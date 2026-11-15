const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const dataset = @import("dataset.zig");
const Color = @import("io_ply.zig").Color;
const render = @import("render.zig");

pub const Triangle = [3]usize;

pub const Mesh = struct {
    points: []Vec3,
    colors: []Color,
    faces: []Triangle,

    pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void {
        allocator.free(self.points);
        allocator.free(self.colors);
        allocator.free(self.faces);
        self.* = undefined;
    }
};

pub const MeshConfig = struct {
    knn: usize = 8,
    outlier_std_mult: f64 = 1.2,
    max_edge_factor: f64 = 4.5,
    min_view_cos: f64 = 0.12,
    dedup_2d_factor: f64 = 0.0,
    laplacian_iters: usize = 0,
    fill_holes: bool = true,
    max_hole_verts: usize = 128,
    max_hole_diam_factor: f64 = 28.0,
    max_hole_depth_factor: f64 = 6.5,
    max_hole_chord_factor: f64 = 13.0,
};

pub const PhotoVerifyConfig = struct {
    long_max_edge_factor: f64 = 24.0,
    long_min_view_cos: f64 = 0.10,
    min_aspect_ratio: f64 = 0.08,
    min_sample_view_cos: f64 = 0.12,
    min_visible_views: usize = 2,
    max_color_error: f64 = 18.0,
    max_view_outlier_error: f64 = 32.0,
    min_normal_align: f64 = 0.32,
    max_depth_cliff_ratio: f64 = 1.8,
};

/// Filter sparse 3D points to those forming coherent local structure (Statistical Outlier Removal)
/// and build a colored 3D triangle mesh via Bowyer-Watson Delaunay triangulation in the view plane
/// with 3D edge-length and grazing-angle pruning.
pub fn buildColoredMesh(
    allocator: std.mem.Allocator,
    raw_points: []const Vec3,
    raw_colors: []const Color,
    poses: []const CameraPose,
    config: MeshConfig,
) !Mesh {
    std.debug.assert(raw_points.len == raw_colors.len);
    if (raw_points.len < 3) {
        return .{
            .points = try allocator.dupe(Vec3, raw_points),
            .colors = try allocator.dupe(Color, raw_colors),
            .faces = try allocator.alloc(Triangle, 0),
        };
    }

    // 1. Statistical Outlier Removal: keep points belonging to coherent local structure
    const filtered = try filterStructurePoints(allocator, raw_points, raw_colors, config.knn, config.outlier_std_mult);
    errdefer {
        allocator.free(filtered.points);
        allocator.free(filtered.colors);
    }

    if (filtered.points.len < 3) {
        return .{
            .points = filtered.points,
            .colors = filtered.colors,
            .faces = try allocator.alloc(Triangle, 0),
        };
    }

    // 2. Compute orthonormal view plane and optionally front-surface deduplicate 2D cells
    const basis = computeViewBasis(filtered.points, poses);

    var mesh_pts = filtered.points;
    var mesh_cols = filtered.colors;
    if (config.dedup_2d_factor > 0) {
        const cell_size = @max(filtered.median_nn * config.dedup_2d_factor, 1e-6);
        const dedup = try frontSurfaceDedup2D(allocator, mesh_pts, mesh_cols, basis, cell_size);
        allocator.free(mesh_pts);
        allocator.free(mesh_cols);
        mesh_pts = dedup.points;
        mesh_cols = dedup.colors;
    }

    if (mesh_pts.len < 3) {
        return .{
            .points = mesh_pts,
            .colors = mesh_cols,
            .faces = try allocator.alloc(Triangle, 0),
        };
    }

    const pts2d = try allocator.alloc(Vec2, mesh_pts.len);
    defer allocator.free(pts2d);
    for (mesh_pts, 0..) |pt, i| {
        const d = pt.sub(basis.origin);
        pts2d[i] = Vec2.init(d.dot(basis.u), d.dot(basis.v));
    }

    // 3. 2D Bowyer-Watson Delaunay triangulation
    const raw_faces = try triangulateDelaunay2D(allocator, pts2d);
    defer allocator.free(raw_faces);

    // 4. Compute median 3D edge length on the deduplicated mesh itself to prune only true gap/jump edges
    var edge_lens: std.ArrayList(f64) = .empty;
    defer edge_lens.deinit(allocator);
    try edge_lens.ensureTotalCapacity(allocator, raw_faces.len * 3);
    for (raw_faces) |tri| {
        const a = mesh_pts[tri[0]];
        const b = mesh_pts[tri[1]];
        const c = mesh_pts[tri[2]];
        try edge_lens.append(allocator, b.sub(a).norm());
        try edge_lens.append(allocator, c.sub(b).norm());
        try edge_lens.append(allocator, a.sub(c).norm());
    }
    std.mem.sort(f64, edge_lens.items, {}, std.sort.asc(f64));
    const med_edge = if (edge_lens.items.len > 0) edge_lens.items[edge_lens.items.len / 2] else 1.0;

    var faces = try filterDelaunayFaces(allocator, mesh_pts, raw_faces, basis.forward, med_edge, config);
    errdefer faces.deinit(allocator);

    if (config.fill_holes and faces.items.len > 0) {
        try fillInteriorHoles(allocator, mesh_pts, pts2d, basis, &faces, med_edge, config);
    }

    const out_faces = try faces.toOwnedSlice(allocator);
    if (config.laplacian_iters > 0 and out_faces.len > 0) {
        try smoothMeshVertices(allocator, mesh_pts, out_faces, config.laplacian_iters, 0.4);
    }

    return .{
        .points = mesh_pts,
        .colors = mesh_cols,
        .faces = out_faces,
    };
}

fn filterDelaunayFaces(
    allocator: std.mem.Allocator,
    mesh_pts: []const Vec3,
    raw_faces: []const Triangle,
    view_forward: Vec3,
    med_edge: f64,
    config: MeshConfig,
) !std.ArrayList(Triangle) {
    const max_edge = @max(med_edge * config.max_edge_factor, 1e-6);
    const max_edge_sq = max_edge * max_edge;

    var faces: std.ArrayList(Triangle) = .empty;
    errdefer faces.deinit(allocator);

    for (raw_faces) |tri| {
        const a = mesh_pts[tri[0]];
        const b = mesh_pts[tri[1]];
        const c = mesh_pts[tri[2]];

        const ab = b.sub(a);
        const bc = c.sub(b);
        const ca = a.sub(c);

        if (ab.normSq() > max_edge_sq or bc.normSq() > max_edge_sq or ca.normSq() > max_edge_sq) {
            continue;
        }

        const cross_n = ab.cross(c.sub(a));
        const area2 = cross_n.norm();
        if (area2 < 1e-12) continue;

        const n_unit = cross_n.scale(1.0 / area2);
        const view_cos = n_unit.dot(view_forward);
        if (@abs(view_cos) < config.min_view_cos) continue;

        if (view_cos > 0) {
            try faces.append(allocator, .{ tri[0], tri[2], tri[1] });
        } else {
            try faces.append(allocator, tri);
        }
    }
    return faces;
}

/// Close remaining enclosed interior boundary loops on an existing mesh using 2D ear clipping
/// with loop-diameter, depth-span, chord-length, and grazing-angle guards.
pub fn fillMeshInteriorHoles(
    allocator: std.mem.Allocator,
    mesh: *Mesh,
    reg_poses: []const CameraPose,
    config: MeshConfig,
) !void {
    if (mesh.points.len < 3 or mesh.faces.len == 0) return;

    const basis = computeViewBasis(mesh.points, reg_poses);
    const pts2d = try allocator.alloc(Vec2, mesh.points.len);
    defer allocator.free(pts2d);
    for (mesh.points, 0..) |pt, i| {
        const d = pt.sub(basis.origin);
        pts2d[i] = Vec2.init(d.dot(basis.u), d.dot(basis.v));
    }

    var edge_lens: std.ArrayList(f64) = .empty;
    defer edge_lens.deinit(allocator);
    try edge_lens.ensureTotalCapacity(allocator, mesh.faces.len * 3);
    for (mesh.faces) |tri| {
        const a = mesh.points[tri[0]];
        const b = mesh.points[tri[1]];
        const c = mesh.points[tri[2]];
        try edge_lens.append(allocator, b.sub(a).norm());
        try edge_lens.append(allocator, c.sub(b).norm());
        try edge_lens.append(allocator, a.sub(c).norm());
    }
    std.mem.sort(f64, edge_lens.items, {}, std.sort.asc(f64));
    const med_edge = if (edge_lens.items.len > 0) edge_lens.items[edge_lens.items.len / 2] else 1.0;

    var faces: std.ArrayList(Triangle) = .empty;
    errdefer faces.deinit(allocator);
    try faces.appendSlice(allocator, mesh.faces);
    const before_count = faces.items.len;

    try fillInteriorHoles(allocator, mesh.points, pts2d, basis, &faces, med_edge, config);

    std.log.info(
        "fillMeshInteriorHoles added {d} triangles ({d} -> {d}, med_edge={d:.4})",
        .{ faces.items.len - before_count, before_count, faces.items.len, med_edge },
    );

    allocator.free(mesh.faces);
    mesh.faces = try faces.toOwnedSlice(allocator);
}

const BoundaryEdge = struct {
    from: usize,
    to: usize,
    used: bool,
};

fn dirEdgeKey(a: usize, b: usize) u64 {
    return (@as(u64, a) << 32) | @as(u64, b);
}

fn cross2d(a: Vec2, b: Vec2) f64 {
    return a.x * b.y - a.y * b.x;
}

fn fillInteriorHoles(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    pts2d: []const Vec2,
    basis: ViewBasis,
    faces: *std.ArrayList(Triangle),
    med_edge: f64,
    config: MeshConfig,
) !void {
    var edge_counts = std.AutoHashMap(u64, u32).init(allocator);
    defer edge_counts.deinit();

    for (faces.items) |tri| {
        const pairs = [3][2]usize{ .{ tri[0], tri[1] }, .{ tri[1], tri[2] }, .{ tri[2], tri[0] } };
        for (pairs) |e| {
            const gop = try edge_counts.getOrPut(dirEdgeKey(e[0], e[1]));
            if (!gop.found_existing) {
                gop.value_ptr.* = 1;
            } else {
                gop.value_ptr.* += 1;
            }
        }
    }

    var boundary_edges: std.ArrayList(BoundaryEdge) = .empty;
    defer boundary_edges.deinit(allocator);

    var it = edge_counts.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const a: usize = @intCast(key >> 32);
        const b: usize = @intCast(key & 0xFFFF_FFFF);
        const rev_cnt = edge_counts.get(dirEdgeKey(b, a)) orelse 0;
        if (entry.value_ptr.* > rev_cnt) {
            const diff = entry.value_ptr.* - rev_cnt;
            for (0..diff) |_| {
                try boundary_edges.append(allocator, .{ .from = a, .to = b, .used = false });
            }
        }
    }

    if (boundary_edges.items.len < 3) return;

    std.mem.sort(BoundaryEdge, boundary_edges.items, {}, struct {
        fn cmp(_: void, lhs: BoundaryEdge, rhs: BoundaryEdge) bool {
            if (lhs.from != rhs.from) return lhs.from < rhs.from;
            return lhs.to < rhs.to;
        }
    }.cmp);

    const head = try allocator.alloc(isize, points.len);
    defer allocator.free(head);
    @memset(head, -1);

    const next_edge = try allocator.alloc(isize, boundary_edges.items.len);
    defer allocator.free(next_edge);

    var ei: usize = boundary_edges.items.len;
    while (ei > 0) {
        ei -= 1;
        const from = boundary_edges.items[ei].from;
        next_edge[ei] = head[from];
        head[from] = @intCast(ei);
    }

    const in_stack = try allocator.alloc(isize, points.len);
    defer allocator.free(in_stack);
    @memset(in_stack, -1);

    var loop_verts: std.ArrayList(usize) = .empty;
    defer loop_verts.deinit(allocator);
    var hole_tris: std.ArrayList(Triangle) = .empty;
    defer hole_tris.deinit(allocator);

    for (0..boundary_edges.items.len) |start_e| {
        if (boundary_edges.items[start_e].used) continue;

        boundary_edges.items[start_e].used = true;
        const start_v = boundary_edges.items[start_e].from;
        var prev_v = start_v;
        var curr_v = boundary_edges.items[start_e].to;

        loop_verts.clearRetainingCapacity();
        try loop_verts.append(allocator, start_v);
        in_stack[start_v] = 0;

        var closed = false;
        while (true) {
            if (curr_v == start_v) {
                closed = true;
                break;
            }
            if (in_stack[curr_v] >= 0) {
                const k_idx: usize = @intCast(in_stack[curr_v]);
                try tryFillHoleLoop(
                    allocator,
                    points,
                    pts2d,
                    basis,
                    loop_verts.items[k_idx..],
                    med_edge,
                    config,
                    faces,
                    &hole_tris,
                );
                for (loop_verts.items[k_idx + 1 ..]) |sv| in_stack[sv] = -1;
                loop_verts.items.len = k_idx + 1;
                prev_v = loop_verts.items[k_idx - 1];
            } else {
                in_stack[curr_v] = @intCast(loop_verts.items.len);
                try loop_verts.append(allocator, curr_v);
            }

            const back = pts2d[prev_v].sub(pts2d[curr_v]);
            const ang_back = std.math.atan2(back.y, back.x);

            var best_ce: ?usize = null;
            var best_cw_delta: f64 = std.math.inf(f64);
            var cand = head[curr_v];
            while (cand >= 0) : (cand = next_edge[@intCast(cand)]) {
                const ce: usize = @intCast(cand);
                if (boundary_edges.items[ce].used) continue;
                const w = boundary_edges.items[ce].to;
                const fwd = pts2d[w].sub(pts2d[curr_v]);
                const ang_fwd = std.math.atan2(fwd.y, fwd.x);
                var cw_delta = ang_back - ang_fwd;
                if (cw_delta <= 0.0) cw_delta += 2.0 * std.math.pi;
                if (cw_delta > 2.0 * std.math.pi) cw_delta -= 2.0 * std.math.pi;
                if (cw_delta < best_cw_delta) {
                    best_cw_delta = cw_delta;
                    best_ce = ce;
                }
            }

            const chosen_ce = best_ce orelse break;
            boundary_edges.items[chosen_ce].used = true;
            prev_v = curr_v;
            curr_v = boundary_edges.items[chosen_ce].to;
        }

        if (closed and loop_verts.items.len >= 3) {
            try tryFillHoleLoop(
                allocator,
                points,
                pts2d,
                basis,
                loop_verts.items,
                med_edge,
                config,
                faces,
                &hole_tris,
            );
        }

        for (loop_verts.items) |sv| in_stack[sv] = -1;
    }
}

fn tryFillHoleLoop(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    pts2d: []const Vec2,
    basis: ViewBasis,
    loop: []const usize,
    med_edge: f64,
    config: MeshConfig,
    faces: *std.ArrayList(Triangle),
    hole_tris: *std.ArrayList(Triangle),
) !void {
    const m = loop.len;
    if (m < 3) return;

    var area2: f64 = 0.0;
    var min_u = pts2d[loop[0]].x;
    var max_u = min_u;
    var min_v = pts2d[loop[0]].y;
    var max_v = min_v;
    var min_d = points[loop[0]].sub(basis.origin).dot(basis.forward);
    var max_d = min_d;

    for (0..m) |i| {
        const va = loop[i];
        const vb = loop[(i + 1) % m];
        const pa = pts2d[va];
        const pb = pts2d[vb];
        area2 += cross2d(pa, pb);

        if (pa.x < min_u) min_u = pa.x;
        if (pa.x > max_u) max_u = pa.x;
        if (pa.y < min_v) min_v = pa.y;
        if (pa.y > max_v) max_v = pa.y;

        const d = points[va].sub(basis.origin).dot(basis.forward);
        if (d < min_d) min_d = d;
        if (d > max_d) max_d = d;
    }

    // Interior holes in CW-oriented mesh always wind CCW (positive 2D shoelace area)
    if (area2 <= 1e-9) return;
    if (m > config.max_hole_verts) return;

    const du = max_u - min_u;
    const dv = max_v - min_v;
    const diag_2d = @sqrt(du * du + dv * dv);
    if (diag_2d > med_edge * config.max_hole_diam_factor) return;

    const depth_span = max_d - min_d;
    if (depth_span > med_edge * config.max_hole_depth_factor) return;

    const max_chord = @max(med_edge * config.max_hole_chord_factor, 1e-6);
    const max_chord_sq = max_chord * max_chord;

    hole_tris.clearRetainingCapacity();
    const ok = try earClipLoop2D(allocator, points, pts2d, basis, loop, max_chord_sq, config.min_view_cos, hole_tris);
    if (!ok) return;

    for (hole_tris.items) |tri| {
        const pa = points[tri[0]];
        const pb = points[tri[1]];
        const pc = points[tri[2]];

        const l0_sq = pb.sub(pa).normSq();
        const l1_sq = pc.sub(pb).normSq();
        const l2_sq = pa.sub(pc).normSq();
        if (l0_sq > max_chord_sq or l1_sq > max_chord_sq or l2_sq > max_chord_sq) continue;

        const cross_n = pb.sub(pa).cross(pc.sub(pa));
        const area3 = cross_n.norm();
        if (area3 < 1e-12) continue;

        const sum_sq = l0_sq + l1_sq + l2_sq;
        if (area3 < 0.015 * sum_sq) continue;

        const view_cos = cross_n.dot(basis.forward) / area3;
        if (view_cos > -config.min_view_cos) continue;

        try faces.append(allocator, tri);
    }
}

fn earClipLoop2D(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    pts2d: []const Vec2,
    basis: ViewBasis,
    loop: []const usize,
    max_chord_sq: f64,
    min_view_cos: f64,
    out_tris: *std.ArrayList(Triangle),
) !bool {
    const m = loop.len;
    if (m < 3) return false;
    if (m == 3) {
        try out_tris.append(allocator, .{ loop[0], loop[2], loop[1] });
        return true;
    }

    const prev_idx = try allocator.alloc(usize, m);
    defer allocator.free(prev_idx);
    const next_idx = try allocator.alloc(usize, m);
    defer allocator.free(next_idx);

    for (0..m) |i| {
        prev_idx[i] = (i + m - 1) % m;
        next_idx[i] = (i + 1) % m;
    }

    var rem = m;
    var cur: usize = 0;
    while (rem > 3) {
        var best_ear: ?usize = null;
        var best_quality: f64 = -1.0;

        var iter = cur;
        for (0..rem) |_| {
            const p = prev_idx[iter];
            const n = next_idx[iter];
            const pa = pts2d[loop[p]];
            const pb = pts2d[loop[iter]];
            const pc = pts2d[loop[n]];

            const ab = pb.sub(pa);
            const ac = pc.sub(pa);
            const area2 = cross2d(ab, ac);

            if (area2 > 1e-12) {
                const bc = pc.sub(pb);
                const ca = pa.sub(pc);
                const len_sum_sq = ab.normSq() + bc.normSq() + ca.normSq();
                const eps = 1e-12 * len_sum_sq;

                var empty = true;
                var chk = next_idx[n];
                while (chk != p) : (chk = next_idx[chk]) {
                    const q = pts2d[loop[chk]];
                    if (cross2d(ab, q.sub(pa)) >= -eps and
                        cross2d(bc, q.sub(pb)) >= -eps and
                        cross2d(ca, q.sub(pc)) >= -eps)
                    {
                        empty = false;
                        break;
                    }
                }

                if (empty) {
                    const p3a = points[loop[p]];
                    const p3b = points[loop[n]];
                    const p3c = points[loop[iter]];
                    const chord_ok = p3b.sub(p3a).normSq() <= max_chord_sq and
                        p3c.sub(p3b).normSq() <= max_chord_sq and
                        p3a.sub(p3c).normSq() <= max_chord_sq;
                    const cross_n = p3b.sub(p3a).cross(p3c.sub(p3a));
                    const area3 = cross_n.norm();
                    const view_ok = area3 >= 1e-12 and (cross_n.dot(basis.forward) / area3) <= -min_view_cos;
                    const bonus: f64 = if (chord_ok and view_ok) 100.0 else if (view_ok) 10.0 else 0.0;
                    const quality = bonus + area2 / @max(len_sum_sq, 1e-18);
                    if (quality > best_quality) {
                        best_quality = quality;
                        best_ear = iter;
                    }
                }
            }
            iter = next_idx[iter];
        }

        const ear = best_ear orelse return false;
        const ep = prev_idx[ear];
        const en = next_idx[ear];

        try out_tris.append(allocator, .{ loop[ep], loop[en], loop[ear] });

        next_idx[ep] = en;
        prev_idx[en] = ep;
        cur = en;
        rem -= 1;
    }

    const p = prev_idx[cur];
    const n = next_idx[cur];
    try out_tris.append(allocator, .{ loop[p], loop[n], loop[cur] });
    return true;
}

const CellKey2D = struct {
    x: i32,
    y: i32,
};

const CellAcc = struct {
    sx: f64,
    sy: f64,
    sz: f64,
    sr: u64,
    sg: u64,
    sb: u64,
    count: u64,
    depth_sum: f64,
};

fn frontSurfaceDedup2D(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    basis: ViewBasis,
    cell_size: f64,
) !struct { points: []Vec3, colors: []Color } {
    var grid = std.AutoHashMap(CellKey2D, CellAcc).init(allocator);
    defer grid.deinit();

    const inv_c = 1.0 / cell_size;
    for (points, colors) |pt, col| {
        const d = pt.sub(basis.origin);
        const u = d.dot(basis.u);
        const v = d.dot(basis.v);
        const depth = d.dot(basis.forward);
        const key = CellKey2D{
            .x = @intFromFloat(@floor(u * inv_c)),
            .y = @intFromFloat(@floor(v * inv_c)),
        };
        const gop = try grid.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{
                .sx = pt.x,
                .sy = pt.y,
                .sz = pt.z,
                .sr = col.r,
                .sg = col.g,
                .sb = col.b,
                .count = 1,
                .depth_sum = depth,
            };
        } else {
            gop.value_ptr.sx += pt.x;
            gop.value_ptr.sy += pt.y;
            gop.value_ptr.sz += pt.z;
            gop.value_ptr.sr += col.r;
            gop.value_ptr.sg += col.g;
            gop.value_ptr.sb += col.b;
            gop.value_ptr.count += 1;
            gop.value_ptr.depth_sum += depth;
        }
    }

    const n_out = grid.count();
    const out_pts = try allocator.alloc(Vec3, n_out);
    errdefer allocator.free(out_pts);
    const out_cols = try allocator.alloc(Color, n_out);

    // 3x3 median depth filtering across neighboring 2D cells removes depth spikes cleanly
    var it = grid.iterator();
    var out_i: usize = 0;
    while (it.next()) |entry| : (out_i += 1) {
        const key = entry.key_ptr.*;
        const acc = entry.value_ptr.*;
        const cf: f64 = @floatFromInt(acc.count);
        var pt = Vec3.init(acc.sx / cf, acc.sy / cf, acc.sz / cf);
        const cur_depth = acc.depth_sum / cf;

        var nbr_depths: [9]f64 = undefined;
        var nd_len: usize = 0;
        var dy: i32 = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: i32 = -1;
            while (dx <= 1) : (dx += 1) {
                if (grid.get(.{ .x = key.x + dx, .y = key.y + dy })) |n_acc| {
                    nbr_depths[nd_len] = n_acc.depth_sum / @as(f64, @floatFromInt(n_acc.count));
                    nd_len += 1;
                }
            }
        }
        if (nd_len >= 3) {
            std.mem.sort(f64, nbr_depths[0..nd_len], {}, std.sort.asc(f64));
            const med_depth = nbr_depths[nd_len / 2];
            if (@abs(cur_depth - med_depth) > cell_size * 1.5) {
                pt = pt.add(basis.forward.scale(med_depth - cur_depth));
            }
        }

        out_pts[out_i] = pt;
        out_cols[out_i] = .{
            .r = @intCast(acc.sr / acc.count),
            .g = @intCast(acc.sg / acc.count),
            .b = @intCast(acc.sb / acc.count),
        };
    }
    return .{ .points = out_pts, .colors = out_cols };
}

fn smoothMeshVertices(
    allocator: std.mem.Allocator,
    points: []Vec3,
    faces: []const Triangle,
    iters: usize,
    lambda: f64,
) !void {
    const accum = try allocator.alloc(Vec3, points.len);
    defer allocator.free(accum);
    const counts = try allocator.alloc(u32, points.len);
    defer allocator.free(counts);

    for (0..iters) |_| {
        @memset(accum, Vec3.zero);
        @memset(counts, 0);

        for (faces) |f| {
            const pairs = [3][2]usize{ .{ f[0], f[1] }, .{ f[1], f[2] }, .{ f[2], f[0] } };
            for (pairs) |e| {
                accum[e[0]] = accum[e[0]].add(points[e[1]]);
                counts[e[0]] += 1;
                accum[e[1]] = accum[e[1]].add(points[e[0]]);
                counts[e[1]] += 1;
            }
        }

        for (points, accum, counts) |*pt, sum, cnt| {
            if (cnt > 0) {
                const avg = sum.scale(1.0 / @as(f64, @floatFromInt(cnt)));
                pt.* = pt.*.add(avg.sub(pt.*).scale(lambda));
            }
        }
    }
}

const FilteredPoints = struct {
    points: []Vec3,
    colors: []Color,
    median_nn: f64,
};

fn filterStructurePoints(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    knn_req: usize,
    std_mult: f64,
) !FilteredPoints {
    const n = points.len;
    const k = @min(knn_req, n - 1);
    if (k == 0) {
        return .{
            .points = try allocator.dupe(Vec3, points),
            .colors = try allocator.dupe(Color, colors),
            .median_nn = 1.0,
        };
    }

    const knn_means = try allocator.alloc(f64, n);
    defer allocator.free(knn_means);
    const nn1_dists = try allocator.alloc(f64, n);
    defer allocator.free(nn1_dists);

    // Sort indices by X so k-NN search can early-exit in both directions
    const order = try allocator.alloc(usize, n);
    defer allocator.free(order);
    for (0..n) |i| order[i] = i;
    std.mem.sort(usize, order, points, struct {
        fn cmp(pts: []const Vec3, a: usize, b: usize) bool {
            return pts[a].x < pts[b].x;
        }
    }.cmp);

    const topk = try allocator.alloc(f64, k);
    defer allocator.free(topk);

    for (0..n) |pos_i| {
        const i = order[pos_i];
        @memset(topk, std.math.inf(f64));
        const pi = points[i];

        // Scan right
        var pos_r = pos_i + 1;
        while (pos_r < n) : (pos_r += 1) {
            const j = order[pos_r];
            const dx = points[j].x - pi.x;
            if (dx * dx >= topk[k - 1]) break;
            const d_sq = pi.sub(points[j]).normSq();
            if (d_sq < topk[k - 1]) {
                insertTopK(topk, d_sq);
            }
        }
        // Scan left
        var pos_l = pos_i;
        while (pos_l > 0) {
            pos_l -= 1;
            const j = order[pos_l];
            const dx = pi.x - points[j].x;
            if (dx * dx >= topk[k - 1]) break;
            const d_sq = pi.sub(points[j]).normSq();
            if (d_sq < topk[k - 1]) {
                insertTopK(topk, d_sq);
            }
        }

        nn1_dists[i] = @sqrt(topk[0]);
        var sum: f64 = 0;
        for (topk) |d_sq| sum += @sqrt(d_sq);
        knn_means[i] = sum / @as(f64, @floatFromInt(k));
    }

    var mean: f64 = 0;
    for (knn_means) |v| mean += v;
    mean /= @as(f64, @floatFromInt(n));

    var var_sum: f64 = 0;
    for (knn_means) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const std_dev = @sqrt(var_sum / @as(f64, @floatFromInt(n)));
    const thresh = mean + std_mult * std_dev;

    var out_pts: std.ArrayList(Vec3) = .empty;
    errdefer out_pts.deinit(allocator);
    var out_cols: std.ArrayList(Color) = .empty;
    errdefer out_cols.deinit(allocator);
    var kept_nn1: std.ArrayList(f64) = .empty;
    defer kept_nn1.deinit(allocator);

    for (0..n) |i| {
        if (knn_means[i] <= thresh) {
            try out_pts.append(allocator, points[i]);
            try out_cols.append(allocator, colors[i]);
            try kept_nn1.append(allocator, nn1_dists[i]);
        }
    }

    std.mem.sort(f64, kept_nn1.items, {}, std.sort.asc(f64));
    const median_nn = if (kept_nn1.items.len > 0) kept_nn1.items[kept_nn1.items.len / 2] else 1.0;

    return .{
        .points = try out_pts.toOwnedSlice(allocator),
        .colors = try out_cols.toOwnedSlice(allocator),
        .median_nn = @max(median_nn, 1e-6),
    };
}

fn insertTopK(topk: []f64, d_sq: f64) void {
    topk[topk.len - 1] = d_sq;
    var pos: usize = topk.len - 1;
    while (pos > 0 and topk[pos] < topk[pos - 1]) : (pos -= 1) {
        std.mem.swap(f64, &topk[pos], &topk[pos - 1]);
    }
}

const ViewBasis = struct {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    forward: Vec3,
};

fn computeViewBasis(points: []const Vec3, poses: []const CameraPose) ViewBasis {
    var origin = Vec3.zero;
    for (points) |p| origin = origin.add(p);
    origin = origin.scale(1.0 / @as(f64, @floatFromInt(points.len)));

    var fwd = Vec3.zero;
    for (poses) |pose| {
        // Camera +Z axis in world coordinates is row 2 of R
        fwd = fwd.add(Vec3.init(pose.r.m[2][0], pose.r.m[2][1], pose.r.m[2][2]));
    }
    if (fwd.normSq() < 1e-12) {
        fwd = Vec3.init(0, 0, 1);
    } else {
        fwd = fwd.normalize();
    }

    const world_up = if (@abs(fwd.y) < 0.9) Vec3.init(0, 1, 0) else Vec3.init(1, 0, 0);
    const u = world_up.cross(fwd).normalize();
    const v = fwd.cross(u).normalize();

    return .{ .origin = origin, .u = u, .v = v, .forward = fwd };
}

const BWTriangle = struct {
    v: [3]usize,
    cx: f64,
    cy: f64,
    r_sq: f64,
    max_x: f64,
};

fn makeBWTriangle(pts: []const Vec2, v0: usize, v1: usize, v2: usize) BWTriangle {
    const a = pts[v0];
    const b = pts[v1];
    const c = pts[v2];

    const ax = a.x;
    const ay = a.y;
    const bx = b.x;
    const by = b.y;
    const cx = c.x;
    const cy = c.y;

    const d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (@abs(d) < 1e-15) {
        return .{ .v = .{ v0, v1, v2 }, .cx = 0, .cy = 0, .r_sq = -1.0, .max_x = -std.math.inf(f64) };
    }

    const a2 = ax * ax + ay * ay;
    const b2 = bx * bx + by * by;
    const c2 = cx * cx + cy * cy;

    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    const dx = ax - ux;
    const dy = ay - uy;
    const r_sq = dx * dx + dy * dy;
    return .{
        .v = .{ v0, v1, v2 },
        .cx = ux,
        .cy = uy,
        .r_sq = r_sq,
        .max_x = ux + @sqrt(r_sq),
    };
}

/// Bowyer-Watson 2D Delaunay triangulation using X-sorted sweep-front pruning.
pub fn triangulateDelaunay2D(allocator: std.mem.Allocator, pts: []const Vec2) ![]Triangle {
    if (pts.len < 3) return try allocator.alloc(Triangle, 0);

    var min_x = pts[0].x;
    var max_x = pts[0].x;
    var min_y = pts[0].y;
    var max_y = pts[0].y;
    for (pts) |p| {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }

    const dx = @max(max_x - min_x, 1e-3);
    const dy = @max(max_y - min_y, 1e-3);
    const delta_max = @max(dx, dy);
    const mid_x = (min_x + max_x) * 0.5;
    const mid_y = (min_y + max_y) * 0.5;

    // Append 3 super-triangle vertices at indices n, n+1, n+2
    const n = pts.len;
    const all_pts = try allocator.alloc(Vec2, n + 3);
    defer allocator.free(all_pts);
    @memcpy(all_pts[0..n], pts);
    all_pts[n + 0] = Vec2.init(mid_x - 20.0 * delta_max, mid_y - delta_max);
    all_pts[n + 1] = Vec2.init(mid_x, mid_y + 20.0 * delta_max);
    all_pts[n + 2] = Vec2.init(mid_x + 20.0 * delta_max, mid_y - delta_max);

    // Sort point indices by (X, Y) ascending so circumcircles left of p.x can be retired permanently
    const order = try allocator.alloc(usize, n);
    defer allocator.free(order);
    for (0..n) |i| order[i] = i;
    std.mem.sort(usize, order, pts, struct {
        fn cmp(p_slice: []const Vec2, a: usize, b: usize) bool {
            if (p_slice[a].x != p_slice[b].x) return p_slice[a].x < p_slice[b].x;
            return p_slice[a].y < p_slice[b].y;
        }
    }.cmp);

    const dup_eps_sq = (delta_max * 1e-7) * (delta_max * 1e-7);

    var active_tris: std.ArrayList(BWTriangle) = .empty;
    defer active_tris.deinit(allocator);
    try active_tris.append(allocator, makeBWTriangle(all_pts, n, n + 1, n + 2));

    var out: std.ArrayList(Triangle) = .empty;
    errdefer out.deinit(allocator);

    var bad_tris: std.ArrayList(BWTriangle) = .empty;
    defer bad_tris.deinit(allocator);
    var edges: std.ArrayList([2]usize) = .empty;
    defer edges.deinit(allocator);

    var prev_inserted: ?Vec2 = null;
    for (order) |p_idx| {
        const p = all_pts[p_idx];
        if (prev_inserted) |prev_p| {
            if (p.sub(prev_p).normSq() <= dup_eps_sq) continue;
        }
        prev_inserted = p;
        bad_tris.clearRetainingCapacity();

        var t_idx: usize = 0;
        while (t_idx < active_tris.items.len) {
            const t = active_tris.items[t_idx];
            if (t.r_sq < 0) {
                _ = active_tris.swapRemove(t_idx);
                continue;
            }
            if (t.max_x < p.x - 1e-9) {
                if (t.v[0] < n and t.v[1] < n and t.v[2] < n) {
                    try out.append(allocator, t.v);
                }
                _ = active_tris.swapRemove(t_idx);
                continue;
            }
            const ex = p.x - t.cx;
            const ey = p.y - t.cy;
            if (ex * ex + ey * ey <= t.r_sq + 1e-12) {
                try bad_tris.append(allocator, t);
                _ = active_tris.swapRemove(t_idx);
                continue;
            }
            t_idx += 1;
        }

        edges.clearRetainingCapacity();
        for (bad_tris.items, 0..) |bt, b_idx| {
            const tv = bt.v;
            const tri_edges = [3][2]usize{
                .{ tv[0], tv[1] },
                .{ tv[1], tv[2] },
                .{ tv[2], tv[0] },
            };
            for (tri_edges) |e| {
                var shared = false;
                for (bad_tris.items, 0..) |other_bt, other_idx| {
                    if (other_idx == b_idx) continue;
                    if (triHasEdge(other_bt.v, e[0], e[1])) {
                        shared = true;
                        break;
                    }
                }
                if (!shared) {
                    try edges.append(allocator, e);
                }
            }
        }

        for (edges.items) |e| {
            try active_tris.append(allocator, makeBWTriangle(all_pts, e[0], e[1], p_idx));
        }
    }

    for (active_tris.items) |t| {
        if (t.r_sq < 0) continue;
        if (t.v[0] >= n or t.v[1] >= n or t.v[2] >= n) continue;
        try out.append(allocator, t.v);
    }
    return try out.toOwnedSlice(allocator);
}

fn triHasEdge(v: [3]usize, a: usize, b: usize) bool {
    const pairs = [3][2]usize{
        .{ v[0], v[1] },
        .{ v[1], v[2] },
        .{ v[2], v[0] },
    };
    for (pairs) |p| {
        if ((p[0] == a and p[1] == b) or (p[0] == b and p[1] == a)) return true;
    }
    return false;
}

/// Subdivide mesh triangles and project RGB pixel colors from the best-facing photo onto every vertex.
pub fn subdivideAndProjectTexture(
    allocator: std.mem.Allocator,
    mesh: *Mesh,
    k: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    subdiv_levels: u32,
) !void {
    if (mesh.faces.len == 0) return;

    var verts: std.ArrayList(Vec3) = .empty;
    errdefer verts.deinit(allocator);
    try verts.appendSlice(allocator, mesh.points);

    var faces: std.ArrayList(Triangle) = .empty;
    errdefer faces.deinit(allocator);
    try faces.appendSlice(allocator, mesh.faces);

    for (0..subdiv_levels) |_| {
        var edge_midpoints = std.AutoHashMap(u64, usize).init(allocator);
        defer edge_midpoints.deinit();

        var next_faces: std.ArrayList(Triangle) = .empty;
        errdefer next_faces.deinit(allocator);
        try next_faces.ensureTotalCapacity(allocator, faces.items.len * 4);

        for (faces.items) |tri| {
            const m01 = try getOrCreateMidpoint(allocator, &verts, &edge_midpoints, tri[0], tri[1]);
            const m12 = try getOrCreateMidpoint(allocator, &verts, &edge_midpoints, tri[1], tri[2]);
            const m20 = try getOrCreateMidpoint(allocator, &verts, &edge_midpoints, tri[2], tri[0]);

            try next_faces.append(allocator, .{ tri[0], m01, m20 });
            try next_faces.append(allocator, .{ m01, tri[1], m12 });
            try next_faces.append(allocator, .{ m20, m12, tri[2] });
            try next_faces.append(allocator, .{ m01, m12, m20 });
        }

        faces.deinit(allocator);
        faces = next_faces;
    }

    // Compute smooth per-vertex normals for choosing the best-facing camera
    const normals = try allocator.alloc(Vec3, verts.items.len);
    defer allocator.free(normals);
    @memset(normals, Vec3.zero);
    for (faces.items) |tri| {
        const a = verts.items[tri[0]];
        const b = verts.items[tri[1]];
        const c = verts.items[tri[2]];
        const fn_vec = b.sub(a).cross(c.sub(a));
        normals[tri[0]] = normals[tri[0]].add(fn_vec);
        normals[tri[1]] = normals[tri[1]].add(fn_vec);
        normals[tri[2]] = normals[tri[2]].add(fn_vec);
    }
    for (normals) |*nrm| {
        nrm.* = nrm.normalize();
    }

    // Precompute camera centers
    const cam_centers = try allocator.alloc(Vec3, poses.len);
    defer allocator.free(cam_centers);
    for (poses, 0..) |p, ci| cam_centers[ci] = p.cameraCenter();

    // Project every vertex onto the best-facing, visibility-consistent registered photo
    const new_cols = try allocator.alloc(Color, verts.items.len);
    errdefer allocator.free(new_cols);
    const has_unocc = try allocator.alloc(bool, verts.items.len);
    defer allocator.free(has_unocc);
    @memset(has_unocc, false);
    const best_scores = try allocator.alloc(f64, verts.items.len);
    defer allocator.free(best_scores);
    @memset(best_scores, -1.0);
    const fallback_scores = try allocator.alloc(f64, verts.items.len);
    defer allocator.free(fallback_scores);
    @memset(fallback_scores, -1.0);
    const fallback_cols = try allocator.alloc(Color, verts.items.len);
    defer allocator.free(fallback_cols);

    for (verts.items, 0..) |_, vi| {
        const init_col: Color = if (vi < mesh.colors.len) mesh.colors[vi] else .{ .r = 160, .g = 160, .b = 160 };
        new_cols[vi] = init_col;
        fallback_cols[vi] = init_col;
    }

    for (images, poses, registered, cam_centers) |img, pose, is_reg, cam_c| {
        if (!is_reg) continue;
        var vis = try render.renderMeshView(allocator, img.width, img.height, k, pose, mesh.points, mesh.colors, mesh.faces);
        defer vis.deinit(allocator);

        const wf: f64 = @floatFromInt(img.width);
        const hf: f64 = @floatFromInt(img.height);
        for (verts.items, normals, 0..) |pt, nrm, vi| {
            const p_cam = pose.transformPoint(pt);
            if (p_cam.z <= 1e-4) continue;
            const uv = lie.projectPoint(k, pose, pt) orelse continue;
            if (uv.x < 1.0 or uv.x >= wf - 2.0 or uv.y < 1.0 or uv.y >= hf - 2.0) continue;

            const ray_to_cam = cam_c.sub(pt).normalize();
            const cos_view = if (nrm.normSq() > 0.5) @max(0.05, nrm.dot(ray_to_cam)) else 1.0;
            const dx = (uv.x - wf * 0.5) / wf;
            const dy = (uv.y - hf * 0.5) / hf;
            const centrality = @max(0.1, 1.0 - @sqrt(dx * dx + dy * dy));
            const score = cos_view * centrality;

            const px_i: usize = @min(img.width - 1, @as(usize, @intFromFloat(@floor(uv.x))));
            const py_i: usize = @min(img.height - 1, @as(usize, @intFromFloat(@floor(uv.y))));
            const p_idx = py_i * img.width + px_i;
            const occ_tol = @max(0.08, 0.015 * p_cam.z);
            const is_visible = !vis.covered[p_idx] or (p_cam.z <= @as(f64, vis.depth[p_idx]) + occ_tol);

            if (is_visible) {
                if (!has_unocc[vi] or score > best_scores[vi]) {
                    has_unocc[vi] = true;
                    best_scores[vi] = score;
                    new_cols[vi] = sampleBilinearRgb(img, uv.x, uv.y);
                }
            } else if (!has_unocc[vi] and score > fallback_scores[vi]) {
                fallback_scores[vi] = score;
                fallback_cols[vi] = sampleBilinearRgb(img, uv.x, uv.y);
            }
        }
    }

    for (0..verts.items.len) |vi| {
        if (!has_unocc[vi] and fallback_scores[vi] > -0.5) {
            new_cols[vi] = fallback_cols[vi];
        }
    }

    allocator.free(mesh.points);
    allocator.free(mesh.colors);
    allocator.free(mesh.faces);
    mesh.points = try verts.toOwnedSlice(allocator);
    mesh.colors = new_cols;
    mesh.faces = try faces.toOwnedSlice(allocator);
}

const TriKey = struct {
    a: usize,
    b: usize,
    c: usize,
};

fn canonicalTriKey(tri: Triangle) TriKey {
    var s = tri;
    if (s[0] > s[1]) std.mem.swap(usize, &s[0], &s[1]);
    if (s[1] > s[2]) std.mem.swap(usize, &s[1], &s[2]);
    if (s[0] > s[1]) std.mem.swap(usize, &s[0], &s[1]);
    return .{ .a = s[0], .b = s[1], .c = s[2] };
}

/// Complete holes in a surface mesh by verifying longer Delaunay triangles against
/// multi-view photometric consistency and local depth/normal continuity.
pub fn completeMeshWithPhotoVerify(
    allocator: std.mem.Allocator,
    mesh: *Mesh,
    k: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    reg_poses: []const CameraPose,
    config: PhotoVerifyConfig,
) !void {
    if (mesh.points.len < 3 or images.len == 0) return;

    const basis = computeViewBasis(mesh.points, reg_poses);
    const pts2d = try allocator.alloc(Vec2, mesh.points.len);
    defer allocator.free(pts2d);
    const depths = try allocator.alloc(f64, mesh.points.len);
    defer allocator.free(depths);
    for (mesh.points, 0..) |pt, vi| {
        const d = pt.sub(basis.origin);
        pts2d[vi] = Vec2.init(d.dot(basis.u), d.dot(basis.v));
        depths[vi] = d.dot(basis.forward);
    }

    const raw_faces = try triangulateDelaunay2D(allocator, pts2d);
    defer allocator.free(raw_faces);
    if (raw_faces.len == 0) return;

    var edge_lens: std.ArrayList(f64) = .empty;
    defer edge_lens.deinit(allocator);
    try edge_lens.ensureTotalCapacity(allocator, raw_faces.len * 3);
    for (raw_faces) |tri| {
        const va = mesh.points[tri[0]];
        const vb = mesh.points[tri[1]];
        const vc = mesh.points[tri[2]];
        try edge_lens.append(allocator, vb.sub(va).norm());
        try edge_lens.append(allocator, vc.sub(vb).norm());
        try edge_lens.append(allocator, va.sub(vc).norm());
    }
    std.mem.sort(f64, edge_lens.items, {}, std.sort.asc(f64));
    const med_edge = if (edge_lens.items.len > 0) edge_lens.items[edge_lens.items.len / 2] else 1.0;
    const long_max_edge = @max(med_edge * config.long_max_edge_factor, 1e-6);
    const long_max_edge_sq = long_max_edge * long_max_edge;

    var existing_faces = std.AutoHashMap(TriKey, void).init(allocator);
    defer existing_faces.deinit();
    try existing_faces.ensureTotalCapacity(@intCast(mesh.faces.len));

    const vertex_normals = try allocator.alloc(Vec3, mesh.points.len);
    defer allocator.free(vertex_normals);
    @memset(vertex_normals, Vec3.zero);

    for (mesh.faces) |tri| {
        try existing_faces.put(canonicalTriKey(tri), {});
        const va = mesh.points[tri[0]];
        const vb = mesh.points[tri[1]];
        const vc = mesh.points[tri[2]];
        const cross_n = vb.sub(va).cross(vc.sub(va));
        const area2 = cross_n.norm();
        if (area2 < 1e-12) continue;
        var n_facing = cross_n.scale(1.0 / area2);
        if (n_facing.dot(basis.forward) > 0) n_facing = n_facing.neg();
        const weighted = n_facing.scale(area2);
        vertex_normals[tri[0]] = vertex_normals[tri[0]].add(weighted);
        vertex_normals[tri[1]] = vertex_normals[tri[1]].add(weighted);
        vertex_normals[tri[2]] = vertex_normals[tri[2]].add(weighted);
    }

    const cam_centers = try allocator.alloc(Vec3, poses.len);
    defer allocator.free(cam_centers);
    for (poses, 0..) |pose, ci| cam_centers[ci] = pose.cameraCenter();

    const obs_r = try allocator.alloc(f64, images.len);
    defer allocator.free(obs_r);
    const obs_g = try allocator.alloc(f64, images.len);
    defer allocator.free(obs_g);
    const obs_b = try allocator.alloc(f64, images.len);
    defer allocator.free(obs_b);
    const sort_buf = try allocator.alloc(f64, images.len);
    defer allocator.free(sort_buf);

    var out_faces: std.ArrayList(Triangle) = .empty;
    errdefer out_faces.deinit(allocator);
    try out_faces.appendSlice(allocator, mesh.faces);

    const sample_weights = [7][3]f64{
        .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },
        .{ 0.60, 0.20, 0.20 },
        .{ 0.20, 0.60, 0.20 },
        .{ 0.20, 0.20, 0.60 },
        .{ 0.45, 0.45, 0.10 },
        .{ 0.45, 0.10, 0.45 },
        .{ 0.10, 0.45, 0.45 },
    };

    var added_count: usize = 0;
    for (raw_faces) |tri| {
        if (existing_faces.contains(canonicalTriKey(tri))) continue;

        const va = mesh.points[tri[0]];
        const vb = mesh.points[tri[1]];
        const vc = mesh.points[tri[2]];

        const len_ab_sq = vb.sub(va).normSq();
        const len_bc_sq = vc.sub(vb).normSq();
        const len_ca_sq = va.sub(vc).normSq();
        const max_len_sq = @max(len_ab_sq, @max(len_bc_sq, len_ca_sq));
        if (max_len_sq > long_max_edge_sq) continue;

        const cross_n = vb.sub(va).cross(vc.sub(va));
        const area2 = cross_n.norm();
        if (area2 < 1e-12) continue;
        if (area2 / max_len_sq < config.min_aspect_ratio) continue;

        const n_unit = cross_n.scale(1.0 / area2);
        const view_cos = n_unit.dot(basis.forward);
        if (@abs(view_cos) < config.long_min_view_cos) continue;

        const n_facing = if (view_cos > 0) n_unit.neg() else n_unit;
        const oriented_tri: Triangle = if (view_cos > 0) .{ tri[0], tri[2], tri[1] } else tri;

        if (hasDepthCliff(tri, pts2d, depths, med_edge, config)) continue;
        if (hasNormalAnomaly(tri, n_facing, vertex_normals, config)) continue;

        var all_samples_ok = true;
        for (sample_weights) |sw| {
            const sp = va.scale(sw[0]).add(vb.scale(sw[1])).add(vc.scale(sw[2]));
            const err = evalSamplePhotoError(
                sp,
                n_facing,
                k,
                images,
                poses,
                registered,
                cam_centers,
                config.min_sample_view_cos,
                config.min_visible_views,
                config.max_view_outlier_error,
                obs_r,
                obs_g,
                obs_b,
                sort_buf,
            ) orelse {
                all_samples_ok = false;
                break;
            };
            if (err > config.max_color_error) {
                all_samples_ok = false;
                break;
            }
        }
        if (!all_samples_ok) continue;

        try out_faces.append(allocator, oriented_tri);
        added_count += 1;
    }

    std.log.info(
        "PhotoVerify completed {d} triangles (base {d} -> total {d}, med_edge={d:.4})",
        .{ added_count, mesh.faces.len, out_faces.items.len, med_edge },
    );

    allocator.free(mesh.faces);
    mesh.faces = try out_faces.toOwnedSlice(allocator);
}

fn hasDepthCliff(
    tri: Triangle,
    pts2d: []const Vec2,
    depths: []const f64,
    med_edge: f64,
    config: PhotoVerifyConfig,
) bool {
    const edges = [3][2]usize{
        .{ tri[0], tri[1] },
        .{ tri[1], tri[2] },
        .{ tri[2], tri[0] },
    };
    for (edges) |edge| {
        const d2d = pts2d[edge[0]].sub(pts2d[edge[1]]).norm();
        const d_depth = @abs(depths[edge[0]] - depths[edge[1]]);
        if (d_depth > med_edge * 2.5 and d_depth > d2d * config.max_depth_cliff_ratio) {
            return true;
        }
    }
    return false;
}

fn hasNormalAnomaly(
    tri: Triangle,
    n_facing: Vec3,
    vertex_normals: []const Vec3,
    config: PhotoVerifyConfig,
) bool {
    var sum_dot: f64 = 0.0;
    var count: usize = 0;
    for (tri) |vi| {
        const vn_sq = vertex_normals[vi].normSq();
        if (vn_sq > 1e-12) {
            const vn_unit = vertex_normals[vi].scale(1.0 / @sqrt(vn_sq));
            const dot = n_facing.dot(vn_unit);
            if (dot < 0.0) return true;
            sum_dot += dot;
            count += 1;
        }
    }
    if (count > 0 and (sum_dot / @as(f64, @floatFromInt(count))) < config.min_normal_align) {
        return true;
    }
    return false;
}

fn evalSamplePhotoError(
    sp: Vec3,
    n_facing: Vec3,
    k: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    cam_centers: []const Vec3,
    min_sample_view_cos: f64,
    min_visible_views: usize,
    max_view_outlier_error: f64,
    obs_r: []f64,
    obs_g: []f64,
    obs_b: []f64,
    sort_buf: []f64,
) ?f64 {
    var m: usize = 0;
    for (images, poses, registered, cam_centers) |img, pose, is_reg, cam_c| {
        if (!is_reg) continue;
        const ray_to_cam = cam_c.sub(sp);
        const dist_sq = ray_to_cam.normSq();
        if (dist_sq < 1e-12) continue;
        const cos_view = n_facing.dot(ray_to_cam.scale(1.0 / @sqrt(dist_sq)));
        if (cos_view < min_sample_view_cos) continue;

        const uv = lie.projectPoint(k, pose, sp) orelse continue;
        const wf: f64 = @floatFromInt(img.width);
        const hf: f64 = @floatFromInt(img.height);
        if (uv.x < 1.0 or uv.x >= wf - 2.0 or uv.y < 1.0 or uv.y >= hf - 2.0) continue;

        const col = sampleBilinearRgb(img, uv.x, uv.y);
        obs_r[m] = @floatFromInt(col.r);
        obs_g[m] = @floatFromInt(col.g);
        obs_b[m] = @floatFromInt(col.b);
        m += 1;
    }

    if (m < min_visible_views or m < 2) return null;

    if (m == 2) {
        const pair_err = (@abs(obs_r[0] - obs_r[1]) + @abs(obs_g[0] - obs_g[1]) + @abs(obs_b[0] - obs_b[1])) / 3.0;
        if (pair_err > max_view_outlier_error) return null;
        return pair_err;
    }

    const med_r = medianSlice(obs_r[0..m], sort_buf[0..m]);
    const med_g = medianSlice(obs_g[0..m], sort_buf[0..m]);
    const med_b = medianSlice(obs_b[0..m], sort_buf[0..m]);

    var err_sum: f64 = 0.0;
    var max_err: f64 = 0.0;
    for (0..m) |idx| {
        const err = (@abs(obs_r[idx] - med_r) + @abs(obs_g[idx] - med_g) + @abs(obs_b[idx] - med_b)) / 3.0;
        err_sum += err;
        if (err > max_err) max_err = err;
    }
    if (max_err > max_view_outlier_error) return null;
    return err_sum / @as(f64, @floatFromInt(m));
}

fn medianSlice(vals: []const f64, buf: []f64) f64 {
    @memcpy(buf, vals);
    std.mem.sort(f64, buf, {}, std.sort.asc(f64));
    return buf[buf.len / 2];
}

fn getOrCreateMidpoint(
    allocator: std.mem.Allocator,
    verts: *std.ArrayList(Vec3),
    edge_map: *std.AutoHashMap(u64, usize),
    a: usize,
    b: usize,
) !usize {
    const lo = @min(a, b);
    const hi = @max(a, b);
    const key = (@as(u64, lo) << 32) | @as(u64, hi);
    const gop = try edge_map.getOrPut(key);
    if (gop.found_existing) return gop.value_ptr.*;

    const mid = verts.items[lo].add(verts.items[hi]).scale(0.5);
    const idx = verts.items.len;
    try verts.append(allocator, mid);
    gop.value_ptr.* = idx;
    return idx;
}

fn sampleBilinearRgb(img: dataset.ImageData, ux: f64, vy: f64) Color {
    const w: usize = img.width;
    const x0: usize = @intFromFloat(@floor(ux));
    const y0: usize = @intFromFloat(@floor(vy));
    const x1 = @min(w - 1, x0 + 1);
    const y1 = @min(@as(usize, img.height) - 1, y0 + 1);
    const wx = ux - @as(f64, @floatFromInt(x0));
    const wy = vy - @as(f64, @floatFromInt(y0));

    var out: [3]u8 = undefined;
    for (0..3) |ch| {
        const p00: f64 = @floatFromInt(img.rgb[(y0 * w + x0) * 3 + ch]);
        const p10: f64 = @floatFromInt(img.rgb[(y0 * w + x1) * 3 + ch]);
        const p01: f64 = @floatFromInt(img.rgb[(y1 * w + x0) * 3 + ch]);
        const p11: f64 = @floatFromInt(img.rgb[(y1 * w + x1) * 3 + ch]);
        const v = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10) + wy * ((1.0 - wx) * p01 + wx * p11);
        out[ch] = @intFromFloat(std.math.clamp(@round(v), 0.0, 255.0));
    }
    return .{ .r = out[0], .g = out[1], .b = out[2] };
}

test "buildColoredMesh filters outliers and triangulates coherent 3D grid" {
    const gpa = std.testing.allocator;
    var pts: std.ArrayList(Vec3) = .empty;
    defer pts.deinit(gpa);
    var cols: std.ArrayList(Color) = .empty;
    defer cols.deinit(gpa);

    // 6x6 curved surface patch
    for (0..6) |gy| {
        for (0..6) |gx| {
            const x: f64 = @floatFromInt(gx);
            const y: f64 = @floatFromInt(gy);
            const z: f64 = 5.0 + 0.1 * (x - 2.5) * (x - 2.5);
            try pts.append(gpa, Vec3.init(x, y, z));
            try cols.append(gpa, .{ .r = @intCast(gx * 40), .g = @intCast(gy * 40), .b = 180 });
        }
    }
    // Add an isolated floating outlier far from the surface
    try pts.append(gpa, Vec3.init(50.0, 50.0, 50.0));
    try cols.append(gpa, .{ .r = 255, .g = 0, .b = 0 });

    const poses = [_]CameraPose{CameraPose.identity};
    var mesh = try buildColoredMesh(gpa, pts.items, cols.items, &poses, .{});
    defer mesh.deinit(gpa);

    // Outlier removed; 36 grid vertices form 2*(5*5) = 50 triangles
    try std.testing.expectEqual(@as(usize, 36), mesh.points.len);
    try std.testing.expectEqual(@as(usize, 50), mesh.faces.len);
}

test "subdivideAndProjectTexture subdivides mesh and projects photo RGB onto vertices" {
    const gpa = std.testing.allocator;
    const w: u32 = 64;
    const h: u32 = 64;
    const k = Mat3.fromRows(
        .{ 64, 0, 32 },
        .{ 0, 64, 32 },
        .{ 0, 0, 1 },
    );

    const rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb);
    const gray = try gpa.alloc(u8, w * h);
    defer gpa.free(gray);
    for (0..h) |y| {
        for (0..w) |x| {
            const idx = y * w + x;
            rgb[idx * 3 + 0] = @intCast(x * 3);
            rgb[idx * 3 + 1] = @intCast(y * 3);
            rgb[idx * 3 + 2] = 200;
            gray[idx] = 128;
        }
    }

    const images = [_]dataset.ImageData{
        .{ .width = w, .height = h, .rgb = rgb, .gray = gray },
    };
    const poses = [_]CameraPose{CameraPose.identity};
    const reg = [_]bool{true};

    const pts = try gpa.dupe(Vec3, &.{
        Vec3.init(-1.0, -1.0, 4.0),
        Vec3.init(1.0, -1.0, 4.0),
        Vec3.init(0.0, 1.0, 4.0),
    });
    const cols = try gpa.dupe(Color, &.{
        .{ .r = 0, .g = 0, .b = 0 },
        .{ .r = 0, .g = 0, .b = 0 },
        .{ .r = 0, .g = 0, .b = 0 },
    });
    const faces = try gpa.dupe(Triangle, &.{.{ 0, 1, 2 }});

    var mesh = Mesh{ .points = pts, .colors = cols, .faces = faces };
    defer mesh.deinit(gpa);

    try subdivideAndProjectTexture(gpa, &mesh, k, &images, &poses, &reg, 1);
    try std.testing.expectEqual(@as(usize, 6), mesh.points.len);
    try std.testing.expectEqual(@as(usize, 4), mesh.faces.len);
    // Center midpoint of bottom edge (0, -1, 4) projects to (u=32, v=16) -> R=96, G=48, B=200
    try std.testing.expectEqual(@as(u8, 96), mesh.colors[3].r);
    try std.testing.expectEqual(@as(u8, 48), mesh.colors[3].g);
    try std.testing.expectEqual(@as(u8, 200), mesh.colors[3].b);
}

test "fillInteriorHoles fills planar interior cavity while leaving outer perimeter open" {
    const gpa = std.testing.allocator;
    const poses = [_]CameraPose{CameraPose.identity};
    const basis = computeViewBasis(&.{Vec3.init(0, 0, 5)}, &poses);

    // 4x4 grid on z = 5 with center 4 triangles removed, leaving an 8-vertex CCW interior hole
    var pts: std.ArrayList(Vec3) = .empty;
    defer pts.deinit(gpa);
    var pts2d: std.ArrayList(Vec2) = .empty;
    defer pts2d.deinit(gpa);
    for (0..4) |gy| {
        for (0..4) |gx| {
            const fx: f64 = @floatFromInt(gx);
            const fy: f64 = @floatFromInt(gy);
            const pt = Vec3.init(fx, fy, 5.0);
            try pts.append(gpa, pt);
            const d = pt.sub(basis.origin);
            try pts2d.append(gpa, Vec2.init(d.dot(basis.u), d.dot(basis.v)));
        }
    }

    var faces: std.ArrayList(Triangle) = .empty;
    defer faces.deinit(gpa);
    for (0..3) |gy| {
        for (0..3) |gx| {
            if (gx == 1 and gy == 1) continue;
            const va = gy * 4 + gx;
            const vb = gy * 4 + gx + 1;
            const vc = (gy + 1) * 4 + gx + 1;
            const vd = (gy + 1) * 4 + gx;
            // Emit CW triangles in 2D (u, v)
            const t1 = Triangle{ va, vb, vc };
            const t2 = Triangle{ va, vc, vd };
            for ([2]Triangle{ t1, t2 }) |tri| {
                const c2 = cross2d(pts2d.items[tri[1]].sub(pts2d.items[tri[0]]), pts2d.items[tri[2]].sub(pts2d.items[tri[0]]));
                if (c2 < 0) {
                    try faces.append(gpa, tri);
                } else {
                    try faces.append(gpa, .{ tri[0], tri[2], tri[1] });
                }
            }
        }
    }

    try std.testing.expectEqual(@as(usize, 16), faces.items.len);
    try fillInteriorHoles(gpa, pts.items, pts2d.items, basis, &faces, 1.0, .{});
    // Center 1x1 square (4-vertex CCW loop) is filled by 2 triangles -> 18 total faces
    try std.testing.expectEqual(@as(usize, 18), faces.items.len);

    for (faces.items) |tri| {
        const pa = pts.items[tri[0]];
        const pb = pts.items[tri[1]];
        const pc = pts.items[tri[2]];
        const n = pb.sub(pa).cross(pc.sub(pa)).normalize();
        try std.testing.expect(n.dot(basis.forward) <= -0.15);
    }
}

test "fillInteriorHoles skips depth-jump blind-spot holes" {
    const gpa = std.testing.allocator;
    const poses = [_]CameraPose{CameraPose.identity};
    const basis = computeViewBasis(&.{Vec3.init(0, 0, 5)}, &poses);

    // 4x4 grid where vertex (1,1) jumps far in depth relative to med_edge = 1.0
    var pts: std.ArrayList(Vec3) = .empty;
    defer pts.deinit(gpa);
    var pts2d: std.ArrayList(Vec2) = .empty;
    defer pts2d.deinit(gpa);
    for (0..4) |gy| {
        for (0..4) |gx| {
            const fx: f64 = @floatFromInt(gx);
            const fy: f64 = @floatFromInt(gy);
            const z: f64 = if (gx == 1 and gy == 1) 20.0 else 5.0;
            const pt = Vec3.init(fx, fy, z);
            try pts.append(gpa, pt);
            const d = pt.sub(basis.origin);
            try pts2d.append(gpa, Vec2.init(d.dot(basis.u), d.dot(basis.v)));
        }
    }

    var faces: std.ArrayList(Triangle) = .empty;
    defer faces.deinit(gpa);
    for (0..3) |gy| {
        for (0..3) |gx| {
            if ((gx == 0 or gx == 1) and (gy == 0 or gy == 1)) continue;
            const va = gy * 4 + gx;
            const vb = gy * 4 + gx + 1;
            const vc = (gy + 1) * 4 + gx + 1;
            const vd = (gy + 1) * 4 + gx;
            for ([2]Triangle{ .{ va, vb, vc }, .{ va, vc, vd } }) |tri| {
                const c2 = cross2d(pts2d.items[tri[1]].sub(pts2d.items[tri[0]]), pts2d.items[tri[2]].sub(pts2d.items[tri[0]]));
                if (c2 < 0) {
                    try faces.append(gpa, tri);
                } else {
                    try faces.append(gpa, .{ tri[0], tri[2], tri[1] });
                }
            }
        }
    }

    try std.testing.expectEqual(@as(usize, 10), faces.items.len);
    try fillInteriorHoles(gpa, pts.items, pts2d.items, basis, &faces, 1.0, .{ .max_hole_depth_factor = 6.5 });
    // Depth span of the interior hole is 15.0 > 6.5 * 1.0, so it remains open
    try std.testing.expectEqual(@as(usize, 10), faces.items.len);
}

test "earClipLoop2D triangulates concave CCW polygon into CW faces" {
    const gpa = std.testing.allocator;
    const pts2d = [_]Vec2{
        Vec2.init(0.0, 0.0),
        Vec2.init(2.0, 0.0),
        Vec2.init(2.0, 1.0),
        Vec2.init(1.0, 1.0),
        Vec2.init(1.0, 2.0),
        Vec2.init(0.0, 2.0),
    };
    const pts3d = [_]Vec3{
        Vec3.init(0.0, 0.0, 5.0),
        Vec3.init(2.0, 0.0, 5.0),
        Vec3.init(2.0, 1.0, 5.0),
        Vec3.init(1.0, 1.0, 5.0),
        Vec3.init(1.0, 2.0, 5.0),
        Vec3.init(0.0, 2.0, 5.0),
    };
    const basis = ViewBasis{
        .origin = Vec3.zero,
        .u = Vec3.init(1, 0, 0),
        .v = Vec3.init(0, 1, 0),
        .forward = Vec3.init(0, 0, 1),
    };
    const loop = [_]usize{ 0, 1, 2, 3, 4, 5 };

    var out: std.ArrayList(Triangle) = .empty;
    defer out.deinit(gpa);

    const ok = try earClipLoop2D(gpa, &pts3d, &pts2d, basis, &loop, 100.0, 0.15, &out);
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(usize, 4), out.items.len);

    var total_area: f64 = 0.0;
    for (out.items) |tri| {
        const c2 = cross2d(pts2d[tri[1]].sub(pts2d[tri[0]]), pts2d[tri[2]].sub(pts2d[tri[0]]));
        try std.testing.expect(c2 < 0.0);
        total_area += -0.5 * c2;
    }
    try std.testing.expectApproxEqAbs(3.0, total_area, 1e-9);
}

test "completeMeshWithPhotoVerify bridges consistent holes and rejects conflicts or single-view blind spots" {
    const gpa = std.testing.allocator;
    const w: u32 = 128;
    const h: u32 = 128;
    const k = Mat3.fromRows(
        .{ 100, 0, 64 },
        .{ 0, 100, 64 },
        .{ 0, 0, 1 },
    );

    const poses = [_]CameraPose{
        CameraPose.identity,
        .{ .r = Mat3.identity, .t = Vec3.init(-0.5, 0.0, 0.0) },
    };

    const rgb0 = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb0);
    const rgb1 = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb1);
    const rgb1_conflict = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb1_conflict);
    const gray = try gpa.alloc(u8, w * h);
    defer gpa.free(gray);
    @memset(gray, 128);

    for (0..h) |py| {
        for (0..w) |px| {
            const idx = py * w + px;
            const wx0 = (@as(f64, @floatFromInt(px)) - 64.0) * 0.05;
            const wy0 = (@as(f64, @floatFromInt(py)) - 64.0) * 0.05;
            rgb0[idx * 3 + 0] = @intFromFloat(std.math.clamp(@round((wx0 + 3.0) * 35.0), 0.0, 255.0));
            rgb0[idx * 3 + 1] = @intFromFloat(std.math.clamp(@round((wy0 + 3.0) * 35.0), 0.0, 255.0));
            rgb0[idx * 3 + 2] = 140;

            const wx1 = (@as(f64, @floatFromInt(px)) - 64.0) * 0.05 + 0.5;
            const wy1 = wy0;
            rgb1[idx * 3 + 0] = @intFromFloat(std.math.clamp(@round((wx1 + 3.0) * 35.0), 0.0, 255.0));
            rgb1[idx * 3 + 1] = @intFromFloat(std.math.clamp(@round((wy1 + 3.0) * 35.0), 0.0, 255.0));
            rgb1[idx * 3 + 2] = 140;

            rgb1_conflict[idx * 3 + 0] = 255 - rgb1[idx * 3 + 0];
            rgb1_conflict[idx * 3 + 1] = 255 - rgb1[idx * 3 + 1];
            rgb1_conflict[idx * 3 + 2] = 30;
        }
    }

    var pts: std.ArrayList(Vec3) = .empty;
    defer pts.deinit(gpa);
    var cols: std.ArrayList(Color) = .empty;
    defer cols.deinit(gpa);

    const x_cols = [_]f64{ -2.0, -1.8, -1.6, -1.4, 0.4, 0.6, 0.8, 1.0 };
    for (0..4) |gy| {
        const y = -0.3 + @as(f64, @floatFromInt(gy)) * 0.2;
        for (x_cols) |x| {
            try pts.append(gpa, Vec3.init(x, y, 5.0));
            try cols.append(gpa, .{ .r = 120, .g = 120, .b = 140 });
        }
    }

    const reg_both = [_]bool{ true, true };
    const images_ok = [_]dataset.ImageData{
        .{ .width = w, .height = h, .rgb = rgb0, .gray = gray },
        .{ .width = w, .height = h, .rgb = rgb1, .gray = gray },
    };

    const mesh_cfg = MeshConfig{ .outlier_std_mult = 3.0, .fill_holes = false };
    var base_mesh = try buildColoredMesh(gpa, pts.items, cols.items, &poses, mesh_cfg);
    defer base_mesh.deinit(gpa);
    const base_faces = base_mesh.faces.len;
    try std.testing.expectEqual(@as(usize, 36), base_faces);

    try completeMeshWithPhotoVerify(gpa, &base_mesh, k, &images_ok, &poses, &reg_both, &poses, .{});
    try std.testing.expect(base_mesh.faces.len > base_faces);

    var conflict_mesh = try buildColoredMesh(gpa, pts.items, cols.items, &poses, mesh_cfg);
    defer conflict_mesh.deinit(gpa);
    const images_bad = [_]dataset.ImageData{
        .{ .width = w, .height = h, .rgb = rgb0, .gray = gray },
        .{ .width = w, .height = h, .rgb = rgb1_conflict, .gray = gray },
    };
    try completeMeshWithPhotoVerify(gpa, &conflict_mesh, k, &images_bad, &poses, &reg_both, &poses, .{});
    try std.testing.expectEqual(base_faces, conflict_mesh.faces.len);

    var single_view_mesh = try buildColoredMesh(gpa, pts.items, cols.items, &poses, mesh_cfg);
    defer single_view_mesh.deinit(gpa);
    const reg_one = [_]bool{ true, false };
    try completeMeshWithPhotoVerify(gpa, &single_view_mesh, k, &images_ok, &poses, &reg_one, &poses, .{});
    try std.testing.expectEqual(base_faces, single_view_mesh.faces.len);
}

test "completeMeshWithPhotoVerify rejects depth-cliff bridges even with uniform texture" {
    const gpa = std.testing.allocator;
    const w: u32 = 64;
    const h: u32 = 64;
    const k = Mat3.fromRows(
        .{ 64, 0, 32 },
        .{ 0, 64, 32 },
        .{ 0, 0, 1 },
    );

    const poses = [_]CameraPose{
        CameraPose.identity,
        .{ .r = Mat3.identity, .t = Vec3.init(-0.4, 0.0, 0.0) },
    };

    const rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb);
    @memset(rgb, 180);
    const gray = try gpa.alloc(u8, w * h);
    defer gpa.free(gray);
    @memset(gray, 180);

    const images = [_]dataset.ImageData{
        .{ .width = w, .height = h, .rgb = rgb, .gray = gray },
        .{ .width = w, .height = h, .rgb = rgb, .gray = gray },
    };

    var pts: std.ArrayList(Vec3) = .empty;
    defer pts.deinit(gpa);
    var cols: std.ArrayList(Color) = .empty;
    defer cols.deinit(gpa);

    for (0..4) |gy| {
        const y = -0.3 + @as(f64, @floatFromInt(gy)) * 0.2;
        for (0..4) |gx| {
            const x = -0.8 + @as(f64, @floatFromInt(gx)) * 0.2;
            try pts.append(gpa, Vec3.init(x, y, 5.0));
            try cols.append(gpa, .{ .r = 180, .g = 180, .b = 180 });
        }
    }
    for (0..4) |gy| {
        const y = -0.3 + @as(f64, @floatFromInt(gy)) * 0.2;
        for (0..4) |gx| {
            const x = 0.0 + @as(f64, @floatFromInt(gx)) * 0.2;
            try pts.append(gpa, Vec3.init(x, y, 6.5));
            try cols.append(gpa, .{ .r = 180, .g = 180, .b = 180 });
        }
    }

    const reg = [_]bool{ true, true };
    var mesh = try buildColoredMesh(gpa, pts.items, cols.items, &poses, .{ .outlier_std_mult = 3.0, .fill_holes = false });
    defer mesh.deinit(gpa);
    const base_faces = mesh.faces.len;
    try std.testing.expectEqual(@as(usize, 36), base_faces);

    try completeMeshWithPhotoVerify(gpa, &mesh, k, &images, &poses, &reg, &poses, .{});
    try std.testing.expectEqual(base_faces, mesh.faces.len);
}
