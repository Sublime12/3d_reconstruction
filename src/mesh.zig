const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const dataset = @import("dataset.zig");
const Color = @import("io_ply.zig").Color;

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
    min_view_cos: f64 = 0.15,
    dedup_2d_factor: f64 = 0.0,
    laplacian_iters: usize = 0,
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
        const view_cos = n_unit.dot(basis.forward);
        if (@abs(view_cos) < config.min_view_cos) continue;

        // Orient winding so normal faces toward cameras (-forward)
        if (view_cos > 0) {
            try faces.append(allocator, .{ tri[0], tri[2], tri[1] });
        } else {
            try faces.append(allocator, tri);
        }
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

    // Project every vertex onto the best-facing registered photo to sample RGB color
    const new_cols = try allocator.alloc(Color, verts.items.len);
    errdefer allocator.free(new_cols);

    for (verts.items, normals, 0..) |pt, nrm, vi| {
        var best_score: f64 = -1.0;
        var best_col: ?Color = null;

        for (images, poses, registered, cam_centers) |img, pose, is_reg, cam_c| {
            if (!is_reg) continue;
            const uv = lie.projectPoint(k, pose, pt) orelse continue;
            const wf: f64 = @floatFromInt(img.width);
            const hf: f64 = @floatFromInt(img.height);
            if (uv.x < 1.0 or uv.x >= wf - 2.0 or uv.y < 1.0 or uv.y >= hf - 2.0) continue;

            const ray_to_cam = cam_c.sub(pt).normalize();
            const cos_view = if (nrm.normSq() > 0.5) @max(0.05, nrm.dot(ray_to_cam)) else 1.0;
            const dx = (uv.x - wf * 0.5) / wf;
            const dy = (uv.y - hf * 0.5) / hf;
            const centrality = @max(0.1, 1.0 - @sqrt(dx * dx + dy * dy));
            const score = cos_view * centrality;

            if (score > best_score) {
                best_score = score;
                best_col = sampleBilinearRgb(img, uv.x, uv.y);
            }
        }

        if (best_col) |col| {
            new_cols[vi] = col;
        } else if (vi < mesh.colors.len) {
            new_cols[vi] = mesh.colors[vi];
        } else {
            new_cols[vi] = .{ .r = 160, .g = 160, .b = 160 };
        }
    }

    allocator.free(mesh.points);
    allocator.free(mesh.colors);
    allocator.free(mesh.faces);
    mesh.points = try verts.toOwnedSlice(allocator);
    mesh.colors = new_cols;
    mesh.faces = try faces.toOwnedSlice(allocator);
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
