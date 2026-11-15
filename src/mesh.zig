const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const CameraPose = @import("math/lie.zig").CameraPose;
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

    // 2. Compute orthonormal view plane from average camera forward axis
    const basis = computeViewBasis(filtered.points, poses);
    const pts2d = try allocator.alloc(Vec2, filtered.points.len);
    defer allocator.free(pts2d);
    for (filtered.points, 0..) |pt, i| {
        const d = pt.sub(basis.origin);
        pts2d[i] = Vec2.init(d.dot(basis.u), d.dot(basis.v));
    }

    // 3. 2D Bowyer-Watson Delaunay triangulation
    const raw_faces = try triangulateDelaunay2D(allocator, pts2d);
    defer allocator.free(raw_faces);

    // 4. Prune triangles spanning depth jumps or empty gaps using 3D median NN distance
    const max_edge = filtered.median_nn * config.max_edge_factor;
    const max_edge_sq = max_edge * max_edge;

    var faces: std.ArrayList(Triangle) = .empty;
    errdefer faces.deinit(allocator);

    for (raw_faces) |tri| {
        const a = filtered.points[tri[0]];
        const b = filtered.points[tri[1]];
        const c = filtered.points[tri[2]];

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

    return .{
        .points = filtered.points,
        .colors = filtered.colors,
        .faces = try faces.toOwnedSlice(allocator),
    };
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

    const topk = try allocator.alloc(f64, k);
    defer allocator.free(topk);

    for (0..n) |i| {
        @memset(topk, std.math.inf(f64));
        const pi = points[i];
        for (0..n) |j| {
            if (i == j) continue;
            const d_sq = pi.sub(points[j]).normSq();
            if (d_sq < topk[k - 1]) {
                topk[k - 1] = d_sq;
                var pos: usize = k - 1;
                while (pos > 0 and topk[pos] < topk[pos - 1]) : (pos -= 1) {
                    std.mem.swap(f64, &topk[pos], &topk[pos - 1]);
                }
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
    active: bool,
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
        return .{ .v = .{ v0, v1, v2 }, .cx = 0, .cy = 0, .r_sq = -1.0, .active = true };
    }

    const a2 = ax * ax + ay * ay;
    const b2 = bx * bx + by * by;
    const c2 = cx * cx + cy * cy;

    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    const dx = ax - ux;
    const dy = ay - uy;
    return .{
        .v = .{ v0, v1, v2 },
        .cx = ux,
        .cy = uy,
        .r_sq = dx * dx + dy * dy,
        .active = true,
    };
}

/// Bowyer-Watson 2D Delaunay triangulation.
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

    var tris: std.ArrayList(BWTriangle) = .empty;
    defer tris.deinit(allocator);
    try tris.append(allocator, makeBWTriangle(all_pts, n, n + 1, n + 2));

    var bad_indices: std.ArrayList(usize) = .empty;
    defer bad_indices.deinit(allocator);
    var edges: std.ArrayList([2]usize) = .empty;
    defer edges.deinit(allocator);

    for (0..n) |p_idx| {
        const p = all_pts[p_idx];
        bad_indices.clearRetainingCapacity();

        for (tris.items, 0..) |t, t_idx| {
            if (!t.active or t.r_sq < 0) continue;
            const ex = p.x - t.cx;
            const ey = p.y - t.cy;
            if (ex * ex + ey * ey <= t.r_sq + 1e-12) {
                try bad_indices.append(allocator, t_idx);
            }
        }

        edges.clearRetainingCapacity();
        for (bad_indices.items) |b_idx| {
            const tv = tris.items[b_idx].v;
            const tri_edges = [3][2]usize{
                .{ tv[0], tv[1] },
                .{ tv[1], tv[2] },
                .{ tv[2], tv[0] },
            };
            for (tri_edges) |e| {
                var shared = false;
                for (bad_indices.items) |other_idx| {
                    if (other_idx == b_idx) continue;
                    const ov = tris.items[other_idx].v;
                    if (triHasEdge(ov, e[0], e[1])) {
                        shared = true;
                        break;
                    }
                }
                if (!shared) {
                    try edges.append(allocator, e);
                }
            }
        }

        for (bad_indices.items) |b_idx| {
            tris.items[b_idx].active = false;
        }

        // Reuse inactive slots before appending
        var e_idx: usize = 0;
        for (tris.items) |*t| {
            if (e_idx >= edges.items.len) break;
            if (!t.active) {
                const e = edges.items[e_idx];
                t.* = makeBWTriangle(all_pts, e[0], e[1], p_idx);
                e_idx += 1;
            }
        }
        while (e_idx < edges.items.len) : (e_idx += 1) {
            const e = edges.items[e_idx];
            try tris.append(allocator, makeBWTriangle(all_pts, e[0], e[1], p_idx));
        }
    }

    var out: std.ArrayList(Triangle) = .empty;
    errdefer out.deinit(allocator);
    for (tris.items) |t| {
        if (!t.active) continue;
        if (t.v[0] >= n or t.v[1] >= n or t.v[2] >= n) continue;
        if (t.r_sq < 0) continue;
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
