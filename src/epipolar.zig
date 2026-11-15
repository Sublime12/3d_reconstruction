const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const DynMat = @import("math/mat.zig").DynMat;
const svd = @import("math/mat.zig").svd;
const svd3x3 = @import("math/mat.zig").svd3x3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const triangulation = @import("triangulation.zig");

const HartleyNorm = struct {
    pts: []Vec2,
    t: Mat3,

    fn deinit(self: *HartleyNorm, allocator: std.mem.Allocator) void {
        allocator.free(self.pts);
    }
};

fn hartleyNormalize(allocator: std.mem.Allocator, pts: []const Vec2) !?HartleyNorm {
    if (pts.len == 0) return null;
    var mx: f64 = 0;
    var my: f64 = 0;
    for (pts) |p| {
        mx += p.x;
        my += p.y;
    }
    const n_f: f64 = @floatFromInt(pts.len);
    mx /= n_f;
    my /= n_f;

    var mean_dist: f64 = 0;
    for (pts) |p| {
        const dx = p.x - mx;
        const dy = p.y - my;
        mean_dist += @sqrt(dx * dx + dy * dy);
    }
    mean_dist /= n_f;
    if (mean_dist < 1e-12) return null;

    const s = std.math.sqrt2 / mean_dist;
    const out = try allocator.alloc(Vec2, pts.len);
    for (pts, 0..) |p, i| {
        out[i] = Vec2.init((p.x - mx) * s, (p.y - my) * s);
    }

    const t = Mat3.fromRows(
        .{ s, 0, -s * mx },
        .{ 0, s, -s * my },
        .{ 0, 0, 1 },
    );
    return .{ .pts = out, .t = t };
}

pub fn pixelToNormalized(k_inv: Mat3, pt: Vec2) Vec2 {
    const v = k_inv.mulVec(Vec3.init(pt.x, pt.y, 1.0));
    return Vec2.init(v.x / v.z, v.y / v.z);
}

/// Normalized 8-point algorithm on already-normalized camera coordinates (x_n = K^-1 * x).
/// Enforces Essential matrix singular values (1, 1, 0).
pub fn eightPointEssential(
    allocator: std.mem.Allocator,
    norm_pts1: []const Vec2,
    norm_pts2: []const Vec2,
) !?Mat3 {
    std.debug.assert(norm_pts1.len == norm_pts2.len);
    if (norm_pts1.len < 8) return null;

    var h1 = (try hartleyNormalize(allocator, norm_pts1)) orelse return null;
    defer h1.deinit(allocator);
    var h2 = (try hartleyNormalize(allocator, norm_pts2)) orelse return null;
    defer h2.deinit(allocator);

    var a = try DynMat.init(allocator, norm_pts1.len, 9);
    defer a.deinit(allocator);

    for (h1.pts, h2.pts, 0..) |p1, p2, r| {
        a.set(r, 0, p2.x * p1.x);
        a.set(r, 1, p2.x * p1.y);
        a.set(r, 2, p2.x);
        a.set(r, 3, p2.y * p1.x);
        a.set(r, 4, p2.y * p1.y);
        a.set(r, 5, p2.y);
        a.set(r, 6, p1.x);
        a.set(r, 7, p1.y);
        a.set(r, 8, 1.0);
    }

    var res = try svd(allocator, a);
    defer res.deinit(allocator);

    const e_raw = Mat3.fromRows(
        .{ res.v.get(0, 8), res.v.get(1, 8), res.v.get(2, 8) },
        .{ res.v.get(3, 8), res.v.get(4, 8), res.v.get(5, 8) },
        .{ res.v.get(6, 8), res.v.get(7, 8), res.v.get(8, 8) },
    );

    const e_unnorm = h2.t.transpose().mul(e_raw).mul(h1.t);
    const s3 = svd3x3(e_unnorm);
    const diag_110 = Mat3.fromRows(
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 0 },
    );
    return s3.u.mul(diag_110).mul(s3.vt);
}

/// Squared Sampson distance for Fundamental (or Essential) matrix F:
/// d^2 = (x2^T F x1)^2 / ( (Fx1)_0^2 + (Fx1)_1^2 + (F^T x2)_0^2 + (F^T x2)_1^2 )
pub fn sampsonErrorSq(f: Mat3, x1: Vec2, x2: Vec2) f64 {
    const p1 = Vec3.init(x1.x, x1.y, 1.0);
    const p2 = Vec3.init(x2.x, x2.y, 1.0);
    const fp1 = f.mulVec(p1);
    const ftp2 = f.transpose().mulVec(p2);
    const num = p2.dot(fp1);
    const denom = fp1.x * fp1.x + fp1.y * fp1.y + ftp2.x * ftp2.x + ftp2.y * ftp2.y;
    if (denom < 1e-18) return std.math.inf(f64);
    return (num * num) / denom;
}

pub fn essentialToFundamental(k_inv: Mat3, e: Mat3) Mat3 {
    return k_inv.transpose().mul(e).mul(k_inv);
}

pub const EssentialRansacResult = struct {
    e: Mat3,
    inliers: []usize,

    pub fn deinit(self: *EssentialRansacResult, allocator: std.mem.Allocator) void {
        allocator.free(self.inliers);
        self.* = undefined;
    }
};

pub fn estimateEssentialRansac(
    allocator: std.mem.Allocator,
    k: Mat3,
    pts1: []const Vec2,
    pts2: []const Vec2,
    iterations: usize,
    pixel_thresh: f64,
    seed: u64,
) !?EssentialRansacResult {
    std.debug.assert(pts1.len == pts2.len);
    if (pts1.len < 8) return null;

    const k_inv = k.inv() orelse return null;
    const norm1 = try allocator.alloc(Vec2, pts1.len);
    defer allocator.free(norm1);
    const norm2 = try allocator.alloc(Vec2, pts2.len);
    defer allocator.free(norm2);

    for (pts1, pts2, 0..) |p1, p2, i| {
        norm1[i] = pixelToNormalized(k_inv, p1);
        norm2[i] = pixelToNormalized(k_inv, p2);
    }

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const thresh_sq = pixel_thresh * pixel_thresh;

    var best_e: ?Mat3 = null;
    var best_inlier_count: usize = 0;

    var sample_idx: [8]usize = undefined;
    var sample1: [8]Vec2 = undefined;
    var sample2: [8]Vec2 = undefined;

    for (0..iterations) |_| {
        // Sample 8 distinct indices
        var count: usize = 0;
        while (count < 8) {
            const candidate = rng.uintLessThan(usize, pts1.len);
            var dup = false;
            for (0..count) |j| {
                if (sample_idx[j] == candidate) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                sample_idx[count] = candidate;
                sample1[count] = norm1[candidate];
                sample2[count] = norm2[candidate];
                count += 1;
            }
        }

        const e_cand = (try eightPointEssential(allocator, &sample1, &sample2)) orelse continue;
        const f_cand = essentialToFundamental(k_inv, e_cand);

        var inlier_count: usize = 0;
        for (pts1, pts2) |p1, p2| {
            if (sampsonErrorSq(f_cand, p1, p2) <= thresh_sq) {
                inlier_count += 1;
            }
        }

        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_e = e_cand;
        }
    }

    const e_initial = best_e orelse return null;
    if (best_inlier_count < 8) return null;

    // Collect inliers & refine E on all inliers
    const f_initial = essentialToFundamental(k_inv, e_initial);
    var inliers: std.ArrayList(usize) = .empty;
    defer inliers.deinit(allocator);
    var in_n1: std.ArrayList(Vec2) = .empty;
    defer in_n1.deinit(allocator);
    var in_n2: std.ArrayList(Vec2) = .empty;
    defer in_n2.deinit(allocator);

    for (pts1, pts2, 0..) |p1, p2, i| {
        if (sampsonErrorSq(f_initial, p1, p2) <= thresh_sq) {
            try inliers.append(allocator, i);
            try in_n1.append(allocator, norm1[i]);
            try in_n2.append(allocator, norm2[i]);
        }
    }

    const e_refined = (try eightPointEssential(allocator, in_n1.items, in_n2.items)) orelse e_initial;
    const f_refined = essentialToFundamental(k_inv, e_refined);

    inliers.clearRetainingCapacity();
    for (pts1, pts2, 0..) |p1, p2, i| {
        if (sampsonErrorSq(f_refined, p1, p2) <= thresh_sq) {
            try inliers.append(allocator, i);
        }
    }

    return .{
        .e = e_refined,
        .inliers = try inliers.toOwnedSlice(allocator),
    };
}

pub fn decomposeEssential(e: Mat3) [4]CameraPose {
    const s = svd3x3(e);
    var u = s.u;
    var vt = s.vt;
    if (u.det() < 0) u = u.scale(-1.0);
    if (vt.det() < 0) vt = vt.scale(-1.0);

    const w = Mat3.fromRows(
        .{ 0, -1, 0 },
        .{ 1, 0, 0 },
        .{ 0, 0, 1 },
    );

    var r1 = u.mul(w).mul(vt);
    var r2 = u.mul(w.transpose()).mul(vt);
    if (r1.det() < 0) r1 = r1.scale(-1.0);
    if (r2.det() < 0) r2 = r2.scale(-1.0);

    const t = Vec3.init(u.m[0][2], u.m[1][2], u.m[2][2]).normalize();
    return .{
        .{ .r = r1, .t = t },
        .{ .r = r1, .t = t.neg() },
        .{ .r = r2, .t = t },
        .{ .r = r2, .t = t.neg() },
    };
}

pub const PoseRecoveryResult = struct {
    pose: CameraPose,
    positive_depth_count: usize,
};

pub fn recoverPoseFromEssential(
    allocator: std.mem.Allocator,
    k: Mat3,
    e: Mat3,
    pts1: []const Vec2,
    pts2: []const Vec2,
    inliers: []const usize,
) !?PoseRecoveryResult {
    const candidates = decomposeEssential(e);
    const p1 = triangulation.makeProjectionMatrix(k, CameraPose.identity);

    var best_idx: ?usize = null;
    var best_count: usize = 0;

    for (candidates, 0..) |cand, c_idx| {
        const p2 = triangulation.makeProjectionMatrix(k, cand);
        var pos_count: usize = 0;
        for (inliers) |idx| {
            const pt3d = (try triangulation.triangulatePoint(allocator, p1, p2, pts1[idx], pts2[idx])) orelse continue;
            const z1 = pt3d.z;
            const z2 = cand.transformPoint(pt3d).z;
            if (z1 > 0 and z2 > 0) {
                pos_count += 1;
            }
        }
        if (pos_count > best_count) {
            best_count = pos_count;
            best_idx = c_idx;
        }
    }

    const idx = best_idx orelse return null;
    return .{
        .pose = candidates[idx],
        .positive_depth_count = best_count,
    };
}

test "Essential RANSAC and pose recovery on synthetic 2-view scene with outliers" {
    const gpa = std.testing.allocator;
    const k = Mat3.fromRows(
        .{ 800, 0, 320 },
        .{ 0, 800, 240 },
        .{ 0, 0, 1 },
    );
    const gt_pose = CameraPose{
        .r = lie.so3Exp(Vec3.init(0.04, -0.15, 0.03)),
        .t = Vec3.init(-1.2, 0.1, 0.25).normalize(),
    };

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const total = 60;
    const n_inliers = 48;
    var pts1: [total]Vec2 = undefined;
    var pts2: [total]Vec2 = undefined;

    for (0..n_inliers) |i| {
        const x = Vec3.init(
            (rng.float(f64) - 0.5) * 4.0,
            (rng.float(f64) - 0.5) * 3.0,
            4.0 + rng.float(f64) * 5.0,
        );
        pts1[i] = lie.projectPoint(k, CameraPose.identity, x).?;
        pts2[i] = lie.projectPoint(k, gt_pose, x).?;
    }
    for (n_inliers..total) |i| {
        pts1[i] = Vec2.init(rng.float(f64) * 640.0, rng.float(f64) * 480.0);
        pts2[i] = Vec2.init(rng.float(f64) * 640.0, rng.float(f64) * 480.0);
    }

    var ransac_res = (try estimateEssentialRansac(gpa, k, &pts1, &pts2, 300, 1.5, 123)).?;
    defer ransac_res.deinit(gpa);

    try std.testing.expect(ransac_res.inliers.len >= n_inliers - 2);

    const rec = (try recoverPoseFromEssential(gpa, k, ransac_res.e, &pts1, &pts2, ransac_res.inliers)).?;
    try std.testing.expect(rec.positive_depth_count >= n_inliers - 2);

    const r_err = rec.pose.r.mul(gt_pose.r.transpose());
    const angle_err = lie.so3Log(r_err).norm();
    try std.testing.expect(angle_err < 1e-3);
    try std.testing.expect(rec.pose.t.dot(gt_pose.t) > 0.999);
}
