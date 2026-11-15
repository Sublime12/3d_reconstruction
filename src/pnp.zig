const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const DynMat = @import("math/mat.zig").DynMat;
const svd = @import("math/mat.zig").svd;
const svd3x3 = @import("math/mat.zig").svd3x3;
const solveLinearSystemInPlace = @import("math/mat.zig").solveLinearSystemInPlace;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;

/// Solve PnP via normalized DLT for N >= 6 2D-3D correspondences.
pub fn solvePnpDlt(
    allocator: std.mem.Allocator,
    k: Mat3,
    pts3d: []const Vec3,
    pts2d: []const Vec2,
) !?CameraPose {
    std.debug.assert(pts3d.len == pts2d.len);
    if (pts3d.len < 6) return null;

    const k_inv = k.inv() orelse return null;

    // Normalize 3D points for numerical stability
    var mean3 = Vec3.zero;
    for (pts3d) |p| mean3 = mean3.add(p);
    mean3 = mean3.scale(1.0 / @as(f64, @floatFromInt(pts3d.len)));

    var mean_dist: f64 = 0;
    for (pts3d) |p| mean_dist += p.sub(mean3).norm();
    mean_dist /= @as(f64, @floatFromInt(pts3d.len));
    if (mean_dist < 1e-12) return null;
    const s3 = @sqrt(3.0) / mean_dist;

    var a = try DynMat.init(allocator, pts3d.len * 2, 12);
    defer a.deinit(allocator);

    for (pts3d, pts2d, 0..) |x_w, uv, i| {
        const xn = k_inv.mulVec(Vec3.init(uv.x, uv.y, 1.0));
        const u = xn.x / xn.z;
        const v = xn.y / xn.z;

        const p = x_w.sub(mean3).scale(s3);
        const r0 = i * 2;
        const r1 = r0 + 1;

        a.set(r0, 0, p.x);
        a.set(r0, 1, p.y);
        a.set(r0, 2, p.z);
        a.set(r0, 3, 1.0);
        a.set(r0, 8, -u * p.x);
        a.set(r0, 9, -u * p.y);
        a.set(r0, 10, -u * p.z);
        a.set(r0, 11, -u);

        a.set(r1, 4, p.x);
        a.set(r1, 5, p.y);
        a.set(r1, 6, p.z);
        a.set(r1, 7, 1.0);
        a.set(r1, 8, -v * p.x);
        a.set(r1, 9, -v * p.y);
        a.set(r1, 10, -v * p.z);
        a.set(r1, 11, -v);
    }

    var res = try svd(allocator, a);
    defer res.deinit(allocator);

    var m_mat = Mat3.fromRows(
        .{ res.v.get(0, 11), res.v.get(1, 11), res.v.get(2, 11) },
        .{ res.v.get(4, 11), res.v.get(5, 11), res.v.get(6, 11) },
        .{ res.v.get(8, 11), res.v.get(9, 11), res.v.get(10, 11) },
    );
    var t_norm = Vec3.init(res.v.get(3, 11), res.v.get(7, 11), res.v.get(11, 11));

    // Check sign so majority of points are in front of the camera
    var pos_z: usize = 0;
    for (pts3d) |x_w| {
        const p = x_w.sub(mean3).scale(s3);
        const z = m_mat.m[2][0] * p.x + m_mat.m[2][1] * p.y + m_mat.m[2][2] * p.z + t_norm.z;
        if (z > 0) pos_z += 1;
    }
    if (pos_z * 2 < pts3d.len) {
        m_mat = m_mat.scale(-1.0);
        t_norm = t_norm.neg();
    }

    const s_m = svd3x3(m_mat);
    var r = s_m.u.mul(s_m.vt);
    if (r.det() < 0) {
        // Flip sign if determinant is negative
        var u_fix = s_m.u;
        u_fix.m[0][2] = -u_fix.m[0][2];
        u_fix.m[1][2] = -u_fix.m[1][2];
        u_fix.m[2][2] = -u_fix.m[2][2];
        r = u_fix.mul(s_m.vt);
    }

    const scale_avg = (s_m.s[0] + s_m.s[1] + s_m.s[2]) / 3.0;
    if (scale_avg < 1e-12) return null;

    // Undo 3D normalization: P_world = [s3 * R | t_norm / scale_avg - s3 * R * mean3]
    // Dividing by s3 gives [R | t_norm / (scale_avg * s3) - R * mean3]
    const t = t_norm.scale(1.0 / (scale_avg * s3)).sub(r.mulVec(mean3));
    return .{ .r = r, .t = t };
}

/// Non-linear Levenberg-Marquardt refinement of a single CameraPose on 2D-3D correspondences.
pub fn refinePoseLM(
    k: Mat3,
    initial_pose: CameraPose,
    pts3d: []const Vec3,
    pts2d: []const Vec2,
    max_iters: usize,
) CameraPose {
    var pose = initial_pose;
    var lambda: f64 = 1e-3;

    var prev_cost = computeCost(k, pose, pts3d, pts2d);

    for (0..max_iters) |_| {
        var jtj = std.mem.zeroes([36]f64);
        var jtr = std.mem.zeroes([6]f64);

        for (pts3d, pts2d) |x_w, uv| {
            const rx = pose.r.mulVec(x_w);
            const xc = rx.add(pose.t);
            if (xc.z <= 1e-6) continue;

            const inv_z = 1.0 / xc.z;
            const inv_z2 = inv_z * inv_z;
            const u_proj = k.m[0][0] * xc.x * inv_z + k.m[0][2];
            const v_proj = k.m[1][1] * xc.y * inv_z + k.m[1][2];

            const ex = u_proj - uv.x;
            const ey = v_proj - uv.y;

            // d(Xc)/d(w, t) = [-[R*X]_x | I_3]
            const fx = k.m[0][0];
            const fy = k.m[1][1];
            const j_u = [6]f64{
                fx * (-rx.y * xc.x * inv_z2),
                fx * (rx.z * inv_z + rx.x * xc.x * inv_z2),
                fx * (-rx.y * inv_z),
                fx * inv_z,
                0,
                -fx * xc.x * inv_z2,
            };
            const j_v = [6]f64{
                fy * (-rx.z * inv_z - rx.y * xc.y * inv_z2),
                fy * (rx.x * xc.y * inv_z2),
                fy * (rx.x * inv_z),
                0,
                fy * inv_z,
                -fy * xc.y * inv_z2,
            };

            for (0..6) |r_i| {
                jtr[r_i] -= j_u[r_i] * ex + j_v[r_i] * ey;
                for (0..6) |c_i| {
                    jtj[r_i * 6 + c_i] += j_u[r_i] * j_u[c_i] + j_v[r_i] * j_v[c_i];
                }
            }
        }

        var damped = jtj;
        for (0..6) |i| {
            damped[i * 6 + i] += lambda * @max(1.0, jtj[i * 6 + i]);
        }
        var delta = jtr;
        if (!solveLinearSystemInPlace(6, &damped, &delta)) {
            lambda *= 10.0;
            continue;
        }

        const dw = Vec3.init(delta[0], delta[1], delta[2]);
        const dt = Vec3.init(delta[3], delta[4], delta[5]);
        if (dw.norm() < 1e-10 and dt.norm() < 1e-10) break;

        const cand_pose = CameraPose{
            .r = lie.so3Exp(dw).mul(pose.r),
            .t = pose.t.add(dt),
        };
        const cand_cost = computeCost(k, cand_pose, pts3d, pts2d);
        if (cand_cost < prev_cost) {
            pose = cand_pose;
            const rel = (prev_cost - cand_cost) / @max(prev_cost, 1e-12);
            prev_cost = cand_cost;
            lambda = @max(lambda * 0.3, 1e-9);
            if (rel < 1e-9) break;
        } else {
            lambda *= 5.0;
        }
    }

    return pose;
}

fn computeCost(k: Mat3, pose: CameraPose, pts3d: []const Vec3, pts2d: []const Vec2) f64 {
    var cost: f64 = 0;
    for (pts3d, pts2d) |x_w, uv| {
        if (lie.projectPoint(k, pose, x_w)) |proj| {
            const d = proj.sub(uv);
            cost += d.normSq();
        } else {
            cost += 1e6;
        }
    }
    return cost;
}

pub const PnpRansacResult = struct {
    pose: CameraPose,
    inliers: []usize,

    pub fn deinit(self: *PnpRansacResult, allocator: std.mem.Allocator) void {
        allocator.free(self.inliers);
        self.* = undefined;
    }
};

pub fn solvePnpRansac(
    allocator: std.mem.Allocator,
    k: Mat3,
    pts3d: []const Vec3,
    pts2d: []const Vec2,
    iterations: usize,
    pixel_thresh: f64,
    seed: u64,
) !?PnpRansacResult {
    std.debug.assert(pts3d.len == pts2d.len);
    if (pts3d.len < 6) return null;

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const thresh_sq = pixel_thresh * pixel_thresh;

    var best_pose: ?CameraPose = null;
    var best_inlier_count: usize = 0;

    var sample_idx: [6]usize = undefined;
    var s3d: [6]Vec3 = undefined;
    var s2d: [6]Vec2 = undefined;

    for (0..iterations) |_| {
        var count: usize = 0;
        while (count < 6) {
            const cand = rng.uintLessThan(usize, pts3d.len);
            var dup = false;
            for (0..count) |j| {
                if (sample_idx[j] == cand) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                sample_idx[count] = cand;
                s3d[count] = pts3d[cand];
                s2d[count] = pts2d[cand];
                count += 1;
            }
        }

        const cand_pose = (try solvePnpDlt(allocator, k, &s3d, &s2d)) orelse continue;
        var inlier_count: usize = 0;
        for (pts3d, pts2d) |x_w, uv| {
            if (lie.projectPoint(k, cand_pose, x_w)) |proj| {
                if (proj.sub(uv).normSq() <= thresh_sq) {
                    inlier_count += 1;
                }
            }
        }

        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_pose = cand_pose;
        }
    }

    const init_pose = best_pose orelse return null;
    if (best_inlier_count < 6) return null;

    var in_3d: std.ArrayList(Vec3) = .empty;
    defer in_3d.deinit(allocator);
    var in_2d: std.ArrayList(Vec2) = .empty;
    defer in_2d.deinit(allocator);
    var inliers: std.ArrayList(usize) = .empty;
    defer inliers.deinit(allocator);

    for (pts3d, pts2d, 0..) |x_w, uv, i| {
        if (lie.projectPoint(k, init_pose, x_w)) |proj| {
            if (proj.sub(uv).normSq() <= thresh_sq) {
                try in_3d.append(allocator, x_w);
                try in_2d.append(allocator, uv);
                try inliers.append(allocator, i);
            }
        }
    }

    const dlt_pose = (try solvePnpDlt(allocator, k, in_3d.items, in_2d.items)) orelse init_pose;
    const refined = refinePoseLM(k, dlt_pose, in_3d.items, in_2d.items, 20);

    inliers.clearRetainingCapacity();
    for (pts3d, pts2d, 0..) |x_w, uv, i| {
        if (lie.projectPoint(k, refined, x_w)) |proj| {
            if (proj.sub(uv).normSq() <= thresh_sq) {
                try inliers.append(allocator, i);
            }
        }
    }

    return .{
        .pose = refined,
        .inliers = try inliers.toOwnedSlice(allocator),
    };
}

test "PnP RANSAC + LM refinement recovers pose with outliers" {
    const gpa = std.testing.allocator;
    const k = Mat3.fromRows(
        .{ 800, 0, 320 },
        .{ 0, 800, 240 },
        .{ 0, 0, 1 },
    );
    const gt_pose = CameraPose{
        .r = lie.so3Exp(Vec3.init(0.1, -0.25, 0.08)),
        .t = Vec3.init(-0.8, 0.3, 1.2),
    };

    var prng = std.Random.DefaultPrng.init(77);
    const rng = prng.random();

    const total = 40;
    const n_inliers = 32;
    var pts3d: [total]Vec3 = undefined;
    var pts2d: [total]Vec2 = undefined;

    for (0..n_inliers) |i| {
        const x = Vec3.init(
            (rng.float(f64) - 0.5) * 4.0,
            (rng.float(f64) - 0.5) * 3.0,
            4.0 + rng.float(f64) * 5.0,
        );
        pts3d[i] = x;
        pts2d[i] = lie.projectPoint(k, gt_pose, x).?;
    }
    for (n_inliers..total) |i| {
        pts3d[i] = Vec3.init(rng.float(f64) * 3.0, rng.float(f64) * 3.0, 5.0);
        pts2d[i] = Vec2.init(rng.float(f64) * 640.0, rng.float(f64) * 480.0);
    }

    var res = (try solvePnpRansac(gpa, k, &pts3d, &pts2d, 200, 2.0, 99)).?;
    defer res.deinit(gpa);

    try std.testing.expect(res.inliers.len >= n_inliers);
    const angle_err = lie.so3Log(res.pose.r.mul(gt_pose.r.transpose())).norm();
    try std.testing.expect(angle_err < 1e-4);
    try std.testing.expect(res.pose.t.dist(gt_pose.t) < 1e-3);
}
