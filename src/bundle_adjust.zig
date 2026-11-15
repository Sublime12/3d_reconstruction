const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const solveLinearSystemInPlace = @import("math/mat.zig").solveLinearSystemInPlace;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;

pub const Observation = struct {
    cam_idx: usize,
    pt_idx: usize,
    uv: Vec2,
};

pub const BaConfig = struct {
    max_iters: usize = 25,
    huber_delta: f64 = 2.0,
    fix_first_camera: bool = true,
};

pub const BaStats = struct {
    initial_rmse: f64,
    final_rmse: f64,
    iters: usize,
};

/// Sparse Levenberg-Marquardt Bundle Adjustment using Schur complement
/// to marginalize 3D point blocks (3x3) and solve the reduced 6C x 6C camera system.
pub fn runBundleAdjustment(
    allocator: std.mem.Allocator,
    k: Mat3,
    poses: []CameraPose,
    points: []Vec3,
    observations: []const Observation,
    config: BaConfig,
) !BaStats {
    const n_cams = poses.len;
    const n_pts = points.len;
    const cam_dim = n_cams * 6;

    if (n_cams == 0 or n_pts == 0 or observations.len == 0) {
        return .{ .initial_rmse = 0, .final_rmse = 0, .iters = 0 };
    }

    // Group observation indices by point so Schur complement V^-1 * W^T is O(nnz)
    const pt_counts = try allocator.alloc(usize, n_pts);
    defer allocator.free(pt_counts);
    @memset(pt_counts, 0);
    for (observations) |obs| pt_counts[obs.pt_idx] += 1;

    const pt_offsets = try allocator.alloc(usize, n_pts + 1);
    defer allocator.free(pt_offsets);
    pt_offsets[0] = 0;
    for (0..n_pts) |j| pt_offsets[j + 1] = pt_offsets[j] + pt_counts[j];

    const pt_obs_idx = try allocator.alloc(usize, observations.len);
    defer allocator.free(pt_obs_idx);
    const pt_cursor = try allocator.alloc(usize, n_pts);
    defer allocator.free(pt_cursor);
    @memcpy(pt_cursor, pt_offsets[0..n_pts]);
    for (observations, 0..) |obs, idx| {
        const j = obs.pt_idx;
        pt_obs_idx[pt_cursor[j]] = idx;
        pt_cursor[j] += 1;
    }

    // Per-camera 6x6 U blocks and 6-vec g_c
    const u_blocks = try allocator.alloc([36]f64, n_cams);
    defer allocator.free(u_blocks);
    const g_c = try allocator.alloc(f64, cam_dim);
    defer allocator.free(g_c);

    // Per-point 3x3 V blocks and 3-vec g_p
    const v_blocks = try allocator.alloc(Mat3, n_pts);
    defer allocator.free(v_blocks);
    const v_inv_blocks = try allocator.alloc(Mat3, n_pts);
    defer allocator.free(v_inv_blocks);
    const g_p = try allocator.alloc(Vec3, n_pts);
    defer allocator.free(g_p);

    // Per-observation 6x3 W_ij = J_c^T * J_p
    const w_blocks = try allocator.alloc([6][3]f64, observations.len);
    defer allocator.free(w_blocks);

    // Reduced Schur camera matrix S (cam_dim x cam_dim) and rhs b_c (cam_dim)
    const s_mat = try allocator.alloc(f64, cam_dim * cam_dim);
    defer allocator.free(s_mat);
    const b_c = try allocator.alloc(f64, cam_dim);
    defer allocator.free(b_c);

    const cand_poses = try allocator.alloc(CameraPose, n_cams);
    defer allocator.free(cand_poses);
    const cand_pts = try allocator.alloc(Vec3, n_pts);
    defer allocator.free(cand_pts);

    var prev_cost = computeHuberCost(k, poses, points, observations, config.huber_delta);
    const initial_rmse = computeRmse(k, poses, points, observations);
    var lambda: f64 = 1e-3;
    var iters_done: usize = 0;

    const fx = k.m[0][0];
    const fy = k.m[1][1];

    for (0..config.max_iters) |iter| {
        iters_done = iter + 1;

        for (u_blocks) |*u| @memset(u, 0);
        @memset(g_c, 0);
        for (v_blocks) |*v| v.* = Mat3.zero;
        for (g_p) |*gp| gp.* = Vec3.zero;

        // 1. Accumulate U, V, W, g_c, g_p
        for (observations, 0..) |obs, obs_idx| {
            const pose = poses[obs.cam_idx];
            const pt = points[obs.pt_idx];
            const rx = pose.r.mulVec(pt);
            const xc = rx.add(pose.t);
            if (xc.z <= 1e-6) {
                w_blocks[obs_idx] = std.mem.zeroes([6][3]f64);
                continue;
            }

            const inv_z = 1.0 / xc.z;
            const inv_z2 = inv_z * inv_z;
            const u_proj = fx * xc.x * inv_z + k.m[0][2];
            const v_proj = fy * xc.y * inv_z + k.m[1][2];

            const ex = u_proj - obs.uv.x;
            const ey = v_proj - obs.uv.y;
            const err_norm = @sqrt(ex * ex + ey * ey);
            const weight = if (err_norm > config.huber_delta) config.huber_delta / err_norm else 1.0;

            const jc_u = [6]f64{
                fx * (-rx.y * xc.x * inv_z2),
                fx * (rx.z * inv_z + rx.x * xc.x * inv_z2),
                fx * (-rx.y * inv_z),
                fx * inv_z,
                0,
                -fx * xc.x * inv_z2,
            };
            const jc_v = [6]f64{
                fy * (-rx.z * inv_z - rx.y * xc.y * inv_z2),
                fy * (rx.x * xc.y * inv_z2),
                fy * (rx.x * inv_z),
                0,
                fy * inv_z,
                -fy * xc.y * inv_z2,
            };

            // d(pi)/d(Xc) * R
            const dpx = [3]f64{ fx * inv_z, 0, -fx * xc.x * inv_z2 };
            const dpy = [3]f64{ 0, fy * inv_z, -fy * xc.y * inv_z2 };
            var jp_u: [3]f64 = undefined;
            var jp_v: [3]f64 = undefined;
            for (0..3) |col| {
                jp_u[col] = dpx[0] * pose.r.m[0][col] + dpx[1] * pose.r.m[1][col] + dpx[2] * pose.r.m[2][col];
                jp_v[col] = dpy[0] * pose.r.m[0][col] + dpy[1] * pose.r.m[1][col] + dpy[2] * pose.r.m[2][col];
            }

            // Camera block U_i and g_c_i
            const ci = obs.cam_idx;
            for (0..6) |r_i| {
                g_c[ci * 6 + r_i] -= weight * (jc_u[r_i] * ex + jc_v[r_i] * ey);
                for (0..6) |c_i| {
                    u_blocks[ci][r_i * 6 + c_i] += weight * (jc_u[r_i] * jc_u[c_i] + jc_v[r_i] * jc_v[c_i]);
                }
            }

            // Point block V_j and g_p_j
            const pj = obs.pt_idx;
            g_p[pj].x -= weight * (jp_u[0] * ex + jp_v[0] * ey);
            g_p[pj].y -= weight * (jp_u[1] * ex + jp_v[1] * ey);
            g_p[pj].z -= weight * (jp_u[2] * ex + jp_v[2] * ey);
            for (0..3) |r_i| {
                for (0..3) |c_i| {
                    v_blocks[pj].m[r_i][c_i] += weight * (jp_u[r_i] * jp_u[c_i] + jp_v[r_i] * jp_v[c_i]);
                }
            }

            // Cross block W_ij
            for (0..6) |r_i| {
                for (0..3) |c_i| {
                    w_blocks[obs_idx][r_i][c_i] = weight * (jc_u[r_i] * jp_u[c_i] + jc_v[r_i] * jp_v[c_i]);
                }
            }
        }

        // 2. Build Schur complement S = U - W V^-1 W^T and b_c = g_c - W V^-1 g_p
        @memset(s_mat, 0);
        @memcpy(b_c, g_c);

        for (0..n_cams) |ci| {
            for (0..6) |r_i| {
                for (0..6) |c_i| {
                    const row = ci * 6 + r_i;
                    const col = ci * 6 + c_i;
                    var val = u_blocks[ci][r_i * 6 + c_i];
                    if (r_i == c_i) {
                        val += lambda * @max(1.0, val);
                        if (config.fix_first_camera and ci == 0) {
                            val += 1e12;
                        }
                    }
                    s_mat[row * cam_dim + col] = val;
                }
            }
        }

        for (0..n_pts) |pj| {
            var v_damped = v_blocks[pj];
            for (0..3) |d| {
                v_damped.m[d][d] += lambda * @max(1.0, v_damped.m[d][d]);
            }
            const v_inv = v_damped.inv() orelse Mat3.zero;
            v_inv_blocks[pj] = v_inv;

            const start = pt_offsets[pj];
            const end = pt_offsets[pj + 1];
            const v_inv_gp = v_inv.mulVec(g_p[pj]);

            for (start..end) |k1| {
                const idx1 = pt_obs_idx[k1];
                const c1 = observations[idx1].cam_idx;
                const w1 = w_blocks[idx1];

                // b_c[c1] -= W1 * V^-1 * g_p
                for (0..6) |r_i| {
                    b_c[c1 * 6 + r_i] -= w1[r_i][0] * v_inv_gp.x + w1[r_i][1] * v_inv_gp.y + w1[r_i][2] * v_inv_gp.z;
                }

                // Compute W1 * V^-1 (6x3)
                var w1_vinv: [6][3]f64 = undefined;
                for (0..6) |r_i| {
                    for (0..3) |c_i| {
                        w1_vinv[r_i][c_i] =
                            w1[r_i][0] * v_inv.m[0][c_i] +
                            w1[r_i][1] * v_inv.m[1][c_i] +
                            w1[r_i][2] * v_inv.m[2][c_i];
                    }
                }

                for (start..end) |k2| {
                    const idx2 = pt_obs_idx[k2];
                    const c2 = observations[idx2].cam_idx;
                    const w2 = w_blocks[idx2];

                    for (0..6) |r_i| {
                        const row = c1 * 6 + r_i;
                        for (0..6) |c_i| {
                            const col = c2 * 6 + c_i;
                            s_mat[row * cam_dim + col] -=
                                w1_vinv[r_i][0] * w2[c_i][0] +
                                w1_vinv[r_i][1] * w2[c_i][1] +
                                w1_vinv[r_i][2] * w2[c_i][2];
                        }
                    }
                }
            }
        }

        // 3. Solve reduced camera system S * dc = b_c (result placed in b_c)
        if (!solveLinearSystemInPlace(cam_dim, s_mat, b_c)) {
            lambda *= 10.0;
            continue;
        }

        // 4. Back-substitute 3D points dp_j = V_j^-1 * (g_p_j - sum W_ij^T * dc_i)
        var max_step_sq: f64 = 0;
        for (0..n_cams) |ci| {
            const base = ci * 6;
            const dw = if (config.fix_first_camera and ci == 0)
                Vec3.zero
            else
                Vec3.init(b_c[base + 0], b_c[base + 1], b_c[base + 2]);
            const dt = if (config.fix_first_camera and ci == 0)
                Vec3.zero
            else
                Vec3.init(b_c[base + 3], b_c[base + 4], b_c[base + 5]);

            max_step_sq = @max(max_step_sq, dw.normSq() + dt.normSq());
            cand_poses[ci] = .{
                .r = lie.so3Exp(dw).mul(poses[ci].r),
                .t = poses[ci].t.add(dt),
            };
        }

        for (0..n_pts) |pj| {
            var rhs = g_p[pj];
            const start = pt_offsets[pj];
            const end = pt_offsets[pj + 1];
            for (start..end) |k1| {
                const idx1 = pt_obs_idx[k1];
                const ci = observations[idx1].cam_idx;
                const w1 = w_blocks[idx1];
                const base = ci * 6;
                for (0..6) |r_i| {
                    const dc = b_c[base + r_i];
                    rhs.x -= w1[r_i][0] * dc;
                    rhs.y -= w1[r_i][1] * dc;
                    rhs.z -= w1[r_i][2] * dc;
                }
            }
            const dp = v_inv_blocks[pj].mulVec(rhs);
            max_step_sq = @max(max_step_sq, dp.normSq());
            cand_pts[pj] = points[pj].add(dp);
        }

        if (max_step_sq < 1e-18) break;

        const cand_cost = computeHuberCost(k, cand_poses, cand_pts, observations, config.huber_delta);
        if (cand_cost < prev_cost) {
            @memcpy(poses, cand_poses);
            @memcpy(points, cand_pts);
            const rel = (prev_cost - cand_cost) / @max(prev_cost, 1e-12);
            prev_cost = cand_cost;
            lambda = @max(lambda * 0.3, 1e-9);
            if (rel < 1e-7) break;
        } else {
            lambda *= 5.0;
        }
    }

    return .{
        .initial_rmse = initial_rmse,
        .final_rmse = computeRmse(k, poses, points, observations),
        .iters = iters_done,
    };
}

fn computeHuberCost(
    k: Mat3,
    poses: []const CameraPose,
    points: []const Vec3,
    observations: []const Observation,
    delta: f64,
) f64 {
    var cost: f64 = 0;
    for (observations) |obs| {
        if (lie.projectPoint(k, poses[obs.cam_idx], points[obs.pt_idx])) |proj| {
            const e = proj.dist(obs.uv);
            if (e <= delta) {
                cost += 0.5 * e * e;
            } else {
                cost += delta * (e - 0.5 * delta);
            }
        } else {
            cost += 1e6;
        }
    }
    return cost;
}

pub fn computeRmse(
    k: Mat3,
    poses: []const CameraPose,
    points: []const Vec3,
    observations: []const Observation,
) f64 {
    if (observations.len == 0) return 0;
    var sum_sq: f64 = 0;
    for (observations) |obs| {
        if (lie.projectPoint(k, poses[obs.cam_idx], points[obs.pt_idx])) |proj| {
            sum_sq += proj.sub(obs.uv).normSq();
        } else {
            sum_sq += 1e6;
        }
    }
    return @sqrt(sum_sq / @as(f64, @floatFromInt(observations.len)));
}

test "Bundle Adjustment reduces reprojection error on noisy synthetic poses and points" {
    const gpa = std.testing.allocator;
    const k = Mat3.fromRows(
        .{ 800, 0, 320 },
        .{ 0, 800, 240 },
        .{ 0, 0, 1 },
    );

    const gt_poses = [_]CameraPose{
        CameraPose.identity,
        .{ .r = lie.so3Exp(Vec3.init(0.02, -0.15, 0.01)), .t = Vec3.init(-1.0, 0.05, 0.1) },
        .{ .r = lie.so3Exp(Vec3.init(-0.03, -0.30, 0.02)), .t = Vec3.init(-2.0, -0.05, 0.25) },
        .{ .r = lie.so3Exp(Vec3.init(0.01, -0.45, -0.02)), .t = Vec3.init(-3.0, 0.1, 0.45) },
    };

    var prng = std.Random.DefaultPrng.init(101);
    const rng = prng.random();

    const n_pts = 30;
    var gt_pts: [n_pts]Vec3 = undefined;
    for (&gt_pts) |*p| {
        p.* = Vec3.init(
            (rng.float(f64) - 0.5) * 4.0,
            (rng.float(f64) - 0.5) * 3.0,
            5.0 + rng.float(f64) * 5.0,
        );
    }

    var obs_list: std.ArrayList(Observation) = .empty;
    defer obs_list.deinit(gpa);
    for (gt_poses, 0..) |pose, ci| {
        for (gt_pts, 0..) |pt, pj| {
            if (lie.projectPoint(k, pose, pt)) |uv| {
                try obs_list.append(gpa, .{ .cam_idx = ci, .pt_idx = pj, .uv = uv });
            }
        }
    }

    // Perturb poses (except camera 0) and 3D points
    var noisy_poses = gt_poses;
    for (1..noisy_poses.len) |ci| {
        noisy_poses[ci].r = lie.so3Exp(Vec3.init(
            (rng.float(f64) - 0.5) * 0.04,
            (rng.float(f64) - 0.5) * 0.04,
            (rng.float(f64) - 0.5) * 0.04,
        )).mul(noisy_poses[ci].r);
        noisy_poses[ci].t = noisy_poses[ci].t.add(Vec3.init(
            (rng.float(f64) - 0.5) * 0.1,
            (rng.float(f64) - 0.5) * 0.1,
            (rng.float(f64) - 0.5) * 0.1,
        ));
    }

    var noisy_pts = gt_pts;
    for (&noisy_pts) |*pt| {
        pt.* = pt.add(Vec3.init(
            (rng.float(f64) - 0.5) * 0.25,
            (rng.float(f64) - 0.5) * 0.25,
            (rng.float(f64) - 0.5) * 0.25,
        ));
    }

    const stats = try runBundleAdjustment(gpa, k, &noisy_poses, &noisy_pts, obs_list.items, .{});
    try std.testing.expect(stats.initial_rmse > 5.0);
    try std.testing.expect(stats.final_rmse < 0.05);
}
