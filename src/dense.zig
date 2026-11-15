const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const dataset = @import("dataset.zig");
const Color = @import("io_ply.zig").Color;

pub const DenseConfig = struct {
    stride: u32 = 3,
    depth_steps: usize = 80,
    patch_radius: isize = 2,
    min_zncc: f64 = 0.68,
    min_grad: f64 = 3.0,
    max_points: usize = 35000,
    num_threads: usize = 0,
};

pub const DenseCloud = struct {
    points: []Vec3,
    colors: []Color,

    pub fn deinit(self: *DenseCloud, allocator: std.mem.Allocator) void {
        allocator.free(self.points);
        allocator.free(self.colors);
        self.* = undefined;
    }
};

/// Multi-view plane-sweep ZNCC dense reconstruction using SfM camera poses and sparse depth bounds.
pub fn runDenseMvs(
    allocator: std.mem.Allocator,
    k: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    sparse_points: []const Vec3,
    config: DenseConfig,
) !DenseCloud {
    const k_inv = k.inv() orelse return error.InvalidIntrinsics;

    // Collect registered camera indices
    var reg_cams: std.ArrayList(usize) = .empty;
    defer reg_cams.deinit(allocator);
    for (registered, 0..) |is_reg, ci| {
        if (is_reg) try reg_cams.append(allocator, ci);
    }
    if (reg_cams.items.len < 2 or sparse_points.len == 0) {
        return .{
            .points = try allocator.alloc(Vec3, 0),
            .colors = try allocator.alloc(Color, 0),
        };
    }

    // Use all registered cameras as reference views for maximum surface coverage
    var ref_cams: std.ArrayList(usize) = .empty;
    defer ref_cams.deinit(allocator);
    for (reg_cams.items) |ci| {
        try ref_cams.append(allocator, ci);
    }

    const per_cam_pts = try allocator.alloc([]Vec3, ref_cams.items.len);
    for (per_cam_pts) |*slot| slot.* = &.{};
    defer {
        for (per_cam_pts) |s| allocator.free(s);
        allocator.free(per_cam_pts);
    }

    const per_cam_cols = try allocator.alloc([]Color, ref_cams.items.len);
    for (per_cam_cols) |*slot| slot.* = &.{};
    defer {
        for (per_cam_cols) |s| allocator.free(s);
        allocator.free(per_cam_cols);
    }

    const n_threads = if (config.num_threads == 0) (std.Thread.getCpuCount() catch 4) else config.num_threads;
    if (n_threads <= 1 or ref_cams.items.len <= 1) {
        for (ref_cams.items, 0..) |cam_idx, slot_idx| {
            try reconstructRefView(
                allocator,
                k,
                k_inv,
                images,
                poses,
                reg_cams.items,
                sparse_points,
                cam_idx,
                config,
                &per_cam_pts[slot_idx],
                &per_cam_cols[slot_idx],
            );
        }
    } else {
        var next_slot = std.atomic.Value(usize).init(0);
        var failed = std.atomic.Value(bool).init(false);
        const worker_count = @min(n_threads, ref_cams.items.len);
        const threads = try allocator.alloc(std.Thread, worker_count);
        defer allocator.free(threads);

        for (threads) |*th| {
            th.* = try std.Thread.spawn(.{}, denseWorker, .{
                allocator,
                k,
                k_inv,
                images,
                poses,
                reg_cams.items,
                sparse_points,
                ref_cams.items,
                config,
                per_cam_pts,
                per_cam_cols,
                &next_slot,
                &failed,
            });
        }
        for (threads) |th| th.join();
        if (failed.load(.acquire)) return error.DenseMvsFailed;
    }

    var total: usize = 0;
    for (per_cam_pts) |s| total += s.len;

    const all_pts = try allocator.alloc(Vec3, total);
    errdefer allocator.free(all_pts);
    const all_cols = try allocator.alloc(Color, total);

    var cursor: usize = 0;
    for (per_cam_pts, per_cam_cols) |pts, cols| {
        @memcpy(all_pts[cursor..][0..pts.len], pts);
        @memcpy(all_cols[cursor..][0..cols.len], cols);
        cursor += pts.len;
    }

    if (config.max_points > 0 and all_pts.len > config.max_points) {
        const fused = try voxelDownsampleToTarget(allocator, all_pts, all_cols, config.max_points);
        allocator.free(all_pts);
        allocator.free(all_cols);
        return fused;
    }

    return .{ .points = all_pts, .colors = all_cols };
}

const VoxelKey = struct {
    x: i32,
    y: i32,
    z: i32,
};

const VoxelAcc = struct {
    sx: f64,
    sy: f64,
    sz: f64,
    sr: u64,
    sg: u64,
    sb: u64,
    count: u64,
};

fn voxelDownsampleToTarget(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    target_max: usize,
) !DenseCloud {
    var min_p = points[0];
    var max_p = points[0];
    for (points) |p| {
        min_p.x = @min(min_p.x, p.x);
        min_p.y = @min(min_p.y, p.y);
        min_p.z = @min(min_p.z, p.z);
        max_p.x = @max(max_p.x, p.x);
        max_p.y = @max(max_p.y, p.y);
        max_p.z = @max(max_p.z, p.z);
    }
    const diag = max_p.sub(min_p).norm();
    // Estimate initial voxel size assuming surface area ~ diag^2
    var voxel_size = @max(diag / @sqrt(@as(f64, @floatFromInt(target_max)) * 1.5), 1e-4);

    var best_pts: []Vec3 = &.{};
    var best_cols: []Color = &.{};

    for (0..4) |_| {
        var map = std.AutoHashMap(VoxelKey, VoxelAcc).init(allocator);
        defer map.deinit();
        const inv_v = 1.0 / voxel_size;

        for (points, colors) |pt, col| {
            const key = VoxelKey{
                .x = @intFromFloat(@floor(pt.x * inv_v)),
                .y = @intFromFloat(@floor(pt.y * inv_v)),
                .z = @intFromFloat(@floor(pt.z * inv_v)),
            };
            const gop = try map.getOrPut(key);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{
                    .sx = pt.x,
                    .sy = pt.y,
                    .sz = pt.z,
                    .sr = col.r,
                    .sg = col.g,
                    .sb = col.b,
                    .count = 1,
                };
            } else {
                gop.value_ptr.sx += pt.x;
                gop.value_ptr.sy += pt.y;
                gop.value_ptr.sz += pt.z;
                gop.value_ptr.sr += col.r;
                gop.value_ptr.sg += col.g;
                gop.value_ptr.sb += col.b;
                gop.value_ptr.count += 1;
            }
        }

        const n_out = map.count();
        if (n_out <= target_max or voxel_size > diag) {
            best_pts = try allocator.alloc(Vec3, n_out);
            errdefer allocator.free(best_pts);
            best_cols = try allocator.alloc(Color, n_out);

            var it = map.valueIterator();
            var idx: usize = 0;
            while (it.next()) |acc| : (idx += 1) {
                const c_f: f64 = @floatFromInt(acc.count);
                best_pts[idx] = Vec3.init(acc.sx / c_f, acc.sy / c_f, acc.sz / c_f);
                best_cols[idx] = .{
                    .r = @intCast(acc.sr / acc.count),
                    .g = @intCast(acc.sg / acc.count),
                    .b = @intCast(acc.sb / acc.count),
                };
            }
            return .{ .points = best_pts, .colors = best_cols };
        }
        voxel_size *= @sqrt(@as(f64, @floatFromInt(n_out)) / @as(f64, @floatFromInt(target_max))) * 1.1;
    }

    return .{
        .points = try allocator.dupe(Vec3, points[0..target_max]),
        .colors = try allocator.dupe(Color, colors[0..target_max]),
    };
}

fn denseWorker(
    allocator: std.mem.Allocator,
    k: Mat3,
    k_inv: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    reg_cams: []const usize,
    sparse_points: []const Vec3,
    ref_cams: []const usize,
    config: DenseConfig,
    per_cam_pts: [][]Vec3,
    per_cam_cols: [][]Color,
    next_slot: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const slot_idx = next_slot.fetchAdd(1, .monotonic);
        if (slot_idx >= ref_cams.len) return;
        reconstructRefView(
            allocator,
            k,
            k_inv,
            images,
            poses,
            reg_cams,
            sparse_points,
            ref_cams[slot_idx],
            config,
            &per_cam_pts[slot_idx],
            &per_cam_cols[slot_idx],
        ) catch {
            failed.store(true, .release);
            return;
        };
    }
}

fn reconstructRefView(
    allocator: std.mem.Allocator,
    k: Mat3,
    k_inv: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    reg_cams: []const usize,
    sparse_points: []const Vec3,
    ref_idx: usize,
    config: DenseConfig,
    out_pts: *[]Vec3,
    out_cols: *[]Color,
) !void {
    const ref_img = images[ref_idx];
    const ref_pose = poses[ref_idx];
    const ref_rt = ref_pose.r.transpose();
    const ref_c = ref_pose.cameraCenter();

    // Estimate [z_min, z_max] in ref camera from sparse points
    var depths: std.ArrayList(f64) = .empty;
    defer depths.deinit(allocator);
    for (sparse_points) |pt| {
        const z = ref_pose.transformPoint(pt).z;
        if (z > 0.1) try depths.append(allocator, z);
    }
    if (depths.items.len < 10) {
        out_pts.* = try allocator.alloc(Vec3, 0);
        out_cols.* = try allocator.alloc(Color, 0);
        return;
    }
    std.mem.sort(f64, depths.items, {}, std.sort.asc(f64));
    const z_min = @max(0.2, depths.items[depths.items.len / 20] * 0.75);
    const z_max = depths.items[(depths.items.len * 19) / 20] * 1.25;
    if (z_max <= z_min) {
        out_pts.* = try allocator.alloc(Vec3, 0);
        out_cols.* = try allocator.alloc(Color, 0);
        return;
    }

    // Pick up to 2 nearest neighboring cameras by camera center distance
    var neighbor_buf: [2]usize = undefined;
    var neighbor_cnt: usize = 0;
    var best_d: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
    for (reg_cams) |ci| {
        if (ci == ref_idx) continue;
        const d = poses[ci].cameraCenter().dist(ref_c);
        if (d < 1e-6) continue;
        if (d < best_d[0]) {
            best_d[1] = best_d[0];
            neighbor_buf[1] = neighbor_buf[0];
            best_d[0] = d;
            neighbor_buf[0] = ci;
            neighbor_cnt = @min(2, neighbor_cnt + 1);
        } else if (d < best_d[1]) {
            best_d[1] = d;
            neighbor_buf[1] = ci;
            neighbor_cnt = @min(2, neighbor_cnt + 1);
        }
    }
    if (neighbor_cnt == 0) {
        out_pts.* = try allocator.alloc(Vec3, 0);
        out_cols.* = try allocator.alloc(Color, 0);
        return;
    }
    const neighbors = neighbor_buf[0..neighbor_cnt];

    var pts: std.ArrayList(Vec3) = .empty;
    errdefer pts.deinit(allocator);
    var cols: std.ArrayList(Color) = .empty;
    errdefer cols.deinit(allocator);

    const w: isize = @intCast(ref_img.width);
    const h: isize = @intCast(ref_img.height);
    const pr = config.patch_radius;
    const margin = pr + 2;
    const inv_z_min = 1.0 / z_max;
    const inv_z_max = 1.0 / z_min;

    var y: isize = margin;
    while (y < h - margin) : (y += @intCast(config.stride)) {
        var x: isize = margin;
        while (x < w - margin) : (x += @intCast(config.stride)) {
            const uf: f64 = @floatFromInt(x);
            const vf: f64 = @floatFromInt(y);

            // Extract reference patch & check texture variance
            var ref_patch: [25]f64 = undefined;
            var ref_mean: f64 = 0;
            var p_len: usize = 0;
            var dy: isize = -pr;
            while (dy <= pr) : (dy += 1) {
                var dx: isize = -pr;
                while (dx <= pr) : (dx += 1) {
                    const idx = @as(usize, @intCast(y + dy)) * ref_img.width + @as(usize, @intCast(x + dx));
                    const val: f64 = @floatFromInt(ref_img.gray[idx]);
                    ref_patch[p_len] = val;
                    ref_mean += val;
                    p_len += 1;
                }
            }
            ref_mean /= @as(f64, @floatFromInt(p_len));

            var ref_var: f64 = 0;
            for (0..p_len) |i| {
                ref_patch[i] -= ref_mean;
                ref_var += ref_patch[i] * ref_patch[i];
            }
            const ref_std = @sqrt(ref_var / @as(f64, @floatFromInt(p_len)));
            if (ref_std < config.min_grad) continue;

            // Ray in world space: X(z) = C_ref + z * R_ref^T * (K^-1 * [u, v, 1]^T)
            const ray_cam = k_inv.mulVec(Vec3.init(uf, vf, 1.0));
            const ray_world = ref_rt.mulVec(ray_cam);

            var best_score: f64 = -1.0;
            var best_ds: usize = 0;
            var scores_buf: [256]f64 = undefined;
            const n_steps = @min(config.depth_steps, scores_buf.len);

            for (0..n_steps) |ds| {
                const alpha = @as(f64, @floatFromInt(ds)) / @as(f64, @floatFromInt(n_steps - 1));
                const inv_z = inv_z_min + alpha * (inv_z_max - inv_z_min);
                const z = 1.0 / inv_z;
                const x_w = ref_c.add(ray_world.scale(z));

                var score_sum: f64 = 0;
                var valid_n: usize = 0;
                for (neighbors) |ni| {
                    const uv_n = lie.projectPoint(k, poses[ni], x_w) orelse continue;
                    const n_img = images[ni];
                    const nw_f: f64 = @floatFromInt(n_img.width);
                    const nh_f: f64 = @floatFromInt(n_img.height);
                    const pr_f: f64 = @floatFromInt(pr);
                    if (uv_n.x < pr_f + 1.0 or uv_n.x >= nw_f - pr_f - 1.0 or
                        uv_n.y < pr_f + 1.0 or uv_n.y >= nh_f - pr_f - 1.0)
                    {
                        continue;
                    }

                    const zncc = computePatchZncc(
                        n_img.gray,
                        n_img.width,
                        uv_n.x,
                        uv_n.y,
                        pr,
                        ref_patch[0..p_len],
                        ref_var,
                    ) orelse continue;
                    score_sum += zncc;
                    valid_n += 1;
                }
                const avg_score = if (valid_n > 0) score_sum / @as(f64, @floatFromInt(valid_n)) else -1.0;
                scores_buf[ds] = avg_score;
                if (avg_score > best_score) {
                    best_score = avg_score;
                    best_ds = ds;
                }
            }

            if (best_score >= config.min_zncc) {
                // Parabolic sub-step depth interpolation to eliminate staircase quantization
                var sub_ds: f64 = @floatFromInt(best_ds);
                if (best_ds > 0 and best_ds + 1 < n_steps) {
                    const s_prev = scores_buf[best_ds - 1];
                    const s_cur = scores_buf[best_ds];
                    const s_next = scores_buf[best_ds + 1];
                    const denom = s_prev - 2.0 * s_cur + s_next;
                    if (s_prev > 0 and s_next > 0 and denom < -1e-6) {
                        const offset = std.math.clamp(0.5 * (s_prev - s_next) / denom, -0.5, 0.5);
                        sub_ds += offset;
                    }
                }
                const alpha = sub_ds / @as(f64, @floatFromInt(n_steps - 1));
                const best_z = 1.0 / (inv_z_min + alpha * (inv_z_max - inv_z_min));
                const x_w = ref_c.add(ray_world.scale(best_z));
                const rgb_idx = (@as(usize, @intCast(y)) * ref_img.width + @as(usize, @intCast(x))) * 3;
                try pts.append(allocator, x_w);
                try cols.append(allocator, .{
                    .r = ref_img.rgb[rgb_idx + 0],
                    .g = ref_img.rgb[rgb_idx + 1],
                    .b = ref_img.rgb[rgb_idx + 2],
                });
            }
        }
    }

    out_pts.* = try pts.toOwnedSlice(allocator);
    out_cols.* = try cols.toOwnedSlice(allocator);
}

fn computePatchZncc(
    gray: []const u8,
    width: u32,
    cx: f64,
    cy: f64,
    pr: isize,
    ref_zero_mean: []const f64,
    ref_var: f64,
) ?f64 {
    var patch: [25]f64 = undefined;
    var mean: f64 = 0;
    var idx: usize = 0;

    var dy: isize = -pr;
    while (dy <= pr) : (dy += 1) {
        const sy = cy + @as(f64, @floatFromInt(dy));
        const y0: usize = @intFromFloat(@floor(sy));
        const wy = sy - @as(f64, @floatFromInt(y0));
        var dx: isize = -pr;
        while (dx <= pr) : (dx += 1) {
            const sx = cx + @as(f64, @floatFromInt(dx));
            const x0: usize = @intFromFloat(@floor(sx));
            const wx = sx - @as(f64, @floatFromInt(x0));

            const p00: f64 = @floatFromInt(gray[y0 * width + x0]);
            const p10: f64 = @floatFromInt(gray[y0 * width + (x0 + 1)]);
            const p01: f64 = @floatFromInt(gray[(y0 + 1) * width + x0]);
            const p11: f64 = @floatFromInt(gray[(y0 + 1) * width + (x0 + 1)]);
            const val = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10) + wy * ((1.0 - wx) * p01 + wx * p11);
            patch[idx] = val;
            mean += val;
            idx += 1;
        }
    }
    mean /= @as(f64, @floatFromInt(idx));

    var var2: f64 = 0;
    var dot: f64 = 0;
    for (0..idx) |i| {
        const d = patch[i] - mean;
        var2 += d * d;
        dot += ref_zero_mean[i] * d;
    }
    const denom = @sqrt(ref_var * var2);
    if (denom < 1e-6) return null;
    return dot / denom;
}

test "Dense MVS ZNCC recovers synthetic textured plane depth" {
    const gpa = std.testing.allocator;
    const w: u32 = 80;
    const h: u32 = 60;
    const k = Mat3.fromRows(
        .{ 80, 0, 40 },
        .{ 0, 80, 30 },
        .{ 0, 0, 1 },
    );

    const pose1 = CameraPose.identity;
    const pose2 = CameraPose{ .r = Mat3.identity, .t = Vec3.init(-0.5, 0, 0) };
    const z_plane: f64 = 4.0;

    const img1_gray = try gpa.alloc(u8, w * h);
    defer gpa.free(img1_gray);
    const img1_rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(img1_rgb);
    const img2_gray = try gpa.alloc(u8, w * h);
    defer gpa.free(img2_gray);
    const img2_rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(img2_rgb);

    // Texture world plane Z = 4.0 and project to both views
    for (0..h) |y| {
        for (0..w) |x| {
            const xw1 = (@as(f64, @floatFromInt(x)) - 40.0) * z_plane / 80.0;
            const yw1 = (@as(f64, @floatFromInt(y)) - 30.0) * z_plane / 80.0;
            const v1 = 127.0 + 60.0 * @sin(xw1 * 6.0) * @cos(yw1 * 6.0) + 40.0 * @sin(xw1 * 11.0 + yw1 * 7.0);
            const p1: u8 = @intFromFloat(std.math.clamp(v1, 0.0, 255.0));
            img1_gray[y * w + x] = p1;
            @memset(img1_rgb[(y * w + x) * 3 ..][0..3], p1);

            // In pose2 (t.x = -0.5), X_cam2 = X_world - 0.5
            const xw2 = ((@as(f64, @floatFromInt(x)) - 40.0) * z_plane / 80.0) + 0.5;
            const yw2 = yw1;
            const v2 = 127.0 + 60.0 * @sin(xw2 * 6.0) * @cos(yw2 * 6.0) + 40.0 * @sin(xw2 * 11.0 + yw2 * 7.0);
            const p2: u8 = @intFromFloat(std.math.clamp(v2, 0.0, 255.0));
            img2_gray[y * w + x] = p2;
            @memset(img2_rgb[(y * w + x) * 3 ..][0..3], p2);
        }
    }

    const images = [_]dataset.ImageData{
        .{ .width = w, .height = h, .rgb = img1_rgb, .gray = img1_gray },
        .{ .width = w, .height = h, .rgb = img2_rgb, .gray = img2_gray },
    };
    const poses = [_]CameraPose{ pose1, pose2 };
    const reg = [_]bool{ true, true };
    var sparse: [12]Vec3 = undefined;
    for (0..12) |i| {
        sparse[i] = Vec3.init(@as(f64, @floatFromInt(i)) * 0.1 - 0.5, 0, z_plane);
    }

    var cloud = try runDenseMvs(gpa, k, &images, &poses, &reg, &sparse, .{
        .stride = 4,
        .depth_steps = 40,
        .min_zncc = 0.85,
        .num_threads = 1,
    });
    defer cloud.deinit(gpa);

    try std.testing.expect(cloud.points.len >= 10);
    var err_sum: f64 = 0;
    for (cloud.points) |pt| err_sum += @abs(pt.z - z_plane);
    try std.testing.expect(err_sum / @as(f64, @floatFromInt(cloud.points.len)) < 0.25);
}
