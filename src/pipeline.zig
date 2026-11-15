const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const features = @import("features.zig");
const epipolar = @import("epipolar.zig");
const triangulation = @import("triangulation.zig");
const pnp = @import("pnp.zig");
const ba = @import("bundle_adjust.zig");
const dataset = @import("dataset.zig");
const Color = @import("io_ply.zig").Color;

pub const TrackObs = struct {
    cam_idx: usize,
    kp_idx: usize,
    uv: Vec2,
};

pub const PairwiseVerified = struct {
    cam1: usize,
    cam2: usize,
    e: Mat3,
    matches: []features.Match,

    pub fn deinit(self: *PairwiseVerified, allocator: std.mem.Allocator) void {
        allocator.free(self.matches);
        self.* = undefined;
    }
};

pub const SfmConfig = struct {
    max_keypoints: usize = 1500,
    match_ratio: f64 = 0.78,
    max_hamming: u32 = 64,
    ransac_iters: usize = 300,
    ransac_thresh_px: f64 = 2.0,
    reproj_thresh_px: f64 = 3.0,
    min_triang_angle_rad: f64 = 0.02, // ~1.15 deg
    seed: u64 = 42,
};

pub const SfmResult = struct {
    poses: []CameraPose,
    registered: []bool,
    points: []Vec3,
    colors: []Color,
    mean_reproj_err: f64,
    registered_count: usize,

    pub fn deinit(self: *SfmResult, allocator: std.mem.Allocator) void {
        allocator.free(self.poses);
        allocator.free(self.registered);
        allocator.free(self.points);
        allocator.free(self.colors);
        self.* = undefined;
    }
};

pub fn runIncrementalSfm(
    allocator: std.mem.Allocator,
    k: Mat3,
    images: []const dataset.ImageData,
    config: SfmConfig,
) !SfmResult {
    const n_cams = images.len;
    if (n_cams < 2) return error.NotEnoughImages;

    // 1. Feature detection
    const all_kps = try allocator.alloc([]features.Keypoint, n_cams);
    defer {
        for (all_kps) |kps| allocator.free(kps);
        allocator.free(all_kps);
    }
    const all_uvs = try allocator.alloc([]Vec2, n_cams);
    defer {
        for (all_uvs) |uvs| allocator.free(uvs);
        allocator.free(all_uvs);
    }
    const all_cols = try allocator.alloc([]Color, n_cams);
    defer {
        for (all_cols) |cols| allocator.free(cols);
        allocator.free(all_cols);
    }

    for (images, 0..) |img, ci| {
        all_kps[ci] = try features.detectAndCompute(
            allocator,
            .{ .width = img.width, .height = img.height, .gray = img.gray },
            config.max_keypoints,
        );
        all_uvs[ci] = try allocator.alloc(Vec2, all_kps[ci].len);
        all_cols[ci] = try allocator.alloc(Color, all_kps[ci].len);
        for (all_kps[ci], 0..) |kp, ki| {
            all_uvs[ci][ki] = Vec2.init(kp.x, kp.y);
            const px: usize = @intFromFloat(std.math.clamp(@round(kp.x), 0.0, @as(f64, @floatFromInt(img.width - 1))));
            const py: usize = @intFromFloat(std.math.clamp(@round(kp.y), 0.0, @as(f64, @floatFromInt(img.height - 1))));
            const rgb_idx = (py * img.width + px) * 3;
            all_cols[ci][ki] = .{
                .r = img.rgb[rgb_idx + 0],
                .g = img.rgb[rgb_idx + 1],
                .b = img.rgb[rgb_idx + 2],
            };
        }
    }

    // 2. Pairwise matching + geometric verification
    var verified_pairs: std.ArrayList(PairwiseVerified) = .empty;
    defer {
        for (verified_pairs.items) |*vp| vp.deinit(allocator);
        verified_pairs.deinit(allocator);
    }

    for (0..n_cams) |i| {
        for (i + 1..n_cams) |j| {
            const raw_matches = try features.matchDescriptors(
                allocator,
                all_kps[i],
                all_kps[j],
                config.match_ratio,
                config.max_hamming,
            );
            defer allocator.free(raw_matches);
            if (raw_matches.len < 16) continue;

            const pts1 = try allocator.alloc(Vec2, raw_matches.len);
            defer allocator.free(pts1);
            const pts2 = try allocator.alloc(Vec2, raw_matches.len);
            defer allocator.free(pts2);
            for (raw_matches, 0..) |m, idx| {
                pts1[idx] = all_uvs[i][m.idx1];
                pts2[idx] = all_uvs[j][m.idx2];
            }

            var er = (try epipolar.estimateEssentialRansac(
                allocator,
                k,
                pts1,
                pts2,
                config.ransac_iters,
                config.ransac_thresh_px,
                config.seed +% @as(u64, i * 1000 + j),
            )) orelse continue;
            defer er.deinit(allocator);

            if (er.inliers.len < 16) continue;

            const inlier_matches = try allocator.alloc(features.Match, er.inliers.len);
            for (er.inliers, 0..) |in_idx, out_idx| {
                inlier_matches[out_idx] = raw_matches[in_idx];
            }

            try verified_pairs.append(allocator, .{
                .cam1 = i,
                .cam2 = j,
                .e = er.e,
                .matches = inlier_matches,
            });
        }
    }

    return runIncrementalSfmFromTracks(
        allocator,
        k,
        all_uvs,
        all_cols,
        verified_pairs.items,
        config,
    );
}

pub fn runIncrementalSfmFromTracks(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_uvs: []const []const Vec2,
    all_cols: ?[]const []const Color,
    verified_pairs: []const PairwiseVerified,
    config: SfmConfig,
) !SfmResult {
    const n_cams = all_uvs.len;
    if (verified_pairs.len == 0) return error.NoVerifiedPairs;

    // Build Union-Find over all (cam_idx, kp_idx) nodes
    const cam_offsets = try allocator.alloc(usize, n_cams + 1);
    defer allocator.free(cam_offsets);
    cam_offsets[0] = 0;
    for (0..n_cams) |ci| {
        cam_offsets[ci + 1] = cam_offsets[ci] + all_uvs[ci].len;
    }
    const total_nodes = cam_offsets[n_cams];

    const parent = try allocator.alloc(usize, total_nodes);
    defer allocator.free(parent);
    for (0..total_nodes) |i| parent[i] = i;

    for (verified_pairs) |vp| {
        for (vp.matches) |m| {
            const n1 = cam_offsets[vp.cam1] + m.idx1;
            const n2 = cam_offsets[vp.cam2] + m.idx2;
            ufUnion(parent, n1, n2);
        }
    }

    // Collect tracks grouped by root
    var root_to_track = std.AutoHashMap(usize, usize).init(allocator);
    defer root_to_track.deinit();

    var tracks: std.ArrayList(std.ArrayList(TrackObs)) = .empty;
    defer {
        for (tracks.items) |*tr| tr.deinit(allocator);
        tracks.deinit(allocator);
    }

    for (0..n_cams) |ci| {
        for (all_uvs[ci], 0..) |uv, ki| {
            const node = cam_offsets[ci] + ki;
            const root = ufFind(parent, node);
            const gop = try root_to_track.getOrPut(root);
            if (!gop.found_existing) {
                gop.value_ptr.* = tracks.items.len;
                try tracks.append(allocator, .empty);
            }
            try tracks.items[gop.value_ptr.*].append(allocator, .{
                .cam_idx = ci,
                .kp_idx = ki,
                .uv = uv,
            });
        }
    }

    // Filter out single-observation nodes and inconsistent tracks (two keypoints in same cam)
    var valid_tracks: std.ArrayList([]TrackObs) = .empty;
    defer {
        for (valid_tracks.items) |tr| allocator.free(tr);
        valid_tracks.deinit(allocator);
    }

    for (tracks.items) |tr| {
        if (tr.items.len < 2) continue;
        var consistent = true;
        for (0..tr.items.len) |a_i| {
            for (a_i + 1..tr.items.len) |b_i| {
                if (tr.items[a_i].cam_idx == tr.items[b_i].cam_idx) {
                    consistent = false;
                    break;
                }
            }
            if (!consistent) break;
        }
        if (consistent) {
            try valid_tracks.append(allocator, try allocator.dupe(TrackObs, tr.items));
        }
    }

    const poses = try allocator.alloc(CameraPose, n_cams);
    errdefer allocator.free(poses);
    for (poses) |*p| p.* = CameraPose.identity;

    const registered = try allocator.alloc(bool, n_cams);
    errdefer allocator.free(registered);
    @memset(registered, false);

    // 3. Pick best initial pair (most inliers among pairs with good pose recovery)
    var best_pair_idx: ?usize = null;
    var best_pair_score: usize = 0;
    var best_pose2: CameraPose = CameraPose.identity;

    for (verified_pairs, 0..) |vp, vp_idx| {
        if (vp.matches.len <= best_pair_score) continue;
        const pts1 = try allocator.alloc(Vec2, vp.matches.len);
        defer allocator.free(pts1);
        const pts2 = try allocator.alloc(Vec2, vp.matches.len);
        defer allocator.free(pts2);
        const idx_list = try allocator.alloc(usize, vp.matches.len);
        defer allocator.free(idx_list);

        for (vp.matches, 0..) |m, i| {
            pts1[i] = all_uvs[vp.cam1][m.idx1];
            pts2[i] = all_uvs[vp.cam2][m.idx2];
            idx_list[i] = i;
        }

        const rec = (try epipolar.recoverPoseFromEssential(allocator, k, vp.e, pts1, pts2, idx_list)) orelse continue;
        if (rec.positive_depth_count > best_pair_score) {
            best_pair_score = rec.positive_depth_count;
            best_pair_idx = vp_idx;
            best_pose2 = rec.pose;
        }
    }

    const init_vp = verified_pairs[best_pair_idx orelse return error.InitializationFailed];
    registered[init_vp.cam1] = true;
    registered[init_vp.cam2] = true;
    poses[init_vp.cam1] = CameraPose.identity;
    poses[init_vp.cam2] = best_pose2;

    // Track triangulation state
    const track_pt_idx = try allocator.alloc(?usize, valid_tracks.items.len);
    defer allocator.free(track_pt_idx);
    @memset(track_pt_idx, null);

    var points3d: std.ArrayList(Vec3) = .empty;
    errdefer points3d.deinit(allocator);
    var point_colors: std.ArrayList(Color) = .empty;
    errdefer point_colors.deinit(allocator);

    try triangulateNewTracks(
        allocator,
        k,
        poses,
        registered,
        valid_tracks.items,
        track_pt_idx,
        &points3d,
        &point_colors,
        all_cols,
        config,
    );

    if (points3d.items.len < 6) return error.TooFewInitialPoints;

    try runGlobalBA(allocator, k, poses, registered, valid_tracks.items, track_pt_idx, points3d.items);

    // 4. Incremental registration loop
    while (true) {
        var best_cam: ?usize = null;
        var best_obs_count: usize = 0;

        for (0..n_cams) |ci| {
            if (registered[ci]) continue;
            var count: usize = 0;
            for (valid_tracks.items, 0..) |tr, ti| {
                if (track_pt_idx[ti] == null) continue;
                for (tr) |obs| {
                    if (obs.cam_idx == ci) {
                        count += 1;
                        break;
                    }
                }
            }
            if (count > best_obs_count) {
                best_obs_count = count;
                best_cam = ci;
            }
        }

        const next_cam = best_cam orelse break;
        if (best_obs_count < 6) break;

        var pnp_3d: std.ArrayList(Vec3) = .empty;
        defer pnp_3d.deinit(allocator);
        var pnp_2d: std.ArrayList(Vec2) = .empty;
        defer pnp_2d.deinit(allocator);

        for (valid_tracks.items, 0..) |tr, ti| {
            const pt_idx = track_pt_idx[ti] orelse continue;
            for (tr) |obs| {
                if (obs.cam_idx == next_cam) {
                    try pnp_3d.append(allocator, points3d.items[pt_idx]);
                    try pnp_2d.append(allocator, obs.uv);
                    break;
                }
            }
        }

        var pnp_res = (try pnp.solvePnpRansac(
            allocator,
            k,
            pnp_3d.items,
            pnp_2d.items,
            config.ransac_iters,
            config.reproj_thresh_px,
            config.seed +% @as(u64, next_cam),
        )) orelse break;
        defer pnp_res.deinit(allocator);

        poses[next_cam] = pnp_res.pose;
        registered[next_cam] = true;

        try triangulateNewTracks(
            allocator,
            k,
            poses,
            registered,
            valid_tracks.items,
            track_pt_idx,
            &points3d,
            &point_colors,
            all_cols,
            config,
        );

        try runGlobalBA(allocator, k, poses, registered, valid_tracks.items, track_pt_idx, points3d.items);
    }

    // 5. Final RMSE and registered count
    var reg_count: usize = 0;
    for (registered) |r| {
        if (r) reg_count += 1;
    }

    var err_sum: f64 = 0;
    var err_cnt: usize = 0;
    for (valid_tracks.items, 0..) |tr, ti| {
        const pt_idx = track_pt_idx[ti] orelse continue;
        for (tr) |obs| {
            if (!registered[obs.cam_idx]) continue;
            if (triangulation.reprojectionError(k, poses[obs.cam_idx], points3d.items[pt_idx], obs.uv)) |e| {
                if (e <= config.reproj_thresh_px * 2.0) {
                    err_sum += e;
                    err_cnt += 1;
                }
            }
        }
    }
    const mean_err = if (err_cnt > 0) err_sum / @as(f64, @floatFromInt(err_cnt)) else 0;

    return .{
        .poses = poses,
        .registered = registered,
        .points = try points3d.toOwnedSlice(allocator),
        .colors = try point_colors.toOwnedSlice(allocator),
        .mean_reproj_err = mean_err,
        .registered_count = reg_count,
    };
}

fn triangulateNewTracks(
    allocator: std.mem.Allocator,
    k: Mat3,
    poses: []const CameraPose,
    registered: []const bool,
    tracks: []const []TrackObs,
    track_pt_idx: []?usize,
    points3d: *std.ArrayList(Vec3),
    point_colors: *std.ArrayList(Color),
    all_cols: ?[]const []const Color,
    config: SfmConfig,
) !void {
    var projs: std.ArrayList([3][4]f64) = .empty;
    defer projs.deinit(allocator);
    var obs2d: std.ArrayList(Vec2) = .empty;
    defer obs2d.deinit(allocator);
    var obs_cams: std.ArrayList(usize) = .empty;
    defer obs_cams.deinit(allocator);

    for (tracks, 0..) |tr, ti| {
        if (track_pt_idx[ti] != null) continue;

        projs.clearRetainingCapacity();
        obs2d.clearRetainingCapacity();
        obs_cams.clearRetainingCapacity();

        for (tr) |obs| {
            if (registered[obs.cam_idx]) {
                try projs.append(allocator, triangulation.makeProjectionMatrix(k, poses[obs.cam_idx]));
                try obs2d.append(allocator, obs.uv);
                try obs_cams.append(allocator, obs.cam_idx);
            }
        }
        if (projs.items.len < 2) continue;

        const pt = (try triangulation.triangulateMultiView(allocator, projs.items, obs2d.items)) orelse continue;

        // Check positive depth, reprojection error, and minimum triangulation angle
        var ok = true;
        var max_angle: f64 = 0;
        for (obs_cams.items, obs2d.items) |ci, uv| {
            const xc = poses[ci].transformPoint(pt);
            if (xc.z <= 1e-4) {
                ok = false;
                break;
            }
            const err = triangulation.reprojectionError(k, poses[ci], pt, uv) orelse {
                ok = false;
                break;
            };
            if (err > config.reproj_thresh_px * 2.0) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        for (0..obs_cams.items.len) |a_i| {
            const c1 = poses[obs_cams.items[a_i]].cameraCenter();
            for (a_i + 1..obs_cams.items.len) |b_i| {
                const c2 = poses[obs_cams.items[b_i]].cameraCenter();
                const ang = triangulation.triangulationAngle(c1, c2, pt);
                if (ang > max_angle) max_angle = ang;
            }
        }
        if (max_angle < config.min_triang_angle_rad) continue;

        track_pt_idx[ti] = points3d.items.len;
        try points3d.append(allocator, pt);
        if (all_cols) |cols| {
            const first = tr[0];
            try point_colors.append(allocator, cols[first.cam_idx][first.kp_idx]);
        } else {
            try point_colors.append(allocator, .{ .r = 220, .g = 220, .b = 220 });
        }
    }
}

fn runGlobalBA(
    allocator: std.mem.Allocator,
    k: Mat3,
    poses: []CameraPose,
    registered: []const bool,
    tracks: []const []TrackObs,
    track_pt_idx: []const ?usize,
    points3d: []Vec3,
) !void {
    // Map global cam_idx -> compact registered cam_idx
    const cam_to_compact = try allocator.alloc(?usize, poses.len);
    defer allocator.free(cam_to_compact);
    var compact_to_cam: std.ArrayList(usize) = .empty;
    defer compact_to_cam.deinit(allocator);

    for (registered, 0..) |is_reg, ci| {
        if (is_reg) {
            cam_to_compact[ci] = compact_to_cam.items.len;
            try compact_to_cam.append(allocator, ci);
        } else {
            cam_to_compact[ci] = null;
        }
    }

    const compact_poses = try allocator.alloc(CameraPose, compact_to_cam.items.len);
    defer allocator.free(compact_poses);
    for (compact_to_cam.items, 0..) |ci, c_idx| {
        compact_poses[c_idx] = poses[ci];
    }

    var obs_list: std.ArrayList(ba.Observation) = .empty;
    defer obs_list.deinit(allocator);

    for (tracks, 0..) |tr, ti| {
        const pt_idx = track_pt_idx[ti] orelse continue;
        for (tr) |obs| {
            if (cam_to_compact[obs.cam_idx]) |c_idx| {
                try obs_list.append(allocator, .{
                    .cam_idx = c_idx,
                    .pt_idx = pt_idx,
                    .uv = obs.uv,
                });
            }
        }
    }

    _ = try ba.runBundleAdjustment(allocator, k, compact_poses, points3d, obs_list.items, .{});

    for (compact_to_cam.items, 0..) |ci, c_idx| {
        poses[ci] = compact_poses[c_idx];
    }
}

fn ufFind(parent: []usize, x: usize) usize {
    var cur = x;
    while (parent[cur] != cur) {
        parent[cur] = parent[parent[cur]];
        cur = parent[cur];
    }
    return cur;
}

fn ufUnion(parent: []usize, a: usize, b: usize) void {
    const ra = ufFind(parent, a);
    const rb = ufFind(parent, b);
    if (ra != rb) {
        parent[rb] = ra;
    }
}

test "End-to-end incremental SfM recovers 5-camera trajectory and 3D structure" {
    const gpa = std.testing.allocator;
    const k = Mat3.fromRows(
        .{ 800, 0, 320 },
        .{ 0, 800, 240 },
        .{ 0, 0, 1 },
    );

    const n_cams = 5;
    const gt_poses = [n_cams]CameraPose{
        CameraPose.identity,
        .{ .r = lie.so3Exp(Vec3.init(0.01, -0.10, 0.0)), .t = Vec3.init(-0.8, 0.02, 0.05) },
        .{ .r = lie.so3Exp(Vec3.init(-0.02, -0.20, 0.01)), .t = Vec3.init(-1.6, -0.02, 0.15) },
        .{ .r = lie.so3Exp(Vec3.init(0.01, -0.30, -0.01)), .t = Vec3.init(-2.4, 0.04, 0.30) },
        .{ .r = lie.so3Exp(Vec3.init(0.0, -0.40, 0.02)), .t = Vec3.init(-3.2, 0.0, 0.50) },
    };

    var prng = std.Random.DefaultPrng.init(2026);
    const rng = prng.random();

    const n_pts = 40;
    var gt_pts: [n_pts]Vec3 = undefined;
    for (&gt_pts) |*p| {
        p.* = Vec3.init(
            (rng.float(f64) - 0.5) * 4.0,
            (rng.float(f64) - 0.5) * 3.0,
            5.0 + rng.float(f64) * 5.0,
        );
    }

    var uvs_storage: [n_cams][n_pts]Vec2 = undefined;
    var uvs_slices: [n_cams][]const Vec2 = undefined;
    for (0..n_cams) |ci| {
        for (0..n_pts) |pi| {
            const proj = lie.projectPoint(k, gt_poses[ci], gt_pts[pi]).?;
            // Add realistic +/- 0.25 px noise
            uvs_storage[ci][pi] = Vec2.init(
                proj.x + (rng.float(f64) - 0.5) * 0.5,
                proj.y + (rng.float(f64) - 0.5) * 0.5,
            );
        }
        uvs_slices[ci] = &uvs_storage[ci];
    }

    // Build consecutive verified pairs (0-1, 1-2, 2-3, 3-4)
    var pairs: [n_cams - 1]PairwiseVerified = undefined;
    for (0..n_cams - 1) |ci| {
        var er = (try epipolar.estimateEssentialRansac(
            gpa,
            k,
            uvs_slices[ci],
            uvs_slices[ci + 1],
            200,
            2.0,
            @as(u64, ci + 1),
        )).?;
        defer er.deinit(gpa);

        const matches = try gpa.alloc(features.Match, er.inliers.len);
        for (er.inliers, 0..) |idx, m_i| {
            matches[m_i] = .{ .idx1 = idx, .idx2 = idx, .dist = 0 };
        }
        pairs[ci] = .{
            .cam1 = ci,
            .cam2 = ci + 1,
            .e = er.e,
            .matches = matches,
        };
    }
    defer for (&pairs) |*vp| vp.deinit(gpa);

    var res = try runIncrementalSfmFromTracks(gpa, k, &uvs_slices, null, &pairs, .{});
    defer res.deinit(gpa);

    try std.testing.expectEqual(@as(usize, n_cams), res.registered_count);
    try std.testing.expect(res.points.len >= 35);
    try std.testing.expect(res.mean_reproj_err < 1.0);
}
