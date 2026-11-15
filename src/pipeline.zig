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
const gpu = @import("gpu.zig");
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
    max_keypoints: usize = 10500,
    match_ratio: f64 = 0.80,
    max_hamming: u32 = 72,
    ransac_iters: usize = 300,
    ransac_thresh_px: f64 = 2.0,
    reproj_thresh_px: f64 = 3.0,
    min_triang_angle_rad: f64 = 0.03,
    seed: u64 = 42,
    /// 1 = single-threaded CPU, 0 = auto-detect CPU core count, >1 = explicit thread count
    num_threads: usize = 1,
    /// Enable WebGPU compute acceleration (Harris corner scoring and batched descriptor matching).
    use_gpu: bool = false,
    enable_guided_densification: bool = true,
    guided_epipolar_thresh_px: f64 = 1.4,
    guided_match_ratio: f64 = 0.90,
    guided_max_hamming: u32 = 86,
    guided_reproj_thresh_px: f64 = 1.0,
    guided_min_angle_rad: f64 = 0.05,
    guided_min_zncc: f64 = 0.82,
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

    const effective_threads = resolveThreadCount(config.num_threads);

    var gpu_ctx_opt: ?gpu.GpuContext = if (config.use_gpu) try gpu.GpuContext.init() else null;
    defer if (gpu_ctx_opt) |*g| g.deinit();

    // 1. Feature detection (CPU single/multi-threaded or WebGPU Harris + CPU NMS/BRIEF)
    const all_kps = try allocator.alloc([]features.Keypoint, n_cams);
    for (all_kps) |*slot| slot.* = &.{};
    defer {
        for (all_kps) |kps| allocator.free(kps);
        allocator.free(all_kps);
    }
    const all_uvs = try allocator.alloc([]Vec2, n_cams);
    for (all_uvs) |*slot| slot.* = &.{};
    defer {
        for (all_uvs) |uvs| allocator.free(uvs);
        allocator.free(all_uvs);
    }
    const all_cols = try allocator.alloc([]Color, n_cams);
    for (all_cols) |*slot| slot.* = &.{};
    defer {
        for (all_cols) |cols| allocator.free(cols);
        allocator.free(all_cols);
    }

    if (gpu_ctx_opt) |*gpu_ctx| {
        const default_tile: features.TileConfig = .{};
        const harris_maps = try allocator.alloc(?gpu.GpuContext.HarrisGpuOutput, n_cams);
        @memset(harris_maps, null);
        defer {
            for (harris_maps) |hm_opt| {
                if (hm_opt) |hm| {
                    allocator.free(hm.blurred);
                    allocator.free(hm.score);
                }
            }
            allocator.free(harris_maps);
        }

        for (images, 0..) |img, ci| {
            if (img.width >= 40 and img.height >= 40 and config.max_keypoints > 0) {
                harris_maps[ci] = try gpu_ctx.computeHarrisScores(
                    allocator,
                    img.gray,
                    img.width,
                    img.height,
                    default_tile.min_harris_score,
                );
            }
        }

        if (effective_threads <= 1 or n_cams <= 1) {
            for (0..n_cams) |ci| {
                try finishGpuImageFeatures(
                    allocator,
                    images[ci],
                    harris_maps[ci],
                    config.max_keypoints,
                    default_tile,
                    &all_kps[ci],
                    &all_uvs[ci],
                    &all_cols[ci],
                );
            }
        } else {
            var next_cam = std.atomic.Value(usize).init(0);
            const worker_count = @min(effective_threads, n_cams);
            const threads = try allocator.alloc(std.Thread, worker_count);
            defer allocator.free(threads);
            var failed = std.atomic.Value(bool).init(false);

            for (threads) |*th| {
                th.* = try std.Thread.spawn(.{}, finishGpuDetectWorker, .{
                    allocator,
                    images,
                    harris_maps,
                    config.max_keypoints,
                    default_tile,
                    all_kps,
                    all_uvs,
                    all_cols,
                    &next_cam,
                    &failed,
                });
            }
            for (threads) |th| th.join();
            if (failed.load(.acquire)) return error.FeatureExtractionFailed;
        }
    } else if (effective_threads <= 1 or n_cams <= 1) {
        for (0..n_cams) |ci| {
            try detectImageFeatures(allocator, images[ci], config.max_keypoints, &all_kps[ci], &all_uvs[ci], &all_cols[ci]);
        }
    } else {
        var next_cam = std.atomic.Value(usize).init(0);
        const worker_count = @min(effective_threads, n_cams);
        const threads = try allocator.alloc(std.Thread, worker_count);
        defer allocator.free(threads);
        var failed = std.atomic.Value(bool).init(false);

        for (threads) |*th| {
            th.* = try std.Thread.spawn(.{}, detectWorker, .{
                allocator,
                images,
                config.max_keypoints,
                all_kps,
                all_uvs,
                all_cols,
                &next_cam,
                &failed,
            });
        }
        for (threads) |th| th.join();
        if (failed.load(.acquire)) return error.FeatureExtractionFailed;
    }

    // 2. Pairwise matching + geometric verification (WebGPU batched or CPU single/multi-threaded)
    const n_pairs = (n_cams * (n_cams - 1)) / 2;
    const pair_indices = try allocator.alloc([2]usize, n_pairs);
    defer allocator.free(pair_indices);
    var p_idx: usize = 0;
    for (0..n_cams) |i| {
        for (i + 1..n_cams) |j| {
            pair_indices[p_idx] = .{ i, j };
            p_idx += 1;
        }
    }

    const pair_slots = try allocator.alloc(?PairwiseVerified, n_pairs);
    @memset(pair_slots, null);
    defer {
        for (pair_slots) |*slot| {
            if (slot.*) |*vp| vp.deinit(allocator);
        }
        allocator.free(pair_slots);
    }

    var gpu_raw_matches: ?[][]features.Match = null;
    if (gpu_ctx_opt) |*gpu_ctx| {
        gpu_raw_matches = try gpu_ctx.matchDescriptorPairsBatch(
            allocator,
            all_kps,
            pair_indices,
            config.match_ratio,
            config.max_hamming,
        );
    }
    defer if (gpu_raw_matches) |batch| {
        for (batch) |ms| allocator.free(ms);
        allocator.free(batch);
    };

    if (effective_threads <= 1 or n_pairs <= 1) {
        for (pair_indices, 0..) |pair, idx| {
            pair_slots[idx] = if (gpu_raw_matches) |batch|
                try verifyPairFromMatches(allocator, k, all_uvs, pair[0], pair[1], batch[idx], config)
            else
                try verifyPair(allocator, k, all_kps, all_uvs, pair[0], pair[1], config);
        }
    } else {
        var next_pair = std.atomic.Value(usize).init(0);
        const worker_count = @min(effective_threads, n_pairs);
        const threads = try allocator.alloc(std.Thread, worker_count);
        defer allocator.free(threads);
        var failed = std.atomic.Value(bool).init(false);

        for (threads) |*th| {
            th.* = try std.Thread.spawn(.{}, verifyPairWorker, .{
                allocator,
                k,
                all_kps,
                all_uvs,
                pair_indices,
                gpu_raw_matches,
                pair_slots,
                config,
                &next_pair,
                &failed,
            });
        }
        for (threads) |th| th.join();
        if (failed.load(.acquire)) return error.PairVerificationFailed;
    }

    var verified_pairs: std.ArrayList(PairwiseVerified) = .empty;
    defer verified_pairs.deinit(allocator);
    for (pair_slots) |slot| {
        if (slot) |vp| {
            try verified_pairs.append(allocator, vp);
        }
    }

    return runIncrementalSfmInternal(
        allocator,
        k,
        all_uvs,
        all_cols,
        verified_pairs.items,
        config,
        if (config.enable_guided_densification) .{
            .images = images,
            .all_kps = all_kps,
            .effective_threads = effective_threads,
        } else null,
    );
}

const GuidedDensifyCtx = struct {
    images: []const dataset.ImageData,
    all_kps: []const []const features.Keypoint,
    effective_threads: usize,
};

pub fn runIncrementalSfmFromTracks(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_uvs: []const []const Vec2,
    all_cols: ?[]const []const Color,
    verified_pairs: []const PairwiseVerified,
    config: SfmConfig,
) !SfmResult {
    return runIncrementalSfmInternal(allocator, k, all_uvs, all_cols, verified_pairs, config, null);
}

fn runIncrementalSfmInternal(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_uvs: []const []const Vec2,
    all_cols: ?[]const []const Color,
    verified_pairs: []const PairwiseVerified,
    config: SfmConfig,
    guided_ctx: ?GuidedDensifyCtx,
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
    var track_pt_idx: std.ArrayList(?usize) = .empty;
    defer track_pt_idx.deinit(allocator);
    try track_pt_idx.resize(allocator, valid_tracks.items.len);
    @memset(track_pt_idx.items, null);

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
        track_pt_idx.items,
        &points3d,
        &point_colors,
        all_cols,
        config,
    );

    if (points3d.items.len < 6) return error.TooFewInitialPoints;

    try runGlobalBA(allocator, k, poses, registered, valid_tracks.items, track_pt_idx.items, points3d.items);

    // 4. Incremental registration loop
    while (true) {
        var best_cam: ?usize = null;
        var best_obs_count: usize = 0;

        for (0..n_cams) |ci| {
            if (registered[ci]) continue;
            var count: usize = 0;
            for (valid_tracks.items, 0..) |tr, ti| {
                if (track_pt_idx.items[ti] == null) continue;
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
            const pt_idx = track_pt_idx.items[ti] orelse continue;
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
            track_pt_idx.items,
            &points3d,
            &point_colors,
            all_cols,
            config,
        );

        try runGlobalBA(allocator, k, poses, registered, valid_tracks.items, track_pt_idx.items, points3d.items);
    }

    if (guided_ctx) |gctx| {
        const added = try densifyViaEpipolarGuidedMatching(
            allocator,
            k,
            poses,
            registered,
            gctx.images,
            gctx.all_kps,
            all_uvs,
            all_cols,
            &valid_tracks,
            &track_pt_idx,
            &points3d,
            &point_colors,
            config,
            gctx.effective_threads,
        );
        if (added > 0) {
            try runGlobalBAWithIters(allocator, k, poses, registered, valid_tracks.items, track_pt_idx.items, points3d.items, 12);
        }
    }

    // 5. Filter extreme depth/reprojection outliers from final point cloud
    var cam_centroid = Vec3.zero;
    var reg_count: usize = 0;
    for (poses, registered) |p, r| {
        if (r) {
            cam_centroid = cam_centroid.add(p.cameraCenter());
            reg_count += 1;
        }
    }
    if (reg_count > 0) {
        cam_centroid = cam_centroid.scale(1.0 / @as(f64, @floatFromInt(reg_count)));
    }

    const pt_dists = try allocator.alloc(f64, points3d.items.len);
    defer allocator.free(pt_dists);
    for (points3d.items, 0..) |pt, idx| {
        pt_dists[idx] = pt.dist(cam_centroid);
    }
    const sorted_dists = try allocator.dupe(f64, pt_dists);
    defer allocator.free(sorted_dists);
    std.mem.sort(f64, sorted_dists, {}, std.sort.asc(f64));
    const median_dist = if (sorted_dists.len > 0) sorted_dists[sorted_dists.len / 2] else 1.0;
    const max_dist = @max(median_dist * 5.0, 1e-3);

    var clean_pts: std.ArrayList(Vec3) = .empty;
    errdefer clean_pts.deinit(allocator);
    var clean_cols: std.ArrayList(Color) = .empty;
    errdefer clean_cols.deinit(allocator);
    const keep_pt = try allocator.alloc(bool, points3d.items.len);
    defer allocator.free(keep_pt);
    for (points3d.items, point_colors.items, pt_dists, 0..) |pt, col, d, idx| {
        if (std.math.isFinite(pt.x) and std.math.isFinite(pt.y) and std.math.isFinite(pt.z) and d <= max_dist) {
            keep_pt[idx] = true;
            try clean_pts.append(allocator, pt);
            try clean_cols.append(allocator, col);
        } else {
            keep_pt[idx] = false;
        }
    }

    var err_sum: f64 = 0;
    var err_cnt: usize = 0;
    for (valid_tracks.items, 0..) |tr, ti| {
        const pt_idx = track_pt_idx.items[ti] orelse continue;
        if (!keep_pt[pt_idx]) continue;
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

    points3d.deinit(allocator);
    point_colors.deinit(allocator);

    return .{
        .poses = poses,
        .registered = registered,
        .points = try clean_pts.toOwnedSlice(allocator),
        .colors = try clean_cols.toOwnedSlice(allocator),
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
    return runGlobalBAWithIters(allocator, k, poses, registered, tracks, track_pt_idx, points3d, 25);
}

fn runGlobalBAWithIters(
    allocator: std.mem.Allocator,
    k: Mat3,
    poses: []CameraPose,
    registered: []const bool,
    tracks: []const []TrackObs,
    track_pt_idx: []const ?usize,
    points3d: []Vec3,
    max_iters: usize,
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

    _ = try ba.runBundleAdjustment(allocator, k, compact_poses, points3d, obs_list.items, .{ .max_iters = max_iters });

    for (compact_to_cam.items, 0..) |ci, c_idx| {
        poses[ci] = compact_poses[c_idx];
    }
}

fn densifyViaEpipolarGuidedMatching(
    allocator: std.mem.Allocator,
    k: Mat3,
    poses: []const CameraPose,
    registered: []const bool,
    images: []const dataset.ImageData,
    all_kps: []const []const features.Keypoint,
    all_uvs: []const []const Vec2,
    all_cols: ?[]const []const Color,
    valid_tracks: *std.ArrayList([]TrackObs),
    track_pt_idx: *std.ArrayList(?usize),
    points3d: *std.ArrayList(Vec3),
    point_colors: *std.ArrayList(Color),
    config: SfmConfig,
    effective_threads: usize,
) !usize {
    const n_cams = all_uvs.len;
    const k_inv = k.inv() orelse return 0;

    // Mark keypoints already belonging to a triangulated 3D point
    const kp_used = try allocator.alloc([]bool, n_cams);
    for (kp_used, 0..) |*row, ci| {
        row.* = try allocator.alloc(bool, all_uvs[ci].len);
        @memset(row.*, false);
    }
    defer {
        for (kp_used) |row| allocator.free(row);
        allocator.free(kp_used);
    }

    for (valid_tracks.items, 0..) |tr, ti| {
        if (track_pt_idx.items[ti] == null) continue;
        for (tr) |obs| {
            kp_used[obs.cam_idx][obs.kp_idx] = true;
        }
    }

    // Compute scene centroid and median depth radius for outlier rejection
    var cam_centroid = Vec3.zero;
    var reg_count: usize = 0;
    for (poses, registered) |p, r| {
        if (r) {
            cam_centroid = cam_centroid.add(p.cameraCenter());
            reg_count += 1;
        }
    }
    if (reg_count < 2) return 0;
    cam_centroid = cam_centroid.scale(1.0 / @as(f64, @floatFromInt(reg_count)));

    var max_dist: f64 = 1e6;
    if (points3d.items.len > 0) {
        const dists = try allocator.alloc(f64, points3d.items.len);
        defer allocator.free(dists);
        for (points3d.items, 0..) |pt, idx| dists[idx] = pt.dist(cam_centroid);
        std.mem.sort(f64, dists, {}, std.sort.asc(f64));
        const med = dists[dists.len / 2];
        max_dist = @max(med * 5.0, 1e-3);
    }

    // Select overlapping registered camera pairs with sufficient viewing overlap and baseline
    var pair_list: std.ArrayList([2]usize) = .empty;
    defer pair_list.deinit(allocator);
    const z_axis = Vec3.init(0, 0, 1);
    for (0..n_cams) |c1| {
        if (!registered[c1]) continue;
        const fwd1 = poses[c1].r.transpose().mulVec(z_axis);
        const cen1 = poses[c1].cameraCenter();
        for (c1 + 1..n_cams) |c2| {
            if (!registered[c2]) continue;
            const fwd2 = poses[c2].r.transpose().mulVec(z_axis);
            if (fwd1.dot(fwd2) < 0.85) continue;
            const cen2 = poses[c2].cameraCenter();
            if (cen1.dist(cen2) < 1e-4) continue;
            try pair_list.append(allocator, .{ c1, c2 });
        }
    }
    if (pair_list.items.len == 0) return 0;

    const pair_matches = try allocator.alloc([]features.Match, pair_list.items.len);
    for (pair_matches) |*slot| slot.* = &.{};
    defer {
        for (pair_matches) |ms| allocator.free(ms);
        allocator.free(pair_matches);
    }

    if (effective_threads <= 1 or pair_list.items.len <= 1) {
        for (pair_list.items, 0..) |pair, idx| {
            pair_matches[idx] = try matchPairEpipolarGuided(
                allocator,
                k,
                k_inv,
                poses[pair[0]],
                poses[pair[1]],
                images[pair[0]],
                images[pair[1]],
                all_kps[pair[0]],
                all_kps[pair[1]],
                all_uvs[pair[0]],
                all_uvs[pair[1]],
                kp_used[pair[0]],
                kp_used[pair[1]],
                cam_centroid,
                max_dist,
                config,
            );
        }
    } else {
        var next_idx = std.atomic.Value(usize).init(0);
        var failed = std.atomic.Value(bool).init(false);
        const worker_count = @min(effective_threads, pair_list.items.len);
        const threads = try allocator.alloc(std.Thread, worker_count);
        defer allocator.free(threads);

        for (threads) |*th| {
            th.* = try std.Thread.spawn(.{}, guidedMatchWorker, .{
                allocator,
                k,
                k_inv,
                poses,
                images,
                all_kps,
                all_uvs,
                kp_used,
                pair_list.items,
                pair_matches,
                cam_centroid,
                max_dist,
                config,
                &next_idx,
                &failed,
            });
        }
        for (threads) |th| th.join();
        if (failed.load(.acquire)) return error.GuidedDensificationFailed;
    }

    // Merge guided matches across pairs into multi-view tracks via Union-Find
    const cam_offsets = try allocator.alloc(usize, n_cams + 1);
    defer allocator.free(cam_offsets);
    cam_offsets[0] = 0;
    for (0..n_cams) |ci| cam_offsets[ci + 1] = cam_offsets[ci] + all_uvs[ci].len;
    const total_nodes = cam_offsets[n_cams];

    const parent = try allocator.alloc(usize, total_nodes);
    defer allocator.free(parent);
    for (0..total_nodes) |i| parent[i] = i;

    const touched = try allocator.alloc(bool, total_nodes);
    defer allocator.free(touched);
    @memset(touched, false);

    for (pair_list.items, pair_matches) |pair, matches| {
        for (matches) |m| {
            const n1 = cam_offsets[pair[0]] + m.idx1;
            const n2 = cam_offsets[pair[1]] + m.idx2;
            touched[n1] = true;
            touched[n2] = true;
            ufUnion(parent, n1, n2);
        }
    }

    var root_to_track = std.AutoHashMap(usize, usize).init(allocator);
    defer root_to_track.deinit();

    var guided_tracks: std.ArrayList(std.ArrayList(TrackObs)) = .empty;
    defer {
        for (guided_tracks.items) |*tr| tr.deinit(allocator);
        guided_tracks.deinit(allocator);
    }

    for (0..n_cams) |ci| {
        for (all_uvs[ci], 0..) |uv, ki| {
            const node = cam_offsets[ci] + ki;
            if (!touched[node]) continue;
            const root = ufFind(parent, node);
            const gop = try root_to_track.getOrPut(root);
            if (!gop.found_existing) {
                gop.value_ptr.* = guided_tracks.items.len;
                try guided_tracks.append(allocator, .empty);
            }
            try guided_tracks.items[gop.value_ptr.*].append(allocator, .{
                .cam_idx = ci,
                .kp_idx = ki,
                .uv = uv,
            });
        }
    }

    var projs: std.ArrayList([3][4]f64) = .empty;
    defer projs.deinit(allocator);
    var obs2d: std.ArrayList(Vec2) = .empty;
    defer obs2d.deinit(allocator);

    var added_count: usize = 0;
    for (guided_tracks.items) |tr| {
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
        if (!consistent) continue;

        projs.clearRetainingCapacity();
        obs2d.clearRetainingCapacity();
        for (tr.items) |obs| {
            try projs.append(allocator, triangulation.makeProjectionMatrix(k, poses[obs.cam_idx]));
            try obs2d.append(allocator, obs.uv);
        }

        const pt = (try triangulation.triangulateMultiView(allocator, projs.items, obs2d.items)) orelse continue;
        if (!std.math.isFinite(pt.x) or !std.math.isFinite(pt.y) or !std.math.isFinite(pt.z)) continue;
        if (pt.dist(cam_centroid) > max_dist) continue;

        const max_reproj = if (tr.items.len == 2) @min(config.guided_reproj_thresh_px, 0.80) else config.guided_reproj_thresh_px;
        const min_ang = if (tr.items.len == 2) @max(config.guided_min_angle_rad, 0.09) else @max(config.guided_min_angle_rad, 0.06);

        var ok = true;
        for (tr.items) |obs| {
            const xc = poses[obs.cam_idx].transformPoint(pt);
            if (xc.z <= 1e-4) {
                ok = false;
                break;
            }
            const err = triangulation.reprojectionError(k, poses[obs.cam_idx], pt, obs.uv) orelse {
                ok = false;
                break;
            };
            if (err > max_reproj) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        var max_angle: f64 = 0;
        for (0..tr.items.len) |a_i| {
            const c1 = poses[tr.items[a_i].cam_idx].cameraCenter();
            for (a_i + 1..tr.items.len) |b_i| {
                const c2 = poses[tr.items[b_i].cam_idx].cameraCenter();
                const ang = triangulation.triangulationAngle(c1, c2, pt);
                if (ang > max_angle) max_angle = ang;
            }
        }
        if (max_angle < min_ang) continue;

        const pt_idx = points3d.items.len;
        const owned_tr = try allocator.dupe(TrackObs, tr.items);
        errdefer allocator.free(owned_tr);
        try valid_tracks.append(allocator, owned_tr);
        try track_pt_idx.append(allocator, pt_idx);
        try points3d.append(allocator, pt);
        if (all_cols) |cols| {
            const first = tr.items[0];
            try point_colors.append(allocator, cols[first.cam_idx][first.kp_idx]);
        } else {
            try point_colors.append(allocator, .{ .r = 220, .g = 220, .b = 220 });
        }
        added_count += 1;
    }

    return added_count;
}

fn matchPairEpipolarGuided(
    allocator: std.mem.Allocator,
    k: Mat3,
    k_inv: Mat3,
    pose1: CameraPose,
    pose2: CameraPose,
    img1: dataset.ImageData,
    img2: dataset.ImageData,
    kps1: []const features.Keypoint,
    kps2: []const features.Keypoint,
    uvs1: []const Vec2,
    uvs2: []const Vec2,
    used1: []const bool,
    used2: []const bool,
    cam_centroid: Vec3,
    max_dist: f64,
    config: SfmConfig,
) ![]features.Match {
    var unm1: std.ArrayList(usize) = .empty;
    defer unm1.deinit(allocator);
    for (used1, 0..) |u, idx| {
        if (!u) try unm1.append(allocator, idx);
    }

    var unm2: std.ArrayList(usize) = .empty;
    defer unm2.deinit(allocator);
    for (used2, 0..) |u, idx| {
        if (!u) try unm2.append(allocator, idx);
    }

    if (unm1.items.len == 0 or unm2.items.len == 0) {
        return try allocator.alloc(features.Match, 0);
    }

    const r_rel = pose2.r.mul(pose1.r.transpose());
    const t_rel = pose2.t.sub(r_rel.mulVec(pose1.t));
    const e = Mat3.skew(t_rel).mul(r_rel);
    const f = epipolar.essentialToFundamental(k_inv, e);
    const ft = f.transpose();

    const l1_inv_norm = try allocator.alloc(f64, unm2.items.len);
    defer allocator.free(l1_inv_norm);
    for (unm2.items, 0..) |kp_idx, j| {
        const uv = uvs2[kp_idx];
        const l1 = ft.mulVec(Vec3.init(uv.x, uv.y, 1.0));
        const n = @sqrt(l1.x * l1.x + l1.y * l1.y);
        l1_inv_norm[j] = if (n > 1e-12) 1.0 / n else 0.0;
    }

    const best_12 = try allocator.alloc(?usize, unm1.items.len);
    defer allocator.free(best_12);
    const best_d_12 = try allocator.alloc(u32, unm1.items.len);
    defer allocator.free(best_d_12);

    const best_21 = try allocator.alloc(?usize, unm2.items.len);
    defer allocator.free(best_21);
    const d1_21 = try allocator.alloc(u32, unm2.items.len);
    defer allocator.free(d1_21);
    const d2_21 = try allocator.alloc(u32, unm2.items.len);
    defer allocator.free(d2_21);
    @memset(best_21, null);
    @memset(d1_21, std.math.maxInt(u32));
    @memset(d2_21, std.math.maxInt(u32));

    const epi_thresh = config.guided_epipolar_thresh_px;

    for (unm1.items, 0..) |kp_idx1, i| {
        const uv1 = uvs1[kp_idx1];
        const l2 = f.mulVec(Vec3.init(uv1.x, uv1.y, 1.0));
        const l2_norm = @sqrt(l2.x * l2.x + l2.y * l2.y);
        if (l2_norm <= 1e-12) {
            best_12[i] = null;
            continue;
        }
        const inv_l2 = 1.0 / l2_norm;
        const desc1 = kps1[kp_idx1].desc;

        var d1: u32 = std.math.maxInt(u32);
        var d2: u32 = std.math.maxInt(u32);
        var b_j: ?usize = null;

        for (unm2.items, 0..) |kp_idx2, j| {
            const inv_l1 = l1_inv_norm[j];
            if (inv_l1 == 0.0) continue;
            const uv2 = uvs2[kp_idx2];
            const num = @abs(l2.x * uv2.x + l2.y * uv2.y + l2.z);
            if (num * inv_l2 > epi_thresh or num * inv_l1 > epi_thresh) continue;

            const d = features.hammingDist(desc1, kps2[kp_idx2].desc);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                b_j = j;
            } else if (d < d2) {
                d2 = d;
            }
            if (d < d1_21[j]) {
                d2_21[j] = d1_21[j];
                d1_21[j] = d;
                best_21[j] = i;
            } else if (d < d2_21[j]) {
                d2_21[j] = d;
            }
        }

        if (b_j != null and d1 <= config.guided_max_hamming and
            @as(f64, @floatFromInt(d1)) < config.guided_match_ratio * @as(f64, @floatFromInt(d2)))
        {
            best_12[i] = b_j;
            best_d_12[i] = d1;
        } else {
            best_12[i] = null;
        }
    }

    const p1 = triangulation.makeProjectionMatrix(k, pose1);
    const p2 = triangulation.makeProjectionMatrix(k, pose2);
    const c1 = pose1.cameraCenter();
    const c2 = pose2.cameraCenter();

    var matches: std.ArrayList(features.Match) = .empty;
    defer matches.deinit(allocator);

    for (best_12, 0..) |opt_j, i| {
        const j = opt_j orelse continue;
        if (best_21[j] != i) continue;
        if (@as(f64, @floatFromInt(d1_21[j])) >= config.guided_match_ratio * @as(f64, @floatFromInt(d2_21[j]))) continue;

        const kp_idx1 = unm1.items[i];
        const kp_idx2 = unm2.items[j];
        const uv1 = uvs1[kp_idx1];
        const uv2 = uvs2[kp_idx2];

        const zncc = computePatchZncc(img1, uv1, img2, uv2, 3) orelse continue;
        if (zncc < config.guided_min_zncc) continue;

        const pt = (try triangulation.triangulatePoint(allocator, p1, p2, uv1, uv2)) orelse continue;
        if (!std.math.isFinite(pt.x) or !std.math.isFinite(pt.y) or !std.math.isFinite(pt.z)) continue;
        if (pt.dist(cam_centroid) > max_dist) continue;

        if (pose1.transformPoint(pt).z <= 1e-4 or pose2.transformPoint(pt).z <= 1e-4) continue;
        const err1 = triangulation.reprojectionError(k, pose1, pt, uv1) orelse continue;
        const err2 = triangulation.reprojectionError(k, pose2, pt, uv2) orelse continue;
        if (err1 > config.guided_reproj_thresh_px or err2 > config.guided_reproj_thresh_px) continue;
        if (triangulation.triangulationAngle(c1, c2, pt) < config.guided_min_angle_rad) continue;

        try matches.append(allocator, .{
            .idx1 = kp_idx1,
            .idx2 = kp_idx2,
            .dist = best_d_12[i],
        });
    }

    return try matches.toOwnedSlice(allocator);
}

fn computePatchZncc(
    img1: dataset.ImageData,
    uv1: Vec2,
    img2: dataset.ImageData,
    uv2: Vec2,
    radius: isize,
) ?f64 {
    const cx1: isize = @intFromFloat(@round(uv1.x));
    const cy1: isize = @intFromFloat(@round(uv1.y));
    const cx2: isize = @intFromFloat(@round(uv2.x));
    const cy2: isize = @intFromFloat(@round(uv2.y));

    if (cx1 < radius or cy1 < radius or cx1 + radius >= @as(isize, @intCast(img1.width)) or cy1 + radius >= @as(isize, @intCast(img1.height))) return null;
    if (cx2 < radius or cy2 < radius or cx2 + radius >= @as(isize, @intCast(img2.width)) or cy2 + radius >= @as(isize, @intCast(img2.height))) return null;

    const w1: usize = img1.width;
    const w2: usize = img2.width;
    var sum1: f64 = 0;
    var sum2: f64 = 0;
    var count: f64 = 0;

    var dy: isize = -radius;
    while (dy <= radius) : (dy += 1) {
        const y1: usize = @intCast(cy1 + dy);
        const y2: usize = @intCast(cy2 + dy);
        var dx: isize = -radius;
        while (dx <= radius) : (dx += 1) {
            const x1: usize = @intCast(cx1 + dx);
            const x2: usize = @intCast(cx2 + dx);
            sum1 += @floatFromInt(img1.gray[y1 * w1 + x1]);
            sum2 += @floatFromInt(img2.gray[y2 * w2 + x2]);
            count += 1.0;
        }
    }

    const m1 = sum1 / count;
    const m2 = sum2 / count;
    var cov: f64 = 0;
    var v1: f64 = 0;
    var v2: f64 = 0;

    dy = -radius;
    while (dy <= radius) : (dy += 1) {
        const y1: usize = @intCast(cy1 + dy);
        const y2: usize = @intCast(cy2 + dy);
        var dx: isize = -radius;
        while (dx <= radius) : (dx += 1) {
            const x1: usize = @intCast(cx1 + dx);
            const x2: usize = @intCast(cx2 + dx);
            const a = @as(f64, @floatFromInt(img1.gray[y1 * w1 + x1])) - m1;
            const b = @as(f64, @floatFromInt(img2.gray[y2 * w2 + x2])) - m2;
            cov += a * b;
            v1 += a * a;
            v2 += b * b;
        }
    }

    if (v1 < 4.0 or v2 < 4.0) return null;
    return cov / @sqrt(v1 * v2);
}

fn guidedMatchWorker(
    allocator: std.mem.Allocator,
    k: Mat3,
    k_inv: Mat3,
    poses: []const CameraPose,
    images: []const dataset.ImageData,
    all_kps: []const []const features.Keypoint,
    all_uvs: []const []const Vec2,
    kp_used: []const []const bool,
    pair_list: []const [2]usize,
    pair_matches: [][]features.Match,
    cam_centroid: Vec3,
    max_dist: f64,
    config: SfmConfig,
    next_idx: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const idx = next_idx.fetchAdd(1, .monotonic);
        if (idx >= pair_list.len) return;
        const pair = pair_list[idx];
        pair_matches[idx] = matchPairEpipolarGuided(
            allocator,
            k,
            k_inv,
            poses[pair[0]],
            poses[pair[1]],
            images[pair[0]],
            images[pair[1]],
            all_kps[pair[0]],
            all_kps[pair[1]],
            all_uvs[pair[0]],
            all_uvs[pair[1]],
            kp_used[pair[0]],
            kp_used[pair[1]],
            cam_centroid,
            max_dist,
            config,
        ) catch {
            failed.store(true, .release);
            return;
        };
    }
}

fn resolveThreadCount(num_threads: usize) usize {
    if (num_threads == 0) {
        return std.Thread.getCpuCount() catch 4;
    }
    return num_threads;
}

fn packImageFeaturesFromKps(
    allocator: std.mem.Allocator,
    img: dataset.ImageData,
    kps: []features.Keypoint,
    out_kps: *[]features.Keypoint,
    out_uvs: *[]Vec2,
    out_cols: *[]Color,
) !void {
    errdefer allocator.free(kps);

    const uvs = try allocator.alloc(Vec2, kps.len);
    errdefer allocator.free(uvs);
    const cols = try allocator.alloc(Color, kps.len);
    errdefer allocator.free(cols);

    for (kps, 0..) |kp, ki| {
        uvs[ki] = Vec2.init(kp.x, kp.y);
        const px: usize = @intFromFloat(std.math.clamp(@round(kp.x), 0.0, @as(f64, @floatFromInt(img.width - 1))));
        const py: usize = @intFromFloat(std.math.clamp(@round(kp.y), 0.0, @as(f64, @floatFromInt(img.height - 1))));
        const rgb_idx = (py * img.width + px) * 3;
        cols[ki] = .{
            .r = img.rgb[rgb_idx + 0],
            .g = img.rgb[rgb_idx + 1],
            .b = img.rgb[rgb_idx + 2],
        };
    }

    out_kps.* = kps;
    out_uvs.* = uvs;
    out_cols.* = cols;
}

fn detectImageFeatures(
    allocator: std.mem.Allocator,
    img: dataset.ImageData,
    max_keypoints: usize,
    out_kps: *[]features.Keypoint,
    out_uvs: *[]Vec2,
    out_cols: *[]Color,
) !void {
    const kps = try features.detectAndCompute(
        allocator,
        .{ .width = img.width, .height = img.height, .gray = img.gray },
        max_keypoints,
    );
    try packImageFeaturesFromKps(allocator, img, kps, out_kps, out_uvs, out_cols);
}

fn finishGpuImageFeatures(
    allocator: std.mem.Allocator,
    img: dataset.ImageData,
    harris_opt: ?gpu.GpuContext.HarrisGpuOutput,
    max_keypoints: usize,
    tile_cfg: features.TileConfig,
    out_kps: *[]features.Keypoint,
    out_uvs: *[]Vec2,
    out_cols: *[]Color,
) !void {
    const kps = if (harris_opt) |hm|
        try features.detectFromScores(allocator, img.width, img.height, hm.blurred, hm.score, max_keypoints, tile_cfg)
    else
        try allocator.alloc(features.Keypoint, 0);
    try packImageFeaturesFromKps(allocator, img, kps, out_kps, out_uvs, out_cols);
}

fn finishGpuDetectWorker(
    allocator: std.mem.Allocator,
    images: []const dataset.ImageData,
    harris_maps: []const ?gpu.GpuContext.HarrisGpuOutput,
    max_keypoints: usize,
    tile_cfg: features.TileConfig,
    all_kps: [][]features.Keypoint,
    all_uvs: [][]Vec2,
    all_cols: [][]Color,
    next_cam: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const ci = next_cam.fetchAdd(1, .monotonic);
        if (ci >= images.len) return;
        finishGpuImageFeatures(allocator, images[ci], harris_maps[ci], max_keypoints, tile_cfg, &all_kps[ci], &all_uvs[ci], &all_cols[ci]) catch {
            failed.store(true, .release);
            return;
        };
    }
}

fn detectWorker(
    allocator: std.mem.Allocator,
    images: []const dataset.ImageData,
    max_keypoints: usize,
    all_kps: [][]features.Keypoint,
    all_uvs: [][]Vec2,
    all_cols: [][]Color,
    next_cam: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const ci = next_cam.fetchAdd(1, .monotonic);
        if (ci >= images.len) return;
        detectImageFeatures(allocator, images[ci], max_keypoints, &all_kps[ci], &all_uvs[ci], &all_cols[ci]) catch {
            failed.store(true, .release);
            return;
        };
    }
}

fn verifyPair(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_kps: []const []const features.Keypoint,
    all_uvs: []const []const Vec2,
    i: usize,
    j: usize,
    config: SfmConfig,
) !?PairwiseVerified {
    const raw_matches = try features.matchDescriptors(
        allocator,
        all_kps[i],
        all_kps[j],
        config.match_ratio,
        config.max_hamming,
    );
    defer allocator.free(raw_matches);
    return verifyPairFromMatches(allocator, k, all_uvs, i, j, raw_matches, config);
}

fn verifyPairFromMatches(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_uvs: []const []const Vec2,
    i: usize,
    j: usize,
    raw_matches: []const features.Match,
    config: SfmConfig,
) !?PairwiseVerified {
    if (raw_matches.len < 16) return null;

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
    )) orelse return null;
    defer er.deinit(allocator);

    if (er.inliers.len < 16) return null;

    const inlier_matches = try allocator.alloc(features.Match, er.inliers.len);
    for (er.inliers, 0..) |in_idx, out_idx| {
        inlier_matches[out_idx] = raw_matches[in_idx];
    }

    return .{
        .cam1 = i,
        .cam2 = j,
        .e = er.e,
        .matches = inlier_matches,
    };
}

fn verifyPairWorker(
    allocator: std.mem.Allocator,
    k: Mat3,
    all_kps: []const []const features.Keypoint,
    all_uvs: []const []const Vec2,
    pair_indices: []const [2]usize,
    gpu_raw_matches: ?[]const []const features.Match,
    pair_slots: []?PairwiseVerified,
    config: SfmConfig,
    next_pair: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const idx = next_pair.fetchAdd(1, .monotonic);
        if (idx >= pair_indices.len) return;
        const pair = pair_indices[idx];
        pair_slots[idx] = (if (gpu_raw_matches) |batch|
            verifyPairFromMatches(allocator, k, all_uvs, pair[0], pair[1], batch[idx], config)
        else
            verifyPair(allocator, k, all_kps, all_uvs, pair[0], pair[1], config)) catch {
            failed.store(true, .release);
            return;
        };
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

test "Guided epipolar densification recovers additional 3D points along epipolar lines" {
    const gpa = std.testing.allocator;
    const w: u32 = 320;
    const h: u32 = 240;
    const k = Mat3.fromRows(
        .{ 400, 0, 160 },
        .{ 0, 400, 120 },
        .{ 0, 0, 1 },
    );

    const poses = [_]CameraPose{
        CameraPose.identity,
        .{ .r = lie.so3Exp(Vec3.init(0.0, 0.15, 0.0)), .t = Vec3.init(-0.8, 0.0, 0.1) },
        .{ .r = lie.so3Exp(Vec3.init(0.0, 0.30, 0.0)), .t = Vec3.init(-1.6, 0.0, 0.25) },
    };
    const registered = [_]bool{ true, true, true };

    var images: [3]dataset.ImageData = undefined;
    for (0..3) |ci| {
        const gray = try gpa.alloc(u8, w * h);
        const rgb = try gpa.alloc(u8, w * h * 3);
        @memset(gray, 128);
        @memset(rgb, 128);
        images[ci] = .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
    }
    defer for (&images) |*img| img.deinit(gpa);

    // Paint distinctive textured 7x7 patches around projected 3D points in all 3 images.
    const pts_gt = [_]Vec3{
        Vec3.init(-0.4, -0.2, 5.0),
        Vec3.init(0.3, 0.2, 5.5),
        Vec3.init(0.0, -0.3, 6.0),
    };

    var kps_storage: [3][3]features.Keypoint = undefined;
    var uvs_storage: [3][3]Vec2 = undefined;
    var kps_slices: [3][]const features.Keypoint = undefined;
    var uvs_slices: [3][]const Vec2 = undefined;

    for (pts_gt, 0..) |pt, pi| {
        for (0..3) |ci| {
            const uv = lie.projectPoint(k, poses[ci], pt).?;
            uvs_storage[ci][pi] = uv;
            kps_storage[ci][pi] = .{
                .x = uv.x,
                .y = uv.y,
                .score = 1000.0,
                .angle = 0.0,
                .desc = .{ @as(u64, pi) + 1, 0, 0, 0 },
            };
            const cx: isize = @intFromFloat(@round(uv.x));
            const cy: isize = @intFromFloat(@round(uv.y));
            var dy: isize = -3;
            while (dy <= 3) : (dy += 1) {
                var dx: isize = -3;
                while (dx <= 3) : (dx += 1) {
                    const px: usize = @intCast(cx + dx);
                    const py: usize = @intCast(cy + dy);
                    const val: u8 = @intCast(80 + @as(isize, @intCast(pi)) * 20 + dx * 7 + dy * 11);
                    images[ci].gray[py * w + px] = val;
                }
            }
        }
    }
    for (0..3) |ci| {
        kps_slices[ci] = &kps_storage[ci];
        uvs_slices[ci] = &uvs_storage[ci];
    }

    var valid_tracks: std.ArrayList([]TrackObs) = .empty;
    defer {
        for (valid_tracks.items) |tr| gpa.free(tr);
        valid_tracks.deinit(gpa);
    }
    var track_pt_idx: std.ArrayList(?usize) = .empty;
    defer track_pt_idx.deinit(gpa);
    var points3d: std.ArrayList(Vec3) = .empty;
    defer points3d.deinit(gpa);
    var point_colors: std.ArrayList(Color) = .empty;
    defer point_colors.deinit(gpa);

    const added = try densifyViaEpipolarGuidedMatching(
        gpa,
        k,
        &poses,
        &registered,
        &images,
        &kps_slices,
        &uvs_slices,
        null,
        &valid_tracks,
        &track_pt_idx,
        &points3d,
        &point_colors,
        .{},
        1,
    );

    try std.testing.expectEqual(@as(usize, 3), added);
    try std.testing.expectEqual(@as(usize, 3), points3d.items.len);
}

test "runIncrementalSfm with use_gpu matches CPU reconstruction" {
    const gpa = std.testing.allocator;
    const w: u32 = 320;
    const h: u32 = 240;
    const k = Mat3.fromRows(
        .{ 320, 0, 160 },
        .{ 0, 320, 120 },
        .{ 0, 0, 1 },
    );

    const n_cams: usize = 3;
    const cam_poses = [_]CameraPose{
        CameraPose.identity,
        .{ .r = lie.so3Exp(Vec3.init(0.0, -0.06, 0.0)), .t = Vec3.init(-0.35, 0.0, 0.02) },
        .{ .r = lie.so3Exp(Vec3.init(0.0, -0.12, 0.0)), .t = Vec3.init(-0.70, 0.0, 0.05) },
    };

    var prng = std.Random.DefaultPrng.init(101);
    const rng = prng.random();
    var world_pts: [90]Vec3 = undefined;
    var world_freq_x: [90]f64 = undefined;
    var world_freq_y: [90]f64 = undefined;
    for (&world_pts, &world_freq_x, &world_freq_y) |*pt, *fx, *fy| {
        pt.* = Vec3.init(
            rng.float(f64) * 3.0 - 1.5,
            rng.float(f64) * 2.0 - 1.0,
            5.0 + rng.float(f64) * 2.5,
        );
        fx.* = 0.25 + rng.float(f64) * 0.45;
        fy.* = 0.25 + rng.float(f64) * 0.45;
    }

    var images: [n_cams]dataset.ImageData = undefined;
    for (0..n_cams) |ci| {
        const gray = try gpa.alloc(u8, w * h);
        const rgb = try gpa.alloc(u8, w * h * 3);
        @memset(gray, 128);
        @memset(rgb, 128);
        for (world_pts, world_freq_x, world_freq_y, 0..) |pt, fx, fy, pi| {
            const uv = lie.projectPoint(k, cam_poses[ci], pt) orelse continue;
            if (uv.x < 12 or uv.x >= @as(f64, @floatFromInt(w - 12)) or uv.y < 12 or uv.y >= @as(f64, @floatFromInt(h - 12))) continue;
            const cx: isize = @intFromFloat(@round(uv.x));
            const cy: isize = @intFromFloat(@round(uv.y));
            const phase = @as(f64, @floatFromInt(pi)) * 0.7;
            var dy: isize = -6;
            while (dy <= 6) : (dy += 1) {
                var dx: isize = -6;
                while (dx <= 6) : (dx += 1) {
                    const idx = @as(usize, @intCast(cy + dy)) * w + @as(usize, @intCast(cx + dx));
                    const dxf: f64 = @floatFromInt(dx);
                    const dyf: f64 = @floatFromInt(dy);
                    const env = @exp(-(dxf * dxf + dyf * dyf) * 0.06);
                    const wave = 95.0 * env * @sin(dxf * fx + phase) * @cos(dyf * fy - phase);
                    const v: u8 = @intFromFloat(std.math.clamp(128.0 + wave, 0.0, 255.0));
                    gray[idx] = v;
                    rgb[idx * 3 + 0] = v;
                    rgb[idx * 3 + 1] = v;
                    rgb[idx * 3 + 2] = v;
                }
            }
        }
        images[ci] = .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
    }
    defer for (&images) |*img| img.deinit(gpa);

    const cfg_cpu: SfmConfig = .{
        .max_keypoints = 600,
        .num_threads = 1,
        .use_gpu = false,
        .enable_guided_densification = false,
        .ransac_iters = 200,
    };
    var cfg_gpu = cfg_cpu;
    cfg_gpu.use_gpu = true;

    var res_cpu = try runIncrementalSfm(gpa, k, &images, cfg_cpu);
    defer res_cpu.deinit(gpa);
    var res_gpu = try runIncrementalSfm(gpa, k, &images, cfg_gpu);
    defer res_gpu.deinit(gpa);

    try std.testing.expectEqual(@as(usize, n_cams), res_cpu.registered_count);
    try std.testing.expectEqual(@as(usize, n_cams), res_gpu.registered_count);
    try std.testing.expect(res_gpu.points.len >= 15);
    try std.testing.expect(res_gpu.mean_reproj_err < 2.0);
}
