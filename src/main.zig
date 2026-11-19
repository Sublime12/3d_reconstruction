const std = @import("std");
const Io = std.Io;
const sfm = @import("3d_reconstruction");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);

    const opts = try sfm.parseCliArgs(args);
    if (opts.show_help) {
        std.debug.print(
            "Usage: 3d_reconstruction [--data <dir>] [--out <ply>] [--max-dim <px>] [--threads <n>] [--gpu] [--nn-model <onnx>] [--npu] [--nn-device <cpu|npu|gpu|auto>] [--nerf] [--nerf-steps <n>] [--verify-views <dir>]\n",
            .{},
        );
        return;
    }
    const data_dir = opts.data_dir;
    const out_path = opts.out_path;
    const max_dim = opts.max_dim;
    const num_threads = opts.num_threads;
    const use_gpu = opts.use_gpu;

    const t_pipeline0 = Io.Clock.now(.awake, io);
    const k_path = try std.fmt.allocPrint(arena, "{s}/K.txt", .{data_dir});
    const k_text = Io.Dir.cwd().readFileAlloc(io, k_path, arena, .limited(64 * 1024)) catch |err| {
        std.log.err("Failed to read intrinsics file '{s}': {t}", .{ k_path, err });
        return err;
    };
    var k = try sfm.dataset.parseIntrinsicsText(k_text);

    // Discover and sort image files in data_dir
    var dir = try Io.Dir.cwd().openDir(io, data_dir, .{ .iterate = true });
    defer dir.close(io);

    var img_names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (isImageFilename(entry.name)) {
            try img_names.append(arena, try arena.dupe(u8, entry.name));
        }
    }
    if (img_names.items.len < 2) {
        std.log.err("Need at least 2 images in '{s}', found {d}", .{ data_dir, img_names.items.len });
        return error.NotEnoughImages;
    }

    std.mem.sort([]const u8, img_names.items, {}, struct {
        fn cmp(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.cmp);

    const images = try gpa.alloc(sfm.dataset.ImageData, img_names.items.len);
    var loaded_count: usize = 0;
    defer {
        for (images[0..loaded_count]) |*img| img.deinit(gpa);
        gpa.free(images);
    }

    var k_scaled = false;
    for (img_names.items, 0..) |name, idx| {
        const full_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ data_dir, name });
        var img = try sfm.dataset.loadImage(gpa, full_path);
        const scale = try sfm.dataset.downscaleIfNeeded(gpa, &img, max_dim);
        if (!k_scaled and scale != 1.0) {
            k.m[0][0] *= scale;
            k.m[0][1] *= scale;
            k.m[0][2] *= scale;
            k.m[1][1] *= scale;
            k.m[1][2] *= scale;
            k_scaled = true;
        }
        images[idx] = img;
        loaded_count += 1;
        std.log.info("Loaded {s} ({d}x{d})", .{ name, img.width, img.height });
    }

    const load_ms = elapsedMs(io, t_pipeline0);

    std.log.info("Running Incremental SfM on {d} images (threads={d}, gpu={any})...", .{ images.len, num_threads, use_gpu });
    const t_start = Io.Clock.now(.awake, io);
    var sfm_timings: sfm.pipeline.SfmTimings = .{};
    var result = try sfm.pipeline.runIncrementalSfm(gpa, k, images, .{
        .num_threads = num_threads,
        .use_gpu = use_gpu,
        .io = io,
        .timings = &sfm_timings,
    });
    defer result.deinit(gpa);
    const elapsed_ms = elapsedMs(io, t_start);

    var reg_poses: std.ArrayList(sfm.lie.CameraPose) = .empty;
    defer reg_poses.deinit(gpa);
    var cam_metas: std.ArrayList(sfm.io_ply.CameraMeta) = .empty;
    defer cam_metas.deinit(gpa);
    for (result.poses, result.registered, img_names.items) |p, is_reg, name| {
        if (is_reg) {
            try reg_poses.append(gpa, p);
            const rel_img = try std.fmt.allocPrint(arena, "{s}/{s}", .{ data_dir, name });
            try cam_metas.append(gpa, .{ .pose = p, .image_path = rel_img });
        }
    }

    var delaunay_ms: f64 = 0;
    var bridge_ms: f64 = 0;
    var loop_ms: f64 = 0;
    var nn_ms: f64 = 0;
    var nn_post_loop_ms: f64 = 0;
    var nn_timings: sfm.neural_depth.NeuralHoleTimings = .{};
    var texture_ms: f64 = 0;

    var nerf_timings: sfm.nerf.NerfTimings = .{};
    var nerf_mc_ms: f64 = 0;
    var nerf_model_opt: ?sfm.nerf.NerfModel = null;
    defer if (nerf_model_opt) |*m| m.deinit(gpa);
    var nerf_gpu_ctx_opt: ?sfm.nerf_gpu.NerfGpuContext = null;
    defer if (nerf_gpu_ctx_opt) |*gc| gc.deinit();

    const nerf_cfg: sfm.nerf.NerfConfig = if (use_gpu) .{
        .num_levels = 12,
        .table_log2 = 18,
        .n_min = 32.0,
        .n_max = 3072.0,
        .geo_hidden = 32,
        .geo_feat_dim = 15,
        .col_hidden = 32,
        .num_steps = opts.nerf_steps,
        .batch_rays = 2048,
        .max_ray_samples = 72,
        .occ_grid_res = 64,
        .occ_update_interval = 16,
        .occ_decay = 0.75,
        .occ_threshold = 0.03,
        .depth_loss_weight = 0.05,
        .distortion_weight = 0.0005,
        .num_threads = num_threads,
        .io = io,
    } else .{
        .num_levels = 12,
        .table_log2 = 17,
        .n_min = 32.0,
        .n_max = 2048.0,
        .occ_grid_res = 48,
        .occ_update_interval = 32,
        .occ_decay = 0.75,
        .occ_threshold = 0.03,
        .depth_loss_weight = 0.05,
        .distortion_weight = 0.0005,
        .max_ray_samples = 72,
        .num_steps = opts.nerf_steps,
        .num_threads = num_threads,
        .io = io,
    };

    var colored_mesh: sfm.mesh.Mesh = undefined;
    var has_colored_mesh = false;
    defer if (has_colored_mesh) colored_mesh.deinit(gpa);

    if (opts.use_nerf) {
        if (use_gpu) {
            nerf_gpu_ctx_opt = try sfm.nerf_gpu.NerfGpuContext.initStandalone();
            std.log.info(
                "Training Instant-NGP NeRF on GPU ({d} steps, batch={d}, levels={d})...",
                .{ nerf_cfg.num_steps, nerf_cfg.batch_rays, nerf_cfg.num_levels },
            );
            nerf_model_opt = try sfm.nerf_gpu.trainOnDatasetGpu(
                gpa,
                &nerf_gpu_ctx_opt.?,
                k,
                images,
                result.poses,
                result.registered,
                result.points,
                nerf_cfg,
                &nerf_timings,
            );
        } else {
            std.log.info("Training Instant-NGP NeRF on CPU ({d} steps, threads={d})...", .{ nerf_cfg.num_steps, num_threads });
            nerf_model_opt = try sfm.nerf.trainOnDataset(gpa, k, images, result.poses, result.registered, result.points, nerf_cfg, &nerf_timings);
        }
        std.log.info(
            "NeRF trained in {d:.1} ms (setup={d:.1}, final_loss={d:.5})",
            .{ nerf_timings.train_ms, nerf_timings.setup_ms, nerf_timings.final_loss },
        );

        const t_mc0 = Io.Clock.now(.awake, io);
        std.log.info("Extracting NeRF isosurface mesh (Marching Cubes, threads={d})...", .{num_threads});
        colored_mesh = try nerf_model_opt.?.extractMesh(
            gpa,
            result.points,
            result.poses,
            result.registered,
            .{
                .max_res = 320,
                .num_threads = num_threads,
                .k = k,
                .img_width = if (images.len > 0) images[0].width else 0,
                .img_height = if (images.len > 0) images[0].height else 0,
            },
        );
        has_colored_mesh = true;
        nerf_mc_ms = elapsedMs(io, t_mc0);
    } else {
        std.log.info("Building colored surface mesh from {d} sparse points...", .{result.points.len});
        const t_delaunay0 = Io.Clock.now(.awake, io);
        colored_mesh = try sfm.mesh.buildColoredMesh(gpa, result.points, result.colors, reg_poses.items, .{ .fill_holes = false });
        has_colored_mesh = true;
        delaunay_ms = elapsedMs(io, t_delaunay0);

        const t_bridge0 = Io.Clock.now(.awake, io);
        try sfm.mesh.completeMeshWithPhotoVerify(
            gpa,
            &colored_mesh,
            k,
            images,
            result.poses,
            result.registered,
            reg_poses.items,
            .{},
        );
        bridge_ms = elapsedMs(io, t_bridge0);

        const t_loop0 = Io.Clock.now(.awake, io);
        try sfm.mesh.fillMeshInteriorHoles(gpa, &colored_mesh, reg_poses.items, .{});
        loop_ms = elapsedMs(io, t_loop0);

        if (opts.nn_model_path) |nn_path| {
            std.log.info("Running 3D Transformer hole completion with '{s}' (device={s})...", .{ nn_path, opts.nn_device.label() });
            const t_nn0 = Io.Clock.now(.awake, io);
            const added_nn = try sfm.neural_depth.completeMeshHolesWithTransformer(
                gpa,
                &colored_mesh,
                k,
                images,
                result.poses,
                result.registered,
                .{
                    .model_path = nn_path,
                    .num_threads = num_threads,
                    .device = opts.nn_device,
                    .io = io,
                    .timings = &nn_timings,
                },
            );
            nn_ms = elapsedMs(io, t_nn0);
            std.log.info("Transformer added {d} hole vertices in {d:.1} ms -> {d} vertices, {d} triangles", .{
                added_nn,
                nn_ms,
                colored_mesh.points.len,
                colored_mesh.faces.len,
            });
            const t_post_loop0 = Io.Clock.now(.awake, io);
            try sfm.mesh.fillMeshInteriorHoles(gpa, &colored_mesh, reg_poses.items, .{});
            nn_post_loop_ms = elapsedMs(io, t_post_loop0);
        }

        std.log.info("Subdividing mesh and projecting photos onto {d} triangles...", .{colored_mesh.faces.len});
        const t_tex0 = Io.Clock.now(.awake, io);
        try sfm.mesh.subdivideAndProjectTexture(
            gpa,
            &colored_mesh,
            k,
            images,
            result.poses,
            result.registered,
            2,
        );
        texture_ms = elapsedMs(io, t_tex0);
    }

    const t_ply0 = Io.Clock.now(.awake, io);
    const ply_bytes = try sfm.io_ply.formatPlyMeshWithMeta(
        gpa,
        colored_mesh.points,
        colored_mesh.colors,
        colored_mesh.faces,
        reg_poses.items,
        .{
            .k = k,
            .img_w = images[0].width,
            .img_h = images[0].height,
            .cameras = cam_metas.items,
        },
    );
    defer gpa.free(ply_bytes);

    if (std.fs.path.dirname(out_path)) |parent| {
        try Io.Dir.cwd().createDirPath(io, parent);
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = ply_bytes });
    const ply_ms = elapsedMs(io, t_ply0);

    const total_elapsed_ms = elapsedMs(io, t_start);
    const wall_total_ms = elapsedMs(io, t_pipeline0);
    std.log.info(
        "Done in {d:.1} ms (SfM {d:.1} ms)! Registered {d}/{d} cameras, {d} mesh vertices, {d} triangles, mean reproj error = {d:.3} px -> {s}",
        .{ total_elapsed_ms, elapsed_ms, result.registered_count, images.len, colored_mesh.points.len, colored_mesh.faces.len, result.mean_reproj_err, out_path },
    );
    std.log.info(
        "Step timings (ms): load={d:.1} | sfm={d:.1} (gpu_init={d:.1}, detect={d:.1}, match_verify={d:.1}, inc_ba={d:.1}, guided={d:.1}, final_ba={d:.1}) | mesh(delaunay={d:.1}, bridge={d:.1}, loops={d:.1}) | nn={d:.1} (init={d:.1}, prerender={d:.1}, infer={d:.1}, stitch={d:.1}, post_loops={d:.1}) | nerf(setup={d:.1}, train={d:.1}, mc={d:.1}) | texture={d:.1} | ply={d:.1} | total={d:.1}",
        .{
            load_ms,
            elapsed_ms,
            sfm_timings.gpu_init_ms,
            sfm_timings.feature_extract_ms,
            sfm_timings.match_verify_ms,
            sfm_timings.incremental_ba_ms,
            sfm_timings.guided_densify_ms,
            sfm_timings.final_ba_ms,
            delaunay_ms,
            bridge_ms,
            loop_ms,
            nn_ms,
            nn_timings.init_ms,
            nn_timings.prerender_ms,
            nn_timings.infer_ms,
            nn_timings.stitch_ms,
            nn_post_loop_ms,
            nerf_timings.setup_ms,
            nerf_timings.train_ms,
            nerf_mc_ms,
            texture_ms,
            ply_ms,
            wall_total_ms,
        },
    );

    if (opts.verify_views_dir) |verify_dir| {
        const t_ver0 = Io.Clock.now(.awake, io);
        if (nerf_model_opt) |*nerf_model| {
            const gctx_ptr: ?*const sfm.nerf_gpu.NerfGpuContext = if (nerf_gpu_ctx_opt) |*gc| gc else null;
            try runVerifyViewsNerf(gpa, arena, io, verify_dir, k, images, &result, nerf_model, nerf_cfg, gctx_ptr);
        } else {
            try runVerifyViews(gpa, arena, io, verify_dir, k, images, reg_poses.items, &result, &colored_mesh);
        }
        std.log.info("View verification completed in {d:.1} ms", .{elapsedMs(io, t_ver0)});
    }
}

fn elapsedMs(io: Io, t0: Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
}

fn runVerifyViews(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    verify_dir: []const u8,
    k: sfm.mat.Mat3,
    images: []const sfm.dataset.ImageData,
    reg_poses: []const sfm.lie.CameraPose,
    result: *const sfm.pipeline.SfmResult,
    colored_mesh: *const sfm.mesh.Mesh,
) !void {
    try Io.Dir.cwd().createDirPath(io, verify_dir);

    var cov_sum: f64 = 0.0;
    var mae_sum: f64 = 0.0;
    var rmse_sum: f64 = 0.0;
    var psnr_sum: f64 = 0.0;
    var ssim_sum: f64 = 0.0;
    var n_eval: usize = 0;

    for (images, 0..) |img, cam_idx| {
        if (!result.registered[cam_idx]) continue;

        var ren = try sfm.render.renderMeshView(
            gpa,
            img.width,
            img.height,
            k,
            result.poses[cam_idx],
            colored_mesh.points,
            colored_mesh.colors,
            colored_mesh.faces,
        );
        defer ren.deinit(gpa);

        const metrics = sfm.render.computeViewMetrics(ren, img);
        cov_sum += metrics.coverage_ratio;
        mae_sum += metrics.mae;
        rmse_sum += metrics.rmse;
        psnr_sum += metrics.psnr_db;
        ssim_sum += metrics.ssim;
        n_eval += 1;

        std.log.info(
            "Verify cam {d:02}: coverage={d:.1}% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ cam_idx, metrics.coverage_ratio * 100.0, metrics.mae, metrics.rmse, metrics.psnr_db, metrics.ssim },
        );

        if (cam_idx == 0 or cam_idx == 3 or cam_idx == 5 or cam_idx == 8 or cam_idx == images.len - 1) {
            const virt_path = try std.fmt.allocPrint(arena, "{s}/cam_{d:02}_virtual.ppm", .{ verify_dir, cam_idx });
            const real_path = try std.fmt.allocPrint(arena, "{s}/cam_{d:02}_real.ppm", .{ verify_dir, cam_idx });
            try sfm.render.writePpm(gpa, io, virt_path, ren.width, ren.height, ren.rgb);
            try sfm.render.writePpm(gpa, io, real_path, img.width, img.height, img.rgb);
        }
    }

    if (n_eval > 0) {
        const nf: f64 = @floatFromInt(n_eval);
        std.log.info(
            "Verify mean ({d} cams): coverage={d:.1}% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ n_eval, (cov_sum / nf) * 100.0, mae_sum / nf, rmse_sum / nf, psnr_sum / nf, ssim_sum / nf },
        );
    }

    // Held-out texture test on middle camera: build same mesh without projecting holdout camera's photo
    const holdout_idx = images.len / 2;
    if (images.len >= 3 and result.registered[holdout_idx]) {
        const reg_holdout = try gpa.dupe(bool, result.registered);
        defer gpa.free(reg_holdout);
        reg_holdout[holdout_idx] = false;

        var holdout_mesh = try sfm.mesh.buildColoredMesh(gpa, result.points, result.colors, reg_poses, .{ .fill_holes = false });
        defer holdout_mesh.deinit(gpa);
        try sfm.mesh.completeMeshWithPhotoVerify(
            gpa,
            &holdout_mesh,
            k,
            images,
            result.poses,
            reg_holdout,
            reg_poses,
            .{},
        );
        try sfm.mesh.fillMeshInteriorHoles(gpa, &holdout_mesh, reg_poses, .{});
        try sfm.mesh.subdivideAndProjectTexture(gpa, &holdout_mesh, k, images, result.poses, reg_holdout, 2);

        const target_img = images[holdout_idx];
        var ren_holdout = try sfm.render.renderMeshView(
            gpa,
            target_img.width,
            target_img.height,
            k,
            result.poses[holdout_idx],
            holdout_mesh.points,
            holdout_mesh.colors,
            holdout_mesh.faces,
        );
        defer ren_holdout.deinit(gpa);

        const hm = sfm.render.computeViewMetrics(ren_holdout, target_img);
        std.log.info(
            "Holdout cam {d:02} (unseen texture): coverage={d:.1}% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ holdout_idx, hm.coverage_ratio * 100.0, hm.mae, hm.rmse, hm.psnr_db, hm.ssim },
        );
        const holdout_path = try std.fmt.allocPrint(arena, "{s}/holdout_{d:02}_virtual.ppm", .{ verify_dir, holdout_idx });
        try sfm.render.writePpm(gpa, io, holdout_path, ren_holdout.width, ren_holdout.height, ren_holdout.rgb);
    }

    // Interpolated novel viewpoint between camera 4 and camera 5
    if (images.len > 5 and result.registered[4] and result.registered[5]) {
        const novel_pose = sfm.render.interpolatePose(result.poses[4], result.poses[5], 0.5);
        const ref_img = images[4];
        var ren_novel = try sfm.render.renderMeshView(
            gpa,
            ref_img.width,
            ref_img.height,
            k,
            novel_pose,
            colored_mesh.points,
            colored_mesh.colors,
            colored_mesh.faces,
        );
        defer ren_novel.deinit(gpa);
        const novel_path = try std.fmt.allocPrint(arena, "{s}/novel_mid45_virtual.ppm", .{verify_dir});
        try sfm.render.writePpm(gpa, io, novel_path, ren_novel.width, ren_novel.height, ren_novel.rgb);
    }
}

fn runVerifyViewsNerf(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    verify_dir: []const u8,
    k: sfm.mat.Mat3,
    images: []const sfm.dataset.ImageData,
    result: *const sfm.pipeline.SfmResult,
    model: *const sfm.nerf.NerfModel,
    nerf_cfg: sfm.nerf.NerfConfig,
    gpu_ctx: ?*const sfm.nerf_gpu.NerfGpuContext,
) !void {
    try Io.Dir.cwd().createDirPath(io, verify_dir);

    var gpu_session_opt: ?sfm.nerf_gpu.NerfGpuSession = if (gpu_ctx) |gc|
        try sfm.nerf_gpu.NerfGpuSession.initFromModel(gc, model, 1)
    else
        null;
    defer if (gpu_session_opt) |*s| s.deinit();

    var mae_sum: f64 = 0.0;
    var rmse_sum: f64 = 0.0;
    var psnr_sum: f64 = 0.0;
    var ssim_sum: f64 = 0.0;
    var n_eval: usize = 0;

    for (images, 0..) |img, cam_idx| {
        if (!result.registered[cam_idx]) continue;

        var ren = if (gpu_session_opt) |*s|
            try s.renderViewGpu(gpa, model, img.width, img.height, k, result.poses[cam_idx], true)
        else
            try model.renderView(gpa, img.width, img.height, k, result.poses[cam_idx], true);
        defer ren.deinit(gpa);

        const metrics = sfm.render.computeViewMetrics(ren, img);
        mae_sum += metrics.mae;
        rmse_sum += metrics.rmse;
        psnr_sum += metrics.psnr_db;
        ssim_sum += metrics.ssim;
        n_eval += 1;

        std.log.info(
            "Verify NeRF cam {d:02}: coverage={d:.1}% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ cam_idx, metrics.coverage_ratio * 100.0, metrics.mae, metrics.rmse, metrics.psnr_db, metrics.ssim },
        );

        if (cam_idx == 0 or cam_idx == 5 or cam_idx == images.len - 1) {
            const virt_path = try std.fmt.allocPrint(arena, "{s}/nerf_cam_{d:02}_virtual.ppm", .{ verify_dir, cam_idx });
            const real_path = try std.fmt.allocPrint(arena, "{s}/nerf_cam_{d:02}_real.ppm", .{ verify_dir, cam_idx });
            try sfm.render.writePpm(gpa, io, virt_path, ren.width, ren.height, ren.rgb);
            try sfm.render.writePpm(gpa, io, real_path, img.width, img.height, img.rgb);
        }
    }

    if (n_eval > 0) {
        const nf: f64 = @floatFromInt(n_eval);
        std.log.info(
            "Verify NeRF mean ({d} cams): coverage=100.0% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ n_eval, mae_sum / nf, rmse_sum / nf, psnr_sum / nf, ssim_sum / nf },
        );
    }

    const holdout_idx = images.len / 2;
    if (images.len >= 3 and result.registered[holdout_idx]) {
        const reg_holdout = try gpa.dupe(bool, result.registered);
        defer gpa.free(reg_holdout);
        reg_holdout[holdout_idx] = false;

        var holdout_model = if (gpu_ctx) |gc|
            try sfm.nerf_gpu.trainOnDatasetGpu(gpa, gc, k, images, result.poses, reg_holdout, result.points, nerf_cfg, null)
        else
            try sfm.nerf.trainOnDataset(gpa, k, images, result.poses, reg_holdout, result.points, nerf_cfg, null);
        defer holdout_model.deinit(gpa);

        const target_img = images[holdout_idx];
        var ren_holdout = if (gpu_ctx) |gc| blk: {
            var hs = try sfm.nerf_gpu.NerfGpuSession.initFromModel(gc, &holdout_model, 1);
            defer hs.deinit();
            break :blk try hs.renderViewGpu(gpa, &holdout_model, target_img.width, target_img.height, k, result.poses[holdout_idx], true);
        } else try holdout_model.renderView(gpa, target_img.width, target_img.height, k, result.poses[holdout_idx], true);
        defer ren_holdout.deinit(gpa);

        const hm = sfm.render.computeViewMetrics(ren_holdout, target_img);
        std.log.info(
            "Holdout NeRF cam {d:02} (unseen view): coverage=100.0% MAE={d:.2} RMSE={d:.2} PSNR={d:.2} dB SSIM={d:.4}",
            .{ holdout_idx, hm.mae, hm.rmse, hm.psnr_db, hm.ssim },
        );
        const holdout_path = try std.fmt.allocPrint(arena, "{s}/nerf_holdout_{d:02}_virtual.ppm", .{ verify_dir, holdout_idx });
        try sfm.render.writePpm(gpa, io, holdout_path, ren_holdout.width, ren_holdout.height, ren_holdout.rgb);
    }

    if (images.len > 5 and result.registered[4] and result.registered[5]) {
        const novel_pose = sfm.render.interpolatePose(result.poses[4], result.poses[5], 0.5);
        const ref_img = images[4];
        var ren_novel = if (gpu_session_opt) |*s|
            try s.renderViewGpu(gpa, model, ref_img.width, ref_img.height, k, novel_pose, true)
        else
            try model.renderView(gpa, ref_img.width, ref_img.height, k, novel_pose, true);
        defer ren_novel.deinit(gpa);
        const novel_path = try std.fmt.allocPrint(arena, "{s}/nerf_novel_mid45_virtual.ppm", .{verify_dir});
        try sfm.render.writePpm(gpa, io, novel_path, ren_novel.width, ren_novel.height, ren_novel.rgb);
    }
}

fn isImageFilename(name: []const u8) bool {
    const exts = [_][]const u8{ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" };
    for (exts) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}
