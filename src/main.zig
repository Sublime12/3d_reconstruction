const std = @import("std");
const Io = std.Io;
const sfm = @import("simple_3d_reconstruction");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);

    const opts = try sfm.parseCliArgs(args);
    if (opts.show_help) {
        std.debug.print(
            "Usage: sfm [--data <dir>] [--out <ply>] [--max-dim <px>] [--threads <n>] [--gpu] [--nn-model <onnx>] [--verify-views <dir>]\n",
            .{},
        );
        return;
    }
    const data_dir = opts.data_dir;
    const out_path = opts.out_path;
    const max_dim = opts.max_dim;
    const num_threads = opts.num_threads;
    const use_gpu = opts.use_gpu;

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

    std.log.info("Running Incremental SfM on {d} images (threads={d}, gpu={any})...", .{ images.len, num_threads, use_gpu });
    const t_start = Io.Clock.now(.awake, io);
    var result = try sfm.pipeline.runIncrementalSfm(gpa, k, images, .{ .num_threads = num_threads, .use_gpu = use_gpu });
    defer result.deinit(gpa);
    const elapsed_ns = t_start.durationTo(Io.Clock.now(.awake, io)).nanoseconds;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1e6;

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

    std.log.info("Building colored surface mesh from {d} sparse points...", .{result.points.len});
    var colored_mesh = try sfm.mesh.buildColoredMesh(gpa, result.points, result.colors, reg_poses.items, .{ .fill_holes = false });
    defer colored_mesh.deinit(gpa);

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

    try sfm.mesh.fillMeshInteriorHoles(gpa, &colored_mesh, reg_poses.items, .{});

    if (opts.nn_model_path) |nn_path| {
        std.log.info("Running 3D Transformer hole completion with '{s}'...", .{nn_path});
        const t_nn0 = Io.Clock.now(.awake, io);
        const added_nn = try sfm.neural_depth.completeMeshHolesWithTransformer(
            gpa,
            &colored_mesh,
            k,
            images,
            result.poses,
            result.registered,
            .{ .model_path = nn_path, .num_threads = num_threads },
        );
        const nn_ms = @as(f64, @floatFromInt(t_nn0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
        std.log.info("Transformer added {d} hole vertices in {d:.1} ms -> {d} vertices, {d} triangles", .{
            added_nn,
            nn_ms,
            colored_mesh.points.len,
            colored_mesh.faces.len,
        });
        try sfm.mesh.fillMeshInteriorHoles(gpa, &colored_mesh, reg_poses.items, .{});
    }

    std.log.info("Subdividing mesh and projecting photos onto {d} triangles...", .{colored_mesh.faces.len});
    try sfm.mesh.subdivideAndProjectTexture(
        gpa,
        &colored_mesh,
        k,
        images,
        result.poses,
        result.registered,
        2,
    );

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

    const total_elapsed_ns = t_start.durationTo(Io.Clock.now(.awake, io)).nanoseconds;
    const total_elapsed_ms = @as(f64, @floatFromInt(total_elapsed_ns)) / 1e6;
    std.log.info(
        "Done in {d:.1} ms (SfM {d:.1} ms)! Registered {d}/{d} cameras, {d} mesh vertices, {d} triangles, mean reproj error = {d:.3} px -> {s}",
        .{ total_elapsed_ms, elapsed_ms, result.registered_count, images.len, colored_mesh.points.len, colored_mesh.faces.len, result.mean_reproj_err, out_path },
    );

    if (opts.verify_views_dir) |verify_dir| {
        try runVerifyViews(gpa, arena, io, verify_dir, k, images, reg_poses.items, &result, &colored_mesh);
    }
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

fn isImageFilename(name: []const u8) bool {
    const exts = [_][]const u8{ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" };
    for (exts) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}
