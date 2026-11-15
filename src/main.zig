const std = @import("std");
const Io = std.Io;
const sfm = @import("simple_3d_reconstruction");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);

    var data_dir: []const u8 = "data/fountain-p11";
    var out_path: []const u8 = "output/reconstruction.ply";
    var max_dim: u32 = 1200;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--data") and i + 1 < args.len) {
            i += 1;
            data_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--out") and i + 1 < args.len) {
            i += 1;
            out_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-dim") and i + 1 < args.len) {
            i += 1;
            max_dim = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            std.debug.print(
                "Usage: sfm [--data <dir>] [--out <ply>] [--max-dim <px>]\n",
                .{},
            );
            return;
        }
    }

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

    std.log.info("Running single-threaded Incremental SfM on {d} images...", .{images.len});
    var result = try sfm.pipeline.runIncrementalSfm(gpa, k, images, .{});
    defer result.deinit(gpa);

    var reg_poses: std.ArrayList(sfm.lie.CameraPose) = .empty;
    defer reg_poses.deinit(gpa);
    for (result.poses, result.registered) |p, is_reg| {
        if (is_reg) try reg_poses.append(gpa, p);
    }

    const ply_bytes = try sfm.io_ply.formatPly(gpa, result.points, result.colors, reg_poses.items);
    defer gpa.free(ply_bytes);

    if (std.fs.path.dirname(out_path)) |parent| {
        try Io.Dir.cwd().createDirPath(io, parent);
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = ply_bytes });

    std.log.info(
        "Done! Registered {d}/{d} cameras, {d} 3D points, mean reproj error = {d:.3} px -> {s}",
        .{ result.registered_count, images.len, result.points.len, result.mean_reproj_err, out_path },
    );
}

fn isImageFilename(name: []const u8) bool {
    const exts = [_][]const u8{ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" };
    for (exts) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}
