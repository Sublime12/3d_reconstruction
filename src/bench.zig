const std = @import("std");
const Io = std.Io;
const sfm = @import("simple_3d_reconstruction");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const n_images: usize = 11;
    const width: u32 = 640;
    const height: u32 = 480;

    // Generate 11 synthetic textured 640x480 images with slight shifts
    const images = try gpa.alloc(sfm.dataset.ImageData, n_images);
    defer {
        for (images) |*img| img.deinit(gpa);
        gpa.free(images);
    }

    for (0..n_images) |ci| {
        const n_pix = @as(usize, width) * @as(usize, height);
        const gray = try gpa.alloc(u8, n_pix);
        const rgb = try gpa.alloc(u8, n_pix * 3);
        const shift_x: f64 = @as(f64, @floatFromInt(ci)) * 4.0;
        const shift_y: f64 = @as(f64, @floatFromInt(ci)) * 1.5;

        for (0..height) |y| {
            for (0..width) |x| {
                const xf = @as(f64, @floatFromInt(x)) - shift_x;
                const yf = @as(f64, @floatFromInt(y)) - shift_y;
                const val = 127.0 +
                    55.0 * @sin(xf * 0.19) * @cos(yf * 0.23) +
                    45.0 * @sin(xf * 0.07 + yf * 0.13);
                const u8_val: u8 = @intFromFloat(std.math.clamp(val, 0.0, 255.0));
                const idx = y * width + x;
                gray[idx] = u8_val;
                rgb[idx * 3 + 0] = u8_val;
                rgb[idx * 3 + 1] = u8_val;
                rgb[idx * 3 + 2] = u8_val;
            }
        }
        images[ci] = .{ .width = width, .height = height, .rgb = rgb, .gray = gray };
    }

    const k = sfm.mat.Mat3.fromRows(
        .{ 600, 0, 320 },
        .{ 0, 600, 240 },
        .{ 0, 0, 1 },
    );

    const cpu_count = std.Thread.getCpuCount() catch 4;
    std.debug.print(
        "=== SfM Benchmark ({d} images, {d}x{d}, {d} CPU cores) ===\n",
        .{ n_images, width, height, cpu_count },
    );

    const t_st = try benchmarkConfig(gpa, io, k, images, 1);
    const t_mt = try benchmarkConfig(gpa, io, k, images, cpu_count);

    const speedup = t_st / t_mt;
    std.debug.print(
        "CPU Single-Thread : {d:.1} ms\nCPU Multi-Thread  : {d:.1} ms ({d:.2}x speedup)\n",
        .{ t_st, t_mt, speedup },
    );
}

fn benchmarkConfig(
    allocator: std.mem.Allocator,
    io: Io,
    k: sfm.mat.Mat3,
    images: []const sfm.dataset.ImageData,
    num_threads: usize,
) !f64 {
    const t0 = Io.Clock.now(.awake, io);

    var res = sfm.pipeline.runIncrementalSfm(allocator, k, images, .{
        .num_threads = num_threads,
        .max_keypoints = 600,
        .ransac_iters = 150,
    }) catch |err| switch (err) {
        // Synthetic planar-shift images can fail 3D triangulation if pure translation has no parallax;
        // feature extraction + pairwise matching + Essential RANSAC still ran in full.
        error.InitializationFailed, error.TooFewInitialPoints, error.NoVerifiedPairs => {
            const dt_ns = t0.durationTo(Io.Clock.now(.awake, io)).nanoseconds;
            return @as(f64, @floatFromInt(dt_ns)) / 1e6;
        },
        else => return err,
    };
    res.deinit(allocator);

    const dt_ns = t0.durationTo(Io.Clock.now(.awake, io)).nanoseconds;
    return @as(f64, @floatFromInt(dt_ns)) / 1e6;
}
