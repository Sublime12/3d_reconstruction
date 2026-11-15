const std = @import("std");
const Io = std.Io;
const sfm = @import("simple_3d_reconstruction");

fn generateSyntheticDataset(allocator: std.mem.Allocator, n_cams: usize, w: u32, h: u32) !struct {
    k: sfm.mat.Mat3,
    images: []sfm.dataset.ImageData,
} {
    const k = sfm.mat.Mat3.fromRows(
        .{ 600, 0, @as(f64, @floatFromInt(w)) * 0.5 },
        .{ 0, 600, @as(f64, @floatFromInt(h)) * 0.5 },
        .{ 0, 0, 1 },
    );

    var prng = std.Random.DefaultPrng.init(2026);
    const rng = prng.random();
    var pts: [300]sfm.vec.Vec3 = undefined;
    var fxs: [300]f64 = undefined;
    var fys: [300]f64 = undefined;
    for (&pts, &fxs, &fys) |*p, *fx, *fy| {
        p.* = sfm.vec.Vec3.init(
            rng.float(f64) * 3.2 - 0.8,
            rng.float(f64) * 2.4 - 1.2,
            4.5 + rng.float(f64) * 2.5,
        );
        fx.* = 0.25 + rng.float(f64) * 0.45;
        fy.* = 0.25 + rng.float(f64) * 0.45;
    }

    const images = try allocator.alloc(sfm.dataset.ImageData, n_cams);
    errdefer allocator.free(images);

    for (0..n_cams) |ci| {
        const gray = try allocator.alloc(u8, w * h);
        const rgb = try allocator.alloc(u8, w * h * 3);
        @memset(gray, 128);
        @memset(rgb, 128);

        const t_step: f64 = @as(f64, @floatFromInt(ci)) - 5.0;
        const ry = t_step * 0.045;
        const r = sfm.lie.so3Exp(sfm.vec.Vec3.init(0, ry, 0));
        const tx = -(t_step * 0.27);
        const pose = sfm.lie.CameraPose{ .r = r, .t = sfm.vec.Vec3.init(tx, 0, @abs(t_step) * 0.02) };

        for (pts, fxs, fys, 0..) |pt, fx, fy, pi| {
            const uv = sfm.lie.projectPoint(k, pose, pt) orelse continue;
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

    return .{ .k = k, .images = images };
}

fn matchWorker(
    allocator: std.mem.Allocator,
    all_kps: []const []const sfm.features.Keypoint,
    pairs: []const [2]usize,
    out_matches: [][]sfm.features.Match,
    next_pair: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
) void {
    while (true) {
        if (failed.load(.acquire)) return;
        const idx = next_pair.fetchAdd(1, .monotonic);
        if (idx >= pairs.len) return;
        const p = pairs[idx];
        out_matches[idx] = sfm.features.matchDescriptors(allocator, all_kps[p[0]], all_kps[p[1]], 0.80, 72) catch {
            failed.store(true, .release);
            return;
        };
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const cpu_count = std.Thread.getCpuCount() catch 4;

    var gpu_ctx = try sfm.gpu.GpuContext.init();
    defer gpu_ctx.deinit();
    std.debug.print("WebGPU Adapter: {s}\n\n", .{gpu_ctx.adapterName()});

    // 1. Harris Corner Scoring Micro-benchmark (11 x 1200x800)
    const hw: u32 = 1200;
    const hh: u32 = 800;
    const n_imgs: usize = 11;
    const img_gray = try gpa.alloc(u8, hw * hh);
    defer gpa.free(img_gray);
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    for (img_gray, 0..) |*px, i| {
        const x = i % hw;
        const y = i / hw;
        const v = 128.0 + 70.0 * @sin(@as(f64, @floatFromInt(x)) * 0.13) * @cos(@as(f64, @floatFromInt(y)) * 0.17);
        px.* = @intFromFloat(std.math.clamp(v + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -5, 5))), 0.0, 255.0));
    }

    const t_h_cpu0 = Io.Clock.now(.awake, io);
    for (0..n_imgs) |_| {
        const b = try sfm.features.gaussianBlur5x5(gpa, img_gray, hw, hh);
        const s = try sfm.features.computeHarrisScores(gpa, b, hw, hh, 250.0);
        gpa.free(b);
        gpa.free(s);
    }
    const h_cpu_ms = @as(f64, @floatFromInt(t_h_cpu0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;

    // Warmup 1 GPU Harris dispatch then time 11 images
    {
        const w0 = try gpu_ctx.computeHarrisScores(gpa, img_gray, hw, hh, 250.0);
        gpa.free(w0.blurred);
        gpa.free(w0.score);
    }
    const t_h_gpu0 = Io.Clock.now(.awake, io);
    for (0..n_imgs) |_| {
        const res = try gpu_ctx.computeHarrisScores(gpa, img_gray, hw, hh, 250.0);
        gpa.free(res.blurred);
        gpa.free(res.score);
    }
    const h_gpu_ms = @as(f64, @floatFromInt(t_h_gpu0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;

    // 2. Descriptor Matching Micro-benchmark (55 pairs, 10,500 keypoints/cam)
    const kps_per_cam: usize = 10500;
    const all_kps = try gpa.alloc([]sfm.features.Keypoint, n_imgs);
    defer {
        for (all_kps) |kps| gpa.free(kps);
        gpa.free(all_kps);
    }
    for (all_kps, 0..) |*slot, ci| {
        slot.* = try gpa.alloc(sfm.features.Keypoint, kps_per_cam);
        for (slot.*, 0..) |*kp, ki| {
            const base_seed: u64 = (@as(u64, ki) + 1) *% 0x9e3779b97f4a7c15;
            kp.* = .{
                .x = @floatFromInt(ki % hw),
                .y = @floatFromInt(ki % hh),
                .score = 1000.0,
                .angle = 0.0,
                .desc = .{
                    base_seed ^ @as(u64, ci),
                    rng.int(u64),
                    rng.int(u64),
                    rng.int(u64),
                },
            };
        }
    }

    const n_pairs = (n_imgs * (n_imgs - 1)) / 2;
    const pairs = try gpa.alloc([2]usize, n_pairs);
    defer gpa.free(pairs);
    var pi: usize = 0;
    for (0..n_imgs) |i| {
        for (i + 1..n_imgs) |j| {
            pairs[pi] = .{ i, j };
            pi += 1;
        }
    }

    const t_m_cpu1_0 = Io.Clock.now(.awake, io);
    for (pairs) |p| {
        const ms = try sfm.features.matchDescriptors(gpa, all_kps[p[0]], all_kps[p[1]], 0.80, 72);
        gpa.free(ms);
    }
    const m_cpu1_ms = @as(f64, @floatFromInt(t_m_cpu1_0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;

    const mt_slots = try gpa.alloc([]sfm.features.Match, n_pairs);
    @memset(mt_slots, &.{});
    defer gpa.free(mt_slots);
    var next_p = std.atomic.Value(usize).init(0);
    var failed = std.atomic.Value(bool).init(false);
    const threads = try gpa.alloc(std.Thread, cpu_count);
    defer gpa.free(threads);

    const t_m_cpumt_0 = Io.Clock.now(.awake, io);
    for (threads) |*th| {
        th.* = try std.Thread.spawn(.{}, matchWorker, .{ gpa, all_kps, pairs, mt_slots, &next_p, &failed });
    }
    for (threads) |th| th.join();
    const m_cpumt_ms = @as(f64, @floatFromInt(t_m_cpumt_0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
    for (mt_slots) |ms| gpa.free(ms);

    const t_m_gpu0 = Io.Clock.now(.awake, io);
    const gpu_batches = try gpu_ctx.matchDescriptorPairsBatch(gpa, all_kps, pairs, 0.80, 72);
    const m_gpu_ms = @as(f64, @floatFromInt(t_m_gpu0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
    for (gpu_batches) |ms| gpa.free(ms);
    gpa.free(gpu_batches);

    // 3. End-to-End Synthetic 11-Camera SfM Benchmark
    const ds = try generateSyntheticDataset(gpa, 11, 640, 480);
    defer {
        for (ds.images) |*im| im.deinit(gpa);
        gpa.free(ds.images);
    }

    const base_cfg: sfm.pipeline.SfmConfig = .{
        .max_keypoints = 2000,
        .ransac_iters = 150,
        .enable_guided_densification = false,
    };

    var cfg_1t = base_cfg;
    cfg_1t.num_threads = 1;
    const t0 = Io.Clock.now(.awake, io);
    var r1 = try sfm.pipeline.runIncrementalSfm(gpa, ds.k, ds.images, cfg_1t);
    const sfm_1t_ms = @as(f64, @floatFromInt(t0.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
    r1.deinit(gpa);

    var cfg_mt = base_cfg;
    cfg_mt.num_threads = 0;
    const t1 = Io.Clock.now(.awake, io);
    var r2 = try sfm.pipeline.runIncrementalSfm(gpa, ds.k, ds.images, cfg_mt);
    const sfm_mt_ms = @as(f64, @floatFromInt(t1.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
    r2.deinit(gpa);

    var cfg_gpu = base_cfg;
    cfg_gpu.num_threads = 0;
    cfg_gpu.use_gpu = true;
    const t2 = Io.Clock.now(.awake, io);
    var r3 = try sfm.pipeline.runIncrementalSfm(gpa, ds.k, ds.images, cfg_gpu);
    const sfm_gpu_ms = @as(f64, @floatFromInt(t2.durationTo(Io.Clock.now(.awake, io)).nanoseconds)) / 1e6;
    r3.deinit(gpa);

    std.debug.print(
        "=== Benchmark Results (logical CPUs={d}) ===\n" ++
            "1) Harris + Blur (11 x 1200x800):\n" ++
            "   CPU 1-Thread : {d:8.2} ms\n" ++
            "   WebGPU GPU   : {d:8.2} ms  ({d:.2}x vs CPU 1T)\n\n" ++
            "2) Pairwise Matching (55 pairs, 10,500 kps/cam):\n" ++
            "   CPU 1-Thread : {d:8.2} ms\n" ++
            "   CPU {d}-Thread : {d:8.2} ms  ({d:.2}x vs CPU 1T)\n" ++
            "   WebGPU GPU   : {d:8.2} ms  ({d:.2}x vs CPU 1T, {d:.2}x vs CPU MT)\n\n" ++
            "3) End-to-End Synthetic SfM (11 cams, 640x480, 2000 kps):\n" ++
            "   CPU 1-Thread : {d:8.2} ms\n" ++
            "   CPU {d}-Thread : {d:8.2} ms  ({d:.2}x vs CPU 1T)\n" ++
            "   WebGPU + {d}T  : {d:8.2} ms  ({d:.2}x vs CPU 1T, {d:.2}x vs CPU MT)\n",
        .{
            cpu_count,
            h_cpu_ms,
            h_gpu_ms,
            h_cpu_ms / h_gpu_ms,
            m_cpu1_ms,
            cpu_count,
            m_cpumt_ms,
            m_cpu1_ms / m_cpumt_ms,
            m_gpu_ms,
            m_cpu1_ms / m_gpu_ms,
            m_cpumt_ms / m_gpu_ms,
            sfm_1t_ms,
            cpu_count,
            sfm_mt_ms,
            sfm_1t_ms / sfm_mt_ms,
            cpu_count,
            sfm_gpu_ms,
            sfm_1t_ms / sfm_gpu_ms,
            sfm_mt_ms / sfm_gpu_ms,
        },
    );
}
