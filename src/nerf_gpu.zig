const std = @import("std");
const c = @import("c_wgpu");
const gpu = @import("gpu.zig");
const nerf = @import("nerf.zig");
const render = @import("render.zig");
const Mat3 = @import("math/mat.zig").Mat3;
const Vec3 = @import("math/vec.zig").Vec3;
const CameraPose = @import("math/lie.zig").CameraPose;
const Io = std.Io;

const ImageData = @import("dataset.zig").ImageData;

const wgsl_common = @embedFile("nerf_common.wgsl");
const wgsl_forward = wgsl_common ++ "\n" ++ @embedFile("nerf_forward.wgsl");
const wgsl_backward = wgsl_common ++ "\n" ++ @embedFile("nerf_backward.wgsl");
const wgsl_occ = wgsl_common ++ "\n" ++ @embedFile("nerf_occ.wgsl");
const wgsl_render = wgsl_common ++ "\n" ++ @embedFile("nerf_render.wgsl");
const wgsl_adam = @embedFile("nerf_adam.wgsl");

pub const NerfUniformsGpu = extern struct {
    bounds_center_inv_r: [4]f32,
    bounds_range: [4]f32,
    grid_dims: [4]u32,
    grid_scales: [4]f32,
    mlp_dims0: [4]u32,
    mlp_dims1: [4]u32,
    mlp_offs0: [4]u32,
    mlp_offs1: [4]u32,
    run_cfg: [4]u32,
    opt_cfg: [4]f32,
    reg_cfg: [4]f32,
};

pub const TrainingRayGpu = extern struct {
    origin_jitter: [4]f32,
    dir_depth: [4]f32,
    target_rgb_pad: [4]f32,
    bg_pad: [4]f32,
};

pub const RayForwardResultGpu = extern struct {
    d_rgb_loss: [4]f32,
    grad_meta: [4]f32,
    extra: [4]f32,
};

pub const HitSampleGpu = extern struct {
    u_dt: [4]f32,
    rgb_t: [4]f32,
    t_a_w_s: [4]f32,
};

pub const AdamUniformsGpu = extern struct {
    len_pad: [4]u32,
    params: [4]f32,
    extra: [4]f32,
};

pub const RenderUniformsGpu = extern struct {
    k_intrinsics: [4]f32,
    k_skew_dims: [4]f32,
    cam_center: [4]f32,
    r_row0: [4]f32,
    r_row1: [4]f32,
    r_row2: [4]f32,
    bg_skip: [4]f32,
};

comptime {
    std.debug.assert(@sizeOf(NerfUniformsGpu) == 176);
    std.debug.assert(@sizeOf(TrainingRayGpu) == 64);
    std.debug.assert(@sizeOf(RayForwardResultGpu) == 48);
    std.debug.assert(@sizeOf(HitSampleGpu) == 48);
    std.debug.assert(@sizeOf(AdamUniformsGpu) == 48);
    std.debug.assert(@sizeOf(RenderUniformsGpu) == 112);
}

const StagePipeline = struct {
    bgl: c.WGPUBindGroupLayout,
    pl: c.WGPUPipelineLayout,
    pipeline: c.WGPUComputePipeline,

    fn deinit(self: *StagePipeline) void {
        c.wgpuComputePipelineRelease(self.pipeline);
        c.wgpuPipelineLayoutRelease(self.pl);
        c.wgpuBindGroupLayoutRelease(self.bgl);
        self.* = undefined;
    }
};

fn createStagePipeline(
    device: c.WGPUDevice,
    label: []const u8,
    wgsl_src: []const u8,
    entries: []const c.WGPUBindGroupLayoutEntry,
) !StagePipeline {
    const sm = gpu.createShaderModule(device, label, wgsl_src) orelse return error.GpuShaderCompileFailed;
    defer c.wgpuShaderModuleRelease(sm);

    const bgl = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .nextInChain = null,
        .label = gpu.wgpuStr(label),
        .entryCount = entries.len,
        .entries = entries.ptr,
    }) orelse return error.GpuPipelineFailed;
    errdefer c.wgpuBindGroupLayoutRelease(bgl);

    const pl = gpu.createPipelineLayout(device, label, bgl) orelse return error.GpuPipelineFailed;
    errdefer c.wgpuPipelineLayoutRelease(pl);

    const pipeline = gpu.createComputePipeline(device, label, pl, sm, "main") orelse return error.GpuPipelineFailed;
    return .{ .bgl = bgl, .pl = pl, .pipeline = pipeline };
}

fn createBindGroup(
    device: c.WGPUDevice,
    label: []const u8,
    bgl: c.WGPUBindGroupLayout,
    entries: []const c.WGPUBindGroupEntry,
) !c.WGPUBindGroup {
    return c.wgpuDeviceCreateBindGroup(device, &.{
        .nextInChain = null,
        .label = gpu.wgpuStr(label),
        .layout = bgl,
        .entryCount = entries.len,
        .entries = entries.ptr,
    }) orelse return error.GpuBindGroupFailed;
}

fn readBufferSlice(
    instance: c.WGPUInstance,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    src_buf: c.WGPUBuffer,
    comptime T: type,
    dst: []T,
) !void {
    if (dst.len == 0) return;
    const raw_bytes = dst.len * @sizeOf(T);
    const aligned_bytes = gpu.alignUp(raw_bytes, 4);
    const stage = try gpu.createBuffer(device, aligned_bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
    defer gpu.releaseBuffer(stage);

    const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
    const enc = c.wgpuDeviceCreateCommandEncoder(device, &enc_desc) orelse return error.GpuEncoderFailed;
    defer c.wgpuCommandEncoderRelease(enc);

    c.wgpuCommandEncoderCopyBufferToBuffer(enc, src_buf, 0, stage, 0, aligned_bytes);
    const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
    const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(queue, 1, &cmd);

    try gpu.mapReadCopy(instance, device, stage, raw_bytes, std.mem.sliceAsBytes(dst));
}

pub const NerfGpuContext = struct {
    dev_owned: ?gpu.DeviceHandle,
    instance: c.WGPUInstance,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    forward: StagePipeline,
    backward: StagePipeline,
    adam: StagePipeline,
    occ: StagePipeline,
    render: StagePipeline,

    pub fn initStandalone() !NerfGpuContext {
        var dev = try gpu.initDeviceContext();
        errdefer dev.deinit();
        return try initWithHandles(dev, dev.instance, dev.device, dev.queue);
    }

    pub fn initFromGpu(g: *const gpu.GpuContext) !NerfGpuContext {
        return try initWithHandles(null, g.instance, g.device, g.queue);
    }

    fn initWithHandles(
        dev_owned: ?gpu.DeviceHandle,
        instance: c.WGPUInstance,
        device: c.WGPUDevice,
        queue: c.WGPUQueue,
    ) !NerfGpuContext {
        const ro = c.WGPUBufferBindingType_ReadOnlyStorage;
        const rw = c.WGPUBufferBindingType_Storage;
        const uni = c.WGPUBufferBindingType_Uniform;

        const fwd_entries = [_]c.WGPUBindGroupLayoutEntry{
            gpu.bufferLayoutEntry(0, uni, false),
            gpu.bufferLayoutEntry(1, ro, false),
            gpu.bufferLayoutEntry(2, ro, false),
            gpu.bufferLayoutEntry(3, ro, false),
            gpu.bufferLayoutEntry(4, ro, false),
            gpu.bufferLayoutEntry(5, rw, false),
            gpu.bufferLayoutEntry(6, rw, false),
        };
        var forward = try createStagePipeline(device, "nerf_forward", wgsl_forward, &fwd_entries);
        errdefer forward.deinit();

        const bwd_entries = [_]c.WGPUBindGroupLayoutEntry{
            gpu.bufferLayoutEntry(0, uni, false),
            gpu.bufferLayoutEntry(1, ro, false),
            gpu.bufferLayoutEntry(2, ro, false),
            gpu.bufferLayoutEntry(3, ro, false),
            gpu.bufferLayoutEntry(4, ro, false),
            gpu.bufferLayoutEntry(5, ro, false),
            gpu.bufferLayoutEntry(6, rw, false),
            gpu.bufferLayoutEntry(7, rw, false),
        };
        var backward = try createStagePipeline(device, "nerf_backward", wgsl_backward, &bwd_entries);
        errdefer backward.deinit();

        const adam_entries = [_]c.WGPUBindGroupLayoutEntry{
            gpu.bufferLayoutEntry(0, uni, false),
            gpu.bufferLayoutEntry(1, rw, false),
            gpu.bufferLayoutEntry(2, rw, false),
            gpu.bufferLayoutEntry(3, rw, false),
            gpu.bufferLayoutEntry(4, rw, false),
        };
        var adam = try createStagePipeline(device, "nerf_adam", wgsl_adam, &adam_entries);
        errdefer adam.deinit();

        const occ_entries = [_]c.WGPUBindGroupLayoutEntry{
            gpu.bufferLayoutEntry(0, uni, false),
            gpu.bufferLayoutEntry(1, ro, false),
            gpu.bufferLayoutEntry(2, ro, false),
            gpu.bufferLayoutEntry(3, rw, false),
            gpu.bufferLayoutEntry(4, rw, false),
        };
        var occ = try createStagePipeline(device, "nerf_occ", wgsl_occ, &occ_entries);
        errdefer occ.deinit();

        const ren_entries = [_]c.WGPUBindGroupLayoutEntry{
            gpu.bufferLayoutEntry(0, uni, false),
            gpu.bufferLayoutEntry(1, ro, false),
            gpu.bufferLayoutEntry(2, ro, false),
            gpu.bufferLayoutEntry(3, ro, false),
            gpu.bufferLayoutEntry(4, uni, false),
            gpu.bufferLayoutEntry(5, rw, false),
        };
        var render_pipe = try createStagePipeline(device, "nerf_render", wgsl_render, &ren_entries);
        errdefer render_pipe.deinit();

        return .{
            .dev_owned = dev_owned,
            .instance = instance,
            .device = device,
            .queue = queue,
            .forward = forward,
            .backward = backward,
            .adam = adam,
            .occ = occ,
            .render = render_pipe,
        };
    }

    pub fn deinit(self: *NerfGpuContext) void {
        self.render.deinit();
        self.occ.deinit();
        self.adam.deinit();
        self.backward.deinit();
        self.forward.deinit();
        if (self.dev_owned) |*d| d.deinit();
        self.* = undefined;
    }
};

fn computeMlpGlobalScale(batch_rays: usize) f32 {
    const wg_count: f32 = @floatFromInt(@max(@as(usize, 1), (batch_rays + 63) / 64));
    return std.math.clamp(262144.0 / wg_count, 4096.0, 262144.0);
}

fn computeHashGlobalScale(batch_rays: usize) f32 {
    return if (batch_rays <= 256) 262144.0 else 65536.0;
}

fn packNerfUniforms(
    model: *const nerf.NerfModel,
    batch_rays: usize,
    step_t: usize,
    skip_empty: bool,
) NerfUniformsGpu {
    const l = model.layout;
    const cfg = model.config;
    const b = model.bounds;
    const inv_b: f32 = 1.0 / @as(f32, @floatFromInt(@max(@as(usize, 1), batch_rays)));
    const jit = nerf.OccupancyGrid.jitterForStep(step_t);
    return .{
        .bounds_center_inv_r = .{ b.center[0], b.center[1], b.center[2], b.inv_radius },
        .bounds_range = .{ b.t_near, b.t_far, cfg.dt_min, if (b.use_contraction) 1.0 else 0.0 },
        .grid_dims = .{
            @intCast(model.hash_grid.num_levels),
            @intCast(model.hash_grid.table_size),
            model.hash_grid.table_mask,
            @intCast(model.occ_grid.res),
        },
        .grid_scales = .{ model.hash_grid.n_min, model.hash_grid.per_level_scale, cfg.depth_loss_weight, inv_b },
        .mlp_dims0 = .{
            @intCast(l.geo_in),
            @intCast(l.geo_h),
            @intCast(l.geo_out),
            @intCast(l.col_in),
        },
        .mlp_dims1 = .{
            @intCast(l.col_h),
            @intCast(l.col_out),
            @intCast(cfg.geo_feat_dim),
            @intCast(cfg.max_ray_samples),
        },
        .mlp_offs0 = .{ @intCast(l.w0_off), @intCast(l.b0_off), @intCast(l.w1_off), @intCast(l.b1_off) },
        .mlp_offs1 = .{ @intCast(l.w2_off), @intCast(l.b2_off), @intCast(l.w3_off), @intCast(l.b3_off) },
        .run_cfg = .{
            @intCast(batch_rays),
            @intCast(step_t),
            if (skip_empty) 1 else 0,
            @intCast(l.total_params),
        },
        .opt_cfg = .{
            cfg.occ_decay,
            cfg.occ_threshold,
            computeMlpGlobalScale(batch_rays),
            computeHashGlobalScale(batch_rays),
        },
        .reg_cfg = .{ jit[0], jit[1], jit[2], cfg.distortion_weight },
    };
}

pub fn packTrainingRays(
    batch: []const nerf.TrainingRay,
    bgs: []const [3]f32,
    jitters: []const f32,
    out: []TrainingRayGpu,
) void {
    for (batch, bgs, jitters, out) |r, bg, jit, *dst| {
        dst.* = .{
            .origin_jitter = .{ r.origin[0], r.origin[1], r.origin[2], jit },
            .dir_depth = .{ r.dir[0], r.dir[1], r.dir[2], r.depth orelse 0.0 },
            .target_rgb_pad = .{ r.rgb[0], r.rgb[1], r.rgb[2], 0.0 },
            .bg_pad = .{ bg[0], bg[1], bg[2], 0.0 },
        };
    }
}

pub const NerfGpuSession = struct {
    ctx: *const NerfGpuContext,
    max_batch_rays: usize,
    mlp_len: usize,
    hash_len: usize,
    occ_len: usize,
    max_ray_samples: usize,

    u_cfg_buf: c.WGPUBuffer,
    adam_mlp_u_buf: c.WGPUBuffer,
    adam_hash_u_buf: c.WGPUBuffer,

    mlp_buf: c.WGPUBuffer,
    mlp_grad_buf: c.WGPUBuffer,
    mlp_m_buf: c.WGPUBuffer,
    mlp_v_buf: c.WGPUBuffer,

    hash_buf: c.WGPUBuffer,
    hash_grad_buf: c.WGPUBuffer,
    hash_m_buf: c.WGPUBuffer,
    hash_v_buf: c.WGPUBuffer,

    occ_density_buf: c.WGPUBuffer,
    occ_flags_buf: c.WGPUBuffer,

    rays_buf: c.WGPUBuffer,
    ray_res_buf: c.WGPUBuffer,
    hit_samples_buf: c.WGPUBuffer,

    fwd_bg: c.WGPUBindGroup,
    bwd_bg: c.WGPUBindGroup,
    adam_mlp_bg: c.WGPUBindGroup,
    adam_hash_bg: c.WGPUBindGroup,
    occ_bg: c.WGPUBindGroup,

    pub fn initFromModel(
        ctx: *const NerfGpuContext,
        model: *const nerf.NerfModel,
        max_batch_rays: usize,
    ) !NerfGpuSession {
        const l = model.layout;
        if (model.hash_grid.num_levels > 12 or
            l.geo_in > 24 or
            l.geo_h > 64 or
            l.geo_out > 32 or
            l.col_in > 48 or
            l.col_h > 64)
        {
            return error.UnsupportedNerfShape;
        }

        const device = ctx.device;
        const queue = ctx.queue;
        const batch_cap = @max(@as(usize, 1), max_batch_rays);
        const mlp_len = l.total_params;
        const hash_len = model.hash_grid.params.len;
        const occ_len = model.occ_grid.density.len;

        const u_cfg_bytes = @sizeOf(NerfUniformsGpu);
        const adam_u_bytes = @sizeOf(AdamUniformsGpu);
        const mlp_bytes = mlp_len * @sizeOf(f32);
        const hash_bytes = hash_len * @sizeOf(f32);
        const occ_bytes = occ_len * @sizeOf(f32);
        const rays_bytes = batch_cap * @sizeOf(TrainingRayGpu);
        const ray_res_bytes = batch_cap * @sizeOf(RayForwardResultGpu);
        const hits_bytes = batch_cap * model.config.max_ray_samples * @sizeOf(HitSampleGpu);

        const rw_cp = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
        const ro_dst = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst;
        const uni = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;

        const u_cfg_buf = try gpu.createBuffer(device, u_cfg_bytes, uni);
        errdefer gpu.releaseBuffer(u_cfg_buf);
        const adam_mlp_u_buf = try gpu.createBuffer(device, adam_u_bytes, uni);
        errdefer gpu.releaseBuffer(adam_mlp_u_buf);
        const adam_hash_u_buf = try gpu.createBuffer(device, adam_u_bytes, uni);
        errdefer gpu.releaseBuffer(adam_hash_u_buf);

        const mlp_buf = try gpu.createBuffer(device, mlp_bytes, rw_cp);
        errdefer gpu.releaseBuffer(mlp_buf);
        const mlp_grad_buf = try gpu.createBuffer(device, mlp_bytes, rw_cp);
        errdefer gpu.releaseBuffer(mlp_grad_buf);
        const mlp_m_buf = try gpu.createBuffer(device, mlp_bytes, rw_cp);
        errdefer gpu.releaseBuffer(mlp_m_buf);
        const mlp_v_buf = try gpu.createBuffer(device, mlp_bytes, rw_cp);
        errdefer gpu.releaseBuffer(mlp_v_buf);

        const hash_buf = try gpu.createBuffer(device, hash_bytes, rw_cp);
        errdefer gpu.releaseBuffer(hash_buf);
        const hash_grad_buf = try gpu.createBuffer(device, hash_bytes, rw_cp);
        errdefer gpu.releaseBuffer(hash_grad_buf);
        const hash_m_buf = try gpu.createBuffer(device, hash_bytes, rw_cp);
        errdefer gpu.releaseBuffer(hash_m_buf);
        const hash_v_buf = try gpu.createBuffer(device, hash_bytes, rw_cp);
        errdefer gpu.releaseBuffer(hash_v_buf);

        const occ_density_buf = try gpu.createBuffer(device, occ_bytes, rw_cp);
        errdefer gpu.releaseBuffer(occ_density_buf);
        const occ_flags_buf = try gpu.createBuffer(device, occ_bytes, rw_cp);
        errdefer gpu.releaseBuffer(occ_flags_buf);

        const rays_buf = try gpu.createBuffer(device, rays_bytes, ro_dst);
        errdefer gpu.releaseBuffer(rays_buf);
        const ray_res_buf = try gpu.createBuffer(device, ray_res_bytes, rw_cp);
        errdefer gpu.releaseBuffer(ray_res_buf);
        const hit_samples_buf = try gpu.createBuffer(device, hits_bytes, c.WGPUBufferUsage_Storage);
        errdefer gpu.releaseBuffer(hit_samples_buf);

        const init_u = packNerfUniforms(model, batch_cap, model.step, true);
        c.wgpuQueueWriteBuffer(queue, u_cfg_buf, 0, &init_u, u_cfg_bytes);
        c.wgpuQueueWriteBuffer(queue, mlp_buf, 0, model.mlp_params.ptr, mlp_bytes);
        c.wgpuQueueWriteBuffer(queue, mlp_m_buf, 0, model.adam_mlp.m.ptr, mlp_bytes);
        c.wgpuQueueWriteBuffer(queue, mlp_v_buf, 0, model.adam_mlp.v.ptr, mlp_bytes);
        c.wgpuQueueWriteBuffer(queue, hash_buf, 0, model.hash_grid.params.ptr, hash_bytes);
        c.wgpuQueueWriteBuffer(queue, hash_m_buf, 0, model.adam_hash.m.ptr, hash_bytes);
        c.wgpuQueueWriteBuffer(queue, hash_v_buf, 0, model.adam_hash.v.ptr, hash_bytes);
        c.wgpuQueueWriteBuffer(queue, occ_density_buf, 0, model.occ_grid.density.ptr, occ_bytes);

        const occ_flags_u32 = try std.heap.page_allocator.alloc(u32, occ_len);
        defer std.heap.page_allocator.free(occ_flags_u32);
        for (model.occ_grid.occupied, 0..) |flag, i| occ_flags_u32[i] = if (flag) 1 else 0;
        c.wgpuQueueWriteBuffer(queue, occ_flags_buf, 0, occ_flags_u32.ptr, occ_bytes);

        {
            const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
            const enc = c.wgpuDeviceCreateCommandEncoder(device, &enc_desc) orelse return error.GpuEncoderFailed;
            defer c.wgpuCommandEncoderRelease(enc);
            c.wgpuCommandEncoderClearBuffer(enc, mlp_grad_buf, 0, gpu.alignUp(mlp_bytes, 4));
            c.wgpuCommandEncoderClearBuffer(enc, hash_grad_buf, 0, gpu.alignUp(hash_bytes, 4));
            const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
            const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
            defer c.wgpuCommandBufferRelease(cmd);
            c.wgpuQueueSubmit(queue, 1, &cmd);
        }

        const fwd_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, u_cfg_buf, 0, u_cfg_bytes),
            gpu.bufEntry(1, mlp_buf, 0, mlp_bytes),
            gpu.bufEntry(2, hash_buf, 0, hash_bytes),
            gpu.bufEntry(3, occ_flags_buf, 0, occ_bytes),
            gpu.bufEntry(4, rays_buf, 0, rays_bytes),
            gpu.bufEntry(5, ray_res_buf, 0, ray_res_bytes),
            gpu.bufEntry(6, hit_samples_buf, 0, hits_bytes),
        };
        const fwd_bg = try createBindGroup(device, "nerf_fwd_bg", ctx.forward.bgl, &fwd_entries);
        errdefer c.wgpuBindGroupRelease(fwd_bg);

        const bwd_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, u_cfg_buf, 0, u_cfg_bytes),
            gpu.bufEntry(1, mlp_buf, 0, mlp_bytes),
            gpu.bufEntry(2, hash_buf, 0, hash_bytes),
            gpu.bufEntry(3, rays_buf, 0, rays_bytes),
            gpu.bufEntry(4, ray_res_buf, 0, ray_res_bytes),
            gpu.bufEntry(5, hit_samples_buf, 0, hits_bytes),
            gpu.bufEntry(6, mlp_grad_buf, 0, mlp_bytes),
            gpu.bufEntry(7, hash_grad_buf, 0, hash_bytes),
        };
        const bwd_bg = try createBindGroup(device, "nerf_bwd_bg", ctx.backward.bgl, &bwd_entries);
        errdefer c.wgpuBindGroupRelease(bwd_bg);

        const adam_mlp_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, adam_mlp_u_buf, 0, adam_u_bytes),
            gpu.bufEntry(1, mlp_buf, 0, mlp_bytes),
            gpu.bufEntry(2, mlp_grad_buf, 0, mlp_bytes),
            gpu.bufEntry(3, mlp_m_buf, 0, mlp_bytes),
            gpu.bufEntry(4, mlp_v_buf, 0, mlp_bytes),
        };
        const adam_mlp_bg = try createBindGroup(device, "nerf_adam_mlp_bg", ctx.adam.bgl, &adam_mlp_entries);
        errdefer c.wgpuBindGroupRelease(adam_mlp_bg);

        const adam_hash_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, adam_hash_u_buf, 0, adam_u_bytes),
            gpu.bufEntry(1, hash_buf, 0, hash_bytes),
            gpu.bufEntry(2, hash_grad_buf, 0, hash_bytes),
            gpu.bufEntry(3, hash_m_buf, 0, hash_bytes),
            gpu.bufEntry(4, hash_v_buf, 0, hash_bytes),
        };
        const adam_hash_bg = try createBindGroup(device, "nerf_adam_hash_bg", ctx.adam.bgl, &adam_hash_entries);
        errdefer c.wgpuBindGroupRelease(adam_hash_bg);

        const occ_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, u_cfg_buf, 0, u_cfg_bytes),
            gpu.bufEntry(1, mlp_buf, 0, mlp_bytes),
            gpu.bufEntry(2, hash_buf, 0, hash_bytes),
            gpu.bufEntry(3, occ_density_buf, 0, occ_bytes),
            gpu.bufEntry(4, occ_flags_buf, 0, occ_bytes),
        };
        const occ_bg = try createBindGroup(device, "nerf_occ_bg", ctx.occ.bgl, &occ_entries);
        errdefer c.wgpuBindGroupRelease(occ_bg);

        return .{
            .ctx = ctx,
            .max_batch_rays = batch_cap,
            .mlp_len = mlp_len,
            .hash_len = hash_len,
            .occ_len = occ_len,
            .max_ray_samples = model.config.max_ray_samples,
            .u_cfg_buf = u_cfg_buf,
            .adam_mlp_u_buf = adam_mlp_u_buf,
            .adam_hash_u_buf = adam_hash_u_buf,
            .mlp_buf = mlp_buf,
            .mlp_grad_buf = mlp_grad_buf,
            .mlp_m_buf = mlp_m_buf,
            .mlp_v_buf = mlp_v_buf,
            .hash_buf = hash_buf,
            .hash_grad_buf = hash_grad_buf,
            .hash_m_buf = hash_m_buf,
            .hash_v_buf = hash_v_buf,
            .occ_density_buf = occ_density_buf,
            .occ_flags_buf = occ_flags_buf,
            .rays_buf = rays_buf,
            .ray_res_buf = ray_res_buf,
            .hit_samples_buf = hit_samples_buf,
            .fwd_bg = fwd_bg,
            .bwd_bg = bwd_bg,
            .adam_mlp_bg = adam_mlp_bg,
            .adam_hash_bg = adam_hash_bg,
            .occ_bg = occ_bg,
        };
    }

    pub fn deinit(self: *NerfGpuSession) void {
        c.wgpuBindGroupRelease(self.occ_bg);
        c.wgpuBindGroupRelease(self.adam_hash_bg);
        c.wgpuBindGroupRelease(self.adam_mlp_bg);
        c.wgpuBindGroupRelease(self.bwd_bg);
        c.wgpuBindGroupRelease(self.fwd_bg);
        gpu.releaseBuffer(self.hit_samples_buf);
        gpu.releaseBuffer(self.ray_res_buf);
        gpu.releaseBuffer(self.rays_buf);
        gpu.releaseBuffer(self.occ_flags_buf);
        gpu.releaseBuffer(self.occ_density_buf);
        gpu.releaseBuffer(self.hash_v_buf);
        gpu.releaseBuffer(self.hash_m_buf);
        gpu.releaseBuffer(self.hash_grad_buf);
        gpu.releaseBuffer(self.hash_buf);
        gpu.releaseBuffer(self.mlp_v_buf);
        gpu.releaseBuffer(self.mlp_m_buf);
        gpu.releaseBuffer(self.mlp_grad_buf);
        gpu.releaseBuffer(self.mlp_buf);
        gpu.releaseBuffer(self.adam_hash_u_buf);
        gpu.releaseBuffer(self.adam_mlp_u_buf);
        gpu.releaseBuffer(self.u_cfg_buf);
        self.* = undefined;
    }

    fn dispatchStage(
        enc: c.WGPUCommandEncoder,
        pipeline: c.WGPUComputePipeline,
        bg: c.WGPUBindGroup,
        wg_x: u32,
        wg_y: u32,
    ) !void {
        var pass_desc = std.mem.zeroes(c.WGPUComputePassDescriptor);
        const pass = c.wgpuCommandEncoderBeginComputePass(enc, &pass_desc) orelse return error.GpuComputePassFailed;
        c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);
    }

    fn dispatch1D(
        enc: c.WGPUCommandEncoder,
        pipeline: c.WGPUComputePipeline,
        bg: c.WGPUBindGroup,
        elem_count: usize,
    ) !void {
        const total_wg = (elem_count + 63) / 64;
        const max_wg: usize = 65535;
        const wg_x: u32 = @intCast(@min(total_wg, max_wg));
        const wg_y: u32 = @intCast((total_wg + max_wg - 1) / max_wg);
        try dispatchStage(enc, pipeline, bg, wg_x, wg_y);
    }

    pub fn uploadUniforms(
        self: *const NerfGpuSession,
        model: *const nerf.NerfModel,
        batch_rays: usize,
        step_t: usize,
        skip_empty: bool,
    ) void {
        const u = packNerfUniforms(model, batch_rays, step_t, skip_empty);
        c.wgpuQueueWriteBuffer(self.ctx.queue, self.u_cfg_buf, 0, &u, @sizeOf(NerfUniformsGpu));
    }

    pub fn forwardBackwardOnly(
        self: *const NerfGpuSession,
        model: *const nerf.NerfModel,
        rays: []const TrainingRayGpu,
        skip_empty: bool,
        out_res: []RayForwardResultGpu,
        out_grad_mlp: []f32,
        out_grad_hash: []f32,
    ) !void {
        std.debug.assert(rays.len > 0 and rays.len <= self.max_batch_rays);
        const batch_rays = rays.len;
        const u = packNerfUniforms(model, batch_rays, model.step, skip_empty);
        c.wgpuQueueWriteBuffer(self.ctx.queue, self.u_cfg_buf, 0, &u, @sizeOf(NerfUniformsGpu));
        c.wgpuQueueWriteBuffer(self.ctx.queue, self.rays_buf, 0, rays.ptr, batch_rays * @sizeOf(TrainingRayGpu));

        const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
        const enc = c.wgpuDeviceCreateCommandEncoder(self.ctx.device, &enc_desc) orelse return error.GpuEncoderFailed;
        defer c.wgpuCommandEncoderRelease(enc);

        c.wgpuCommandEncoderClearBuffer(enc, self.mlp_grad_buf, 0, gpu.alignUp(self.mlp_len * @sizeOf(i32), 4));
        c.wgpuCommandEncoderClearBuffer(enc, self.hash_grad_buf, 0, gpu.alignUp(self.hash_len * @sizeOf(i32), 4));

        const wg_rays_fwd: u32 = @intCast((batch_rays + 63) / 64);
        const wg_rays_bwd: u32 = @intCast((batch_rays + 63) / 64);
        try dispatchStage(enc, self.ctx.forward.pipeline, self.fwd_bg, wg_rays_fwd, 1);
        try dispatchStage(enc, self.ctx.backward.pipeline, self.bwd_bg, wg_rays_bwd, 1);

        const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
        const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
        defer c.wgpuCommandBufferRelease(cmd);
        c.wgpuQueueSubmit(self.ctx.queue, 1, &cmd);

        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.ray_res_buf, RayForwardResultGpu, out_res[0..batch_rays]);

        const alloc = std.heap.page_allocator;
        const raw_mlp = try alloc.alloc(i32, self.mlp_len);
        defer alloc.free(raw_mlp);
        const raw_hash = try alloc.alloc(i32, self.hash_len);
        defer alloc.free(raw_hash);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.mlp_grad_buf, i32, raw_mlp);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.hash_grad_buf, i32, raw_hash);

        const inv_mlp = 1.0 / computeMlpGlobalScale(batch_rays);
        const inv_hash = 1.0 / computeHashGlobalScale(batch_rays);
        for (raw_mlp, 0..) |v, i| out_grad_mlp[i] = @as(f32, @floatFromInt(v)) * inv_mlp;
        for (raw_hash, 0..) |v, i| out_grad_hash[i] = @as(f32, @floatFromInt(v)) * inv_hash;
    }

    pub fn submitTrainStep(
        self: *const NerfGpuSession,
        model: *const nerf.NerfModel,
        rays: []const TrainingRayGpu,
        step_t: usize,
        lr_scale: f32,
        update_occ: bool,
        skip_empty: bool,
    ) !void {
        const batch_rays = rays.len;
        std.debug.assert(batch_rays > 0 and batch_rays <= self.max_batch_rays);

        c.wgpuQueueWriteBuffer(self.ctx.queue, self.rays_buf, 0, rays.ptr, batch_rays * @sizeOf(TrainingRayGpu));

        const tf: f32 = @floatFromInt(step_t);
        const inv_bc1 = 1.0 / (1.0 - std.math.pow(f32, 0.9, tf));
        const inv_bc2 = 1.0 / (1.0 - std.math.pow(f32, 0.999, tf));
        const lr = model.config.lr * lr_scale;
        const inv_batch: f32 = 1.0 / @as(f32, @floatFromInt(batch_rays));

        const adam_mlp_u = AdamUniformsGpu{
            .len_pad = .{ @intCast(self.mlp_len), 0, 0, 0 },
            .params = .{ lr, model.config.weight_decay, inv_bc1, inv_bc2 },
            .extra = .{ inv_batch / computeMlpGlobalScale(batch_rays), 0.0, 0.0, 0.0 },
        };
        const adam_hash_u = AdamUniformsGpu{
            .len_pad = .{ @intCast(self.hash_len), 0, 0, 0 },
            .params = .{ lr, 0.0, inv_bc1, inv_bc2 },
            .extra = .{ inv_batch / computeHashGlobalScale(batch_rays), 0.0, 0.0, 0.0 },
        };
        c.wgpuQueueWriteBuffer(self.ctx.queue, self.adam_mlp_u_buf, 0, &adam_mlp_u, @sizeOf(AdamUniformsGpu));
        c.wgpuQueueWriteBuffer(self.ctx.queue, self.adam_hash_u_buf, 0, &adam_hash_u, @sizeOf(AdamUniformsGpu));
        if (update_occ or !skip_empty or batch_rays != self.max_batch_rays) {
            self.uploadUniforms(model, batch_rays, step_t, skip_empty);
        }

        const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
        const enc = c.wgpuDeviceCreateCommandEncoder(self.ctx.device, &enc_desc) orelse return error.GpuEncoderFailed;
        defer c.wgpuCommandEncoderRelease(enc);

        const wg_rays_fwd: u32 = @intCast((batch_rays + 63) / 64);
        const wg_rays_bwd: u32 = @intCast((batch_rays + 63) / 64);

        try dispatchStage(enc, self.ctx.forward.pipeline, self.fwd_bg, wg_rays_fwd, 1);
        try dispatchStage(enc, self.ctx.backward.pipeline, self.bwd_bg, wg_rays_bwd, 1);
        try dispatch1D(enc, self.ctx.adam.pipeline, self.adam_mlp_bg, self.mlp_len);
        try dispatch1D(enc, self.ctx.adam.pipeline, self.adam_hash_bg, self.hash_len);

        if (update_occ) {
            try dispatch1D(enc, self.ctx.occ.pipeline, self.occ_bg, self.occ_len);
        }

        const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
        const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
        defer c.wgpuCommandBufferRelease(cmd);
        c.wgpuQueueSubmit(self.ctx.queue, 1, &cmd);
    }

    pub fn readLastBatchLoss(self: *const NerfGpuSession, allocator: std.mem.Allocator, batch_rays: usize) !f64 {
        const res = try allocator.alloc(RayForwardResultGpu, batch_rays);
        defer allocator.free(res);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.ray_res_buf, RayForwardResultGpu, res);
        var sum: f64 = 0.0;
        for (res) |r| sum += @as(f64, r.d_rgb_loss[3]);
        return sum / @as(f64, @floatFromInt(@max(@as(usize, 1), batch_rays)));
    }

    pub fn syncToCpu(self: *const NerfGpuSession, model: *nerf.NerfModel, step: usize) !void {
        _ = c.wgpuDevicePoll(self.ctx.device, 1, null);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.mlp_buf, f32, model.mlp_params);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.hash_buf, f32, model.hash_grid.params);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.occ_density_buf, f32, model.occ_grid.density);

        const flags_u32 = try std.heap.page_allocator.alloc(u32, self.occ_len);
        defer std.heap.page_allocator.free(flags_u32);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, self.occ_flags_buf, u32, flags_u32);
        for (flags_u32, 0..) |f, i| model.occ_grid.occupied[i] = f != 0;
        model.step = step;
    }

    pub fn renderViewGpu(
        self: *const NerfGpuSession,
        allocator: std.mem.Allocator,
        model: *const nerf.NerfModel,
        width: u32,
        height: u32,
        k: Mat3,
        pose: CameraPose,
        skip_empty: bool,
    ) !render.RenderedView {
        const n_pix = @as(usize, width) * @as(usize, height);
        const out_bytes = n_pix * @sizeOf([4]f32);

        const r_u_buf = try gpu.createBuffer(self.ctx.device, @sizeOf(RenderUniformsGpu), c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
        defer gpu.releaseBuffer(r_u_buf);
        const out_buf = try gpu.createBuffer(self.ctx.device, out_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
        defer gpu.releaseBuffer(out_buf);

        const cam_c = pose.cameraCenter();
        const ren_entries = [_]c.WGPUBindGroupEntry{
            gpu.bufEntry(0, self.u_cfg_buf, 0, @sizeOf(NerfUniformsGpu)),
            gpu.bufEntry(1, self.mlp_buf, 0, self.mlp_len * @sizeOf(f32)),
            gpu.bufEntry(2, self.hash_buf, 0, self.hash_len * @sizeOf(f32)),
            gpu.bufEntry(3, self.occ_flags_buf, 0, self.occ_len * @sizeOf(u32)),
            gpu.bufEntry(4, r_u_buf, 0, @sizeOf(RenderUniformsGpu)),
            gpu.bufEntry(5, out_buf, 0, out_bytes),
        };
        const ren_bg = try createBindGroup(self.ctx.device, "nerf_ren_bg", self.ctx.render.bgl, &ren_entries);
        defer c.wgpuBindGroupRelease(ren_bg);

        var y0: u32 = 0;
        const tile_rows: u32 = 200;
        while (y0 < height) : (y0 += tile_rows) {
            const tile_h = @min(tile_rows, height - y0);
            const r_u = RenderUniformsGpu{
                .k_intrinsics = .{ @floatCast(k.m[0][0]), @floatCast(k.m[1][1]), @floatCast(k.m[0][2]), @floatCast(k.m[1][2]) },
                .k_skew_dims = .{ @floatCast(k.m[0][1]), @floatFromInt(width), @floatFromInt(height), @floatFromInt(y0) },
                .cam_center = .{ @floatCast(cam_c.x), @floatCast(cam_c.y), @floatCast(cam_c.z), 0.0 },
                .r_row0 = .{ @floatCast(pose.r.m[0][0]), @floatCast(pose.r.m[0][1]), @floatCast(pose.r.m[0][2]), 0.0 },
                .r_row1 = .{ @floatCast(pose.r.m[1][0]), @floatCast(pose.r.m[1][1]), @floatCast(pose.r.m[1][2]), 0.0 },
                .r_row2 = .{ @floatCast(pose.r.m[2][0]), @floatCast(pose.r.m[2][1]), @floatCast(pose.r.m[2][2]), 0.0 },
                .bg_skip = .{ model.config.bg_color[0], model.config.bg_color[1], model.config.bg_color[2], if (skip_empty) 1.0 else 0.0 },
            };
            c.wgpuQueueWriteBuffer(self.ctx.queue, r_u_buf, 0, &r_u, @sizeOf(RenderUniformsGpu));

            const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
            const enc = c.wgpuDeviceCreateCommandEncoder(self.ctx.device, &enc_desc) orelse return error.GpuEncoderFailed;
            defer c.wgpuCommandEncoderRelease(enc);

            try dispatchStage(enc, self.ctx.render.pipeline, ren_bg, (width + 7) / 8, (tile_h + 7) / 8);

            const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
            const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
            defer c.wgpuCommandBufferRelease(cmd);
            c.wgpuQueueSubmit(self.ctx.queue, 1, &cmd);
            _ = c.wgpuDevicePoll(self.ctx.device, 1, null);
        }

        const raw_pixels = try allocator.alloc([4]f32, n_pix);
        defer allocator.free(raw_pixels);
        try readBufferSlice(self.ctx.instance, self.ctx.device, self.ctx.queue, out_buf, [4]f32, raw_pixels);

        const rgb = try allocator.alloc(u8, n_pix * 3);
        errdefer allocator.free(rgb);
        const depth = try allocator.alloc(f32, n_pix);
        errdefer allocator.free(depth);
        const covered = try allocator.alloc(bool, n_pix);
        errdefer allocator.free(covered);
        @memset(covered, true);

        for (raw_pixels, 0..) |px, i| {
            for (0..3) |ch| {
                rgb[i * 3 + ch] = @intFromFloat(std.math.clamp(px[ch] * 255.0 + 0.5, 0.0, 255.0));
            }
            depth[i] = px[3];
        }

        return .{
            .width = width,
            .height = height,
            .rgb = rgb,
            .depth = depth,
            .covered = covered,
        };
    }
};

pub fn trainOnDatasetGpu(
    allocator: std.mem.Allocator,
    ctx: *const NerfGpuContext,
    k: Mat3,
    images: []const ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    points: []const Vec3,
    cfg: nerf.NerfConfig,
    timings_out: ?*nerf.NerfTimings,
) !nerf.NerfModel {
    const t_setup0: ?Io.Timestamp = if (cfg.io) |io_v| Io.Clock.now(.awake, io_v) else null;

    var reg_cams: std.ArrayList(usize) = .empty;
    defer reg_cams.deinit(allocator);
    for (registered, 0..) |ok, ci| {
        if (ok) try reg_cams.append(allocator, ci);
    }
    if (reg_cams.items.len == 0) return error.NoRegisteredCameras;

    const bounds = nerf.SceneBounds.fromSfm(points, poses, registered, cfg.use_contraction);
    var model = try nerf.NerfModel.init(allocator, cfg, bounds);
    errdefer model.deinit(allocator);

    const depth_maps = if (cfg.depth_loss_weight > 0.0 and points.len > 0)
        try nerf.buildSparseDepthMaps(allocator, points, k, images, poses, registered)
    else
        null;
    defer if (depth_maps) |maps| {
        for (maps) |m| allocator.free(m);
        allocator.free(maps);
    };
    if (points.len >= 32) {
        model.occ_grid.seedFromViews(bounds, k, images, poses, registered, depth_maps);
    }

    var prng = std.Random.DefaultPrng.init(cfg.seed ^ 0x55aa55aa);
    const rng = prng.random();

    const batch = try allocator.alloc(nerf.TrainingRay, cfg.batch_rays);
    defer allocator.free(batch);
    const bgs = try allocator.alloc([3]f32, cfg.batch_rays);
    defer allocator.free(bgs);
    const jitters = try allocator.alloc(f32, cfg.batch_rays);
    defer allocator.free(jitters);
    const gpu_rays = try allocator.alloc(TrainingRayGpu, cfg.batch_rays);
    defer allocator.free(gpu_rays);

    var session = try NerfGpuSession.initFromModel(ctx, &model, cfg.batch_rays);
    defer session.deinit();

    const setup_ms = nerf.elapsedMsOpt(cfg.io, t_setup0);
    const t_train0: ?Io.Timestamp = if (cfg.io) |io_v| Io.Clock.now(.awake, io_v) else null;

    for (0..cfg.num_steps) |step_idx| {
        nerf.sampleTrainingBatch(k, images, poses, reg_cams.items, depth_maps, cfg, rng, batch, bgs, jitters);
        packTrainingRays(batch, bgs, jitters, gpu_rays);

        const lr_scale = nerf.cosineLrScale(step_idx, cfg.num_steps);
        const do_occ = (step_idx + 1) % cfg.occ_update_interval == 0;
        try session.submitTrainStep(&model, gpu_rays, step_idx + 1, lr_scale, do_occ, true);
        if ((step_idx + 1) % 4 == 0) {
            _ = c.wgpuDevicePoll(ctx.device, 1, null);
        }
        if (cfg.io != null and cfg.num_steps >= 200 and ((step_idx + 1) % 250 == 0 or step_idx + 1 == cfg.num_steps)) {
            _ = c.wgpuDevicePoll(ctx.device, 1, null);
            const cur_loss = session.readLastBatchLoss(allocator, cfg.batch_rays) catch 0.0;
            const ms_so_far = nerf.elapsedMsOpt(cfg.io, t_train0);
            std.log.info("NeRF GPU step {d}/{d} ({d:.0} ms, loss={d:.5})", .{ step_idx + 1, cfg.num_steps, ms_so_far, cur_loss });
        }
    }

    _ = c.wgpuDevicePoll(ctx.device, 1, null);
    const final_loss = if (cfg.num_steps > 0) try session.readLastBatchLoss(allocator, cfg.batch_rays) else 0.0;
    try session.syncToCpu(&model, cfg.num_steps);

    if (timings_out) |tm| {
        tm.setup_ms = setup_ms;
        tm.train_ms = nerf.elapsedMsOpt(cfg.io, t_train0);
        tm.final_loss = final_loss;
    }

    return model;
}

test "GPU NeRF forward pass and gradients match CPU reference" {
    const gpa = std.testing.allocator;
    var ctx = try NerfGpuContext.initStandalone();
    defer ctx.deinit();

    const cfg = nerf.NerfConfig{
        .num_levels = 4,
        .table_log2 = 10,
        .n_min = 4.0,
        .n_max = 32.0,
        .geo_hidden = 16,
        .geo_feat_dim = 7,
        .col_hidden = 16,
        .batch_rays = 8,
        .max_ray_samples = 32,
        .occ_grid_res = 8,
        .depth_loss_weight = 0.05,
        .use_contraction = true,
    };
    const bounds = nerf.SceneBounds{
        .center = .{ 0.0, 0.0, 0.0 },
        .inv_radius = 0.5,
        .use_contraction = true,
        .t_near = 0.2,
        .t_far = 6.0,
    };

    var model = try nerf.NerfModel.init(gpa, cfg, bounds);
    defer model.deinit(gpa);
    for (model.hash_grid.params, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) * 0.025;
    }

    var session = try NerfGpuSession.initFromModel(&ctx, &model, cfg.batch_rays);
    defer session.deinit();

    var prng = std.Random.DefaultPrng.init(2026);
    const rng = prng.random();

    const batch = try gpa.alloc(nerf.TrainingRay, cfg.batch_rays);
    defer gpa.free(batch);
    const bgs = try gpa.alloc([3]f32, cfg.batch_rays);
    defer gpa.free(bgs);
    const jitters = try gpa.alloc(f32, cfg.batch_rays);
    defer gpa.free(jitters);
    const gpu_rays = try gpa.alloc(TrainingRayGpu, cfg.batch_rays);
    defer gpa.free(gpu_rays);

    for (0..cfg.batch_rays) |bi| {
        const d_raw = Vec3.init(rng.float(f32) - 0.5, rng.float(f32) - 0.5, 1.0).normalize();
        batch[bi] = .{
            .origin = .{ (rng.float(f32) - 0.5) * 0.6, (rng.float(f32) - 0.5) * 0.6, -2.5 },
            .dir = .{ @floatCast(d_raw.x), @floatCast(d_raw.y), @floatCast(d_raw.z) },
            .rgb = .{ rng.float(f32), rng.float(f32), rng.float(f32) },
            .depth = if (bi % 2 == 0) 2.5 else null,
        };
        bgs[bi] = .{ 0.2, 0.4, 0.6 };
        jitters[bi] = rng.float(f32);
    }
    packTrainingRays(batch, bgs, jitters, gpu_rays);

    const cpu_g_mlp = try gpa.alloc(f32, model.mlp_params.len);
    defer gpa.free(cpu_g_mlp);
    const cpu_g_hash = try gpa.alloc(f32, model.hash_grid.params.len);
    defer gpa.free(cpu_g_hash);
    @memset(cpu_g_mlp, 0.0);
    @memset(cpu_g_hash, 0.0);

    var cpu_losses: [8]f32 = undefined;
    for (batch, 0..) |r, bi| {
        cpu_losses[bi] = model.rayLossAndBackward(r.origin, r.dir, r.rgb, r.depth, bgs[bi], jitters[bi], true, cpu_g_mlp, cpu_g_hash);
    }

    const gpu_res = try gpa.alloc(RayForwardResultGpu, cfg.batch_rays);
    defer gpa.free(gpu_res);
    const gpu_g_mlp = try gpa.alloc(f32, model.mlp_params.len);
    defer gpa.free(gpu_g_mlp);
    const gpu_g_hash = try gpa.alloc(f32, model.hash_grid.params.len);
    defer gpa.free(gpu_g_hash);

    try session.forwardBackwardOnly(&model, gpu_rays, true, gpu_res, gpu_g_mlp, gpu_g_hash);

    for (cpu_losses, gpu_res) |cl, gr| {
        try std.testing.expectApproxEqAbs(cl, gr.d_rgb_loss[3], 1e-4);
    }

    var max_mlp_diff: f32 = 0.0;
    for (cpu_g_mlp, gpu_g_mlp) |cg, gg| {
        max_mlp_diff = @max(max_mlp_diff, @abs(cg - gg));
    }
    var max_hash_diff: f32 = 0.0;
    for (cpu_g_hash, gpu_g_hash) |cg, gg| {
        max_hash_diff = @max(max_hash_diff, @abs(cg - gg));
    }
    try std.testing.expect(max_mlp_diff < 2e-4);
    try std.testing.expect(max_hash_diff < 2e-4);
}

test "GPU NeRF single Adam step, occupancy update, and view render match CPU" {
    const gpa = std.testing.allocator;
    var ctx = try NerfGpuContext.initStandalone();
    defer ctx.deinit();

    const cfg = nerf.NerfConfig{
        .num_levels = 4,
        .table_log2 = 10,
        .n_min = 4.0,
        .n_max = 32.0,
        .geo_hidden = 16,
        .geo_feat_dim = 7,
        .col_hidden = 16,
        .batch_rays = 8,
        .max_ray_samples = 32,
        .occ_grid_res = 8,
        .depth_loss_weight = 0.0,
        .use_contraction = true,
    };
    const bounds = nerf.SceneBounds{
        .center = .{ 0.0, 0.0, 0.0 },
        .inv_radius = 0.5,
        .use_contraction = true,
        .t_near = 0.2,
        .t_far = 6.0,
    };

    var cpu_model = try nerf.NerfModel.init(gpa, cfg, bounds);
    defer cpu_model.deinit(gpa);
    for (cpu_model.hash_grid.params, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.03;
    }

    var gpu_model = try nerf.NerfModel.init(gpa, cfg, bounds);
    defer gpu_model.deinit(gpa);
    @memcpy(gpu_model.hash_grid.params, cpu_model.hash_grid.params);
    @memcpy(gpu_model.mlp_params, cpu_model.mlp_params);

    var session = try NerfGpuSession.initFromModel(&ctx, &gpu_model, cfg.batch_rays);
    defer session.deinit();

    const batch = try gpa.alloc(nerf.TrainingRay, cfg.batch_rays);
    defer gpa.free(batch);
    const bgs = try gpa.alloc([3]f32, cfg.batch_rays);
    defer gpa.free(bgs);
    const jitters = try gpa.alloc(f32, cfg.batch_rays);
    defer gpa.free(jitters);
    const gpu_rays = try gpa.alloc(TrainingRayGpu, cfg.batch_rays);
    defer gpa.free(gpu_rays);

    for (0..cfg.batch_rays) |bi| {
        const fi: f32 = @floatFromInt(bi);
        const dir = Vec3.init((fi - 3.5) * 0.08, 0.05, 1.0).normalize();
        batch[bi] = .{
            .origin = .{ 0.0, 0.0, -2.2 },
            .dir = .{ @floatCast(dir.x), @floatCast(dir.y), @floatCast(dir.z) },
            .rgb = .{ 0.8, 0.2, 0.3 },
            .depth = null,
        };
        bgs[bi] = cfg.bg_color;
        jitters[bi] = 0.5;
    }
    packTrainingRays(batch, bgs, jitters, gpu_rays);

    const grad_mlp = try gpa.alloc(f32, cpu_model.mlp_params.len);
    defer gpa.free(grad_mlp);
    const grad_hash = try gpa.alloc(f32, cpu_model.hash_grid.params.len);
    defer gpa.free(grad_hash);
    @memset(grad_mlp, 0.0);
    @memset(grad_hash, 0.0);
    for (batch, 0..) |r, bi| {
        _ = cpu_model.rayLossAndBackward(r.origin, r.dir, r.rgb, r.depth, bgs[bi], jitters[bi], true, grad_mlp, grad_hash);
    }
    const inv_b: f32 = 1.0 / @as(f32, @floatFromInt(cfg.batch_rays));
    for (grad_mlp) |*g| g.* *= inv_b;
    for (grad_hash) |*g| g.* *= inv_b;
    cpu_model.applyGradients(grad_mlp, grad_hash, 1.0);
    cpu_model.occ_grid.updateFromModel(&cpu_model, cfg.occ_decay, cfg.occ_threshold);

    try session.submitTrainStep(&gpu_model, gpu_rays, 1, 1.0, true, true);
    try session.syncToCpu(&gpu_model, 1);

    var max_sig_w_diff: f32 = 0.0;
    var checked_sig: usize = 0;
    for (cpu_model.mlp_params, gpu_model.mlp_params, grad_mlp) |cw, gw, g| {
        if (@abs(g) >= 5e-4) {
            max_sig_w_diff = @max(max_sig_w_diff, @abs(cw - gw));
            checked_sig += 1;
        }
    }
    try std.testing.expect(checked_sig > 20);
    try std.testing.expect(max_sig_w_diff < 2e-4);

    for (cpu_model.occ_grid.occupied, gpu_model.occ_grid.occupied) |co, go| {
        try std.testing.expectEqual(co, go);
    }

    const k = Mat3.fromRows(
        .{ 40.0, 0.0, 16.0 },
        .{ 0.0, 40.0, 12.0 },
        .{ 0.0, 0.0, 1.0 },
    );
    const pose = CameraPose{ .r = Mat3.identity, .t = Vec3.init(0.0, 0.0, 2.2) };
    var cpu_view = try cpu_model.renderView(gpa, 32, 24, k, pose, true);
    defer cpu_view.deinit(gpa);
    var gpu_view = try session.renderViewGpu(gpa, &gpu_model, 32, 24, k, pose, true);
    defer gpu_view.deinit(gpa);

    var max_px_diff: u32 = 0;
    for (cpu_view.rgb, gpu_view.rgb) |c_px, g_px| {
        const d = @abs(@as(i32, c_px) - @as(i32, g_px));
        if (d > max_px_diff) max_px_diff = d;
    }
    try std.testing.expect(max_px_diff <= 2);
}

test "GPU trainOnDatasetGpu overfits synthetic 3D scene rapidly" {
    const gpa = std.testing.allocator;
    var ctx = try NerfGpuContext.initStandalone();
    defer ctx.deinit();

    const w: u32 = 36;
    const h: u32 = 36;
    const k = Mat3.fromRows(
        .{ 38.0, 0.0, 18.0 },
        .{ 0.0, 38.0, 18.0 },
        .{ 0.0, 0.0, 1.0 },
    );

    const n_cams: usize = 10;
    const target = Vec3.zero;
    const sphere_c = Vec3.zero;
    const sphere_r: f64 = 0.72;
    const up = Vec3.init(0.0, 1.0, 0.0);

    var poses: [n_cams]CameraPose = undefined;
    var images: [n_cams]ImageData = undefined;
    var reg_train: [n_cams]bool = undefined;
    for (0..n_cams) |ci| {
        const ang = (@as(f64, @floatFromInt(ci)) / @as(f64, @floatFromInt(n_cams))) * std.math.tau;
        const cam_c = Vec3.init(2.8 * @cos(ang), 0.35 * @sin(ang * 2.0), 2.8 * @sin(ang));
        const fwd = target.sub(cam_c).normalize();
        const right = up.cross(fwd).normalize();
        const true_up = fwd.cross(right).normalize();
        const r_mat = Mat3.fromRows(
            .{ right.x, right.y, right.z },
            .{ true_up.x, true_up.y, true_up.z },
            .{ fwd.x, fwd.y, fwd.z },
        );
        const t_vec = r_mat.mulVec(cam_c).neg();
        poses[ci] = .{ .r = r_mat, .t = t_vec };
        reg_train[ci] = (ci != 5);

        const rgb = try gpa.alloc(u8, @as(usize, w) * h * 3);
        const gray = try gpa.alloc(u8, @as(usize, w) * h);
        for (0..h) |y| {
            for (0..w) |x| {
                const ray = nerf.generateCameraRay(k, poses[ci], @as(f64, @floatFromInt(x)) + 0.5, @as(f64, @floatFromInt(y)) + 0.5);
                const ro = Vec3.init(ray.origin[0], ray.origin[1], ray.origin[2]);
                const rd = Vec3.init(ray.dir[0], ray.dir[1], ray.dir[2]);
                const oc = ro.sub(sphere_c);
                const b_v = oc.dot(rd);
                const c_v = oc.dot(oc) - sphere_r * sphere_r;
                const disc = b_v * b_v - c_v;
                var col = [3]f64{ 0.78, 0.82, 0.88 };
                if (disc > 0.0) {
                    const hit_t = -b_v - @sqrt(disc);
                    if (hit_t > 0.0) {
                        const p = ro.add(rd.scale(hit_t));
                        const n = p.sub(sphere_c).normalize();
                        col = .{
                            0.2 + 0.75 * std.math.clamp(0.5 * n.x + 0.5, 0.0, 1.0),
                            0.2 + 0.75 * std.math.clamp(0.5 * n.y + 0.5, 0.0, 1.0),
                            0.2 + 0.75 * std.math.clamp(0.5 * n.z + 0.5, 0.0, 1.0),
                        };
                    }
                }
                const pidx = y * w + x;
                for (0..3) |ch| {
                    rgb[pidx * 3 + ch] = @intFromFloat(std.math.clamp(col[ch] * 255.0 + 0.5, 0.0, 255.0));
                }
                gray[pidx] = rgb[pidx * 3 + 1];
            }
        }
        images[ci] = .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
    }
    defer for (images) |im| {
        gpa.free(im.rgb);
        gpa.free(im.gray);
    };

    const sfm_pts = [_]Vec3{
        Vec3.init(0.7, 0.0, 0.0),
        Vec3.init(-0.7, 0.0, 0.0),
        Vec3.init(0.0, 0.7, 0.0),
        Vec3.init(0.0, -0.7, 0.0),
        Vec3.init(0.0, 0.0, -0.7),
    };

    const cfg = nerf.NerfConfig{
        .num_levels = 6,
        .table_log2 = 13,
        .n_min = 4.0,
        .n_max = 64.0,
        .geo_hidden = 32,
        .geo_feat_dim = 15,
        .col_hidden = 32,
        .num_steps = 300,
        .batch_rays = 512,
        .max_ray_samples = 160,
        .lr = 1.5e-2,
        .occ_grid_res = 24,
        .occ_update_interval = 32,
        .use_contraction = false,
        .depth_loss_weight = 0.0,
    };

    var model = try trainOnDatasetGpu(gpa, &ctx, k, &images, &poses, &reg_train, &sfm_pts, cfg, null);
    defer model.deinit(gpa);

    var session = try NerfGpuSession.initFromModel(&ctx, &model, 1);
    defer session.deinit();
    var view = try session.renderViewGpu(gpa, &model, w, h, k, poses[5], true);
    defer view.deinit(gpa);

    const metrics = render.computeViewMetrics(view, images[5]);
    try std.testing.expect(metrics.psnr_db >= 26.0);
}
