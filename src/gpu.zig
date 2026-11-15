const std = @import("std");
const c = @import("c_wgpu");
const features = @import("features.zig");

const MatchResultGpu = extern struct {
    best_idx: u32,
    d1: u32,
    d2: u32,
    _pad: u32,
};

const MatchParamsUniform = extern struct {
    q_offset: u32,
    q_count: u32,
    t_offset: u32,
    t_count: u32,
    out_offset: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const HarrisParamsUniform = extern struct {
    width: u32,
    height: u32,
    min_score: f32,
    _pad: u32 = 0,
};

const match_wgsl =
    \\struct MatchParams {
    \\    q_offset: u32,
    \\    q_count: u32,
    \\    t_offset: u32,
    \\    t_count: u32,
    \\    out_offset: u32,
    \\    _pad0: u32,
    \\    _pad1: u32,
    \\    _pad2: u32,
    \\};
    \\
    \\@group(0) @binding(0) var<storage, read> q_descs: array<vec4<u32>>;
    \\@group(0) @binding(1) var<storage, read> t_descs: array<vec4<u32>>;
    \\@group(0) @binding(2) var<storage, read_write> out_best: array<vec4<u32>>;
    \\@group(0) @binding(3) var<uniform> params: MatchParams;
    \\
    \\var<workgroup> tile_t: array<vec4<u32>, 128>;
    \\
    \\@compute @workgroup_size(64, 1, 1)
    \\fn main(
    \\    @builtin(global_invocation_id) gid: vec3<u32>,
    \\    @builtin(local_invocation_id) lid: vec3<u32>,
    \\) {
    \\    let tid = lid.x;
    \\    let q_local = gid.x;
    \\    let valid_q = q_local < params.q_count;
    \\    var q0 = vec4<u32>(0u);
    \\    var q1 = vec4<u32>(0u);
    \\    if (valid_q) {
    \\        let base = (params.q_offset + q_local) * 2u;
    \\        q0 = q_descs[base];
    \\        q1 = q_descs[base + 1u];
    \\    }
    \\
    \\    var d1: u32 = 0xFFFFFFFFu;
    \\    var d2: u32 = 0xFFFFFFFFu;
    \\    var best_idx: u32 = 0u;
    \\
    \\    let t_count = params.t_count;
    \\    for (var tile_start: u32 = 0u; tile_start < t_count; tile_start += 64u) {
    \\        let load_t = tile_start + tid;
    \\        if (load_t < t_count) {
    \\            let tbase = (params.t_offset + load_t) * 2u;
    \\            tile_t[tid * 2u] = t_descs[tbase];
    \\            tile_t[tid * 2u + 1u] = t_descs[tbase + 1u];
    \\        }
    \\        workgroupBarrier();
    \\
    \\        if (valid_q) {
    \\            let tile_end = min(64u, t_count - tile_start);
    \\            for (var j: u32 = 0u; j < tile_end; j++) {
    \\                let t0 = tile_t[j * 2u];
    \\                let t1 = tile_t[j * 2u + 1u];
    \\                let x0 = countOneBits(q0 ^ t0);
    \\                let x1 = countOneBits(q1 ^ t1);
    \\                let dist = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
    \\                if (dist < d1) {
    \\                    d2 = d1;
    \\                    d1 = dist;
    \\                    best_idx = tile_start + j;
    \\                } else if (dist < d2) {
    \\                    d2 = dist;
    \\                }
    \\            }
    \\        }
    \\        workgroupBarrier();
    \\    }
    \\
    \\    if (valid_q) {
    \\        out_best[params.out_offset + q_local] = vec4<u32>(best_idx, d1, d2, 0u);
    \\    }
    \\}
;

const harris_wgsl =
    \\struct HarrisParams {
    \\    width: u32,
    \\    height: u32,
    \\    min_score: f32,
    \\    _pad: u32,
    \\};
    \\
    \\@group(0) @binding(0) var<storage, read> src_gray: array<u32>;
    \\@group(0) @binding(1) var<storage, read_write> tmp_buf: array<u32>;
    \\@group(0) @binding(2) var<storage, read_write> blurred_out: array<u32>;
    \\@group(0) @binding(3) var<storage, read_write> score_out: array<f32>;
    \\@group(0) @binding(4) var<uniform> params: HarrisParams;
    \\
    \\fn read_src_byte(x: u32, y: u32) -> u32 {
    \\    let idx = y * params.width + x;
    \\    return (src_gray[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    \\}
    \\
    \\fn read_blurred_byte(x: u32, y: u32) -> u32 {
    \\    let idx = y * params.width + x;
    \\    return (blurred_out[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    \\}
    \\
    \\@compute @workgroup_size(64, 1, 1)
    \\fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\    let x = gid.x;
    \\    let y = gid.y;
    \\    if (x >= params.width || y >= params.height) {
    \\        return;
    \\    }
    \\    let max_x = params.width - 1u;
    \\    let xm2 = select(0u, x - 2u, x >= 2u);
    \\    let xm1 = select(0u, x - 1u, x >= 1u);
    \\    let xp1 = min(max_x, x + 1u);
    \\    let xp2 = min(max_x, x + 2u);
    \\    let sum =
    \\        read_src_byte(xm2, y) +
    \\        4u * read_src_byte(xm1, y) +
    \\        6u * read_src_byte(x, y) +
    \\        4u * read_src_byte(xp1, y) +
    \\        read_src_byte(xp2, y);
    \\    tmp_buf[y * params.width + x] = sum;
    \\}
    \\
    \\@compute @workgroup_size(64, 1, 1)
    \\fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\    let word_idx = gid.x;
    \\    let total_pixels = params.width * params.height;
    \\    let base_idx = word_idx * 4u;
    \\    if (base_idx >= total_pixels) {
    \\        return;
    \\    }
    \\    let max_y = params.height - 1u;
    \\    var packed_word: u32 = 0u;
    \\    for (var k: u32 = 0u; k < 4u; k++) {
    \\        let idx = base_idx + k;
    \\        if (idx < total_pixels) {
    \\            let x = idx % params.width;
    \\            let y = idx / params.width;
    \\            let ym2 = select(0u, y - 2u, y >= 2u);
    \\            let ym1 = select(0u, y - 1u, y >= 1u);
    \\            let yp1 = min(max_y, y + 1u);
    \\            let yp2 = min(max_y, y + 2u);
    \\            let sum =
    \\                tmp_buf[ym2 * params.width + x] +
    \\                4u * tmp_buf[ym1 * params.width + x] +
    \\                6u * tmp_buf[y * params.width + x] +
    \\                4u * tmp_buf[yp1 * params.width + x] +
    \\                tmp_buf[yp2 * params.width + x];
    \\            let val = (sum >> 8u) & 0xFFu;
    \\            packed_word |= (val << (k * 8u));
    \\        }
    \\    }
    \\    blurred_out[word_idx] = packed_word;
    \\}
    \\
    \\var<workgroup> tile_b: array<u32, 400>;
    \\var<workgroup> tile_ix2: array<f32, 324>;
    \\var<workgroup> tile_iy2: array<f32, 324>;
    \\var<workgroup> tile_ixy: array<f32, 324>;
    \\
    \\@compute @workgroup_size(16, 16, 1)
    \\fn harris_score(
    \\    @builtin(workgroup_id) wg_id: vec3<u32>,
    \\    @builtin(local_invocation_id) lid: vec3<u32>,
    \\    @builtin(global_invocation_id) gid: vec3<u32>,
    \\) {
    \\    let tid = lid.y * 16u + lid.x;
    \\    let wg_x0 = i32(wg_id.x * 16u);
    \\    let wg_y0 = i32(wg_id.y * 16u);
    \\    let w_i = i32(params.width);
    \\    let h_i = i32(params.height);
    \\
    \\    var i = tid;
    \\    while (i < 400u) {
    \\        let lx = i32(i % 20u);
    \\        let ly = i32(i / 20u);
    \\        let px = u32(clamp(wg_x0 + lx - 2, 0, w_i - 1));
    \\        let py = u32(clamp(wg_y0 + ly - 2, 0, h_i - 1));
    \\        tile_b[i] = read_blurred_byte(px, py);
    \\        i += 256u;
    \\    }
    \\    workgroupBarrier();
    \\
    \\    i = tid;
    \\    while (i < 324u) {
    \\        let sx = i % 18u;
    \\        let sy = i / 18u;
    \\        let bx = sx + 1u;
    \\        let by = sy + 1u;
    \\        let p00 = f32(tile_b[(by - 1u) * 20u + (bx - 1u)]);
    \\        let p01 = f32(tile_b[(by - 1u) * 20u + bx]);
    \\        let p02 = f32(tile_b[(by - 1u) * 20u + (bx + 1u)]);
    \\        let p10 = f32(tile_b[by * 20u + (bx - 1u)]);
    \\        let p12 = f32(tile_b[by * 20u + (bx + 1u)]);
    \\        let p20 = f32(tile_b[(by + 1u) * 20u + (bx - 1u)]);
    \\        let p21 = f32(tile_b[(by + 1u) * 20u + bx]);
    \\        let p22 = f32(tile_b[(by + 1u) * 20u + (bx + 1u)]);
    \\        let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
    \\        let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;
    \\        tile_ix2[i] = gx * gx;
    \\        tile_iy2[i] = gy * gy;
    \\        tile_ixy[i] = gx * gy;
    \\        i += 256u;
    \\    }
    \\    workgroupBarrier();
    \\
    \\    let x = gid.x;
    \\    let y = gid.y;
    \\    if (x >= params.width || y >= params.height) {
    \\        return;
    \\    }
    \\    if (x < 10u || x >= params.width - 10u || y < 10u || y >= params.height - 10u) {
    \\        score_out[y * params.width + x] = 0.0;
    \\        return;
    \\    }
    \\
    \\    var sxx: f32 = 0.0;
    \\    var syy: f32 = 0.0;
    \\    var sxy: f32 = 0.0;
    \\    for (var dy: u32 = 0u; dy < 3u; dy++) {
    \\        let ry = lid.y + dy;
    \\        for (var dx: u32 = 0u; dx < 3u; dx++) {
    \\            let rx = lid.x + dx;
    \\            let tidx = ry * 18u + rx;
    \\            sxx += tile_ix2[tidx];
    \\            syy += tile_iy2[tidx];
    \\            sxy += tile_ixy[tidx];
    \\        }
    \\    }
    \\    let det_m = sxx * syy - sxy * sxy;
    \\    let trace_m = sxx + syy;
    \\    let r = det_m - 0.04 * trace_m * trace_m;
    \\    score_out[y * params.width + x] = select(0.0, r, r > params.min_score);
    \\}
;

pub const GpuContext = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    match_bgl: c.WGPUBindGroupLayout,
    match_pl: c.WGPUPipelineLayout,
    match_pipeline: c.WGPUComputePipeline,
    harris_bgl: c.WGPUBindGroupLayout,
    harris_pl: c.WGPUPipelineLayout,
    blur_h_pipeline: c.WGPUComputePipeline,
    blur_v_pipeline: c.WGPUComputePipeline,
    harris_pipeline: c.WGPUComputePipeline,
    adapter_name_buf: [256]u8 = undefined,
    adapter_name_len: usize = 0,
    backend_type: u32 = 0,
    adapter_type: u32 = 0,

    pub fn adapterName(self: *const GpuContext) []const u8 {
        return self.adapter_name_buf[0..self.adapter_name_len];
    }

    pub fn init() !GpuContext {
        const inst_desc = std.mem.zeroes(c.WGPUInstanceDescriptor);
        const instance = c.wgpuCreateInstance(&inst_desc) orelse return error.GpuInitFailed;
        errdefer c.wgpuInstanceRelease(instance);

        const AdapterState = struct {
            done: bool = false,
            status: c.WGPURequestAdapterStatus = c.WGPURequestAdapterStatus_Unknown,
            adapter: c.WGPUAdapter = null,
        };
        var adapter_state = AdapterState{};
        const adapter_cb = struct {
            fn cb(
                status: c.WGPURequestAdapterStatus,
                adapter: c.WGPUAdapter,
                _: c.WGPUStringView,
                userdata1: ?*anyopaque,
                _: ?*anyopaque,
            ) callconv(.c) void {
                const st: *AdapterState = @ptrCast(@alignCast(userdata1.?));
                st.status = status;
                st.adapter = adapter;
                st.done = true;
            }
        }.cb;

        var req_opts = std.mem.zeroes(c.WGPURequestAdapterOptions);
        req_opts.powerPreference = c.WGPUPowerPreference_HighPerformance;
        req_opts.backendType = c.WGPUBackendType_Vulkan;
        _ = c.wgpuInstanceRequestAdapter(instance, &req_opts, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = adapter_cb,
            .userdata1 = &adapter_state,
            .userdata2 = null,
        });
        while (!adapter_state.done) c.wgpuInstanceProcessEvents(instance);

        if (adapter_state.status != c.WGPURequestAdapterStatus_Success or adapter_state.adapter == null) {
            return error.GpuAdapterUnavailable;
        }
        const adapter = adapter_state.adapter;
        errdefer c.wgpuAdapterRelease(adapter);

        var name_buf: [256]u8 = undefined;
        var name_len: usize = 0;
        var info = std.mem.zeroes(c.WGPUAdapterInfo);
        if (c.wgpuAdapterGetInfo(adapter, &info) == c.WGPUStatus_Success) {
            defer c.wgpuAdapterInfoFreeMembers(info);
            if (info.device.data != null and info.device.length > 0) {
                name_len = @min(name_buf.len, info.device.length);
                @memcpy(name_buf[0..name_len], info.device.data[0..name_len]);
            }
        }

        const DeviceState = struct {
            done: bool = false,
            status: c.WGPURequestDeviceStatus = c.WGPURequestDeviceStatus_Unknown,
            device: c.WGPUDevice = null,
        };
        var device_state = DeviceState{};
        const device_cb = struct {
            fn cb(
                status: c.WGPURequestDeviceStatus,
                device: c.WGPUDevice,
                _: c.WGPUStringView,
                userdata1: ?*anyopaque,
                _: ?*anyopaque,
            ) callconv(.c) void {
                const st: *DeviceState = @ptrCast(@alignCast(userdata1.?));
                st.status = status;
                st.device = device;
                st.done = true;
            }
        }.cb;

        var dev_desc = std.mem.zeroes(c.WGPUDeviceDescriptor);
        dev_desc.label = wgpuStr("sfm_gpu_device");
        dev_desc.defaultQueue.label = wgpuStr("sfm_gpu_queue");
        dev_desc.deviceLostCallbackInfo = .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = null,
            .userdata1 = null,
            .userdata2 = null,
        };
        _ = c.wgpuAdapterRequestDevice(adapter, &dev_desc, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = device_cb,
            .userdata1 = &device_state,
            .userdata2 = null,
        });
        while (!device_state.done) c.wgpuInstanceProcessEvents(instance);

        if (device_state.status != c.WGPURequestDeviceStatus_Success or device_state.device == null) {
            return error.GpuDeviceFailed;
        }
        const device = device_state.device;
        errdefer c.wgpuDeviceRelease(device);
        const queue = c.wgpuDeviceGetQueue(device) orelse return error.GpuQueueFailed;
        errdefer c.wgpuQueueRelease(queue);

        const match_sm = createShaderModule(device, "match_shader", match_wgsl) orelse return error.GpuShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(match_sm);

        var match_entries = [4]c.WGPUBindGroupLayoutEntry{
            bufferLayoutEntry(0, c.WGPUBufferBindingType_ReadOnlyStorage, false),
            bufferLayoutEntry(1, c.WGPUBufferBindingType_ReadOnlyStorage, false),
            bufferLayoutEntry(2, c.WGPUBufferBindingType_Storage, false),
            bufferLayoutEntry(3, c.WGPUBufferBindingType_Uniform, true),
        };
        const match_bgl = c.wgpuDeviceCreateBindGroupLayout(device, &.{
            .nextInChain = null,
            .label = wgpuStr("match_bgl"),
            .entryCount = match_entries.len,
            .entries = &match_entries,
        }) orelse return error.GpuPipelineFailed;
        errdefer c.wgpuBindGroupLayoutRelease(match_bgl);

        const match_pl = createPipelineLayout(device, "match_pl", match_bgl) orelse return error.GpuPipelineFailed;
        errdefer c.wgpuPipelineLayoutRelease(match_pl);
        const match_pipeline = createComputePipeline(device, "match_pipe", match_pl, match_sm, "main") orelse return error.GpuPipelineFailed;
        errdefer c.wgpuComputePipelineRelease(match_pipeline);

        const harris_sm = createShaderModule(device, "harris_shader", harris_wgsl) orelse return error.GpuShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(harris_sm);

        var harris_entries = [5]c.WGPUBindGroupLayoutEntry{
            bufferLayoutEntry(0, c.WGPUBufferBindingType_ReadOnlyStorage, false),
            bufferLayoutEntry(1, c.WGPUBufferBindingType_Storage, false),
            bufferLayoutEntry(2, c.WGPUBufferBindingType_Storage, false),
            bufferLayoutEntry(3, c.WGPUBufferBindingType_Storage, false),
            bufferLayoutEntry(4, c.WGPUBufferBindingType_Uniform, false),
        };
        const harris_bgl = c.wgpuDeviceCreateBindGroupLayout(device, &.{
            .nextInChain = null,
            .label = wgpuStr("harris_bgl"),
            .entryCount = harris_entries.len,
            .entries = &harris_entries,
        }) orelse return error.GpuPipelineFailed;
        errdefer c.wgpuBindGroupLayoutRelease(harris_bgl);

        const harris_pl = createPipelineLayout(device, "harris_pl", harris_bgl) orelse return error.GpuPipelineFailed;
        errdefer c.wgpuPipelineLayoutRelease(harris_pl);
        const blur_h_pipeline = createComputePipeline(device, "blur_h_pipe", harris_pl, harris_sm, "blur_h") orelse return error.GpuPipelineFailed;
        errdefer c.wgpuComputePipelineRelease(blur_h_pipeline);
        const blur_v_pipeline = createComputePipeline(device, "blur_v_pipe", harris_pl, harris_sm, "blur_v") orelse return error.GpuPipelineFailed;
        errdefer c.wgpuComputePipelineRelease(blur_v_pipeline);
        const harris_pipeline = createComputePipeline(device, "harris_pipe", harris_pl, harris_sm, "harris_score") orelse return error.GpuPipelineFailed;
        errdefer c.wgpuComputePipelineRelease(harris_pipeline);

        var ctx: GpuContext = .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
            .match_bgl = match_bgl,
            .match_pl = match_pl,
            .match_pipeline = match_pipeline,
            .harris_bgl = harris_bgl,
            .harris_pl = harris_pl,
            .blur_h_pipeline = blur_h_pipeline,
            .blur_v_pipeline = blur_v_pipeline,
            .harris_pipeline = harris_pipeline,
            .adapter_name_len = name_len,
            .backend_type = @intCast(info.backendType),
            .adapter_type = @intCast(info.adapterType),
        };
        if (name_len > 0) @memcpy(ctx.adapter_name_buf[0..name_len], name_buf[0..name_len]);
        return ctx;
    }

    pub fn deinit(self: *GpuContext) void {
        c.wgpuComputePipelineRelease(self.harris_pipeline);
        c.wgpuComputePipelineRelease(self.blur_v_pipeline);
        c.wgpuComputePipelineRelease(self.blur_h_pipeline);
        c.wgpuPipelineLayoutRelease(self.harris_pl);
        c.wgpuBindGroupLayoutRelease(self.harris_bgl);
        c.wgpuComputePipelineRelease(self.match_pipeline);
        c.wgpuPipelineLayoutRelease(self.match_pl);
        c.wgpuBindGroupLayoutRelease(self.match_bgl);
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuInstanceRelease(self.instance);
        self.* = undefined;
    }

    pub const HarrisGpuOutput = struct {
        blurred: []u8,
        score: []f32,
    };

    pub fn computeHarrisScores(
        self: *GpuContext,
        allocator: std.mem.Allocator,
        gray: []const u8,
        w: u32,
        h: u32,
        min_harris_score: f32,
    ) !HarrisGpuOutput {
        const n = @as(usize, w) * @as(usize, h);
        const packed_bytes = alignUp(n, 4);
        const tmp_bytes = n * @sizeOf(u32);
        const score_bytes = n * @sizeOf(f32);

        const src_buf = try createBuffer(self.device, packed_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
        defer releaseBuffer(src_buf);
        const tmp_buf = try createBuffer(self.device, tmp_bytes, c.WGPUBufferUsage_Storage);
        defer releaseBuffer(tmp_buf);
        const blurred_buf = try createBuffer(self.device, packed_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
        defer releaseBuffer(blurred_buf);
        const score_buf = try createBuffer(self.device, score_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
        defer releaseBuffer(score_buf);
        const uniform_buf = try createBuffer(self.device, @sizeOf(HarrisParamsUniform), c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
        defer releaseBuffer(uniform_buf);
        const stage_blurred = try createBuffer(self.device, packed_bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        defer releaseBuffer(stage_blurred);
        const stage_score = try createBuffer(self.device, score_bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        defer releaseBuffer(stage_score);

        // Pad gray tail up to 4-byte multiple cleanly when uploading
        if (packed_bytes == n) {
            c.wgpuQueueWriteBuffer(self.queue, src_buf, 0, gray.ptr, n);
        } else {
            const aligned_src = try allocator.alloc(u8, packed_bytes);
            defer allocator.free(aligned_src);
            @memcpy(aligned_src[0..n], gray);
            @memset(aligned_src[n..], 0);
            c.wgpuQueueWriteBuffer(self.queue, src_buf, 0, aligned_src.ptr, packed_bytes);
        }

        const udata = HarrisParamsUniform{ .width = w, .height = h, .min_score = min_harris_score };
        c.wgpuQueueWriteBuffer(self.queue, uniform_buf, 0, &udata, @sizeOf(HarrisParamsUniform));

        var bg_entries = [5]c.WGPUBindGroupEntry{
            bufEntry(0, src_buf, 0, packed_bytes),
            bufEntry(1, tmp_buf, 0, tmp_bytes),
            bufEntry(2, blurred_buf, 0, packed_bytes),
            bufEntry(3, score_buf, 0, score_bytes),
            bufEntry(4, uniform_buf, 0, @sizeOf(HarrisParamsUniform)),
        };
        const bg = c.wgpuDeviceCreateBindGroup(self.device, &.{
            .nextInChain = null,
            .label = wgpuStr("harris_bg"),
            .layout = self.harris_bgl,
            .entryCount = bg_entries.len,
            .entries = &bg_entries,
        }) orelse return error.GpuBindGroupFailed;
        defer c.wgpuBindGroupRelease(bg);

        const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
        const enc = c.wgpuDeviceCreateCommandEncoder(self.device, &enc_desc) orelse return error.GpuEncoderFailed;
        defer c.wgpuCommandEncoderRelease(enc);

        var pass_desc = std.mem.zeroes(c.WGPUComputePassDescriptor);
        const pass = c.wgpuCommandEncoderBeginComputePass(enc, &pass_desc) orelse return error.GpuComputePassFailed;
        c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);

        c.wgpuComputePassEncoderSetPipeline(pass, self.blur_h_pipeline);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, (w + 63) / 64, h, 1);

        c.wgpuComputePassEncoderSetPipeline(pass, self.blur_v_pipeline);
        const num_words: u32 = @intCast((n + 3) / 4);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, (num_words + 63) / 64, 1, 1);

        c.wgpuComputePassEncoderSetPipeline(pass, self.harris_pipeline);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, (w + 15) / 16, (h + 15) / 16, 1);

        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);

        c.wgpuCommandEncoderCopyBufferToBuffer(enc, blurred_buf, 0, stage_blurred, 0, packed_bytes);
        c.wgpuCommandEncoderCopyBufferToBuffer(enc, score_buf, 0, stage_score, 0, score_bytes);

        const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
        const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
        defer c.wgpuCommandBufferRelease(cmd);
        c.wgpuQueueSubmit(self.queue, 1, &cmd);

        const blurred = try allocator.alloc(u8, n);
        errdefer allocator.free(blurred);
        const score = try allocator.alloc(f32, n);
        errdefer allocator.free(score);

        try mapReadCopy(self.instance, self.device, stage_blurred, packed_bytes, blurred);
        try mapReadCopy(self.instance, self.device, stage_score, score_bytes, std.mem.sliceAsBytes(score));
        return .{ .blurred = blurred, .score = score };
    }

    pub fn matchDescriptors(
        self: *GpuContext,
        allocator: std.mem.Allocator,
        kps1: []const features.Keypoint,
        kps2: []const features.Keypoint,
        ratio_thresh: f64,
        max_hamming: u32,
    ) ![]features.Match {
        if (kps1.len == 0 or kps2.len < 2) return try allocator.alloc(features.Match, 0);
        const all_kps = [_][]const features.Keypoint{ kps1, kps2 };
        const pairs = [_][2]usize{.{ 0, 1 }};
        const batch = try self.matchDescriptorPairsBatch(allocator, &all_kps, &pairs, ratio_thresh, max_hamming);
        defer allocator.free(batch);
        return batch[0];
    }

    /// Uploads all cameras' descriptors once and executes batched mutual Lowe-ratio matching across all pairs on GPU.
    pub fn matchDescriptorPairsBatch(
        self: *GpuContext,
        allocator: std.mem.Allocator,
        all_kps: []const []const features.Keypoint,
        pairs: []const [2]usize,
        ratio_thresh: f64,
        max_hamming: u32,
    ) ![][]features.Match {
        const out_lists = try allocator.alloc([]features.Match, pairs.len);
        @memset(out_lists, &.{});
        errdefer {
            for (out_lists) |m| allocator.free(m);
            allocator.free(out_lists);
        }
        if (pairs.len == 0) return out_lists;

        const n_cams = all_kps.len;
        const cam_offsets = try allocator.alloc(u32, n_cams);
        defer allocator.free(cam_offsets);

        var total_kps: usize = 0;
        for (all_kps, 0..) |kps, ci| {
            cam_offsets[ci] = @intCast(total_kps);
            total_kps += kps.len;
        }
        if (total_kps == 0) return out_lists;

        // Pack all descriptors tightly into [8]u32 slots (32 bytes per keypoint)
        const desc_bytes = total_kps * 32;
        const packed_descs = try allocator.alloc([8]u32, total_kps);
        defer allocator.free(packed_descs);
        for (all_kps, 0..) |kps, ci| {
            const off = cam_offsets[ci];
            for (kps, 0..) |kp, ki| {
                packed_descs[off + ki] = @bitCast(kp.desc);
            }
        }

        const desc_buf = try createBuffer(self.device, desc_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
        defer releaseBuffer(desc_buf);
        c.wgpuQueueWriteBuffer(self.queue, desc_buf, 0, packed_descs.ptr, desc_bytes);

        // Chunk pair dispatches so output staging stays within conservative 32MB storage buffer budget
        const max_results_per_chunk: usize = 2_000_000;
        const result_stride = @sizeOf(MatchResultGpu);
        const uniform_stride: usize = 256;

        const host_results = try allocator.alloc(MatchResultGpu, max_results_per_chunk);
        defer allocator.free(host_results);

        var p_start: usize = 0;
        while (p_start < pairs.len) {
            var p_end = p_start;
            var chunk_results: usize = 0;
            var chunk_dispatches: usize = 0;

            while (p_end < pairs.len) {
                const p = pairs[p_end];
                const n1 = all_kps[p[0]].len;
                const n2 = all_kps[p[1]].len;
                if (n1 == 0 or n2 < 2) {
                    p_end += 1;
                    continue;
                }
                const needed = n1 + n2;
                if (chunk_results > 0 and chunk_results + needed > max_results_per_chunk) break;
                if (needed > max_results_per_chunk) return error.GpuBatchTooLarge;
                chunk_results += needed;
                chunk_dispatches += 2;
                p_end += 1;
            }

            if (chunk_dispatches == 0) {
                p_start = p_end;
                continue;
            }

            const out_bytes = chunk_results * result_stride;
            const uniform_bytes = chunk_dispatches * uniform_stride;

            const out_buf = try createBuffer(self.device, out_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
            defer releaseBuffer(out_buf);
            const stage_buf = try createBuffer(self.device, out_bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
            defer releaseBuffer(stage_buf);
            const uniform_buf = try createBuffer(self.device, uniform_bytes, c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
            defer releaseBuffer(uniform_buf);

            const uniform_blob = try allocator.alloc(u8, uniform_bytes);
            defer allocator.free(uniform_blob);
            @memset(uniform_blob, 0);

            const DispatchMeta = struct {
                pair_idx: usize,
                off12: usize,
                off21: usize,
                n1: usize,
                n2: usize,
                dyn12: u32,
                dyn21: u32,
            };
            var metas: std.ArrayList(DispatchMeta) = .empty;
            defer metas.deinit(allocator);

            var cur_res_off: usize = 0;
            var cur_disp: usize = 0;
            for (p_start..p_end) |pi| {
                const pair = pairs[pi];
                const c1 = pair[0];
                const c2 = pair[1];
                const n1 = all_kps[c1].len;
                const n2 = all_kps[c2].len;
                if (n1 == 0 or n2 < 2) continue;

                const dyn12: u32 = @intCast(cur_disp * uniform_stride);
                const p12 = MatchParamsUniform{
                    .q_offset = cam_offsets[c1],
                    .q_count = @intCast(n1),
                    .t_offset = cam_offsets[c2],
                    .t_count = @intCast(n2),
                    .out_offset = @intCast(cur_res_off),
                };
                @memcpy(uniform_blob[dyn12 .. dyn12 + @sizeOf(MatchParamsUniform)], std.mem.asBytes(&p12));

                const dyn21: u32 = @intCast((cur_disp + 1) * uniform_stride);
                const p21 = MatchParamsUniform{
                    .q_offset = cam_offsets[c2],
                    .q_count = @intCast(n2),
                    .t_offset = cam_offsets[c1],
                    .t_count = @intCast(n1),
                    .out_offset = @intCast(cur_res_off + n1),
                };
                @memcpy(uniform_blob[dyn21 .. dyn21 + @sizeOf(MatchParamsUniform)], std.mem.asBytes(&p21));

                try metas.append(allocator, .{
                    .pair_idx = pi,
                    .off12 = cur_res_off,
                    .off21 = cur_res_off + n1,
                    .n1 = n1,
                    .n2 = n2,
                    .dyn12 = dyn12,
                    .dyn21 = dyn21,
                });

                cur_res_off += n1 + n2;
                cur_disp += 2;
            }

            c.wgpuQueueWriteBuffer(self.queue, uniform_buf, 0, uniform_blob.ptr, uniform_bytes);

            var bg_entries = [4]c.WGPUBindGroupEntry{
                bufEntry(0, desc_buf, 0, desc_bytes),
                bufEntry(1, desc_buf, 0, desc_bytes),
                bufEntry(2, out_buf, 0, out_bytes),
                bufEntry(3, uniform_buf, 0, @sizeOf(MatchParamsUniform)),
            };
            const bg = c.wgpuDeviceCreateBindGroup(self.device, &.{
                .nextInChain = null,
                .label = wgpuStr("match_bg"),
                .layout = self.match_bgl,
                .entryCount = bg_entries.len,
                .entries = &bg_entries,
            }) orelse return error.GpuBindGroupFailed;
            defer c.wgpuBindGroupRelease(bg);

            const enc_desc = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
            const enc = c.wgpuDeviceCreateCommandEncoder(self.device, &enc_desc) orelse return error.GpuEncoderFailed;
            defer c.wgpuCommandEncoderRelease(enc);

            var pass_desc = std.mem.zeroes(c.WGPUComputePassDescriptor);
            const pass = c.wgpuCommandEncoderBeginComputePass(enc, &pass_desc) orelse return error.GpuComputePassFailed;
            c.wgpuComputePassEncoderSetPipeline(pass, self.match_pipeline);

            for (metas.items) |m| {
                var off12 = m.dyn12;
                c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 1, &off12);
                c.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast((m.n1 + 63) / 64), 1, 1);

                var off21 = m.dyn21;
                c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 1, &off21);
                c.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast((m.n2 + 63) / 64), 1, 1);
            }

            c.wgpuComputePassEncoderEnd(pass);
            c.wgpuComputePassEncoderRelease(pass);

            c.wgpuCommandEncoderCopyBufferToBuffer(enc, out_buf, 0, stage_buf, 0, out_bytes);
            const cb_desc = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
            const cmd = c.wgpuCommandEncoderFinish(enc, &cb_desc) orelse return error.GpuCommandFinishFailed;
            defer c.wgpuCommandBufferRelease(cmd);
            c.wgpuQueueSubmit(self.queue, 1, &cmd);

            try mapReadCopy(
                self.instance,
                self.device,
                stage_buf,
                out_bytes,
                std.mem.sliceAsBytes(host_results[0..chunk_results]),
            );

            for (metas.items) |m| {
                const res12 = host_results[m.off12 .. m.off12 + m.n1];
                const res21 = host_results[m.off21 .. m.off21 + m.n2];

                var list: std.ArrayList(features.Match) = .empty;
                defer list.deinit(allocator);

                for (res12, 0..) |r12, i| {
                    if (r12.d1 <= max_hamming and @as(f64, @floatFromInt(r12.d1)) < ratio_thresh * @as(f64, @floatFromInt(r12.d2))) {
                        const j: usize = r12.best_idx;
                        if (j < res21.len and res21[j].best_idx == i) {
                            try list.append(allocator, .{
                                .idx1 = i,
                                .idx2 = j,
                                .dist = r12.d1,
                            });
                        }
                    }
                }
                out_lists[m.pair_idx] = try list.toOwnedSlice(allocator);
            }

            p_start = p_end;
        }

        return out_lists;
    }
};

fn wgpuStr(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

inline fn alignUp(n: usize, a: usize) usize {
    return (n + a - 1) & ~(a - 1);
}

fn createShaderModule(device: c.WGPUDevice, label: []const u8, wgsl_src: []const u8) c.WGPUShaderModule {
    var wgsl_desc = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_desc.chain = .{
        .next = null,
        .sType = c.WGPUSType_ShaderSourceWGSL,
    };
    wgsl_desc.code = wgpuStr(wgsl_src);

    var sm_desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    sm_desc.nextInChain = &wgsl_desc.chain;
    sm_desc.label = wgpuStr(label);
    return c.wgpuDeviceCreateShaderModule(device, &sm_desc);
}

fn bufferLayoutEntry(binding: u32, buf_type: c.WGPUBufferBindingType, has_dynamic_offset: bool) c.WGPUBindGroupLayoutEntry {
    var e = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
    e.binding = binding;
    e.visibility = c.WGPUShaderStage_Compute;
    e.buffer = .{
        .nextInChain = null,
        .type = buf_type,
        .hasDynamicOffset = @intFromBool(has_dynamic_offset),
        .minBindingSize = 0,
    };
    return e;
}

fn createPipelineLayout(device: c.WGPUDevice, label: []const u8, bgl: c.WGPUBindGroupLayout) c.WGPUPipelineLayout {
    const bgls = [1]c.WGPUBindGroupLayout{bgl};
    return c.wgpuDeviceCreatePipelineLayout(device, &.{
        .nextInChain = null,
        .label = wgpuStr(label),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgls,
    });
}

fn createComputePipeline(
    device: c.WGPUDevice,
    label: []const u8,
    layout: c.WGPUPipelineLayout,
    module: c.WGPUShaderModule,
    entry: []const u8,
) c.WGPUComputePipeline {
    return c.wgpuDeviceCreateComputePipeline(device, &.{
        .nextInChain = null,
        .label = wgpuStr(label),
        .layout = layout,
        .compute = .{
            .nextInChain = null,
            .module = module,
            .entryPoint = wgpuStr(entry),
            .constantCount = 0,
            .constants = null,
        },
    });
}

fn createBuffer(device: c.WGPUDevice, size: usize, usage: c.WGPUBufferUsage) !c.WGPUBuffer {
    const buf = c.wgpuDeviceCreateBuffer(device, &.{
        .nextInChain = null,
        .label = wgpuStr("sfm_buf"),
        .usage = usage,
        .size = alignUp(@max(size, 4), 4),
        .mappedAtCreation = 0,
    }) orelse return error.GpuBufferAllocFailed;
    return buf;
}

fn releaseBuffer(buf: c.WGPUBuffer) void {
    c.wgpuBufferDestroy(buf);
    c.wgpuBufferRelease(buf);
}

fn bufEntry(binding: u32, buf: c.WGPUBuffer, offset: u64, size: u64) c.WGPUBindGroupEntry {
    var e = std.mem.zeroes(c.WGPUBindGroupEntry);
    e.binding = binding;
    e.buffer = buf;
    e.offset = offset;
    e.size = alignUp(@max(@as(usize, @intCast(size)), 4), 4);
    return e;
}

fn mapReadCopy(
    instance: c.WGPUInstance,
    device: c.WGPUDevice,
    staging: c.WGPUBuffer,
    size: usize,
    dst: []u8,
) !void {
    const MapState = struct {
        done: bool = false,
        status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Unknown,
    };
    var state = MapState{};
    const map_cb = struct {
        fn cb(
            status: c.WGPUMapAsyncStatus,
            _: c.WGPUStringView,
            userdata1: ?*anyopaque,
            _: ?*anyopaque,
        ) callconv(.c) void {
            const st: *MapState = @ptrCast(@alignCast(userdata1.?));
            st.status = status;
            st.done = true;
        }
    }.cb;

    const aligned_size = alignUp(@max(size, 4), 4);
    _ = c.wgpuBufferMapAsync(staging, c.WGPUMapMode_Read, 0, aligned_size, .{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowSpontaneous,
        .callback = map_cb,
        .userdata1 = &state,
        .userdata2 = null,
    });
    _ = c.wgpuDevicePoll(device, 1, null);
    while (!state.done) c.wgpuInstanceProcessEvents(instance);
    if (state.status != c.WGPUMapAsyncStatus_Success) return error.GpuBufferMapFailed;

    const mapped = c.wgpuBufferGetConstMappedRange(staging, 0, aligned_size) orelse {
        c.wgpuBufferUnmap(staging);
        return error.GpuMappedRangeNull;
    };
    const src_bytes: [*]const u8 = @ptrCast(mapped);
    @memcpy(dst, src_bytes[0..dst.len]);
    c.wgpuBufferUnmap(staging);
}

test "WebGPU compute Harris scores and descriptor matcher match CPU reference" {
    const gpa = std.testing.allocator;
    var gpu = try GpuContext.init();
    defer gpu.deinit();

    const w: u32 = 161;
    const h: u32 = 121;
    const gray = try gpa.alloc(u8, @as(usize, w) * @as(usize, h));
    defer gpa.free(gray);

    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();
    for (0..h) |y| {
        for (0..w) |x| {
            const xf: f64 = @floatFromInt(x);
            const yf: f64 = @floatFromInt(y);
            const val = 128.0 + 60.0 * @sin(xf * 0.23) * @cos(yf * 0.19) + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -10, 10)));
            gray[y * w + x] = @intFromFloat(std.math.clamp(val, 0.0, 255.0));
        }
    }

    const cpu_blurred = try features.gaussianBlur5x5(gpa, gray, w, h);
    defer gpa.free(cpu_blurred);
    const cpu_score = try features.computeHarrisScores(gpa, cpu_blurred, w, h, 250.0);
    defer gpa.free(cpu_score);

    const gpu_res = try gpu.computeHarrisScores(gpa, gray, w, h, 250.0);
    defer gpa.free(gpu_res.blurred);
    defer gpa.free(gpu_res.score);

    try std.testing.expectEqualSlices(u8, cpu_blurred, gpu_res.blurred);
    for (cpu_score, gpu_res.score) |cs, gs| {
        if (cs == 0) {
            try std.testing.expectEqual(@as(f32, 0), gs);
        } else {
            try std.testing.expectApproxEqRel(cs, gs, 1e-3);
        }
    }

    const n1: usize = 120;
    const n2: usize = 135;
    const kps1 = try gpa.alloc(features.Keypoint, n1);
    defer gpa.free(kps1);
    const kps2 = try gpa.alloc(features.Keypoint, n2);
    defer gpa.free(kps2);
    const kps3 = try gpa.alloc(features.Keypoint, 90);
    defer gpa.free(kps3);

    for (kps1, 0..) |*kp, i| {
        kp.* = .{
            .x = @floatFromInt(i),
            .y = 10.0,
            .score = 100.0,
            .angle = 0.0,
            .desc = .{ rng.int(u64), rng.int(u64), rng.int(u64), rng.int(u64) },
        };
    }
    for (kps2, 0..) |*kp, j| {
        var d = [4]u64{ rng.int(u64), rng.int(u64), rng.int(u64), rng.int(u64) };
        if (j < 40) {
            d = kps1[j].desc;
            d[0] ^= 0x3;
        }
        kp.* = .{ .x = @floatFromInt(j), .y = 12.0, .score = 100.0, .angle = 0.0, .desc = d };
    }
    for (kps3, 0..) |*kp, k| {
        var d = [4]u64{ rng.int(u64), rng.int(u64), rng.int(u64), rng.int(u64) };
        if (k < 25) {
            d = kps2[k].desc;
            d[1] ^= 0x7;
        }
        kp.* = .{ .x = @floatFromInt(k), .y = 14.0, .score = 100.0, .angle = 0.0, .desc = d };
    }

    const cpu_matches = try features.matchDescriptors(gpa, kps1, kps2, 0.80, 72);
    defer gpa.free(cpu_matches);
    const gpu_matches = try gpu.matchDescriptors(gpa, kps1, kps2, 0.80, 72);
    defer gpa.free(gpu_matches);

    try std.testing.expectEqual(cpu_matches.len, gpu_matches.len);
    for (cpu_matches, gpu_matches) |cm, gm| {
        try std.testing.expectEqual(cm, gm);
    }

    const all_kps = [_][]const features.Keypoint{ kps1, kps2, kps3 };
    const pairs = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } };
    const batch = try gpu.matchDescriptorPairsBatch(gpa, &all_kps, &pairs, 0.80, 72);
    defer {
        for (batch) |m| gpa.free(m);
        gpa.free(batch);
    }
    for (pairs, 0..) |pair, pi| {
        const ref = try features.matchDescriptors(gpa, all_kps[pair[0]], all_kps[pair[1]], 0.80, 72);
        defer gpa.free(ref);
        try std.testing.expectEqual(ref.len, batch[pi].len);
        for (ref, batch[pi]) |rm, bm| {
            try std.testing.expectEqual(rm, bm);
        }
    }
}
