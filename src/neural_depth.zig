const std = @import("std");
const c = @import("c_ort");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const dataset = @import("dataset.zig");
const Color = @import("io_ply.zig").Color;
const mesh_mod = @import("mesh.zig");
const render = @import("render.zig");

pub const NeuralHoleConfig = struct {
    model_path: []const u8,
    num_threads: usize = 0,
    input_max_dim: u32 = 504,
    num_tokens: i64 = 1800,
    max_views: usize = 11,
    mask_thresh: f32 = 0.55,
    harmonic_iters: usize = 96,
    grid_target_nodes_short: u32 = 170,
    max_added_points: usize = 120000,
    max_depth_grad_rel: f32 = 0.14,
};

pub const Prediction = struct {
    width: u32,
    height: u32,
    depth: []f32,
    mask: []f32,
    metric_scale: f32,

    pub fn deinit(self: *Prediction, allocator: std.mem.Allocator) void {
        allocator.free(self.depth);
        allocator.free(self.mask);
        self.* = undefined;
    }
};

pub const NeuralSession = struct {
    api: *const c.OrtApi,
    env: *c.OrtEnv,
    opts: *c.OrtSessionOptions,
    session: *c.OrtSession,
    mem_info: *c.OrtMemoryInfo,
    allocator_ptr: *c.OrtAllocator,
    in_names_z: [][*:0]u8,
    out_names_z: [][*:0]u8,
    num_tokens_ndim: usize,
    zig_alloc: std.mem.Allocator,

    fn checkStatus(api: *const c.OrtApi, status: ?*c.OrtStatus) !void {
        if (status) |st| {
            const msg = api.GetErrorMessage.?(st);
            std.log.err("ONNX Runtime error: {s}", .{std.mem.span(msg)});
            api.ReleaseStatus.?(st);
            return error.OnnxRuntimeError;
        }
    }

    pub fn init(zig_alloc: std.mem.Allocator, model_path: []const u8, num_threads: usize) !NeuralSession {
        const api_base = c.OrtGetApiBase();
        if (api_base == null) return error.OnnxInitFailed;
        const api_c = api_base.*.GetApi.?(c.ORT_API_VERSION);
        if (api_c == null) return error.OnnxApiVersionMismatch;
        const api: *const c.OrtApi = @ptrCast(api_c);

        var env_opt: ?*c.OrtEnv = null;
        try checkStatus(api, api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "sfm_neural", &env_opt));
        const env = env_opt orelse return error.OnnxInitFailed;
        errdefer api.ReleaseEnv.?(env);

        var opts_opt: ?*c.OrtSessionOptions = null;
        try checkStatus(api, api.CreateSessionOptions.?(&opts_opt));
        const opts = opts_opt orelse return error.OnnxInitFailed;
        errdefer api.ReleaseSessionOptions.?(opts);

        const threads: c_int = @intCast(if (num_threads == 0) (std.Thread.getCpuCount() catch 4) else num_threads);
        try checkStatus(api, api.SetIntraOpNumThreads.?(opts, threads));
        try checkStatus(api, api.SetSessionGraphOptimizationLevel.?(opts, c.ORT_ENABLE_ALL));

        const path_z = try zig_alloc.allocSentinel(u8, model_path.len, 0);
        defer zig_alloc.free(path_z);
        @memcpy(path_z, model_path);

        var sess_opt: ?*c.OrtSession = null;
        try checkStatus(api, api.CreateSession.?(env, path_z.ptr, opts, &sess_opt));
        const session = sess_opt orelse return error.OnnxSessionFailed;
        errdefer api.ReleaseSession.?(session);

        var mem_info_opt: ?*c.OrtMemoryInfo = null;
        try checkStatus(api, api.CreateCpuMemoryInfo.?(c.OrtDeviceAllocator, c.OrtMemTypeDefault, &mem_info_opt));
        const mem_info = mem_info_opt orelse return error.OnnxInitFailed;
        errdefer api.ReleaseMemoryInfo.?(mem_info);

        var ort_alloc_opt: ?*c.OrtAllocator = null;
        try checkStatus(api, api.GetAllocatorWithDefaultOptions.?(&ort_alloc_opt));
        const ort_alloc = ort_alloc_opt orelse return error.OnnxInitFailed;

        var in_count: usize = 0;
        var out_count: usize = 0;
        try checkStatus(api, api.SessionGetInputCount.?(session, &in_count));
        try checkStatus(api, api.SessionGetOutputCount.?(session, &out_count));

        const in_names_z = try zig_alloc.alloc([*:0]u8, in_count);
        var in_loaded: usize = 0;
        errdefer {
            for (in_names_z[0..in_loaded]) |s| zig_alloc.free(std.mem.span(s));
            zig_alloc.free(in_names_z);
        }
        for (0..in_count) |i| {
            var c_name: [*c]u8 = null;
            try checkStatus(api, api.SessionGetInputName.?(session, i, ort_alloc, &c_name));
            defer _ = api.AllocatorFree.?(ort_alloc, c_name);
            const span = std.mem.span(@as([*:0]const u8, @ptrCast(c_name)));
            const copy = try zig_alloc.allocSentinel(u8, span.len, 0);
            @memcpy(copy, span);
            in_names_z[i] = copy.ptr;
            in_loaded += 1;
        }

        const out_names_z = try zig_alloc.alloc([*:0]u8, out_count);
        var out_loaded: usize = 0;
        errdefer {
            for (out_names_z[0..out_loaded]) |s| zig_alloc.free(std.mem.span(s));
            zig_alloc.free(out_names_z);
        }
        for (0..out_count) |i| {
            var c_name: [*c]u8 = null;
            try checkStatus(api, api.SessionGetOutputName.?(session, i, ort_alloc, &c_name));
            defer _ = api.AllocatorFree.?(ort_alloc, c_name);
            const span = std.mem.span(@as([*:0]const u8, @ptrCast(c_name)));
            const copy = try zig_alloc.allocSentinel(u8, span.len, 0);
            @memcpy(copy, span);
            out_names_z[i] = copy.ptr;
            out_loaded += 1;
        }

        var num_tokens_ndim: usize = 0;
        if (in_count >= 2) {
            var ti: ?*c.OrtTypeInfo = null;
            try checkStatus(api, api.SessionGetInputTypeInfo.?(session, 1, &ti));
            defer if (ti) |t| api.ReleaseTypeInfo.?(t);
            var t_info: ?*const c.OrtTensorTypeAndShapeInfo = null;
            try checkStatus(api, api.CastTypeInfoToTensorInfo.?(ti, &t_info));
            if (t_info) |tensor_info| {
                try checkStatus(api, api.GetDimensionsCount.?(tensor_info, &num_tokens_ndim));
            }
        }

        return .{
            .api = api,
            .env = env,
            .opts = opts,
            .session = session,
            .mem_info = mem_info,
            .allocator_ptr = ort_alloc,
            .in_names_z = in_names_z,
            .out_names_z = out_names_z,
            .num_tokens_ndim = num_tokens_ndim,
            .zig_alloc = zig_alloc,
        };
    }

    pub fn deinit(self: *NeuralSession) void {
        for (self.in_names_z) |s| self.zig_alloc.free(std.mem.span(s));
        self.zig_alloc.free(self.in_names_z);
        for (self.out_names_z) |s| self.zig_alloc.free(std.mem.span(s));
        self.zig_alloc.free(self.out_names_z);
        self.api.ReleaseMemoryInfo.?(self.mem_info);
        self.api.ReleaseSession.?(self.session);
        self.api.ReleaseSessionOptions.?(self.opts);
        self.api.ReleaseEnv.?(self.env);
        self.* = undefined;
    }

    pub fn predict(
        self: *NeuralSession,
        allocator: std.mem.Allocator,
        rgb: []const u8,
        src_w: u32,
        src_h: u32,
        target_w: u32,
        target_h: u32,
        num_tokens: i64,
    ) !Prediction {
        if (self.in_names_z.len == 0 or self.out_names_z.len == 0) return error.OnnxInvalidModelIo;

        const tw: usize = target_w;
        const th: usize = target_h;
        const plane = tw * th;
        const nchw = try allocator.alloc(f32, 3 * plane);
        defer allocator.free(nchw);

        fillResizedNchw01(rgb, src_w, src_h, target_w, target_h, nchw);

        const img_shape = [4]i64{ 1, 3, @intCast(th), @intCast(tw) };
        var img_val: ?*c.OrtValue = null;
        try checkStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
            self.mem_info,
            nchw.ptr,
            nchw.len * @sizeOf(f32),
            &img_shape,
            img_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &img_val,
        ));
        defer if (img_val) |v| self.api.ReleaseValue.?(v);

        var tok_scalar = num_tokens;
        const tok_shape = [1]i64{1};
        var tok_val: ?*c.OrtValue = null;
        if (self.in_names_z.len >= 2) {
            try checkStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                self.mem_info,
                &tok_scalar,
                @sizeOf(i64),
                if (self.num_tokens_ndim > 0) &tok_shape else &tok_shape,
                @min(self.num_tokens_ndim, 1),
                c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                &tok_val,
            ));
        }
        defer if (tok_val) |v| self.api.ReleaseValue.?(v);

        const in_names = try allocator.alloc([*c]const u8, self.in_names_z.len);
        defer allocator.free(in_names);
        const in_vals = try allocator.alloc(?*const c.OrtValue, self.in_names_z.len);
        defer allocator.free(in_vals);
        for (self.in_names_z, 0..) |nm, idx| in_names[idx] = nm;
        in_vals[0] = img_val;
        if (self.in_names_z.len >= 2) in_vals[1] = tok_val;

        const out_names = try allocator.alloc([*c]const u8, self.out_names_z.len);
        defer allocator.free(out_names);
        const out_vals = try allocator.alloc(?*c.OrtValue, self.out_names_z.len);
        defer {
            for (out_vals) |v_opt| {
                if (v_opt) |v| self.api.ReleaseValue.?(v);
            }
            allocator.free(out_vals);
        }
        for (self.out_names_z, 0..) |nm, idx| {
            out_names[idx] = nm;
            out_vals[idx] = null;
        }

        try checkStatus(self.api, self.api.Run.?(
            self.session,
            null,
            in_names.ptr,
            in_vals.ptr,
            in_vals.len,
            out_names.ptr,
            out_names.len,
            out_vals.ptr,
        ));

        var depth_out: ?[]f32 = null;
        errdefer if (depth_out) |d| allocator.free(d);
        var mask_out: ?[]f32 = null;
        errdefer if (mask_out) |m| allocator.free(m);
        var out_w: u32 = target_w;
        var out_h: u32 = target_h;
        var metric_scale: f32 = 1.0;

        for (self.out_names_z, 0..) |nm_z, oi| {
            const val = out_vals[oi] orelse continue;
            const nm = std.mem.span(nm_z);

            var shape_info_opt: ?*c.OrtTensorTypeAndShapeInfo = null;
            try checkStatus(self.api, self.api.GetTensorTypeAndShape.?(val, &shape_info_opt));
            const shape_info = shape_info_opt orelse continue;
            defer self.api.ReleaseTensorTypeAndShapeInfo.?(shape_info);

            var ndim: usize = 0;
            try checkStatus(self.api, self.api.GetDimensionsCount.?(shape_info, &ndim));
            var dims_buf: [8]i64 = @splat(0);
            const safe_ndim = @min(ndim, dims_buf.len);
            if (safe_ndim > 0) {
                try checkStatus(self.api, self.api.GetDimensions.?(shape_info, &dims_buf, safe_ndim));
            }
            var elem_type: c.ONNXTensorElementDataType = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            try checkStatus(self.api, self.api.GetTensorElementType.?(shape_info, &elem_type));

            var elem_count: usize = 0;
            try checkStatus(self.api, self.api.GetTensorShapeElementCount.?(shape_info, &elem_count));

            var raw_ptr_opt: ?*anyopaque = null;
            try checkStatus(self.api, self.api.GetTensorMutableData.?(val, &raw_ptr_opt));
            const raw_ptr = raw_ptr_opt orelse continue;

            if (std.mem.eql(u8, nm, "points") or (oi == 0 and depth_out == null)) {
                if (elem_type != c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return error.OnnxUnexpectedOutputType;
                const data: [*]const f32 = @ptrCast(@alignCast(raw_ptr));
                // Handle both [1, H, W, 3] (MoGe standard) and [1, 3, H, W]
                var h_out: usize = th;
                var w_out: usize = tw;
                var is_nhwc = true;
                if (safe_ndim == 4 and dims_buf[3] == 3) {
                    h_out = @intCast(dims_buf[1]);
                    w_out = @intCast(dims_buf[2]);
                    is_nhwc = true;
                } else if (safe_ndim == 4 and dims_buf[1] == 3) {
                    h_out = @intCast(dims_buf[2]);
                    w_out = @intCast(dims_buf[3]);
                    is_nhwc = false;
                } else if (safe_ndim == 3) {
                    h_out = @intCast(dims_buf[1]);
                    w_out = @intCast(dims_buf[2]);
                    is_nhwc = false;
                }
                out_w = @intCast(w_out);
                out_h = @intCast(h_out);
                const np = h_out * w_out;
                const d_slice = try allocator.alloc(f32, np);
                if (safe_ndim == 3) {
                    @memcpy(d_slice, data[0..np]);
                } else if (is_nhwc) {
                    for (0..np) |p| d_slice[p] = data[p * 3 + 2];
                } else {
                    @memcpy(d_slice, data[2 * np .. 3 * np]);
                }
                depth_out = d_slice;
            } else if (std.mem.eql(u8, nm, "mask")) {
                const np = @as(usize, out_w) * @as(usize, out_h);
                if (elem_count >= np) {
                    const m_slice = try allocator.alloc(f32, np);
                    if (elem_type == c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                        const data: [*]const f32 = @ptrCast(@alignCast(raw_ptr));
                        @memcpy(m_slice, data[0..np]);
                    } else if (elem_type == c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL or elem_type == c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
                        const bdata: [*]const u8 = @ptrCast(raw_ptr);
                        for (0..np) |p| m_slice[p] = if (bdata[p] != 0) 1.0 else 0.0;
                    } else {
                        @memset(m_slice, 1.0);
                    }
                    mask_out = m_slice;
                }
            } else if (std.mem.eql(u8, nm, "metric_scale") and elem_count >= 1 and elem_type == c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const data: [*]const f32 = @ptrCast(@alignCast(raw_ptr));
                metric_scale = data[0];
            }
        }

        const final_depth = depth_out orelse return error.OnnxMissingDepthOutput;
        const np = @as(usize, out_w) * @as(usize, out_h);
        const final_mask = mask_out orelse blk: {
            const m = try allocator.alloc(f32, np);
            @memset(m, 1.0);
            break :blk m;
        };

        return .{
            .width = out_w,
            .height = out_h,
            .depth = final_depth,
            .mask = final_mask,
            .metric_scale = metric_scale,
        };
    }
};

pub fn fillResizedNchw01(
    rgb: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    nchw_out: []f32,
) void {
    const plane = @as(usize, dst_w) * @as(usize, dst_h);
    const sx_scale = if (dst_w > 1) @as(f64, @floatFromInt(src_w - 1)) / @as(f64, @floatFromInt(dst_w - 1)) else 0.0;
    const sy_scale = if (dst_h > 1) @as(f64, @floatFromInt(src_h - 1)) / @as(f64, @floatFromInt(dst_h - 1)) else 0.0;

    for (0..dst_h) |y| {
        const fy = @as(f64, @floatFromInt(y)) * sy_scale;
        const y0: usize = @intFromFloat(@floor(fy));
        const y1: usize = @min(@as(usize, src_h - 1), y0 + 1);
        const wy = fy - @as(f64, @floatFromInt(y0));
        for (0..dst_w) |x| {
            const fx = @as(f64, @floatFromInt(x)) * sx_scale;
            const x0: usize = @intFromFloat(@floor(fx));
            const x1: usize = @min(@as(usize, src_w - 1), x0 + 1);
            const wx = fx - @as(f64, @floatFromInt(x0));

            const idx00 = (y0 * src_w + x0) * 3;
            const idx10 = (y0 * src_w + x1) * 3;
            const idx01 = (y1 * src_w + x0) * 3;
            const idx11 = (y1 * src_w + x1) * 3;

            const w00 = (1.0 - wx) * (1.0 - wy);
            const w10 = wx * (1.0 - wy);
            const w01 = (1.0 - wx) * wy;
            const w11 = wx * wy;

            const p_idx = y * dst_w + x;
            for (0..3) |ch| {
                const v = w00 * @as(f64, @floatFromInt(rgb[idx00 + ch])) +
                    w10 * @as(f64, @floatFromInt(rgb[idx10 + ch])) +
                    w01 * @as(f64, @floatFromInt(rgb[idx01 + ch])) +
                    w11 * @as(f64, @floatFromInt(rgb[idx11 + ch]));
                nchw_out[ch * plane + p_idx] = @floatCast(v / 255.0);
            }
        }
    }
}

pub const AffineFit = struct {
    scale: f64,
    shift: f64,
    quad: f64 = 0.0,
    inlier_count: usize,
    mad_residual: f64,
};

pub fn fitAffineDepthIrlS(pred: []const f32, target: []const f32, weights: []const bool) ?AffineFit {
    return fitAffineDepthWeightedIrlS(pred, target, weights, null, false);
}

pub fn fitAffineDepthWeightedIrlS(
    pred: []const f32,
    target: []const f32,
    valid: []const bool,
    base_weights: ?[]const f64,
    relative_error: bool,
) ?AffineFit {
    var s: f64 = 1.0;
    var b: f64 = 0.0;
    var valid_count: usize = 0;

    {
        var sw: f64 = 0;
        var sp: f64 = 0;
        var st: f64 = 0;
        var spp: f64 = 0;
        var spt: f64 = 0;
        for (pred, target, valid, 0..) |pv, tv, ok, i| {
            if (!ok) continue;
            if (!std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
            const p: f64 = pv;
            const t: f64 = tv;
            const bw: f64 = if (base_weights) |bws| bws[i] else 1.0;
            if (!(bw > 0.0 and std.math.isFinite(bw))) continue;
            sw += bw;
            sp += bw * p;
            st += bw * t;
            spp += bw * p * p;
            spt += bw * p * t;
            valid_count += 1;
        }
        if (valid_count < 32 or sw <= 1e-12) return null;
        const det = spp * sw - sp * sp;
        if (@abs(det) < 1e-12) return null;
        s = (spt * sw - sp * st) / det;
        b = (spp * st - sp * spt) / det;
        if (!(s > 0.0 and std.math.isFinite(s) and std.math.isFinite(b))) return null;
    }

    var delta: f64 = if (relative_error) 0.06 else 0.25;
    for (0..6) |_| {
        var sw: f64 = 0;
        var sp: f64 = 0;
        var st: f64 = 0;
        var spp: f64 = 0;
        var spt: f64 = 0;
        var err_wsum: f64 = 0;
        var bw_sum: f64 = 0;
        var cnt: usize = 0;

        for (pred, target, valid, 0..) |pv, tv, ok, i| {
            if (!ok) continue;
            if (!std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
            const p: f64 = pv;
            const t: f64 = tv;
            const bw: f64 = if (base_weights) |bws| bws[i] else 1.0;
            if (!(bw > 0.0 and std.math.isFinite(bw))) continue;
            const abs_err = @abs(s * p + b - t);
            const metric_err = if (relative_error) abs_err / t else abs_err;
            err_wsum += bw * metric_err;
            bw_sum += bw;
            cnt += 1;
            const huber_w = if (metric_err <= delta) 1.0 else delta / metric_err;
            const norm_scale = if (relative_error) 1.0 / (t * t) else 1.0;
            const w = bw * huber_w * norm_scale;
            sw += w;
            sp += w * p;
            st += w * t;
            spp += w * p * p;
            spt += w * p * t;
        }
        if (cnt < 32 or bw_sum <= 1e-12) break;
        const mean_err = err_wsum / bw_sum;
        delta = if (relative_error) @max(0.015, 1.5 * mean_err) else @max(0.03, 1.5 * mean_err);
        const det = spp * sw - sp * sp;
        if (@abs(det) < 1e-12) break;
        const ns = (spt * sw - sp * st) / det;
        const nb = (spp * st - sp * spt) / det;
        if (ns > 0.0 and std.math.isFinite(ns) and std.math.isFinite(nb)) {
            s = ns;
            b = nb;
        }
    }

    var err_sum: f64 = 0;
    var n_err: usize = 0;
    for (pred, target, valid) |pv, tv, ok| {
        if (!ok or !std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
        err_sum += @abs(s * @as(f64, pv) + b - @as(f64, tv));
        n_err += 1;
    }
    return .{
        .scale = s,
        .shift = b,
        .inlier_count = valid_count,
        .mad_residual = if (n_err > 0) err_sum / @as(f64, @floatFromInt(n_err)) else 0.0,
    };
}

/// Refines a weighted relative affine depth calibration with a monotone quadratic term
/// (`z = q*p^2 + s*p + b`) when scenes span a wide depth ratio, preventing long-range compression.
pub fn fitMonotoneQuadDepthWeightedIrlS(
    pred: []const f32,
    target: []const f32,
    valid: []const bool,
    base_weights: ?[]const f64,
) ?AffineFit {
    const base = fitAffineDepthWeightedIrlS(pred, target, valid, base_weights, true) orelse return null;

    var min_t: f64 = std.math.inf(f64);
    var max_t: f64 = 0.0;
    var max_p: f64 = 0.0;
    var far_cnt: usize = 0;
    for (pred, target, valid) |pv, tv, ok| {
        if (!ok or !std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
        const p: f64 = pv;
        const t: f64 = tv;
        min_t = @min(min_t, t);
        max_t = @max(max_t, t);
        max_p = @max(max_p, p);
        if (t >= 9.0) far_cnt += 1;
    }
    if (far_cnt < 8 or max_t < 2.8 * min_t or max_p <= 1e-3) return base;

    const evalRelLoss = struct {
        fn call(p_arr: []const f32, t_arr: []const f32, v_arr: []const bool, bw_opt: ?[]const f64, qq: f64, ss: f64, bb: f64) f64 {
            var sum_l: f64 = 0;
            var sum_w: f64 = 0;
            for (p_arr, t_arr, v_arr, 0..) |pv, tv, ok, i| {
                if (!ok or !std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
                const bw = if (bw_opt) |bws| bws[i] else 1.0;
                if (!(bw > 0.0 and std.math.isFinite(bw))) continue;
                const p: f64 = pv;
                const t: f64 = tv;
                const e = @abs((qq * p + ss) * p + bb - t) / t;
                sum_l += bw * @min(e, 0.25);
                sum_w += bw;
            }
            return if (sum_w > 0) sum_l / sum_w else std.math.inf(f64);
        }
    }.call;

    const base_loss = evalRelLoss(pred, target, valid, base_weights, 0.0, base.scale, base.shift);

    var q: f64 = 0.0;
    var s: f64 = base.scale;
    var b: f64 = base.shift;
    var delta: f64 = 0.05;

    for (0..6) |_| {
        var ata = [3][3]f64{
            .{ 0, 0, 0 },
            .{ 0, 0, 0 },
            .{ 0, 0, 0 },
        };
        var aty = [3]f64{ 0, 0, 0 };
        var err_wsum: f64 = 0;
        var bw_sum: f64 = 0;

        for (pred, target, valid, 0..) |pv, tv, ok, i| {
            if (!ok or !std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
            const bw: f64 = if (base_weights) |bws| bws[i] else 1.0;
            if (!(bw > 0.0 and std.math.isFinite(bw))) continue;
            const p: f64 = pv;
            const t: f64 = tv;
            const z_est = (q * p + s) * p + b;
            const rel_err = @abs(z_est - t) / t;
            err_wsum += bw * rel_err;
            bw_sum += bw;
            const huber_w = if (rel_err <= delta) 1.0 else delta / rel_err;
            const w = bw * huber_w / (t * t);
            const phi = [3]f64{ 1.0, p, p * p };
            for (0..3) |r| {
                aty[r] += w * phi[r] * t;
                for (0..3) |c_i| ata[r][c_i] += w * phi[r] * phi[c_i];
            }
        }
        if (bw_sum <= 1e-12) break;
        delta = @max(0.015, 1.5 * (err_wsum / bw_sum));
        ata[2][2] += 1e-6;

        // Solve 3x3 symmetric system via Gaussian elimination with partial pivoting
        var aug = [3][4]f64{
            .{ ata[0][0], ata[0][1], ata[0][2], aty[0] },
            .{ ata[1][0], ata[1][1], ata[1][2], aty[1] },
            .{ ata[2][0], ata[2][1], ata[2][2], aty[2] },
        };
        var singular = false;
        for (0..3) |col| {
            var piv = col;
            var best = @abs(aug[col][col]);
            for (col + 1..3) |row| {
                if (@abs(aug[row][col]) > best) {
                    best = @abs(aug[row][col]);
                    piv = row;
                }
            }
            if (best < 1e-12) {
                singular = true;
                break;
            }
            if (piv != col) {
                const tmp = aug[col];
                aug[col] = aug[piv];
                aug[piv] = tmp;
            }
            const inv_p = 1.0 / aug[col][col];
            for (col..4) |k_i| aug[col][k_i] *= inv_p;
            for (0..3) |row| {
                if (row == col) continue;
                const f = aug[row][col];
                for (col..4) |k_i| aug[row][k_i] -= f * aug[col][k_i];
            }
        }
        if (singular) break;
        const nb = aug[0][3];
        const ns = aug[1][3];
        const nq = aug[2][3];
        if (!std.math.isFinite(nb) or !std.math.isFinite(ns) or !std.math.isFinite(nq)) break;
        // Require positive monotonic slope across all observed and extrapolated depths
        if (ns < 0.25 * base.scale or (2.0 * nq * (max_p * 1.5) + ns) < 0.25 * base.scale) break;
        b = nb;
        s = ns;
        q = nq;
    }

    const quad_loss = evalRelLoss(pred, target, valid, base_weights, q, s, b);
    if (!(quad_loss < base_loss * 0.92 and q >= -0.05 * s)) return base;

    var err_sum: f64 = 0;
    var n_err: usize = 0;
    for (pred, target, valid) |pv, tv, ok| {
        if (!ok or !std.math.isFinite(pv) or !std.math.isFinite(tv) or pv <= 1e-4 or tv <= 1e-4) continue;
        const p: f64 = pv;
        err_sum += @abs((q * p + s) * p + b - @as(f64, tv));
        n_err += 1;
    }
    return .{
        .scale = s,
        .shift = b,
        .quad = q,
        .inlier_count = base.inlier_count,
        .mad_residual = if (n_err > 0) err_sum / @as(f64, @floatFromInt(n_err)) else 0.0,
    };
}

fn roundUpDiv14(v: u32) u32 {
    const blocks = @max(@as(u32, 8), (v + 7) / 14);
    return blocks * 14;
}

fn sampleBilinear(src: []const f32, sw: u32, sh: u32, fx: f64, fy: f64) f32 {
    const max_x: f64 = @floatFromInt(sw - 1);
    const max_y: f64 = @floatFromInt(sh - 1);
    const cx = std.math.clamp(fx, 0.0, max_x);
    const cy = std.math.clamp(fy, 0.0, max_y);
    const x0: usize = @intFromFloat(@floor(cx));
    const y0: usize = @intFromFloat(@floor(cy));
    const x1: usize = @min(@as(usize, sw - 1), x0 + 1);
    const y1: usize = @min(@as(usize, sh - 1), y0 + 1);
    const wx: f32 = @floatCast(cx - @as(f64, @floatFromInt(x0)));
    const wy: f32 = @floatCast(cy - @as(f64, @floatFromInt(y0)));

    const v00 = src[y0 * sw + x0];
    const v10 = src[y0 * sw + x1];
    const v01 = src[y1 * sw + x0];
    const v11 = src[y1 * sw + x1];
    const top = v00 * (1.0 - wx) + v10 * wx;
    const bot = v01 * (1.0 - wx) + v11 * wx;
    return top * (1.0 - wy) + bot * wy;
}

fn selectBalancedViewIndices(allocator: std.mem.Allocator, n: usize, max_views: usize) ![]usize {
    const target = @min(n, max_views);
    if (target == 0) return allocator.alloc(usize, 0);

    const used = try allocator.alloc(bool, n);
    defer allocator.free(used);
    @memset(used, false);

    var order: std.ArrayList(usize) = .empty;
    errdefer order.deinit(allocator);

    const pushIfUnused = struct {
        fn call(alloc: std.mem.Allocator, list: *std.ArrayList(usize), seen: []bool, idx: usize) !void {
            if (!seen[idx]) {
                seen[idx] = true;
                try list.append(alloc, idx);
            }
        }
    }.call;

    try pushIfUnused(allocator, &order, used, n / 2);
    if (order.items.len < target) try pushIfUnused(allocator, &order, used, 0);
    if (order.items.len < target) try pushIfUnused(allocator, &order, used, n - 1);

    const Interval = struct { lo: usize, hi: usize };
    var queue: std.ArrayList(Interval) = .empty;
    defer queue.deinit(allocator);
    if (n > 2) {
        try queue.append(allocator, .{ .lo = 0, .hi = n / 2 });
        try queue.append(allocator, .{ .lo = n / 2, .hi = n - 1 });
    }

    var q_head: usize = 0;
    while (order.items.len < target and q_head < queue.items.len) : (q_head += 1) {
        const iv = queue.items[q_head];
        if (iv.hi <= iv.lo + 1) continue;
        const mid = (iv.lo + iv.hi) / 2;
        try pushIfUnused(allocator, &order, used, mid);
        try queue.append(allocator, .{ .lo = iv.lo, .hi = mid });
        try queue.append(allocator, .{ .lo = mid, .hi = iv.hi });
    }

    if (order.items.len < target) {
        for (0..n) |i| {
            if (order.items.len >= target) break;
            try pushIfUnused(allocator, &order, used, i);
        }
    }

    return order.toOwnedSlice(allocator);
}

pub const OtherViewsContext = struct {
    renders: []const render.RenderedView,
    poses: []const CameraPose,
    images: []const dataset.ImageData,
    reg_cams: []const usize,
    self_ri: usize,
};

pub const GridPatchInput = struct {
    gw: usize,
    gh: usize,
    step_px: usize = 10,
    px_x: []const usize,
    px_y: []const usize,
    cov: []const bool,
    cov_depth: []const f32,
    z_global: []const f32,
    mask: []const f32,
    occ_blocked: ?[]const bool = null,
    other_views: ?OtherViewsContext = null,
    fit_mad: f64,
    med_edge: f64,
    bb_min: Vec3,
    bb_max: Vec3,
    k: Mat3,
    pose: CameraPose,
    img: dataset.ImageData,
    vert_offset: usize,
    max_new_verts: usize,
    config: NeuralHoleConfig,
};

pub const GridPatchOutput = struct {
    pts: []Vec3,
    cols: []Color,
    faces: [][3]usize,

    pub fn deinit(self: *GridPatchOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.pts);
        allocator.free(self.cols);
        allocator.free(self.faces);
        self.* = undefined;
    }
};

fn unprojectPixelWorld(k: Mat3, inv_rt: Mat3, t: Vec3, px: usize, py: usize, z_val: f64) Vec3 {
    const px_c: f64 = @as(f64, @floatFromInt(px)) + 0.5;
    const py_c: f64 = @as(f64, @floatFromInt(py)) + 0.5;
    const yn = (py_c - k.m[1][2]) / k.m[1][1];
    const xn = (px_c - k.m[0][2] - k.m[0][1] * yn) / k.m[0][0];
    const x_cam = Vec3.init(xn * z_val, yn * z_val, z_val);
    return inv_rt.mulVec(x_cam.sub(t));
}

fn relaxHarmonicNode(
    res: []f32,
    anchored: []const bool,
    in: GridPatchInput,
    gx: usize,
    gy: usize,
    max_anchor_res: f32,
) void {
    const gw = in.gw;
    const gi = gy * gw + gx;
    if (in.cov[gi] or anchored[gi] or in.mask[gi] <= in.config.mask_thresh) return;
    const zc = in.z_global[gi];
    if (!(zc > 0.1)) return;
    const edge_lim = zc * in.config.max_depth_grad_rel;

    var nb_buf: [4]usize = undefined;
    var nb_len: usize = 0;
    if (gx > 0) {
        nb_buf[nb_len] = gi - 1;
        nb_len += 1;
    }
    if (gx + 1 < gw) {
        nb_buf[nb_len] = gi + 1;
        nb_len += 1;
    }
    if (gy > 0) {
        nb_buf[nb_len] = gi - gw;
        nb_len += 1;
    }
    if (gy + 1 < in.gh) {
        nb_buf[nb_len] = gi + gw;
        nb_len += 1;
    }

    var sum_r: f32 = 0;
    var cnt_nb: f32 = 0;
    for (nb_buf[0..nb_len]) |nb| {
        if (in.cov[nb]) {
            if (!anchored[nb]) continue;
            if (@abs(in.z_global[nb] - zc) > edge_lim) continue;
            if (@abs(in.cov_depth[nb] - zc) > @max(edge_lim, max_anchor_res)) continue;
            sum_r += res[nb];
            cnt_nb += 1.0;
        } else if (in.mask[nb] > in.config.mask_thresh and @abs(in.z_global[nb] - zc) <= edge_lim) {
            sum_r += res[nb];
            cnt_nb += 1.0;
        }
    }
    if (cnt_nb > 0) {
        res[gi] = 0.9985 * (sum_r / cnt_nb);
    }
}

/// Solves edge-aware screened harmonic boundary residuals over a camera-space grid and emits
/// watertight, normal-oriented triangles covering only missing hole regions.
pub fn extractHolePatchFromGrid(allocator: std.mem.Allocator, in: GridPatchInput) !GridPatchOutput {
    const n_nodes = in.gw * in.gh;
    const res = try allocator.alloc(f32, n_nodes);
    defer allocator.free(res);
    @memset(res, 0);

    const anchored = try allocator.alloc(bool, n_nodes);
    defer allocator.free(anchored);
    @memset(anchored, false);

    const max_anchor_res: f32 = @floatCast(@max(0.45, 4.5 * in.fit_mad));
    for (0..n_nodes) |gi| {
        if (in.cov[gi] and in.cov_depth[gi] > 0.05 and in.z_global[gi] > 0.05) {
            const diff = in.cov_depth[gi] - in.z_global[gi];
            if (@abs(diff) <= max_anchor_res) {
                res[gi] = diff;
                anchored[gi] = true;
            }
        }
    }

    // Symmetric Gauss-Seidel sweeps propagate SfM boundary residuals rapidly across wide holes
    var it: usize = 0;
    while (it < in.config.harmonic_iters) : (it += 1) {
        if (it & 1 == 0) {
            for (0..in.gh) |gy| {
                for (0..in.gw) |gx| {
                    relaxHarmonicNode(res, anchored, in, gx, gy, max_anchor_res);
                }
            }
        } else {
            var gy = in.gh;
            while (gy > 0) {
                gy -= 1;
                var gx = in.gw;
                while (gx > 0) {
                    gx -= 1;
                    relaxHarmonicNode(res, anchored, in, gx, gy, max_anchor_res);
                }
            }
        }
    }

    const node_pos = try allocator.alloc(Vec3, n_nodes);
    defer allocator.free(node_pos);
    const node_z = try allocator.alloc(f64, n_nodes);
    defer allocator.free(node_z);
    const node_valid = try allocator.alloc(bool, n_nodes);
    defer allocator.free(node_valid);
    @memset(node_valid, false);

    var n_skip_mask: usize = 0;
    var n_skip_occ: usize = 0;
    var n_skip_bb: usize = 0;

    const inv_rt = in.pose.r.transpose();
    for (0..in.gh) |gy| {
        const py = in.px_y[gy];
        for (0..in.gw) |gx| {
            const px = in.px_x[gx];
            const gi = gy * in.gw + gx;
            if (in.cov[gi]) {
                const z_cov: f64 = in.cov_depth[gi];
                if (!std.math.isFinite(z_cov) or z_cov <= 0.1) continue;
                node_pos[gi] = unprojectPixelWorld(in.k, inv_rt, in.pose.t, px, py, z_cov);
                node_z[gi] = z_cov;
                node_valid[gi] = true;
            } else {
                if (in.mask[gi] <= in.config.mask_thresh) {
                    n_skip_mask += 1;
                    continue;
                }
                if (in.occ_blocked) |pre_occ| {
                    if (pre_occ[gi]) {
                        n_skip_occ += 1;
                        continue;
                    }
                }
                const z_val: f64 = @as(f64, in.z_global[gi] + res[gi]);
                if (!std.math.isFinite(z_val) or z_val <= 0.15) continue;
                const x_world = unprojectPixelWorld(in.k, inv_rt, in.pose.t, px, py, z_val);
                if (x_world.x < in.bb_min.x or x_world.x > in.bb_max.x or
                    x_world.y < in.bb_min.y or x_world.y > in.bb_max.y or
                    x_world.z < in.bb_min.z or x_world.z > in.bb_max.z)
                {
                    n_skip_bb += 1;
                    continue;
                }
                if (in.other_views) |ov| {
                    var occ_hits: usize = 0;
                    for (ov.reg_cams, 0..) |oci, ori| {
                        if (ori == ov.self_ri) continue;
                        const oc = ov.poses[oci].transformPoint(x_world);
                        if (oc.z <= 0.1) continue;
                        const oinv_z = 1.0 / oc.z;
                        const oxn = oc.x * oinv_z;
                        const oyn = oc.y * oinv_z;
                        const opx = in.k.m[0][0] * oxn + in.k.m[0][1] * oyn + in.k.m[0][2];
                        const opy = in.k.m[1][1] * oyn + in.k.m[1][2];
                        if (opx < 0 or opy < 0) continue;
                        const oxi: usize = @intFromFloat(@floor(opx));
                        const oyi: usize = @intFromFloat(@floor(opy));
                        if (oxi >= ov.renders[ori].width or oyi >= ov.renders[ori].height) continue;
                        const op_idx = oyi * ov.renders[ori].width + oxi;
                        if (ov.renders[ori].covered[op_idx]) {
                            const z_sfm_other: f64 = ov.renders[ori].depth[op_idx];
                            const occ_margin = @max(in.med_edge * 5.0, 0.045 * z_sfm_other);
                            if (oc.z < z_sfm_other - occ_margin) {
                                occ_hits += 1;
                                if (occ_hits >= 2) break;
                            }
                        }
                    }
                    if (occ_hits >= 2) {
                        n_skip_occ += 1;
                        continue;
                    }
                }
                node_pos[gi] = x_world;
                node_z[gi] = z_val;
                node_valid[gi] = true;
            }
        }
    }

    const sentinel = std.math.maxInt(usize);
    const grid_to_vert = try allocator.alloc(usize, n_nodes);
    defer allocator.free(grid_to_vert);
    @memset(grid_to_vert, sentinel);

    var out_pts: std.ArrayList(Vec3) = .empty;
    errdefer out_pts.deinit(allocator);
    var out_cols: std.ArrayList(Color) = .empty;
    errdefer out_cols.deinit(allocator);
    var out_faces: std.ArrayList([3]usize) = .empty;
    errdefer out_faces.deinit(allocator);

    const cam_c = in.pose.cameraCenter();
    const focal_px = @max(in.k.m[0][0], 1.0);
    const step_f: f64 = @floatFromInt(in.step_px);

    var n_skip_zjump: usize = 0;
    var n_skip_edge: usize = 0;
    var n_skip_aspect: usize = 0;
    var n_skip_grazing: usize = 0;
    var n_skip_photo: usize = 0;

    var err_buf: [64]f64 = undefined;

    for (0..in.gh - 1) |gy| {
        for (0..in.gw - 1) |gx| {
            if (out_pts.items.len >= in.max_new_verts) break;
            const g00 = gy * in.gw + gx;
            const g10 = g00 + 1;
            const g01 = g00 + in.gw;
            const g11 = g01 + 1;

            if (in.cov[g00] and in.cov[g10] and in.cov[g01] and in.cov[g11]) continue;

            const use_diag_00_11 = !node_valid[g10] or !node_valid[g01] or
                (node_valid[g00] and node_valid[g11] and @abs(node_z[g00] - node_z[g11]) <= @abs(node_z[g10] - node_z[g01]));

            const tris: [2][3]usize = if (use_diag_00_11)
                .{ .{ g00, g01, g11 }, .{ g00, g11, g10 } }
            else
                .{ .{ g00, g01, g10 }, .{ g10, g01, g11 } };

            for (tris) |cand| {
                const a = cand[0];
                const b = cand[1];
                const c_idx = cand[2];
                if (!node_valid[a] or !node_valid[b] or !node_valid[c_idx]) continue;
                if (in.cov[a] and in.cov[b] and in.cov[c_idx]) continue;

                const za = node_z[a];
                const zb = node_z[b];
                const zc_v = node_z[c_idx];
                const has_cov_mix = in.cov[a] or in.cov[b] or in.cov[c_idx];
                const z_rel_lim: f64 = if (has_cov_mix) 0.10 else 0.18;
                if (@abs(za - zb) > z_rel_lim * @min(za, zb) or
                    @abs(zb - zc_v) > z_rel_lim * @min(zb, zc_v) or
                    @abs(zc_v - za) > z_rel_lim * @min(zc_v, za))
                {
                    n_skip_zjump += 1;
                    continue;
                }

                const pa = node_pos[a];
                const pb = node_pos[b];
                const pc = node_pos[c_idx];
                const l_ab_sq = pb.sub(pa).normSq();
                const l_bc_sq = pc.sub(pb).normSq();
                const l_ca_sq = pa.sub(pc).normSq();
                const max_len_sq = @max(l_ab_sq, @max(l_bc_sq, l_ca_sq));

                const cross_n = pb.sub(pa).cross(pc.sub(pa));
                const area2 = cross_n.norm();
                const min_aspect: f64 = if (has_cov_mix) 0.065 else 0.038;
                if (area2 < 1e-10 or area2 / max_len_sq < min_aspect) {
                    n_skip_aspect += 1;
                    continue;
                }

                const ray_to_cam = cam_c.sub(pa).normalize();
                const view_dot = cross_n.scale(1.0 / area2).dot(ray_to_cam);
                const abs_vd = @abs(view_dot);
                if (abs_vd < 0.055) {
                    n_skip_grazing += 1;
                    continue;
                }

                const z_mean = (za + zb + zc_v) / 3.0;
                const grazing_scale = @min(@as(f64, 22.0), @max(@as(f64, 12.0), 2.2 / @max(abs_vd, 0.055)));
                const local_max_e = @max(in.med_edge * 9.0, (step_f * z_mean / focal_px) * grazing_scale);
                if (max_len_sq > local_max_e * local_max_e or max_len_sq < 1e-12) {
                    n_skip_edge += 1;
                    continue;
                }

                // Multi-view photometric check when other registered cameras also view this triangle unoccluded
                if (in.other_views) |ov| {
                    var n_vis: usize = 0;
                    const ga_y = a / in.gw;
                    const ga_x = a % in.gw;
                    const gb_y = b / in.gw;
                    const gb_x = b % in.gw;
                    const gc_y = c_idx / in.gw;
                    const gc_x = c_idx % in.gw;
                    const src_ia = (in.px_y[ga_y] * in.img.width + in.px_x[ga_x]) * 3;
                    const src_ib = (in.px_y[gb_y] * in.img.width + in.px_x[gb_x]) * 3;
                    const src_ic = (in.px_y[gc_y] * in.img.width + in.px_x[gc_x]) * 3;

                    for (ov.reg_cams, 0..) |oci, ori| {
                        if (ori == ov.self_ri or n_vis >= err_buf.len) continue;
                        const oimg = ov.images[oci];
                        const opose = ov.poses[oci];
                        const oren = ov.renders[ori];
                        const wf: f64 = @floatFromInt(oimg.width);
                        const hf: f64 = @floatFromInt(oimg.height);

                        const oca = opose.transformPoint(pa);
                        const ocb = opose.transformPoint(pb);
                        const occ = opose.transformPoint(pc);
                        if (oca.z <= 0.1 or ocb.z <= 0.1 or occ.z <= 0.1) continue;

                        const uva = lie.projectPoint(in.k, opose, pa) orelse continue;
                        const uvb = lie.projectPoint(in.k, opose, pb) orelse continue;
                        const uvc = lie.projectPoint(in.k, opose, pc) orelse continue;
                        if (uva.x < 1.0 or uva.x >= wf - 2.0 or uva.y < 1.0 or uva.y >= hf - 2.0) continue;
                        if (uvb.x < 1.0 or uvb.x >= wf - 2.0 or uvb.y < 1.0 or uvb.y >= hf - 2.0) continue;
                        if (uvc.x < 1.0 or uvc.x >= wf - 2.0 or uvc.y < 1.0 or uvc.y >= hf - 2.0) continue;

                        const pma = @as(usize, @intFromFloat(@floor(uva.y))) * oimg.width + @as(usize, @intFromFloat(@floor(uva.x)));
                        const pmb = @as(usize, @intFromFloat(@floor(uvb.y))) * oimg.width + @as(usize, @intFromFloat(@floor(uvb.x)));
                        const pmc = @as(usize, @intFromFloat(@floor(uvc.y))) * oimg.width + @as(usize, @intFromFloat(@floor(uvc.x)));
                        const o_tol = @max(0.25, 0.05 * oca.z);
                        if ((oren.covered[pma] and oca.z > @as(f64, oren.depth[pma]) + o_tol) or
                            (oren.covered[pmb] and ocb.z > @as(f64, oren.depth[pmb]) + o_tol) or
                            (oren.covered[pmc] and occ.z > @as(f64, oren.depth[pmc]) + o_tol))
                        {
                            continue;
                        }

                        const diff =
                            @abs(@as(f64, oimg.rgb[pma * 3 + 0]) - @as(f64, in.img.rgb[src_ia + 0])) +
                            @abs(@as(f64, oimg.rgb[pma * 3 + 1]) - @as(f64, in.img.rgb[src_ia + 1])) +
                            @abs(@as(f64, oimg.rgb[pma * 3 + 2]) - @as(f64, in.img.rgb[src_ia + 2])) +
                            @abs(@as(f64, oimg.rgb[pmb * 3 + 0]) - @as(f64, in.img.rgb[src_ib + 0])) +
                            @abs(@as(f64, oimg.rgb[pmb * 3 + 1]) - @as(f64, in.img.rgb[src_ib + 1])) +
                            @abs(@as(f64, oimg.rgb[pmb * 3 + 2]) - @as(f64, in.img.rgb[src_ib + 2])) +
                            @abs(@as(f64, oimg.rgb[pmc * 3 + 0]) - @as(f64, in.img.rgb[src_ic + 0])) +
                            @abs(@as(f64, oimg.rgb[pmc * 3 + 1]) - @as(f64, in.img.rgb[src_ic + 1])) +
                            @abs(@as(f64, oimg.rgb[pmc * 3 + 2]) - @as(f64, in.img.rgb[src_ic + 2]));
                        err_buf[n_vis] = diff / 9.0;
                        n_vis += 1;
                    }

                    if (n_vis >= 2) {
                        std.mem.sort(f64, err_buf[0..n_vis], {}, std.sort.asc(f64));
                        const q25_photo_err = err_buf[n_vis / 4];
                        if (q25_photo_err > 45.0) {
                            n_skip_photo += 1;
                            continue;
                        }
                    }
                }

                const tri_g: [3]usize = if (view_dot < 0) .{ a, c_idx, b } else .{ a, b, c_idx };
                var new_needed: usize = 0;
                for (tri_g) |gi| {
                    if (grid_to_vert[gi] == sentinel) new_needed += 1;
                }
                if (out_pts.items.len + new_needed > in.max_new_verts) continue;

                var tri_v: [3]usize = undefined;
                for (tri_g, 0..) |gi, k_v| {
                    if (grid_to_vert[gi] == sentinel) {
                        const v_idx = in.vert_offset + out_pts.items.len;
                        grid_to_vert[gi] = v_idx;
                        try out_pts.append(allocator, node_pos[gi]);
                        const g_y = gi / in.gw;
                        const g_x = gi % in.gw;
                        const rgb_idx = (in.px_y[g_y] * in.img.width + in.px_x[g_x]) * 3;
                        try out_cols.append(allocator, .{
                            .r = in.img.rgb[rgb_idx + 0],
                            .g = in.img.rgb[rgb_idx + 1],
                            .b = in.img.rgb[rgb_idx + 2],
                        });
                    }
                    tri_v[k_v] = grid_to_vert[gi];
                }
                try out_faces.append(allocator, tri_v);
            }
        }
    }

    if (in.other_views != null) {
        std.log.debug(
            "  skip stats: mask={d} occ={d} bb={d} zjump={d} edge={d} aspect={d} grazing={d} photo={d}",
            .{ n_skip_mask, n_skip_occ, n_skip_bb, n_skip_zjump, n_skip_edge, n_skip_aspect, n_skip_grazing, n_skip_photo },
        );
    }

    return .{
        .pts = try out_pts.toOwnedSlice(allocator),
        .cols = try out_cols.toOwnedSlice(allocator),
        .faces = try out_faces.toOwnedSlice(allocator),
    };
}

/// Runs a local 3D geometry Transformer (MoGe-2 ONNX) on calibrated views to predict missing
/// geometry in untextured holes, anchors hole boundaries harmonically to existing SfM surfaces,
/// verifies empty-space consistency across all cameras, and stitches clean hole triangles onto the mesh.
pub fn completeMeshHolesWithTransformer(
    allocator: std.mem.Allocator,
    mesh: *mesh_mod.Mesh,
    k: Mat3,
    images: []const dataset.ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    config: NeuralHoleConfig,
) !usize {
    if (mesh.points.len < 16 or mesh.faces.len == 0 or images.len == 0) return 0;

    var sess = try NeuralSession.init(allocator, config.model_path, config.num_threads);
    defer sess.deinit();

    var reg_cams: std.ArrayList(usize) = .empty;
    defer reg_cams.deinit(allocator);
    for (registered, 0..) |is_reg, ci| {
        if (is_reg) try reg_cams.append(allocator, ci);
    }
    if (reg_cams.items.len == 0) return 0;

    // Rasterize initial SfM mesh into all registered cameras for scale fitting & empty-space checks
    const renders_base = try allocator.alloc(render.RenderedView, reg_cams.items.len);
    for (reg_cams.items, 0..) |ci, ri| {
        renders_base[ri] = try render.renderMeshView(
            allocator,
            images[ci].width,
            images[ci].height,
            k,
            poses[ci],
            mesh.points,
            mesh.colors,
            mesh.faces,
        );
    }
    defer {
        for (renders_base) |*rv| rv.deinit(allocator);
        allocator.free(renders_base);
    }

    var bb_min = Vec3.init(std.math.inf(f64), std.math.inf(f64), std.math.inf(f64));
    var bb_max = Vec3.init(-std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64));
    for (mesh.points) |pt| {
        bb_min.x = @min(bb_min.x, pt.x);
        bb_min.y = @min(bb_min.y, pt.y);
        bb_min.z = @min(bb_min.z, pt.z);
        bb_max.x = @max(bb_max.x, pt.x);
        bb_max.y = @max(bb_max.y, pt.y);
        bb_max.z = @max(bb_max.z, pt.z);
    }
    const bb_ext = bb_max.sub(bb_min);
    const pad = Vec3.init(@max(bb_ext.x * 2.5, 20.0), @max(bb_ext.y * 2.5, 15.0), @max(bb_ext.z * 3.5, 30.0));
    bb_min = bb_min.sub(pad);
    bb_max = bb_max.add(pad);

    var edge_lens: std.ArrayList(f64) = .empty;
    defer edge_lens.deinit(allocator);
    try edge_lens.ensureTotalCapacity(allocator, mesh.faces.len);
    for (mesh.faces) |tri| {
        edge_lens.appendAssumeCapacity(mesh.points[tri[1]].sub(mesh.points[tri[0]]).norm());
    }
    std.mem.sort(f64, edge_lens.items, {}, std.sort.asc(f64));
    const med_edge = if (edge_lens.items.len > 0) edge_lens.items[edge_lens.items.len / 2] else 0.05;

    const selected_ri = try selectBalancedViewIndices(allocator, reg_cams.items.len, config.max_views);
    defer allocator.free(selected_ri);

    // Process widest-hole views first so broad background geometry sets global depth context for closer views
    const uncov_frac = try allocator.alloc(f64, reg_cams.items.len);
    defer allocator.free(uncov_frac);
    for (renders_base, 0..) |rv, ri_idx| {
        var u_cnt: usize = 0;
        for (rv.covered) |is_cov| {
            if (!is_cov) u_cnt += 1;
        }
        uncov_frac[ri_idx] = if (rv.covered.len > 0) @as(f64, @floatFromInt(u_cnt)) / @as(f64, @floatFromInt(rv.covered.len)) else 0.0;
    }
    const SortCtx = struct {
        scores: []const f64,
        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.scores[a] > ctx.scores[b];
        }
    };
    std.mem.sort(usize, selected_ri, SortCtx{ .scores = uncov_frac }, SortCtx.lessThan);

    var pts_acc: std.ArrayList(Vec3) = .empty;
    errdefer pts_acc.deinit(allocator);
    try pts_acc.appendSlice(allocator, mesh.points);

    var cols_acc: std.ArrayList(Color) = .empty;
    errdefer cols_acc.deinit(allocator);
    try cols_acc.appendSlice(allocator, mesh.colors);

    var faces_acc: std.ArrayList([3]usize) = .empty;
    errdefer faces_acc.deinit(allocator);
    try faces_acc.appendSlice(allocator, mesh.faces);

    const base_vert_count = mesh.points.len;

    for (selected_ri, 0..) |ri, view_step| {
        const added_so_far = pts_acc.items.len - base_vert_count;
        if (added_so_far >= config.max_added_points) break;

        const ci = reg_cams.items[ri];
        const img = images[ci];
        const pose = poses[ci];
        const base_ren = renders_base[ri];
        const w = img.width;
        const h = img.height;

        // Rasterize accumulated mesh before inference so already-complete intermediate views skip ONNX
        var dyn_ren: ?render.RenderedView = if (view_step > 0 and pts_acc.items.len > base_vert_count)
            try render.renderMeshView(allocator, w, h, k, pose, pts_acc.items, cols_acc.items, faces_acc.items)
        else
            null;
        defer if (dyn_ren) |*dr| dr.deinit(allocator);
        const cur_ren = if (dyn_ren) |dr| dr else base_ren;

        var cur_cov_cnt: usize = 0;
        for (cur_ren.covered) |is_cov| {
            if (is_cov) cur_cov_cnt += 1;
        }
        if (cur_cov_cnt * 100 >= cur_ren.covered.len * 97) continue;

        const max_dim_f: f64 = @floatFromInt(@max(w, h));
        const scale_in = @as(f64, @floatFromInt(config.input_max_dim)) / max_dim_f;
        const target_w = roundUpDiv14(@intFromFloat(@max(112.0, @as(f64, @floatFromInt(w)) * scale_in)));
        const target_h = roundUpDiv14(@intFromFloat(@max(112.0, @as(f64, @floatFromInt(h)) * scale_in)));

        var pred = try sess.predict(allocator, img.rgb, w, h, target_w, target_h, config.num_tokens);
        defer pred.deinit(allocator);

        // Fit global affine (scale, shift) across all depth scales combining dense SfM renders and unoccluded sparse SfM vertices
        var fit_pred_list: std.ArrayList(f32) = .empty;
        defer fit_pred_list.deinit(allocator);
        var fit_tgt_list: std.ArrayList(f32) = .empty;
        defer fit_tgt_list.deinit(allocator);
        var fit_ok_list: std.ArrayList(bool) = .empty;
        defer fit_ok_list.deinit(allocator);
        var fit_bin_list: std.ArrayList(u8) = .empty;
        defer fit_bin_list.deinit(allocator);

        const sx_pred = if (w > 1) @as(f64, @floatFromInt(pred.width - 1)) / @as(f64, @floatFromInt(w - 1)) else 0.0;
        const sy_pred = if (h > 1) @as(f64, @floatFromInt(pred.height - 1)) / @as(f64, @floatFromInt(h - 1)) else 0.0;
        var bin_cnt: [16]usize = @splat(0);

        const depthBin = struct {
            fn idx(z: f64) u8 {
                const lz = std.math.log2(@max(0.25, z)) + 2.0; // 0..8 -> [0.25m..64m]
                return @intCast(std.math.clamp(@as(i32, @intFromFloat(@floor(lz * 2.0))), 0, 15));
            }
        }.idx;

        const fit_sw: usize = (w + 3) / 4;
        const fit_sh: usize = (h + 3) / 4;
        try fit_pred_list.ensureTotalCapacity(allocator, fit_sw * fit_sh);
        try fit_tgt_list.ensureTotalCapacity(allocator, fit_sw * fit_sh);
        try fit_ok_list.ensureTotalCapacity(allocator, fit_sw * fit_sh);
        try fit_bin_list.ensureTotalCapacity(allocator, fit_sw * fit_sh);

        for (0..fit_sh) |fy| {
            const y = @min(h - 1, fy * 4);
            const py_pred = @as(f64, @floatFromInt(y)) * sy_pred;
            for (0..fit_sw) |fx| {
                const x = @min(w - 1, fx * 4);
                const p = y * w + x;
                const pz = sampleBilinear(pred.depth, pred.width, pred.height, @as(f64, @floatFromInt(x)) * sx_pred, py_pred);
                const pm = sampleBilinear(pred.mask, pred.width, pred.height, @as(f64, @floatFromInt(x)) * sx_pred, py_pred);
                const tgt = base_ren.depth[p];
                const is_ok = base_ren.covered[p] and pm > config.mask_thresh and pz > 1e-4 and tgt > 0.1;
                const b_idx: u8 = if (is_ok) depthBin(tgt) else 0;
                if (is_ok) bin_cnt[b_idx] += 1;
                try fit_pred_list.append(allocator, pz);
                try fit_tgt_list.append(allocator, tgt);
                try fit_ok_list.append(allocator, is_ok);
                try fit_bin_list.append(allocator, b_idx);
            }
        }

        const dense_sample_len = fit_pred_list.items.len;
        const wf: f64 = @floatFromInt(w);
        const hf: f64 = @floatFromInt(h);
        for (pts_acc.items[0..base_vert_count]) |pt| {
            const pc = pose.transformPoint(pt);
            if (pc.z <= 0.2 or pc.z > 80.0) continue;
            const uv = lie.projectPoint(k, pose, pt) orelse continue;
            if (uv.x < 0.0 or uv.x >= wf - 1.0 or uv.y < 0.0 or uv.y >= hf - 1.0) continue;
            const ui: usize = @intFromFloat(@floor(uv.x));
            const vi: usize = @intFromFloat(@floor(uv.y));
            const p_idx = vi * w + ui;
            if (base_ren.covered[p_idx]) {
                const bd: f64 = base_ren.depth[p_idx];
                if (pc.z > bd + @max(0.15, 0.03 * bd)) continue;
            }
            const pz = sampleBilinear(pred.depth, pred.width, pred.height, uv.x * sx_pred, uv.y * sy_pred);
            const pm = sampleBilinear(pred.mask, pred.width, pred.height, uv.x * sx_pred, uv.y * sy_pred);
            if (pm <= config.mask_thresh or pz <= 1e-4) continue;
            const b_idx = depthBin(pc.z);
            bin_cnt[b_idx] += 1;
            try fit_pred_list.append(allocator, pz);
            try fit_tgt_list.append(allocator, @floatCast(pc.z));
            try fit_ok_list.append(allocator, true);
            try fit_bin_list.append(allocator, b_idx);
        }

        const sparse_end_len = fit_pred_list.items.len;
        // Anchor depth bins unpopulated by ground-truth SfM using already-stitched multi-view geometry
        if (dyn_ren != null) {
            const gt_bin_cnt = bin_cnt;
            for (0..fit_sh) |fy| {
                const y = @min(h - 1, fy * 4);
                const py_pred = @as(f64, @floatFromInt(y)) * sy_pred;
                for (0..fit_sw) |fx| {
                    const x = @min(w - 1, fx * 4);
                    const p = y * w + x;
                    if (base_ren.covered[p] or !cur_ren.covered[p]) continue;
                    const tgt = cur_ren.depth[p];
                    if (tgt <= 0.1) continue;
                    const b_idx = depthBin(tgt);
                    if (gt_bin_cnt[b_idx] >= 32) continue;
                    const pz = sampleBilinear(pred.depth, pred.width, pred.height, @as(f64, @floatFromInt(x)) * sx_pred, py_pred);
                    const pm = sampleBilinear(pred.mask, pred.width, pred.height, @as(f64, @floatFromInt(x)) * sx_pred, py_pred);
                    if (pm <= config.mask_thresh or pz <= 1e-4) continue;
                    bin_cnt[b_idx] += 1;
                    try fit_pred_list.append(allocator, pz);
                    try fit_tgt_list.append(allocator, tgt);
                    try fit_ok_list.append(allocator, true);
                    try fit_bin_list.append(allocator, b_idx);
                }
            }
        }

        const fit_weights = try allocator.alloc(f64, fit_pred_list.items.len);
        defer allocator.free(fit_weights);
        for (fit_ok_list.items, fit_bin_list.items, 0..) |is_ok, b_idx, fi| {
            const src_mult: f64 = if (fi >= sparse_end_len) 0.5 else 1.0;
            fit_weights[fi] = if (is_ok) src_mult / @max(@as(f64, 16.0), @as(f64, @floatFromInt(bin_cnt[b_idx]))) else 0.0;
        }

        const fit = fitMonotoneQuadDepthWeightedIrlS(
            fit_pred_list.items,
            fit_tgt_list.items,
            fit_ok_list.items,
            fit_weights,
        ) orelse continue;

        var n_sparse_far: usize = 0;
        var sparse_far_abs: f64 = 0;
        for (fit_pred_list.items[dense_sample_len..sparse_end_len], fit_tgt_list.items[dense_sample_len..sparse_end_len]) |pz, tz| {
            if (tz >= 9.0) {
                const p_d: f64 = pz;
                sparse_far_abs += @abs((fit.quad * p_d + fit.scale) * p_d + fit.shift - @as(f64, tz));
                n_sparse_far += 1;
            }
        }
        const far_mae = if (n_sparse_far > 0) sparse_far_abs / @as(f64, @floatFromInt(n_sparse_far)) else 0.0;

        std.log.info(
            "Transformer cam {d:02} ({d}x{d}): scale={d:.3} shift={d:.3} quad={d:.3} metric_scale={d:.3} mean_fit_err={d:.4} far_sparse({d})_mae={d:.3}",
            .{ ci, pred.width, pred.height, fit.scale, fit.shift, fit.quad, pred.metric_scale, fit.mad_residual, n_sparse_far, far_mae },
        );

        const step: usize = @max(@as(usize, 6), @as(usize, @min(w, h)) / config.grid_target_nodes_short);
        const gw: usize = (w - 1 + step - 1) / step + 1;
        const gh: usize = (h - 1 + step - 1) / step + 1;
        const n_nodes = gw * gh;

        const px_x = try allocator.alloc(usize, gw);
        defer allocator.free(px_x);
        for (0..gw) |gx| px_x[gx] = @min(w - 1, gx * step);

        const px_y = try allocator.alloc(usize, gh);
        defer allocator.free(px_y);
        for (0..gh) |gy| px_y[gy] = @min(h - 1, gy * step);

        const cov = try allocator.alloc(bool, n_nodes);
        defer allocator.free(cov);
        const cov_depth = try allocator.alloc(f32, n_nodes);
        defer allocator.free(cov_depth);
        const z_global = try allocator.alloc(f32, n_nodes);
        defer allocator.free(z_global);
        const mask = try allocator.alloc(f32, n_nodes);
        defer allocator.free(mask);

        for (0..gh) |gy| {
            const py = px_y[gy];
            const py_pred = @as(f64, @floatFromInt(py)) * sy_pred;
            for (0..gw) |gx| {
                const px = px_x[gx];
                const gi = gy * gw + gx;
                const p = py * w + px;
                cov[gi] = cur_ren.covered[p];
                cov_depth[gi] = cur_ren.depth[p];

                const fx_pred = @as(f64, @floatFromInt(px)) * sx_pred;
                const pz = sampleBilinear(pred.depth, pred.width, pred.height, fx_pred, py_pred);
                mask[gi] = sampleBilinear(pred.mask, pred.width, pred.height, fx_pred, py_pred);
                const p_d: f64 = pz;
                z_global[gi] = @floatCast((fit.quad * p_d + fit.scale) * p_d + fit.shift);
            }
        }

        var patch = try extractHolePatchFromGrid(allocator, .{
            .gw = gw,
            .gh = gh,
            .step_px = step,
            .px_x = px_x,
            .px_y = px_y,
            .cov = cov,
            .cov_depth = cov_depth,
            .z_global = z_global,
            .mask = mask,
            .other_views = .{
                .renders = renders_base,
                .poses = poses,
                .images = images,
                .reg_cams = reg_cams.items,
                .self_ri = ri,
            },
            .fit_mad = fit.mad_residual,
            .med_edge = med_edge,
            .bb_min = bb_min,
            .bb_max = bb_max,
            .k = k,
            .pose = pose,
            .img = img,
            .vert_offset = pts_acc.items.len,
            .max_new_verts = config.max_added_points - added_so_far,
            .config = config,
        });
        defer patch.deinit(allocator);

        if (patch.pts.len > 0 and patch.faces.len > 0) {
            try pts_acc.appendSlice(allocator, patch.pts);
            try cols_acc.appendSlice(allocator, patch.cols);
            try faces_acc.appendSlice(allocator, patch.faces);
        }
        std.log.info("Transformer stitched {d} hole vertices, {d} triangles from cam {d:02}", .{
            patch.pts.len,
            patch.faces.len,
            ci,
        });
    }

    const total_added = pts_acc.items.len - base_vert_count;
    if (total_added == 0) {
        pts_acc.deinit(allocator);
        cols_acc.deinit(allocator);
        faces_acc.deinit(allocator);
        return 0;
    }

    allocator.free(mesh.points);
    allocator.free(mesh.colors);
    allocator.free(mesh.faces);
    mesh.points = try pts_acc.toOwnedSlice(allocator);
    mesh.colors = try cols_acc.toOwnedSlice(allocator);
    mesh.faces = try faces_acc.toOwnedSlice(allocator);
    return total_added;
}

test "fillResizedNchw01 and fitAffineDepthIrlS recover affine transform with outliers" {
    const gpa = std.testing.allocator;
    const rgb = [_]u8{
        0,   0,   0, 255, 128, 64,
        128, 255, 0, 255, 255, 255,
    };
    var out: [3 * 2 * 2]f32 = undefined;
    fillResizedNchw01(&rgb, 2, 2, 2, 2, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0 / 255.0), out[4 + 1], 1e-5);

    const n: usize = 100;
    const pred = try gpa.alloc(f32, n);
    defer gpa.free(pred);
    const target = try gpa.alloc(f32, n);
    defer gpa.free(target);
    const ok = try gpa.alloc(bool, n);
    defer gpa.free(ok);

    for (0..n) |i| {
        const p: f32 = 1.0 + @as(f32, @floatFromInt(i)) * 0.05;
        pred[i] = p;
        target[i] = if (i % 15 == 0) 25.0 else 2.2 * p + 0.7; // inject ~7% gross outliers
        ok[i] = true;
    }

    const fit = fitAffineDepthIrlS(pred, target, ok).?;
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), fit.scale, 0.08);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), fit.shift, 0.25);

    // Weighted relative IRLS balances dense near samples against sparse far anchors
    var b_pred: [40]f32 = undefined;
    var b_tgt: [40]f32 = undefined;
    var b_ok: [40]bool = @splat(true);
    var b_w: [40]f64 = undefined;
    for (0..35) |i| {
        const p: f32 = 2.0 + 0.02 * @as(f32, @floatFromInt(i));
        b_pred[i] = p;
        b_tgt[i] = 1.5 * p + 0.4;
        b_w[i] = 1.0 / 35.0;
    }
    for (35..40) |i| {
        const p: f32 = 10.0 + 0.5 * @as(f32, @floatFromInt(i - 35));
        b_pred[i] = p;
        b_tgt[i] = 1.5 * p + 0.4;
        b_w[i] = 1.0 / 5.0;
    }
    const wfit = fitAffineDepthWeightedIrlS(&b_pred, &b_tgt, &b_ok, &b_w, true).?;
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), wfit.scale, 0.02);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), wfit.shift, 0.1);

    // Quadratic monotone depth fit recovers compression across 1x-8x depth ranges
    for (0..30) |i| {
        const p: f32 = 0.7 + 0.05 * @as(f32, @floatFromInt(i));
        b_pred[i] = p;
        b_tgt[i] = 0.35 * p * p + 3.2 * p + 0.15;
        b_w[i] = 1.0 / 30.0;
    }
    for (30..40) |i| {
        const p: f32 = 3.0 + 0.15 * @as(f32, @floatFromInt(i - 30));
        b_pred[i] = p;
        b_tgt[i] = 0.35 * p * p + 3.2 * p + 0.15;
        b_w[i] = 1.0 / 10.0;
    }
    const qfit = fitMonotoneQuadDepthWeightedIrlS(&b_pred, &b_tgt, &b_ok, &b_w).?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.35), qfit.quad, 0.04);
    try std.testing.expectApproxEqAbs(@as(f64, 3.2), qfit.scale, 0.15);
}

test "selectBalancedViewIndices and extractHolePatchFromGrid stitch holes with camera-facing normals and cliff rejection" {
    const gpa = std.testing.allocator;
    const order = try selectBalancedViewIndices(gpa, 11, 8);
    defer gpa.free(order);
    try std.testing.expectEqual(@as(usize, 8), order.len);
    try std.testing.expectEqual(@as(usize, 5), order[0]);
    try std.testing.expectEqual(@as(usize, 0), order[1]);
    try std.testing.expectEqual(@as(usize, 10), order[2]);

    // Synthetic 6x6 camera grid over a Z=5.0m wall with an uncovered center hole + foreground step
    const gw: usize = 6;
    const gh: usize = 6;
    const step: usize = 10;
    const w: u32 = 60;
    const h: u32 = 60;
    const rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb);
    @memset(rgb, 200);

    const px_x = [_]usize{ 0, 10, 20, 30, 40, 50 };
    const px_y = [_]usize{ 0, 10, 20, 30, 40, 50 };
    var cov: [36]bool = @splat(true);
    var cov_depth: [36]f32 = @splat(5.0);
    var z_global: [36]f32 = @splat(4.8);
    const mask: [36]f32 = @splat(0.95);
    const occ: [36]bool = @splat(false);

    // Punch a 2x2 uncovered hole in the wall (nodes 2..3, 2..3)
    cov[2 * gw + 2] = false;
    cov[2 * gw + 3] = false;
    cov[3 * gw + 2] = false;
    cov[3 * gw + 3] = false;
    // Inject an isolated foreground cliff column at (0, 2..3) at Z=2.5m (should not bridge to Z=5.0)
    cov_depth[2 * gw + 0] = 2.5;
    cov_depth[3 * gw + 0] = 2.5;
    z_global[2 * gw + 0] = 2.5;
    z_global[3 * gw + 0] = 2.5;
    cov[2 * gw + 1] = false;
    _ = step;

    const k = Mat3{ .m = .{
        .{ 500.0, 0.0, 30.0 },
        .{ 0.0, 500.0, 30.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const pose = CameraPose{ .r = Mat3.identity, .t = Vec3.zero };

    var patch = try extractHolePatchFromGrid(gpa, .{
        .gw = gw,
        .gh = gh,
        .px_x = &px_x,
        .px_y = &px_y,
        .cov = &cov,
        .cov_depth = &cov_depth,
        .z_global = &z_global,
        .mask = &mask,
        .occ_blocked = &occ,
        .fit_mad = 0.05,
        .med_edge = 0.12,
        .bb_min = Vec3.init(-5.0, -5.0, 0.5),
        .bb_max = Vec3.init(5.0, 5.0, 15.0),
        .k = k,
        .pose = pose,
        .img = .{ .width = w, .height = h, .rgb = rgb, .gray = rgb[0 .. w * h] },
        .vert_offset = 100,
        .max_new_verts = 200,
        .config = .{ .model_path = "" },
    });
    defer patch.deinit(gpa);

    try std.testing.expect(patch.pts.len > 0);
    try std.testing.expect(patch.faces.len > 0);
    // Harmonic boundary anchor must bring Z=4.8 hole nodes close to Z=5.0 wall
    var near_wall_cnt: usize = 0;
    for (patch.pts) |pt| {
        if (pt.z > 4.88 and pt.z < 5.05) near_wall_cnt += 1;
    }
    try std.testing.expect(near_wall_cnt > 0);
    // Every emitted triangle normal must face the camera (-Z) and have no cliff-spanning Z jump
    for (patch.faces) |tri| {
        try std.testing.expect(tri[0] >= 100 and tri[1] >= 100 and tri[2] >= 100);
        const pa = patch.pts[tri[0] - 100];
        const pb = patch.pts[tri[1] - 100];
        const pc = patch.pts[tri[2] - 100];
        const z_min = @min(pa.z, @min(pb.z, pc.z));
        const z_max = @max(pa.z, @max(pb.z, pc.z));
        try std.testing.expect(z_max - z_min <= 0.15 * z_min);
        const n = pb.sub(pa).cross(pc.sub(pa));
        try std.testing.expect(n.dot(pose.cameraCenter().sub(pa)) > 0);
    }
}

test "MoGe-2 ONNX model loads and predicts 3D geometry via Zig when present" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "models/moge-2-vitb-normal.onnx", .{}) catch return;

    var sess = try NeuralSession.init(gpa, "models/moge-2-vitb-normal.onnx", 4);
    defer sess.deinit();

    try std.testing.expect(sess.in_names_z.len >= 1);
    try std.testing.expect(sess.out_names_z.len >= 1);

    const w: u32 = 140;
    const h: u32 = 112;
    const rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb);
    for (0..h) |y| {
        for (0..w) |x| {
            const idx = (y * w + x) * 3;
            rgb[idx + 0] = @intCast((x * 255) / w);
            rgb[idx + 1] = @intCast((y * 255) / h);
            rgb[idx + 2] = 128;
        }
    }

    var pred = try sess.predict(gpa, rgb, w, h, w, h, 256);
    defer pred.deinit(gpa);

    try std.testing.expectEqual(@as(usize, w) * @as(usize, h), pred.depth.len);
    try std.testing.expectEqual(@as(usize, w) * @as(usize, h), pred.mask.len);
    var any_finite_pos = false;
    for (pred.depth) |d| {
        if (std.math.isFinite(d) and d > 0) {
            any_finite_pos = true;
            break;
        }
    }
    try std.testing.expect(any_finite_pos);
}
