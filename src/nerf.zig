const std = @import("std");
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const ImageData = @import("dataset.zig").ImageData;
const render = @import("render.zig");
const RenderedView = render.RenderedView;
const mc = @import("marching_cubes.zig");
const mesh_mod = @import("mesh.zig");
const Color = @import("io_ply.zig").Color;
const Io = std.Io;

pub const NerfConfig = struct {
    num_levels: usize = 8,
    table_log2: u5 = 17,
    n_min: f32 = 16.0,
    n_max: f32 = 512.0,
    geo_hidden: usize = 32,
    geo_feat_dim: usize = 15,
    col_hidden: usize = 32,
    num_steps: usize = 1500,
    batch_rays: usize = 1024,
    max_ray_samples: usize = 96,
    lr: f32 = 1e-2,
    weight_decay: f32 = 1e-6,
    dt_min: f32 = 0.0,
    t_near: f32 = 0.05,
    t_far: f32 = 16.0,
    use_contraction: bool = true,
    occ_grid_res: usize = 32,
    occ_update_interval: usize = 64,
    occ_decay: f32 = 0.70,
    occ_threshold: f32 = 0.02,
    depth_loss_weight: f32 = 0.05,
    distortion_weight: f32 = 0.0,
    num_threads: usize = 1,
    seed: u64 = 1337,
    render_scale: f64 = 0.5,
    bg_color: [3]f32 = .{ 0.78, 0.82, 0.88 },
    random_bg: bool = false,
    io: ?Io = null,
};

pub const NerfTimings = struct {
    setup_ms: f64 = 0.0,
    train_ms: f64 = 0.0,
    render_ms: f64 = 0.0,
    final_loss: f64 = 0.0,
};

pub const SceneBounds = struct {
    center: [3]f32,
    inv_radius: f32,
    use_contraction: bool,
    t_near: f32,
    t_far: f32,

    pub fn rayStepDt(self: *const SceneBounds, t: f32, dt_override: f32, max_samples: usize) f32 {
        const core_r = 1.0 / @max(self.inv_radius, 1e-6);
        const dt_base = if (dt_override > 0.0) dt_override else core_r * 0.012;
        const n_f: f32 = @floatFromInt(@max(@as(usize, 16), max_samples));
        if (!self.use_contraction) return dt_base;
        const cone_step = 2.4 / n_f;
        return @max(dt_base, t * cone_step);
    }

    pub fn fromSfm(points: []const Vec3, poses: []const CameraPose, registered: []const bool, use_contraction: bool) SceneBounds {
        if (points.len == 0) {
            return .{
                .center = .{ 0.0, 0.0, 0.0 },
                .inv_radius = 0.1,
                .use_contraction = use_contraction,
                .t_near = 0.1,
                .t_far = 40.0,
            };
        }

        var sum = Vec3.zero;
        for (points) |p| sum = sum.add(p);
        const nf: f64 = @floatFromInt(points.len);
        const mean = sum.scale(1.0 / nf);

        var max_r: f64 = 1.0;
        var sum_r: f64 = 0.0;
        for (points) |p| {
            const d = p.sub(mean).norm();
            sum_r += d;
            if (d > max_r) max_r = d;
        }
        const mean_r = sum_r / nf;

        var cam_max_r: f64 = mean_r;
        for (poses, registered) |pose, is_reg| {
            if (!is_reg) continue;
            const d = pose.cameraCenter().sub(mean).norm();
            if (d > cam_max_r) cam_max_r = d;
        }

        const core_r = if (use_contraction)
            @max(1.0, @max(mean_r * 1.35, cam_max_r * 1.15))
        else
            @max(1.0, @max(max_r * 1.1, cam_max_r * 1.15));

        const t_far = @as(f32, @floatCast(@max(core_r * 4.5, max_r * 1.6)));
        return .{
            .center = .{
                @floatCast(mean.x),
                @floatCast(mean.y),
                @floatCast(mean.z),
            },
            .inv_radius = @floatCast(1.0 / core_r),
            .use_contraction = use_contraction,
            .t_near = @floatCast(@max(0.05, core_r * 0.02)),
            .t_far = t_far,
        };
    }

    /// Map a 3D world point into `[0, 1]^3`, returning `null` if outside uncontracted bounds.
    pub fn toUnit01(self: SceneBounds, p: [3]f32) ?[3]f32 {
        const nx = (p[0] - self.center[0]) * self.inv_radius;
        const ny = (p[1] - self.center[1]) * self.inv_radius;
        const nz = (p[2] - self.center[2]) * self.inv_radius;

        if (!self.use_contraction) {
            const ux = 0.5 * nx + 0.5;
            const uy = 0.5 * ny + 0.5;
            const uz = 0.5 * nz + 0.5;
            if (ux < 0.0 or ux >= 1.0 or uy < 0.0 or uy >= 1.0 or uz < 0.0 or uz >= 1.0) return null;
            return .{ ux, uy, uz };
        }

        const norm_sq = nx * nx + ny * ny + nz * nz;
        var cx = nx;
        var cy = ny;
        var cz = nz;
        if (norm_sq > 1.0) {
            const norm = @sqrt(norm_sq);
            const s = (2.0 - 1.0 / norm) / norm;
            cx *= s;
            cy *= s;
            cz *= s;
        }
        const eps: f32 = 1e-6;
        return .{
            std.math.clamp(0.25 * cx + 0.5, 0.0, 1.0 - eps),
            std.math.clamp(0.25 * cy + 0.5, 0.0, 1.0 - eps),
            std.math.clamp(0.25 * cz + 0.5, 0.0, 1.0 - eps),
        };
    }

    pub fn fromUnit01(self: SceneBounds, u: [3]f32) ?[3]f32 {
        const core_r = 1.0 / @max(self.inv_radius, 1e-6);
        if (!self.use_contraction) {
            return .{
                self.center[0] + (2.0 * u[0] - 1.0) * core_r,
                self.center[1] + (2.0 * u[1] - 1.0) * core_r,
                self.center[2] + (2.0 * u[2] - 1.0) * core_r,
            };
        }
        const cx = (u[0] - 0.5) * 4.0;
        const cy = (u[1] - 0.5) * 4.0;
        const cz = (u[2] - 0.5) * 4.0;
        const rc = @sqrt(cx * cx + cy * cy + cz * cz);
        if (rc >= 1.99) return null;
        const scale = if (rc <= 1.0) core_r else (core_r / ((2.0 - rc) * rc));
        return .{
            self.center[0] + cx * scale,
            self.center[1] + cy * scale,
            self.center[2] + cz * scale,
        };
    }
};

pub const SH_DIM: usize = 16;

/// Encode unit view direction into real spherical harmonics up to degree 3 (16 floats).
pub fn encodeSphericalHarmonics(d: [3]f32) [SH_DIM]f32 {
    const x = d[0];
    const y = d[1];
    const z = d[2];
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const yz = y * z;
    const xz = x * z;

    return .{
        0.28209479,
        -0.48860251 * y,
        0.48860251 * z,
        -0.48860251 * x,
        1.09254843 * xy,
        -1.09254843 * yz,
        0.31539157 * (3.0 * zz - 1.0),
        -1.09254843 * xz,
        0.54627422 * (xx - yy),
        -0.59004359 * y * (3.0 * xx - yy),
        2.89061144 * xy * z,
        -0.45704580 * y * (5.0 * zz - 1.0),
        0.37317633 * z * (5.0 * zz - 3.0),
        -0.45704580 * x * (5.0 * zz - 1.0),
        1.44530572 * z * (xx - yy),
        -0.59004359 * x * (xx - 3.0 * yy),
    };
}

pub fn generateCameraRay(k: Mat3, pose: CameraPose, px: f64, py: f64) struct { origin: [3]f32, dir: [3]f32 } {
    const fx = k.m[0][0];
    const skew = k.m[0][1];
    const cx = k.m[0][2];
    const fy = k.m[1][1];
    const cy = k.m[1][2];

    const yn = (py - cy) / fy;
    const xn = (px - cx - skew * yn) / fx;
    const d_cam = Vec3.init(xn, yn, 1.0).normalize();
    const d_world = pose.r.transpose().mulVec(d_cam).normalize();
    const c_world = pose.cameraCenter();
    return .{
        .origin = .{ @floatCast(c_world.x), @floatCast(c_world.y), @floatCast(c_world.z) },
        .dir = .{ @floatCast(d_world.x), @floatCast(d_world.y), @floatCast(d_world.z) },
    };
}

pub const HashGrid = struct {
    num_levels: usize,
    table_size: usize,
    table_mask: u32,
    n_min: f32,
    per_level_scale: f32,
    params: []f32,

    pub fn init(allocator: std.mem.Allocator, cfg: NerfConfig, rng: std.Random) !HashGrid {
        const table_size = @as(usize, 1) << cfg.table_log2;
        const total = cfg.num_levels * table_size * 2;
        const params = try allocator.alloc(f32, total);
        for (params) |*v| {
            v.* = (rng.float(f32) * 2.0 - 1.0) * 1e-4;
        }
        const scale = if (cfg.num_levels > 1)
            @exp((@log(cfg.n_max) - @log(cfg.n_min)) / @as(f32, @floatFromInt(cfg.num_levels - 1)))
        else
            1.0;

        return .{
            .num_levels = cfg.num_levels,
            .table_size = table_size,
            .table_mask = @intCast(table_size - 1),
            .n_min = cfg.n_min,
            .per_level_scale = scale,
            .params = params,
        };
    }

    pub fn deinit(self: *HashGrid, allocator: std.mem.Allocator) void {
        allocator.free(self.params);
        self.* = undefined;
    }

    pub fn encDim(self: *const HashGrid) usize {
        return self.num_levels * 2;
    }

    fn gridResAtLevel(self: *const HashGrid, level: usize) u32 {
        const scale = std.math.pow(f32, self.per_level_scale, @as(f32, @floatFromInt(level)));
        return @max(@as(u32, 2), @as(u32, @intFromFloat(@floor(self.n_min * scale))));
    }

    fn hashIndex(self: *const HashGrid, res: u32, ix: u32, iy: u32, iz: u32) usize {
        const stride = res + 1;
        const stride_u64 = @as(u64, stride);
        if (stride_u64 * stride_u64 * stride_u64 <= self.table_size) {
            return @intCast(ix + stride * (iy + stride * iz));
        }
        const h: u32 = ix *% 1 ^ (iy *% 2654435761) ^ (iz *% 805459861);
        return @intCast(h & self.table_mask);
    }

    const LevelCorner = struct {
        slot: usize,
        weight: f32,
    };

    fn trilinearCorners(self: *const HashGrid, lvl: usize, u: [3]f32) [8]LevelCorner {
        const res = self.gridResAtLevel(lvl);
        const res_f: f32 = @floatFromInt(res - 1);
        const px = std.math.clamp(u[0] * res_f, 0.0, res_f - 1e-5);
        const py = std.math.clamp(u[1] * res_f, 0.0, res_f - 1e-5);
        const pz = std.math.clamp(u[2] * res_f, 0.0, res_f - 1e-5);

        const ix: u32 = @intFromFloat(@floor(px));
        const iy: u32 = @intFromFloat(@floor(py));
        const iz: u32 = @intFromFloat(@floor(pz));
        const fx = px - @as(f32, @floatFromInt(ix));
        const fy = py - @as(f32, @floatFromInt(iy));
        const fz = pz - @as(f32, @floatFromInt(iz));
        const wx = [2]f32{ 1.0 - fx, fx };
        const wy = [2]f32{ 1.0 - fy, fy };
        const wz = [2]f32{ 1.0 - fz, fz };

        const base = lvl * self.table_size * 2;
        var out: [8]LevelCorner = undefined;
        var k: usize = 0;
        for (0..2) |dz| {
            for (0..2) |dy| {
                for (0..2) |dx| {
                    const idx = self.hashIndex(res, ix + @as(u32, @intCast(dx)), iy + @as(u32, @intCast(dy)), iz + @as(u32, @intCast(dz)));
                    out[k] = .{
                        .slot = base + idx * 2,
                        .weight = wx[dx] * wy[dy] * wz[dz],
                    };
                    k += 1;
                }
            }
        }
        return out;
    }

    pub fn encode(self: *const HashGrid, u: [3]f32, out_feat: []f32) void {
        for (0..self.num_levels) |lvl| {
            const corners = self.trilinearCorners(lvl, u);
            var acc0: f32 = 0.0;
            var acc1: f32 = 0.0;
            for (corners) |c| {
                acc0 += c.weight * self.params[c.slot + 0];
                acc1 += c.weight * self.params[c.slot + 1];
            }
            out_feat[lvl * 2 + 0] = acc0;
            out_feat[lvl * 2 + 1] = acc1;
        }
    }

    pub fn backward(self: *const HashGrid, u: [3]f32, grad_in: []const f32, grad_table: []f32) void {
        for (0..self.num_levels) |lvl| {
            const g0 = grad_in[lvl * 2 + 0];
            const g1 = grad_in[lvl * 2 + 1];
            if (g0 == 0.0 and g1 == 0.0) continue;

            const corners = self.trilinearCorners(lvl, u);
            for (corners) |c| {
                grad_table[c.slot + 0] += c.weight * g0;
                grad_table[c.slot + 1] += c.weight * g1;
            }
        }
    }
};

pub const MlpLayout = struct {
    geo_in: usize,
    geo_h: usize,
    geo_out: usize,
    col_in: usize,
    col_h: usize,
    col_out: usize,

    w0_off: usize,
    b0_off: usize,
    w1_off: usize,
    b1_off: usize,
    w2_off: usize,
    b2_off: usize,
    w3_off: usize,
    b3_off: usize,
    total_params: usize,

    pub fn init(cfg: NerfConfig) MlpLayout {
        const geo_in = cfg.num_levels * 2;
        const geo_h = cfg.geo_hidden;
        const geo_out = 1 + cfg.geo_feat_dim;
        const col_in = cfg.geo_feat_dim + SH_DIM;
        const col_h = cfg.col_hidden;
        const col_out = 3;

        var off: usize = 0;
        const w0_off = off;
        off += geo_h * geo_in;
        const b0_off = off;
        off += geo_h;
        const w1_off = off;
        off += geo_out * geo_h;
        const b1_off = off;
        off += geo_out;
        const w2_off = off;
        off += col_h * col_in;
        const b2_off = off;
        off += col_h;
        const w3_off = off;
        off += col_out * col_h;
        const b3_off = off;
        off += col_out;

        return .{
            .geo_in = geo_in,
            .geo_h = geo_h,
            .geo_out = geo_out,
            .col_in = col_in,
            .col_h = col_h,
            .col_out = col_out,
            .w0_off = w0_off,
            .b0_off = b0_off,
            .w1_off = w1_off,
            .b1_off = b1_off,
            .w2_off = w2_off,
            .b2_off = b2_off,
            .w3_off = w3_off,
            .b3_off = b3_off,
            .total_params = off,
        };
    }
};

const Activation = enum { linear, leaky_relu };

const LEAKY_ALPHA: f32 = 0.01;

fn linearForward(
    in_vec: []const f32,
    weights: []const f32,
    bias: []const f32,
    out_vec: []f32,
    act: Activation,
) void {
    const in_dim = in_vec.len;
    const out_dim = out_vec.len;
    for (0..out_dim) |o| {
        const w_row = weights[o * in_dim .. (o + 1) * in_dim];
        var sum = bias[o];
        for (in_vec, w_row) |x, w| sum += x * w;
        out_vec[o] = switch (act) {
            .linear => sum,
            .leaky_relu => if (sum > 0.0) sum else LEAKY_ALPHA * sum,
        };
    }
}

fn linearBackward(
    in_vec: []const f32,
    weights: []const f32,
    out_val: []const f32,
    grad_out: []const f32,
    grad_in: []f32,
    grad_weights: []f32,
    grad_bias: []f32,
    act: Activation,
) void {
    const in_dim = in_vec.len;
    const out_dim = out_val.len;
    @memset(grad_in, 0.0);
    for (0..out_dim) |o| {
        const go = grad_out[o];
        if (go == 0.0) continue;
        const act_deriv: f32 = switch (act) {
            .linear => 1.0,
            .leaky_relu => if (out_val[o] > 0.0) 1.0 else LEAKY_ALPHA,
        };
        const gz = go * act_deriv;
        grad_bias[o] += gz;
        const w_row = weights[o * in_dim .. (o + 1) * in_dim];
        const gw_row = grad_weights[o * in_dim .. (o + 1) * in_dim];
        for (0..in_dim) |i| {
            gw_row[i] += gz * in_vec[i];
            grad_in[i] += gz * w_row[i];
        }
    }
}

inline fn truncExp(x: f32) f32 {
    const clamped = std.math.clamp(x, -15.0, 10.0);
    return @exp(clamped);
}

inline fn truncExpDeriv(raw: f32, sigma: f32) f32 {
    return if (raw > -15.0 and raw < 10.0) sigma else 0.0;
}

inline fn sigmoidf(x: f32) f32 {
    const c = std.math.clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + @exp(-c));
}

pub const OccupancyGrid = struct {
    res: usize,
    density: []f32,
    occupied: []bool,

    pub fn init(allocator: std.mem.Allocator, res: usize) !OccupancyGrid {
        const n = res * res * res;
        const density = try allocator.alloc(f32, n);
        errdefer allocator.free(density);
        @memset(density, 0.12);
        const occupied = try allocator.alloc(bool, n);
        @memset(occupied, true);
        return .{ .res = res, .density = density, .occupied = occupied };
    }

    pub fn jitterForStep(step: usize) [3]f32 {
        if (step == 0) return .{ 0.5, 0.5, 0.5 };
        const sf: f32 = @floatFromInt(step);
        return .{
            @mod(sf * 0.75487767 + 0.5, 1.0),
            @mod(sf * 0.56984029 + 0.5, 1.0),
            @mod(sf * 0.43015971 + 0.5, 1.0),
        };
    }

    pub fn deinit(self: *OccupancyGrid, allocator: std.mem.Allocator) void {
        allocator.free(self.density);
        allocator.free(self.occupied);
        self.* = undefined;
    }

    pub fn isOccupiedUnit01(self: *const OccupancyGrid, u: [3]f32) bool {
        const r_f: f32 = @floatFromInt(self.res);
        const ix = @min(self.res - 1, @as(usize, @intFromFloat(@max(0.0, u[0] * r_f))));
        const iy = @min(self.res - 1, @as(usize, @intFromFloat(@max(0.0, u[1] * r_f))));
        const iz = @min(self.res - 1, @as(usize, @intFromFloat(@max(0.0, u[2] * r_f))));
        return self.occupied[ix + self.res * (iy + self.res * iz)];
    }

    pub fn isOccupied3x3Unit01(self: *const OccupancyGrid, u: [3]f32) bool {
        const r = self.res;
        const r_f: f32 = @floatFromInt(r);
        const ix = @min(r - 1, @as(usize, @intFromFloat(@max(0.0, u[0] * r_f))));
        const iy = @min(r - 1, @as(usize, @intFromFloat(@max(0.0, u[1] * r_f))));
        const iz = @min(r - 1, @as(usize, @intFromFloat(@max(0.0, u[2] * r_f))));
        const z0 = if (iz > 0) iz - 1 else 0;
        const z1 = @min(r - 1, iz + 1);
        const y0 = if (iy > 0) iy - 1 else 0;
        const y1 = @min(r - 1, iy + 1);
        const x0 = if (ix > 0) ix - 1 else 0;
        const x1 = @min(r - 1, ix + 1);
        for (z0..z1 + 1) |zz| {
            for (y0..y1 + 1) |yy| {
                for (x0..x1 + 1) |xx| {
                    if (self.occupied[xx + r * (yy + r * zz)]) return true;
                }
            }
        }
        return false;
    }

    pub fn updateFromModel(self: *OccupancyGrid, model: *const NerfModel, decay: f32, threshold: f32) void {
        const res = self.res;
        const inv_r = 1.0 / @as(f32, @floatFromInt(res));
        const jit = jitterForStep(model.step);
        for (0..res) |iz| {
            for (0..res) |iy| {
                for (0..res) |ix| {
                    const idx = ix + res * (iy + res * iz);
                    if (self.density[idx] < 0.0) {
                        self.occupied[idx] = false;
                        continue;
                    }
                    const u = [3]f32{
                        (@as(f32, @floatFromInt(ix)) + jit[0]) * inv_r,
                        (@as(f32, @floatFromInt(iy)) + jit[1]) * inv_r,
                        (@as(f32, @floatFromInt(iz)) + jit[2]) * inv_r,
                    };
                    const sigma = model.evalDensityUnit01(u);
                    const blended = @max(self.density[idx] * decay, sigma);
                    self.density[idx] = blended;
                    self.occupied[idx] = blended > threshold;
                }
            }
        }
    }

    pub fn seedFromViews(
        self: *OccupancyGrid,
        bounds: SceneBounds,
        k: Mat3,
        images: []const ImageData,
        poses: []const CameraPose,
        registered: []const bool,
        depth_maps: ?[]const []f32,
    ) void {
        const res = self.res;
        const inv_r = 1.0 / @as(f32, @floatFromInt(res));
        const core_r = 1.0 / @max(bounds.inv_radius, 1e-6);
        const fx: f32 = @floatCast(k.m[0][0]);
        const fy: f32 = @floatCast(k.m[1][1]);
        const cx_k: f32 = @floatCast(k.m[0][2]);
        const cy_k: f32 = @floatCast(k.m[1][2]);
        const scale_u: f32 = if (bounds.use_contraction) 4.0 else 2.0;
        const base_cell_r = 0.8660254 * inv_r * scale_u * core_r;

        for (0..res) |iz| {
            for (0..res) |iy| {
                for (0..res) |ix| {
                    const idx = ix + res * (iy + res * iz);
                    const u = [3]f32{
                        (@as(f32, @floatFromInt(ix)) + 0.5) * inv_r,
                        (@as(f32, @floatFromInt(iy)) + 0.5) * inv_r,
                        (@as(f32, @floatFromInt(iz)) + 0.5) * inv_r,
                    };
                    const p = bounds.fromUnit01(u) orelse {
                        self.density[idx] = -1.0;
                        self.occupied[idx] = false;
                        continue;
                    };

                    const p_vec = Vec3.init(p[0], p[1], p[2]);
                    const dx_c = (u[0] - 0.5) * scale_u;
                    const dy_c = (u[1] - 0.5) * scale_u;
                    const dz_c = (u[2] - 0.5) * scale_u;
                    const rc = @sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c);
                    const jac = if (bounds.use_contraction and rc > 1.0)
                        1.0 / ((2.0 - rc) * (2.0 - rc))
                    else
                        1.0;
                    const r_cell = base_cell_r * jac;

                    var in_any_frustum = false;
                    var has_depth_cov = false;
                    var behind_all_depths = true;
                    var in_front_all_depths = true;
                    var near_any_surface = false;

                    for (poses, images, registered, 0..) |pose, img, is_reg, ci| {
                        if (!is_reg) continue;
                        const w_f: f32 = @floatFromInt(img.width);
                        const h_f: f32 = @floatFromInt(img.height);
                        const pc = pose.r.mulVec(p_vec).add(pose.t);
                        const zc: f32 = @floatCast(pc.z);
                        if (zc + r_cell <= 0.25) continue;

                        const z_eff = @max(zc, 0.25);
                        const up = fx * @as(f32, @floatCast(pc.x)) / z_eff + cx_k;
                        const vp = fy * @as(f32, @floatCast(pc.y)) / z_eff + cy_k;
                        const margin_x = r_cell * fx / z_eff + 0.08 * w_f;
                        const margin_y = r_cell * fy / z_eff + 0.08 * h_f;
                        if (up >= -margin_x and up <= w_f + margin_x and vp >= -margin_y and vp <= h_f + margin_y) {
                            in_any_frustum = true;
                        }

                        if (depth_maps) |dmaps| {
                            if (zc > 0.25 and up >= 0.0 and up < w_f and vp >= 0.0 and vp < h_f) {
                                const px_i: usize = @min(img.width - 1, @as(usize, @intFromFloat(up)));
                                const py_i: usize = @min(img.height - 1, @as(usize, @intFromFloat(vp)));
                                const z_gt = dmaps[ci][py_i * img.width + px_i];
                                if (z_gt > 0.0) {
                                    const cam_dist: f32 = @floatCast(p_vec.sub(pose.cameraCenter()).norm());
                                    has_depth_cov = true;
                                    if (cam_dist <= z_gt + r_cell + 0.80) behind_all_depths = false;
                                    if (cam_dist >= z_gt - r_cell - 0.50) in_front_all_depths = false;
                                    if (@abs(cam_dist - z_gt) <= r_cell + 0.45) near_any_surface = true;
                                }
                            }
                        }
                    }

                    if (!in_any_frustum or (has_depth_cov and rc <= 1.05 and behind_all_depths)) {
                        self.density[idx] = -1.0;
                        self.occupied[idx] = false;
                    } else if (near_any_surface or !has_depth_cov) {
                        self.density[idx] = 0.12;
                        self.occupied[idx] = true;
                    } else if (in_front_all_depths) {
                        self.density[idx] = 0.01;
                        self.occupied[idx] = false;
                    } else {
                        self.density[idx] = 0.06;
                        self.occupied[idx] = true;
                    }
                }
            }
        }
    }
};

const AdamBuffers = struct {
    m: []f32,
    v: []f32,

    fn init(allocator: std.mem.Allocator, len: usize) !AdamBuffers {
        const m = try allocator.alloc(f32, len);
        errdefer allocator.free(m);
        const v = try allocator.alloc(f32, len);
        @memset(m, 0.0);
        @memset(v, 0.0);
        return .{ .m = m, .v = v };
    }

    fn deinit(self: *AdamBuffers, allocator: std.mem.Allocator) void {
        allocator.free(self.m);
        allocator.free(self.v);
        self.* = undefined;
    }
};

fn adamStepSlice(
    params: []f32,
    grads: []const f32,
    adam: *AdamBuffers,
    lr: f32,
    weight_decay: f32,
    step_t: usize,
) void {
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    const tf: f32 = @floatFromInt(step_t);
    const bc1 = 1.0 - std.math.pow(f32, beta1, tf);
    const bc2 = 1.0 - std.math.pow(f32, beta2, tf);
    const inv_bc1 = 1.0 / bc1;
    const inv_bc2 = 1.0 / bc2;

    for (0..params.len) |i| {
        const g = grads[i];
        if (g == 0.0 and adam.m[i] == 0.0) continue;
        const grad = g + weight_decay * params[i];
        const m = beta1 * adam.m[i] + (1.0 - beta1) * grad;
        const v = beta2 * adam.v[i] + (1.0 - beta2) * grad * grad;
        adam.m[i] = m;
        adam.v[i] = v;
        const m_hat = m * inv_bc1;
        const v_hat = v * inv_bc2;
        params[i] -= lr * m_hat / (@sqrt(v_hat) + eps);
    }
}

const PointEvalCache = struct {
    u: [3]f32,
    t: f32,
    delta: f32,
    geo_in: [32]f32,
    geo_h: [64]f32,
    sigma_raw: f32,
    sigma: f32,
    col_in: [64]f32,
    col_h: [64]f32,
    col_logits: [3]f32,
    rgb: [3]f32,
    alpha: f32,
    transmittance: f32,
    weight: f32,
};

pub const RayRenderOutput = struct {
    rgb: [3]f32,
    depth: f32,
    transmittance_rem: f32,
    opacity: f32,
};

pub const NerfModel = struct {
    config: NerfConfig,
    bounds: SceneBounds,
    layout: MlpLayout,
    hash_grid: HashGrid,
    mlp_params: []f32,
    occ_grid: OccupancyGrid,
    adam_mlp: AdamBuffers,
    adam_hash: AdamBuffers,
    step: usize,

    pub fn init(allocator: std.mem.Allocator, cfg: NerfConfig, bounds: SceneBounds) !NerfModel {
        var prng = std.Random.DefaultPrng.init(cfg.seed);
        const rng = prng.random();

        const layout = MlpLayout.init(cfg);
        var hash_grid = try HashGrid.init(allocator, cfg, rng);
        errdefer hash_grid.deinit(allocator);

        const mlp_params = try allocator.alloc(f32, layout.total_params);
        errdefer allocator.free(mlp_params);
        initMlpWeights(mlp_params, layout, rng);

        var occ_grid = try OccupancyGrid.init(allocator, cfg.occ_grid_res);
        errdefer occ_grid.deinit(allocator);

        var adam_mlp = try AdamBuffers.init(allocator, layout.total_params);
        errdefer adam_mlp.deinit(allocator);

        var adam_hash = try AdamBuffers.init(allocator, hash_grid.params.len);
        errdefer adam_hash.deinit(allocator);

        return .{
            .config = cfg,
            .bounds = bounds,
            .layout = layout,
            .hash_grid = hash_grid,
            .mlp_params = mlp_params,
            .occ_grid = occ_grid,
            .adam_mlp = adam_mlp,
            .adam_hash = adam_hash,
            .step = 0,
        };
    }

    pub fn deinit(self: *NerfModel, allocator: std.mem.Allocator) void {
        self.hash_grid.deinit(allocator);
        allocator.free(self.mlp_params);
        self.occ_grid.deinit(allocator);
        self.adam_mlp.deinit(allocator);
        self.adam_hash.deinit(allocator);
        self.* = undefined;
    }

    fn initMlpWeights(params: []f32, l: MlpLayout, rng: std.Random) void {
        const pairs = [_][4]usize{
            .{ l.w0_off, l.b0_off, l.geo_in, l.geo_h },
            .{ l.w1_off, l.b1_off, l.geo_h, l.geo_out },
            .{ l.w2_off, l.b2_off, l.col_in, l.col_h },
            .{ l.w3_off, l.b3_off, l.col_h, l.col_out },
        };
        for (pairs) |spec| {
            const w_off = spec[0];
            const b_off = spec[1];
            const fan_in = spec[2];
            const fan_out = spec[3];
            const bound = @sqrt(6.0 / @as(f32, @floatFromInt(fan_in + fan_out)));
            for (params[w_off .. w_off + fan_in * fan_out]) |*w| {
                w.* = (rng.float(f32) * 2.0 - 1.0) * bound;
            }
            @memset(params[b_off .. b_off + fan_out], 0.0);
        }
        // Negative density bias keeps empty space initially transparent
        params[l.b1_off] = -1.5;
    }

    pub fn evalDensityUnit01(self: *const NerfModel, u: [3]f32) f32 {
        const l = self.layout;
        var geo_in_buf: [32]f32 = undefined;
        var geo_h_buf: [64]f32 = undefined;
        var geo_out_buf: [64]f32 = undefined;

        const geo_in = geo_in_buf[0..l.geo_in];
        const geo_h = geo_h_buf[0..l.geo_h];
        const geo_out = geo_out_buf[0..l.geo_out];

        self.hash_grid.encode(u, geo_in);
        linearForward(geo_in, self.mlp_params[l.w0_off .. l.w0_off + l.geo_h * l.geo_in], self.mlp_params[l.b0_off .. l.b0_off + l.geo_h], geo_h, .leaky_relu);
        linearForward(geo_h, self.mlp_params[l.w1_off .. l.w1_off + l.geo_out * l.geo_h], self.mlp_params[l.b1_off .. l.b1_off + l.geo_out], geo_out, .linear);
        return truncExp(geo_out[0]);
    }

    fn evalSample(self: *const NerfModel, u: [3]f32, sh: [SH_DIM]f32, cache: *PointEvalCache) void {
        const l = self.layout;
        const geo_in = cache.geo_in[0..l.geo_in];
        const geo_h = cache.geo_h[0..l.geo_h];
        var geo_out_buf: [64]f32 = undefined;
        const geo_out = geo_out_buf[0..l.geo_out];

        self.hash_grid.encode(u, geo_in);
        linearForward(geo_in, self.mlp_params[l.w0_off .. l.w0_off + l.geo_h * l.geo_in], self.mlp_params[l.b0_off .. l.b0_off + l.geo_h], geo_h, .leaky_relu);
        linearForward(geo_h, self.mlp_params[l.w1_off .. l.w1_off + l.geo_out * l.geo_h], self.mlp_params[l.b1_off .. l.b1_off + l.geo_out], geo_out, .linear);

        cache.u = u;
        cache.sigma_raw = geo_out[0];
        cache.sigma = truncExp(geo_out[0]);

        const col_in = cache.col_in[0..l.col_in];
        @memcpy(col_in[0..self.config.geo_feat_dim], geo_out[1..l.geo_out]);
        @memcpy(col_in[self.config.geo_feat_dim .. self.config.geo_feat_dim + SH_DIM], &sh);

        const col_h = cache.col_h[0..l.col_h];
        linearForward(col_in, self.mlp_params[l.w2_off .. l.w2_off + l.col_h * l.col_in], self.mlp_params[l.b2_off .. l.b2_off + l.col_h], col_h, .leaky_relu);
        linearForward(col_h, self.mlp_params[l.w3_off .. l.w3_off + l.col_out * l.col_h], self.mlp_params[l.b3_off .. l.b3_off + l.col_out], &cache.col_logits, .linear);

        for (0..3) |c| cache.rgb[c] = sigmoidf(cache.col_logits[c]);
    }

    fn backwardSample(
        self: *const NerfModel,
        cache: *const PointEvalCache,
        d_loss_rgb: [3]f32,
        d_loss_sigma: f32,
        grad_mlp: []f32,
        grad_hash: []f32,
    ) void {
        const l = self.layout;

        var d_col_logits: [3]f32 = undefined;
        for (0..3) |c| {
            const s = cache.rgb[c];
            d_col_logits[c] = d_loss_rgb[c] * s * (1.0 - s);
        }

        var d_col_h_buf: [64]f32 = undefined;
        const d_col_h = d_col_h_buf[0..l.col_h];
        linearBackward(
            cache.col_h[0..l.col_h],
            self.mlp_params[l.w3_off .. l.w3_off + l.col_out * l.col_h],
            &cache.col_logits,
            &d_col_logits,
            d_col_h,
            grad_mlp[l.w3_off .. l.w3_off + l.col_out * l.col_h],
            grad_mlp[l.b3_off .. l.b3_off + l.col_out],
            .linear,
        );

        var d_col_in_buf: [64]f32 = undefined;
        const d_col_in = d_col_in_buf[0..l.col_in];
        linearBackward(
            cache.col_in[0..l.col_in],
            self.mlp_params[l.w2_off .. l.w2_off + l.col_h * l.col_in],
            cache.col_h[0..l.col_h],
            d_col_h,
            d_col_in,
            grad_mlp[l.w2_off .. l.w2_off + l.col_h * l.col_in],
            grad_mlp[l.b2_off .. l.b2_off + l.col_h],
            .leaky_relu,
        );

        var d_geo_out_buf: [64]f32 = undefined;
        const d_geo_out = d_geo_out_buf[0..l.geo_out];
        d_geo_out[0] = d_loss_sigma * truncExpDeriv(cache.sigma_raw, cache.sigma);
        @memcpy(d_geo_out[1..l.geo_out], d_col_in[0..self.config.geo_feat_dim]);

        var d_geo_h_buf: [64]f32 = undefined;
        const d_geo_h = d_geo_h_buf[0..l.geo_h];
        linearBackward(
            cache.geo_h[0..l.geo_h],
            self.mlp_params[l.w1_off .. l.w1_off + l.geo_out * l.geo_h],
            d_geo_out, // linear activation ignores out_val
            d_geo_out,
            d_geo_h,
            grad_mlp[l.w1_off .. l.w1_off + l.geo_out * l.geo_h],
            grad_mlp[l.b1_off .. l.b1_off + l.geo_out],
            .linear,
        );

        var d_geo_in_buf: [32]f32 = undefined;
        const d_geo_in = d_geo_in_buf[0..l.geo_in];
        linearBackward(
            cache.geo_in[0..l.geo_in],
            self.mlp_params[l.w0_off .. l.w0_off + l.geo_h * l.geo_in],
            cache.geo_h[0..l.geo_h],
            d_geo_h,
            d_geo_in,
            grad_mlp[l.w0_off .. l.w0_off + l.geo_h * l.geo_in],
            grad_mlp[l.b0_off .. l.b0_off + l.geo_h],
            .leaky_relu,
        );

        self.hash_grid.backward(cache.u, d_geo_in, grad_hash);
    }

    pub fn renderRay(
        self: *const NerfModel,
        origin: [3]f32,
        dir: [3]f32,
        bg: [3]f32,
        jitter: f32,
        skip_empty: bool,
    ) RayRenderOutput {
        const sh = encodeSphericalHarmonics(dir);
        const n_max = self.config.max_ray_samples;

        var t = self.bounds.t_near;
        var transmittance: f32 = 1.0;
        var rgb_acc = [3]f32{ 0.0, 0.0, 0.0 };
        var depth_acc: f32 = 0.0;
        var hits: usize = 0;
        var cache: PointEvalCache = undefined;

        for (0..n_max * 3) |_| {
            if (t >= self.bounds.t_far or transmittance < 1e-4 or hits >= n_max) break;
            const dt = self.bounds.rayStepDt(t, self.config.dt_min, n_max);
            const t_mid = t + dt * (0.25 + 0.5 * jitter);
            const pos = [3]f32{
                origin[0] + t_mid * dir[0],
                origin[1] + t_mid * dir[1],
                origin[2] + t_mid * dir[2],
            };
            t += dt;

            const u = self.bounds.toUnit01(pos) orelse continue;
            if (skip_empty and !self.occ_grid.isOccupiedUnit01(u)) continue;

            self.evalSample(u, sh, &cache);
            hits += 1;
            const alpha = 1.0 - @exp(-cache.sigma * dt);
            if (alpha < 1e-5) continue;

            const w = transmittance * alpha;
            for (0..3) |c| rgb_acc[c] += w * cache.rgb[c];
            depth_acc += w * t_mid;
            transmittance *= (1.0 - alpha);
        }

        for (0..3) |c| rgb_acc[c] += transmittance * bg[c];
        return .{
            .rgb = rgb_acc,
            .depth = depth_acc,
            .transmittance_rem = transmittance,
            .opacity = 1.0 - transmittance,
        };
    }

    pub fn rayLossAndBackward(
        self: *const NerfModel,
        origin: [3]f32,
        dir: [3]f32,
        target_rgb: [3]f32,
        target_depth: ?f32,
        bg: [3]f32,
        jitter: f32,
        skip_empty: bool,
        grad_mlp: []f32,
        grad_hash: []f32,
    ) f32 {
        const sh = encodeSphericalHarmonics(dir);
        const n_max = self.config.max_ray_samples;

        var caches_buf: [256]PointEvalCache = undefined;
        const limit = @min(n_max, caches_buf.len);
        var count: usize = 0;

        var t = self.bounds.t_near;
        var transmittance: f32 = 1.0;
        var rgb_acc = [3]f32{ 0.0, 0.0, 0.0 };
        var depth_acc: f32 = 0.0;
        var pref_w: f32 = 0.0;
        var pref_wt: f32 = 0.0;
        var dist_loss: f32 = 0.0;

        for (0..limit * 3) |_| {
            if (t >= self.bounds.t_far or transmittance < 1e-4 or count >= limit) break;
            const dt = self.bounds.rayStepDt(t, self.config.dt_min, n_max);
            const t_mid = t + dt * (0.25 + 0.5 * jitter);
            const pos = [3]f32{
                origin[0] + t_mid * dir[0],
                origin[1] + t_mid * dir[1],
                origin[2] + t_mid * dir[2],
            };
            t += dt;

            const u = self.bounds.toUnit01(pos) orelse continue;
            if (skip_empty and !self.occ_grid.isOccupiedUnit01(u)) continue;

            var c_slot = &caches_buf[count];
            self.evalSample(u, sh, c_slot);
            c_slot.t = t_mid;
            c_slot.delta = dt;
            c_slot.transmittance = transmittance;
            const alpha = 1.0 - @exp(-c_slot.sigma * dt);
            c_slot.alpha = alpha;
            const w = transmittance * alpha;
            c_slot.weight = w;

            for (0..3) |c| rgb_acc[c] += w * c_slot.rgb[c];
            depth_acc += w * t_mid;
            dist_loss += 2.0 * w * (t_mid * pref_w - pref_wt) + (1.0 / 3.0) * w * w * dt;
            pref_w += w;
            pref_wt += w * t_mid;
            transmittance *= (1.0 - alpha);
            count += 1;
        }

        for (0..3) |c| rgb_acc[c] += transmittance * bg[c];

        var loss: f32 = 0.0;
        var d_loss_rgb_hat: [3]f32 = undefined;
        const inv3: f32 = 1.0 / 3.0;
        for (0..3) |c| {
            const diff = rgb_acc[c] - target_rgb[c];
            loss += diff * diff * inv3;
            d_loss_rgb_hat[c] = (2.0 * inv3) * diff;
        }

        var d_loss_depth_hat: f32 = 0.0;
        if (target_depth) |z_gt| {
            if (self.config.depth_loss_weight > 0.0 and z_gt > 0.0) {
                const inv_z = 1.0 / @max(z_gt, 1.0);
                const dz = (depth_acc - z_gt) * inv_z;
                loss += self.config.depth_loss_weight * dz * dz;
                d_loss_depth_hat = 2.0 * self.config.depth_loss_weight * dz * inv_z;
            }
        }
        const dist_w = self.config.distortion_weight;
        if (dist_w > 0.0) loss += dist_w * dist_loss;

        var suffix_all = transmittance * (d_loss_rgb_hat[0] * bg[0] + d_loss_rgb_hat[1] * bg[1] + d_loss_rgb_hat[2] * bg[2]);
        var suf_w: f32 = 0.0;
        var suf_wt: f32 = 0.0;

        var rev = count;
        while (rev > 0) {
            rev -= 1;
            const s = &caches_buf[rev];
            const t_next = s.transmittance * (1.0 - s.alpha);
            var g_w = d_loss_rgb_hat[0] * s.rgb[0] + d_loss_rgb_hat[1] * s.rgb[1] + d_loss_rgb_hat[2] * s.rgb[2];
            if (d_loss_depth_hat != 0.0) g_w += d_loss_depth_hat * s.t;
            if (dist_w > 0.0) {
                const bw = pref_w - suf_w - s.weight;
                const bwt = pref_wt - suf_wt - s.weight * s.t;
                const g_dist = 2.0 * (s.t * (bw - suf_w) - (bwt - suf_wt) + (1.0 / 3.0) * s.weight * s.delta);
                g_w += dist_w * g_dist;
            }

            const d_sigma = s.delta * (t_next * g_w - suffix_all);
            suffix_all += s.weight * g_w;
            suf_w += s.weight;
            suf_wt += s.weight * s.t;

            const d_rgb_sample = [3]f32{
                s.weight * d_loss_rgb_hat[0],
                s.weight * d_loss_rgb_hat[1],
                s.weight * d_loss_rgb_hat[2],
            };
            self.backwardSample(s, d_rgb_sample, d_sigma, grad_mlp, grad_hash);
        }

        return loss;
    }

    pub fn applyGradients(
        self: *NerfModel,
        grad_mlp: []const f32,
        grad_hash: []const f32,
        lr_scale: f32,
    ) void {
        self.step += 1;
        const lr = self.config.lr * lr_scale;
        adamStepSlice(self.mlp_params, grad_mlp, &self.adam_mlp, lr, self.config.weight_decay, self.step);
        adamStepSlice(self.hash_grid.params, grad_hash, &self.adam_hash, lr, 0.0, self.step);
    }

    pub fn renderView(
        self: *const NerfModel,
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        k: Mat3,
        pose: CameraPose,
        skip_empty: bool,
    ) !RenderedView {
        const w_us: usize = width;
        const h_us: usize = height;
        const n_pix = w_us * h_us;
        const rgb = try allocator.alloc(u8, n_pix * 3);
        errdefer allocator.free(rgb);
        const depth = try allocator.alloc(f32, n_pix);
        errdefer allocator.free(depth);
        const covered = try allocator.alloc(bool, n_pix);
        errdefer allocator.free(covered);

        const n_threads = @min(@max(@as(usize, 1), self.config.num_threads), h_us);
        if (n_threads <= 1) {
            renderRowsWorker(self, width, 0, h_us, k, pose, skip_empty, rgb, depth, covered);
        } else {
            var next_row = std.atomic.Value(usize).init(0);
            const threads = try allocator.alloc(std.Thread, n_threads);
            defer allocator.free(threads);
            for (threads) |*th| {
                th.* = try std.Thread.spawn(.{}, renderRowLoop, .{
                    self, width, h_us, k, pose, skip_empty, rgb, depth, covered, &next_row,
                });
            }
            for (threads) |th| th.join();
        }

        return .{
            .width = width,
            .height = height,
            .rgb = rgb,
            .depth = depth,
            .covered = covered,
        };
    }

    fn renderRowLoop(
        self: *const NerfModel,
        width: u32,
        height: usize,
        k: Mat3,
        pose: CameraPose,
        skip_empty: bool,
        rgb: []u8,
        depth: []f32,
        covered: []bool,
        next_row: *std.atomic.Value(usize),
    ) void {
        while (true) {
            const y = next_row.fetchAdd(1, .monotonic);
            if (y >= height) break;
            renderRowsWorker(self, width, y, y + 1, k, pose, skip_empty, rgb, depth, covered);
        }
    }

    fn renderRowsWorker(
        self: *const NerfModel,
        width: u32,
        y0: usize,
        y1: usize,
        k: Mat3,
        pose: CameraPose,
        skip_empty: bool,
        rgb: []u8,
        depth: []f32,
        covered: []bool,
    ) void {
        const w_us: usize = width;
        for (y0..y1) |y| {
            for (0..w_us) |x| {
                const ray = generateCameraRay(k, pose, @as(f64, @floatFromInt(x)) + 0.5, @as(f64, @floatFromInt(y)) + 0.5);
                const out = self.renderRay(ray.origin, ray.dir, self.config.bg_color, 0.5, skip_empty);
                const idx = y * w_us + x;
                for (0..3) |c| {
                    rgb[idx * 3 + c] = @intFromFloat(std.math.clamp(out.rgb[c] * 255.0 + 0.5, 0.0, 255.0));
                }
                depth[idx] = out.depth;
                covered[idx] = true;
            }
        }
    }

    pub const ExtractMeshConfig = struct {
        max_res: usize = 128,
        iso_threshold: f32 = 0.0,
        num_threads: usize = 1,
        k: ?Mat3 = null,
        img_width: u32 = 0,
        img_height: u32 = 0,
    };

    const ExtractCamView = struct {
        pose: CameraPose,
        center: Vec3,
    };

    fn dsuFind(parent: []u32, x0: u32) u32 {
        var x = x0;
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    fn dsuUnion(parent: []u32, a: u32, b: u32) void {
        const ra = dsuFind(parent, a);
        const rb = dsuFind(parent, b);
        if (ra != rb) parent[rb] = ra;
    }

    fn smoothScalarField(allocator: std.mem.Allocator, field: []f32, nx: usize, ny: usize, nz: usize) !void {
        if (nx < 3 or ny < 3 or nz < 3) return;
        const tmp = try allocator.alloc(f32, field.len);
        defer allocator.free(tmp);

        for (0..nz) |iz| {
            for (0..ny) |iy| {
                const row = (iz * ny + iy) * nx;
                tmp[row] = field[row];
                tmp[row + nx - 1] = field[row + nx - 1];
                for (1..nx - 1) |ix| {
                    tmp[row + ix] = 0.10 * field[row + ix - 1] + 0.80 * field[row + ix] + 0.10 * field[row + ix + 1];
                }
            }
        }
        for (0..nz) |iz| {
            const sl = iz * ny * nx;
            for (0..nx) |ix| {
                field[sl + ix] = tmp[sl + ix];
                field[sl + (ny - 1) * nx + ix] = tmp[sl + (ny - 1) * nx + ix];
            }
            for (1..ny - 1) |iy| {
                for (0..nx) |ix| {
                    const idx = sl + iy * nx + ix;
                    field[idx] = 0.10 * tmp[idx - nx] + 0.80 * tmp[idx] + 0.10 * tmp[idx + nx];
                }
            }
        }
        const stride_z = ny * nx;
        @memcpy(tmp[0..stride_z], field[0..stride_z]);
        @memcpy(tmp[(nz - 1) * stride_z .. nz * stride_z], field[(nz - 1) * stride_z .. nz * stride_z]);
        for (1..nz - 1) |iz| {
            for (0..stride_z) |xy| {
                const idx = iz * stride_z + xy;
                tmp[idx] = 0.10 * field[idx - stride_z] + 0.80 * field[idx] + 0.10 * field[idx + stride_z];
            }
        }
        @memcpy(field, tmp);
    }

    fn sampleFieldTrilinear(field: []const f32, nx: usize, ny: usize, nz: usize, gx_in: f64, gy_in: f64, gz_in: f64) f32 {
        const gx = std.math.clamp(gx_in, 0.0, @as(f64, @floatFromInt(nx - 1)) - 1e-4);
        const gy = std.math.clamp(gy_in, 0.0, @as(f64, @floatFromInt(ny - 1)) - 1e-4);
        const gz = std.math.clamp(gz_in, 0.0, @as(f64, @floatFromInt(nz - 1)) - 1e-4);
        const x0: usize = @intFromFloat(@floor(gx));
        const y0: usize = @intFromFloat(@floor(gy));
        const z0: usize = @intFromFloat(@floor(gz));
        const x1 = @min(nx - 1, x0 + 1);
        const y1 = @min(ny - 1, y0 + 1);
        const z1 = @min(nz - 1, z0 + 1);
        const fx: f32 = @floatCast(gx - @as(f64, @floatFromInt(x0)));
        const fy: f32 = @floatCast(gy - @as(f64, @floatFromInt(y0)));
        const fz: f32 = @floatCast(gz - @as(f64, @floatFromInt(z0)));
        const stride_z = ny * nx;
        const c00 = field[z0 * stride_z + y0 * nx + x0] * (1.0 - fx) + field[z0 * stride_z + y0 * nx + x1] * fx;
        const c01 = field[z0 * stride_z + y1 * nx + x0] * (1.0 - fx) + field[z0 * stride_z + y1 * nx + x1] * fx;
        const c10 = field[z1 * stride_z + y0 * nx + x0] * (1.0 - fx) + field[z1 * stride_z + y0 * nx + x1] * fx;
        const c11 = field[z1 * stride_z + y1 * nx + x0] * (1.0 - fx) + field[z1 * stride_z + y1 * nx + x1] * fx;
        const c0 = c00 * (1.0 - fy) + c01 * fy;
        const c1 = c10 * (1.0 - fy) + c11 * fy;
        return c0 * (1.0 - fz) + c1 * fz;
    }

    fn evalFieldOutwardNormal(field: []const f32, nx: usize, ny: usize, nz: usize, min_p: Vec3, step: Vec3, v: Vec3) Vec3 {
        const gx = (v.x - min_p.x) / step.x;
        const gy = (v.y - min_p.y) / step.y;
        const gz = (v.z - min_p.z) / step.z;
        const dx = (sampleFieldTrilinear(field, nx, ny, nz, gx + 0.5, gy, gz) -
            sampleFieldTrilinear(field, nx, ny, nz, gx - 0.5, gy, gz)) / @as(f32, @floatCast(step.x));
        const dy = (sampleFieldTrilinear(field, nx, ny, nz, gx, gy + 0.5, gz) -
            sampleFieldTrilinear(field, nx, ny, nz, gx, gy - 0.5, gz)) / @as(f32, @floatCast(step.y));
        const dz = (sampleFieldTrilinear(field, nx, ny, nz, gx, gy, gz + 0.5) -
            sampleFieldTrilinear(field, nx, ny, nz, gx, gy, gz - 0.5)) / @as(f32, @floatCast(step.z));
        const g = Vec3.init(-dx, -dy, -dz);
        return if (g.normSq() > 1e-12) g.normalize() else Vec3.init(0.0, 0.0, -1.0);
    }

    fn colorVerticesWorker(
        self: *const NerfModel,
        verts: []const Vec3,
        best_dirs: []const [3]f32,
        vox_s: f32,
        colors: []Color,
        next_idx: *std.atomic.Value(usize),
    ) void {
        var cache: PointEvalCache = undefined;
        const step_d: f32 = 0.40 * vox_s;
        const micro_dt: f32 = @max(step_d, 0.015);
        while (true) {
            const vi = next_idx.fetchAdd(1, .monotonic);
            if (vi >= verts.len) break;
            const v = verts[vi];
            const v_f = [3]f32{ @floatCast(v.x), @floatCast(v.y), @floatCast(v.z) };
            const dir = best_dirs[vi];
            const sh_v = encodeSphericalHarmonics(dir);

            var col_acc = [3]f32{ 0.0, 0.0, 0.0 };
            var trans: f32 = 1.0;
            for (0..5) |si| {
                const off = (@as(f32, @floatFromInt(si)) - 1.5) * step_d;
                const sp = [3]f32{
                    v_f[0] + off * dir[0],
                    v_f[1] + off * dir[1],
                    v_f[2] + off * dir[2],
                };
                const su = self.bounds.toUnit01(sp) orelse continue;
                self.evalSample(su, sh_v, &cache);
                const a = 1.0 - @exp(-cache.sigma * micro_dt);
                const w = trans * a;
                for (0..3) |ch| col_acc[ch] += w * cache.rgb[ch];
                trans *= (1.0 - a);
            }
            const cov = 1.0 - trans;
            if (cov > 0.05) {
                for (0..3) |ch| col_acc[ch] /= cov;
            } else {
                const u = self.bounds.toUnit01(v_f) orelse [3]f32{ 0.5, 0.5, 0.5 };
                self.evalSample(u, sh_v, &cache);
                col_acc = cache.rgb;
            }
            colors[vi] = .{
                .r = @intFromFloat(std.math.clamp(col_acc[0] * 255.0 + 0.5, 0.0, 255.0)),
                .g = @intFromFloat(std.math.clamp(col_acc[1] * 255.0 + 0.5, 0.0, 255.0)),
                .b = @intFromFloat(std.math.clamp(col_acc[2] * 255.0 + 0.5, 0.0, 255.0)),
            };
        }
    }

    pub fn extractMesh(
        self: *const NerfModel,
        allocator: std.mem.Allocator,
        points: []const Vec3,
        poses: []const CameraPose,
        registered: []const bool,
        cfg: ExtractMeshConfig,
    ) !mesh_mod.Mesh {
        var min_p: Vec3 = undefined;
        var max_p: Vec3 = undefined;
        if (points.len >= 4) {
            const xs = try allocator.alloc(f64, points.len);
            defer allocator.free(xs);
            const ys = try allocator.alloc(f64, points.len);
            defer allocator.free(ys);
            const zs = try allocator.alloc(f64, points.len);
            defer allocator.free(zs);
            for (points, 0..) |p, i| {
                xs[i] = p.x;
                ys[i] = p.y;
                zs[i] = p.z;
            }
            std.sort.heap(f64, xs, {}, std.sort.asc(f64));
            std.sort.heap(f64, ys, {}, std.sort.asc(f64));
            std.sort.heap(f64, zs, {}, std.sort.asc(f64));
            const lo_idx = (points.len * 5) / 1000;
            const hi_idx = @min(points.len - 1, (points.len * 995) / 1000);
            const q05_idx = (points.len * 5) / 100;
            const q95_idx = @min(points.len - 1, (points.len * 95) / 100);
            const ext90_x = @max(xs[q95_idx] - xs[q05_idx], 0.5);
            const ext90_y = @max(ys[q95_idx] - ys[q05_idx], 0.5);
            const ext90_z = @max(zs[q95_idx] - zs[q05_idx], 0.5);
            min_p = Vec3.init(
                @max(xs[lo_idx], xs[q05_idx] - 0.18 * ext90_x),
                @max(ys[lo_idx], ys[q05_idx] - 0.18 * ext90_y),
                @max(zs[lo_idx], zs[q05_idx] - 0.18 * ext90_z),
            );
            max_p = Vec3.init(
                @min(xs[hi_idx], xs[q95_idx] + 0.18 * ext90_x),
                @min(ys[hi_idx], ys[q95_idx] + 0.18 * ext90_y),
                @min(zs[hi_idx], zs[q95_idx] + 0.18 * ext90_z),
            );
            const pad_frac: f64 = if (points.len >= 32) 0.06 else 0.12;
            const pad = Vec3.init(
                @max((max_p.x - min_p.x) * pad_frac, 0.05),
                @max((max_p.y - min_p.y) * pad_frac, 0.05),
                @max((max_p.z - min_p.z) * pad_frac, 0.05),
            );
            min_p = min_p.sub(pad);
            max_p = max_p.add(pad);
        } else {
            const r = 1.0 / @max(@as(f64, self.bounds.inv_radius), 1e-4);
            const c = Vec3.init(self.bounds.center[0], self.bounds.center[1], self.bounds.center[2]);
            min_p = c.sub(Vec3.init(r, r, r));
            max_p = c.add(Vec3.init(r, r, r));
        }

        const ext = max_p.sub(min_p);
        const max_ext = @max(ext.x, @max(ext.y, ext.z));
        const res_cap = @max(@as(usize, 8), cfg.max_res);
        const target_step = @max(max_ext / @as(f64, @floatFromInt(res_cap - 1)), 1e-4);
        const nx = @max(@as(usize, 8), @min(res_cap, @as(usize, @intFromFloat(@ceil(ext.x / target_step))) + 1));
        const ny = @max(@as(usize, 8), @min(res_cap, @as(usize, @intFromFloat(@ceil(ext.y / target_step))) + 1));
        const nz = @max(@as(usize, 8), @min(res_cap, @as(usize, @intFromFloat(@ceil(ext.z / target_step))) + 1));
        const step = Vec3.init(
            ext.x / @as(f64, @floatFromInt(nx - 1)),
            ext.y / @as(f64, @floatFromInt(ny - 1)),
            ext.z / @as(f64, @floatFromInt(nz - 1)),
        );

        const field = try allocator.alloc(f32, nx * ny * nz);
        defer allocator.free(field);

        const n_threads = @min(@max(@as(usize, 1), cfg.num_threads), nz);
        if (n_threads <= 1) {
            sampleDensitySlices(self, 0, nz, nx, ny, min_p, step, field);
        } else {
            var next_z = std.atomic.Value(usize).init(0);
            const threads = try allocator.alloc(std.Thread, n_threads);
            defer allocator.free(threads);
            for (threads) |*th| {
                th.* = try std.Thread.spawn(.{}, sampleDensityLoop, .{
                    self, nz, nx, ny, min_p, step, field, &next_z,
                });
            }
            for (threads) |th| th.join();
        }

        if (res_cap >= 64) {
            try smoothScalarField(allocator, field, nx, ny, nz);
        }

        const scene_t = 1.0 / @max(self.bounds.inv_radius, 1e-4);
        const scene_dt = self.bounds.rayStepDt(scene_t, self.config.dt_min, self.config.max_ray_samples);
        const iso = if (cfg.iso_threshold > 0.0)
            cfg.iso_threshold
        else
            std.math.clamp(0.75 / @max(scene_dt, 1e-4), 2.8, 7.5);

        var raw = try mc.extractIsosurface(allocator, nx, ny, nz, min_p, step, field, iso, true);
        if (raw.vertices.len == 0 and cfg.iso_threshold <= 0.0) {
            raw.deinit(allocator);
            raw = try mc.extractIsosurface(allocator, nx, ny, nz, min_p, step, field, iso * 0.35, true);
        }
        defer raw.deinit(allocator);

        var cams: std.ArrayList(ExtractCamView) = .empty;
        defer cams.deinit(allocator);
        for (poses, registered) |pose, is_reg| {
            if (!is_reg) continue;
            try cams.append(allocator, .{
                .pose = pose,
                .center = pose.cameraCenter(),
            });
        }

        const raw_norms = try allocator.alloc([3]f32, raw.vertices.len);
        defer allocator.free(raw_norms);
        const raw_dirs = try allocator.alloc([3]f32, raw.vertices.len);
        defer allocator.free(raw_dirs);
        const keep_vert = try allocator.alloc(bool, raw.vertices.len);
        defer allocator.free(keep_vert);

        const use_k_filter = cfg.k != null and cfg.img_width > 0 and cfg.img_height > 0 and cams.items.len > 0;
        const k_mat = cfg.k orelse Mat3.identity;
        const w_f: f64 = @floatFromInt(cfg.img_width);
        const h_f: f64 = @floatFromInt(cfg.img_height);

        for (raw.vertices, 0..) |v, vi| {
            const normal = evalFieldOutwardNormal(field, nx, ny, nz, min_p, step, v);
            raw_norms[vi] = .{ @floatCast(normal.x), @floatCast(normal.y), @floatCast(normal.z) };
            var best_dir = [3]f32{ 0.0, 0.0, 1.0 };

            if (use_k_filter) {
                var best_score: f64 = -1.0;
                for (cams.items) |cv| {
                    const pc = cv.pose.r.mulVec(v).add(cv.pose.t);
                    if (pc.z <= 0.25) continue;
                    const up = k_mat.m[0][0] * pc.x / pc.z + k_mat.m[0][2];
                    const vp = k_mat.m[1][1] * pc.y / pc.z + k_mat.m[1][2];
                    if (up < 2.0 or up >= w_f - 2.0 or vp < 2.0 or vp >= h_f - 2.0) continue;

                    const to_cam = cv.center.sub(v);
                    const dist = to_cam.norm();
                    if (dist < 1e-6) continue;
                    const dot_n = to_cam.dot(normal) / dist;
                    if (dot_n <= -0.25) continue;

                    const margin = @min(@min(up, w_f - 1.0 - up), @min(vp, h_f - 1.0 - vp));
                    const border_w = std.math.clamp(margin / 32.0, 0.15, 1.0);
                    const score = (dot_n + 0.35) * border_w / (dist + 0.5);
                    if (score > best_score) {
                        best_score = score;
                        const dn = v.sub(cv.center).scale(1.0 / dist);
                        best_dir = .{ @floatCast(dn.x), @floatCast(dn.y), @floatCast(dn.z) };
                    }
                }
                keep_vert[vi] = best_score > 0.0;
            } else {
                var best_dist = std.math.inf(f64);
                for (cams.items) |cv| {
                    const pc = cv.pose.r.mulVec(v).add(cv.pose.t);
                    if (pc.z <= 0.0) continue;
                    const diff = v.sub(cv.center);
                    const d2 = diff.normSq();
                    if (d2 < best_dist) {
                        best_dist = d2;
                        const dn = diff.normalize();
                        best_dir = .{ @floatCast(dn.x), @floatCast(dn.y), @floatCast(dn.z) };
                    }
                }
                keep_vert[vi] = true;
            }
            raw_dirs[vi] = best_dir;
        }

        var cand_faces: std.ArrayList([3]usize) = .empty;
        defer cand_faces.deinit(allocator);
        for (raw.faces) |fc| {
            if (keep_vert[fc[0]] and keep_vert[fc[1]] and keep_vert[fc[2]]) {
                try cand_faces.append(allocator, fc);
            }
        }
        if (cand_faces.items.len == 0 and raw.faces.len > 0) {
            try cand_faces.appendSlice(allocator, raw.faces);
            @memset(keep_vert, true);
        }

        const final_faces_source = if (use_k_filter and cand_faces.items.len > 0) blk: {
            const parent = try allocator.alloc(u32, raw.vertices.len);
            defer allocator.free(parent);
            for (0..raw.vertices.len) |i| parent[i] = @intCast(i);
            for (cand_faces.items) |fc| {
                dsuUnion(parent, @intCast(fc[0]), @intCast(fc[1]));
                dsuUnion(parent, @intCast(fc[0]), @intCast(fc[2]));
            }
            const comp_counts = try allocator.alloc(usize, raw.vertices.len);
            defer allocator.free(comp_counts);
            @memset(comp_counts, 0);
            var max_comp: usize = 0;
            for (cand_faces.items) |fc| {
                const r = dsuFind(parent, @intCast(fc[0]));
                comp_counts[r] += 1;
                if (comp_counts[r] > max_comp) max_comp = comp_counts[r];
            }
            const min_comp = @min(max_comp, @max(@as(usize, 128), cand_faces.items.len / 200));
            var clean_faces: std.ArrayList([3]usize) = .empty;
            for (cand_faces.items) |fc| {
                const r = dsuFind(parent, @intCast(fc[0]));
                if (comp_counts[r] >= min_comp) {
                    try clean_faces.append(allocator, fc);
                }
            }
            break :blk try clean_faces.toOwnedSlice(allocator);
        } else try allocator.dupe([3]usize, cand_faces.items);
        defer allocator.free(final_faces_source);

        const remap = try allocator.alloc(u32, raw.vertices.len);
        defer allocator.free(remap);
        @memset(remap, std.math.maxInt(u32));
        var kept_v_list: std.ArrayList(Vec3) = .empty;
        errdefer kept_v_list.deinit(allocator);
        var kept_d_list: std.ArrayList([3]f32) = .empty;
        defer kept_d_list.deinit(allocator);

        const out_faces = try allocator.alloc([3]usize, final_faces_source.len);
        errdefer allocator.free(out_faces);
        for (final_faces_source, 0..) |fc, fi| {
            for (0..3) |ci| {
                const old_idx = fc[ci];
                if (remap[old_idx] == std.math.maxInt(u32)) {
                    remap[old_idx] = @intCast(kept_v_list.items.len);
                    try kept_v_list.append(allocator, raw.vertices[old_idx]);
                    try kept_d_list.append(allocator, raw_dirs[old_idx]);
                }
                out_faces[fi][ci] = remap[old_idx];
            }
        }

        const out_verts = try kept_v_list.toOwnedSlice(allocator);
        errdefer allocator.free(out_verts);
        const colors = try allocator.alloc(Color, out_verts.len);
        errdefer allocator.free(colors);

        const vox_s: f32 = @floatCast(@min(step.x, @min(step.y, step.z)));
        const col_threads = @min(@max(@as(usize, 1), cfg.num_threads), @max(@as(usize, 1), out_verts.len));
        var next_vi = std.atomic.Value(usize).init(0);
        if (col_threads <= 1) {
            colorVerticesWorker(self, out_verts, kept_d_list.items, vox_s, colors, &next_vi);
        } else {
            const threads = try allocator.alloc(std.Thread, col_threads);
            defer allocator.free(threads);
            for (threads) |*th| {
                th.* = try std.Thread.spawn(.{}, colorVerticesWorker, .{
                    self,
                    out_verts,
                    kept_d_list.items,
                    vox_s,
                    colors,
                    &next_vi,
                });
            }
            for (threads) |th| th.join();
        }

        return .{
            .points = out_verts,
            .colors = colors,
            .faces = out_faces,
        };
    }

    fn sampleDensityLoop(
        self: *const NerfModel,
        nz: usize,
        nx: usize,
        ny: usize,
        origin: Vec3,
        step: Vec3,
        field: []f32,
        next_z: *std.atomic.Value(usize),
    ) void {
        while (true) {
            const iz = next_z.fetchAdd(1, .monotonic);
            if (iz >= nz) break;
            sampleDensitySlices(self, iz, iz + 1, nx, ny, origin, step, field);
        }
    }

    fn sampleDensitySlices(
        self: *const NerfModel,
        z0: usize,
        z1: usize,
        nx: usize,
        ny: usize,
        origin: Vec3,
        step: Vec3,
        field: []f32,
    ) void {
        for (z0..z1) |iz| {
            const pz: f32 = @floatCast(origin.z + @as(f64, @floatFromInt(iz)) * step.z);
            for (0..ny) |iy| {
                const py: f32 = @floatCast(origin.y + @as(f64, @floatFromInt(iy)) * step.y);
                const row_base = (iz * ny + iy) * nx;
                for (0..nx) |ix| {
                    const px: f32 = @floatCast(origin.x + @as(f64, @floatFromInt(ix)) * step.x);
                    const u = self.bounds.toUnit01(.{ px, py, pz }) orelse {
                        field[row_base + ix] = 0.0;
                        continue;
                    };
                    field[row_base + ix] = if (self.occ_grid.isOccupied3x3Unit01(u))
                        self.evalDensityUnit01(u)
                    else
                        0.0;
                }
            }
        }
    }
};

pub const TrainingRay = struct {
    origin: [3]f32,
    dir: [3]f32,
    rgb: [3]f32,
    depth: ?f32,
};

pub fn sampleTrainingBatch(
    k: Mat3,
    images: []const ImageData,
    poses: []const CameraPose,
    reg_cams: []const usize,
    depth_maps: ?[]const []f32,
    cfg: NerfConfig,
    rng: std.Random,
    batch: []TrainingRay,
    bgs: [][3]f32,
    jitters: []f32,
) void {
    for (0..batch.len) |bi| {
        const ci = reg_cams[rng.intRangeLessThan(usize, 0, reg_cams.len)];
        const img = images[ci];
        const px_u = rng.intRangeLessThan(u32, 0, img.width);
        const py_u = rng.intRangeLessThan(u32, 0, img.height);
        const px_f = @as(f64, @floatFromInt(px_u)) + @as(f64, rng.float(f32));
        const py_f = @as(f64, @floatFromInt(py_u)) + @as(f64, rng.float(f32));
        const ray = generateCameraRay(k, poses[ci], px_f, py_f);
        const pidx = @as(usize, py_u) * img.width + @as(usize, px_u);
        const z_prior: ?f32 = if (depth_maps) |maps| blk: {
            const z = maps[ci][pidx];
            break :blk if (z > 0.0) z else null;
        } else null;

        batch[bi] = .{
            .origin = ray.origin,
            .dir = ray.dir,
            .rgb = .{
                @as(f32, @floatFromInt(img.rgb[pidx * 3 + 0])) / 255.0,
                @as(f32, @floatFromInt(img.rgb[pidx * 3 + 1])) / 255.0,
                @as(f32, @floatFromInt(img.rgb[pidx * 3 + 2])) / 255.0,
            },
            .depth = z_prior,
        };
        bgs[bi] = if (cfg.random_bg)
            .{ rng.float(f32), rng.float(f32), rng.float(f32) }
        else
            cfg.bg_color;
        jitters[bi] = rng.float(f32);
    }
}

pub fn cosineLrScale(step_idx: usize, num_steps: usize) f32 {
    if (num_steps >= 200 and step_idx < 32) {
        return 0.25 + 0.75 * (@as(f32, @floatFromInt(step_idx + 1)) / 32.0);
    }
    const prog = @as(f32, @floatFromInt(step_idx)) / @as(f32, @floatFromInt(@max(@as(usize, 1), num_steps)));
    return 0.10 + 0.90 * 0.5 * (1.0 + @cos(std.math.pi * prog));
}

fn trainBatchWorker(
    model: *const NerfModel,
    rays: []const TrainingRay,
    bg_colors: [][3]f32,
    jitters: []const f32,
    grad_mlp_local: []f32,
    grad_hash_local: []f32,
    loss_acc: *f64,
    next_idx: *std.atomic.Value(usize),
) void {
    @memset(grad_mlp_local, 0.0);
    @memset(grad_hash_local, 0.0);
    var local_loss: f64 = 0.0;
    while (true) {
        const i = next_idx.fetchAdd(1, .monotonic);
        if (i >= rays.len) break;
        const r = rays[i];
        local_loss += @as(f64, model.rayLossAndBackward(
            r.origin,
            r.dir,
            r.rgb,
            r.depth,
            bg_colors[i],
            jitters[i],
            true,
            grad_mlp_local,
            grad_hash_local,
        ));
    }
    loss_acc.* = local_loss;
}

/// Build sparse depth priors aligned to camera rays by Z-buffering registered SfM points.
pub fn buildSparseDepthMaps(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    k: Mat3,
    images: []const ImageData,
    poses: []const CameraPose,
    registered: []const bool,
) ![][]f32 {
    const maps = try allocator.alloc([]f32, images.len);
    errdefer allocator.free(maps);
    var init_n: usize = 0;
    errdefer for (maps[0..init_n]) |m| allocator.free(m);

    for (images, 0..) |img, ci| {
        const n_pix = @as(usize, img.width) * @as(usize, img.height);
        const dm = try allocator.alloc(f32, n_pix);
        @memset(dm, 0.0);
        maps[ci] = dm;
        init_n += 1;
    }

    if (points.len >= 32) {
        var reg_poses: std.ArrayList(CameraPose) = .empty;
        defer reg_poses.deinit(allocator);
        for (poses, registered) |p, ok| {
            if (ok) try reg_poses.append(allocator, p);
        }
        if (reg_poses.items.len >= 2) {
            const dummy_cols = try allocator.alloc(Color, points.len);
            defer allocator.free(dummy_cols);
            @memset(dummy_cols, .{ .r = 128, .g = 128, .b = 128 });
            if (mesh_mod.buildColoredMesh(allocator, points, dummy_cols, reg_poses.items, .{ .fill_holes = false })) |coarse_val| {
                var coarse = coarse_val;
                defer coarse.deinit(allocator);
                mesh_mod.completeMeshWithPhotoVerify(allocator, &coarse, k, images, poses, registered, reg_poses.items, .{}) catch {};
                mesh_mod.fillMeshInteriorHoles(allocator, &coarse, reg_poses.items, .{}) catch {};

                const fx = k.m[0][0];
                const skew = k.m[0][1];
                const cx = k.m[0][2];
                const fy = k.m[1][1];
                const cy = k.m[1][2];

                for (images, 0..) |img, ci| {
                    if (!registered[ci]) continue;
                    var ren = try render.renderMeshView(allocator, img.width, img.height, k, poses[ci], coarse.points, coarse.colors, coarse.faces);
                    defer ren.deinit(allocator);
                    const dm = maps[ci];
                    for (0..img.height) |py| {
                        const yn = (@as(f64, @floatFromInt(py)) + 0.5 - cy) / fy;
                        const row_off = py * img.width;
                        for (0..img.width) |px| {
                            const idx = row_off + px;
                            if (!ren.covered[idx]) continue;
                            const zc = @as(f64, ren.depth[idx]);
                            if (!std.math.isFinite(zc) or zc <= 0.1) continue;
                            const xn = (@as(f64, @floatFromInt(px)) + 0.5 - cx - skew * yn) / fx;
                            dm[idx] = @floatCast(zc * @sqrt(xn * xn + yn * yn + 1.0));
                        }
                    }
                }
            } else |_| {}
        }
    }

    for (images, 0..) |img, ci| {
        if (!registered[ci]) continue;
        const dm = maps[ci];
        const wf: f64 = @floatFromInt(img.width);
        const hf: f64 = @floatFromInt(img.height);
        const pose = poses[ci];
        for (points) |pt| {
            const xc = pose.transformPoint(pt);
            if (xc.z <= 0.1) continue;
            const uv = lie.projectPoint(k, pose, pt) orelse continue;
            if (uv.x < 0.0 or uv.x >= wf or uv.y < 0.0 or uv.y >= hf) continue;
            const px: usize = @intFromFloat(@floor(uv.x));
            const py: usize = @intFromFloat(@floor(uv.y));
            const ray = generateCameraRay(k, pose, uv.x, uv.y);
            const c = pose.cameraCenter();
            const ray_t: f32 = @floatCast(pt.sub(c).dot(Vec3.init(ray.dir[0], ray.dir[1], ray.dir[2])));
            if (ray_t <= 0.1) continue;
            const idx = py * img.width + px;
            dm[idx] = ray_t;
        }
    }
    return maps;
}

pub fn elapsedMsOpt(io: ?Io, t0: ?Io.Timestamp) f64 {
    const io_v = io orelse return 0.0;
    const t_v = t0 orelse return 0.0;
    return @as(f64, @floatFromInt(t_v.durationTo(Io.Clock.now(.awake, io_v)).nanoseconds)) / 1e6;
}

/// Train an Instant-NGP Hash-Grid NeRF from calibrated SfM cameras and sparse points.
pub fn trainOnDataset(
    allocator: std.mem.Allocator,
    k: Mat3,
    images: []const ImageData,
    poses: []const CameraPose,
    registered: []const bool,
    points: []const Vec3,
    cfg: NerfConfig,
    timings_out: ?*NerfTimings,
) !NerfModel {
    const t_setup0: ?Io.Timestamp = if (cfg.io) |io_v| Io.Clock.now(.awake, io_v) else null;

    var reg_cams: std.ArrayList(usize) = .empty;
    defer reg_cams.deinit(allocator);
    for (registered, 0..) |ok, ci| {
        if (ok) try reg_cams.append(allocator, ci);
    }
    if (reg_cams.items.len == 0) return error.NoRegisteredCameras;

    const bounds = SceneBounds.fromSfm(points, poses, registered, cfg.use_contraction);
    var model = try NerfModel.init(allocator, cfg, bounds);
    errdefer model.deinit(allocator);

    const depth_maps = if (cfg.depth_loss_weight > 0.0 and points.len > 0)
        try buildSparseDepthMaps(allocator, points, k, images, poses, registered)
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

    const batch = try allocator.alloc(TrainingRay, cfg.batch_rays);
    defer allocator.free(batch);
    const bgs = try allocator.alloc([3]f32, cfg.batch_rays);
    defer allocator.free(bgs);
    const jitters = try allocator.alloc(f32, cfg.batch_rays);
    defer allocator.free(jitters);

    const n_workers = @max(@as(usize, 1), cfg.num_threads);
    const worker_grad_mlp = try allocator.alloc([]f32, n_workers);
    defer {
        for (worker_grad_mlp) |g| allocator.free(g);
        allocator.free(worker_grad_mlp);
    }
    const worker_grad_hash = try allocator.alloc([]f32, n_workers);
    defer {
        for (worker_grad_hash) |g| allocator.free(g);
        allocator.free(worker_grad_hash);
    }
    for (0..n_workers) |wi| {
        worker_grad_mlp[wi] = try allocator.alloc(f32, model.mlp_params.len);
        worker_grad_hash[wi] = try allocator.alloc(f32, model.hash_grid.params.len);
    }

    const sum_grad_mlp = try allocator.alloc(f32, model.mlp_params.len);
    defer allocator.free(sum_grad_mlp);
    const sum_grad_hash = try allocator.alloc(f32, model.hash_grid.params.len);
    defer allocator.free(sum_grad_hash);
    const worker_losses = try allocator.alloc(f64, n_workers);
    defer allocator.free(worker_losses);

    const setup_ms = elapsedMsOpt(cfg.io, t_setup0);
    const t_train0: ?Io.Timestamp = if (cfg.io) |io_v| Io.Clock.now(.awake, io_v) else null;

    var final_loss: f64 = 0.0;
    const inv_batch: f32 = 1.0 / @as(f32, @floatFromInt(cfg.batch_rays));

    for (0..cfg.num_steps) |step_idx| {
        sampleTrainingBatch(k, images, poses, reg_cams.items, depth_maps, cfg, rng, batch, bgs, jitters);

        var next_idx = std.atomic.Value(usize).init(0);
        if (n_workers <= 1) {
            trainBatchWorker(&model, batch, bgs, jitters, worker_grad_mlp[0], worker_grad_hash[0], &worker_losses[0], &next_idx);
            @memcpy(sum_grad_mlp, worker_grad_mlp[0]);
            @memcpy(sum_grad_hash, worker_grad_hash[0]);
            final_loss = worker_losses[0] * inv_batch;
        } else {
            const threads = try allocator.alloc(std.Thread, n_workers);
            defer allocator.free(threads);
            for (threads, 0..) |*th, wi| {
                th.* = try std.Thread.spawn(.{}, trainBatchWorker, .{
                    &model, batch, bgs, jitters, worker_grad_mlp[wi], worker_grad_hash[wi], &worker_losses[wi], &next_idx,
                });
            }
            for (threads) |th| th.join();

            @memset(sum_grad_mlp, 0.0);
            @memset(sum_grad_hash, 0.0);
            var tot_loss: f64 = 0.0;
            for (0..n_workers) |wi| {
                tot_loss += worker_losses[wi];
                for (sum_grad_mlp, worker_grad_mlp[wi]) |*dst, src| dst.* += src;
                for (sum_grad_hash, worker_grad_hash[wi]) |*dst, src| dst.* += src;
            }
            final_loss = tot_loss * inv_batch;
        }

        for (sum_grad_mlp) |*g| g.* *= inv_batch;
        for (sum_grad_hash) |*g| g.* *= inv_batch;

        const lr_scale = cosineLrScale(step_idx, cfg.num_steps);
        model.applyGradients(sum_grad_mlp, sum_grad_hash, lr_scale);

        if ((step_idx + 1) % cfg.occ_update_interval == 0) {
            model.occ_grid.updateFromModel(&model, cfg.occ_decay, cfg.occ_threshold);
        }
    }

    if (timings_out) |tm| {
        tm.setup_ms = setup_ms;
        tm.train_ms = elapsedMsOpt(cfg.io, t_train0);
        tm.final_loss = final_loss;
    }

    return model;
}

test "SphericalHarmonics produces normalized zero-order term" {
    const sh0 = encodeSphericalHarmonics(.{ 0.0, 0.0, 1.0 });
    const sh1 = encodeSphericalHarmonics(.{ 0.57735027, 0.57735027, 0.57735027 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.28209479), sh0[0], 1e-6);
    try std.testing.expectApproxEqAbs(sh0[0], sh1[0], 1e-6);
    try std.testing.expect(sh0[2] > 0.0);
    try std.testing.expect(sh0[1] == 0.0);
}

fn evalRayMseForTest(model: *const NerfModel, origin: [3]f32, dir: [3]f32, target_rgb: [3]f32, bg: [3]f32) f32 {
    var dummy_mlp: [1024]f32 = undefined;
    var dummy_hash: [8192]f32 = undefined;
    const gm = dummy_mlp[0..model.mlp_params.len];
    const gh = dummy_hash[0..model.hash_grid.params.len];
    @memset(gm, 0.0);
    @memset(gh, 0.0);
    return model.rayLossAndBackward(origin, dir, target_rgb, null, bg, 0.37, false, gm, gh);
}

fn checkFiniteDiffSlice(
    model: *const NerfModel,
    params: []f32,
    grads: []const f32,
    origin: [3]f32,
    dir: [3]f32,
    target_rgb: [3]f32,
    bg: [3]f32,
    max_checks: usize,
    min_required: usize,
) !void {
    const eps: f32 = 1e-3;
    var checked: usize = 0;
    for (0..params.len) |i| {
        if (@abs(grads[i]) < 1e-4) continue;
        const orig = params[i];
        params[i] = orig + eps;
        const loss_p = evalRayMseForTest(model, origin, dir, target_rgb, bg);
        params[i] = orig - eps;
        const loss_m = evalRayMseForTest(model, origin, dir, target_rgb, bg);
        params[i] = orig;
        const fd = (loss_p - loss_m) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(fd, grads[i], 2e-3);
        checked += 1;
        if (checked >= max_checks) break;
    }
    try std.testing.expect(checked >= min_required);
}

test "HashGrid and NerfModel analytical gradients match finite differences" {
    const gpa = std.testing.allocator;
    const cfg = NerfConfig{
        .num_levels = 4,
        .table_log2 = 9,
        .n_min = 4.0,
        .n_max = 16.0,
        .geo_hidden = 16,
        .geo_feat_dim = 7,
        .col_hidden = 16,
        .max_ray_samples = 24,
        .dt_min = 0.05,
        .use_contraction = false,
        .occ_grid_res = 8,
    };
    const bounds = SceneBounds{
        .center = .{ 0.5, 0.5, 0.5 },
        .inv_radius = 2.0,
        .use_contraction = false,
        .t_near = 0.05,
        .t_far = 4.0,
    };
    var model = try NerfModel.init(gpa, cfg, bounds);
    defer model.deinit(gpa);
    model.mlp_params[model.layout.b1_off] = 0.0;
    var fd_prng = std.Random.DefaultPrng.init(42);
    for (model.hash_grid.params) |*hp| {
        hp.* = (fd_prng.random().float(f32) * 2.0 - 1.0) * 0.15;
    }

    const origin = [3]f32{ 0.5, 0.5, -0.5 };
    const dir = [3]f32{ 0.04, -0.03, 0.99875 };
    const target_rgb = [3]f32{ 0.72, 0.31, 0.55 };
    const bg = [3]f32{ 0.1, 0.1, 0.1 };

    const grad_mlp = try gpa.alloc(f32, model.mlp_params.len);
    defer gpa.free(grad_mlp);
    const grad_hash = try gpa.alloc(f32, model.hash_grid.params.len);
    defer gpa.free(grad_hash);
    @memset(grad_mlp, 0.0);
    @memset(grad_hash, 0.0);

    _ = model.rayLossAndBackward(origin, dir, target_rgb, null, bg, 0.37, false, grad_mlp, grad_hash);
    try checkFiniteDiffSlice(&model, model.mlp_params, grad_mlp, origin, dir, target_rgb, bg, 200, 40);
    try checkFiniteDiffSlice(&model, model.hash_grid.params, grad_hash, origin, dir, target_rgb, bg, 30, 8);
}

test "NerfModel overfits synthetic 3D textured sphere and generalizes to held-out view" {
    const gpa = std.testing.allocator;
    const w: u32 = 40;
    const h: u32 = 40;
    const n_cams: usize = 16;

    const k = Mat3{ .m = .{
        .{ 48.0, 0.0, 20.0 },
        .{ 0.0, 48.0, 20.0 },
        .{ 0.0, 0.0, 1.0 },
    } };

    const sphere_center = Vec3.init(0.5, 0.5, 0.5);
    const sphere_r: f64 = 0.28;
    const bg = [3]f32{ 0.12, 0.15, 0.22 };

    const poses = try gpa.alloc(CameraPose, n_cams);
    defer gpa.free(poses);
    const images = try gpa.alloc(ImageData, n_cams);
    defer {
        for (images) |*img| img.deinit(gpa);
        gpa.free(images);
    }
    const registered = try gpa.alloc(bool, n_cams);
    defer gpa.free(registered);

    for (0..n_cams) |ci| {
        const angle = (@as(f64, @floatFromInt(ci)) / @as(f64, @floatFromInt(n_cams))) * std.math.tau;
        const cam_c = Vec3.init(
            0.5 + 1.15 * @cos(angle),
            0.5 + 0.22 * @sin(angle * 2.0),
            0.5 + 1.15 * @sin(angle),
        );
        const fwd = sphere_center.sub(cam_c).normalize();
        const right = Vec3.init(0, 1, 0).cross(fwd).normalize();
        const down = fwd.cross(right).normalize();
        const r = Mat3{ .m = .{
            .{ right.x, right.y, right.z },
            .{ down.x, down.y, down.z },
            .{ fwd.x, fwd.y, fwd.z },
        } };
        poses[ci] = .{ .r = r, .t = r.mulVec(cam_c).neg() };
        registered[ci] = (ci != 8);

        const n_pix = @as(usize, w) * @as(usize, h);
        const rgb = try gpa.alloc(u8, n_pix * 3);
        const gray = try gpa.alloc(u8, n_pix);
        for (0..h) |y| {
            for (0..w) |x| {
                const ray = generateCameraRay(k, poses[ci], @as(f64, @floatFromInt(x)) + 0.5, @as(f64, @floatFromInt(y)) + 0.5);
                const ro = Vec3.init(ray.origin[0], ray.origin[1], ray.origin[2]);
                const rd = Vec3.init(ray.dir[0], ray.dir[1], ray.dir[2]);
                const oc = ro.sub(sphere_center);
                const b = oc.dot(rd);
                const c_val = oc.dot(oc) - sphere_r * sphere_r;
                const disc = b * b - c_val;
                var col = bg;
                if (disc >= 0.0) {
                    const hit_t = -b - @sqrt(disc);
                    if (hit_t > 0.0) {
                        const hit_p = ro.add(rd.scale(hit_t));
                        const nrm = hit_p.sub(sphere_center).normalize();
                        col = .{
                            @floatCast(0.5 + 0.42 * nrm.x),
                            @floatCast(0.5 + 0.42 * nrm.y),
                            @floatCast(0.5 + 0.42 * nrm.z),
                        };
                    }
                }
                const pidx = y * w + x;
                for (0..3) |ch| rgb[pidx * 3 + ch] = @intFromFloat(std.math.clamp(col[ch] * 255.0 + 0.5, 0.0, 255.0));
                gray[pidx] = rgb[pidx * 3 + 1];
            }
        }
        images[ci] = .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
    }

    const sparse_pts = [_]Vec3{
        Vec3.init(0.5 + 0.28, 0.5, 0.5),
        Vec3.init(0.5 - 0.28, 0.5, 0.5),
        Vec3.init(0.5, 0.5 + 0.28, 0.5),
        Vec3.init(0.5, 0.5 - 0.28, 0.5),
        Vec3.init(0.5, 0.5, 0.5 + 0.28),
        Vec3.init(0.5, 0.5, 0.5 - 0.28),
    };

    const cfg = NerfConfig{
        .num_levels = 6,
        .table_log2 = 12,
        .n_min = 8.0,
        .n_max = 64.0,
        .geo_hidden = 24,
        .geo_feat_dim = 11,
        .col_hidden = 24,
        .num_steps = 600,
        .batch_rays = 256,
        .max_ray_samples = 64,
        .lr = 2e-2,
        .dt_min = 0.032,
        .use_contraction = false,
        .occ_grid_res = 16,
        .occ_update_interval = 40,
        .occ_decay = 0.75,
        .depth_loss_weight = 0.0,
        .bg_color = bg,
    };

    var model = try trainOnDataset(gpa, k, images, poses, registered, &sparse_pts, cfg, null);
    defer model.deinit(gpa);

    var holdout_view = try model.renderView(gpa, w, h, k, poses[8], true);
    defer holdout_view.deinit(gpa);
    const hm = render.computeViewMetrics(holdout_view, images[8]);
    try std.testing.expect(hm.psnr_db > 25.0);

    var extracted = try model.extractMesh(gpa, &sparse_pts, poses, registered, .{ .max_res = 32 });
    defer extracted.deinit(gpa);
    try std.testing.expect(extracted.points.len > 100);
    try std.testing.expect(extracted.faces.len > 100);
    var sum_r: f64 = 0.0;
    for (extracted.points) |pt| sum_r += pt.sub(sphere_center).norm();
    const mean_r = sum_r / @as(f64, @floatFromInt(extracted.points.len));
    try std.testing.expectApproxEqAbs(sphere_r, mean_r, 0.08);
}

test "SceneBounds.fromUnit01 round-trips contraction and OccupancyGrid preserves culled cells" {
    const gpa = std.testing.allocator;
    const bounds = SceneBounds{
        .center = .{ 1.0, -0.5, 2.0 },
        .inv_radius = 0.25,
        .use_contraction = true,
        .t_near = 0.1,
        .t_far = 20.0,
    };
    const p_in = [3]f32{ 2.0, -1.0, 3.5 };
    const u_in = bounds.toUnit01(p_in).?;
    const r_in = bounds.fromUnit01(u_in).?;
    for (0..3) |d| try std.testing.expectApproxEqAbs(p_in[d], r_in[d], 1e-4);

    const p_out = [3]f32{ 1.0, -0.5, 10.0 };
    const u_out = bounds.toUnit01(p_out).?;
    const r_out = bounds.fromUnit01(u_out).?;
    for (0..3) |d| try std.testing.expectApproxEqAbs(p_out[d], r_out[d], 1e-4);
    try std.testing.expect(bounds.fromUnit01(.{ 0.01, 0.01, 0.01 }) == null);

    var model = try NerfModel.init(gpa, .{
        .num_levels = 4,
        .table_log2 = 8,
        .geo_hidden = 16,
        .geo_feat_dim = 7,
        .col_hidden = 16,
        .occ_grid_res = 4,
    }, bounds);
    defer model.deinit(gpa);
    model.occ_grid.density[0] = -1.0;
    model.occ_grid.occupied[0] = false;
    model.occ_grid.updateFromModel(&model, 0.75, 0.02);
    try std.testing.expect(model.occ_grid.density[0] < 0.0);
    try std.testing.expect(!model.occ_grid.occupied[0]);
}
