const std = @import("std");
const Io = std.Io;
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const Color = @import("io_ply.zig").Color;
const ImageData = @import("dataset.zig").ImageData;

pub const RenderedView = struct {
    width: u32,
    height: u32,
    rgb: []u8,
    depth: []f32,
    covered: []bool,

    pub fn deinit(self: *RenderedView, allocator: std.mem.Allocator) void {
        allocator.free(self.rgb);
        allocator.free(self.depth);
        allocator.free(self.covered);
        self.* = undefined;
    }
};

pub const ViewMetrics = struct {
    covered_pixels: usize,
    total_pixels: usize,
    coverage_ratio: f64,
    mae: f64,
    rmse: f64,
    psnr_db: f64,
    ssim: f64,
};

const ProjectedVertex = struct {
    px: f64,
    py: f64,
    z: f64,
    inv_z: f64,
    valid: bool,
};

pub fn interpolatePose(a: CameraPose, b: CameraPose, alpha: f64) CameraPose {
    const r_rel = b.r.mul(a.r.transpose());
    const w = lie.so3Log(r_rel);
    const r = lie.so3Exp(w.scale(alpha)).mul(a.r);
    const ca = a.cameraCenter();
    const cb = b.cameraCenter();
    const c = ca.scale(1.0 - alpha).add(cb.scale(alpha));
    return .{
        .r = r,
        .t = r.mulVec(c).neg(),
    };
}

pub fn renderMeshView(
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    k: Mat3,
    pose: CameraPose,
    points: []const Vec3,
    colors: []const Color,
    faces: []const [3]usize,
) !RenderedView {
    const n_pix = @as(usize, width) * @as(usize, height);
    const rgb = try allocator.alloc(u8, n_pix * 3);
    errdefer allocator.free(rgb);
    @memset(rgb, 0);

    const depth = try allocator.alloc(f32, n_pix);
    errdefer allocator.free(depth);
    @memset(depth, std.math.inf(f32));

    const covered = try allocator.alloc(bool, n_pix);
    errdefer allocator.free(covered);
    @memset(covered, false);

    const proj = try allocator.alloc(ProjectedVertex, points.len);
    defer allocator.free(proj);

    for (points, 0..) |pt, idx| {
        const xc = pose.transformPoint(pt);
        if (xc.z <= 1e-4) {
            proj[idx] = .{ .px = 0, .py = 0, .z = 0, .inv_z = 0, .valid = false };
            continue;
        }
        const inv_z = 1.0 / xc.z;
        const xn = xc.x * inv_z;
        const yn = xc.y * inv_z;
        const px = k.m[0][0] * xn + k.m[0][1] * yn + k.m[0][2];
        const py = k.m[1][1] * yn + k.m[1][2];
        proj[idx] = .{ .px = px, .py = py, .z = xc.z, .inv_z = inv_z, .valid = true };
    }

    const w_f: f64 = @floatFromInt(width);
    const h_f: f64 = @floatFromInt(height);

    for (faces) |tri| {
        const v_a = proj[tri[0]];
        const v_b = proj[tri[1]];
        const v_c = proj[tri[2]];
        if (!v_a.valid or !v_b.valid or !v_c.valid) continue;

        const area = (v_b.px - v_a.px) * (v_c.py - v_a.py) - (v_b.py - v_a.py) * (v_c.px - v_a.px);
        if (@abs(area) < 1e-4) continue;

        const min_x_f = @max(0.0, @floor(@min(v_a.px, @min(v_b.px, v_c.px))));
        const max_x_f = @min(w_f - 1.0, @ceil(@max(v_a.px, @max(v_b.px, v_c.px))));
        const min_y_f = @max(0.0, @floor(@min(v_a.py, @min(v_b.py, v_c.py))));
        const max_y_f = @min(h_f - 1.0, @ceil(@max(v_a.py, @max(v_b.py, v_c.py))));
        if (max_x_f < min_x_f or max_y_f < min_y_f) continue;

        const min_x: usize = @intFromFloat(min_x_f);
        const max_x: usize = @intFromFloat(max_x_f);
        const min_y: usize = @intFromFloat(min_y_f);
        const max_y: usize = @intFromFloat(max_y_f);

        const c_a = colors[tri[0]];
        const c_b = colors[tri[1]];
        const c_c = colors[tri[2]];

        const inv_area = 1.0 / area;

        const r_a_iz = @as(f64, @floatFromInt(c_a.r)) * v_a.inv_z;
        const g_a_iz = @as(f64, @floatFromInt(c_a.g)) * v_a.inv_z;
        const b_a_iz = @as(f64, @floatFromInt(c_a.b)) * v_a.inv_z;

        const r_b_iz = @as(f64, @floatFromInt(c_b.r)) * v_b.inv_z;
        const g_b_iz = @as(f64, @floatFromInt(c_b.g)) * v_b.inv_z;
        const b_b_iz = @as(f64, @floatFromInt(c_b.b)) * v_b.inv_z;

        const r_c_iz = @as(f64, @floatFromInt(c_c.r)) * v_c.inv_z;
        const g_c_iz = @as(f64, @floatFromInt(c_c.g)) * v_c.inv_z;
        const b_c_iz = @as(f64, @floatFromInt(c_c.b)) * v_c.inv_z;

        var py_idx: usize = min_y;
        while (py_idx <= max_y) : (py_idx += 1) {
            const p_y: f64 = @as(f64, @floatFromInt(py_idx)) + 0.5;
            const row_off = py_idx * @as(usize, width);
            var px_idx: usize = min_x;
            while (px_idx <= max_x) : (px_idx += 1) {
                const p_x: f64 = @as(f64, @floatFromInt(px_idx)) + 0.5;
                const l0 = ((v_b.px - p_x) * (v_c.py - p_y) - (v_b.py - p_y) * (v_c.px - p_x)) * inv_area;
                if (l0 < -1e-5) continue;
                const l1 = ((v_c.px - p_x) * (v_a.py - p_y) - (v_c.py - p_y) * (v_a.px - p_x)) * inv_area;
                if (l1 < -1e-5) continue;
                const l2 = 1.0 - l0 - l1;
                if (l2 < -1e-5) continue;

                const inv_z = l0 * v_a.inv_z + l1 * v_b.inv_z + l2 * v_c.inv_z;
                if (inv_z <= 1e-9) continue;
                const z_val = 1.0 / inv_z;
                const z_f32: f32 = @floatCast(z_val);

                const pix_idx = row_off + px_idx;
                if (z_f32 >= depth[pix_idx]) continue;

                depth[pix_idx] = z_f32;
                covered[pix_idx] = true;

                const rr = std.math.clamp((l0 * r_a_iz + l1 * r_b_iz + l2 * r_c_iz) * z_val, 0.0, 255.0);
                const gg = std.math.clamp((l0 * g_a_iz + l1 * g_b_iz + l2 * g_c_iz) * z_val, 0.0, 255.0);
                const bb = std.math.clamp((l0 * b_a_iz + l1 * b_b_iz + l2 * b_c_iz) * z_val, 0.0, 255.0);
                rgb[pix_idx * 3 + 0] = @intFromFloat(@round(rr));
                rgb[pix_idx * 3 + 1] = @intFromFloat(@round(gg));
                rgb[pix_idx * 3 + 2] = @intFromFloat(@round(bb));
            }
        }
    }

    return .{
        .width = width,
        .height = height,
        .rgb = rgb,
        .depth = depth,
        .covered = covered,
    };
}

pub fn computeViewMetrics(rendered: RenderedView, real: ImageData) ViewMetrics {
    std.debug.assert(rendered.width == real.width and rendered.height == real.height);
    const total_pixels = @as(usize, rendered.width) * @as(usize, rendered.height);

    var covered_count: usize = 0;
    var sum_abs: f64 = 0.0;
    var sum_sq: f64 = 0.0;

    for (0..total_pixels) |idx| {
        if (!rendered.covered[idx]) continue;
        covered_count += 1;
        for (0..3) |ch| {
            const diff = @as(f64, @floatFromInt(rendered.rgb[idx * 3 + ch])) -
                @as(f64, @floatFromInt(real.rgb[idx * 3 + ch]));
            sum_abs += @abs(diff);
            sum_sq += diff * diff;
        }
    }

    if (covered_count == 0) {
        return .{
            .covered_pixels = 0,
            .total_pixels = total_pixels,
            .coverage_ratio = 0.0,
            .mae = 255.0,
            .rmse = 255.0,
            .psnr_db = 0.0,
            .ssim = 0.0,
        };
    }

    const n_vals = @as(f64, @floatFromInt(covered_count * 3));
    const mae = sum_abs / n_vals;
    const mse = sum_sq / n_vals;
    const rmse = @sqrt(mse);
    const psnr_db = if (mse <= 1e-12)
        99.99
    else
        10.0 * std.math.log10((255.0 * 255.0) / mse);

    const ssim = computeMaskedSsim(rendered, real);

    return .{
        .covered_pixels = covered_count,
        .total_pixels = total_pixels,
        .coverage_ratio = @as(f64, @floatFromInt(covered_count)) / @as(f64, @floatFromInt(total_pixels)),
        .mae = mae,
        .rmse = rmse,
        .psnr_db = psnr_db,
        .ssim = ssim,
    };
}

fn rgbLuminance(rgb: []const u8, idx: usize) f64 {
    const r: f64 = @floatFromInt(rgb[idx * 3 + 0]);
    const g: f64 = @floatFromInt(rgb[idx * 3 + 1]);
    const b: f64 = @floatFromInt(rgb[idx * 3 + 2]);
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

fn computeMaskedSsim(rendered: RenderedView, real: ImageData) f64 {
    const c1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const c2: f64 = (0.03 * 255.0) * (0.03 * 255.0);
    const block: usize = 8;

    const w: usize = rendered.width;
    const h: usize = rendered.height;

    var ssim_sum: f64 = 0.0;
    var block_count: usize = 0;

    var by: usize = 0;
    while (by + block <= h) : (by += block) {
        var bx: usize = 0;
        while (bx + block <= w) : (bx += block) {
            var n_cov: usize = 0;
            var sum_x: f64 = 0.0;
            var sum_y: f64 = 0.0;
            for (0..block) |dy| {
                const row = (by + dy) * w + bx;
                for (0..block) |dx| {
                    const idx = row + dx;
                    if (!rendered.covered[idx]) continue;
                    n_cov += 1;
                    sum_x += rgbLuminance(rendered.rgb, idx);
                    sum_y += rgbLuminance(real.rgb, idx);
                }
            }
            if (n_cov < (block * block * 3) / 4) continue;

            const nf: f64 = @floatFromInt(n_cov);
            const mu_x = sum_x / nf;
            const mu_y = sum_y / nf;

            var var_x: f64 = 0.0;
            var var_y: f64 = 0.0;
            var cov_xy: f64 = 0.0;
            for (0..block) |dy| {
                const row = (by + dy) * w + bx;
                for (0..block) |dx| {
                    const idx = row + dx;
                    if (!rendered.covered[idx]) continue;
                    const lx = rgbLuminance(rendered.rgb, idx) - mu_x;
                    const ly = rgbLuminance(real.rgb, idx) - mu_y;
                    var_x += lx * lx;
                    var_y += ly * ly;
                    cov_xy += lx * ly;
                }
            }
            const denom_var = @max(1.0, nf - 1.0);
            var_x /= denom_var;
            var_y /= denom_var;
            cov_xy /= denom_var;

            const num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2);
            const den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);
            ssim_sum += num / den;
            block_count += 1;
        }
    }

    if (block_count == 0) return 0.0;
    return ssim_sum / @as(f64, @floatFromInt(block_count));
}

pub fn writePpm(allocator: std.mem.Allocator, io: Io, path: []const u8, width: u32, height: u32, rgb: []const u8) !void {
    const header = try std.fmt.allocPrint(allocator, "P6\n{d} {d}\n255\n", .{ width, height });
    defer allocator.free(header);

    const out = try allocator.alloc(u8, header.len + rgb.len);
    defer allocator.free(out);
    @memcpy(out[0..header.len], header);
    @memcpy(out[header.len..], rgb);

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out });
}

test "renderMeshView rasterizes triangle with Z-buffer occlusion" {
    const gpa = std.testing.allocator;
    const k = Mat3{ .m = .{
        .{ 100.0, 0.0, 16.0 },
        .{ 0.0, 100.0, 16.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const pose = CameraPose.identity;

    const pts = [_]Vec3{
        // Far red triangle at z = 5
        Vec3.init(-1.0, -1.0, 5.0),
        Vec3.init(1.0, -1.0, 5.0),
        Vec3.init(0.0, 1.0, 5.0),
        // Closer green triangle at z = 3 covering center
        Vec3.init(-0.5, -0.5, 3.0),
        Vec3.init(0.5, -0.5, 3.0),
        Vec3.init(0.0, 0.5, 3.0),
    };
    const cols = [_]Color{
        .{ .r = 255, .g = 0, .b = 0 },
        .{ .r = 255, .g = 0, .b = 0 },
        .{ .r = 255, .g = 0, .b = 0 },
        .{ .r = 0, .g = 255, .b = 0 },
        .{ .r = 0, .g = 255, .b = 0 },
        .{ .r = 0, .g = 255, .b = 0 },
    };
    const faces = [_][3]usize{
        .{ 0, 1, 2 },
        .{ 3, 4, 5 },
    };

    var view = try renderMeshView(gpa, 32, 32, k, pose, &pts, &cols, &faces);
    defer view.deinit(gpa);

    const center_idx = 16 * 32 + 16;
    try std.testing.expect(view.covered[center_idx]);
    try std.testing.expectEqual(@as(u8, 0), view.rgb[center_idx * 3 + 0]);
    try std.testing.expectEqual(@as(u8, 255), view.rgb[center_idx * 3 + 1]);
}

test "computeViewMetrics returns exact match on identical image and drops on mismatch" {
    const gpa = std.testing.allocator;
    const w: u32 = 16;
    const h: u32 = 16;
    const n_pix = @as(usize, w) * @as(usize, h);

    const rgb_ren = try gpa.alloc(u8, n_pix * 3);
    const depth = try gpa.alloc(f32, n_pix);
    const cov = try gpa.alloc(bool, n_pix);
    @memset(depth, 1.0);
    @memset(cov, true);

    const rgb_real = try gpa.alloc(u8, n_pix * 3);
    const gray_real = try gpa.alloc(u8, n_pix);

    for (0..n_pix) |i| {
        const val: u8 = @intCast((i * 7) % 255);
        rgb_ren[i * 3 + 0] = val;
        rgb_ren[i * 3 + 1] = val / 2;
        rgb_ren[i * 3 + 2] = 255 - val;
        rgb_real[i * 3 + 0] = val;
        rgb_real[i * 3 + 1] = val / 2;
        rgb_real[i * 3 + 2] = 255 - val;
        gray_real[i] = val;
    }

    var ren = RenderedView{ .width = w, .height = h, .rgb = rgb_ren, .depth = depth, .covered = cov };
    defer ren.deinit(gpa);
    var real = ImageData{ .width = w, .height = h, .rgb = rgb_real, .gray = gray_real };
    defer real.deinit(gpa);

    const exact = computeViewMetrics(ren, real);
    try std.testing.expectApproxEqAbs(1.0, exact.coverage_ratio, 1e-9);
    try std.testing.expectApproxEqAbs(0.0, exact.mae, 1e-9);
    try std.testing.expectApproxEqAbs(1.0, exact.ssim, 1e-6);

    for (0..n_pix * 3) |j| ren.rgb[j] = 255 - ren.rgb[j];
    const diff = computeViewMetrics(ren, real);
    try std.testing.expect(diff.mae > 50.0);
    try std.testing.expect(diff.ssim < 0.2);
}

test "interpolatePose interpolates endpoints and midpoint" {
    const a = CameraPose.identity;
    const b_r = lie.so3Exp(Vec3.init(0.0, 0.4, 0.0));
    const b_c = Vec3.init(2.0, 0.0, 0.0);
    const b = CameraPose{ .r = b_r, .t = b_r.mulVec(b_c).neg() };

    const p0 = interpolatePose(a, b, 0.0);
    const p5 = interpolatePose(a, b, 0.5);
    const p1 = interpolatePose(a, b, 1.0);

    try std.testing.expectApproxEqAbs(0.0, p0.cameraCenter().x, 1e-6);
    try std.testing.expectApproxEqAbs(1.0, p5.cameraCenter().x, 1e-6);
    try std.testing.expectApproxEqAbs(2.0, p1.cameraCenter().x, 1e-6);
}
