const std = @import("std");
const Mat3 = @import("math/mat.zig").Mat3;

const c = @import("c_img");

pub const ImageData = struct {
    width: u32,
    height: u32,
    rgb: []u8,
    gray: []u8,

    pub fn deinit(self: *ImageData, allocator: std.mem.Allocator) void {
        allocator.free(self.rgb);
        allocator.free(self.gray);
        self.* = undefined;
    }
};

pub const Dataset = struct {
    k: Mat3,
    images: []ImageData,
    names: [][]u8,

    pub fn deinit(self: *Dataset, allocator: std.mem.Allocator) void {
        for (self.images) |*img| img.deinit(allocator);
        allocator.free(self.images);
        for (self.names) |n| allocator.free(n);
        allocator.free(self.names);
        self.* = undefined;
    }
};

fn rgbToGray(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32) ![]u8 {
    const n = @as(usize, w) * @as(usize, h);
    const gray = try allocator.alloc(u8, n);
    for (0..n) |i| {
        const r: u32 = rgb[i * 3 + 0];
        const g: u32 = rgb[i * 3 + 1];
        const b: u32 = rgb[i * 3 + 2];
        gray[i] = @intCast((77 * r + 150 * g + 29 * b) >> 8);
    }
    return gray;
}

pub fn loadPng(allocator: std.mem.Allocator, path_z: [:0]const u8) !ImageData {
    var img: c.png_image = std.mem.zeroes(c.png_image);
    img.version = c.PNG_IMAGE_VERSION;

    if (c.png_image_begin_read_from_file(&img, path_z.ptr) == 0) {
        return error.PngReadFailed;
    }
    errdefer c.png_image_free(&img);

    img.format = c.PNG_FORMAT_RGB;
    const w: u32 = img.width;
    const h: u32 = img.height;
    const rgb_len = @as(usize, w) * @as(usize, h) * 3;
    const rgb = try allocator.alloc(u8, rgb_len);
    errdefer allocator.free(rgb);

    if (c.png_image_finish_read(&img, null, rgb.ptr, 0, null) == 0) {
        return error.PngDecodeFailed;
    }
    const gray = try rgbToGray(allocator, rgb, w, h);
    return .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
}

pub fn loadJpeg(allocator: std.mem.Allocator, path_z: [:0]const u8) !ImageData {
    const fp = c.fopen(path_z.ptr, "rb") orelse return error.FileOpenFailed;
    defer _ = c.fclose(fp);

    var cinfo: c.jpeg_decompress_struct = std.mem.zeroes(c.jpeg_decompress_struct);
    var jerr: c.jpeg_error_mgr = std.mem.zeroes(c.jpeg_error_mgr);
    cinfo.err = c.jpeg_std_error(&jerr);
    c.jpeg_CreateDecompress(&cinfo, c.JPEG_LIB_VERSION, @sizeOf(c.jpeg_decompress_struct));
    defer c.jpeg_destroy_decompress(&cinfo);

    c.jpeg_stdio_src(&cinfo, fp);
    _ = c.jpeg_read_header(&cinfo, 1);
    cinfo.out_color_space = c.JCS_RGB;
    _ = c.jpeg_start_decompress(&cinfo);

    const w: u32 = cinfo.output_width;
    const h: u32 = cinfo.output_height;
    const stride = @as(usize, w) * @as(usize, @intCast(cinfo.output_components));
    const rgb = try allocator.alloc(u8, @as(usize, h) * stride);
    errdefer allocator.free(rgb);

    while (cinfo.output_scanline < h) {
        const row_ptr: [*c]u8 = rgb.ptr + @as(usize, cinfo.output_scanline) * stride;
        var rows = [1][*c]u8{row_ptr};
        _ = c.jpeg_read_scanlines(&cinfo, &rows, 1);
    }
    _ = c.jpeg_finish_decompress(&cinfo);

    const gray = try rgbToGray(allocator, rgb, w, h);
    return .{ .width = w, .height = h, .rgb = rgb, .gray = gray };
}

pub fn loadImage(allocator: std.mem.Allocator, path: []const u8) !ImageData {
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);

    if (std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".PNG")) {
        return loadPng(allocator, path_z);
    } else if (std.mem.endsWith(u8, path, ".jpg") or std.mem.endsWith(u8, path, ".jpeg") or
        std.mem.endsWith(u8, path, ".JPG") or std.mem.endsWith(u8, path, ".JPEG"))
    {
        return loadJpeg(allocator, path_z);
    }
    return error.UnsupportedImageFormat;
}

/// Downscale image using bilinear interpolation so max(width, height) <= max_dim.
pub fn downscaleIfNeeded(allocator: std.mem.Allocator, img: *ImageData, max_dim: u32) !f64 {
    const cur_max = @max(img.width, img.height);
    if (max_dim == 0 or cur_max <= max_dim) return 1.0;

    const scale = @as(f64, @floatFromInt(max_dim)) / @as(f64, @floatFromInt(cur_max));
    const nw: u32 = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(img.width)) * scale))));
    const nh: u32 = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(img.height)) * scale))));

    const new_rgb = try allocator.alloc(u8, @as(usize, nw) * @as(usize, nh) * 3);
    errdefer allocator.free(new_rgb);

    const ow: usize = img.width;
    const oh: usize = img.height;

    for (0..nh) |y| {
        const sy = (@as(f64, @floatFromInt(y)) + 0.5) / scale - 0.5;
        const y0: usize = @intFromFloat(std.math.clamp(@floor(sy), 0.0, @as(f64, @floatFromInt(oh - 1))));
        const y1: usize = @min(oh - 1, y0 + 1);
        const wy = std.math.clamp(sy - @as(f64, @floatFromInt(y0)), 0.0, 1.0);

        for (0..nw) |x| {
            const sx = (@as(f64, @floatFromInt(x)) + 0.5) / scale - 0.5;
            const x0: usize = @intFromFloat(std.math.clamp(@floor(sx), 0.0, @as(f64, @floatFromInt(ow - 1))));
            const x1: usize = @min(ow - 1, x0 + 1);
            const wx = std.math.clamp(sx - @as(f64, @floatFromInt(x0)), 0.0, 1.0);

            for (0..3) |ch| {
                const v00: f64 = @floatFromInt(img.rgb[(y0 * ow + x0) * 3 + ch]);
                const v10: f64 = @floatFromInt(img.rgb[(y0 * ow + x1) * 3 + ch]);
                const v01: f64 = @floatFromInt(img.rgb[(y1 * ow + x0) * 3 + ch]);
                const v11: f64 = @floatFromInt(img.rgb[(y1 * ow + x1) * 3 + ch]);
                const v = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v10) + wy * ((1.0 - wx) * v01 + wx * v11);
                new_rgb[(y * nw + x) * 3 + ch] = @intFromFloat(std.math.clamp(@round(v), 0.0, 255.0));
            }
        }
    }

    const new_gray = try rgbToGray(allocator, new_rgb, nw, nh);
    allocator.free(img.rgb);
    allocator.free(img.gray);
    img.width = nw;
    img.height = nh;
    img.rgb = new_rgb;
    img.gray = new_gray;
    return scale;
}

pub fn parseIntrinsicsText(text: []const u8) !Mat3 {
    var vals: [9]f64 = undefined;
    var count: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |tok| {
        if (count >= 9) break;
        vals[count] = try std.fmt.parseFloat(f64, tok);
        count += 1;
    }
    if (count < 9) return error.InvalidIntrinsicsFile;
    return Mat3.fromRows(
        .{ vals[0], vals[1], vals[2] },
        .{ vals[3], vals[4], vals[5] },
        .{ vals[6], vals[7], vals[8] },
    );
}

test "parseIntrinsicsText parses 3x3 K matrix" {
    const txt =
        \\2759.48 0 1520.69
        \\0 2764.16 1006.81
        \\0 0 1
    ;
    const k = try parseIntrinsicsText(txt);
    try std.testing.expectApproxEqAbs(@as(f64, 2759.48), k.m[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1520.69), k.m[0][2], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2764.16), k.m[1][1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1006.81), k.m[1][2], 1e-9);
}
