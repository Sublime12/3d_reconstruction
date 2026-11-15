const std = @import("std");

pub const vec = @import("math/vec.zig");
pub const mat = @import("math/mat.zig");
pub const lie = @import("math/lie.zig");
pub const features = @import("features.zig");
pub const dataset = @import("dataset.zig");
pub const triangulation = @import("triangulation.zig");
pub const epipolar = @import("epipolar.zig");
pub const pnp = @import("pnp.zig");
pub const bundle_adjust = @import("bundle_adjust.zig");
pub const io_ply = @import("io_ply.zig");
pub const pipeline = @import("pipeline.zig");
pub const mesh = @import("mesh.zig");
pub const dense = @import("dense.zig");
pub const render = @import("render.zig");
pub const viewer_math = @import("viewer_math.zig");
pub const gpu = @import("gpu.zig");

pub const CliOptions = struct {
    data_dir: []const u8 = "data/fountain-p11",
    out_path: []const u8 = "output/reconstruction.ply",
    max_dim: u32 = 1200,
    num_threads: usize = 1,
    use_gpu: bool = false,
    verify_views_dir: ?[]const u8 = null,
    show_help: bool = false,
};

pub fn parseCliArgs(args: []const []const u8) !CliOptions {
    var opts: CliOptions = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--data") and i + 1 < args.len) {
            i += 1;
            opts.data_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--out") and i + 1 < args.len) {
            i += 1;
            opts.out_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-dim") and i + 1 < args.len) {
            i += 1;
            opts.max_dim = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            i += 1;
            opts.num_threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--gpu")) {
            opts.use_gpu = true;
        } else if (std.mem.eql(u8, args[i], "--verify-views") and i + 1 < args.len) {
            i += 1;
            opts.verify_views_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            opts.show_help = true;
        }
    }
    return opts;
}

test "parseCliArgs parses flags" {
    const argv = [_][]const u8{
        "sfm",
        "--data",
        "data/test",
        "--out",
        "out/m.ply",
        "--max-dim",
        "800",
        "--threads",
        "4",
        "--gpu",
        "--verify-views",
        "output/views",
    };
    const opts = try parseCliArgs(&argv);
    try std.testing.expectEqualStrings("data/test", opts.data_dir);
    try std.testing.expectEqualStrings("out/m.ply", opts.out_path);
    try std.testing.expectEqual(@as(u32, 800), opts.max_dim);
    try std.testing.expectEqual(@as(usize, 4), opts.num_threads);
    try std.testing.expect(opts.use_gpu);
    try std.testing.expectEqualStrings("output/views", opts.verify_views_dir.?);
    try std.testing.expect(!opts.show_help);
}

test {
    std.testing.refAllDecls(@This());
}
