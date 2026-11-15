const std = @import("std");
const Vec3 = @import("math/vec.zig").Vec3;
const CameraPose = @import("math/lie.zig").CameraPose;

pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub fn formatPly(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    poses: []const CameraPose,
) ![]u8 {
    return formatPlyMesh(allocator, points, colors, &.{}, poses);
}

pub fn formatPlyMesh(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    faces: []const [3]usize,
    poses: []const CameraPose,
) ![]u8 {
    std.debug.assert(points.len == colors.len);
    const total_verts = points.len + poses.len;

    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.print(allocator,
        \\ply
        \\format ascii 1.0
        \\element vertex {d}
        \\property float x
        \\property float y
        \\property float z
        \\property uchar red
        \\property uchar green
        \\property uchar blue
        \\element face {d}
        \\property list uchar int vertex_indices
        \\end_header
        \\
    , .{ total_verts, faces.len });

    for (points, colors) |pt, col| {
        try buf.print(allocator, "{d:.6} {d:.6} {d:.6} {d} {d} {d}\n", .{
            cleanZero(pt.x), cleanZero(pt.y), cleanZero(pt.z), col.r, col.g, col.b,
        });
    }

    for (poses) |pose| {
        const c = pose.cameraCenter();
        try buf.print(allocator, "{d:.6} {d:.6} {d:.6} 0 255 0\n", .{
            cleanZero(c.x), cleanZero(c.y), cleanZero(c.z),
        });
    }

    for (faces) |f| {
        try buf.print(allocator, "3 {d} {d} {d}\n", .{ f[0], f[1], f[2] });
    }

    return try buf.toOwnedSlice(allocator);
}

fn cleanZero(v: f64) f64 {
    return if (v == 0) 0.0 else v;
}

test "formatPly generates valid PLY header and vertices" {
    const gpa = std.testing.allocator;
    const pts = [_]Vec3{Vec3.init(1.0, 2.0, 3.0)};
    const cols = [_]Color{.{ .r = 10, .g = 20, .b = 30 }};
    const poses = [_]CameraPose{CameraPose.identity};

    const ply = try formatPly(gpa, &pts, &cols, &poses);
    defer gpa.free(ply);

    try std.testing.expect(std.mem.indexOf(u8, ply, "element vertex 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, ply, "1.000000 2.000000 3.000000 10 20 30") != null);
    try std.testing.expect(std.mem.indexOf(u8, ply, "0.000000 0.000000 0.000000 0 255 0") != null);
}
