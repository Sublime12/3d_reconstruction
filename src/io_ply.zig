const std = @import("std");
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const CameraPose = @import("math/lie.zig").CameraPose;

pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub const CameraMeta = struct {
    pose: CameraPose,
    image_path: []const u8,
};

pub const SceneMetadata = struct {
    k: Mat3,
    img_w: u32,
    img_h: u32,
    cameras: []const CameraMeta,
};

pub const LoadedPlyScene = struct {
    points: []Vec3,
    colors: []Color,
    faces: [][3]usize,
    k: ?Mat3,
    img_w: u32,
    img_h: u32,
    cameras: []CameraMeta,

    pub fn deinit(self: *LoadedPlyScene, allocator: std.mem.Allocator) void {
        allocator.free(self.points);
        allocator.free(self.colors);
        allocator.free(self.faces);
        for (self.cameras) |cam| allocator.free(cam.image_path);
        allocator.free(self.cameras);
        self.* = undefined;
    }
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
    return formatPlyMeshWithMeta(allocator, points, colors, faces, poses, null);
}

pub fn formatPlyMeshWithMeta(
    allocator: std.mem.Allocator,
    points: []const Vec3,
    colors: []const Color,
    faces: []const [3]usize,
    poses: []const CameraPose,
    meta: ?SceneMetadata,
) ![]u8 {
    std.debug.assert(points.len == colors.len);
    const total_verts = points.len + poses.len;

    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.print(allocator, "ply\nformat ascii 1.0\n", .{});
    if (meta) |m| {
        try buf.print(allocator, "comment k {d:.6} {d:.6} {d:.6} {d:.6} {d} {d}\n", .{
            m.k.m[0][0], m.k.m[1][1], m.k.m[0][2], m.k.m[1][2], m.img_w, m.img_h,
        });
        for (m.cameras, 0..) |cam, idx| {
            const r = cam.pose.r.m;
            const t = cam.pose.t;
            try buf.print(
                allocator,
                "comment cam {d} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {s}\n",
                .{
                    idx,
                    r[0][0], r[0][1], r[0][2],
                    r[1][0], r[1][1], r[1][2],
                    r[2][0], r[2][1], r[2][2],
                    t.x,     t.y,     t.z,
                    cam.image_path,
                },
            );
        }
    }

    try buf.print(allocator,
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

pub fn parsePlyMesh(allocator: std.mem.Allocator, bytes: []const u8) !LoadedPlyScene {
    var line_it = std.mem.splitScalar(u8, bytes, '\n');

    var n_verts: usize = 0;
    var n_faces: usize = 0;
    var parsed_k: ?Mat3 = null;
    var img_w: u32 = 1200;
    var img_h: u32 = 800;

    var cameras: std.ArrayList(CameraMeta) = .empty;
    errdefer {
        for (cameras.items) |cam| allocator.free(cam.image_path);
        cameras.deinit(allocator);
    }

    while (line_it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "end_header")) break;

        if (std.mem.startsWith(u8, line, "element vertex ")) {
            n_verts = try std.fmt.parseInt(usize, line["element vertex ".len..], 10);
        } else if (std.mem.startsWith(u8, line, "element face ")) {
            n_faces = try std.fmt.parseInt(usize, line["element face ".len..], 10);
        } else if (std.mem.startsWith(u8, line, "comment k ")) {
            var tok = std.mem.tokenizeScalar(u8, line["comment k ".len..], ' ');
            const fx = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const fy = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const cx = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const cy = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            img_w = try std.fmt.parseInt(u32, tok.next() orelse return error.InvalidPlyHeader, 10);
            img_h = try std.fmt.parseInt(u32, tok.next() orelse return error.InvalidPlyHeader, 10);
            parsed_k = Mat3{ .m = .{
                .{ fx, 0.0, cx },
                .{ 0.0, fy, cy },
                .{ 0.0, 0.0, 1.0 },
            } };
        } else if (std.mem.startsWith(u8, line, "comment cam ")) {
            var tok = std.mem.tokenizeScalar(u8, line["comment cam ".len..], ' ');
            _ = tok.next() orelse return error.InvalidPlyHeader; // cam index
            var r: [3][3]f64 = undefined;
            for (0..3) |rr| {
                for (0..3) |cc| {
                    r[rr][cc] = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
                }
            }
            const tx = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const ty = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const tz = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyHeader);
            const img_path = std.mem.trim(u8, tok.rest(), " \r\t");
            try cameras.append(allocator, .{
                .pose = .{ .r = .{ .m = r }, .t = Vec3.init(tx, ty, tz) },
                .image_path = try allocator.dupe(u8, img_path),
            });
        }
    }

    var points_list: std.ArrayList(Vec3) = .empty;
    errdefer points_list.deinit(allocator);
    var colors_list: std.ArrayList(Color) = .empty;
    errdefer colors_list.deinit(allocator);
    var faces_list: std.ArrayList([3]usize) = .empty;
    errdefer faces_list.deinit(allocator);

    try points_list.ensureTotalCapacity(allocator, n_verts);
    try colors_list.ensureTotalCapacity(allocator, n_verts);
    try faces_list.ensureTotalCapacity(allocator, n_faces);

    var v_read: usize = 0;
    while (v_read < n_verts) {
        const raw = line_it.next() orelse break;
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) continue;
        var tok = std.mem.tokenizeScalar(u8, line, ' ');
        const x = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyVertex);
        const y = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyVertex);
        const z = try std.fmt.parseFloat(f64, tok.next() orelse return error.InvalidPlyVertex);
        const cr = try std.fmt.parseInt(u8, tok.next() orelse return error.InvalidPlyVertex, 10);
        const cg = try std.fmt.parseInt(u8, tok.next() orelse return error.InvalidPlyVertex, 10);
        const cb = try std.fmt.parseInt(u8, tok.next() orelse return error.InvalidPlyVertex, 10);
        try points_list.append(allocator, Vec3.init(x, y, z));
        try colors_list.append(allocator, .{ .r = cr, .g = cg, .b = cb });
        v_read += 1;
    }

    var f_read: usize = 0;
    while (f_read < n_faces) {
        const raw = line_it.next() orelse break;
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) continue;
        var tok = std.mem.tokenizeScalar(u8, line, ' ');
        const n_idx = try std.fmt.parseInt(usize, tok.next() orelse return error.InvalidPlyFace, 10);
        if (n_idx == 3) {
            const f0 = try std.fmt.parseInt(usize, tok.next() orelse return error.InvalidPlyFace, 10);
            const f1 = try std.fmt.parseInt(usize, tok.next() orelse return error.InvalidPlyFace, 10);
            const f2 = try std.fmt.parseInt(usize, tok.next() orelse return error.InvalidPlyFace, 10);
            try faces_list.append(allocator, .{ f0, f1, f2 });
        }
        f_read += 1;
    }

    // Strip trailing unreferenced camera marker vertices so camera markers don't linger as stray points
    if (cameras.items.len > 0 and points_list.items.len >= cameras.items.len) {
        const trim_len = points_list.items.len - cameras.items.len;
        points_list.shrinkAndFree(allocator, trim_len);
        colors_list.shrinkAndFree(allocator, trim_len);
    }

    return .{
        .points = try points_list.toOwnedSlice(allocator),
        .colors = try colors_list.toOwnedSlice(allocator),
        .faces = try faces_list.toOwnedSlice(allocator),
        .k = parsed_k,
        .img_w = img_w,
        .img_h = img_h,
        .cameras = try cameras.toOwnedSlice(allocator),
    };
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

test "formatPlyMeshWithMeta and parsePlyMesh round-trip mesh and camera metadata" {
    const gpa = std.testing.allocator;
    const pts = [_]Vec3{
        Vec3.init(0, 0, 5),
        Vec3.init(1, 0, 5),
        Vec3.init(0, 1, 5),
    };
    const cols = [_]Color{
        .{ .r = 200, .g = 10, .b = 20 },
        .{ .r = 10, .g = 200, .b = 30 },
        .{ .r = 20, .g = 30, .b = 200 },
    };
    const faces = [_][3]usize{.{ 0, 1, 2 }};
    const pose = CameraPose{ .r = Mat3.identity, .t = Vec3.init(0.5, -0.25, 1.0) };
    const poses = [_]CameraPose{pose};
    const cam_metas = [_]CameraMeta{.{ .pose = pose, .image_path = "data/fountain-p11/0000.jpg" }};
    const k = Mat3{ .m = .{
        .{ 950.0, 0.0, 600.0 },
        .{ 0.0, 950.0, 400.0 },
        .{ 0.0, 0.0, 1.0 },
    } };

    const ply = try formatPlyMeshWithMeta(gpa, &pts, &cols, &faces, &poses, .{
        .k = k,
        .img_w = 1200,
        .img_h = 800,
        .cameras = &cam_metas,
    });
    defer gpa.free(ply);

    var loaded = try parsePlyMesh(gpa, ply);
    defer loaded.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 3), loaded.points.len);
    try std.testing.expectEqual(@as(usize, 1), loaded.faces.len);
    try std.testing.expectEqual(@as(usize, 1), loaded.cameras.len);
    try std.testing.expectEqualStrings("data/fountain-p11/0000.jpg", loaded.cameras[0].image_path);
    try std.testing.expectApproxEqAbs(950.0, loaded.k.?.m[0][0], 1e-4);
    try std.testing.expectApproxEqAbs(0.5, loaded.cameras[0].pose.t.x, 1e-6);
}
