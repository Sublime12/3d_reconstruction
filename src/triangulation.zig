const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const DynMat = @import("math/mat.zig").DynMat;
const svd = @import("math/mat.zig").svd;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;

/// Build 3x4 projection matrix P = K * [R | t].
pub fn makeProjectionMatrix(k: Mat3, pose: CameraPose) [3][4]f64 {
    const kr = k.mul(pose.r);
    const kt = k.mulVec(pose.t);
    return .{
        .{ kr.m[0][0], kr.m[0][1], kr.m[0][2], kt.x },
        .{ kr.m[1][0], kr.m[1][1], kr.m[1][2], kt.y },
        .{ kr.m[2][0], kr.m[2][1], kr.m[2][2], kt.z },
    };
}

/// Linear DLT triangulation from two views (P1, P2) and pixel coordinates (x1, x2).
pub fn triangulatePoint(
    allocator: std.mem.Allocator,
    p1: [3][4]f64,
    p2: [3][4]f64,
    x1: Vec2,
    x2: Vec2,
) !?Vec3 {
    const projs = [_][3][4]f64{ p1, p2 };
    const obs = [_]Vec2{ x1, x2 };
    return triangulateMultiView(allocator, &projs, &obs);
}

/// Linear DLT triangulation from N >= 2 views.
pub fn triangulateMultiView(
    allocator: std.mem.Allocator,
    projections: []const [3][4]f64,
    observations: []const Vec2,
) !?Vec3 {
    std.debug.assert(projections.len == observations.len);
    std.debug.assert(projections.len >= 2);

    var a = try DynMat.init(allocator, projections.len * 2, 4);
    defer a.deinit(allocator);

    for (projections, observations, 0..) |p, x, i| {
        const r0 = i * 2;
        const r1 = r0 + 1;
        for (0..4) |c| {
            a.set(r0, c, x.x * p[2][c] - p[0][c]);
            a.set(r1, c, x.y * p[2][c] - p[1][c]);
        }
    }

    var res = try svd(allocator, a);
    defer res.deinit(allocator);

    const w = res.v.get(3, 3);
    if (@abs(w) < 1e-12) return null;

    return Vec3.init(
        res.v.get(0, 3) / w,
        res.v.get(1, 3) / w,
        res.v.get(2, 3) / w,
    );
}

/// Compute triangulation angle (in radians) between rays from camera centers c1, c2 to 3D point x.
pub fn triangulationAngle(c1: Vec3, c2: Vec3, x: Vec3) f64 {
    const r1 = x.sub(c1).normalize();
    const r2 = x.sub(c2).normalize();
    const cos_a = std.math.clamp(r1.dot(r2), -1.0, 1.0);
    return std.math.acos(cos_a);
}

/// Euclidean pixel reprojection error ||project(K, pose, X) - x2d||, or null if behind camera.
pub fn reprojectionError(k: Mat3, pose: CameraPose, x3d: Vec3, x2d: Vec2) ?f64 {
    const proj = lie.projectPoint(k, pose, x3d) orelse return null;
    return proj.dist(x2d);
}

test "Linear DLT triangulation recovers synthetic 3D point" {
    const gpa = std.testing.allocator;
    const k = Mat3.fromRows(
        .{ 800, 0, 320 },
        .{ 0, 800, 240 },
        .{ 0, 0, 1 },
    );
    const pose1 = CameraPose.identity;
    const pose2 = CameraPose{
        .r = lie.so3Exp(Vec3.init(0.05, -0.2, 0.02)),
        .t = Vec3.init(-1.5, 0.1, 0.2),
    };

    const p1 = makeProjectionMatrix(k, pose1);
    const p2 = makeProjectionMatrix(k, pose2);

    const pts = [_]Vec3{
        Vec3.init(0.5, -0.3, 5.0),
        Vec3.init(-1.2, 0.8, 7.5),
        Vec3.init(2.0, 1.1, 10.0),
    };

    for (pts) |pt| {
        const uv1 = lie.projectPoint(k, pose1, pt).?;
        const uv2 = lie.projectPoint(k, pose2, pt).?;
        const est = (try triangulatePoint(gpa, p1, p2, uv1, uv2)).?;
        try std.testing.expectApproxEqAbs(pt.x, est.x, 1e-8);
        try std.testing.expectApproxEqAbs(pt.y, est.y, 1e-8);
        try std.testing.expectApproxEqAbs(pt.z, est.z, 1e-8);
    }
}
