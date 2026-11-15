const std = @import("std");
const Vec2 = @import("vec.zig").Vec2;
const Vec3 = @import("vec.zig").Vec3;
const Mat3 = @import("mat.zig").Mat3;

pub const CameraPose = struct {
    r: Mat3,
    t: Vec3,

    pub const identity: CameraPose = .{
        .r = Mat3.identity,
        .t = Vec3.zero,
    };

    /// Transform a 3D world point into camera coordinates: X_cam = R * X_world + t.
    pub fn transformPoint(self: CameraPose, x_world: Vec3) Vec3 {
        return self.r.mulVec(x_world).add(self.t);
    }

    /// Camera center in world coordinates: C = -R^T * t.
    pub fn cameraCenter(self: CameraPose) Vec3 {
        return self.r.transpose().mulVec(self.t).neg();
    }
};

/// Rodrigues exponential map: axis-angle vector w -> rotation matrix R in SO(3).
pub fn so3Exp(w: Vec3) Mat3 {
    const theta_sq = w.normSq();
    const k = Mat3.skew(w);
    const k2 = k.mul(k);

    var a: f64 = undefined;
    var b: f64 = undefined;
    if (theta_sq < 1e-16) {
        a = 1.0 - theta_sq / 6.0;
        b = 0.5 - theta_sq / 24.0;
    } else {
        const theta = @sqrt(theta_sq);
        a = @sin(theta) / theta;
        b = (1.0 - @cos(theta)) / theta_sq;
    }

    return Mat3.identity.add(k.scale(a)).add(k2.scale(b));
}

/// Rodrigues logarithm map: rotation matrix R in SO(3) -> axis-angle vector w.
pub fn so3Log(r: Mat3) Vec3 {
    const v_skew = Vec3.init(
        (r.m[2][1] - r.m[1][2]) * 0.5,
        (r.m[0][2] - r.m[2][0]) * 0.5,
        (r.m[1][0] - r.m[0][1]) * 0.5,
    );
    const sin_theta = v_skew.norm();
    const cos_theta = std.math.clamp((r.m[0][0] + r.m[1][1] + r.m[2][2] - 1.0) * 0.5, -1.0, 1.0);
    const theta = std.math.atan2(sin_theta, cos_theta);

    if (theta < 1e-8) {
        const factor = 1.0 + theta * theta / 6.0;
        return v_skew.scale(factor);
    }

    if (std.math.pi - theta < 1e-4) {
        // Near pi: extract axis from diagonal of R + I
        const xx = @max(0.0, (r.m[0][0] + 1.0) * 0.5);
        const yy = @max(0.0, (r.m[1][1] + 1.0) * 0.5);
        const zz = @max(0.0, (r.m[2][2] + 1.0) * 0.5);
        var axis: Vec3 = undefined;
        if (xx >= yy and xx >= zz) {
            const x = @sqrt(xx);
            axis = Vec3.init(x, (r.m[0][1] + r.m[1][0]) / (4.0 * x), (r.m[0][2] + r.m[2][0]) / (4.0 * x));
        } else if (yy >= xx and yy >= zz) {
            const y = @sqrt(yy);
            axis = Vec3.init((r.m[0][1] + r.m[1][0]) / (4.0 * y), y, (r.m[1][2] + r.m[2][1]) / (4.0 * y));
        } else {
            const z = @sqrt(zz);
            axis = Vec3.init((r.m[0][2] + r.m[2][0]) / (4.0 * z), (r.m[1][2] + r.m[2][1]) / (4.0 * z), z);
        }
        return axis.normalize().scale(theta);
    }

    return v_skew.scale(theta / sin_theta);
}

/// Project 3D world point X into 2D pixel coordinates using intrinsics K and pose (R, t).
/// Returns null if point is behind or on the camera plane (Z <= 1e-9).
pub fn projectPoint(k: Mat3, pose: CameraPose, x_world: Vec3) ?Vec2 {
    const xc = pose.transformPoint(x_world);
    if (xc.z <= 1e-9) return null;
    const xn = xc.x / xc.z;
    const yn = xc.y / xc.z;
    return Vec2.init(
        k.m[0][0] * xn + k.m[0][1] * yn + k.m[0][2],
        k.m[1][0] * xn + k.m[1][1] * yn + k.m[1][2],
    );
}

test "so3Exp and so3Log round-trip" {
    const axes = [_]Vec3{
        Vec3.zero,
        Vec3.init(1e-10, -2e-10, 3e-10),
        Vec3.init(0.3, -0.5, 0.7),
        Vec3.init(std.math.pi, 0, 0),
        Vec3.init(0, std.math.pi * 0.9999, 0),
    };

    for (axes) |w| {
        const r = so3Exp(w);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), r.det(), 1e-10);
        const rrt = r.mul(r.transpose());
        for (0..3) |i| {
            for (0..3) |j| {
                const expected: f64 = if (i == j) 1.0 else 0.0;
                try std.testing.expectApproxEqAbs(expected, rrt.m[i][j], 1e-10);
            }
        }
        const w2 = so3Log(r);
        const r2 = so3Exp(w2);
        for (0..3) |i| {
            for (0..3) |j| {
                try std.testing.expectApproxEqAbs(r.m[i][j], r2.m[i][j], 1e-9);
            }
        }
    }
}
