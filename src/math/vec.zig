const std = @import("std");

pub const Vec2 = struct {
    x: f64,
    y: f64,

    pub fn init(x: f64, y: f64) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }

    pub fn scale(a: Vec2, s: f64) Vec2 {
        return .{ .x = a.x * s, .y = a.y * s };
    }

    pub fn dot(a: Vec2, b: Vec2) f64 {
        return a.x * b.x + a.y * b.y;
    }

    pub fn normSq(a: Vec2) f64 {
        return a.dot(a);
    }

    pub fn norm(a: Vec2) f64 {
        return @sqrt(a.normSq());
    }

    pub fn dist(a: Vec2, b: Vec2) f64 {
        return a.sub(b).norm();
    }
};

pub const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    pub const zero: Vec3 = .{ .x = 0, .y = 0, .z = 0 };

    pub fn init(x: f64, y: f64, z: f64) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn fromSlice(s: [3]f64) Vec3 {
        return .{ .x = s[0], .y = s[1], .z = s[2] };
    }

    pub fn toArray(a: Vec3) [3]f64 {
        return .{ a.x, a.y, a.z };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub fn neg(a: Vec3) Vec3 {
        return .{ .x = -a.x, .y = -a.y, .z = -a.z };
    }

    pub fn scale(a: Vec3, s: f64) Vec3 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s };
    }

    pub fn dot(a: Vec3, b: Vec3) f64 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn normSq(a: Vec3) f64 {
        return a.dot(a);
    }

    pub fn norm(a: Vec3) f64 {
        return @sqrt(a.normSq());
    }

    pub fn normalize(a: Vec3) Vec3 {
        const n = a.norm();
        if (n == 0) return a;
        return a.scale(1.0 / n);
    }

    pub fn dist(a: Vec3, b: Vec3) f64 {
        return a.sub(b).norm();
    }
};

test "Vec2 and Vec3 basic operations" {
    const a = Vec3.init(1, 0, 0);
    const b = Vec3.init(0, 1, 0);
    const c = a.cross(b);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c.y, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c.z, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), Vec2.init(3, 4).norm(), 1e-12);
}
