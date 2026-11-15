const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;

pub const Mat3 = struct {
    m: [3][3]f64,

    pub const identity: Mat3 = .{ .m = .{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    } };

    pub const zero: Mat3 = .{ .m = .{
        .{ 0, 0, 0 },
        .{ 0, 0, 0 },
        .{ 0, 0, 0 },
    } };

    pub fn fromRows(r0: [3]f64, r1: [3]f64, r2: [3]f64) Mat3 {
        return .{ .m = .{ r0, r1, r2 } };
    }

    pub fn mul(a: Mat3, b: Mat3) Mat3 {
        var res = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                var sum: f64 = 0;
                for (0..3) |k| {
                    sum += a.m[i][k] * b.m[k][j];
                }
                res.m[i][j] = sum;
            }
        }
        return res;
    }

    pub fn mulVec(a: Mat3, v: Vec3) Vec3 {
        return .{
            .x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
            .y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
            .z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z,
        };
    }

    pub fn add(a: Mat3, b: Mat3) Mat3 {
        var res = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                res.m[i][j] = a.m[i][j] + b.m[i][j];
            }
        }
        return res;
    }

    pub fn scale(a: Mat3, s: f64) Mat3 {
        var res = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                res.m[i][j] = a.m[i][j] * s;
            }
        }
        return res;
    }

    pub fn transpose(a: Mat3) Mat3 {
        var res = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                res.m[i][j] = a.m[j][i];
            }
        }
        return res;
    }

    pub fn det(a: Mat3) f64 {
        return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
            a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
            a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
    }

    pub fn inv(a: Mat3) ?Mat3 {
        const d = a.det();
        if (@abs(d) < 1e-15) return null;
        const inv_d = 1.0 / d;
        return .{ .m = .{
            .{
                (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) * inv_d,
                (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * inv_d,
                (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * inv_d,
            },
            .{
                (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) * inv_d,
                (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * inv_d,
                (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * inv_d,
            },
            .{
                (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) * inv_d,
                (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * inv_d,
                (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * inv_d,
            },
        } };
    }

    pub fn skew(v: Vec3) Mat3 {
        return .{ .m = .{
            .{ 0, -v.z, v.y },
            .{ v.z, 0, -v.x },
            .{ -v.y, v.x, 0 },
        } };
    }

    pub fn frobeniusNorm(a: Mat3) f64 {
        var sum: f64 = 0;
        for (0..3) |i| {
            for (0..3) |j| {
                sum += a.m[i][j] * a.m[i][j];
            }
        }
        return @sqrt(sum);
    }
};

pub const Svd3Result = struct {
    u: Mat3,
    s: [3]f64,
    vt: Mat3,
};

/// Stack-allocated 3x3 SVD using one-sided Jacobi rotations.
/// Singular values are returned in descending order.
pub fn svd3x3(a: Mat3) Svd3Result {
    var u_data: [9]f64 = undefined;
    for (0..3) |r| {
        for (0..3) |c| {
            u_data[r * 3 + c] = a.m[r][c];
        }
    }
    var v_data: [9]f64 = .{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    var s: [3]f64 = undefined;
    jacobiSvdInPlace(3, 3, &u_data, &s, &v_data);

    var u = Mat3.zero;
    var vt = Mat3.zero;
    for (0..3) |r| {
        for (0..3) |c| {
            u.m[r][c] = u_data[r * 3 + c];
            vt.m[c][r] = v_data[r * 3 + c];
        }
    }
    if (s[2] <= 1e-12) {
        const col0 = Vec3.init(u.m[0][0], u.m[1][0], u.m[2][0]);
        const col1 = Vec3.init(u.m[0][1], u.m[1][1], u.m[2][1]);
        const col2 = col0.cross(col1).normalize();
        u.m[0][2] = col2.x;
        u.m[1][2] = col2.y;
        u.m[2][2] = col2.z;
    }
    return .{ .u = u, .s = s, .vt = vt };
}

pub const DynMat = struct {
    rows: usize,
    cols: usize,
    data: []f64,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !DynMat {
        const data = try allocator.alloc(f64, rows * cols);
        @memset(data, 0);
        return .{ .rows = rows, .cols = cols, .data = data };
    }

    pub fn deinit(self: *DynMat, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }

    pub inline fn get(self: DynMat, r: usize, c: usize) f64 {
        return self.data[r * self.cols + c];
    }

    pub inline fn set(self: *DynMat, r: usize, c: usize, val: f64) void {
        self.data[r * self.cols + c] = val;
    }

    pub fn clone(self: DynMat, allocator: std.mem.Allocator) !DynMat {
        const data = try allocator.dupe(f64, self.data);
        return .{ .rows = self.rows, .cols = self.cols, .data = data };
    }
};

pub const SvdResult = struct {
    u: DynMat,
    s: []f64,
    v: DynMat,

    pub fn deinit(self: *SvdResult, allocator: std.mem.Allocator) void {
        self.u.deinit(allocator);
        allocator.free(self.s);
        self.v.deinit(allocator);
        self.* = undefined;
    }
};

/// Thin SVD for m x n matrix: A = U * diag(S) * V^T.
/// If m < n, A is padded with zero rows internally so V (n x n) still contains the full right-singular nullspace.
pub fn svd(allocator: std.mem.Allocator, a: DynMat) !SvdResult {
    const m_pad = @max(a.rows, a.cols);
    var u = try DynMat.init(allocator, m_pad, a.cols);
    errdefer u.deinit(allocator);
    for (0..a.rows) |r| {
        for (0..a.cols) |c| {
            u.set(r, c, a.get(r, c));
        }
    }

    const s = try allocator.alloc(f64, a.cols);
    errdefer allocator.free(s);

    var v = try DynMat.init(allocator, a.cols, a.cols);
    errdefer v.deinit(allocator);
    for (0..a.cols) |i| {
        v.set(i, i, 1.0);
    }

    jacobiSvdInPlace(m_pad, a.cols, u.data, s, v.data);

    return .{ .u = u, .s = s, .v = v };
}

fn jacobiSvdInPlace(m: usize, n: usize, u: []f64, s: []f64, v: []f64) void {
    const max_sweeps: usize = 60;
    const eps: f64 = 1e-15;

    for (0..max_sweeps) |_| {
        var converged = true;
        for (0..n) |p| {
            for (p + 1..n) |q| {
                var alpha: f64 = 0;
                var beta: f64 = 0;
                var gamma: f64 = 0;
                for (0..m) |i| {
                    const up = u[i * n + p];
                    const uq = u[i * n + q];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }

                const denom = @sqrt(alpha * beta);
                if (denom > 0 and @abs(gamma) > eps * denom) {
                    converged = false;
                    const zeta = (beta - alpha) / (2.0 * gamma);
                    const t_val = if (zeta >= 0)
                        1.0 / (zeta + @sqrt(1.0 + zeta * zeta))
                    else
                        -1.0 / (-zeta + @sqrt(1.0 + zeta * zeta));
                    const c = 1.0 / @sqrt(1.0 + t_val * t_val);
                    const s_rot = c * t_val;

                    for (0..m) |i| {
                        const up = u[i * n + p];
                        const uq = u[i * n + q];
                        u[i * n + p] = c * up - s_rot * uq;
                        u[i * n + q] = s_rot * up + c * uq;
                    }
                    for (0..n) |i| {
                        const vp = v[i * n + p];
                        const vq = v[i * n + q];
                        v[i * n + p] = c * vp - s_rot * vq;
                        v[i * n + q] = s_rot * vp + c * vq;
                    }
                }
            }
        }
        if (converged) break;
    }

    for (0..n) |j| {
        var norm_sq: f64 = 0;
        for (0..m) |i| {
            const val = u[i * n + j];
            norm_sq += val * val;
        }
        const sigma = @sqrt(norm_sq);
        s[j] = sigma;
        if (sigma > 1e-15) {
            const inv_s = 1.0 / sigma;
            for (0..m) |i| {
                u[i * n + j] *= inv_s;
            }
        } else {
            for (0..m) |i| {
                u[i * n + j] = 0;
            }
        }
    }

    // Sort singular values descending
    for (0..n) |i| {
        var max_idx = i;
        for (i + 1..n) |j| {
            if (s[j] > s[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            std.mem.swap(f64, &s[i], &s[max_idx]);
            for (0..m) |r| {
                std.mem.swap(f64, &u[r * n + i], &u[r * n + max_idx]);
            }
            for (0..n) |r| {
                std.mem.swap(f64, &v[r * n + i], &v[r * n + max_idx]);
            }
        }
    }
}

/// Solve A * x = b for square n x n matrix A using Gaussian elimination with partial pivoting.
/// Mutates `a` and `b` in-place; result is written into `b`.
pub fn solveLinearSystemInPlace(n: usize, a: []f64, b: []f64) bool {
    for (0..n) |col| {
        var pivot_row = col;
        var max_val = @abs(a[col * n + col]);
        for (col + 1..n) |row| {
            const v = @abs(a[row * n + col]);
            if (v > max_val) {
                max_val = v;
                pivot_row = row;
            }
        }
        if (max_val < 1e-14) return false;

        if (pivot_row != col) {
            for (col..n) |k| {
                std.mem.swap(f64, &a[col * n + k], &a[pivot_row * n + k]);
            }
            std.mem.swap(f64, &b[col], &b[pivot_row]);
        }

        const inv_pivot = 1.0 / a[col * n + col];
        for (col + 1..n) |row| {
            const factor = a[row * n + col] * inv_pivot;
            if (factor == 0) continue;
            a[row * n + col] = 0;
            for (col + 1..n) |k| {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    var i: usize = n;
    while (i > 0) {
        i -= 1;
        var sum = b[i];
        for (i + 1..n) |k| {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    return true;
}

test "Mat3 inverse and determinant" {
    const a = Mat3.fromRows(
        .{ 2, -1, 0 },
        .{ -1, 2, -1 },
        .{ 0, -1, 2 },
    );
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), a.det(), 1e-12);
    const a_inv = a.inv().?;
    const prod = a.mul(a_inv);
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, prod.m[i][j], 1e-12);
        }
    }
}

test "svd3x3 reconstruction and ordering" {
    const a = Mat3.fromRows(
        .{ 3, 1, 1 },
        .{ -1, 3, 1 },
        .{ 1, 2, -2 },
    );
    const res = svd3x3(a);
    try std.testing.expect(res.s[0] >= res.s[1] and res.s[1] >= res.s[2]);

    const diag_s = Mat3.fromRows(
        .{ res.s[0], 0, 0 },
        .{ 0, res.s[1], 0 },
        .{ 0, 0, res.s[2] },
    );
    const recon = res.u.mul(diag_s).mul(res.vt);
    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expectApproxEqAbs(a.m[i][j], recon.m[i][j], 1e-11);
        }
    }
}

test "DynMat SVD nullspace recovery" {
    const gpa = std.testing.allocator;
    var a = try DynMat.init(gpa, 5, 4);
    defer a.deinit(gpa);

    // Construct rank-3 matrix whose nullspace vector is proportional to (1, -2, 3, -4)
    const rows = [5][4]f64{
        .{ 2, 1, 0, 0 },
        .{ 0, 3, 2, 0 },
        .{ 1, 0, 1, 1 },
        .{ 4, 2, 0, 0 },
        .{ 1, 1, 1, 0.5 },
    };
    for (0..5) |r| {
        // Adjust last column so row dot (1, -2, 3, -4) == 0
        var dot3: f64 = 0;
        const x = [4]f64{ 1, -2, 3, -4 };
        for (0..3) |c| {
            a.set(r, c, rows[r][c]);
            dot3 += rows[r][c] * x[c];
        }
        a.set(r, 3, -dot3 / x[3]);
    }

    var res = try svd(gpa, a);
    defer res.deinit(gpa);

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), res.s[3], 1e-11);

    // Last column of V should be proportional to (1, -2, 3, -4)
    const v3 = [4]f64{ res.v.get(0, 3), res.v.get(1, 3), res.v.get(2, 3), res.v.get(3, 3) };
    const scale = 1.0 / v3[0];
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), v3[1] * scale, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), v3[2] * scale, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, -4.0), v3[3] * scale, 1e-10);
}

test "solveLinearSystemInPlace" {
    var a = [_]f64{
        3,  2, -1,
        2, -2,  4,
        -1, 0.5, -1,
    };
    var b = [_]f64{ 1, -2, 0 };
    try std.testing.expect(solveLinearSystemInPlace(3, &a, &b));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), b[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), b[2], 1e-12);
}
