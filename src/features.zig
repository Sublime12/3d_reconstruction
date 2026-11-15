const std = @import("std");
const Vec2 = @import("math/vec.zig").Vec2;

pub const Keypoint = struct {
    x: f64,
    y: f64,
    score: f64,
    angle: f64,
    desc: [4]u64,
};

pub const Match = struct {
    idx1: usize,
    idx2: usize,
    dist: u32,
};

pub const ImageView = struct {
    width: u32,
    height: u32,
    gray: []const u8,
};

// 256 deterministic rotated-BRIEF sampling pairs in [-15, 15]^2
const brief_pairs: [256][4]i8 = blk: {
    @setEvalBranchQuota(10000);
    var pairs: [256][4]i8 = undefined;
    var state: u64 = 0x9e3779b97f4a7c15;
    for (0..256) |i| {
        for (0..4) |c| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            const v: i8 = @intCast(@as(i32, @intCast(state % 29)) - 14);
            pairs[i][c] = v;
        }
    }
    break :blk pairs;
};

pub const TileConfig = struct {
    grid_cols: usize = 24,
    grid_rows: usize = 16,
    min_harris_score: f32 = 250.0,
    min_tile_dist_px: f64 = 3.5,
};

const Candidate = struct {
    x: usize,
    y: usize,
    score: f32,
};

pub fn detectAndCompute(
    allocator: std.mem.Allocator,
    img: ImageView,
    max_keypoints: usize,
) ![]Keypoint {
    return detectAndComputeWithConfig(allocator, img, max_keypoints, .{});
}

/// Detect Harris corners using spatially-uniform tiled selection and compute oriented 256-bit BRIEF descriptors.
pub fn detectAndComputeWithConfig(
    allocator: std.mem.Allocator,
    img: ImageView,
    max_keypoints: usize,
    tile_cfg: TileConfig,
) ![]Keypoint {
    const w: usize = img.width;
    const h: usize = img.height;
    if (w < 40 or h < 40 or max_keypoints == 0) return try allocator.alloc(Keypoint, 0);

    const blurred = try gaussianBlur5x5(allocator, img.gray, w, h);
    defer allocator.free(blurred);

    const score = try computeHarrisScores(allocator, blurred, w, h, tile_cfg.min_harris_score);
    defer allocator.free(score);

    return try detectFromScores(allocator, w, h, blurred, score, max_keypoints, tile_cfg);
}

pub fn computeHarrisScores(
    allocator: std.mem.Allocator,
    blurred: []const u8,
    w: usize,
    h: usize,
    min_harris_score: f32,
) ![]f32 {
    const n = w * h;
    const ix2 = try allocator.alloc(f32, n);
    defer allocator.free(ix2);
    const iy2 = try allocator.alloc(f32, n);
    defer allocator.free(iy2);
    const ixy = try allocator.alloc(f32, n);
    defer allocator.free(ixy);
    @memset(ix2, 0);
    @memset(iy2, 0);
    @memset(ixy, 0);

    for (1..h - 1) |y| {
        for (1..w - 1) |x| {
            const p = y * w + x;
            const gx: f32 =
                -@as(f32, @floatFromInt(blurred[(y - 1) * w + (x - 1)])) +
                @as(f32, @floatFromInt(blurred[(y - 1) * w + (x + 1)])) -
                2.0 * @as(f32, @floatFromInt(blurred[y * w + (x - 1)])) +
                2.0 * @as(f32, @floatFromInt(blurred[y * w + (x + 1)])) -
                @as(f32, @floatFromInt(blurred[(y + 1) * w + (x - 1)])) +
                @as(f32, @floatFromInt(blurred[(y + 1) * w + (x + 1)]));
            const gy: f32 =
                -@as(f32, @floatFromInt(blurred[(y - 1) * w + (x - 1)])) -
                2.0 * @as(f32, @floatFromInt(blurred[(y - 1) * w + x])) -
                @as(f32, @floatFromInt(blurred[(y - 1) * w + (x + 1)])) +
                @as(f32, @floatFromInt(blurred[(y + 1) * w + (x - 1)])) +
                2.0 * @as(f32, @floatFromInt(blurred[(y + 1) * w + x])) +
                @as(f32, @floatFromInt(blurred[(y + 1) * w + (x + 1)]));
            ix2[p] = gx * gx;
            iy2[p] = gy * gy;
            ixy[p] = gx * gy;
        }
    }

    const score = try allocator.alloc(f32, n);
    @memset(score, 0);

    const border: usize = 10;
    for (border..h - border) |y| {
        for (border..w - border) |x| {
            var sxx: f32 = 0;
            var syy: f32 = 0;
            var sxy: f32 = 0;
            for (0..3) |dy| {
                const ry = y + dy - 1;
                for (0..3) |dx| {
                    const rx = x + dx - 1;
                    const idx = ry * w + rx;
                    sxx += ix2[idx];
                    syy += iy2[idx];
                    sxy += ixy[idx];
                }
            }
            const det_m = sxx * syy - sxy * sxy;
            const trace_m = sxx + syy;
            const r = det_m - 0.04 * trace_m * trace_m;
            if (r > min_harris_score) {
                score[y * w + x] = r;
            }
        }
    }

    return score;
}

/// Runs tiled 5x5 NMS, spatially-uniform round-robin selection, orientation, and BRIEF from precomputed blur & Harris scores.
pub fn detectFromScores(
    allocator: std.mem.Allocator,
    w: usize,
    h: usize,
    blurred: []const u8,
    score: []const f32,
    max_keypoints: usize,
    tile_cfg: TileConfig,
) ![]Keypoint {
    if (w < 40 or h < 40 or max_keypoints == 0) return try allocator.alloc(Keypoint, 0);
    const border: usize = 10;

    const cols = @max(@as(usize, 1), @min(tile_cfg.grid_cols, w / 24));
    const rows = @max(@as(usize, 1), @min(tile_cfg.grid_rows, h / 24));
    const n_tiles = cols * rows;

    const tile_cands = try allocator.alloc(std.ArrayList(Candidate), n_tiles);
    for (tile_cands) |*tc| tc.* = .empty;
    defer {
        for (tile_cands) |*tc| tc.deinit(allocator);
        allocator.free(tile_cands);
    }

    // 5x5 NMS binned directly into spatial tiles
    for (border..h - border) |y| {
        const ty = @min(rows - 1, (y * rows) / h);
        for (border..w - border) |x| {
            const val = score[y * w + x];
            if (val <= 0) continue;

            var is_max = true;
            var dy: isize = -2;
            while (dy <= 2) : (dy += 1) {
                var dx: isize = -2;
                while (dx <= 2) : (dx += 1) {
                    if (dx == 0 and dy == 0) continue;
                    const ny: usize = @intCast(@as(isize, @intCast(y)) + dy);
                    const nx: usize = @intCast(@as(isize, @intCast(x)) + dx);
                    if (score[ny * w + nx] > val) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) break;
            }
            if (!is_max) continue;

            const tx = @min(cols - 1, (x * cols) / w);
            try tile_cands[ty * cols + tx].append(allocator, .{ .x = x, .y = y, .score = val });
        }
    }

    for (tile_cands) |*tc| {
        std.mem.sort(Candidate, tc.items, {}, struct {
            fn cmp(_: void, a: Candidate, b: Candidate) bool {
                return a.score > b.score;
            }
        }.cmp);
    }

    const tile_sel = try allocator.alloc(std.ArrayList(Candidate), n_tiles);
    for (tile_sel) |*ts| ts.* = .empty;
    defer {
        for (tile_sel) |*ts| ts.deinit(allocator);
        allocator.free(tile_sel);
    }
    const tile_cursor = try allocator.alloc(usize, n_tiles);
    defer allocator.free(tile_cursor);
    @memset(tile_cursor, 0);

    const min_dist_sq = tile_cfg.min_tile_dist_px * tile_cfg.min_tile_dist_px;
    const base_quota = @max(@as(usize, 1), max_keypoints / n_tiles);

    var selected: std.ArrayList(Candidate) = .empty;
    defer selected.deinit(allocator);

    // Stage 1: Give every tile up to base_quota well-spaced keypoints
    for (0..n_tiles) |ti| {
        const cands = tile_cands[ti].items;
        var idx: usize = 0;
        while (idx < cands.len and tile_sel[ti].items.len < base_quota and selected.items.len < max_keypoints) : (idx += 1) {
            const c = cands[idx];
            if (isWellSpaced(tile_sel[ti].items, c, min_dist_sq)) {
                try tile_sel[ti].append(allocator, c);
                try selected.append(allocator, c);
            }
        }
        tile_cursor[ti] = idx;
    }

    // Stage 2: Round-robin remaining budget across tiles that still have spaced candidates
    while (selected.items.len < max_keypoints) {
        var added_any = false;
        for (0..n_tiles) |ti| {
            if (selected.items.len >= max_keypoints) break;
            const cands = tile_cands[ti].items;
            while (tile_cursor[ti] < cands.len) {
                const c = cands[tile_cursor[ti]];
                tile_cursor[ti] += 1;
                if (isWellSpaced(tile_sel[ti].items, c, min_dist_sq)) {
                    try tile_sel[ti].append(allocator, c);
                    try selected.append(allocator, c);
                    added_any = true;
                    break;
                }
            }
        }
        if (!added_any) break;
    }

    const out = try allocator.alloc(Keypoint, selected.items.len);
    for (selected.items, 0..) |c, i| {
        const angle = computeCentroidAngle(blurred, w, c.x, c.y);
        const kx: f64 = @floatFromInt(c.x);
        const ky: f64 = @floatFromInt(c.y);
        out[i] = .{
            .x = kx,
            .y = ky,
            .score = c.score,
            .angle = angle,
            .desc = computeBrief(blurred, w, h, kx, ky, angle),
        };
    }
    return out;
}

fn isWellSpaced(existing: []const Candidate, cand: Candidate, min_dist_sq: f64) bool {
    if (min_dist_sq <= 0) return true;
    const cx: f64 = @floatFromInt(cand.x);
    const cy: f64 = @floatFromInt(cand.y);
    for (existing) |e| {
        const dx = cx - @as(f64, @floatFromInt(e.x));
        const dy = cy - @as(f64, @floatFromInt(e.y));
        if (dx * dx + dy * dy < min_dist_sq) return false;
    }
    return true;
}

fn computeCentroidAngle(blurred: []const u8, w: usize, x: usize, y: usize) f64 {
    var m10: f64 = 0;
    var m01: f64 = 0;
    var oy: isize = -7;
    while (oy <= 7) : (oy += 1) {
        var ox: isize = -7;
        while (ox <= 7) : (ox += 1) {
            if (ox * ox + oy * oy > 49) continue;
            const py: usize = @intCast(@as(isize, @intCast(y)) + oy);
            const px: usize = @intCast(@as(isize, @intCast(x)) + ox);
            const intensity: f64 = @floatFromInt(blurred[py * w + px]);
            m10 += @as(f64, @floatFromInt(ox)) * intensity;
            m01 += @as(f64, @floatFromInt(oy)) * intensity;
        }
    }
    return std.math.atan2(m01, m10);
}

fn computeBrief(img: []const u8, w: usize, h: usize, kx: f64, ky: f64, angle: f64) [4]u64 {
    const cos_a = @cos(angle);
    const sin_a = @sin(angle);
    var desc = [4]u64{ 0, 0, 0, 0 };

    for (brief_pairs, 0..) |pair, bit_idx| {
        const ax_f = @as(f64, @floatFromInt(pair[0]));
        const ay_f = @as(f64, @floatFromInt(pair[1]));
        const bx_f = @as(f64, @floatFromInt(pair[2]));
        const by_f = @as(f64, @floatFromInt(pair[3]));

        const rax = @as(isize, @intFromFloat(@round(kx + ax_f * cos_a - ay_f * sin_a)));
        const ray = @as(isize, @intFromFloat(@round(ky + ax_f * sin_a + ay_f * cos_a)));
        const rbx = @as(isize, @intFromFloat(@round(kx + bx_f * cos_a - by_f * sin_a)));
        const rby = @as(isize, @intFromFloat(@round(ky + bx_f * sin_a + by_f * cos_a)));

        const ax: usize = @intCast(std.math.clamp(rax, 0, @as(isize, @intCast(w - 1))));
        const ay: usize = @intCast(std.math.clamp(ray, 0, @as(isize, @intCast(h - 1))));
        const bx: usize = @intCast(std.math.clamp(rbx, 0, @as(isize, @intCast(w - 1))));
        const by: usize = @intCast(std.math.clamp(rby, 0, @as(isize, @intCast(h - 1))));

        if (img[ay * w + ax] < img[by * w + bx]) {
            const word = bit_idx >> 6;
            const shift: u6 = @intCast(bit_idx & 63);
            desc[word] |= (@as(u64, 1) << shift);
        }
    }
    return desc;
}

pub fn gaussianBlur5x5(allocator: std.mem.Allocator, src: []const u8, w: usize, h: usize) ![]u8 {
    const tmp = try allocator.alloc(u16, w * h);
    defer allocator.free(tmp);
    const dst = try allocator.alloc(u8, w * h);

    // Horizontal pass [1 4 6 4 1] / 16
    for (0..h) |y| {
        for (0..w) |x| {
            const xm2 = if (x >= 2) x - 2 else 0;
            const xm1 = if (x >= 1) x - 1 else 0;
            const xp1 = @min(w - 1, x + 1);
            const xp2 = @min(w - 1, x + 2);
            const sum: u16 =
                @as(u16, src[y * w + xm2]) +
                4 * @as(u16, src[y * w + xm1]) +
                6 * @as(u16, src[y * w + x]) +
                4 * @as(u16, src[y * w + xp1]) +
                @as(u16, src[y * w + xp2]);
            tmp[y * w + x] = sum;
        }
    }

    // Vertical pass [1 4 6 4 1] / 256
    for (0..h) |y| {
        const ym2 = if (y >= 2) y - 2 else 0;
        const ym1 = if (y >= 1) y - 1 else 0;
        const yp1 = @min(h - 1, y + 1);
        const yp2 = @min(h - 1, y + 2);
        for (0..w) |x| {
            const sum: u32 =
                @as(u32, tmp[ym2 * w + x]) +
                4 * @as(u32, tmp[ym1 * w + x]) +
                6 * @as(u32, tmp[y * w + x]) +
                4 * @as(u32, tmp[yp1 * w + x]) +
                @as(u32, tmp[yp2 * w + x]);
            dst[y * w + x] = @intCast(sum >> 8);
        }
    }
    return dst;
}

pub inline fn hammingDist(a: [4]u64, b: [4]u64) u32 {
    return @as(u32, @popCount(a[0] ^ b[0])) +
        @as(u32, @popCount(a[1] ^ b[1])) +
        @as(u32, @popCount(a[2] ^ b[2])) +
        @as(u32, @popCount(a[3] ^ b[3]));
}

/// Pairwise descriptor matching with Lowe's ratio test and mutual cross-check.
pub fn matchDescriptors(
    allocator: std.mem.Allocator,
    kps1: []const Keypoint,
    kps2: []const Keypoint,
    ratio_thresh: f64,
    max_hamming: u32,
) ![]Match {
    if (kps1.len == 0 or kps2.len < 2) return try allocator.alloc(Match, 0);

    const best_12 = try allocator.alloc(?usize, kps1.len);
    defer allocator.free(best_12);
    const best_dist_12 = try allocator.alloc(u32, kps1.len);
    defer allocator.free(best_dist_12);

    const best_21 = try allocator.alloc(usize, kps2.len);
    defer allocator.free(best_21);
    const best_dist_21 = try allocator.alloc(u32, kps2.len);
    defer allocator.free(best_dist_21);
    @memset(best_21, 0);
    @memset(best_dist_21, std.math.maxInt(u32));

    for (kps1, 0..) |kp1, i| {
        var d1: u32 = std.math.maxInt(u32);
        var d2: u32 = std.math.maxInt(u32);
        var b_idx: usize = 0;
        for (kps2, 0..) |kp2, j| {
            const d = hammingDist(kp1.desc, kp2.desc);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                b_idx = j;
            } else if (d < d2) {
                d2 = d;
            }
            if (d < best_dist_21[j]) {
                best_dist_21[j] = d;
                best_21[j] = i;
            }
        }
        if (d1 <= max_hamming and @as(f64, @floatFromInt(d1)) < ratio_thresh * @as(f64, @floatFromInt(d2))) {
            best_12[i] = b_idx;
            best_dist_12[i] = d1;
        } else {
            best_12[i] = null;
        }
    }

    var matches: std.ArrayList(Match) = .empty;
    defer matches.deinit(allocator);
    for (best_12, 0..) |opt_j, i| {
        if (opt_j) |j| {
            if (best_21[j] == i) {
                try matches.append(allocator, .{
                    .idx1 = i,
                    .idx2 = j,
                    .dist = best_dist_12[i],
                });
            }
        }
    }
    return try matches.toOwnedSlice(allocator);
}

test "detectAndCompute and matchDescriptors recover synthetic image translation" {
    const gpa = std.testing.allocator;
    const w: u32 = 160;
    const h: u32 = 120;
    const img1 = try gpa.alloc(u8, w * h);
    defer gpa.free(img1);
    const img2 = try gpa.alloc(u8, w * h);
    defer gpa.free(img2);

    // Generate deterministic textured pattern
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    for (0..h) |y| {
        for (0..w) |x| {
            const xf: f64 = @floatFromInt(x);
            const yf: f64 = @floatFromInt(y);
            const val = 127.0 +
                50.0 * @sin(xf * 0.25) * @cos(yf * 0.25) +
                40.0 * @sin(xf * 0.11 + yf * 0.17) +
                @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -8, 8)));
            img1[y * w + x] = @intFromFloat(std.math.clamp(val, 0.0, 255.0));
        }
    }

    // Shift img1 by (dx=8, dy=-5) into img2
    const dx_gt: isize = 8;
    const dy_gt: isize = -5;
    @memset(img2, 128);
    for (0..h) |y| {
        for (0..w) |x| {
            const sx = @as(isize, @intCast(x)) - dx_gt;
            const sy = @as(isize, @intCast(y)) - dy_gt;
            if (sx >= 0 and sx < w and sy >= 0 and sy < h) {
                img2[y * w + x] = img1[@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))];
            }
        }
    }

    const kps1 = try detectAndCompute(gpa, .{ .width = w, .height = h, .gray = img1 }, 300);
    defer gpa.free(kps1);
    const kps2 = try detectAndCompute(gpa, .{ .width = w, .height = h, .gray = img2 }, 300);
    defer gpa.free(kps2);

    try std.testing.expect(kps1.len >= 20);
    try std.testing.expect(kps2.len >= 20);

    const matches = try matchDescriptors(gpa, kps1, kps2, 0.8, 64);
    defer gpa.free(matches);
    try std.testing.expect(matches.len >= 10);

    var correct: usize = 0;
    for (matches) |m| {
        const dx = kps2[m.idx2].x - kps1[m.idx1].x;
        const dy = kps2[m.idx2].y - kps1[m.idx1].y;
        if (@abs(dx - @as(f64, @floatFromInt(dx_gt))) <= 2.0 and @abs(dy - @as(f64, @floatFromInt(dy_gt))) <= 2.0) {
            correct += 1;
        }
    }
    try std.testing.expect(correct * 2 >= matches.len);
}

test "detectAndCompute distributes keypoints across high and low contrast regions" {
    const gpa = std.testing.allocator;
    const w: u32 = 320;
    const h: u32 = 240;
    const img = try gpa.alloc(u8, w * h);
    defer gpa.free(img);

    // Top-left quadrant has very high contrast; other 3 quadrants have subtle low-contrast texture.
    for (0..h) |y| {
        for (0..w) |x| {
            const xf: f64 = @floatFromInt(x);
            const yf: f64 = @floatFromInt(y);
            const amp: f64 = if (x < w / 2 and y < h / 2) 100.0 else 10.0;
            const val = 127.0 + amp * @sin(xf * 0.35) * @cos(yf * 0.35);
            img[y * w + x] = @intFromFloat(std.math.clamp(val, 0.0, 255.0));
        }
    }

    const kps = try detectAndCompute(gpa, .{ .width = w, .height = h, .gray = img }, 200);
    defer gpa.free(kps);

    var quad_counts = [4]usize{ 0, 0, 0, 0 };
    for (kps) |kp| {
        const qx: usize = if (kp.x >= @as(f64, @floatFromInt(w / 2))) 1 else 0;
        const qy: usize = if (kp.y >= @as(f64, @floatFromInt(h / 2))) 2 else 0;
        quad_counts[qy + qx] += 1;
    }

    for (quad_counts) |cnt| {
        try std.testing.expect(cnt >= 15);
    }
}
