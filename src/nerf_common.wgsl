const MAX_LEVELS: u32 = 12u;
const MAX_GEO_IN: u32 = 24u;
const MAX_GEO_H: u32 = 64u;
const MAX_GEO_OUT: u32 = 32u;
const MAX_COL_IN: u32 = 48u;
const MAX_COL_H: u32 = 64u;
const SH_DIM: u32 = 16u;
const LEAKY_ALPHA: f32 = 0.01;
const FP_SCALE_WG: f32 = 262144.0;

struct NerfUniforms {
    bounds_center_inv_r: vec4<f32>,
    bounds_range: vec4<f32>,
    grid_dims: vec4<u32>,
    grid_scales: vec4<f32>,
    mlp_dims0: vec4<u32>,
    mlp_dims1: vec4<u32>,
    mlp_offs0: vec4<u32>,
    mlp_offs1: vec4<u32>,
    run_cfg: vec4<u32>,
    opt_cfg: vec4<f32>,
    reg_cfg: vec4<f32>,
};

struct TrainingRayGpu {
    origin_jitter: vec4<f32>,
    dir_depth: vec4<f32>,
    target_rgb_pad: vec4<f32>,
    bg_pad: vec4<f32>,
};

struct RayForwardResult {
    d_rgb_loss: vec4<f32>,
    grad_meta: vec4<f32>,
    extra: vec4<f32>,
};

struct HitSample {
    u_dt: vec4<f32>,
    rgb_t: vec4<f32>,
    t_a_w_s: vec4<f32>,
};

struct RenderUniforms {
    k_intrinsics: vec4<f32>,
    k_skew_dims: vec4<f32>,
    cam_center: vec4<f32>,
    r_row0: vec4<f32>,
    r_row1: vec4<f32>,
    r_row2: vec4<f32>,
    bg_skip: vec4<f32>,
};

struct SampleCache {
    geo_in: array<f32, MAX_GEO_IN>,
    geo_h: array<f32, MAX_GEO_H>,
    col_in: array<f32, MAX_COL_IN>,
    col_h: array<f32, MAX_COL_H>,
    col_logits: array<f32, 3>,
    rgb: vec3<f32>,
    sigma_raw: f32,
    sigma: f32,
};

@group(0) @binding(0) var<uniform> u_cfg: NerfUniforms;
@group(0) @binding(1) var<storage, read> mlp_params: array<f32>;
@group(0) @binding(2) var<storage, read> hash_params: array<f32>;

var<private> s_cache: SampleCache;

fn truncExp(x: f32) -> f32 {
    return exp(clamp(x, -15.0, 10.0));
}

fn truncExpDeriv(raw: f32, sigma: f32) -> f32 {
    return select(0.0, sigma, raw > -15.0 && raw < 10.0);
}

fn sigmoidf(x: f32) -> f32 {
    let cx = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-cx));
}

fn toUnit01(p: vec3<f32>) -> vec4<f32> {
    let n = (p - u_cfg.bounds_center_inv_r.xyz) * u_cfg.bounds_center_inv_r.w;
    if (u_cfg.bounds_range.w < 0.5) {
        let uu = 0.5 * n + vec3<f32>(0.5);
        if (uu.x < 0.0 || uu.x >= 1.0 || uu.y < 0.0 || uu.y >= 1.0 || uu.z < 0.0 || uu.z >= 1.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        return vec4<f32>(uu, 1.0);
    }
    let norm_sq = dot(n, n);
    var c = n;
    if (norm_sq > 1.0) {
        let norm = sqrt(norm_sq);
        let s = (2.0 - 1.0 / norm) / norm;
        c = n * s;
    }
    let clamped = clamp(0.25 * c + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0 - 1e-6));
    return vec4<f32>(clamped, 1.0);
}

fn rayStepDt(t: f32) -> f32 {
    let core_r = 1.0 / max(u_cfg.bounds_center_inv_r.w, 1e-6);
    let dt_override = u_cfg.bounds_range.z;
    let dt_base = select(core_r * 0.012, dt_override, dt_override > 0.0);
    if (u_cfg.bounds_range.w < 0.5) {
        return dt_base;
    }
    let n_f = f32(max(16u, u_cfg.mlp_dims1.w));
    return max(dt_base, t * (2.4 / n_f));
}

fn occVoxelIndex(pos_u: vec3<f32>, res: u32) -> u32 {
    let rf = f32(res);
    let ix = min(res - 1u, u32(max(0.0, pos_u.x * rf)));
    let iy = min(res - 1u, u32(max(0.0, pos_u.y * rf)));
    let iz = min(res - 1u, u32(max(0.0, pos_u.z * rf)));
    return ix + res * (iy + res * iz);
}

fn encodeSphericalHarmonics(d: vec3<f32>) -> array<f32, 16> {
    let x = d.x;
    let y = d.y;
    let z = d.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;
    return array<f32, 16>(
        0.28209479,
        -0.48860251 * y,
        0.48860251 * z,
        -0.48860251 * x,
        1.09254843 * xy,
        -1.09254843 * yz,
        0.31539157 * (3.0 * zz - 1.0),
        -1.09254843 * xz,
        0.54627422 * (xx - yy),
        -0.59004359 * y * (3.0 * xx - yy),
        2.89061144 * xy * z,
        -0.45704580 * y * (5.0 * zz - 1.0),
        0.37317633 * z * (5.0 * zz - 3.0),
        -0.45704580 * x * (5.0 * zz - 1.0),
        1.44530572 * z * (xx - yy),
        -0.59004359 * x * (xx - 3.0 * yy),
    );
}

fn gridResAtLevel(lvl: u32) -> u32 {
    let sc = select(pow(u_cfg.grid_scales.y, f32(lvl)), 1.0, lvl == 0u);
    return max(2u, u32(floor(u_cfg.grid_scales.x * sc)));
}

fn hashIndex(res: u32, ix: u32, iy: u32, iz: u32) -> u32 {
    let stride = res + 1u;
    if (stride <= 1024u && stride * stride * stride <= u_cfg.grid_dims.y) {
        return ix + stride * (iy + stride * iz);
    }
    let h = ix ^ (iy * 2654435761u) ^ (iz * 805459861u);
    return h & u_cfg.grid_dims.z;
}

fn encodeHashGrid(pos_u: vec3<f32>) {
    let num_levels = u_cfg.grid_dims.x;
    let table_size = u_cfg.grid_dims.y;
    for (var lvl: u32 = 0u; lvl < num_levels; lvl += 1u) {
        let res = gridResAtLevel(lvl);
        let res_f = f32(res - 1u);
        let p = clamp(pos_u * res_f, vec3<f32>(0.0), vec3<f32>(res_f - 1e-5));
        let ip = vec3<u32>(floor(p));
        let fp = p - vec3<f32>(ip);
        let wp0 = vec3<f32>(1.0) - fp;
        let base = lvl * table_size * 2u;

        var acc0: f32 = 0.0;
        var acc1: f32 = 0.0;
        for (var c: u32 = 0u; c < 8u; c += 1u) {
            let dx = c & 1u;
            let dy = (c >> 1u) & 1u;
            let dz = (c >> 2u) & 1u;
            let wx = select(wp0.x, fp.x, dx != 0u);
            let wy = select(wp0.y, fp.y, dy != 0u);
            let wz = select(wp0.z, fp.z, dz != 0u);
            let w = wx * wy * wz;
            let slot = base + hashIndex(res, ip.x + dx, ip.y + dy, ip.z + dz) * 2u;
            acc0 += w * hash_params[slot + 0u];
            acc1 += w * hash_params[slot + 1u];
        }
        s_cache.geo_in[lvl * 2u + 0u] = acc0;
        s_cache.geo_in[lvl * 2u + 1u] = acc1;
    }
}

fn evalDensityUnit01(pos_u: vec3<f32>) -> f32 {
    encodeHashGrid(pos_u);
    let geo_in = u_cfg.mlp_dims0.x;
    let geo_h = u_cfg.mlp_dims0.y;
    let w0_off = u_cfg.mlp_offs0.x;
    let b0_off = u_cfg.mlp_offs0.y;
    let w1_off = u_cfg.mlp_offs0.z;
    let b1_off = u_cfg.mlp_offs0.w;

    for (var o: u32 = 0u; o < geo_h; o += 1u) {
        var sum = mlp_params[b0_off + o];
        let row = w0_off + o * geo_in;
        for (var i: u32 = 0u; i < geo_in; i += 1u) {
            sum += s_cache.geo_in[i] * mlp_params[row + i];
        }
        s_cache.geo_h[o] = select(LEAKY_ALPHA * sum, sum, sum > 0.0);
    }

    var raw_sigma = mlp_params[b1_off];
    for (var i: u32 = 0u; i < geo_h; i += 1u) {
        raw_sigma += s_cache.geo_h[i] * mlp_params[w1_off + i];
    }
    return truncExp(raw_sigma);
}

fn evalSample(pos_u: vec3<f32>, sh: array<f32, 16>) {
    encodeHashGrid(pos_u);
    let geo_in = u_cfg.mlp_dims0.x;
    let geo_h = u_cfg.mlp_dims0.y;
    let geo_out = u_cfg.mlp_dims0.z;
    let col_in = u_cfg.mlp_dims0.w;
    let col_h = u_cfg.mlp_dims1.x;
    let geo_feat_dim = u_cfg.mlp_dims1.z;

    let w0_off = u_cfg.mlp_offs0.x;
    let b0_off = u_cfg.mlp_offs0.y;
    let w1_off = u_cfg.mlp_offs0.z;
    let b1_off = u_cfg.mlp_offs0.w;
    let w2_off = u_cfg.mlp_offs1.x;
    let b2_off = u_cfg.mlp_offs1.y;
    let w3_off = u_cfg.mlp_offs1.z;
    let b3_off = u_cfg.mlp_offs1.w;

    for (var o: u32 = 0u; o < geo_h; o += 1u) {
        var sum = mlp_params[b0_off + o];
        let row = w0_off + o * geo_in;
        for (var i: u32 = 0u; i < geo_in; i += 1u) {
            sum += s_cache.geo_in[i] * mlp_params[row + i];
        }
        s_cache.geo_h[o] = select(LEAKY_ALPHA * sum, sum, sum > 0.0);
    }

    for (var o: u32 = 0u; o < geo_out; o += 1u) {
        var sum = mlp_params[b1_off + o];
        let row = w1_off + o * geo_h;
        for (var i: u32 = 0u; i < geo_h; i += 1u) {
            sum += s_cache.geo_h[i] * mlp_params[row + i];
        }
        if (o == 0u) {
            s_cache.sigma_raw = sum;
            s_cache.sigma = truncExp(sum);
        } else {
            s_cache.col_in[o - 1u] = sum;
        }
    }

    for (var k: u32 = 0u; k < SH_DIM; k += 1u) {
        s_cache.col_in[geo_feat_dim + k] = sh[k];
    }

    for (var o: u32 = 0u; o < col_h; o += 1u) {
        var sum = mlp_params[b2_off + o];
        let row = w2_off + o * col_in;
        for (var i: u32 = 0u; i < col_in; i += 1u) {
            sum += s_cache.col_in[i] * mlp_params[row + i];
        }
        s_cache.col_h[o] = select(LEAKY_ALPHA * sum, sum, sum > 0.0);
    }

    for (var o: u32 = 0u; o < 3u; o += 1u) {
        var sum = mlp_params[b3_off + o];
        let row = w3_off + o * col_h;
        for (var i: u32 = 0u; i < col_h; i += 1u) {
            sum += s_cache.col_h[i] * mlp_params[row + i];
        }
        s_cache.col_logits[o] = sum;
    }

    s_cache.rgb = vec3<f32>(
        sigmoidf(s_cache.col_logits[0]),
        sigmoidf(s_cache.col_logits[1]),
        sigmoidf(s_cache.col_logits[2]),
    );
}
