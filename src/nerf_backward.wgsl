@group(0) @binding(3) var<storage, read> rays: array<TrainingRayGpu>;
@group(0) @binding(4) var<storage, read> ray_res: array<RayForwardResult>;
@group(0) @binding(5) var<storage, read> hit_samples: array<HitSample>;
@group(0) @binding(6) var<storage, read_write> mlp_grad: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> hash_grad: array<atomic<i32>>;

const MAX_WG_MLP: u32 = 3840u;
var<workgroup> wg_mlp_grad: array<atomic<i32>, MAX_WG_MLP>;

fn addMlpGrad(idx: u32, val: f32, use_wg: bool, mlp_global_scale: f32) {
    if (val == 0.0) {
        return;
    }
    let clamped = clamp(val, -32.0, 32.0);
    if (use_wg) {
        let q = i32(round(clamped * FP_SCALE_WG));
        if (q != 0) {
            atomicAdd(&wg_mlp_grad[idx], q);
        }
    } else {
        let q = i32(round(clamped * mlp_global_scale));
        if (q != 0) {
            atomicAdd(&mlp_grad[idx], q);
        }
    }
}

fn addHashGrad(idx: u32, val: f32, hash_global_scale: f32) {
    if (val == 0.0) {
        return;
    }
    let q = i32(round(clamp(val, -32.0, 32.0) * hash_global_scale));
    if (q != 0) {
        atomicAdd(&hash_grad[idx], q);
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let mlp_total = u_cfg.run_cfg.w;
    let use_wg = mlp_total <= MAX_WG_MLP;
    let mlp_global_scale = u_cfg.opt_cfg.z;
    let hash_global_scale = u_cfg.opt_cfg.w;

    if (use_wg) {
        var k = lid.x;
        while (k < mlp_total) {
            atomicStore(&wg_mlp_grad[k], 0);
            k += 64u;
        }
        workgroupBarrier();
    }

    let r_idx = gid.x;
    let batch_rays = u_cfg.run_cfg.x;
    if (r_idx < batch_rays) {
        let ray = rays[r_idx];
        let res = ray_res[r_idx];
        let d_loss_rgb_hat = res.d_rgb_loss.xyz;
        let d_loss_depth_hat = res.grad_meta.x;
        var suffix_all = res.grad_meta.y;
        let count = u32(res.grad_meta.w + 0.5);
        let w_tot = res.extra.x;
        let wt_tot = res.extra.y;
        let dist_w = u_cfg.reg_cfg.w;
        var suf_w: f32 = 0.0;
        var suf_wt: f32 = 0.0;

        let sh = encodeSphericalHarmonics(ray.dir_depth.xyz);

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

        let num_levels = u_cfg.grid_dims.x;
        let table_size = u_cfg.grid_dims.y;

        let rot_3 = lid.x % 3u;
        let rot_col_h = lid.x % col_h;
        let rot_col_in = lid.x % col_in;
        let rot_geo_out = lid.x % geo_out;
        let rot_geo_h = lid.x % geo_h;
        let rot_geo_in = lid.x % geo_in;

        var d_col_h: array<f32, MAX_COL_H>;
        var d_col_in: array<f32, MAX_COL_IN>;
        var d_geo_out: array<f32, MAX_GEO_OUT>;
        var d_geo_h: array<f32, MAX_GEO_H>;
        var d_geo_in: array<f32, MAX_GEO_IN>;

        var rev = count;
        while (rev > 0u) {
            rev -= 1u;
            let hs = hit_samples[rev * batch_rays + r_idx];
            let pos_u = hs.u_dt.xyz;
            let dt = hs.u_dt.w;
            let t_mid = hs.rgb_t.w;
            let transm = hs.t_a_w_s.x;
            let alpha = hs.t_a_w_s.y;
            let weight = hs.t_a_w_s.z;

            evalSample(pos_u, sh);

            let t_next = transm * (1.0 - alpha);
            var g_w = dot(d_loss_rgb_hat, s_cache.rgb);
            if (d_loss_depth_hat != 0.0) {
                g_w += d_loss_depth_hat * t_mid;
            }
            if (dist_w > 0.0) {
                let bw = w_tot - suf_w - weight;
                let bwt = wt_tot - suf_wt - weight * t_mid;
                let g_dist = 2.0 * (t_mid * (bw - suf_w) - (bwt - suf_wt) + (1.0 / 3.0) * weight * dt);
                g_w += dist_w * g_dist;
            }
            let d_sigma = dt * (t_next * g_w - suffix_all);
            suffix_all += weight * g_w;
            suf_w += weight;
            suf_wt += weight * t_mid;

            let d_rgb_sample = d_loss_rgb_hat * weight;
            var d_logits = array<f32, 3>(
                d_rgb_sample.x * s_cache.rgb.x * (1.0 - s_cache.rgb.x),
                d_rgb_sample.y * s_cache.rgb.y * (1.0 - s_cache.rgb.y),
                d_rgb_sample.z * s_cache.rgb.z * (1.0 - s_cache.rgb.z),
            );
            if (abs(d_sigma * s_cache.sigma) < 1e-6 && max(abs(d_logits[0]), max(abs(d_logits[1]), abs(d_logits[2]))) < 1e-6) {
                continue;
            }

            for (var i: u32 = 0u; i < col_h; i += 1u) {
                d_col_h[i] = 0.0;
            }
            var o3 = rot_3;
            for (var raw_o: u32 = 0u; raw_o < 3u; raw_o += 1u) {
                let gz = d_logits[o3];
                if (gz != 0.0) {
                    addMlpGrad(b3_off + o3, gz, use_wg, mlp_global_scale);
                    let row = w3_off + o3 * col_h;
                    var i = rot_col_h;
                    for (var raw_i: u32 = 0u; raw_i < col_h; raw_i += 1u) {
                        addMlpGrad(row + i, gz * s_cache.col_h[i], use_wg, mlp_global_scale);
                        d_col_h[i] += gz * mlp_params[row + i];
                        i += 1u;
                        if (i == col_h) { i = 0u; }
                    }
                }
                o3 += 1u;
                if (o3 == 3u) { o3 = 0u; }
            }

            for (var i: u32 = 0u; i < col_in; i += 1u) {
                d_col_in[i] = 0.0;
            }
            var o_ch = rot_col_h;
            for (var raw_o: u32 = 0u; raw_o < col_h; raw_o += 1u) {
                let go = d_col_h[o_ch];
                if (go != 0.0) {
                    let act = select(LEAKY_ALPHA, 1.0, s_cache.col_h[o_ch] > 0.0);
                    let gz = go * act;
                    addMlpGrad(b2_off + o_ch, gz, use_wg, mlp_global_scale);
                    let row = w2_off + o_ch * col_in;
                    var i = rot_col_in;
                    for (var raw_i: u32 = 0u; raw_i < col_in; raw_i += 1u) {
                        addMlpGrad(row + i, gz * s_cache.col_in[i], use_wg, mlp_global_scale);
                        d_col_in[i] += gz * mlp_params[row + i];
                        i += 1u;
                        if (i == col_in) { i = 0u; }
                    }
                }
                o_ch += 1u;
                if (o_ch == col_h) { o_ch = 0u; }
            }

            d_geo_out[0] = d_sigma * truncExpDeriv(s_cache.sigma_raw, s_cache.sigma);
            for (var i: u32 = 0u; i < geo_feat_dim; i += 1u) {
                d_geo_out[1u + i] = d_col_in[i];
            }

            for (var i: u32 = 0u; i < geo_h; i += 1u) {
                d_geo_h[i] = 0.0;
            }
            var o_go = rot_geo_out;
            for (var raw_o: u32 = 0u; raw_o < geo_out; raw_o += 1u) {
                let gz = d_geo_out[o_go];
                if (gz != 0.0) {
                    addMlpGrad(b1_off + o_go, gz, use_wg, mlp_global_scale);
                    let row = w1_off + o_go * geo_h;
                    var i = rot_geo_h;
                    for (var raw_i: u32 = 0u; raw_i < geo_h; raw_i += 1u) {
                        addMlpGrad(row + i, gz * s_cache.geo_h[i], use_wg, mlp_global_scale);
                        d_geo_h[i] += gz * mlp_params[row + i];
                        i += 1u;
                        if (i == geo_h) { i = 0u; }
                    }
                }
                o_go += 1u;
                if (o_go == geo_out) { o_go = 0u; }
            }

            for (var i: u32 = 0u; i < geo_in; i += 1u) {
                d_geo_in[i] = 0.0;
            }
            var o_gh = rot_geo_h;
            for (var raw_o: u32 = 0u; raw_o < geo_h; raw_o += 1u) {
                let go = d_geo_h[o_gh];
                if (go != 0.0) {
                    let act = select(LEAKY_ALPHA, 1.0, s_cache.geo_h[o_gh] > 0.0);
                    let gz = go * act;
                    addMlpGrad(b0_off + o_gh, gz, use_wg, mlp_global_scale);
                    let row = w0_off + o_gh * geo_in;
                    var i = rot_geo_in;
                    for (var raw_i: u32 = 0u; raw_i < geo_in; raw_i += 1u) {
                        addMlpGrad(row + i, gz * s_cache.geo_in[i], use_wg, mlp_global_scale);
                        d_geo_in[i] += gz * mlp_params[row + i];
                        i += 1u;
                        if (i == geo_in) { i = 0u; }
                    }
                }
                o_gh += 1u;
                if (o_gh == geo_h) { o_gh = 0u; }
            }

            for (var lvl: u32 = 0u; lvl < num_levels; lvl += 1u) {
                let g0 = d_geo_in[lvl * 2u + 0u];
                let g1 = d_geo_in[lvl * 2u + 1u];
                if (g0 == 0.0 && g1 == 0.0) {
                    continue;
                }
                let res = gridResAtLevel(lvl);
                let res_f = f32(res - 1u);
                let p = clamp(pos_u * res_f, vec3<f32>(0.0), vec3<f32>(res_f - 1e-5));
                let ip = vec3<u32>(floor(p));
                let fp = p - vec3<f32>(ip);
                let wp0 = vec3<f32>(1.0) - fp;
                let base = lvl * table_size * 2u;

                for (var c: u32 = 0u; c < 8u; c += 1u) {
                    let dx = c & 1u;
                    let dy = (c >> 1u) & 1u;
                    let dz = (c >> 2u) & 1u;
                    let wx = select(wp0.x, fp.x, dx != 0u);
                    let wy = select(wp0.y, fp.y, dy != 0u);
                    let wz = select(wp0.z, fp.z, dz != 0u);
                    let w = wx * wy * wz;
                    let slot = base + hashIndex(res, ip.x + dx, ip.y + dy, ip.z + dz) * 2u;
                    addHashGrad(slot + 0u, w * g0, hash_global_scale);
                    addHashGrad(slot + 1u, w * g1, hash_global_scale);
                }
            }
        }
    }

    if (use_wg) {
        workgroupBarrier();
        let flush_ratio = mlp_global_scale / FP_SCALE_WG;
        var k = lid.x;
        while (k < mlp_total) {
            let raw = atomicLoad(&wg_mlp_grad[k]);
            if (raw != 0) {
                let q = i32(round(f32(raw) * flush_ratio));
                if (q != 0) {
                    atomicAdd(&mlp_grad[k], q);
                }
            }
            k += 64u;
        }
    }
}
