@group(0) @binding(3) var<storage, read> occ_flags: array<u32>;
@group(0) @binding(4) var<storage, read> rays: array<TrainingRayGpu>;
@group(0) @binding(5) var<storage, read_write> ray_res: array<RayForwardResult>;
@group(0) @binding(6) var<storage, read_write> hit_samples: array<HitSample>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r_idx = gid.x;
    let batch_rays = u_cfg.run_cfg.x;
    if (r_idx >= batch_rays) {
        return;
    }

    let ray = rays[r_idx];
    let origin = ray.origin_jitter.xyz;
    let jitter = ray.origin_jitter.w;
    let dir = ray.dir_depth.xyz;
    let target_depth = ray.dir_depth.w;
    let target_rgb = ray.target_rgb_pad.xyz;
    let bg = ray.bg_pad.xyz;

    let sh = encodeSphericalHarmonics(dir);
    let n_max = u_cfg.mlp_dims1.w;
    let skip_empty = u_cfg.run_cfg.z != 0u;
    let occ_res = u_cfg.grid_dims.w;
    let t_far = u_cfg.bounds_range.y;

    var t = u_cfg.bounds_range.x;
    var transmittance: f32 = 1.0;
    var rgb_acc = vec3<f32>(0.0);
    var depth_acc: f32 = 0.0;
    var pref_w: f32 = 0.0;
    var pref_wt: f32 = 0.0;
    var dist_loss: f32 = 0.0;
    var count: u32 = 0u;
    let max_steps = n_max * 3u;

    for (var s: u32 = 0u; s < max_steps; s += 1u) {
        if (t >= t_far || transmittance < 1e-4 || count >= n_max) {
            break;
        }
        let dt = rayStepDt(t);
        let t_mid = t + dt * (0.25 + 0.5 * jitter);
        let pos = origin + dir * t_mid;
        t += dt;

        let u4 = toUnit01(pos);
        if (u4.w < 0.5) {
            continue;
        }
        let pos_u = u4.xyz;
        if (skip_empty && occ_flags[occVoxelIndex(pos_u, occ_res)] == 0u) {
            continue;
        }

        evalSample(pos_u, sh);
        let alpha = 1.0 - exp(-s_cache.sigma * dt);
        let w = transmittance * alpha;

        hit_samples[count * batch_rays + r_idx] = HitSample(
            vec4<f32>(pos_u, dt),
            vec4<f32>(s_cache.rgb, t_mid),
            vec4<f32>(transmittance, alpha, w, s_cache.sigma),
        );

        rgb_acc += s_cache.rgb * w;
        depth_acc += w * t_mid;
        dist_loss += 2.0 * w * (t_mid * pref_w - pref_wt) + (1.0 / 3.0) * w * w * dt;
        pref_w += w;
        pref_wt += w * t_mid;
        transmittance *= (1.0 - alpha);
        count += 1u;
    }

    rgb_acc += bg * transmittance;

    let diff = rgb_acc - target_rgb;
    let inv3 = 1.0 / 3.0;
    var loss = dot(diff, diff) * inv3;
    let d_loss_rgb_hat = diff * (2.0 * inv3);

    var d_loss_depth_hat: f32 = 0.0;
    let depth_w = u_cfg.grid_scales.z;
    if (depth_w > 0.0 && target_depth > 0.0) {
        let inv_z = 1.0 / max(target_depth, 1.0);
        let dz = (depth_acc - target_depth) * inv_z;
        loss += depth_w * dz * dz;
        d_loss_depth_hat = 2.0 * depth_w * dz * inv_z;
    }
    let dist_w = u_cfg.reg_cfg.w;
    if (dist_w > 0.0) {
        loss += dist_w * dist_loss;
    }

    let suffix_rgb_init = transmittance * dot(d_loss_rgb_hat, bg);
    ray_res[r_idx] = RayForwardResult(
        vec4<f32>(d_loss_rgb_hat, loss),
        vec4<f32>(d_loss_depth_hat, suffix_rgb_init, transmittance, f32(count)),
        vec4<f32>(pref_w, pref_wt, 0.0, 0.0),
    );
}
