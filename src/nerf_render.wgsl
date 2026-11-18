@group(0) @binding(3) var<storage, read> occ_flags: array<u32>;
@group(0) @binding(4) var<uniform> r_cfg: RenderUniforms;
@group(0) @binding(5) var<storage, read_write> out_pixels: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(r_cfg.k_skew_dims.y + 0.5);
    let h = u32(r_cfg.k_skew_dims.z + 0.5);
    let y_off = u32(r_cfg.k_skew_dims.w + 0.5);
    let py_u = gid.y + y_off;
    if (gid.x >= w || py_u >= h) {
        return;
    }

    let fx = r_cfg.k_intrinsics.x;
    let fy = r_cfg.k_intrinsics.y;
    let cx = r_cfg.k_intrinsics.z;
    let cy = r_cfg.k_intrinsics.w;
    let skew = r_cfg.k_skew_dims.x;

    let px = f32(gid.x) + 0.5;
    let py = f32(py_u) + 0.5;
    let yn = (py - cy) / fy;
    let xn = (px - cx - skew * yn) / fx;
    let d_cam = normalize(vec3<f32>(xn, yn, 1.0));
    let dir = normalize(
        r_cfg.r_row0.xyz * d_cam.x +
        r_cfg.r_row1.xyz * d_cam.y +
        r_cfg.r_row2.xyz * d_cam.z
    );
    let origin = r_cfg.cam_center.xyz;

    let sh = encodeSphericalHarmonics(dir);
    let n_max = u_cfg.mlp_dims1.w;
    let skip_empty = r_cfg.bg_skip.w > 0.5;
    let occ_res = u_cfg.grid_dims.w;
    let t_far = u_cfg.bounds_range.y;

    var t = u_cfg.bounds_range.x;
    var transmittance: f32 = 1.0;
    var rgb_acc = vec3<f32>(0.0);
    var depth_acc: f32 = 0.0;
    var hits: u32 = 0u;
    let max_steps = n_max * 3u;

    for (var s: u32 = 0u; s < max_steps; s += 1u) {
        if (t >= t_far || transmittance < 1e-4 || hits >= n_max) {
            break;
        }
        let dt = rayStepDt(t);
        let t_mid = t + dt * 0.5;
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
        hits += 1u;
        let alpha = 1.0 - exp(-s_cache.sigma * dt);
        if (alpha < 1e-5) {
            continue;
        }
        let w_s = transmittance * alpha;
        rgb_acc += s_cache.rgb * w_s;
        depth_acc += w_s * t_mid;
        transmittance *= (1.0 - alpha);
    }

    rgb_acc += r_cfg.bg_skip.xyz * transmittance;
    out_pixels[py_u * w + gid.x] = vec4<f32>(rgb_acc, depth_acc);
}
