@group(0) @binding(3) var<storage, read_write> occ_density: array<f32>;
@group(0) @binding(4) var<storage, read_write> occ_flags: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let res = u_cfg.grid_dims.w;
    let total = res * res * res;
    let idx = gid.y * (num_wg.x * 64u) + gid.x;
    if (idx >= total) {
        return;
    }

    let ix = idx % res;
    let t = idx / res;
    let iy = t % res;
    let iz = t / res;

    let prev = occ_density[idx];
    if (prev < 0.0) {
        occ_flags[idx] = 0u;
        return;
    }

    let inv_r = 1.0 / f32(res);
    let pos_u = (vec3<f32>(f32(ix), f32(iy), f32(iz)) + u_cfg.reg_cfg.xyz) * inv_r;
    let sigma = evalDensityUnit01(pos_u);

    let decay = u_cfg.opt_cfg.x;
    let threshold = u_cfg.opt_cfg.y;
    let blended = max(prev * decay, sigma);
    occ_density[idx] = blended;
    occ_flags[idx] = select(0u, 1u, blended > threshold);
}
