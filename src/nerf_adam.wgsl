struct AdamUniforms {
    len_pad: vec4<u32>,
    params: vec4<f32>,
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> adam_cfg: AdamUniforms;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> grads: array<i32>;
@group(0) @binding(3) var<storage, read_write> adam_m: array<f32>;
@group(0) @binding(4) var<storage, read_write> adam_v: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.y * (num_wg.x * 64u) + gid.x;
    if (i >= adam_cfg.len_pad.x) {
        return;
    }

    let g_raw = grads[i];
    let m_prev = adam_m[i];
    if (g_raw == 0 && m_prev == 0.0) {
        return;
    }
    grads[i] = 0;

    let grad_to_mean = adam_cfg.extra.x;
    let g = f32(g_raw) * grad_to_mean;
    let lr = adam_cfg.params.x;
    let weight_decay = adam_cfg.params.y;
    let inv_bc1 = adam_cfg.params.z;
    let inv_bc2 = adam_cfg.params.w;

    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let eps: f32 = 1e-8;

    let w_prev = weights[i];
    let grad = g + weight_decay * w_prev;
    let m = beta1 * m_prev + (1.0 - beta1) * grad;
    let v = beta2 * adam_v[i] + (1.0 - beta2) * grad * grad;
    adam_m[i] = m;
    adam_v[i] = v;

    let m_hat = m * inv_bc1;
    let v_hat = v * inv_bc2;
    weights[i] = w_prev - lr * m_hat / (sqrt(v_hat) + eps);
}
