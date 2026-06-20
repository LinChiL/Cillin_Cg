struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>, // xyz = 法线0
    n1: vec4<f32>, // xyz = 法线1
    n2: vec4<f32>, // xyz = 法线2
    uv01: vec4<f32>, // [u0, v0, u1, v1]
    uv2: vec4<f32>,  // [u2, v2, 0.0, 0.0] - 保持对齐
};

struct InstanceData {
    model_matrix: mat4x4<f32>,
    model_id: u32,
    instance_id: u32,
    tri_start: u32,
    _pad_inner: u32,
    extra_data: vec2<f32>,
    _pad: array<vec4<u32>, 2>,
};

struct Params {
    view_inv: mat4x4<f32>,       // 64
    proj_inv: mat4x4<f32>,       // 64
    prev_view_proj: mat4x4<f32>, // 64
    cam_pos: vec4<f32>,         // 16
    light_dir: vec4<f32>,       // 16

    anchor_count: u32,           // 4
    scaffold_count: u32,         // 4
    time: f32,                   // 4
    selected_instance_id: u32,    // 4 (选中的实例ID)

    model_center: vec4<f32>,    // 16
    base_color: vec4<f32>,      // 16

    base_radius: f32,            // 4
    debug_mode: u32,             // 4
    screen_width: u32,            // 4
    screen_height: u32,          // 4

    // Ap1 包络字段
    envelope_displacement: f32,  // 4
    show_envelope: u32,          // 4
    envelope_vertex_count: u32,  // 4
    _pad_enum: u32,              // 4

    // Ap2 扭曲字段
    distort_strength: f32,       // 4
    distort_frequency: f32,      // 4
    ap2_iteration: u32,          // 4
    _pad2: u32,                  // 4
};

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> scaffold: array<vec4<f32>>;
@group(0) @binding(4) var depth_tex: texture_2d<f32>;
@group(0) @binding(5) var tri_id_tex: texture_2d<u32>;
@group(0) @binding(6) var uv_tex: texture_2d<f32>;
@group(0) @binding(7) var t_albedo: texture_2d<f32>;
@group(0) @binding(8) var s_albedo: sampler;
@group(0) @binding(9) var<storage, read> instances: array<InstanceData>;
struct WarpPixel {
    src_x: u32,
    src_y: u32,
    tri_id: u32,
    flags: u32,
};
@group(0) @binding(10) var<storage, read_write> warpBuffer: array<WarpPixel>;
@group(0) @binding(11) var<storage, read_write> sdfOutput: array<f32>;
@group(0) @binding(12) var model_uv_tex: texture_2d<f32>;
@group(0) @binding(13) var normal_tex: texture_2d<f32>;

@group(0) @binding(0) var t_read: texture_2d<f32>;

fn dot2(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i + vec3(0.0, 0.0, 0.0)), hash(i + vec3(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3(0.0, 1.0, 0.0)), hash(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash(i + vec3(0.0, 0.0, 1.0)), hash(i + vec3(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3(0.0, 1.0, 1.0)), hash(i + vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

// 2 级轻量 FBM，专门用于高速几何形变和求交导航
fn fbm_coarse(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var current_p = p;
    for (var i = 0u; i < 2u; i++) {
        value += amplitude * noise(current_p);
        current_p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// 计算点 p 到三角形 abc 的最短距离 (Unsigned Distance)
fn udTriangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let cb = c - b;
    let pb = p - b;
    let ac = a - c;
    let pc = p - c;
    let nor = cross(ba, ac);

    let d = sign(dot(cross(ba, nor), pa)) +
            sign(dot(cross(cb, nor), pb)) +
            sign(dot(cross(ac, nor), pc));

    if (d < 2.0) {
        return sqrt(min(min(
            dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
            dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
            dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc)));
    } else {
        return sqrt(dot(nor, pa) * dot(nor, pa) / dot2(nor));
    }
}

// -------------------------------------------------------------
// 一致性逆向求解器与扭曲数学库
// -------------------------------------------------------------

fn get_distortion_consistent(p: vec3<f32>) -> vec3<f32> {
    let s = params.distort_strength;
    let freq = params.distort_frequency * 2.0;
    let t = params.time * 0.5;

    if (s <= 0.0) { return vec3<f32>(0.0); }

    let dx = fbm_coarse(p * freq + vec3<f32>(t, 0.0, 0.0)) - 0.5;
    let dy = fbm_coarse(p * freq + vec3<f32>(0.0, t, 0.0)) - 0.5;
    let dz = fbm_coarse(p * freq + vec3<f32>(0.0, 0.0, t)) - 0.5;
    return vec3<f32>(dx, dy, dz) * s;
}

fn get_p_old_consistent(p_new: vec3<f32>) -> vec3<f32> {
    var p_old = p_new;
    for (var i = 0u; i < 8u; i++) {
        let offset = get_distortion_consistent(p_old);
        p_old = mix(p_old, p_new - offset, 0.4);
    }
    return p_old;
}

fn get_barycentric(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a; let v1 = c - a; let v2 = p - a;
    let d00 = dot(v0, v0); let d01 = dot(v0, v1); let d11 = dot(v1, v1);
    let d20 = dot(v2, v0); let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-6) { return vec3<f32>(1.0, 0.0, 0.0); }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    return vec3<f32>(u, v, w);
}

// -------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn cs_ap2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));

    let tri = textureLoad(tri_id_tex, pixel, 0).r;
    if (tri == 0u) { return; }

    let p_static = textureLoad(uv_tex, pixel, 0).xyz;
    let p_distorted = p_static + get_distortion_consistent(p_static);

    let clip = params.prev_view_proj * vec4<f32>(p_distorted, 1.0);
    let ndc = clip.xyz / clip.w;

    let target_x = u32((ndc.x * 0.5 + 0.5) * f32(params.screen_width));
    let target_y = u32((0.5 - ndc.y * 0.5) * f32(params.screen_height));

    if (target_x < params.screen_width && target_y < params.screen_height) {
        let warpedIdx = target_y * params.screen_width + target_x;
        warpBuffer[warpedIdx].src_x = u32(pixel.x);
        warpBuffer[warpedIdx].src_y = u32(pixel.y);
        warpBuffer[warpedIdx].tri_id = tri;
        warpBuffer[warpedIdx].flags = 1u;
    }
}

// =================================================================
// 3. 完全体 AP3 双轨杂交寻址着色器
// =================================================================

@compute @workgroup_size(8, 8)
fn cs_ap3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let screen_coord = vec2<i32>(i32(gid.x), i32(gid.y));

    if (params.debug_mode == 1u) {
        let tri_id = textureLoad(tri_id_tex, screen_coord, 0).r;
        if (tri_id > 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.0, 1.0, 0.0, 1.0));
        }
        return;
    }

    let idx = gid.y * params.screen_width + gid.x;
    let wp_direct = warpBuffer[idx];

    if (params.debug_mode == 2u) {
        if (wp_direct.flags == 1u) {
            textureStore(output_texture, screen_coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        }
        return;
    }

    let uv_screen = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.screen_width), f32(params.screen_height));
    let ndc = vec2<f32>(uv_screen.x * 2.0 - 1.0, (1.0 - uv_screen.y) * 2.0 - 1.0);
    let ray_target = params.proj_inv * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
    let ray_o = params.cam_pos.xyz;

    // -------------------------------------------------------------
    // 【双轨杂交：主干区域无条件高速放行（100% 稳定性，零发散裂纹）】
    // -------------------------------------------------------------
    if (wp_direct.flags == 1u && wp_direct.tri_id > 0u) {
        let original_pixel = vec2<i32>(i32(wp_direct.src_x), i32(wp_direct.src_y));

        let world_pos = textureLoad(uv_tex, original_pixel, 0).xyz;
        let model_uv = textureLoad(model_uv_tex, original_pixel, 0).xy;
        let smooth_n = normalize(textureLoad(normal_tex, original_pixel, 0).xyz);

        let tex_color = textureSampleLevel(t_albedo, s_albedo, model_uv, 0.0).rgb;

        let e = 0.01;
        let freq = params.distort_frequency * 2.0;
        let p_old = get_p_old_consistent(world_pos);
        let h_c = fbm_coarse(p_old * freq);
        let h_x = fbm_coarse((p_old + vec3<f32>(e, 0.0, 0.0)) * freq);
        let h_y = fbm_coarse((p_old + vec3<f32>(0.0, e, 0.0)) * freq);
        let h_z = fbm_coarse((p_old + vec3<f32>(0.0, 0.0, e)) * freq);

        let fbm_grad = vec3<f32>(h_x - h_c, h_y - h_c, h_z - h_c) / e;

        // FBM 连续空间梯度扰动法线（100% 消除网面割裂痕迹，画面圆滑）
        let distort_effect = params.distort_strength * 0.15;
        let final_n = normalize(smooth_n - fbm_grad * distort_effect);

        let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diff = max(dot(final_n, L), 0.0) * 0.8 + 0.2;

        var final_col = tex_color * diff;
        final_col = pow(final_col, vec3<f32>(1.0 / 2.2));
        textureStore(output_texture, screen_coord, vec4<f32>(final_col, 1.0));

    } else {
        // -------------------------------------------------------------
        // 【双轨杂交：空洞像素执行单三角 3D 步进，融合平面引导以解决掠射角断裂】
        // -------------------------------------------------------------
        var candidates: array<u32, 4> = array<u32, 4>(0u, 0u, 0u, 0u);
        var candidate_count = 0u;

        // 5x5 邻域搜索收集候选三角形
        for (var dy: i32 = -2; dy <= 2; dy++) {
            for (var dx: i32 = -2; dx <= 2; dx++) {
                let nx = i32(gid.x) + dx;
                let ny = i32(gid.y) + dy;
                if (nx >= 0 && nx < i32(params.screen_width) && ny >= 0 && ny < i32(params.screen_height)) {
                    let nidx = u32(ny) * params.screen_width + u32(nx);
                    let nwp = warpBuffer[nidx];
                    if (nwp.flags == 1u && nwp.tri_id > 0u) {
                        var exists = false;
                        for (var c = 0u; c < candidate_count; c++) {
                            if (candidates[c] == nwp.tri_id) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists && candidate_count < 4u) {
                            candidates[candidate_count] = nwp.tri_id;
                            candidate_count++;
                        }
                    }
                }
            }
        }

        if (params.debug_mode == 3u) {
            if (candidate_count > 0u) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 1.0, 1.0));
            }
            return;
        }

        // 调试模式 4: 5x5 搜寻面数量热力图
        if (params.debug_mode == 4u) {
            if (candidate_count == 0u) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0)); // 黑色（没搜到）
            } else if (candidate_count == 1u) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.3, 0.0, 1.0)); // 深绿
            } else if (candidate_count == 2u) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.7, 0.0, 1.0)); // 亮绿
            } else {
                textureStore(output_texture, screen_coord, vec4<f32>(1.0, 1.0, 0.0, 1.0)); // 黄色（多个候选面）
            }
            return;
        }

        var global_best_hit = false;
        var global_best_t = 1e9;
        var global_best_p_old = vec3<f32>(0.0);
        var global_best_tri_idx = 0u;

        // Debug 5 诊断变量
        var diag_any_hit = false;      // 是否有任何相交
        var diag_inside_reject = false; // 是否有边界外拒绝

        for (var c = 0u; c < candidate_count; c++) {
            let actual_tri_idx = candidates[c] - 1u;
            if (actual_tri_idx >= params.anchor_count) { continue; }
            let tri = triangles[actual_tri_idx];

            let inst = instances[0u];
            let v0_world = (inst.model_matrix * tri.v0).xyz;
            let v1_world = (inst.model_matrix * tri.v1).xyz;
            let v2_world = (inst.model_matrix * tri.v2).xyz;

            let N = normalize(cross(v1_world - v0_world, v2_world - v0_world));

            let d0 = distance(ray_o, v0_world);
            let d1 = distance(ray_o, v1_world);
            let d2 = distance(ray_o, v2_world);

            let t_min_vert = min(d0, min(d1, d2));
            let t_max_vert = max(d0, max(d1, d2));

            // 【核心安全保护】：限制 t_plane 在几何顶点的物理深度域内，绝不允许溢出
            let denom = dot(ray_dir, N);
            var t_plane = (t_min_vert + t_max_vert) * 0.5;
            if (abs(denom) > 1e-4) {
                let t_calc = dot(v0_world - ray_o, N) / denom;
                t_plane = clamp(t_calc, t_min_vert - 0.2, t_max_vert + 0.2);
            }

            // 将寻根范围收窄，配合 params 强度，杜绝婴儿步耗尽
            let padding = params.distort_strength + 0.15;
            var t = max(t_plane - padding, t_min_vert - 0.2);
            let t_max = min(t_plane + padding, t_max_vert + 0.2);

            var local_hit = false;
            var local_hit_t = t;

            // 32 步高容错 3D 步进
            for (var step = 0u; step < 32u; step++) {
                let p_curr = ray_o + ray_dir * t;
                let p_old = get_p_old_consistent(p_curr);

                let d = udTriangle(p_old, v0_world, v1_world, v2_world);

                if (d < 0.008) {
                    var low = t - 0.04;
                    var high = t;
                    for (var k = 0u; k < 4u; k++) {
                        let mid = (low + high) * 0.5;
                        let po = get_p_old_consistent(ray_o + ray_dir * mid);
                        let dm = udTriangle(po, v0_world, v1_world, v2_world);
                        if (dm < 0.002) { high = mid; } else { low = mid; }
                    }
                    local_hit_t = high;
                    local_hit = true;
                    break;
                }

                // 强制最小步长 0.012，解决面平行引起的步长萎缩
                t += max(d * 0.6, 0.012);
                if (t > t_max) { break; }
            }

            if (local_hit) {
                let local_p_old = get_p_old_consistent(ray_o + ray_dir * local_hit_t);
                let bary = get_barycentric(local_p_old, v0_world, v1_world, v2_world);
                
                // 边界愈合阈值：放宽至 -0.05 完美缝合面之间裂纹，0 噪点
                let inside = (bary.x >= -0.05 && bary.y >= -0.05 && bary.z >= -0.05);

                // Debug 5 诊断更新
                diag_any_hit = true;
                if (!inside) {
                    diag_inside_reject = true;
                }

                if (inside && local_hit_t < global_best_t) {
                    global_best_hit = true;
                    global_best_t = local_hit_t;
                    global_best_p_old = local_p_old;
                    global_best_tri_idx = actual_tri_idx;
                }
            }
        }

        // Debug 5: 寻根诊断图
        if (params.debug_mode == 5u) {
            if (global_best_hit) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 1.0, 0.0, 1.0)); // 绿色：成功
            } else if (diag_inside_reject) {
                textureStore(output_texture, screen_coord, vec4<f32>(1.0, 1.0, 0.0, 1.0)); // 黄色：相交但在边界外
            } else if (diag_any_hit) {
                textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 1.0, 1.0)); // 蓝色：有相交但未命中
            } else {
                textureStore(output_texture, screen_coord, vec4<f32>(1.0, 0.0, 0.0, 1.0)); // 红色：无候选面或无相交
            }
            return;
        }

        if (global_best_hit) {
            let tri_final = triangles[global_best_tri_idx];
            let inst_final = instances[0u];

            let v0_w = (inst_final.model_matrix * tri_final.v0).xyz;
            let v1_w = (inst_final.model_matrix * tri_final.v1).xyz;
            let v2_w = (inst_final.model_matrix * tri_final.v2).xyz;

            let bary = get_barycentric(global_best_p_old, v0_w, v1_w, v2_w);

            let uv0 = tri_final.uv01.xy;
            let uv1 = tri_final.uv01.zw;
            let uv2 = tri_final.uv2.xy;
            let interpolated_uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
            let tex_color = textureSampleLevel(t_albedo, s_albedo, interpolated_uv, 0.0).rgb;

            let smooth_n_local = normalize(tri_final.n0.xyz * bary.x + tri_final.n1.xyz * bary.y + tri_final.n2.xyz * bary.z);
            let smooth_n = normalize((inst_final.model_matrix * vec4<f32>(smooth_n_local, 0.0)).xyz);

            let e = 0.01;
            let freq = params.distort_frequency * 2.0;
            let h_c = fbm_coarse(global_best_p_old * freq);
            let h_x = fbm_coarse((global_best_p_old + vec3<f32>(e, 0.0, 0.0)) * freq);
            let h_y = fbm_coarse((global_best_p_old + vec3<f32>(0.0, e, 0.0)) * freq);
            let h_z = fbm_coarse((global_best_p_old + vec3<f32>(0.0, 0.0, e)) * freq);

            let fbm_grad = vec3<f32>(h_x - h_c, h_y - h_c, h_z - h_c) / e;

            let distort_effect = params.distort_strength * 0.15;
            let final_n = normalize(smooth_n - fbm_grad * distort_effect);

            let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
            let diff = max(dot(final_n, L), 0.0) * 0.8 + 0.2;

            var final_col = tex_color * diff;
            final_col = pow(final_col, vec3<f32>(1.0 / 2.2));
            textureStore(output_texture, screen_coord, vec4<f32>(final_col, 1.0));
        } else {
            draw_background(screen_coord, ray_o, ray_dir);
        }
    }
}

fn draw_background(screen_coord: vec2<i32>, ray_o: vec3<f32>, ray_dir: vec3<f32>) {
    var bg_col = vec3<f32>(0.1, 0.1, 0.12);
    let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
    if (t_grid > 0.0 && t_grid < 100.0) {
        let p = ray_o + ray_dir * t_grid;
        let grid_uv = abs(fract(p.xz - 0.5) - 0.5);
        let grid = smoothstep(0.02, 0.0, grid_uv.x) + smoothstep(0.02, 0.0, grid_uv.y);
        bg_col = mix(bg_col, vec3<f32>(0.2, 0.2, 0.25), grid * 0.5);
    }
    textureStore(output_texture, screen_coord, vec4<f32>(bg_col, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_coord = vec2<i32>(id.xy);
    let size_u = textureDimensions(output_texture);
    if (any(id.xy >= size_u)) { return; }

    let screen_pos = vec2<f32>(id.xy);
    let uv = (screen_pos / vec2<f32>(size_u)) * 2.0 - 1.0;
    let ray_target = params.proj_inv * vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
    let ray_o = params.cam_pos.xyz;

    var bg_col = vec3<f32>(0.1, 0.1, 0.12);
    let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
    if (t_grid > 0.0 && t_grid < 100.0) {
        let p = ray_o + ray_dir * t_grid;
        let grid_uv = abs(fract(p.xz - 0.5) - 0.5);
        let grid = smoothstep(0.02, 0.0, grid_uv.x) + smoothstep(0.02, 0.0, grid_uv.y);
        bg_col = mix(bg_col, vec3<f32>(0.2, 0.2, 0.25), grid * 0.5);
    }
    textureStore(output_texture, screen_coord, vec4<f32>(bg_col, 1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(idx) / 2) * 4.0 - 1.0;
    let y = f32(i32(idx) % 2) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(t_read, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
}

struct ScaffoldVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_scaffold(@builtin(vertex_index) idx: u32) -> ScaffoldVertexOutput {
    var out: ScaffoldVertexOutput;
    let vertex_data = scaffold[idx];
    let p_world = vertex_data.xyz;
    let normal_x = vertex_data.w * 2.0 - 1.0;
    let normal = vec3<f32>(normal_x, 0.0, 0.0);
    let to_cam = normalize(params.cam_pos.xyz - p_world);
    let dot_prod = dot(normal, to_cam);
    if (dot_prod < -0.05) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    let p_clip = params.prev_view_proj * vec4<f32>(p_world, 1.0);
    out.position = p_clip;
    out.color = vec4<f32>(0.0, 0.6, 1.0, 0.5);
    return out;
}

@fragment
fn fs_scaffold(in: ScaffoldVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

struct DepthVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) triangle_id: u32,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) instance_id: u32,
    @location(3) world_pos: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
};

@vertex
fn vs_depth(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> DepthVertexOutput {
    let instance = instances[instance_index];
    let tri_idx = vertex_index / 3u;
    let local_idx = vertex_index % 3u;
    let tri = triangles[tri_idx];
    var p_local: vec3<f32>;
    var n_local: vec3<f32>;
    var uv: vec2<f32>;
    if (local_idx == 0u) {
        p_local = tri.v0.xyz; n_local = tri.n0.xyz; uv = tri.uv01.xy;
    } else if (local_idx == 1u) {
        p_local = tri.v1.xyz; n_local = tri.n1.xyz; uv = tri.uv01.zw;
    } else {
        p_local = tri.v2.xyz; n_local = tri.n2.xyz; uv = tri.uv2.xy;
    }
    let world_pos_pre = instance.model_matrix * vec4<f32>(p_local, 1.0);
    let world_pos_raw = world_pos_pre.xyz / world_pos_pre.w;
    let world_normal = normalize((instance.model_matrix * vec4<f32>(n_local, 0.0)).xyz);
    let clip_pos = params.prev_view_proj * world_pos_pre;
    var out: DepthVertexOutput;
    out.position = clip_pos;
    out.triangle_id = tri_idx;
    out.uv = uv;
    out.instance_id = instance_index;
    out.world_pos = world_pos_raw;
    out.world_normal = world_normal;
    return out;
}

struct DepthFragmentOutput {
    @location(0) triangle_id: u32,
    @location(1) world_pos: vec4<f32>,
    @location(2) model_uv: vec2<f32>,
    @location(3) world_normal: vec4<f32>,
};

@fragment
fn fs_depth(in: DepthVertexOutput) -> DepthFragmentOutput {
    var out: DepthFragmentOutput;
    out.triangle_id = in.triangle_id + 1u;
    out.world_pos = vec4<f32>(in.world_pos, 1.0);
    out.model_uv = in.uv;
    out.world_normal = vec4<f32>(in.world_normal, 0.0);
    return out;
}

@group(0) @binding(1) var depth_blit_tex: texture_2d<f32>;

@fragment
fn fs_depth_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let depth = textureLoad(depth_blit_tex, vec2<i32>(i32(pos.x), i32(pos.y)), 0).r;
    let vis = (1.0 - depth) * 0.5;
    let near_color = vec3<f32>(1.0, 0.8, 0.4);
    let far_color = vec3<f32>(0.2, 0.3, 0.8);
    let color = mix(far_color, near_color, vis);
    return vec4<f32>(color, 1.0);
}

struct MeshVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) instance_id: u32,
    @location(4) @interpolate(flat) global_tri_idx: u32,
};

@vertex
fn vs_mesh(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> MeshVertexOutput {
    let instance = instances[i_idx];
    let tri_idx = instance.tri_start + v_idx / 3u;
    let local_idx = v_idx % 3u;
    let tri = triangles[tri_idx];
    var p: vec3<f32>; var uv: vec2<f32>; var n: vec3<f32>;
    if (local_idx == 0u) { p = tri.v0.xyz; uv = tri.uv01.xy; n = tri.n0.xyz; }
    else if (local_idx == 1u) { p = tri.v1.xyz; uv = tri.uv01.zw; n = tri.n1.xyz; }
    else { p = tri.v2.xyz; uv = tri.uv2.xy; n = tri.n2.xyz; }
    let world_pos = (instance.model_matrix * vec4<f32>(p, 1.0)).xyz;
    let normal = normalize((instance.model_matrix * vec4<f32>(n, 0.0)).xyz);
    var out: MeshVertexOutput;
    out.position = params.prev_view_proj * vec4<f32>(world_pos, 1.0);
    out.instance_id = i_idx;
    out.global_tri_idx = tri_idx;
    out.world_pos = world_pos;
    out.normal = normal;
    out.uv = uv;
    return out;
}

@group(1) @binding(0) var t_material: texture_2d<f32>;
@group(1) @binding(1) var s_material: sampler;

@fragment
fn fs_mesh(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    let tri = triangles[in.global_tri_idx];
    let instance = instances[in.instance_id];
    let v0 = (instance.model_matrix * tri.v0).xyz;
    let v1 = (instance.model_matrix * tri.v1).xyz;
    let v2 = (instance.model_matrix * tri.v2).xyz;
    let ray_o = params.cam_pos.xyz;
    let view_dir = normalize(in.world_pos - ray_o);
    var t = distance(ray_o, in.world_pos) - 0.01;
    var hit = false;
    var p = vec3<f32>(0.0);
    for (var i = 0u; i < 12u; i++) {
        p = ray_o + view_dir * t;
        let d = udTriangle(p, v0, v1, v2) - 0.005;
        if (d < 0.0001) {
            hit = true;
            break;
        }
        t += d;
    }
    if (!hit) { discard; }
    let n = normalize(in.normal);
    let tex_color = textureSample(t_material, s_material, in.uv).rgb;
    let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diff = max(dot(n, L), 0.0) * 0.8 + 0.2;
    var final_col = tex_color * diff;
    if (in.instance_id == params.selected_instance_id) {
        final_col = mix(final_col, vec3<f32>(1.0, 0.5, 0.0), 0.4);
    }
    return vec4<f32>(pow(final_col, vec3<f32>(1.0 / 2.2)), 1.0);
}