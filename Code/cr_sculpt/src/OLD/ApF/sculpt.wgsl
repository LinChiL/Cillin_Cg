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

    _pad: vec4<u32>,            // 16 (对齐填充)
};

// 1. Compute 阶段使用的声明 (Group 0)
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

// 2. Render 阶段使用的声明 (注意：我们让它也用 Group 0，因为它们在不同的 Pass 运行)
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

fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    let shift = vec3(100.0);
    var p_mut = p;
    for (var i = 0; i < 6; i = i + 1) {
        v = v + a * noise(p_mut);
        p_mut = p_mut * 2.0 + shift;
        a = a * 0.5;
    }
    return v;
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

// 使用 Triangle ID 直接计算完美几何法线
fn get_geometry_normal(tri_idx: u32) -> vec3<f32> {
    if (tri_idx >= params.anchor_count) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    let tri = triangles[tri_idx];
    let e1 = tri.v1.xyz - tri.v0.xyz;
    let e2 = tri.v2.xyz - tri.v0.xyz;
    
    return normalize(cross(e1, e2));
}

@compute @workgroup_size(8, 8)
fn cs_ap2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));

    let tri = textureLoad(tri_id_tex, pixel, 0).r;
    if (tri == 0u) { return; }

    // 【关键】：直接从 uv_tex 读取稳定的世界坐标
    let p_static = textureLoad(uv_tex, pixel, 0).xyz;

    // 从 normal_tex 读取世界空间法线
    let normal = textureLoad(normal_tex, pixel, 0).xyz;

    // FBM 噪声驱动的向内坍缩：位移量由噪声决定
    let roughness = fbm(p_static * 10.0);
    let displacement = roughness * 0.15;
    let p_distorted = p_static - normal * displacement;

    // 将扭曲后的点重新投影到屏幕空间
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

fn compute_sdf(src_x: u32, src_y: u32, tri_id: u32) -> f32 {
    if (tri_id >= params.anchor_count) {
        return 1e9;
    }

    let tri = triangles[tri_id];
    let pixel_pos = vec3<f32>(f32(src_x), f32(src_y), 0.0);

    let edge0 = tri.v1.xyz - tri.v0.xyz;
    let edge1 = tri.v2.xyz - tri.v0.xyz;
    let v0v0 = tri.v0.xyz - pixel_pos;

    let d00 = dot(edge0, edge0);
    let d01 = dot(edge0, edge1);
    let d11 = dot(edge1, edge1);
    let d20 = dot(v0v0, edge0);
    let d21 = dot(v0v0, edge1);

    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-10) {
        return 1e9;
    }

    let inv_denom = 1.0 / denom;
    let alpha = (d11 * d20 - d01 * d21) * inv_denom;
    let beta = (d00 * d21 - d01 * d20) * inv_denom;

    if (alpha >= 0.0 && beta >= 0.0 && alpha + beta < 1.0) {
        let p = tri.v0.xyz + alpha * edge0 + beta * edge1;
        return distance(p, pixel_pos);
    }

    let d1 = dot(edge0, v0v0);
    let d2 = dot(edge1, v0v0);
    let e2 = d00 * d11 - d01 * d01;

    let s = clamp((d11 * d1 - d01 * d2) / e2, 0.0, 1.0);
    let t = clamp((d00 * d2 - d01 * d1) / e2, 0.0, 1.0);
    let s互补 = 1.0 - s;
    let t互补 = 1.0 - t;

    if (s互补 + t互补 > 1.0) {
        return 1e9;
    }

    let p = tri.v0.xyz + s * edge0 + t * edge1;
    return distance(p, pixel_pos);
}

fn find_nearest_warp_pixel(curr_pos: vec2<u32>) -> WarpPixel {
    var nearest: WarpPixel;
    nearest.flags = 0u;
    var min_dist = 100.0;

    for (var dy: i32 = -3; dy <= 3; dy++) {
        for (var dx: i32 = -3; dx <= 3; dx++) {
            let nx = i32(curr_pos.x) + dx;
            let ny = i32(curr_pos.y) + dy;

            if (nx < 0 || nx >= i32(params.screen_width)) { continue; }

            let nidx = u32(ny) * params.screen_width + u32(nx);
            let wp = warpBuffer[nidx];

            if (wp.flags == 1u) {
                let d = length(vec2<f32>(f32(dx), f32(dy)));
                if (d < min_dist) {
                    min_dist = d;
                    nearest = wp;
                }
            }
        }
    }
    return nearest;
}

@compute @workgroup_size(8, 8)
fn cs_ap3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let idx = gid.y * params.screen_width + gid.x;
    let screen_coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // --- 调试模式 1: 直接查看 Ap1 的 ID 图 ---
    if (params.debug_mode == 1u) {
        let tri_id = textureLoad(tri_id_tex, screen_coord, 0).r;
        if (tri_id > 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.0, 1.0, 0.0, 1.0)); // 绿图
        }
        return;
    }

    // --- 调试模式 2: 查看 Ap2 投射过来的点（原始 9 个像素） ---
    let wp_direct = warpBuffer[idx];
    if (params.debug_mode == 2u) {
        if (wp_direct.flags == 1u) {
            textureStore(output_texture, screen_coord, vec4<f32>(1.0, 0.0, 0.0, 1.0)); // 红点
        }
        return;
    }

    // --- 核心逻辑：执行 9 -> 14 的补洞 ---
    var final_wp: WarpPixel;
    var found = false;
    var is_gap_filled = false;

    if (wp_direct.flags == 1u) {
        final_wp = wp_direct;
        found = true;
    } else {
        // 搜索周围找最近的有效像素
        let search_radius: i32 = 2;
        var min_dist = 100.0;

        for (var i: i32 = -search_radius; i <= search_radius; i++) {
            for (var j: i32 = -search_radius; j <= search_radius; j++) {
                let nx = i32(gid.x) + i;
                let ny = i32(gid.y) + j;
                if (nx < 0 || nx >= i32(params.screen_width)) { continue; }

                let nidx = u32(ny) * params.screen_width + u32(nx);
                let nwp = warpBuffer[nidx];

                if (nwp.flags == 1u) {
                    let d = length(vec2<f32>(f32(i), f32(j)));
                    if (d < min_dist) {
                        min_dist = d;
                        final_wp = nwp;
                        found = true;
                        is_gap_filled = true;
                    }
                }
            }
        }
    }

    // --- 调试模式 3: 查看哪些是补出来的洞 ---
    if (params.debug_mode == 3u) {
        if (found && wp_direct.flags == 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 1.0, 1.0)); // 蓝点（补出来的）
        }
        return;
    }

    // --- 最终着色：只计算一次 ---
    if (found) {
        var world_pos: vec3<f32>;
        let original_pixel = vec2<i32>(i32(final_wp.src_x), i32(final_wp.src_y));

        if (is_gap_filled && final_wp.tri_id > 0u) {
            // --- 【几何重构模式】：补洞像素走 SDF 步进 ---
            // 借用邻居的三角形 ID
            let tri = triangles[final_wp.tri_id - 1u];

            // 生成当前像素自己的射线
            let uv_screen = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.screen_width), f32(params.screen_height));
            let ndc = vec2<f32>(uv_screen.x * 2.0 - 1.0, (1.0 - uv_screen.y) * 2.0 - 1.0);
            let ray_target = params.proj_inv * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
            let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
            let ray_o = params.cam_pos.xyz;

            // SDF 步进求交（只跑 12 次，局部精度足够）
            var t = 0.0;
            var p = ray_o;
            var hit = false;
            
            for(var i=0u; i<12u; i++) {
                p = ray_o + ray_dir * t;
                let d = udTriangle(p, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz) - 0.005;
                if (d < 0.0001) { hit = true; break; }
                t += d;
            }

            if (hit) {
                world_pos = p;
            } else {
                // 求交失败， fallback 到邻居的 world_pos
                world_pos = textureLoad(uv_tex, original_pixel, 0).xyz;
            }
        } else {
            // --- 【直接读取模式】：原始 9 个像素直接读取 ---
            world_pos = textureLoad(uv_tex, original_pixel, 0).xyz;
        }

        let model_uv = textureLoad(model_uv_tex, original_pixel, 0).xy;
        let tex_color = textureSampleLevel(t_albedo, s_albedo, model_uv, 0.0).rgb;

        let macro_n = normalize(textureLoad(normal_tex, original_pixel, 0).xyz);

        let eps = 0.002;
        let h_center = fbm(world_pos * 10.0);
        let h_x = fbm((world_pos + vec3(eps, 0.0, 0.0)) * 10.0);
        let h_y = fbm((world_pos + vec3(0.0, eps, 0.0)) * 10.0);
        let h_z = fbm((world_pos + vec3(0.0, 0.0, eps)) * 10.0);
        let grad = vec3(h_x - h_center, h_y - h_center, h_z - h_center) / eps;
        let final_n = normalize(macro_n - grad * 0.3);

        let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diff = max(dot(final_n, L), 0.0) * 0.8 + 0.2;

        var final_col = tex_color * diff;
        final_col = pow(final_col, vec3<f32>(1.0 / 2.2));
        textureStore(output_texture, screen_coord, vec4<f32>(final_col, 1.0));
    }
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_coord = vec2<i32>(id.xy);
    let size_u = textureDimensions(output_texture);
    if (any(id.xy >= size_u)) { return; }

    // 【修复】：背景现在无条件全屏绘制，不再依赖 tri_id
    // 这样当模型扭曲走后，原位置立即被背景填充，不会留下"三不管地带"
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

// --- 展示管线 (Blit) ---
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(idx) / 2) * 4.0 - 1.0;
    let y = f32(i32(idx) % 2) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // 关键：现在 t_read 是普通 texture_2d，使用 textureLoad 是合法的
    return textureLoad(t_read, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
}

// --- 点云渲染管线 ---
struct ScaffoldVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_scaffold(@builtin(vertex_index) idx: u32) -> ScaffoldVertexOutput {
    var out: ScaffoldVertexOutput;
    
    // 直接从 Storage Buffer 拿顶点（格式：[pos.x, pos.y, pos.z, packed_normal]）
    let vertex_data = scaffold[idx];
    let p_world = vertex_data.xyz;
    
    // 从 w 分量解码法线 x 分量（存储时从 [-1,1] 映射到 [0,1]）
    let normal_x = vertex_data.w * 2.0 - 1.0;
    let normal = vec3<f32>(normal_x, 0.0, 0.0); // 简化：只使用法线 x 分量
    
    // 计算点到相机的向量
    let to_cam = normalize(params.cam_pos.xyz - p_world);
    
    // 背面剔除判断（使用简化的法线）
    let dot_prod = dot(normal, to_cam);
    
    // 如果点朝向背面，直接把它扔到裁剪空间外（不渲染）
    if (dot_prod < -0.05) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // 正常的投影逻辑
    let p_clip = params.prev_view_proj * vec4<f32>(p_world, 1.0);
    
    out.position = p_clip;
    out.color = vec4<f32>(0.0, 0.6, 1.0, 0.5); // 蓝紫色半透明点
    return out;
}

@fragment
fn fs_scaffold(in: ScaffoldVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

// ====================== 轻量深度图管线 ======================
struct DepthVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) triangle_id: u32,
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
    out.triangle_id = in.instance_id + 1u;
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
// ============================================================

// ====================== 网格渲染管线 ======================
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
    
    // 不在 VS 阶段扭曲！扭曲将在 Ap2 阶段从稳定的世界坐标出发
    
    // 使用平滑法线
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

// 材质绑定组（BindGroup 1）
@group(1) @binding(0) var t_material: texture_2d<f32>;
@group(1) @binding(1) var s_material: sampler;

@fragment
fn fs_mesh(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    // 1. 使用平滑法线进行射线步进
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

    // 2. 使用平滑法线进行光照
    let n = normalize(in.normal);

    // 3. 材质与渲染
    let tex_color = textureSample(t_material, s_material, in.uv).rgb;
    let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diff = max(dot(n, L), 0.0) * 0.8 + 0.2;
    
    var final_col = tex_color * diff;
    if (in.instance_id == params.selected_instance_id) {
        final_col = mix(final_col, vec3<f32>(1.0, 0.5, 0.0), 0.4);
    }

    return vec4<f32>(pow(final_col, vec3<f32>(1.0 / 2.2)), 1.0);
}

// ============================================================