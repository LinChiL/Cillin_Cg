struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>, // xyz = 视觉法线0
    n1: vec4<f32>, // xyz = 视觉法线1
    n2: vec4<f32>, // xyz = 视觉法线2
    warp_n0: vec4<f32>, // xyz = 扭曲法线0
    warp_n1: vec4<f32>, // xyz = 扭曲法线1
    warp_n2: vec4<f32>, // xyz = 扭曲法线2
    uv01: vec4<f32>, // [u0, v0, u1, v1]
    uv2: vec4<f32>,  // [u2, v2, 0.0, 0.0] - 保持对齐
};

struct InstanceData {
    model_matrix: mat4x4<f32>,
    model_id: u32,
    instance_id: u32,
    tri_start: u32,
    _pad_inner: u32,
    material_color: vec4<f32>,
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

    envelope_displacement: f32,
    show_envelope: u32,
    envelope_vertex_count: u32,
    _pad: u32,

    distort_strength: f32,
    distort_frequency: f32,
    ap2_iteration: u32,
    instance_count: u32,
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
    flags: atomic<u32>,
    barycentric: vec4<f32>,
};
@group(0) @binding(10) var<storage, read_write> warpBuffer: array<WarpPixel>;
@group(0) @binding(11) var<storage, read_write> sdfOutput: array<f32>;
@group(0) @binding(12) var model_uv_tex: texture_2d<f32>;
@group(0) @binding(13) var normal_tex: texture_2d<f32>;
@group(0) @binding(15) var warp_normal_tex: texture_2d<f32>;

// 2. Render 阶段使用的声明
@group(0) @binding(14) var t_read: texture_2d<f32>;

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
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)), hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    let shift = vec3<f32>(100.0);
    var p_mut = p;
    for (var i = 0; i < 6; i = i + 1) {
        v = v + a * noise(p_mut);
        p_mut = p_mut * 2.0 + shift;
        a = a * 0.5;
    }
    return v;
}

fn warp_displacement(world_pos: vec3<f32>) -> f32 {
    let frequency = max(params.distort_frequency, 0.001);
    let roughness = fbm(world_pos * 10.0 * frequency);
    return roughness * 0.5 * params.distort_strength;
}

fn warp_world_pos(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    return world_pos - normalize(world_normal) * warp_displacement(world_pos);
}

fn sample_warp_normal(pixel: vec2<i32>) -> vec3<f32> {
    let warp_normal = textureLoad(warp_normal_tex, pixel, 0).xyz;
    return select(normalize(textureLoad(normal_tex, pixel, 0).xyz), normalize(warp_normal), dot(warp_normal, warp_normal) > 0.0001);
}

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

fn project_warped_pixel(pixel: vec2<i32>) -> vec4<f32> {
    let p_static = textureLoad(uv_tex, pixel, 0).xyz;
    let normal = sample_warp_normal(pixel);
    let p_distorted = warp_world_pos(p_static, normal);
    let clip = params.prev_view_proj * vec4<f32>(p_distorted, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>(
        (ndc.x * 0.5 + 0.5) * f32(params.screen_width),
        (0.5 - ndc.y * 0.5) * f32(params.screen_height),
        ndc.z,
        clip.w,
    );
}

fn write_warp_pixel(dest: vec2<i32>, source: vec2<i32>, tri: u32, z_ndc: f32, barycentric: vec3<f32>) {
    if (dest.x < 0 || dest.x >= i32(params.screen_width) || dest.y < 0 || dest.y >= i32(params.screen_height)) {
        return;
    }

    let warped_idx = u32(dest.y) * params.screen_width + u32(dest.x);
    let packed_depth = u32(clamp((z_ndc + 1.0) * 0.5, 0.0, 1.0) * 4294967295.0);

    // 【核心优化】先读后写：读操作不占独占总线，直接把被遮挡的样本在门口拦下
    let fast_check = atomicLoad(&warpBuffer[warped_idx].flags);
    if (fast_check != 0u && packed_depth >= fast_check) {
        return; // 白嫖失败，直接退出，不进入 CAS loop
    }

    var existing_depth = fast_check;
    loop {
        if (existing_depth != 0u && packed_depth >= existing_depth) {
            break;
        }
        let result = atomicCompareExchangeWeak(&warpBuffer[warped_idx].flags, existing_depth, packed_depth);
        if (result.exchanged) {
            warpBuffer[warped_idx].src_x = u32(source.x);
            warpBuffer[warped_idx].src_y = u32(source.y);
            warpBuffer[warped_idx].tri_id = tri;
            warpBuffer[warped_idx].barycentric = vec4<f32>(barycentric, 0.0);
            break;
        }
        existing_depth = result.old_value;
    }
}

@compute @workgroup_size(8, 8)
fn cs_ap2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));

    let tri = textureLoad(tri_id_tex, pixel, 0).r;
    if (tri == 0u) { return; }

    let world_pos = textureLoad(uv_tex, pixel, 0).xyz;
    let normal = normalize(textureLoad(normal_tex, pixel, 0).xyz);
    let model_uv = textureLoad(model_uv_tex, pixel, 0).xy;
    let center = project_warped_pixel(pixel);
    if (center.w <= 0.0) { return; }

    let base_target = vec2<i32>(i32(floor(center.x)), i32(floor(center.y)));
    let radius = 2i;
    for (var oy = -radius; oy <= radius; oy = oy + 1) {
        for (var ox = -radius; ox <= radius; ox = ox + 1) {
            let offset = vec2<f32>(f32(ox), f32(oy));
            if (dot(offset, offset) <= f32(radius * radius) + 0.25) {
                write_warp_pixel(base_target + vec2<i32>(ox, oy), pixel, tri, center.z, vec3<f32>(1.0, 0.0, 0.0));
            }
        }
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
    
    let s_comp = 1.0 - s;
    let t_comp = 1.0 - t;

    if (s_comp + t_comp > 1.0) {
        return 1e9;
    }

    let p = tri.v0.xyz + s * edge0 + t * edge1;
    return distance(p, pixel_pos);
}

fn get_search_dir(d: u32) -> vec2<i32> {
    switch (d) {
        case 0u:  { return vec2<i32>(1, 0); }
        case 1u:  { return vec2<i32>(-1, 0); }
        case 2u:  { return vec2<i32>(0, 1); }
        case 3u:  { return vec2<i32>(0, -1); }
        case 4u:  { return vec2<i32>(1, 1); }
        case 5u:  { return vec2<i32>(-1, -1); }
        case 6u:  { return vec2<i32>(1, -1); }
        default:  { return vec2<i32>(-1, 1); } 
    }
}

fn get_search_step(s: u32) -> i32 {
    switch (s) {
        case 0u:  { return 1; }
        case 1u:  { return 2; }
        case 2u:  { return 4; }
        case 3u:  { return 8; }
        case 4u:  { return 12; }
        default:  { return 16; } 
    }
}


@compute @workgroup_size(8, 8)
fn cs_ap3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let screen_coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // --- 调试模式 1: 直接查看 Ap1 的 ID 图 ---
    if (params.debug_mode == 1u) {
        let tri_id = textureLoad(tri_id_tex, screen_coord, 0).r;
        if (tri_id > 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.0, 1.0, 0.0, 1.0)); 
        }
        return;
    }

    let warp_idx = gid.y * params.screen_width + gid.x;
    let flags = atomicLoad(&warpBuffer[warp_idx].flags);

    // --- 调试模式 2: 查看 AP2 直接投射点 ---
    if (params.debug_mode == 2u) {
        if (flags > 0u) { 
            textureStore(output_texture, screen_coord, vec4<f32>(1.0, 0.0, 0.0, 1.0)); 
        }
        return;
    }

    // --- 调试模式 3: 查看 AP2 未覆盖缺口 ---
    if (params.debug_mode == 3u) {
        if (flags == 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 1.0, 1.0));
        }
        return;
    }

    // --- 最终着色 ---
    if (flags > 0u) {
        let tri_id = warpBuffer[warp_idx].tri_id;
        let instance_id = (tri_id >> 24u) - 1u;
        let tri_idx = (tri_id & 0x00FFFFFFu) - 1u;
        let instance = instances[instance_id];
        let tri = triangles[tri_idx];
        let b = warpBuffer[warp_idx].barycentric.xyz;

        let local_pos = tri.v0.xyz * b.x + tri.v1.xyz * b.y + tri.v2.xyz * b.z;
        let world_pos_pre = instance.model_matrix * vec4<f32>(local_pos, 1.0);
        let world_pos = world_pos_pre.xyz / world_pos_pre.w;

        let model_uv = tri.uv01.xy * b.x + tri.uv01.zw * b.y + tri.uv2.xy * b.z;
        let tex_color = textureSampleLevel(t_albedo, s_albedo, model_uv, 0.0).rgb;

        let local_n = normalize(tri.n0.xyz * b.x + tri.n1.xyz * b.y + tri.n2.xyz * b.z);
        let macro_n = normalize((instance.model_matrix * vec4<f32>(local_n, 0.0)).xyz);

        let eps = 0.002;
        let freq = max(params.distort_frequency, 0.001);
        let h_center = fbm(world_pos * 10.0 * freq);
        let h_x = fbm((world_pos + vec3<f32>(eps, 0.0, 0.0)) * 10.0 * freq);
        let h_y = fbm((world_pos + vec3<f32>(0.0, eps, 0.0)) * 10.0 * freq);
        let h_z = fbm((world_pos + vec3<f32>(0.0, 0.0, eps)) * 10.0 * freq);
        let grad = vec3<f32>(h_x - h_center, h_y - h_center, h_z - h_center) / eps;
        let final_n = normalize(macro_n - grad * params.distort_strength);

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
    @location(0) triangle_id: u32,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) instance_id: u32,
    @location(3) world_pos: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) warp_normal: vec3<f32>,
    @location(6) barycentric: vec3<f32>,
};

fn warped_screen_clip(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    let warped = warp_world_pos(world_pos, world_normal);
    let clip = params.prev_view_proj * vec4<f32>(warped, 1.0);

    // 【核心修复】防近裁剪面奇点：若 clip.w 接近 0 或为负（跑到相机背后），强行限制为极小正数
    let safe_w = max(clip.w, 0.00001);
    let ndc = clip.xyz / safe_w;

    return vec4<f32>(
        (ndc.x * 0.5 + 0.5) * f32(params.screen_width),
        (0.5 - ndc.y * 0.5) * f32(params.screen_height),
        ndc.z,
        clip.w, // 原样返回真实 w 供后续判断
    );
}

fn warped_screen_pos(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec2<f32> {
    return warped_screen_clip(world_pos, world_normal).xy;
}

fn unwrap_local_uv(tri: Triangle, local_pos: vec3<f32>) -> vec2<f32> {
    let e0 = tri.v1.xyz - tri.v0.xyz;
    let e1 = tri.v2.xyz - tri.v0.xyz;
    let face_n = abs(normalize(cross(e0, e1)));

    var p0: vec2<f32>;
    var p1: vec2<f32>;
    var p2: vec2<f32>;
    var p: vec2<f32>;

    if (face_n.x > face_n.y && face_n.x > face_n.z) {
        p0 = tri.v0.yz; p1 = tri.v1.yz; p2 = tri.v2.yz; p = local_pos.yz;
    } else if (face_n.y > face_n.z) {
        p0 = tri.v0.xz; p1 = tri.v1.xz; p2 = tri.v2.xz; p = local_pos.xz;
    } else {
        p0 = tri.v0.xy; p1 = tri.v1.xy; p2 = tri.v2.xy; p = local_pos.xy;
    }

    let min_p = min(min(p0, p1), p2);
    let max_p = max(max(p0, p1), p2);
    let extent = max(max_p - min_p, vec2<f32>(0.0001));
    return clamp((p - min_p) / extent, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn unwrap_triangle_pos(tri: Triangle, local_pos: vec3<f32>, instance: InstanceData) -> vec4<f32> {
    let w0 = (instance.model_matrix * tri.v0).xyz;
    let w1 = (instance.model_matrix * tri.v1).xyz;
    let w2 = (instance.model_matrix * tri.v2).xyz;
    let n0 = normalize((instance.model_matrix * vec4<f32>(tri.warp_n0.xyz, 0.0)).xyz);
    let n1 = normalize((instance.model_matrix * vec4<f32>(tri.warp_n1.xyz, 0.0)).xyz);
    let n2 = normalize((instance.model_matrix * vec4<f32>(tri.warp_n2.xyz, 0.0)).xyz);

    let s0 = warped_screen_pos(w0, n0);
    let s1 = warped_screen_pos(w1, n1);
    let s2 = warped_screen_pos(w2, n2);
    let min_s = floor(max(min(min(s0, s1), s2) - vec2<f32>(2.0), vec2<f32>(0.0)));
    let max_s = ceil(min(max(max(s0, s1), s2) + vec2<f32>(2.0), vec2<f32>(f32(params.screen_width - 1u), f32(params.screen_height - 1u))));
    let bbox_size = max(max_s - min_s, vec2<f32>(1.0));
    let local_uv = unwrap_local_uv(tri, local_pos);
    let screen_pos = min_s + local_uv * bbox_size;
    let ndc = vec2<f32>(
        (screen_pos.x / f32(params.screen_width)) * 2.0 - 1.0,
        1.0 - (screen_pos.y / f32(params.screen_height)) * 2.0,
    );
    return vec4<f32>(ndc, 0.0, 1.0);
}

@vertex
fn vs_depth(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> DepthVertexOutput {
    let instance = instances[instance_index];
    let local_tri_idx = vertex_index / 3u;
    let tri_idx = instance.tri_start + local_tri_idx;
    let local_idx = vertex_index % 3u;

    let tri = triangles[tri_idx];
    var p_local: vec3<f32>;
    var n_local: vec3<f32>;
    var warp_n_local: vec3<f32>;
    var uv: vec2<f32>;

    var barycentric: vec3<f32>;
    if (local_idx == 0u) {
        p_local = tri.v0.xyz; n_local = tri.n0.xyz; warp_n_local = tri.warp_n0.xyz; uv = tri.uv01.xy; barycentric = vec3<f32>(1.0, 0.0, 0.0);
    } else if (local_idx == 1u) {
        p_local = tri.v1.xyz; n_local = tri.n1.xyz; warp_n_local = tri.warp_n1.xyz; uv = tri.uv01.zw; barycentric = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        p_local = tri.v2.xyz; n_local = tri.n2.xyz; warp_n_local = tri.warp_n2.xyz; uv = tri.uv2.xy; barycentric = vec3<f32>(0.0, 0.0, 1.0);
    }

    let world_pos_pre = instance.model_matrix * vec4<f32>(p_local, 1.0);
    let world_pos_raw = world_pos_pre.xyz / world_pos_pre.w;
    let world_normal = normalize((instance.model_matrix * vec4<f32>(n_local, 0.0)).xyz);
    let warp_normal = normalize((instance.model_matrix * vec4<f32>(warp_n_local, 0.0)).xyz);
    var out: DepthVertexOutput;
    out.position = unwrap_triangle_pos(tri, p_local, instance);
    out.triangle_id = tri_idx;
    out.uv = uv;
    out.instance_id = instance_index;
    out.world_pos = world_pos_raw;
    out.world_normal = world_normal;
    out.warp_normal = warp_normal;
    out.barycentric = barycentric;
    return out;
}

struct DepthFragmentOutput {
    @location(0) triangle_id: u32,
    @location(1) world_pos: vec4<f32>,
    @location(2) model_uv: vec2<f32>,
    @location(3) world_normal: vec4<f32>,
    @location(4) warp_normal: vec4<f32>,
};

@fragment
fn fs_depth(in: DepthVertexOutput) -> DepthFragmentOutput {
    let packed_tri = ((in.instance_id + 1u) << 24u) | (in.triangle_id + 1u);
    let projected = warped_screen_clip(in.world_pos, in.warp_normal);
    if (projected.w > 0.0) {
        let base_target = vec2<i32>(i32(floor(projected.x)), i32(floor(projected.y)));
        let source = vec2<i32>(i32(floor(in.position.x)), i32(floor(in.position.y)));
        let view_dir = normalize(params.cam_pos.xyz - in.world_pos);
        let facing = abs(dot(normalize(in.world_normal), view_dir));
        let grazing_stability = smoothstep(0.08, 0.35, facing);
        let radius = select(1i, 2i, grazing_stability > 0.5);
        for (var oy = -radius; oy <= radius; oy = oy + 1) {
            for (var ox = -radius; ox <= radius; ox = ox + 1) {
                let offset = vec2<f32>(f32(ox), f32(oy));
                if (dot(offset, offset) <= f32(radius * radius) + 0.25) {
                    write_warp_pixel(base_target + vec2<i32>(ox, oy), source, packed_tri, projected.z, in.barycentric);
                }
            }
        }

        let raw_proj_dx = dpdx(projected);
        let raw_proj_dy = dpdy(projected);
        let raw_bary_dx = dpdx(in.barycentric);
        let raw_bary_dy = dpdy(in.barycentric);

        let max_axis = mix(3.0, 12.0, grazing_stability);
        let dx_len = length(raw_proj_dx.xy);
        let dy_len = length(raw_proj_dy.xy);
        let dx_scale = min(1.0, max_axis / max(dx_len, 0.0001));
        let dy_scale = min(1.0, max_axis / max(dy_len, 0.0001));
        let proj_dx = raw_proj_dx * dx_scale;
        let proj_dy = raw_proj_dy * dy_scale;
        let bary_dx = raw_bary_dx * dx_scale;
        let bary_dy = raw_bary_dy * dy_scale;

        let footprint_area = abs(proj_dx.x * proj_dy.y - proj_dx.y * proj_dy.x);
        let area_scale = select(1.0, 0.5, footprint_area < 0.25 || footprint_area > 96.0);
        let stable_proj_dx = proj_dx * area_scale;
        let stable_proj_dy = proj_dy * area_scale;
        let stable_bary_dx = bary_dx * area_scale;
        let stable_bary_dy = bary_dy * area_scale;

        let x_steps = min(max(i32(ceil(length(stable_proj_dx.xy))), 1), 12);
        let y_steps = min(max(i32(ceil(length(stable_proj_dy.xy))), 1), 12);
        for (var sy = 0i; sy <= y_steps; sy = sy + 1) {
            let v = f32(sy) / f32(y_steps) - 0.5;
            for (var sx = 0i; sx <= x_steps; sx = sx + 1) {
                let u = f32(sx) / f32(x_steps) - 0.5;
                let p = projected + stable_proj_dx * u + stable_proj_dy * v;
                let b = in.barycentric + stable_bary_dx * u + stable_bary_dy * v;
                write_warp_pixel(vec2<i32>(i32(floor(p.x)), i32(floor(p.y))), source, packed_tri, p.z, b);
            }
        }
    }

    var out: DepthFragmentOutput;
    out.triangle_id = packed_tri;
    out.world_pos = vec4<f32>(in.world_pos, 1.0);
    out.model_uv = in.uv;
    out.world_normal = vec4<f32>(in.world_normal, 0.0);
    out.warp_normal = vec4<f32>(in.warp_normal, 0.0);
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
    let tri_idx = v_idx / 3u;
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

// 材质绑定组（BindGroup 1）
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
        if (d < 0.0001) { hit = true; break; }
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