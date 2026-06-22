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
    tri_count: u32,
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
@group(0) @binding(16) var pixel_shadow_tex: texture_2d<f32>;
@group(0) @binding(17) var pixel_shadow_out: texture_storage_2d<rgba8unorm, write>;

// 太阳空间网格分箱（阴影加速）
const GRID_RES = 1024u;
const GRID_HALF_SIZE = 1350.0;
const MAX_SHADOW_LIST_STEPS = 4096u;

struct GridNode {
    next: u32,
    packed_tri_id: u32,
    cell_idx: u32,
    _pad: u32,
};
@group(0) @binding(18) var<storage, read_write> grid_head: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> grid_nodes: array<GridNode>;
@group(0) @binding(20) var<storage, read_write> global_counter: atomic<u32>;

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

fn fbm_lod(p: vec3<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    let shift = vec3<f32>(100.0);
    var p_mut = p;
    for (var i = 0; i < 6; i = i + 1) {
        if (i >= octaves) { break; }
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

fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let horizon = pow(1.0 - max(ray_dir.y, 0.0), 2.0);
    let zenith = vec3<f32>(0.12, 0.38, 0.85);
    let horizon_col = vec3<f32>(0.68, 0.82, 1.0);
    var col = mix(zenith, horizon_col, horizon);
    let sun_amount = max(dot(ray_dir, sun_dir), 0.0);
    col = col + vec3<f32>(1.0, 0.72, 0.38) * pow(sun_amount, 96.0);
    col = col + vec3<f32>(0.55, 0.68, 0.95) * pow(sun_amount, 8.0) * 0.25;
    return col;
}

fn cloud_density(p: vec3<f32>) -> f32 {
    let wind = vec3<f32>(params.time * 0.015, 0.0, params.time * 0.006);
    let shape = fbm((p + wind) * vec3<f32>(0.010, 0.045, 0.010));
    let detail = fbm((p + wind * 2.7) * vec3<f32>(0.045, 0.12, 0.045));
    let height_fade = smoothstep(35.0, 85.0, p.y) * (1.0 - smoothstep(150.0, 230.0, p.y));
    return clamp((shape * 0.78 + detail * 0.22 - 0.50) * 3.2, 0.0, 1.0) * height_fade;
}

fn volumetric_clouds(ray_o: vec3<f32>, ray_dir: vec3<f32>, base_sky: vec3<f32>) -> vec3<f32> {
    if (ray_dir.y <= 0.02) { return base_sky; }
    let t0 = max((45.0 - ray_o.y) / ray_dir.y, 0.0);
    let t1 = min((220.0 - ray_o.y) / ray_dir.y, 1200.0);
    if (t1 <= t0) { return base_sky; }
    var transmittance = 1.0;
    var cloud_light = vec3<f32>(0.0);
    let sun_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let step_len = (t1 - t0) / 18.0;
    for (var i = 0; i < 18; i = i + 1) {
        let t = t0 + (f32(i) + 0.5) * step_len;
        let p = ray_o + ray_dir * t;
        let d = cloud_density(p);
        let alpha = 1.0 - exp(-d * step_len * 0.018);
        let lighting = 0.55 + 0.45 * max(dot(ray_dir, sun_dir), 0.0);
        let cloud_col = mix(vec3<f32>(0.55, 0.58, 0.62), vec3<f32>(1.0, 0.96, 0.88), lighting);
        cloud_light = cloud_light + transmittance * alpha * cloud_col;
        transmittance = transmittance * (1.0 - alpha);
        if (transmittance < 0.04) { break; }
    }
    return base_sky * transmittance + cloud_light;
}

fn apply_distance_fog(col: vec3<f32>, dist: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    let fog_col = sky_color(ray_dir);
    let fog_dist = max(dist - 500.0, 0.0);
    let fog_amount = 1.0 - exp(-fog_dist * 0.0008);
    let height_fog = clamp(1.0 - ray_dir.y * 0.65, 0.25, 1.0);
    return mix(col, fog_col, clamp(fog_amount * height_fog, 0.0, 0.92));
}

fn transform_normal(model_matrix: mat4x4<f32>, local_normal: vec3<f32>) -> vec3<f32> {
    let x = model_matrix[0].xyz;
    let y = model_matrix[1].xyz;
    let z = model_matrix[2].xyz;
    let nx = cross(y, z);
    let ny = cross(z, x);
    let nz = cross(x, y);
    let det = dot(x, nx);
    let normal = nx * local_normal.x + ny * local_normal.y + nz * local_normal.z;
    return normalize(select(normal, -normal, det < 0.0));
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

fn point_segment_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    return distance(p, a + ab * h);
}

fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let n = normalize(cross(ab, ac));
    let plane_p = p - n * dot(p - a, n);

    let c0 = cross(b - a, plane_p - a);
    let c1 = cross(c - b, plane_p - b);
    let c2 = cross(a - c, plane_p - c);
    let inside = dot(c0, n) >= 0.0 && dot(c1, n) >= 0.0 && dot(c2, n) >= 0.0;
    if (inside) {
        return abs(dot(p - a, n));
    }

    return min(
        point_segment_distance(p, a, b),
        min(point_segment_distance(p, b, c), point_segment_distance(p, c, a)),
    );
}

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn compute_triangle_local_sdf_debug(tri: Triangle, barycentric: vec3<f32>) -> vec2<f32> {
    let p = tri.v0.xyz * barycentric.x + tri.v1.xyz * barycentric.y + tri.v2.xyz * barycentric.z;
    let face_n = normalize(cross(tri.v1.xyz - tri.v0.xyz, tri.v2.xyz - tri.v0.xyz));
    let plane_dist = dot(p - tri.v0.xyz, face_n);
    let edge_dist = min(
        point_segment_distance(p, tri.v0.xyz, tri.v1.xyz),
        min(
            point_segment_distance(p, tri.v1.xyz, tri.v2.xyz),
            point_segment_distance(p, tri.v2.xyz, tri.v0.xyz),
        ),
    );
    let inside = all(barycentric >= vec3<f32>(0.0)) && dot(barycentric, vec3<f32>(1.0)) <= 1.0;
    let surface_sdf = select(sqrt(plane_dist * plane_dist + edge_dist * edge_dist), abs(plane_dist), inside);
    return vec2<f32>(surface_sdf, edge_dist);
}

fn compute_fusion_sdf(world_pos: vec3<f32>, source_instance_id: u32, source_tri_idx: u32) -> vec2<f32> {
    var hard_min = 1e6;
    var blended = 1e6;
    var hits = 0u;
    for (var inst_i = 0u; inst_i < params.instance_count; inst_i = inst_i + 1u) {
        let instance = instances[inst_i];
        let tri_base = instance.tri_start;

        let tri_count = min(instance.tri_count, params.anchor_count - min(tri_base, params.anchor_count));
        for (var local_tri = 0u; local_tri < tri_count; local_tri = local_tri + 1u) {
            let tri_idx = tri_base + local_tri;
            if (inst_i == source_instance_id && tri_idx == source_tri_idx) {
                continue;
            }
            let tri = triangles[tri_idx];
            let w0 = (instance.model_matrix * tri.v0).xyz;
            let w1 = (instance.model_matrix * tri.v1).xyz;
            let w2 = (instance.model_matrix * tri.v2).xyz;
            let d = point_triangle_distance(world_pos, w0, w1, w2);
            hard_min = min(hard_min, d);
            blended = smooth_min(blended, d, 0.12);
            hits = hits + 1u;
        }
    }

    if (hits == 0u) {
        return vec2<f32>(1e6, 1e6);
    }
    return vec2<f32>(hard_min, blended);
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

fn shadow_at_pixel(pixel: vec2<i32>) -> f32 {
    return textureLoad(pixel_shadow_tex, pixel, 0).r;
}

fn ground_occludes_world_pos(world_pos: vec3<f32>) -> bool {
    let cam_y = params.cam_pos.y;
    let frag_y = world_pos.y;
    return (cam_y > 0.0 && frag_y < 0.0) || (cam_y < 0.0 && frag_y > 0.0);
}

fn world_to_sun_grid(w: vec3<f32>) -> vec2<i32> {
    let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
    var Up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(L.y) > 0.99) { Up = vec3<f32>(0.0, 0.0, 1.0); }
    let U = normalize(cross(L, Up));
    let V = cross(L, U);

    // 【核心改动】将世界坐标 w 偏置到以相机当前 XZ 坐标为中心
    // 这样无论场景物体多远，只要在相机周围 GRID_HALF_SIZE 范围内，就能获得阴影
    let camera_anchor = vec3<f32>(params.cam_pos.x, 0.0, params.cam_pos.z);
    let relative_w = w - camera_anchor;

    let x_proj = dot(relative_w, U);
    let y_proj = dot(relative_w, V);
    let u = i32(((x_proj / GRID_HALF_SIZE) * 0.5 + 0.5) * f32(GRID_RES));
    let v = i32(((y_proj / GRID_HALF_SIZE) * 0.5 + 0.5) * f32(GRID_RES));
    return vec2<i32>(u, v);
}

fn sun_grid_inside(g: vec2<i32>) -> bool {
    return g.x >= 0i && g.y >= 0i && g.x < i32(GRID_RES) && g.y < i32(GRID_RES);
}

fn ray_triangle_t(origin: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let e1 = b - a;
    let e2 = c - a;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if (abs(det) < 1e-6) { return -1.0; }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    return dot(e2, qvec) * inv_det;
}

fn trace_pixel_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, source_tri_id: u32) -> f32 {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let n = normalize(world_normal);
    let self_shadow_bias = 0.08;
    let source_instance_id = source_tri_id >> 20u;

    // 【核心优化 1】背光面直接判定为阴影，跳过所有链表和求交计算
    let cos_theta = dot(n, light_dir);
    if (cos_theta <= 0.0) {
        return 0.35;
    }

    let normal_shadow = mix(0.45, 1.0, smoothstep(0.0, 0.35, cos_theta));
    let origin = world_pos + n * 0.015 + light_dir * 0.02;

    if (light_dir.y < -0.001) {
        let ground_t = -origin.y / light_dir.y;
        if (ground_t > 0.02 && ground_t < 10000.0) { return 0.35; }
    }

    // 投影到太阳空间网格，遍历链表节点
    let g = world_to_sun_grid(origin);
    if (!sun_grid_inside(g)) {
        return normal_shadow;
    }
    let cell_idx = u32(g.y) * GRID_RES + u32(g.x);
    var curr_node_idx = atomicLoad(&grid_head[cell_idx]);

    if (curr_node_idx == 0u) {
        return normal_shadow;
    }

    var steps = 0u;
    while (curr_node_idx != 0u && steps < MAX_SHADOW_LIST_STEPS) {
        let node = grid_nodes[curr_node_idx];
        if (node.packed_tri_id != source_tri_id) {
            let instance_id = (node.packed_tri_id >> 20u) - 1u;
            let tri_idx = (node.packed_tri_id & 0x000fffffu) - 1u;
            let model_matrix = instances[instance_id].model_matrix;
            let tri = triangles[tri_idx];
            let a = model_matrix[0].xyz * tri.v0.x + model_matrix[1].xyz * tri.v0.y + model_matrix[2].xyz * tri.v0.z + model_matrix[3].xyz;
            let b = model_matrix[0].xyz * tri.v1.x + model_matrix[1].xyz * tri.v1.y + model_matrix[2].xyz * tri.v1.z + model_matrix[3].xyz;
            let c = model_matrix[0].xyz * tri.v2.x + model_matrix[1].xyz * tri.v2.y + model_matrix[2].xyz * tri.v2.z + model_matrix[3].xyz;
            let t = ray_triangle_t(origin, light_dir, a, b, c);
            if (t > self_shadow_bias && t < 10000.0) {
                let blocker_instance_id = node.packed_tri_id >> 20u;
                if (blocker_instance_id != source_instance_id || t > self_shadow_bias * 4.0) {
                    return 0.35;
                }
            }
        }
        curr_node_idx = node.next;
        steps = steps + 1u;
    }
    if (curr_node_idx != 0u) {
        return 0.35;
    }
    return normal_shadow;
}

@compute @workgroup_size(64, 1, 1)
fn cs_binning(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let instance_idx = wid.y;
    let local_tri_idx = gid.x;
    if (instance_idx >= params.instance_count) { return; }
    let instance = instances[instance_idx];
    if (local_tri_idx >= instance.tri_count) { return; }
    let global_tri_idx = instance.tri_start + local_tri_idx;
    let tri = triangles[global_tri_idx];
    let w0 = (instance.model_matrix * tri.v0).xyz;
    let w1 = (instance.model_matrix * tri.v1).xyz;
    let w2 = (instance.model_matrix * tri.v2).xyz;

    // 【核心优化 2】背向光源的三角形（Back-facing）不可能投射阴影，直接剔除
    let face_normal = cross(w1 - w0, w2 - w0);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    if (dot(face_normal, light_dir) >= 0.0) {
        return;
    }

    let g0 = world_to_sun_grid(w0);
    let g1 = world_to_sun_grid(w1);
    let g2 = world_to_sun_grid(w2);
    let x_min_raw = min(g0.x, min(g1.x, g2.x));
    let x_max_raw = max(g0.x, max(g1.x, g2.x));
    let y_min_raw = min(g0.y, min(g1.y, g2.y));
    let y_max_raw = max(g0.y, max(g1.y, g2.y));
    if (x_max_raw < 0i || y_max_raw < 0i || x_min_raw >= i32(GRID_RES) || y_min_raw >= i32(GRID_RES)) { return; }
    let x_min = max(x_min_raw, 0i);
    let x_max = min(x_max_raw, i32(GRID_RES - 1u));
    let y_min = max(y_min_raw, 0i);
    let y_max = min(y_max_raw, i32(GRID_RES - 1u));

    // 【核心优化 3】防止超大三角形过度分箱，挤占公共链表池
    // 注意：GRID_RES=1024 时，同尺寸三角形覆盖格子数是 256 分辨率下的 16 倍，
    //       因此阈值从 256 放宽到 4096 以容纳巨物表面
    let binned_area = (x_max - x_min + 1) * (y_max - y_min + 1);
    if (binned_area > 4096) {
        return;
    }

    let packed_tri_id = ((instance_idx + 1u) << 20u) | (global_tri_idx + 1u);
    for (var y = y_min; y <= y_max; y = y + 1) {
        for (var x = x_min; x <= x_max; x = x + 1) {
            let cell_idx = u32(y) * GRID_RES + u32(x);
            let node_idx = atomicAdd(&global_counter, 1u) + 1u;
            if (node_idx < arrayLength(&grid_nodes)) {
                let old_head = atomicExchange(&grid_head[cell_idx], node_idx);
                grid_nodes[node_idx].next = old_head;
                grid_nodes[node_idx].packed_tri_id = packed_tri_id;
                grid_nodes[node_idx].cell_idx = cell_idx;
                grid_nodes[node_idx]._pad = 0u;
            }
        }
    }
}

@compute @workgroup_size(8, 8)
fn cs_shadow_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));
    let warp_idx = gid.y * params.screen_width + gid.x;
    let flags = atomicLoad(&warpBuffer[warp_idx].flags);

    var source_tri_id = 0u;
    var world_pos: vec3<f32>;
    var world_normal = vec3<f32>(0.0, 1.0, 0.0);

    if (flags > 0u) {
        source_tri_id = warpBuffer[warp_idx].tri_id;
        let instance_id = (source_tri_id >> 20u) - 1u;
        let tri_idx = (source_tri_id & 0x000fffffu) - 1u;
        let instance = instances[instance_id];
        let tri = triangles[tri_idx];
        let b = warpBuffer[warp_idx].barycentric.xyz;
        let local_pos = tri.v0.xyz * b.x + tri.v1.xyz * b.y + tri.v2.xyz * b.z;
        let world_pos_pre = instance.model_matrix * vec4<f32>(local_pos, 1.0);
        world_pos = world_pos_pre.xyz / world_pos_pre.w;
        let local_normal = normalize(tri.n0.xyz * b.x + tri.n1.xyz * b.y + tri.n2.xyz * b.z);
        world_normal = transform_normal(instance.model_matrix, local_normal);
    } else {
        let screen_pos = vec2<f32>(f32(gid.x), f32(gid.y));
        let size_f = vec2<f32>(f32(params.screen_width), f32(params.screen_height));
        let uv = (screen_pos / size_f) * 2.0 - 1.0;
        let ray_target = params.proj_inv * vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
        let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
        let ray_o = params.cam_pos.xyz;
        let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
        
        if (t_grid <= 0.0 || t_grid >= GRID_HALF_SIZE) {
            textureStore(pixel_shadow_out, pixel, vec4<f32>(1.0, 0.0, 0.0, 1.0));
            return;
        }
        world_pos = ray_o + ray_dir * t_grid;
    }

    let shadow = trace_pixel_shadow(world_pos, world_normal, source_tri_id);
    textureStore(pixel_shadow_out, pixel, vec4<f32>(shadow, 0.0, 0.0, 1.0));
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

    // --- 调试模式 4: 查看隐性 SDF 距离场 ---
    if (params.debug_mode == 4u) {
        if (flags == 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.02, 0.02, 0.04, 1.0));
        } else {
            let tri_id = warpBuffer[warp_idx].tri_id;
            let tri_idx = (tri_id & 0x000FFFFFu) - 1u;
            let tri = triangles[tri_idx];
            let b = warpBuffer[warp_idx].barycentric.xyz;
            let sdf_debug = compute_triangle_local_sdf_debug(tri, b);
            sdfOutput[warp_idx] = sdf_debug.x;
            let edge_heat = clamp(sdf_debug.y * 40.0, 0.0, 1.0);
            let contour = 1.0 - abs(fract(sdf_debug.y * 120.0) * 2.0 - 1.0);
            textureStore(output_texture, screen_coord, vec4<f32>(edge_heat, 1.0 - edge_heat, contour * 0.8, 1.0));
        }
        return;
    }

    // --- 调试模式 5: 查看实例间融合 SDF 预览 ---
    if (params.debug_mode == 5u) {
        if (flags == 0u) {
            textureStore(output_texture, screen_coord, vec4<f32>(0.02, 0.02, 0.04, 1.0));
        } else {
            let tri_id = warpBuffer[warp_idx].tri_id;
            let instance_id = (tri_id >> 20u) - 1u;
            let tri_idx = (tri_id & 0x000FFFFFu) - 1u;
            let instance = instances[instance_id];
            let tri = triangles[tri_idx];
            let b = warpBuffer[warp_idx].barycentric.xyz;
            let local_pos = tri.v0.xyz * b.x + tri.v1.xyz * b.y + tri.v2.xyz * b.z;
            let world_pos_pre = instance.model_matrix * vec4<f32>(local_pos, 1.0);
            let world_pos = world_pos_pre.xyz / world_pos_pre.w;
            let fusion = compute_fusion_sdf(world_pos, instance_id, tri_idx);
            let blend_gain = clamp((fusion.x - fusion.y) * 18.0, 0.0, 1.0);
            let near = 1.0 - clamp(fusion.x * 6.0, 0.0, 1.0);
            sdfOutput[warp_idx] = fusion.y;
            textureStore(output_texture, screen_coord, vec4<f32>(blend_gain, near, 1.0 - blend_gain, 1.0));
        }
        return;
    }

    if (flags == 0u) {
        sdfOutput[warp_idx] = 1e6;
    }

    // --- 最终着色 ---
    if (flags > 0u) {
        let tri_id = warpBuffer[warp_idx].tri_id;
        let instance_id = (tri_id >> 20u) - 1u;
        let tri_idx = (tri_id & 0x000FFFFFu) - 1u;
        let instance = instances[instance_id];
        let tri = triangles[tri_idx];
        let b = warpBuffer[warp_idx].barycentric.xyz;
        let sdf_debug = compute_triangle_local_sdf_debug(tri, b);
        sdfOutput[warp_idx] = sdf_debug.x;

        let local_pos = tri.v0.xyz * b.x + tri.v1.xyz * b.y + tri.v2.xyz * b.z;
        let world_pos_pre = instance.model_matrix * vec4<f32>(local_pos, 1.0);
        let world_pos = world_pos_pre.xyz / world_pos_pre.w;

        let model_uv = tri.uv01.xy * b.x + tri.uv01.zw * b.y + tri.uv2.xy * b.z;
        let tex_color = textureSampleLevel(t_albedo, s_albedo, model_uv, 0.0).rgb;

        let local_n = normalize(tri.n0.xyz * b.x + tri.n1.xyz * b.y + tri.n2.xyz * b.z);
        let macro_n = transform_normal(instance.model_matrix, local_n);

        var final_n = macro_n;
        if (params.distort_strength > 0.001) {
            let dist = distance(params.cam_pos.xyz, world_pos);
            var octaves = 6;
            if (dist > 120.0) {
                octaves = 1;
            } else if (dist > 40.0) {
                octaves = 3;
            }
            let eps = 0.002;
            let freq = max(params.distort_frequency, 0.001);
            let h_center = fbm_lod(world_pos * 10.0 * freq, octaves);
            let h_x = fbm_lod((world_pos + vec3<f32>(eps, 0.0, 0.0)) * 10.0 * freq, octaves);
            let h_y = fbm_lod((world_pos + vec3<f32>(0.0, eps, 0.0)) * 10.0 * freq, octaves);
            let h_z = fbm_lod((world_pos + vec3<f32>(0.0, 0.0, eps)) * 10.0 * freq, octaves);
            let grad = vec3<f32>(h_x - h_center, h_y - h_center, h_z - h_center) / eps;
            final_n = normalize(macro_n - grad * params.distort_strength);
        }

        let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diff = max(dot(final_n, L), 0.0) * 0.8 + 0.2;
        let shadow = shadow_at_pixel(screen_coord);

        var final_col = tex_color * diff * shadow;
        let ray_dir = normalize(world_pos - params.cam_pos.xyz);
        let view_dist = distance(params.cam_pos.xyz, world_pos);
        final_col = apply_distance_fog(final_col, view_dist, ray_dir);
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

    var bg_col = volumetric_clouds(ray_o, ray_dir, sky_color(ray_dir));
    let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
    if (t_grid > 0.0 && t_grid < 10000.0) {
        let p = ray_o + ray_dir * t_grid;
        let cell = floor(p.xz);
        let checker = select(0.0, 1.0, (i32(cell.x + cell.y) & 1) == 0);
        let base = mix(vec3<f32>(0.18, 0.18, 0.19), vec3<f32>(0.28, 0.28, 0.30), checker);
        let grid_uv = abs(fract(p.xz - 0.5) - 0.5);
        let grid = min(1.0, smoothstep(0.025, 0.0, grid_uv.x) + smoothstep(0.025, 0.0, grid_uv.y));
        let axis_x = smoothstep(0.035, 0.0, abs(p.z));
        let axis_z = smoothstep(0.035, 0.0, abs(p.x));
        bg_col = mix(base, vec3<f32>(0.05, 0.05, 0.06), grid * 0.45);
        bg_col = mix(bg_col, vec3<f32>(0.75, 0.18, 0.18), axis_x * 0.7);
        bg_col = mix(bg_col, vec3<f32>(0.18, 0.35, 0.75), axis_z * 0.7);
        bg_col *= shadow_at_pixel(screen_coord);
        bg_col = apply_distance_fog(bg_col, t_grid, ray_dir);
    }
    bg_col = pow(bg_col, vec3<f32>(1.0 / 2.2));
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
    if (params.debug_mode != 6u) {
        let normal_x = vertex_data.w * 2.0 - 1.0;
        let normal = vec3<f32>(normal_x, 0.0, 0.0);
        let to_cam = normalize(params.cam_pos.xyz - p_world);
        let dot_prod = dot(normal, to_cam);
        if (dot_prod < -0.05) {
            out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
            out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return out;
        }
    }
    let p_clip = params.prev_view_proj * vec4<f32>(p_world, 1.0);
    out.position = p_clip;
    out.color = select(vec4<f32>(0.0, 0.6, 1.0, 0.5), vec4<f32>(1.0, 0.85, 0.0, 0.9), params.debug_mode == 6u);
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

fn original_screen_clip(world_pos: vec3<f32>) -> vec4<f32> {
    let clip = params.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let safe_w = max(clip.w, 0.00001);
    let ndc = clip.xyz / safe_w;
    return vec4<f32>(
        (ndc.x * 0.5 + 0.5) * f32(params.screen_width),
        (0.5 - ndc.y * 0.5) * f32(params.screen_height),
        ndc.z,
        clip.w,
    );
}

fn original_screen_pos(world_pos: vec3<f32>) -> vec2<f32> {
    return original_screen_clip(world_pos).xy;
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

    let s0 = original_screen_pos(w0);
    let s1 = original_screen_pos(w1);
    let s2 = original_screen_pos(w2);
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

    let view_proj = params.prev_view_proj;
    let c0 = view_proj * (instance.model_matrix * tri.v0);
    let c1 = view_proj * (instance.model_matrix * tri.v1);
    let c2 = view_proj * (instance.model_matrix * tri.v2);

    var out: DepthVertexOutput;
    if (c0.w <= 0.0 || c1.w <= 0.0 || c2.w <= 0.0) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

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
    let world_normal = transform_normal(instance.model_matrix, n_local);
    let warp_normal = transform_normal(instance.model_matrix, warp_n_local);
    
    if (params.distort_strength <= 0.001) {
        out.position = params.prev_view_proj * world_pos_pre;
    } else {
        out.position = unwrap_triangle_pos(tri, p_local, instance);
    }
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
    if (ground_occludes_world_pos(in.world_pos)) {
        discard;
    }

    let packed_tri = ((in.instance_id + 1u) << 20u) | (in.triangle_id + 1u);

    let projected = warped_screen_clip(in.world_pos, in.warp_normal);
    if (projected.w > 0.0) {
        let base_target = vec2<i32>(i32(floor(projected.x)), i32(floor(projected.y)));

        if (base_target.x >= 0 && base_target.x < i32(params.screen_width) &&
            base_target.y >= 0 && base_target.y < i32(params.screen_height)) {
            let check_idx = u32(base_target.y) * params.screen_width + u32(base_target.x);
            let current_depth = atomicLoad(&warpBuffer[check_idx].flags);
            let my_depth = u32(clamp(((projected.z) + 1.0) * 0.5, 0.0, 1.0) * 4294967295.0);
            if (current_depth != 0u && my_depth >= current_depth) {
                discard;
            }
        }

        let source = vec2<i32>(i32(floor(in.position.x)), i32(floor(in.position.y)));
        if (params.distort_strength <= 0.001) {
            write_warp_pixel(base_target, source, packed_tri, projected.z, in.barycentric);
        } else {
            let view_dir = normalize(params.cam_pos.xyz - in.world_pos);
            let dist = distance(params.cam_pos.xyz, in.world_pos);

            let facing = abs(dot(normalize(in.world_normal), view_dir));
            let grazing_coverage = 1.0 - smoothstep(0.08, 0.35, facing);

            var radius = select(1i, 3i, grazing_coverage > 0.35);
            if (dist > 100.0) {
                radius = 1i;
            }

            let radius_sq_limit = f32(radius * radius) + 0.25;
            for (var oy = -radius; oy <= radius; oy = oy + 1) {
                let oy_sq = f32(oy * oy);
                for (var ox = -radius; ox <= radius; ox = ox + 1) {
                    let dist_sq = f32(ox * ox) + oy_sq;
                    if (dist_sq <= radius_sq_limit) {
                        write_warp_pixel(base_target + vec2<i32>(ox, oy), source, packed_tri, projected.z, in.barycentric);
                    }
                }
            }

            let raw_proj_dx = dpdx(projected);
            let raw_proj_dy = dpdy(projected);
            let raw_bary_dx = dpdx(in.barycentric);
            let raw_bary_dy = dpdy(in.barycentric);

            let max_axis = mix(5.0, 18.0, grazing_coverage);
            let dx_len = length(raw_proj_dx.xy);
            let dy_len = length(raw_proj_dy.xy);
            let dx_scale = min(1.0, max_axis / max(dx_len, 0.0001));
            let dy_scale = min(1.0, max_axis / max(dy_len, 0.0001));
            let proj_dx = raw_proj_dx * dx_scale;
            let proj_dy = raw_proj_dy * dy_scale;
            let bary_dx = raw_bary_dx * dx_scale;
            let bary_dy = raw_bary_dy * dy_scale;

            let footprint_area = abs(proj_dx.x * proj_dy.y - proj_dx.y * proj_dy.x);
            let area_scale = select(1.0, mix(0.75, 1.0, grazing_coverage), footprint_area < 0.25 || footprint_area > 144.0);
            let stable_proj_dx = proj_dx * area_scale;
            let stable_proj_dy = proj_dy * area_scale;
            let stable_bary_dx = bary_dx * area_scale;
            let stable_bary_dy = bary_dy * area_scale;

            let fast_dx = length(stable_proj_dx.xy);
            let fast_dy = length(stable_proj_dy.xy);

            if (fast_dx <= 1.2 && fast_dy <= 1.2) {
                write_warp_pixel(base_target, source, packed_tri, projected.z, in.barycentric);
            } else {
                let max_step = i32(clamp(8.0 - max(dist - 60.0, 0.0) / 20.0, 1.0, 8.0));
                let x_steps = min(max(i32(ceil(fast_dx)), 1), max_step);
                let y_steps = min(max(i32(ceil(fast_dy)), 1), max_step);
                let dx_p = stable_proj_dx / f32(x_steps);
                let dx_b = stable_bary_dx / f32(x_steps);
                let dy_p = stable_proj_dy / f32(y_steps);
                let dy_b = stable_bary_dy / f32(y_steps);

                let start_p = projected - stable_proj_dx * 0.5 - stable_proj_dy * 0.5;
                let start_b = in.barycentric - stable_bary_dx * 0.5 - stable_bary_dy * 0.5;

                for (var sy = 0i; sy <= y_steps; sy = sy + 1) {
                    let row_p = start_p + dy_p * f32(sy);
                    let row_b = start_b + dy_b * f32(sy);

                    var curr_p = row_p;
                    var curr_b = row_b;

                    for (var sx = 0i; sx <= x_steps; sx = sx + 1) {
                        write_warp_pixel(vec2<i32>(i32(floor(curr_p.x)), i32(floor(curr_p.y))), source, packed_tri, curr_p.z, curr_b.xyz);
                        curr_p = curr_p + dx_p;
                        curr_b = curr_b + dx_b;
                    }
                }
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
    let normal = transform_normal(instance.model_matrix, n);

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
    if (ground_occludes_world_pos(in.world_pos)) {
        discard;
    }

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