const INFLATION: f32 = 0.3;

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv01: vec4<f32>,
    uv2: vec4<f32>,
    neighbors: vec4<u32>,
};

struct InstanceData {
    model_matrix: mat4x4<f32>,
    model_matrix_inv: mat4x4<f32>,
    model_id: u32,
    instance_id: u32,
    tri_start: u32,
    _pad_inner: u32,
    extra_data: vec2<f32>,
    _pad: array<u32, 10>,
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

// 2. Render 阶段使用的声明 (注意：我们让它也用 Group 0，因为它们在不同的 Pass 运行)
@group(0) @binding(0) var t_read: texture_2d<f32>;

fn dot2(v: vec3<f32>) -> f32 {
    return dot(v, v);
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
    @location(1) uv: vec2<f32>, // 传递插值 UV
    @location(2) @interpolate(flat) instance_id: u32, // 新增：平直插值传递实例ID
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
    var uv: vec2<f32>;
    var vertex_normal: vec3<f32>;

    if (local_idx == 0u) {
        p_local = tri.v0.xyz; uv = tri.uv01.xy; vertex_normal = tri.n0.xyz;
    } else if (local_idx == 1u) {
        p_local = tri.v1.xyz; uv = tri.uv01.zw; vertex_normal = tri.n1.xyz;
    } else {
        p_local = tri.v2.xyz; uv = tri.uv2.xy; vertex_normal = tri.n2.xyz;
    }

    let inflated_p = p_local + vertex_normal * INFLATION;

    let world_pos = instance.model_matrix * vec4<f32>(inflated_p, 1.0);
    let clip_pos = params.prev_view_proj * world_pos;

    var out: DepthVertexOutput;
    out.position = clip_pos;
    out.triangle_id = tri_idx;
    out.uv = uv;
    out.instance_id = instance_index;
    return out;
}

struct DepthFragmentOutput {
    @location(0) triangle_id: u32,
    @location(1) uv: vec4<f32>, // 扩展为 vec4 以匹配 Rgba16Float 格式
};

@fragment
fn fs_depth(in: DepthVertexOutput) -> DepthFragmentOutput {
    var out: DepthFragmentOutput;
    // 简化：直接存 InstanceID + 1，0 代表背景
    out.triangle_id = in.instance_id + 1u;
    out.uv = vec4<f32>(in.uv, 0.0, 0.0);
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
    let tri_idx = v_idx / 3u;
    let local_idx = v_idx % 3u;
    let tri = triangles[tri_idx];

    var p: vec3<f32>; var uv: vec2<f32>; var vertex_normal: vec3<f32>;
    if (local_idx == 0u) { p = tri.v0.xyz; uv = tri.uv01.xy; vertex_normal = tri.n0.xyz; }
    else if (local_idx == 1u) { p = tri.v1.xyz; uv = tri.uv01.zw; vertex_normal = tri.n1.xyz; }
    else { p = tri.v2.xyz; uv = tri.uv2.xy; vertex_normal = tri.n2.xyz; }

    let inflated_p = p + vertex_normal * INFLATION;

    let world_pos = (instance.model_matrix * vec4<f32>(inflated_p, 1.0)).xyz;

    let normal = normalize((instance.model_matrix * vec4<f32>(vertex_normal, 0.0)).xyz);

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

fn apply_distortion(p: vec3<f32>) -> vec3<f32> {
    let offset = vec3<f32>(
        sin(p.y * 10.0 + params.time * 2.0) * 0.05,
        0.0,
        cos(p.y * 10.0 + params.time * 2.0) * 0.05
    );
    return p + offset;
}

fn smin(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

fn inverse_warp(p: vec3<f32>) -> vec3<f32> {
    let distortion = sin(p.y * 15.0 + params.time * 3.0) * 0.05;
    return p - vec3<f32>(distortion, 0.0, 0.0);
}

fn get_local_sdf(
    p_local: vec3<f32>,
    tri_idx: u32
) -> f32
{
    let p = p_local;

    var d = 1e9;

    let tri = triangles[tri_idx];

    d = min(
        d,
        udTriangle(
            p,
            tri.v0.xyz,
            tri.v1.xyz,
            tri.v2.xyz
        )
    );

    for(var i=0u;i<3u;i++)
    {
        let nid = tri.neighbors[i];

        if(nid!=0xFFFFFFFFu)
        {
            let nt = triangles[nid];

            d = min(
                d,
                udTriangle(
                    p,
                    nt.v0.xyz,
                    nt.v1.xyz,
                    nt.v2.xyz
                )
            );

            for(var j=0u;j<3u;j++)
            {
                let nid2 = nt.neighbors[j];

                if(
                    nid2!=0xFFFFFFFFu &&
                    nid2!=tri_idx
                )
                {
                    let nt2 = triangles[nid2];

                    d = min(
                        d,
                        udTriangle(
                            p,
                            nt2.v0.xyz,
                            nt2.v1.xyz,
                            nt2.v2.xyz
                        )
                    );
                }
            }
        }
    }

    let distortion =
        sin(p.y*15.0 + params.time*3.0)
        *0.05;

    return d - 0.01 - abs(distortion);
}

fn get_barycentric_uv(p: vec3<f32>, tri: Triangle) -> vec2<f32> {
    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;

    let uv0 = tri.uv01.xy;
    let uv1 = tri.uv01.zw;
    let uv2 = tri.uv2.xy;

    let v10 = v1 - v0;
    let v20 = v2 - v0;
    let vp0 = p - v0;

    let d00 = dot(v10, v10);
    let d01 = dot(v10, v20);
    let d11 = dot(v20, v20);
    let d20 = dot(vp0, v10);
    let d21 = dot(vp0, v20);
    let denom = d00 * d11 - d01 * d01;

    if (abs(denom) < 1e-6) { return uv0; }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return uv0 * u + uv1 * v + uv2 * w;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_mesh(in: MeshVertexOutput) -> FragmentOutput {
    let instance = instances[in.instance_id];
    let ray_o_world = params.cam_pos.xyz;
    let view_dir_world = normalize(in.world_pos - ray_o_world);

    let inv_model = instance.model_matrix_inv;
    let ray_o = (inv_model * vec4<f32>(ray_o_world, 1.0)).xyz;
    let ray_dir = normalize((inv_model * vec4<f32>(view_dir_world, 0.0)).xyz);

    let local_hit_pos = (inv_model * vec4<f32>(in.world_pos, 1.0)).xyz;
    var t = 0.0;

    var hit = false;
    var p: vec3<f32>;

    for (var i = 0u; i < 256u; i++) {
        p = ray_o + ray_dir * t;
        let d = get_local_sdf(p, in.global_tri_idx);

        if (d < 0.0005) { hit = true; break; }

        t += max(d,0.0005);

        if (t > 20.0) { break; }
    }

    if (!hit)
    {
        discard;
    }

    let h = 0.01;
    let k = vec2<f32>(1.0, -1.0);
    let n_local = normalize(
        k.xyy * get_local_sdf(p + k.xyy * h, in.global_tri_idx) +
        k.yyx * get_local_sdf(p + k.yyx * h, in.global_tri_idx) +
        k.yxy * get_local_sdf(p + k.yxy * h, in.global_tri_idx) +
        k.xxx * get_local_sdf(p + k.xxx * h, in.global_tri_idx)
    );
    let n = normalize((instance.model_matrix * vec4<f32>(n_local, 0.0)).xyz);

    let tri = triangles[in.global_tri_idx];
    var real_uv = get_barycentric_uv(p, tri);

    let tex_color = textureSample(t_material, s_material, real_uv).rgb;

    let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diff = max(dot(n, L), 0.0) * 0.8 + 0.2;
    var final_col = tex_color * diff;

    if (in.instance_id == params.selected_instance_id) {
        final_col = mix(final_col, vec3<f32>(1.0, 0.5, 0.0), 0.4);
    }

    let world_pos = instance.model_matrix * vec4<f32>(p, 1.0);
    let clip_pos = params.prev_view_proj * world_pos;
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragmentOutput;
    out.color = vec4<f32>(pow(final_col, vec3<f32>(1.0 / 2.2)), 1.0);
    out.depth = ndc_depth;

    return out;
}

// ============================================================