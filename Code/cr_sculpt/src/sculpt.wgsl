struct Primitive {
    inv_model_matrix: mat4x4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
};

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
};

const TRI_INDEX_MAP_SIZE: u32 = 64u;
const TRI_PER_CELL: u32 = 4u;



struct Params {
    view_inv: mat4x4<f32>,      // 64 bytes
    proj_inv: mat4x4<f32>,      // 64 bytes
    prev_view_proj: mat4x4<f32>, // 64 bytes
    cam_pos: vec4<f32>,         // 16 bytes
    light_dir: vec4<f32>,       // 16 bytes
    
    // 数据包 A (16 bytes)
    prim_count: u32,      // 4
    anchor_count: u32,    // 4
    scaffold_count: u32,  // 4
    is_moving: u32,       // 4
    
    grid_origin: vec4<f32>,  // 16 bytes
    
    // 数据包 B (16 bytes)
    time: f32,    // 4
    _pad1: u32,   // 4
    _pad2: u32,   // 4
    _pad3: u32,   // 4

    model_center: vec4<f32>, // 16 bytes
    
    disk_center: vec4<f32>,  // 16 bytes
    disk_radius: f32,        // 4
    base_radius: f32,        // 4
    debug_mode: u32,         // 4 - 0=正常, 1=圆盘调试, 2=圆球调试
    _padding: u32,           // 4 ← 确保总大小为 16 的倍数（400字节）
};

// 1. Compute 阶段使用的声明 (Group 0)
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> scaffold: array<vec4<f32>>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var tri_id_tex: texture_2d<u32>;

// 2. Render 阶段使用的声明 (注意：我们让它也用 Group 0，因为它们在不同的 Pass 运行)
@group(0) @binding(0) var t_read: texture_2d<f32>;

// 平滑并集 (Smooth Union)
fn smin(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

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

// ====================== 圆盘极坐标调试 ======================
fn is_in_disk(screen_pos: vec2<f32>) -> bool {
    let c = params.disk_center.xy;
    let r = params.disk_radius * 1.12;   // 留一点 margin
    return distance(screen_pos, c) <= r + 8.0; // 额外再加 8 像素 padding
}

fn get_polar_coords(screen_pos: vec2<f32>) -> vec3<f32> {
    let c = params.disk_center.xy;
    let delta = screen_pos - c;
    let r = length(delta) / params.disk_radius;
    let theta = atan2(delta.y, delta.x) / (3.1415926 * 2.0) + 0.5;
    return vec3<f32>(
        r,
        theta,
        1.0 - r * 0.7
    );
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(fract(h * 6.0) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    let sector = i32(h * 6.0);

    if (sector == 0)      { rgb = vec3<f32>(c, x, 0.0); }
    else if (sector == 1) { rgb = vec3<f32>(x, c, 0.0); }
    else if (sector == 2) { rgb = vec3<f32>(0.0, c, x); }
    else if (sector == 3) { rgb = vec3<f32>(0.0, x, c); }
    else if (sector == 4) { rgb = vec3<f32>(x, 0.0, c); }
    else                  { rgb = vec3<f32>(c, 0.0, x); }

    return rgb + m;
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

fn get_sdf_precise(p: vec3<f32>, target_tri_id: u32) -> vec4<f32> {
    let base_center = params.model_center.xyz;
    let base_radius = primitives[0].params.x;
    let d_base = length(p - base_center) - base_radius;
    
    if (target_tri_id == 0u) {
        return vec4<f32>(primitives[0].color.rgb, d_base);
    }

    // 直接使用三角形距离，不再融合球
    let tri = triangles[target_tri_id - 1u];
    let d_tri = udTriangle(p, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz) - 0.001;
    
    return vec4<f32>(primitives[0].color.rgb, d_tri);
}

fn get_blended_normal(p: vec3<f32>, tri_id: u32) -> vec3<f32> {
    let h = 0.001;
    let k = vec4<f32>(1.0, -1.0, -1.0, 1.0);
    return normalize(
        k.xyy * get_sdf_precise(p + k.xyy * h, tri_id).w +
        k.yyx * get_sdf_precise(p + k.yyx * h, tri_id).w +
        k.yxy * get_sdf_precise(p + k.yxy * h, tri_id).w +
        k.xxx * get_sdf_precise(p + k.xxx * h, tri_id).w
    );
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size_u = textureDimensions(output_texture);
    let screen_coord = vec2<i32>(i32(id.x), i32(id.y));
    
    if (screen_coord.x >= i32(size_u.x) || screen_coord.y >= i32(size_u.y)) { return; }

    // === 圆盘 + Voronoi 调试模式 ===
    if (params.debug_mode == 1u) {
        let screen_pos = vec2<f32>(f32(id.x), f32(id.y));

        var col = vec3<f32>(0.06, 0.06, 0.10);

        if (is_in_disk(screen_pos)) {
            var closest_dist = 999999.0;
            var closest_idx: u32 = 0u;
            var second_dist = 999999.0;

            for (var i = 0u; i < params.scaffold_count; i = i + 1u) {
                let p = scaffold[i].xy;
                let d = distance(screen_pos, p);

                if (d < closest_dist) {
                    second_dist = closest_dist;
                    closest_dist = d;
                    closest_idx = i;
                } else if (d < second_dist) {
                    second_dist = d;
                }
            }

            let hue = f32(closest_idx % 37u) / 37.0 * 6.28;
            col = hsv_to_rgb(hue / 6.28, 0.9, 0.95);

            let edge = abs(closest_dist - second_dist);
            if (edge < 2.5) {
                col = mix(col, vec3<f32>(0.0, 0.0, 0.0), 0.85);
            }

            if (closest_dist < 5.0) {
                col = vec3<f32>(1.0, 1.0, 1.0);
            }
        } else {
            col = vec3<f32>(0.4, 0.05, 0.05) * 0.6;
        }

        textureStore(output_texture, screen_coord, vec4<f32>(col, 1.0));
        return;
    }

    // === 圆球调试模式 (Vs球可视化) ===
    if (params.debug_mode == 3u) {
        let screen_size = vec2<f32>(f32(size_u.x), f32(size_u.y));
        let screen_pos = vec2<f32>(f32(id.x), f32(id.y));
        let uv = (screen_pos / screen_size) * 2.0 - 1.0;
        let ray_target = params.proj_inv * vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
        let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
        let ray_o = params.cam_pos.xyz;
        
        let sphere_center = params.model_center.xyz;
        let sphere_radius = params.base_radius;
        
        let oc = ray_o - sphere_center;
        let b = dot(oc, ray_dir);
        let c = dot(oc, oc) - sphere_radius * sphere_radius;
        let discriminant = b * b - c;
        
        if (discriminant > 0.0) {
            let t = -b - sqrt(discriminant);
            if (t > 0.0) {
                let hit_pos = ray_o + ray_dir * t;
                let normal = normalize(hit_pos - sphere_center);
                
                let light_dir = normalize(params.light_dir.xyz);
                let diffuse = max(dot(normal, light_dir), 0.0);
                let col = vec3<f32>(0.8, 0.2, 0.2) * (diffuse * 0.8 + 0.2);
                textureStore(output_texture, screen_coord, vec4<f32>(col, 1.0));
                return;
            }
        }
        
        textureStore(output_texture, screen_coord, vec4<f32>(0.1, 0.1, 0.12, 1.0));
        return;
    }

    // 1. 获取该像素对应的三角形 ID (来自光栅化 Pass)
    // 0 = 背景/无三角形, >0 = 三角形索引+1
    let tri_id = textureLoad(tri_id_tex, screen_coord, 0).r;

    // 2. 初始化射线和背景网格
    let screen_pos = vec2<f32>(f32(id.x), f32(id.y));
    let uv = (screen_pos / vec2<f32>(size_u)) * 2.0 - 1.0;
    let ray_target = params.proj_inv * vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
    let ray_o = params.cam_pos.xyz;

    // 绘制背景网格
    var final_col = vec3<f32>(0.1, 0.1, 0.12);
    let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
    if (t_grid > 0.0 && t_grid < 100.0) {
        let p = ray_o + ray_dir * t_grid;
        let grid_uv = abs(fract(p.xz - 0.5) - 0.5);
        let grid = smoothstep(0.05, 0.0, grid_uv.x) + smoothstep(0.05, 0.0, grid_uv.y);
        final_col = mix(final_col, vec3<f32>(0.2, 0.2, 0.25), grid);
    }

    // 3. 【核心改进】收集邻域内的所有三角形候选者
    var candidates: array<u32, 9>;
    var candidate_count = 0u;

    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let neighbor_coord = screen_coord + vec2<i32>(ox, oy);
            let tid = textureLoad(tri_id_tex, neighbor_coord, 0).r;
            
            if (tid > 0u) {
                candidates[candidate_count] = tid;
                candidate_count++;
            }
        }
    }

    // 4. 射线步进
    var t = 0.0;
    var hit = false;
    
    // 只有当有候选三角形时才进行步进
    if (candidate_count > 0u) {
        let sphere_center = params.model_center.xyz;
        let sphere_radius = params.base_radius;
        
        let oc = ray_o - sphere_center;
        let b = dot(oc, ray_dir);
        let c = dot(oc, oc) - sphere_radius * sphere_radius;
        let discriminant = b * b - c;
        
        if (discriminant > 0.0) {
            let t_sphere = -b - sqrt(discriminant);
            if (t_sphere > 0.0) {
                t = t_sphere; // 从球表面开始
            }
        }
        
        for (var i = 0u; i < 128u; i = i + 1u) {
            let p = ray_o + ray_dir * t;
            
            var min_d_tri = 1000.0;
            
            // 【核心改进】不再只算一个，而是算这几个候选三角形里最近的
            for (var c = 0u; c < candidate_count; c++) {
                let tri = triangles[candidates[c] - 1u];
                let d_tri = udTriangle(p, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz) - 0.001;
                min_d_tri = min(min_d_tri, d_tri);
            }
            
            // 直接使用三角形距离，不再融合球
            let d = min_d_tri;

            if (d < 0.0005) { hit = true; break; }
            t = t + d;
            if (t > 20.0) { break; }
        }
    }
    // candidate_count == 0 时不步进，直接显示背景网格

    // 5. 最终着色
    if (hit) {
        let p = ray_o + ray_dir * t;
        
        // 获取法线
        var n: vec3<f32>;
        if (candidate_count > 0u) {
            // 使用主像素的 tri_id 获取法线
            if (tri_id > 0u) {
                n = get_geometry_normal(tri_id - 1u);
            } else {
                // 如果主像素没有 tri_id，使用第一个候选的法线
                n = get_geometry_normal(candidates[0] - 1u);
            }
        } else {
            n = normalize(p - params.model_center.xyz);
        }

        // === 法线调试模式：直接显示法线颜色 ===
        if (params.debug_mode == 4u) {
            final_col = (n * 0.5 + 0.5);
            textureStore(output_texture, screen_coord, vec4<f32>(final_col, 1.0));
            return;
        }

        // 光照计算
        let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diff = max(dot(n, L), 0.0) * 0.8 + 0.2;
        
        // 如果是三角形区域，给点不一样的颜色看看
        var base_color = primitives[0].color.rgb;
        if (tri_id > 0u) {
            base_color = vec3<f32>(0.4, 0.6, 1.0); // 蓝色模型
        }
        
        final_col = base_color * diff;
        final_col = pow(final_col, vec3<f32>(1.0 / 2.2));
    }

    // 统一输出，保证网格线不消失
    textureStore(output_texture, screen_coord, vec4<f32>(final_col, 1.0));
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
    
    // 直接从 Storage Buffer 拿顶点
    let p_world = scaffold[idx].xyz;
    
    // 利用预计算的视图投影矩阵进行投影
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
};

@vertex
fn vs_depth(@builtin(vertex_index) vertex_index: u32) -> DepthVertexOutput {
    let tri_idx = vertex_index / 3u;
    let local_idx = vertex_index % 3u;

    let tri = triangles[tri_idx];
    var p: vec3<f32>;

    if (local_idx == 0u) {
        p = tri.v0.xyz;
    } else if (local_idx == 1u) {
        p = tri.v1.xyz;
    } else {
        p = tri.v2.xyz;
    }

    let clip_pos = params.prev_view_proj * vec4<f32>(p, 1.0);

    var out: DepthVertexOutput;
    out.position = clip_pos;
    out.triangle_id = tri_idx;
    return out;
}

struct DepthFragmentOutput {
    @location(0) triangle_id: u32,
};

@fragment
fn fs_depth(in: DepthVertexOutput) -> DepthFragmentOutput {
    var out: DepthFragmentOutput;
    out.triangle_id = in.triangle_id + 1u;
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