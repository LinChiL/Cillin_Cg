// BVH 节点结构
struct BVHNode {
    aabb_min: vec4<f32>, // xyz = min bounds, w = left_child (internal) or tri_start (leaf)
    aabb_max: vec4<f32>, // xyz = max bounds, w = right_child (internal) or negative tri_count (leaf)
};

struct BVHResult {
    distance: f32,
    tri_idx: u32,
};

// BVH 节点绑定 (Group 0, Binding 10)
@group(0) @binding(10) var<storage, read> bvh_nodes: array<BVHNode>;

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
    bvh_start: u32, // 复用原 _pad_inner 字段
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
    _pad: u32,                   // 4

    // Ap2 扭曲字段
    distort_strength: f32,       // 4
    distort_frequency: f32,      // 4
    ap2_iteration: u32,          // 4
    _pad2: u32,                  // 4
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

// 专用于 TRS（平移、旋转、缩放）仿射矩阵的快速求逆算法
fn invert_trs_matrix(m: mat4x4<f32>) -> mat4x4<f32> {
    let t = m[3].xyz;
    let c0 = m[0].xyz;
    let c1 = m[1].xyz;
    let c2 = m[2].xyz;

    let s0 = dot(c0, c0);
    let s1 = dot(c1, c1);
    let s2 = dot(c2, c2);

    // 逆矩阵的旋转行向量
    let r0 = c0 / max(s0, 1e-6);
    let r1 = c1 / max(s1, 1e-6);
    let r2 = c2 / max(s2, 1e-6);

    // 修复：使用点积精准投影逆平移分量
    let inv_t = vec3<f32>(
        -dot(r0, t),
        -dot(r1, t),
        -dot(r2, t)
    );

    return mat4x4<f32>(
        vec4<f32>(r0.x, r1.x, r2.x, 0.0),
        vec4<f32>(r0.y, r1.y, r2.y, 0.0),
        vec4<f32>(r0.z, r1.z, r2.z, 0.0),
        vec4<f32>(inv_t.x, inv_t.y, inv_t.z, 1.0)
    );
}

// 4x4 解析矩阵求逆（备用）
fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {
    let n11 = m[0][0]; let n12 = m[1][0]; let n13 = m[2][0]; let n14 = m[3][0];
    let n21 = m[0][1]; let n22 = m[1][1]; let n23 = m[2][1]; let n24 = m[3][1];
    let n31 = m[0][2]; let n32 = m[1][2]; let n33 = m[2][2]; let n34 = m[3][2];
    let n41 = m[0][3]; let n42 = m[1][3]; let n43 = m[2][3]; let n44 = m[3][3];

    let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n11 * n34 * n43 + n13 * n32 * n44 - n11 * n33 * n44;
    let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n11 * n24 * n43 - n13 * n22 * n44 + n11 * n23 * n44;
    let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n11 * n24 * n33 + n13 * n22 * n34 - n11 * n23 * n34;

    let det = n11 * t11 + n12 * t12 + n13 * t13 + n14 * t14;
    if (abs(det) < 1e-6) {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        );
    }
    let idet = 1.0 / det;

    var res: mat4x4<f32>;
    res[0][0] = t11 * idet;
    res[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n42 + n21 * n34 * n42 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    res[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    res[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    res[1][0] = t12 * idet;
    res[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    res[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    res[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    res[2][0] = t13 * idet;
    res[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    res[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - m[0][0] * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    res[2][3] = (n13 * n22 * n41 - n12 * n23 * m[3][0] - n13 * n21 * n42 + m[0][0] * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    res[3][0] = t14 * idet;
    res[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    res[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    res[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return res;
}

// 点到 AABB 的距离平方
fn distance_sq_point_aabb(p: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> f32 {
    let dx = max(0.0, max(aabb_min.x - p.x, p.x - aabb_max.x));
    let dy = max(0.0, max(aabb_min.y - p.y, p.y - aabb_max.y));
    let dz = max(0.0, max(aabb_min.z - p.z, p.z - aabb_max.z));
    return dx * dx + dy * dy + dz * dz;
}

// 基于栈的局部空间 BVH 遍历引擎（支持温启动优化）
fn get_closest_bvh(
    p_local: vec3<f32>, 
    tri_offset: u32, 
    bvh_start: u32, 
    initial_dist_sq: f32, // 传入母体三角形的距离平方作为初始剪枝上限
    initial_tri_idx: u32  // 传入母体三角形索引作为初始最接近三角形
) -> BVHResult {
    var min_dist_sq = initial_dist_sq;
    var closest_tri = initial_tri_idx;
    
    var stack: array<u32, 16>; // 栈大小安全缩减至 16，大幅降低寄存器压力，提升 Occupancy
    var stack_ptr = 0u;

    stack[stack_ptr] = bvh_start;
    stack_ptr = stack_ptr + 1u;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let d_sq = distance_sq_point_aabb(p_local, node.aabb_min.xyz, node.aabb_max.xyz);
        // 使用母体距离进行极其苛刻的早期剪枝，95% 的无关分支在第一层就会被这里直接干掉！
        if (d_sq >= min_dist_sq) {
            continue;
        }

        if (node.aabb_max.w < 0.0) {
            let tri_start = tri_offset + u32(node.aabb_min.w);
            let tri_count = u32(-node.aabb_max.w);
            for (var i = 0u; i < tri_count; i = i + 1u) {
                let global_idx = tri_start + i;
                let tri = triangles[global_idx];
                let dist = udTriangle(p_local, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
                let d_sq_tri = dist * dist;
                if (d_sq_tri < min_dist_sq) {
                    min_dist_sq = d_sq_tri;
                    closest_tri = global_idx;
                }
            }
        } else {
            let left_child = u32(node.aabb_min.w);
            let right_child = u32(node.aabb_max.w);

            let d_left = distance_sq_point_aabb(p_local, bvh_nodes[left_child].aabb_min.xyz, bvh_nodes[left_child].aabb_max.xyz);
            let d_right = distance_sq_point_aabb(p_local, bvh_nodes[right_child].aabb_min.xyz, bvh_nodes[right_child].aabb_max.xyz);

            if (d_left < d_right) {
                if (d_right < min_dist_sq && stack_ptr < 16u) {
                    stack[stack_ptr] = right_child;
                    stack_ptr = stack_ptr + 1u;
                }
                if (d_left < min_dist_sq && stack_ptr < 16u) {
                    stack[stack_ptr] = left_child;
                    stack_ptr = stack_ptr + 1u;
                }
            } else {
                if (d_left < min_dist_sq && stack_ptr < 16u) {
                    stack[stack_ptr] = left_child;
                    stack_ptr = stack_ptr + 1u;
                }
                if (d_right < min_dist_sq && stack_ptr < 16u) {
                    stack[stack_ptr] = right_child;
                    stack_ptr = stack_ptr + 1u;
                }
            }
        }
    }

    var res: BVHResult;
    res.distance = sqrt(min_dist_sq);
    res.tri_idx = closest_tri;
    return res;
}

// Ap2 扭曲相关函数

// 快速随机哈希
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D 值噪声 (Value Noise)
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // 缓动曲线 (Quintic smoothstep)
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash31(i + vec3<f32>(0.,0.,0.)), hash31(i + vec3<f32>(1.,0.,0.)), u.x),
            mix(hash31(i + vec3<f32>(0.,1.,0.)), hash31(i + vec3<f32>(1.,1.,0.)), u.x), u.y),
        mix(mix(hash31(i + vec3<f32>(0.,0.,1.)), hash31(i + vec3<f32>(1.,0.,1.)), u.x),
            mix(hash31(i + vec3<f32>(0.,1.,1.)), hash31(i + vec3<f32>(1.,1.,1.)), u.x), u.y), u.z
    );
}

// FBM 函数 (叠加 4 层噪声)
fn fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var current_p = p;

    // 4层迭代（Ap2 性能平衡点）
    for (var i = 0u; i < 4u; i++) {
        value += amplitude * noise3d(current_p);
        current_p *= 2.0; // 频率翻倍
        amplitude *= 0.5; // 振幅减半
    }
    return value;
}

// 扭曲函数 - 使用平滑的正弦波组合
fn get_distortion(p: vec3<f32>) -> vec3<f32> {
    let s = params.distort_strength;
    let f = params.distort_frequency * 2.0; // 适当调高频率看波动
    let t = params.time;

    // 使用互不相关的正弦波组合，确保三个轴都有平滑变形
    let dx = sin(p.y * f + t) * cos(p.z * f * 0.8 + t);
    let dy = sin(p.z * f * 1.1 + t) * cos(p.x * f * 0.9 + t);
    let dz = sin(p.x * f * 0.7 + t) * cos(p.y * f * 1.2 + t);

    return vec3<f32>(dx, dy, dz) * s;
}

// Ap2 关键：反向传播求解器
// 我们已知 P_new，要求 P_old，使得 P_old + get_distortion(p_old) = P_new
fn get_p_old(p_new: vec3<f32>) -> vec3<f32> {
    var p_old = p_new;
    // 增加迭代次数以保证凹陷处的精度（8次以上）
    for (var i = 0u; i < 8u; i++) {
        p_old = p_new - get_distortion(p_old);
    }
    return p_old;
}

// 增强型反向传播（带松弛迭代，解决掠射角黑洞）
fn get_p_old_refined(p_new: vec3<f32>) -> vec3<f32> {
    var p_old = p_new;
    // 增加到 12 次迭代，这是解决掠射角黑洞的关键
    for (var i = 0u; i < 12u; i++) {
        let offset = get_distortion(p_old);
        // 使用软约束迭代，防止在剧烈扭曲处震荡
        p_old = p_old * 0.2 + (p_new - offset) * 0.8;
    }
    return p_old;
}

// 优化 2：在全局作用域实现自适应距离场查询（Coarse-to-Fine Reject）
fn get_world_SDF(pos_world: vec3<f32>, inst: InstanceData, inv_m: mat4x4<f32>) -> f32 {
    let po = get_p_old_final(pos_world);
    let po_local = (inv_m * vec4<f32>(po, 1.0)).xyz;
    
    // 快速读取 BVH 根节点 AABB
    let root = bvh_nodes[inst.bvh_start];
    let d_root = sqrt(distance_sq_point_aabb(po_local, root.aabb_min.xyz, root.aabb_max.xyz));
    
    // 如果光线距离模型根包围盒较远，直接使用包围盒距离，跳过昂贵的递归遍历
    if (d_root > 0.05) {
        return d_root;
    }
    
    return get_closest_bvh(po_local, inst.tri_start, inst.bvh_start, 1e10, 0u).distance;
}

// 优化版本：利用已知的命中三角形进行温启动剪枝（用于法线偏导数计算）
fn get_world_SDF_optimized(
    pos_world: vec3<f32>,
    inst: InstanceData,
    inv_m: mat4x4<f32>,
    home_tri_idx: u32
) -> f32 {
    let po = get_p_old_final(pos_world);
    let po_local = (inv_m * vec4<f32>(po, 1.0)).xyz;
    
    let root = bvh_nodes[inst.bvh_start];
    let d_root = sqrt(distance_sq_point_aabb(po_local, root.aabb_min.xyz, root.aabb_max.xyz));
    
    if (d_root > 0.05) {
        return d_root;
    }
    
    // 利用已知的命中三角形进行温启动剪枝
    let home_tri = triangles[home_tri_idx];
    let d_home = udTriangle(po_local, home_tri.v0.xyz, home_tri.v1.xyz, home_tri.v2.xyz);
    let d_home_sq = d_home * d_home;
    
    return get_closest_bvh(po_local, inst.tri_start, inst.bvh_start, d_home_sq, home_tri_idx).distance;
}

// 距离场函数（用于法线计算）
fn d_field(pos: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let po = get_p_old(pos);
    return udTriangle(po, v0, v1, v2);
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

    let tri_id = textureLoad(tri_id_tex, screen_coord, 0).r;
    
    // 只有背景才画网格
    if (tri_id == 0u) {
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
    } else {
        // 物体区域留空，交给 fs_mesh 画
        textureStore(output_texture, screen_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
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

    if (local_idx == 0u) {
        p_local = tri.v0.xyz; uv = tri.uv01.xy;
    } else if (local_idx == 1u) {
        p_local = tri.v1.xyz; uv = tri.uv01.zw;
    } else {
        p_local = tri.v2.xyz; uv = tri.uv2.xy;
    }

    let world_pos = instance.model_matrix * vec4<f32>(p_local, 1.0);
    let clip_pos = params.prev_view_proj * world_pos;

    var out: DepthVertexOutput;
    out.position = clip_pos;
    out.triangle_id = tri_idx;
    out.uv = uv;
    out.instance_id = instance_index; // 传入实例索引
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
    // 修复：v_idx 已经带有 offset，直接除以 3 即可获得正确的全局三角形索引
    let tri_idx = v_idx / 3u;
    let local_idx = v_idx % 3u;
    let tri = triangles[tri_idx];

    var p: vec3<f32>; var uv: vec2<f32>; var n: vec3<f32>;
    if (local_idx == 0u) { p = tri.v0.xyz; uv = tri.uv01.xy; n = tri.n0.xyz; }
    else if (local_idx == 1u) { p = tri.v1.xyz; uv = tri.uv01.zw; n = tri.n1.xyz; }
    else { p = tri.v2.xyz; uv = tri.uv2.xy; n = tri.n2.xyz; }

    // --- 纯法向膨胀：得益于 180 度平滑焊接，这里直接沿法线拉伸，面绝对不会开裂，且外观完全符合原网格流向 ---
    let local_normal = normalize(n);
    let expansion = params.distort_strength * 2.5;
    let p_expanded = p + local_normal * expansion;

    let world_pos = (instance.model_matrix * vec4<f32>(p_expanded, 1.0)).xyz;

    // 依然使用平滑法线
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

// ============================================================
// 强化的自适应非线性震荡抑制求解器（彻底消除发散黑洞）
// ============================================================
fn get_p_old_final(p_new: vec3<f32>) -> vec3<f32> {
    let s = params.distort_strength;
    if (s < 1e-5) {
        return p_new;
    }

    var p_old = p_new;
    var p_prev = p_new;
    
    // 初始赋予一个较为温和的松弛因子
    var alpha = 0.35; 

    for (var i = 0u; i < 32u; i++) {
        let offset = get_distortion(p_old);
        let p_target = p_new - offset;
        let p_next = mix(p_old, p_target, alpha);

        let diff = p_next - p_old;
        if (dot(diff, diff) < 1e-7) {
            p_old = p_next;
            break;
        }

        // 数学核心：检测更新方向的夹角
        let step_dir = p_next - p_old;
        let prev_dir = p_old - p_prev;
        
        if (i > 0u) {
            let correlation = dot(step_dir, prev_dir);
            if (correlation < 0.0) {
                // 检测到震荡（超调），大幅度削减松弛因子以强行平抑
                alpha = max(alpha * 0.45, 0.01);
            } else {
                // 收敛平稳，稍微加速以提高效率
                alpha = min(alpha * 1.05, 0.5);
            }
        }

        p_prev = p_old;
        p_old = p_next;
    }
    return p_old;
}

// 终极距离场函数（用于法线计算）
fn dist_func_final(pos: vec3<f32>, inst: InstanceData, tri_idx: u32) -> f32 {
    let po = get_p_old_final(pos);
    let tri = triangles[tri_idx];
    return udTriangle(po, (inst.model_matrix*tri.v0).xyz, (inst.model_matrix*tri.v1).xyz, (inst.model_matrix*tri.v2).xyz);
}

// 辅助函数：计算点 p 在三角形 (v0, v1, v2) 上的重心坐标
fn get_barycentric(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a; let v1 = c - a; let v2 = p - a;
    let d00 = dot(v0, v0); let d01 = dot(v0, v1); let d11 = dot(v1, v1);
    let d20 = dot(v2, v0); let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    return vec3<f32>(u, v, w);
}

// 辅助函数：计算扭曲空间中的三角形距离场
fn get_d_field(pos: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, m: mat4x4<f32>) -> f32 {
    let po = get_p_old_final(pos);
    return udTriangle(po, (m * vec4(v0, 1.0)).xyz, (m * vec4(v1, 1.0)).xyz, (m * vec4(v2, 1.0)).xyz);
}

@fragment
fn fs_mesh(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    if (params.debug_mode == 1u) {
        let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diff = dot(normalize(in.normal), L) * 0.5 + 0.5;
        return vec4<f32>(vec3<f32>(0.1, 0.5, 1.0) * diff, 0.7); 
    }

    let ray_o = params.cam_pos.xyz;
    let view_dir = normalize(in.world_pos - ray_o);
    
    // ============================================================
    // 暴力提升 1：动态起点收拢
    // 扭曲小的时候，回退距离极度收窄，确保光线几乎从外壳表面直接贴脸开始步进
    // ============================================================
    let start_back = max(params.distort_strength * 1.5, 0.005);
    var t = distance(ray_o, in.world_pos) - start_back;
    
    // ============================================================
    // 暴力提升 2：拓宽绝对射程 t_max 
    // 即使在强度极低时，也给予至少 0.45 的绝对航程，防止掠射角由于三角拉伸而提前终止
    // ============================================================
    let t_max = t + params.distort_strength * 8.0 + 0.45; 

    var hit = false;
    var hit_tri_idx = 0u;
    var prev_t = t;
    var final_p_old = vec3<f32>(0.0);

    let inst = instances[in.instance_id];
    let inv_model_mat = invert_trs_matrix(inst.model_matrix);
    let root_node = bvh_nodes[inst.bvh_start];

    let L_est = params.distort_strength * params.distort_frequency * 2.0;
    let step_mult = 0.45 / (1.0 + L_est);

    // ============================================================
    // 暴力提升 3：将最大迭代步数提升至 280 步（硬核保底）
    // 得益于之前的 BVH 温启动，正常像素会极速退出，只有掠射角才会消耗高步数
    // ============================================================
    for (var i = 0u; i < 280u; i++) {
        let p_curr = ray_o + view_dir * t;
        let p_old = get_p_old_final(p_curr);
        let po_local = (inv_model_mat * vec4<f32>(p_old, 1.0)).xyz;
        
        let d_root = sqrt(distance_sq_point_aabb(po_local, root_node.aabb_min.xyz, root_node.aabb_max.xyz));
        var min_d = d_root;

        if (d_root <= 0.05) {
            let home_tri_idx = in.global_tri_idx;
            let home_tri = triangles[home_tri_idx];
            
            let d_home = udTriangle(po_local, home_tri.v0.xyz, home_tri.v1.xyz, home_tri.v2.xyz);
            let d_home_sq = d_home * d_home;

            let bvh_res = get_closest_bvh(po_local, inst.tri_start, inst.bvh_start, d_home_sq, home_tri_idx);
            min_d = bvh_res.distance;

            if (min_d < 0.003) {
                var low = prev_t;
                var high = t;
                for (var k = 0u; k < 15u; k++) {
                    let mid = (low + high) * 0.5;
                    let p_mid = get_p_old_final(ray_o + view_dir * mid);
                    let p_mid_local = (inv_model_mat * vec4<f32>(p_mid, 1.0)).xyz;
                    
                    let d_home_mid = udTriangle(p_mid_local, home_tri.v0.xyz, home_tri.v1.xyz, home_tri.v2.xyz);
                    let dm = get_closest_bvh(p_mid_local, inst.tri_start, inst.bvh_start, d_home_mid * d_home_mid, home_tri_idx).distance;
                    if (dm < 0.003) { 
                        high = mid; 
                    } else { 
                        low = mid; 
                    }
                }
                t = high;
                final_p_old = get_p_old_final(ray_o + view_dir * t);
                
                let po_local_refined = (inv_model_mat * vec4<f32>(final_p_old, 1.0)).xyz;
                let d_home_refined = udTriangle(po_local_refined, home_tri.v0.xyz, home_tri.v1.xyz, home_tri.v2.xyz);
                hit_tri_idx = get_closest_bvh(po_local_refined, inst.tri_start, inst.bvh_start, d_home_refined * d_home_refined, home_tri_idx).tri_idx;
                
                hit = true;
                break;
            }
        }

        prev_t = t;
        // ============================================================
        // 暴力提升 4：硬核保底步长提升至 0.0015
        // 防止光线在近乎平行的掠射面上由于微小的 min_d 陷入无限慢滑步
        // ============================================================
        t += max(min_d * step_mult, 0.0015);
        if (t > t_max) { break; }
    }

    if (!hit) { discard; }

    // --- 后续法线与光照计算保持一致 ---
    let tri_f = triangles[hit_tri_idx];
    let final_p_old_local = (inv_model_mat * vec4<f32>(final_p_old, 1.0)).xyz;
    let bary = get_barycentric(final_p_old_local, tri_f.v0.xyz, tri_f.v1.xyz, tri_f.v2.xyz);
    let smooth_n = normalize(tri_f.n0.xyz * bary.x + tri_f.n1.xyz * bary.y + tri_f.n2.xyz * bary.z);
    let world_n = normalize((inst.model_matrix * vec4<f32>(smooth_n, 0.0)).xyz);

    let e = 0.005;
    let p_h = ray_o + view_dir * t;
    
    // 修复：使用 get_world_SDF_optimized 并传入确切的 hit_tri_idx
    let g_n = normalize(vec3<f32>(
        get_world_SDF_optimized(p_h + vec3<f32>(e, 0., 0.), inst, inv_model_mat, hit_tri_idx) - get_world_SDF_optimized(p_h - vec3<f32>(e, 0., 0.), inst, inv_model_mat, hit_tri_idx),
        get_world_SDF_optimized(p_h + vec3<f32>(0., e, 0.), inst, inv_model_mat, hit_tri_idx) - get_world_SDF_optimized(p_h - vec3<f32>(0., e, 0.), inst, inv_model_mat, hit_tri_idx),
        get_world_SDF_optimized(p_h + vec3<f32>(0., 0., e), inst, inv_model_mat, hit_tri_idx) - get_world_SDF_optimized(p_h - vec3<f32>(0., 0., e), inst, inv_model_mat, hit_tri_idx)
    ));

    let mix_factor = clamp(params.distort_strength * 2.0, 0.0, 0.5);
    let final_n = normalize(mix(world_n, g_n, mix_factor));

    let L = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diff = dot(final_n, L) * 0.5 + 0.5;

    let uv = tri_f.uv01.xy * bary.x + tri_f.uv01.zw * bary.y + tri_f.uv2.xy * bary.z;
    let tex_color = textureSample(t_material, s_material, uv);
    
    return vec4<f32>(tex_color.rgb * diff, tex_color.a);
}

// ============================================================
// ====================== Ap1 保守包络 (Envelope) ======================
// ============================================================

struct EnvelopeVertex {
    pos: vec4<f32>,    // xyz = 位置, w = padding
    normal: vec4<f32>, // xyz = 法线, w = padding
};

// Ap1 专用的绑定组 (Group 2)
@group(0) @binding(0) var<uniform> env_params: Params;
@group(0) @binding(1) var<storage, read> envelope_vertices: array<EnvelopeVertex>;
@group(0) @binding(2) var envelope_tex: texture_storage_2d<r32uint, write>;

// 清空包络纹理
@compute @workgroup_size(8, 8)
fn cs_clear_envelope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(envelope_tex);
    if (any(gid.xy >= dims)) { return; }
    textureStore(envelope_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(0u, 0u, 0u, 0u));
}

@compute @workgroup_size(64)
fn cs_ap1_generate_envelope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri_idx = gid.x;
    let total_v = arrayLength(&envelope_vertices);
    if (tri_idx * 3u + 2u >= total_v) { return; }

    let v0 = envelope_vertices[tri_idx * 3u + 0u];
    let v1 = envelope_vertices[tri_idx * 3u + 1u];
    let v2 = envelope_vertices[tri_idx * 3u + 2u];

    let dist = env_params.envelope_displacement;

    var min_s = vec2<f32>(f32(env_params.screen_width), f32(env_params.screen_height));
    var max_s = vec2<f32>(0.0, 0.0);
    var inside = false;

    // 手动处理 6 个极值点（展开循环以避免 WGSL 动态索引限制）
    // 顶点 0 的两个偏移方向
    {
        let p = v0.pos.xyz + v0.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }
    {
        let p = v0.pos.xyz - v0.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }
    // 顶点 1 的两个偏移方向
    {
        let p = v1.pos.xyz + v1.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }
    {
        let p = v1.pos.xyz - v1.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }
    // 顶点 2 的两个偏移方向
    {
        let p = v2.pos.xyz + v2.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }
    {
        let p = v2.pos.xyz - v2.normal.xyz * dist;
        let clip = env_params.prev_view_proj * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let screen = vec2<f32>(
                (ndc.x * 0.5 + 0.5) * f32(env_params.screen_width),
                (0.5 - ndc.y * 0.5) * f32(env_params.screen_height)
            );
            min_s = min(min_s, screen);
            max_s = max(max_s, screen);
            inside = true;
        }
    }

    if (!inside) { return; }

    // 保守填充：将该三角形覆盖的屏幕矩形全部标记为 1
    let x_start = u32(clamp(min_s.x - 2.0, 0.0, f32(env_params.screen_width - 1u)));
    let x_end   = u32(clamp(max_s.x + 2.0, 0.0, f32(env_params.screen_width - 1u)));
    let y_start = u32(clamp(min_s.y - 2.0, 0.0, f32(env_params.screen_height - 1u)));
    let y_end   = u32(clamp(max_s.y + 2.0, 0.0, f32(env_params.screen_height - 1u)));

    for (var y = y_start; y <= y_end; y++) {
        for (var x = x_start; x <= x_end; x++) {
            textureStore(envelope_tex, vec2<i32>(i32(x), i32(y)), vec4<u32>(1u, 0u, 0u, 0u));
        }
    }
}

// Ap1 渲染管线 — 使用全屏四边形，采样 envelope_tex 显示红色
struct EnvelopeVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_envelope(@builtin(vertex_index) idx: u32) -> EnvelopeVertexOutput {
    let x = f32(i32(idx) / 2) * 4.0 - 1.0;
    let y = f32(i32(idx) % 2) * 4.0 - 1.0;
    var out: EnvelopeVertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var<uniform> env_render_params: Params;
@group(0) @binding(3) var envelope_tex_ro: texture_2d<u32>;

@fragment
fn fs_envelope(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    if (env_render_params.show_envelope == 0u) { discard; }

    let x = u32(frag_coord.x);
    let y = u32(frag_coord.y);
    let covered = textureLoad(envelope_tex_ro, vec2<i32>(i32(x), i32(y)), 0).r;

    if (covered != 0u) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.3); // 半透明红
    }
    discard;
}

// ============================================================