struct Primitive {
    inv_model_matrix: mat4x4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
};

struct Anchor {
    pos: vec4<f32>,
    offset_attr: vec4<f32>,
};

struct GridCell {
    offset: u32,
    count: u32,
};

struct Params {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>, // 重投影矩阵
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,
    
    // 数据包 A
    prim_count: u32,
    anchor_count: u32,
    scaffold_count: u32,
    is_moving: u32,
    
    // --- 新增：空间网格参数 ---
    grid_origin: vec4<f32>, // 网格左下角起点 [x, y, z, cell_size]
    
    // 数据包 B
    time: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,

    // 最终填充
    _final_padding: array<vec4<f32>, 4>,

    // 模型几何中心 (用于径向位移场)
    model_center: vec4<f32>,
};

// 1. Compute 阶段使用的声明 (Group 0)
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(3) var<storage, read> anchors: array<Anchor>;
@group(0) @binding(4) var<storage, read> grid_data: array<GridCell>;
@group(0) @binding(5) var<storage, read> scaffold: array<vec4<f32>>;

// 2. Render 阶段使用的声明 (注意：我们让它也用 Group 0，因为它们在不同的 Pass 运行)
@group(0) @binding(0) var t_read: texture_2d<f32>;

// 平滑并集 (Smooth Union)
fn smin(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// 场景 SDF 采样 (基础层)
fn get_base_sdf(p: vec3<f32>) -> vec4<f32> {
    var min_d = 1000.0;
    var col = vec3<f32>(0.8);

    for (var i = 0u; i < params.prim_count; i = i + 1u) {
        let prim = primitives[i];
        let local_p = (prim.inv_model_matrix * vec4<f32>(p, 1.0)).xyz;

        var d = 1000.0;
        let type_id = u32(prim.params.w);

        if (type_id == 0u) { // 球体
            d = length(local_p) - prim.params.x;
        } else if (type_id == 1u) { // 圆角立方体
            let q = abs(local_p) - vec3<f32>(prim.params.x);
            d = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - prim.params.y;
        }

        let k = prim.params.z;
        if (i == 0u) {
            min_d = d;
            col = prim.color.rgb;
        } else {
            let h = clamp(0.5 + 0.5 * (min_d - d) / k, 0.0, 1.0);
            min_d = smin(min_d, d, k);
            col = mix(col, prim.color.rgb, h);
        }
    }
    return vec4<f32>(col, min_d);
}

// 修正后的 RBF 权重函数
fn get_rbf_weight(dist: f32, radius: f32) -> f32 {
    let x = clamp(dist / radius, 0.0, 1.0);
    // 使用更高阶的平滑多项式，彻底消除"阶梯感"
    return (1.0 - x * x) * (1.0 - x * x);
}

// 从空间网格中获取细节偏移
fn get_detail_offset(p: vec3<f32>) -> f32 {
    let origin = params.grid_origin.xyz;
    let cell_len = params.grid_origin.w;
    
    // 1. 定位当前格子坐标
    let g = vec3<i32>(floor((p - origin) / cell_len));
    
    if (any(g < vec3<i32>(0)) || any(g >= vec3<i32>(16))) { return 0.0; }
    
    var sum_off = 0.0;
    var sum_w = 0.0;

    // 2. 采样当前格子和周围 8 个邻居格子
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbor_g = g + vec3<i32>(dx, dy, dz);
                if (all(neighbor_g >= vec3<i32>(0)) && all(neighbor_g < vec3<i32>(16))) {
                    let cell_idx = u32(neighbor_g.z * 256 + neighbor_g.y * 16 + neighbor_g.x);
                    let cell = grid_data[cell_idx];

                    for (var i = 0u; i < cell.count; i = i + 1u) {
                        let a = anchors[cell.offset + i];
                        let d = length(p - a.pos.xyz);
                        let r = a.pos.w;
                        
                        if (d < r) {
                            let w = exp(-5.0 * (d/r) * (d/r)); // 高斯平滑
                            sum_off = sum_off + a.offset_attr.x * w;
                            sum_w = sum_w + w;
                        }
                    }
                }
            }
        }
    }
    
    return select(0.0, sum_off / max(sum_w, 0.001), sum_w > 0.0);
}

fn get_sdf(p: vec3<f32>) -> vec4<f32> {
    let base_res = get_base_sdf(p);
    let d_base = base_res.w;

    // --- 极致优化 1：洋葱皮剪裁 ---
    // 只有当射线距离基座表面 0.15 单位内，才开启锚点计算
    if (abs(d_base) > 0.15) { return base_res; }

    // --- 极致优化 2：单格快速查表 ---
    // 32x32x32 网格，只查 1 个格子
    let g = vec3<i32>(floor((p - params.grid_origin.xyz) / params.grid_origin.w));
    let cell_idx = u32(g.z * 1024 + g.y * 32 + g.x);
    let cell = grid_data[cell_idx];

    var weighted_offset = 0.0;
    var total_w = 0.0;

    // 此时 cell.count 应该被控制在 8-16 个点以内
    for (var i = 0u; i < cell.count; i = i + 1u) {
        let a = anchors[cell.offset + i];
        let diff = p - a.pos.xyz;
        let d = length(diff);
        let r = a.pos.w;

        if (d < r) {
            // --- 核心精度修复：4次幂核函数 ---
            let weight = pow(1.0 - d / r, 4.0); // 更锐利的权重，增加细节
            weighted_offset += a.offset_attr.x * weight;
            total_w += weight;
        }
    }

    // --- 归一化蒙皮 ---
    var displacement = 0.0;
    if (total_w > 0.0) {
        displacement = weighted_offset / total_w;
    }

    // --- 极致精度补丁：Lipschitz 纠偏 ---
    // 强行限制位移场的影响范围，防止梯度溢出导致的缩放异常
    let influence = smoothstep(0.15, 0.0, abs(d_base));
    let d_final = d_base + (displacement * influence);

    return vec4<f32>(base_res.rgb, d_final);
}

// 高精度四面体法线 (Tetrahedron Normal)
fn get_normal(p: vec3<f32>) -> vec3<f32> {
    // 关键：对于解析几何，步长 h 设置为 0.005 能捕捉到极致的边缘锐度
    let h = 0.005;
    let k = vec4<f32>(1.0, -1.0, -1.0, 1.0);
    return normalize(
        k.xyy * get_sdf(p + k.xyy * h).w +
        k.yyx * get_sdf(p + k.yyx * h).w +
        k.yxy * get_sdf(p + k.yxy * h).w +
        k.xxx * get_sdf(p + k.xxx * h).w
    );
}



@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // 关键修正 1：显式处理 textureDimensions 的返回值
    let size_u = textureDimensions(output_texture);
    let screen_size = vec2<f32>(f32(size_u.x), f32(size_u.y));

    if (f32(id.x) >= screen_size.x || f32(id.y) >= screen_size.y) { return; }

    // 2. 初始化射线 (使用更稳健的坐标转换)
    let screen_pos = vec2<f32>(f32(id.x), f32(id.y));
    let uv = (screen_pos / screen_size) * 2.0 - 1.0;

    // 注意：翻转 Y 轴以对齐 WGPU 坐标系
    let ray_target = params.proj_inv * vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(ray_target.xyz / ray_target.w), 0.0)).xyz);
    let ray_o = params.cam_pos.xyz;

    // 2. 初始背景：深色工业感背景
    var final_col = vec3<f32>(0.1, 0.1, 0.12);
    let t_grid = -ray_o.y / (ray_dir.y + 0.00001);
    if (t_grid > 0.0 && t_grid < 100.0) {
        let p = ray_o + ray_dir * t_grid;
        let grid_uv = abs(fract(p.xz - 0.5) - 0.5);
        let grid = smoothstep(0.05, 0.0, grid_uv.x) + smoothstep(0.05, 0.0, grid_uv.y);
        final_col = mix(final_col, vec3<f32>(0.2, 0.2, 0.25), grid);
    }



    // 3. 步进 (Raymarching)
    var t = 0.0;
    var hit = false;
    var res = vec4<f32>(0.0);
    for (var i = 0u; i < 200u; i = i + 1u) { // 步数加满
        let p = ray_o + ray_dir * t;
        res = get_sdf(p);
        let d = res.w;

        if (abs(d) < 0.0001) { hit = true; break; }

        // --- 核心性能修复：双速步进 ---
        var step_dist = d;
        if (abs(d) > 0.1) {
            step_dist = d * 0.9; // 远距离大跳
        } else {
            step_dist = d * 0.35; // 表面附近细查细节
        }

        t = t + step_dist;
        if (t > 50.0) { break; }
    }

    // 3. 最终混合渲染
    if (hit) {
        // 渲染实体模型 (带摄影棚光照)
        let p = ray_o + ray_dir * t;
        let n = get_normal(p);
        let base_color = res.rgb;

        // --- 摄影棚级光照逻辑 (Snapshot Material) ---
        let L_key = normalize(vec3<f32>(0.5, 1.0, 0.5));   // 主光
        let L_back = normalize(vec3<f32>(0.0, -0.5, -1.0)); // 轮廓背光

        // A. 粘土感漫反射
        let key_diff = max(dot(n, L_key), 0.0);
        
        // B. 边缘高光 (Rim Light)：突出模型的体积感
        let rim = pow(1.0 - max(dot(n, -ray_dir), 0.0), 4.0);
        let rim_light = rim * max(dot(n, L_back), 0.0) * 0.4;

        // C. 高亮反射 (Specular)：模拟抛光效果
        let h_vec = normalize(L_key - ray_dir);
        let specular = pow(max(dot(n, h_vec), 0.0), 32.0) * 0.3;

        // D. 菲涅尔环境光：让暗部不至于死黑，且带有空气感
        let ambient = (n.y * 0.5 + 0.5) * 0.15;

        // 最终合成
        var col = base_color * (key_diff + ambient);
        col = col + (rim_light * vec3<f32>(0.7, 0.8, 1.0)); // 蓝色冷调背光
        col = col + specular;

        // E. 调色板映射与 Gamma 修正
        col = col / (col + vec3<f32>(1.0)); // 简单的 ToneMapping
        final_col = pow(col, vec3<f32>(1.0 / 2.2));
    }

    // 关键修正 2：确保 textureStore 参数符合 2D Storage Texture 规范 (3个参数)
    // 1. Texture, 2. vec2<i32> 坐标, 3. vec4 值
    textureStore(output_texture, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(final_col, 1.0));
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