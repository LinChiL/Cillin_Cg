@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> scene: Scene;
@group(0) @binding(2) var t_cem_data: texture_3d<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> palette: array<vec4<f32>>;
// 6. Tile 数据 (Stride 132 = 1 count + 3 pad + 128 indices)
@group(0) @binding(6) var<storage, read> tile_data: array<u32>;
const TILE_STRIDE: u32 = 132u;

const TILE_SIZE: u32 = 16;
const MAX_ENTITIES_PER_TILE: u32 = 128;

// 射线区间结构体
struct RayRange {
    t_in: f32,
    t_out: f32,
    e_idx: u32,
};

// 核心求交算法 (Ray-AABB Slab Method)
fn ray_aabb_intersection(ray_o: vec3<f32>, ray_dir: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> vec2<f32> {
    let inv_d = 1.0 / ray_dir;
    let t0 = (b_min - ray_o) * inv_d;
    let t1 = (b_max - ray_o) * inv_d;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let t_entry = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit = min(min(t_max.x, t_max.y), t_max.z);
    return vec2<f32>(t_entry, t_exit);
}

// 射线-AABB 求交：算出进入和离开的时间
fn intersect_aabb(ray_o: vec3<f32>, ray_d: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> vec2<f32> {
    let inv_d = 1.0 / ray_d;
    let t0 = (b_min - ray_o) * inv_d;
    let t1 = (b_max - ray_o) * inv_d;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_in = max(max(tmin.x, tmin.y), tmin.z);
    let t_out = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_in, t_out);
}

// shader.wgsl -> intersect_aabb 修正版
fn intersect_aabb_safe(ray_o: vec3<f32>, ray_d: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> vec2<f32> {
    // 关键修正 1：防止 ray_dir 为 0 导致的无穷大
    let safe_d = select(ray_d, vec3<f32>(0.00001), abs(ray_d) < vec3<f32>(0.00001));
    let inv_d = 1.0 / safe_d;
    let t0 = (b_min - ray_o) * inv_d;
    let t1 = (b_max - ray_o) * inv_d;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_in = max(max(tmin.x, tmin.y), tmin.z);
    let t_out = min(min(tmax.x, tmax.y), tmax.z);
    
    // 关键修正 2：如果射线就在盒子内部，t_in 会是负数
    // 我们返回一个带有微小"提前量"的 t_in
    return vec2<f32>(t_in - 0.05, t_out + 0.05);
}

// 2. 【生命感求交】：带安全余量的 AABB 检测
fn intersect_aabb_alive(ray_o: vec3<f32>, ray_d: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> vec2<f32> {
    let inv_d = 1.0 / (ray_d + vec3<f32>(1e-6));
    let t0 = (b_min - ray_o) * inv_d;
    let t1 = (b_max - ray_o) * inv_d;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    // 增加 0.02 的"苏醒余量"，防止射线出生在表面之后
    return vec2<f32>(max(max(t_min.x, t_min.y), t_min.z) - 0.02, min(min(t_max.x, t_max.y), t_max.z) + 0.02);
}

// 将候选名单扩大到 24 个以节省寄存器，提升 FPS
fn sort_ranges_robust(ranges: ptr<function, array<RayRange, 24>>, count: u32) {
    if (count <= 1u) { return; }
    for (var i = 1u; i < count; i++) {
        let key = (*ranges)[i];
        var j = i;
        // 增加安全上限判定
        while (j > 0u && (*ranges)[j - 1u].t_in > key.t_in) {
            (*ranges)[j] = (*ranges)[j - 1u];
            j--;
        }
        (*ranges)[j] = key;
    }
}

// 针对特定物体的精确采样
fn get_entity_dist_precise(world_p: vec3<f32>, e_idx: u32) -> f32 {
    let e = scene.entities[e_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let model_max_dim = max(dims.x, max(dims.y, dims.z));
    
    let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
    let voxel = sample_dna_smooth(uvw, e.sdf_index);
    
    // --- 核心修正：SDF 膨胀 ---
    // 减去 0.002 意味着我们将物体的“场”向外扩张了体素宽度的 1/8 左右
    // 这能有效抵消线性插值带来的“消融”效应
    return (voxel.dist - 0.002) * model_max_dim * e.instance_scale;
}

// 获取特定物体的 Voxel 数据
fn get_voxel_data_precise(world_p: vec3<f32>, e_idx: u32) -> Voxel {
    let e = scene.entities[e_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    
    let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
    var voxel = sample_dna_smooth(uvw, e.sdf_index);
    
    voxel.entity_id = e_idx;
    return voxel;
}

// 1. 核心修正：绝对平滑的 Tile 采样器
fn get_tile_sdf_safe(world_p: vec3<f32>, tile_base: u32, count: u32) -> Voxel {
    var min_v: Voxel;
    min_v.dist = 1000.0;
    min_v.entity_id = 0u;

    for (var j: u32 = 0u; j < count; j++) {
        let e_idx = tile_data[tile_base + 4u + j];
        let e = scene.entities[e_idx];
        let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
        let dims = e.aabb_max.xyz - e.aabb_min.xyz;
        let model_max_dim = max(dims.x, max(dims.y, dims.z));
        
        // 使用带符号的 AABB 距离 (内部为负)
        let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz) * e.instance_scale;

        // 采样 DNA（即使在 AABB 外也利用 clamp 保证连续性）
        let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
        let voxel = sample_dna_smooth(uvw, e.sdf_index);
        let d_dna = voxel.dist * model_max_dim * e.instance_scale;

        // --- 核心公式：max(d_box, d_dna) ---
        // 这是 SDF 交集的标准公式。
        // 在盒子外：d_box 占主导，提供远距离步进推力。
        // 在盒子边缘的空气区：d_box 为 0，d_dna 为 0.5，结果为 0.5。
        // 数值在跨越 AABB 表面时是完全连续的，阴影射线感知不到盒子的存在！
        let d_final = max(d_box, d_dna);

        if (abs(d_final) < abs(min_v.dist)) {
            min_v = voxel;
            min_v.dist = d_final;
            min_v.entity_id = e_idx;
        }
    }
    return min_v;
}

// 专为影子设计的极速采样（只针对特定物体，不循环）
fn get_entity_dist_shadow(p: vec3<f32>, e_idx: u32) -> f32 {
    let e = scene.entities[e_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(p, 1.0)).xyz;
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let voxel = sample_dna_smooth((local_p - e.aabb_min.xyz) / dims, e.sdf_index);
    return voxel.dist * max(dims.x, max(dims.y, dims.z)) * e.instance_scale;
}

fn calculate_luoer_tile_optimized(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, id_xy: vec2<u32>, tile_base: u32, count: u32, e_idx_hit: u32) -> f32 {
    let dither = interleavedGradientNoise(vec2<f32>(id_xy));
    
    // 1. 动态 Bias：基于模型缩放，解决影子偏移和漏光
    let e_hit = scene.entities[e_idx_hit];
    let dims_hit = e_hit.aabb_max.xyz - e_hit.aabb_min.xyz;
    let dynamic_bias = max(dims_hit.x, max(dims_hit.y, dims_hit.z)) * e_hit.instance_scale / 64.0 * 1.5;
    
    // 影子射线起点：利用法线和光源方向的合力纠偏
    let bias_dir = normalize(normal + L * 0.5);
    let shadow_o = p + bias_dir * dynamic_bias;
    
    var res: f32 = 1.0;
    let k: f32 = 16.0; // 软阴影系数：值越小影子越柔和

    for (var j: u32 = 0u; j < count; j++) {
        let e_idx = tile_data[tile_base + 4u + j];
        let e = scene.entities[e_idx];
        
        let r = intersect_aabb_safe((e.inv_model_matrix * vec4<f32>(shadow_o, 1.0)).xyz, (e.inv_model_matrix * vec4<f32>(L, 0.0)).xyz, e.aabb_min.xyz, e.aabb_max.xyz);
        
        // 只有物体在光照路径上才有资格贡献阴影
        if (r.y > 0.0 && r.x < 50.0) {
            // st 是相对于起跳点 shadow_o 的距离
            // 我们从 max(0.001, r.x) 开始，保证分母不为 0
            var st = max(0.001, r.x);
            let send = min(r.y, 50.0);
            
            // 如果是检测“自己”，起跳距离要翻倍，防止自遮挡
            if (e_idx == e_idx_hit) {
                st = max(st, dynamic_bias * 2.0);
            }

            var shadow_steps = 0u;
            while (st < send) {
                shadow_steps++;
                let sd = get_entity_dist_shadow(shadow_o + L * st, e_idx);
                
                // 核心判定：如果有任何一处 dist < 0，说明绝对遮挡
                if (sd < 0.0) { return 0.0; }
                
                // --- 软阴影公式：重点在于 st 的全局一致性 ---
                // 这里的 st 诚实地记录了从 shadow_o 到当前采样点的距离
                res = min(res, k * sd / st);
                
                // 步进：0.8 折扣保证精度，0.02 保底步长跨越台阶
                st += max(sd * 0.8, 0.02);
                
                // 优化：如果影子已经黑透了，或者跑了太远，提前下班
                if (res < 0.01 || st > 50.0 || shadow_steps > 64u) { break; }
            }
        }
    }
    // 最后的平滑过渡，消除硬边缘感
    return smoothstep(0.0, 1.0, res);
}

// 对应的法线函数也需要修改为针对特定物体
fn get_normal_fast(p: vec3<f32>, e_idx: u32) -> vec3<f32> {
    let h = 0.02; 
    let k = vec2<f32>(1.0, -1.0);
    // 直接复用你之前写的 get_entity_dist_precise
    return normalize(
        k.xyy * get_entity_dist_precise(p + k.xyy * h, e_idx) +
        k.yyx * get_entity_dist_precise(p + k.yyx * h, e_idx) +
        k.yxy * get_entity_dist_precise(p + k.yxy * h, e_idx) +
        k.xxx * get_entity_dist_precise(p + k.xxx * h, e_idx)
    );
}

struct Params {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,
    entity_count: u32,
    debug_mode: u32,
    _pad1: vec2<f32>,
    _pad2: array<vec4<f32>, 3>,
};

struct Entity {
    model_matrix: mat4x4<f32>,
    inv_model_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    sdf_index: u32,
    instance_scale: f32,
    _pad: array<u32, 18>,
};

struct Scene { entities: array<Entity, 64>, };

struct Voxel {
    dist: f32,
    color: vec3<f32>, // 修改：从 color_id 变为直接存储颜色
    normal: vec3<f32>, // 从 64-bit 数据中解码的法线
    modifier: f32,
    entity_id: u32, // 记录所属实体ID，用于法线隔离
};

fn unpack_voxel_safe(data_r: u32, data_g: u32) -> Voxel {
    var v: Voxel;
    
    // --- 1. 处理 R 通道 ---
    let r_val = data_r;
    let d_bits = (r_val >> 12u) & 0xFFFFFu; // 20bit
    var d_i32 = i32(d_bits);
    if (d_bits >= 524288u) { d_i32 -= 1048576i; } // 20bit 补码修正
    v.dist = f32(d_i32) / 524287.0;

    let col_id = r_val & 0xFFFu; // 12bit
    v.color = palette[col_id].rgb;

    // --- 2. 处理 G 通道 ---
    let g_val = data_g;
    // 解码法线 (10bit)
    v.normal = decode_normal_10bit(g_val >> 22u); 
    
    // 解码 Ko (8bit)
    v.modifier = f32((g_val >> 14u) & 0xFFu) / 255.0;

    return v;
}

// 八面体解码 10bit 版
fn decode_normal_10bit(packed: u32) -> vec3<f32> {
    let ux = f32((packed >> 5u) & 0x1Fu) / 31.0;
    let uy = f32(packed & 0x1Fu) / 31.0;
    var v = vec2<f32>(ux * 2.0 - 1.0, uy * 2.0 - 1.0);
    var n = vec3<f32>(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    if (n.z < 0.0) {
        let sign_v = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        v = (1.0 - abs(v.yx)) * sign_v;
        n = vec3<f32>(v.x, v.y, n.z);
    }
    return normalize(n);
}

fn sample_dna_smooth(uvw: vec3<f32>, sdf_index: u32) -> Voxel {
    let tex_coord = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)) * 63.0;
    let i0 = vec3<i32>(floor(tex_coord));
    let i1 = min(i0 + vec3<i32>(1), vec3<i32>(63));
    let f = fract(tex_coord);
    
    // 关键修正：深度偏移恢复为 64，不再需要 *2
    let z_off = i32(sdf_index) * 64;

    // 采样周围 8 个点，直接一次性拿到 R 和 G 通道
    let v000_raw = textureLoad(t_cem_data, vec3<i32>(i0.x, i0.y, i0.z + z_off), 0).xy;
    let v100_raw = textureLoad(t_cem_data, vec3<i32>(i1.x, i0.y, i0.z + z_off), 0).xy;
    let v010_raw = textureLoad(t_cem_data, vec3<i32>(i0.x, i1.y, i0.z + z_off), 0).xy;
    let v110_raw = textureLoad(t_cem_data, vec3<i32>(i1.x, i1.y, i0.z + z_off), 0).xy;
    let v001_raw = textureLoad(t_cem_data, vec3<i32>(i0.x, i0.y, i1.z + z_off), 0).xy;
    let v101_raw = textureLoad(t_cem_data, vec3<i32>(i1.x, i0.y, i1.z + z_off), 0).xy;
    let v011_raw = textureLoad(t_cem_data, vec3<i32>(i0.x, i1.y, i1.z + z_off), 0).xy;
    let v111_raw = textureLoad(t_cem_data, vec3<i32>(i1.x, i1.y, i1.z + z_off), 0).xy;

    let v000 = unpack_voxel_safe(v000_raw.x, v000_raw.y);
    let v100 = unpack_voxel_safe(v100_raw.x, v100_raw.y);
    let v010 = unpack_voxel_safe(v010_raw.x, v010_raw.y);
    let v110 = unpack_voxel_safe(v110_raw.x, v110_raw.y);
    let v001 = unpack_voxel_safe(v001_raw.x, v001_raw.y);
    let v101 = unpack_voxel_safe(v101_raw.x, v101_raw.y);
    let v011 = unpack_voxel_safe(v011_raw.x, v011_raw.y);
    let v111 = unpack_voxel_safe(v111_raw.x, v111_raw.y);

    var res: Voxel;
    // 距离插值
    res.dist = mix(mix(mix(v000.dist, v100.dist, f.x), mix(v010.dist, v110.dist, f.x), f.y),
                   mix(mix(v001.dist, v101.dist, f.x), mix(v011.dist, v111.dist, f.x), f.y), f.z);
    
    // 法线插值：这是 64 位重构最强的武器，必须保留！
    res.normal = normalize(mix(mix(mix(v000.normal, v100.normal, f.x), mix(v010.normal, v110.normal, f.x), f.y),
                               mix(mix(v001.normal, v101.normal, f.x), mix(v011.normal, v111.normal, f.x), f.y), f.z));
    
    res.color = v000.color;
    res.modifier = v000.modifier;
    res.entity_id = v000.entity_id;
    return res;
}

fn sdAABB(p: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> f32 {
    let center = (b_min + b_max) * 0.5;
    let half_extents = (b_max - b_min) * 0.5;
    let q = abs(p - center) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// 1. 修正：只返回外部距离的 AABB 函数
fn sdAABB_ext_only(p: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> f32 {
    let center = (b_min + b_max) * 0.5;
    let half_extents = (b_max - b_min) * 0.5;
    let q = abs(p - center) - half_extents;
    // 关键：外部返回正值，内部返回 0。这样包围盒就不会在内部产生实体感。
    return length(max(q, vec3<f32>(0.0)));
}

// 2. 核心修正：无缝衔接的场景采样
fn get_scene_sdf_safe(world_p: vec3<f32>) -> Voxel {
    var min_v: Voxel;
    min_v.dist = 1000.0;

    for (var i: u32 = 0u; i < params.entity_count; i++) {
        let e = scene.entities[i];
        let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
        let dims = e.aabb_max.xyz - e.aabb_min.xyz;
        let model_max_dim = max(dims.x, max(dims.y, dims.z));
        
        // 获取 AABB 的纯外部距离（盒子内永远是 0）
        let d_box_ext = sdAABB_ext_only(local_p, e.aabb_min.xyz, e.aabb_max.xyz) * e.instance_scale;

        // 性能过滤：如果离盒子非常远，直接返回盒子距离，跳过纹理采样
        if (d_box_ext > 1.0) {
            if (d_box_ext < abs(min_v.dist)) {
                min_v.dist = d_box_ext;
                min_v.entity_id = i;
            }
            continue;
        }

        // 采样 DNA（使用你代码里现成的五次插值 sample_dna_smooth）
        let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
        let voxel = sample_dna_smooth(uvw, e.sdf_index);
        
        // 还原物理距离（保留符号！！这是防止镂空的关键）
        let d_dna = voxel.dist * model_max_dim * e.instance_scale;

        // --- 终极公式：距离 = 外部盒体推力 + 内部细节 ---
        // 在盒子外：d_box_ext > 0，两者相加。
        // 在盒子内：d_box_ext = 0，距离完全由 d_dna 决定。
        // 因为相加操作是连续的，阴影射线在跨越边界时不会看到任何数值跳变！
        let d_final = d_box_ext + d_dna;

        // 比较绝对值找到最近表面，但保留符号返回以供命中判定
        if (abs(d_final) < abs(min_v.dist)) {
            min_v = voxel;
            min_v.dist = d_final;
            min_v.entity_id = i;
        }
    }
    return min_v;
}

fn get_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.015; // 采样步长
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * get_scene_sdf_safe(p + k.xyy * h).dist +
        k.yyx * get_scene_sdf_safe(p + k.yyx * h).dist +
        k.yxy * get_scene_sdf_safe(p + k.yxy * h).dist +
        k.xxx * get_scene_sdf_safe(p + k.xxx * h).dist
    );
}

fn get_normal_custom(p: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * get_scene_sdf_safe(p + k.xyy * h).dist +
        k.yyx * get_scene_sdf_safe(p + k.yyx * h).dist +
        k.yxy * get_scene_sdf_safe(p + k.yxy * h).dist +
        k.xxx * get_scene_sdf_safe(p + k.xxx * h).dist
    );
}

// 专门采样某个特定实体的距离场（法线计算专用，杜绝干扰）
fn get_entity_sdf_only(world_p: vec3<f32>, entity_idx: u32) -> f32 {
    let e = scene.entities[entity_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
    
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let model_world_size = max(dims.x, max(dims.y, dims.z));
    
    // UVW 映射
    let uvw = (local_p - e.aabb_min.xyz) / dims;
    
    // 即使在盒子外也返回插值后的 DNA 距离，保证导数连续
    let voxel = sample_dna_smooth(clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)), e.sdf_index);
    
    // 如果在盒子外面，我们要补上到盒子的欧几里得距离，确保法线方向正确
    let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz);
    
    // 核心公式：将 DNA 比例还原为世界空间距离
    return (voxel.dist * model_world_size + max(d_box, 0.0)) * e.instance_scale;
}

// 智能法线函数 - 将 entity ID 传进去，实现"采样锁定"
// 辅助函数：确保法线计算使用的公式与采样完全一致
fn get_entity_sdf_combined(world_p: vec3<f32>, e_idx: u32) -> f32 {
    let e = scene.entities[e_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let model_max_dim = max(dims.x, max(dims.y, dims.z));
    
    let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz);
    let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
    let voxel = sample_dna_smooth(uvw, e.sdf_index);
    
    // 必须使用与 get_scene_sdf_safe 完全一致的 max 逻辑
    return max(d_box, voxel.dist) * model_max_dim * e.instance_scale;
}

fn get_normal_isolated(p: vec3<f32>, entity_idx: u32) -> vec3<f32> {
    let e = scene.entities[entity_idx];
    
    // --- 核心修正 1：收缩步长 ---
    // 对于硬表面，h 必须极小，才能捕捉到锐利的转折
    // 0.01 左右能有效防止法线"跨越棱角"去偷看另一面的光
    let h = 0.01 * e.instance_scale;

    let k = vec2<f32>(1.0, -1.0);
    // 注意：这里移除 p_dithered，法线需要绝对的确定性
    return normalize(
        k.xyy * get_entity_sdf_combined(p + k.xyy * h, entity_idx) +
        k.yyx * get_entity_sdf_combined(p + k.yyx * h, entity_idx) +
        k.yxy * get_entity_sdf_combined(p + k.yxy * h, entity_idx) +
        k.xxx * get_entity_sdf_combined(p + k.xxx * h, entity_idx)
    );
}

fn get_entity_sdf_only_abs(world_p: vec3<f32>, entity_idx: u32) -> f32 {
    let e = scene.entities[entity_idx];
    let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let model_world_size = max(dims.x, max(dims.y, dims.z));
    let uvw = (local_p - e.aabb_min.xyz) / dims;
    
    let voxel = sample_dna_smooth(clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)), e.sdf_index);
    let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz);
    
    // 统一使用 abs 还原距离
    return (abs(voxel.dist) * model_world_size + max(d_box, 0.0)) * e.instance_scale;
}

fn calculate_luoer(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>) -> f32 {
    var t: f32 = 0.25; // 起跳偏置，防止自遮挡
    var res: f32 = 1.0;
    let k: f32 = 8.0;   // 软阴影硬度
    
    for (var i = 0; i < 24; i++) {
        let d = get_scene_sdf_safe(p + normal * 0.1 + L * t).dist;
        if (d < 0.001) { return 0.0; }
        res = min(res, k * d / t);
        t += max(d, 0.08);
        if (t > 15.0) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

// 1. 增加一个极其快速的伪随机函数（打碎干涉条纹）
fn interleavedGradientNoise(n: vec2<f32>) -> f32 {
    let f = fract(sin(dot(n, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    return f;
}

// 2. 改进版罗尔值计算（影子逻辑）
fn calculate_luoer_smooth(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, id_xy: vec2<u32>) -> f32 {
    // 关键修正 1：利用像素坐标产生抖动偏移
    let dither = interleavedGradientNoise(vec2<f32>(id_xy));
    
    // 关键修正 2：初始偏移量。从 0.1 增加到 0.2，并加上随机抖动
    // 这样每一根阴影射线的起始点都不一样，彻底打破相干性
    var t: f32 = 0.15 + 0.05 * dither;
    
    var res: f32 = 1.0;
    let k: f32 = 16.0;   // 影子硬度
    
    for (var i = 0; i < 32; i++) {
        // 关键修正 3：计算采样位置时，沿法线方向再推出一点点 (normal * 0.05)
        // 这能帮射线逃离 12bit 数据在表面产生的"底噪区"
        let sample_p = p + normal * 0.04 + L * t;
        let d = get_scene_sdf_safe(sample_p).dist;
        
        // 撞击判定
        if (d < 0.001) { return 0.0; }
        
        // 软阴影核心公式
        res = min(res, k * d / t);
        
        // 关键修正 4：步长优化
        // 这里的步长不能太大也不能太小，max(d, 0.015) 刚好约等于一个体素的尺寸
        t += max(d, 0.02);
        
        if (t > 15.0 || res < 0.01) { break; }
    }
    
    // 让影子的边缘过渡更符合视觉习惯
    return smoothstep(0.0, 1.0, res);
}

// 3. 高精度薄片阴影函数（针对薄几何体优化）
fn calculate_luoer_precision(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, id_xy: vec2<u32>) -> f32 {
    let dither = interleavedGradientNoise(vec2<f32>(id_xy));
    
    // 核心修正：沿法线推开一个显著的距离（Bias）
    // 0.1 是一个安全值，它能让阴影射线彻底飞离"台阶区"
    var t: f32 = 0.1 + 0.05 * dither;
    
    var res: f32 = 1.0;
    let k: f32 = 16.0; // 降低硬度，让阴影边缘更自然
    
    for (var i = 0; i < 32; i++) {
        // 采样点沿法线再偏移一丁点
        let sample_p = p + normal * 0.05 + L * t;
        let d = get_scene_sdf_safe(sample_p).dist;
        
        // 如果 d 为负，说明绝对撞上了
        if (d < 0.0) { return 0.0; }
        if (d < 0.001) { return 0.0; }
        
        res = min(res, k * d / t);
        
        // 强制最小步进，防止卡在表面
        t += max(d, 0.02);
        
        if (t > 20.0 || res < 0.01) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

// 影子函数现在需要传入 tile 信息，否则它依然是性能黑洞
fn calculate_luoer_tile(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, id_xy: vec2<u32>, base_addr: u32, count: u32) -> f32 {
    let dither = interleavedGradientNoise(vec2<f32>(id_xy));
    var t: f32 = 0.1 + 0.05 * dither;
    var res: f32 = 1.0;
    let k: f32 = 16.0;
    
    for (var i = 0; i < 24; i++) { // 影子步数可以稍微减少
        let sample_p = p + normal * 0.05 + L * t;
        
        // 关键：影子也只查当前 Tile 里的物体！
        let voxel = get_tile_sdf_safe(sample_p, base_addr, count);
        let d = voxel.dist;
        
        if (d < 0.001) { return 0.0; }
        res = min(res, k * d / t);
        t += max(d, 0.02);
        if (t > 15.0 || res < 0.01) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

// 辅助颜色转换，用于观察负值
fn debug_color(val: f32) -> vec3<f32> {
    if (val > 0.0) { return vec3<f32>(val, 0.0, 0.0); } // 正值显红色
    return vec3<f32>(0.0, abs(val), 0.0);             // 负值显绿色
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(output_texture);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // 1. 初始化射线
    let screen_uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);
    let uv = vec2<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0);
    let target_pos = params.proj_inv * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(target_pos.xyz / target_pos.w), 0.0)).xyz);
    let ray_p = params.cam_pos.xyz;

    // 1. 定位当前像素的竞技场 (Tile)
    let tiles_x = (screen_size.x + 15u) / 16u;
    let tile_idx = (id.y / 16u) * tiles_x + (id.x / 16u);
    let base_addr = tile_idx * TILE_STRIDE;
    let entity_count = min(tile_data[base_addr], 128u);
    
    // --- 模式 6: 分片热力图可视化 ---
    if (params.debug_mode == 6u) {
        let count = tile_data[base_addr];

        // 1. 绘制 Tile 边界线 (16x16 像素的网格)
        let is_border = (id.x % 16u == 0u) || (id.y % 16u == 0u);
        
        // 2. 根据物体数量生成颜色 (热力图)
        // 0个: 蓝色, 1-10个: 绿色, 11-32个: 黄色, 33-64个: 红色
        var col = vec3<f32>(0.0, 0.0, 0.2); // 默认深蓝 (空)
        if (count > 0u) {
            let t = f32(count) / 128.0; // 归一化到 0-1
            if (t < 0.2) {
                col = mix(vec3<f32>(0.0, 0.1, 0.5), vec3<f32>(0.0, 1.0, 0.0), t * 5.0);
            } else if (t < 0.5) {
                col = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.2) * 3.33);
            } else {
                col = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
            }
        }

        // 3. 叠加边界线
        if (is_border) {
            textureStore(output_texture, id.xy, vec4<f32>(1.0, 1.0, 1.0, 0.5)); // 白色网格
        } else {
            textureStore(output_texture, id.xy, vec4<f32>(col, 1.0));
        }
        return; // 立即返回，不进行后续渲染
    }

    // 2. 【候选人海选】：只看射线路径上的物体
    var ranges: array<RayRange, 24>; // 减少到 24 以节省寄存器，提升 FPS
    var cand_count = 0u;
    for (var j = 0u; j < entity_count; j++) {
        let e_idx = tile_data[base_addr + 4u + j];
        let e = scene.entities[e_idx];
        let r = intersect_aabb_alive((e.inv_model_matrix * vec4<f32>(ray_p, 1.0)).xyz, (e.inv_model_matrix * vec4<f32>(ray_dir, 0.0)).xyz, e.aabb_min.xyz, e.aabb_max.xyz);
        if (r.y > 0.0 && r.x < r.y) {
            ranges[cand_count] = RayRange(max(r.x, 0.0), r.y, e_idx);
            cand_count++; if (cand_count >= 24u) { break; }
        }
    }

    // 3. 【竞技场排序】：按深度排队
    sort_ranges_robust(&ranges, cand_count);

    // 4. 【深度优先步进】：遮挡剔除的灵魂
    var t_hit_limit = 1000.0;
    var hit = false;
    var best_entity_id = 0u;

    for (var i = 0u; i < cand_count; i++) {
        let range = ranges[i];
        
        // --- 核心"自觉"逻辑：如果当前盒子入口比已知撞击点还远，后面的直接物理蒸发 ---
        if (range.t_in > t_hit_limit) { break; }

        var t_march = range.t_in;
        let march_end = min(range.t_out, t_hit_limit);
        var steps = 0u;

        while (t_march < march_end) {
            steps++;
            let d = get_entity_dist_precise(ray_p + ray_dir * t_march, range.e_idx);
            
            // 远距离精度补偿：让判定更灵敏
            let threshold = 0.001 + t_march * 0.00002;

            if (d < threshold) {
                if (t_march < t_hit_limit) {
                    t_hit_limit = t_march;
                    hit = true;
                    best_entity_id = range.e_idx;
                }
                break; // 撞到了，跳出当前物体的 march，但可能还要检查重叠的其他物体
            }

            // --- 核心修正：步长压缩 ---
            // 当距离小于 0.05（接近表面）时，强制使用更小的步长系数（0.5 而不是 0.8）
            // 防止光线跳过细杆子
            let step_factor = select(0.8, 0.5, d < 0.05);
            t_march += max(d * step_factor, 0.0008);
            
            if (steps > 160u) { break; }
        }
    }

    if (hit) {
        // --- 最终采样：确保使用撞击时的 ID 和坐标 ---
        let hit_p = ray_p + ray_dir * t_hit_limit;
        let final_v = get_voxel_data_precise(hit_p, best_entity_id);
        
        // --- 性能飞跃：直接转换法线到世界空间 ---
        // 不再需要采样 4 次 SDF，只需要 1 次矩阵变换
        let e = scene.entities[best_entity_id];
        let world_normal = normalize((e.model_matrix * vec4<f32>(final_v.normal, 0.0)).xyz);
        
        let L = normalize(params.light_dir.xyz);
        
        // 影子也必须限制在名单内
        let luoer = calculate_luoer_tile_optimized(hit_p, world_normal, L, id.xy, base_addr, entity_count, best_entity_id);
        
        // 最终光照：0.32 是环境底色。只有当 diffuse > 0 且没被影子遮挡时，光照才会提升
        let lighting = mix(0.32, 1.0, luoer * max(0.0, dot(world_normal, L)));
        
        // 调试模式
        if (params.debug_mode == 1u) {
            // 1: 距离场(红正绿负)
            let sdf_color = select(
                vec3<f32>(final_v.dist, 0.0, 0.0),  // 正值：红色
                vec3<f32>(0.0, -final_v.dist, 0.0), // 负值：绿色
                final_v.dist < 0.0
            );
            textureStore(output_texture, id.xy, vec4<f32>(sdf_color, 1.0));
        } else if (params.debug_mode == 2u) {
            // 2: 坐标映射
            let e = scene.entities[best_entity_id];
            let local_p = (e.inv_model_matrix * vec4<f32>(hit_p, 1.0)).xyz;
            let uvw = (local_p - e.aabb_min.xyz) / (e.aabb_max.xyz - e.aabb_min.xyz);
            textureStore(output_texture, id.xy, vec4<f32>(uvw, 1.0));
        } else if (params.debug_mode == 3u) {
            // 3: 法线
            let normal_color = (world_normal + 1.0) * 0.5; // 将法线从 [-1,1] 映射到 [0,1]
            textureStore(output_texture, id.xy, vec4<f32>(normal_color, 1.0));
        } else if (params.debug_mode == 4u) {
            // 4: 原始颜色 (乘以微调整值)
            textureStore(output_texture, id.xy, vec4<f32>(final_v.color * final_v.modifier, 1.0));
        } else if (params.debug_mode == 5u) {
            // 5: 距离场扫描(红内蓝外)
            let sdf_color = select(
                vec3<f32>(0.0, 0.0, 1.0), // 外部：蓝色
                vec3<f32>(1.0, 0.0, 0.0), // 内部：红色
                final_v.dist < 0.0
            );
            textureStore(output_texture, id.xy, vec4<f32>(sdf_color, 1.0));
        } else {
            // 0: 关，正常渲染
            textureStore(output_texture, id.xy, vec4<f32>(final_v.color * final_v.modifier * lighting, 1.0));
        }
    } else {
        textureStore(output_texture, id.xy, vec4<f32>(0.05, 0.05, 0.1, 1.0));
    }
}

// 展示部分保持不动...
@group(0) @binding(0) var t_screen_read: texture_2d<f32>; 

@vertex
fn vs_blit(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let pos_x = select(select(-1.0, 3.0, idx == 1u), -1.0, idx == 2u);
    let pos_y = select(select(-1.0, -1.0, idx == 1u), 3.0, idx == 2u);
    return vec4<f32>(pos_x, pos_y, 0.0, 1.0);
}

@fragment
fn fs_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(t_screen_read, vec2<i32>(pos.xy), 0);
}