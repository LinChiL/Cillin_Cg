@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> scene: Scene;
@group(0) @binding(2) var t_cem_data: texture_3d<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> palette: array<vec4<f32>>;
// 6. Tile 数据 (Stride 132 = 1 count + 3 pad + 128 indices)
@group(0) @binding(6) var<storage, read> tile_data: array<u32>;
@group(0) @binding(7) var t_history: texture_2d<f32>;
@group(0) @binding(8) var s_linear: sampler;
const TILE_STRIDE: u32 = 132u;
const TILE_SIZE: u32 = 16u;
const MAX_ENTITIES_PER_TILE: u32 = 128u;
const GROUND_Y: f32 = 0.0;
const FOG_DIST: f32 = 1000.0;
const GROUND_ENTITY_ID: u32 = 0xFFFFFFFFu;

// --- 核心重构：深度感知重投影 ---
fn sample_history_reprojected(id: vec2<u32>, screen_size: vec2<u32>) -> vec4<f32> {
    let uv = vec2<f32>(id) / vec2<f32>(screen_size);
    
    // 1. 获取上一帧在这个位置的真实深度
    let history_raw = textureLoad(t_history, vec2<i32>(id.xy), 0);
    let depth = history_raw.a; // 我们在上一帧存的 t_hit
    
    // 如果深度太小（接近0）或太大（天空），深度感知会失效，回归默认值
    var safe_depth = depth;
    if (depth <= 0.0) { safe_depth = 1000.0; }

    // 2. 利用真实深度还原世界坐标
    // 计算当前像素的射线方向
    let screen_uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);
    let uv_ndc = vec2<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0);
    let target_pos = params.proj_inv * vec4<f32>(uv_ndc.x, uv_ndc.y, 1.0, 1.0);
    let current_ray_dir = normalize((params.view_inv * vec4<f32>(normalize(target_pos.xyz / target_pos.w), 0.0)).xyz);
    
    // 核心公式：世界坐标 = 相机位置 + 射线方向 * 真实深度
    let world_p = params.cam_pos.xyz + current_ray_dir * safe_depth;

    // 3. 投射回上一帧
    let prev_clip = params.prev_view_proj * vec4<f32>(world_p, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2<f32>((prev_ndc.x + 1.0) * 0.5, (1.0 - prev_ndc.y) * 0.5);

    // 边缘安全检查
    if (any(prev_uv < vec2<f32>(0.01)) || any(prev_uv > vec2<f32>(0.99))) {
        return vec4<f32>(0.0);
    }

    // 获取颜色，同时保留深度供下一帧使用
    let history_color = textureSampleLevel(t_history, s_linear, prev_uv, 0.0);
    return vec4<f32>(history_color.rgb, depth);
}

// 基础哈希函数
fn hash(n: vec2<f32>) -> f32 {
    return fract(sin(dot(n, vec2<f32>(12.9898, 4.1414))) * 43758.5453);
}

// 平滑噪声
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x), 
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// 4层分形布朗运动 (fBm)
fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 4; i++) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

// 极简版噪声，只采样两次，不跑循环
fn simple_mist(p: vec2<f32>, t: f32) -> f32 {
    // 第一层：大尺度慢速漂移
    let p1 = p * 0.2 + vec2<f32>(t * 0.02, t * 0.01);
    // 第二层：小尺度快速微调
    let p2 = p * 0.5 - vec2<f32>(t * 0.04, t * -0.02);
    
    // 直接取均值，产生自然的干涉效果
    return (noise(p1) + noise(p2)) * 0.5;
}

// 射线区间结构体
struct RayRange {
    t_in: f32,
    t_out: f32,
    e_idx: u32,
};

// 统一步进命中结果结构体
struct SceneHit {
    dist: f32,
    entity_id: u32,
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

// 2. 【鲁棒 AABB 求交】：使用 Robust Slab Method 解决拉伸问题
fn intersect_aabb_robust(ray_o: vec3<f32>, ray_d: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> vec2<f32> {
    let inv_d = 1.0 / (ray_d + select(vec3<f32>(1e-9), vec3<f32>(-1e-9), ray_d < vec3<f32>(0.0)));
    let t0 = (b_min - ray_o) * inv_d;
    let t1 = (b_max - ray_o) * inv_d;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    
    let t_entry = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit = min(min(t_max.x, t_max.y), t_max.z);
    
    // 返回：x 为进入距离，y 为退出距离
    // 如果 t_entry > t_exit，说明射线没撞到 AABB
    return vec2<f32>(t_entry, t_exit);
}

// 将候选名单扩大到 12 个以节省寄存器，提升 FPS
fn sort_ranges_robust(ranges: ptr<function, array<RayRange, 12>>, count: u32) {
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
    // 减去 0.005 意味着我们将物体的“场”向外扩张，以抵消线性插值带来的“消融”效应
    // 这对细杆子特别重要，可以防止它们在插值过程中被淡化
    return (voxel.dist - 0.01) * model_max_dim * e.instance_scale;
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

fn get_tile_sdf_dist_only(world_p: vec3<f32>, tile_base: u32, count: u32) -> SceneHit {
    var hit: SceneHit;
    hit.dist = 1000.0;
    hit.entity_id = 0u;

    for (var j: u32 = 0u; j < count; j++) {
        let e_idx = tile_data[tile_base + 4u + j];
        let e = scene.entities[e_idx];
        let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
        let dims = e.aabb_max.xyz - e.aabb_min.xyz;
        let model_max_dim = max(dims.x, max(dims.y, dims.z));

        let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz) * e.instance_scale;
        let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
        let d_dna_raw = sample_dna_dist_only(uvw, e.sdf_index);
        let d_dna = d_dna_raw * model_max_dim * e.instance_scale;

        let d_final = max(d_box, d_dna);

        if (d_final < hit.dist) {
            hit.dist = d_final;
            hit.entity_id = e_idx;
        }
    }
    return hit;
}

// 专为影子设计的极速采样（局部空间版，带三线性插值）
fn get_entity_dist_shadow_local_smooth(local_p: vec3<f32>, e_idx: u32) -> f32 {
    let e = scene.entities[e_idx];
    let dims = e.aabb_max.xyz - e.aabb_min.xyz;
    let uvw = clamp((local_p - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
    let d_raw = sample_dna_dist_only(uvw, e.sdf_index);
    return (d_raw - 0.008) * max(dims.x, max(dims.y, dims.z)) * e.instance_scale;
}

fn calculate_luoer_tile_optimized(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, id_xy: vec2<u32>, tile_base: u32, count: u32, e_idx_hit: u32) -> f32 {
    let dither = interleavedGradientNoise(vec2<f32>(id_xy));

    if (e_idx_hit == GROUND_ENTITY_ID) {
        var res: f32 = 1.0;
        let k: f32 = 16.0;

        for (var j: u32 = 0u; j < count; j++) {
            if (res < 0.01) { break; }

            let e_idx = tile_data[tile_base + 4u + j];
            let e = scene.entities[e_idx];

            let r = intersect_aabb_safe((e.inv_model_matrix * vec4<f32>(p, 1.0)).xyz, (e.inv_model_matrix * vec4<f32>(L, 0.0)).xyz, e.aabb_min.xyz, e.aabb_max.xyz);

            if (r.y > 0.0 && r.x < 100.0) {
                var st = max(0.001, r.x) + dither * 0.02;
                let send = min(r.y, 100.0);

                var shadow_steps = 0u;
                let local_shadow_o = (e.inv_model_matrix * vec4<f32>(p, 1.0)).xyz;
                let local_L = (e.inv_model_matrix * vec4<f32>(L, 0.0)).xyz;
                while (st < send) {
                    if (st >= send) { break; }

                    shadow_steps++;
                    let local_p = local_shadow_o + local_L * st;
                    let sd = get_entity_dist_shadow_local_smooth(local_p, e_idx);

                    if (sd < 0.0) {
                        res = 0.0;
                        break;
                    }

                    res = min(res, 12.0 * sd / max(st, 0.05));
                    st += max(sd * 0.95, 0.02);

                    if (res < 0.01 || shadow_steps > 48u) { break; }
                }
            }
        }
        return smoothstep(0.0, 1.0, res);
    }

    let e_hit = scene.entities[e_idx_hit];
    let dims_hit = e_hit.aabb_max.xyz - e_hit.aabb_min.xyz;
    let dynamic_bias = max(dims_hit.x, max(dims_hit.y, dims_hit.z)) * e_hit.instance_scale / 64.0 * 1.5;

    let bias_dir = normalize(normal + L * 0.02);
    let shadow_o = p + bias_dir * dynamic_bias * 0.05;

    var res: f32 = 1.0;
    let k: f32 = 16.0;

    for (var j: u32 = 0u; j < count; j++) {
        // --- 核心优化 A：遮挡早停 ---
        // 如果当前累积的影子已经黑透了（res 极小），直接下班！
        // 这样即使 Tile 里有 128 个物体，如果前两个就把光挡了，后面 126 个就不算了。
        if (res < 0.01) { break; }

        let e_idx = tile_data[tile_base + 4u + j];
        let e = scene.entities[e_idx];
        
        let r = intersect_aabb_safe((e.inv_model_matrix * vec4<f32>(shadow_o, 1.0)).xyz, (e.inv_model_matrix * vec4<f32>(L, 0.0)).xyz, e.aabb_min.xyz, e.aabb_max.xyz);
        
        // 只要光线路径穿过 AABB，就去算，不再做距离限制
        if (r.y > 0.0 && r.x < 100.0) { // 100.0 是一个足够大的世界半径
            var st = max(0.001, r.x) + dither * 0.02;
            let send = min(r.y, 100.0);
            
            // 如果是检测"自己"，起跳距离要翻倍，防止自遮挡
            if (e_idx == e_idx_hit) {
                st = max(st, dynamic_bias * 2.5);
            }

            var shadow_steps = 0u;
            let local_shadow_o = (e.inv_model_matrix * vec4<f32>(shadow_o, 1.0)).xyz;
            let local_L = (e.inv_model_matrix * vec4<f32>(L, 0.0)).xyz;
            while (st < send) {
                if (st >= send) { break; }

                shadow_steps++;
                let local_p = local_shadow_o + local_L * st;
                let sd = get_entity_dist_shadow_local_smooth(local_p, e_idx);
                
                // 核心判定：如果有任何一处 dist < 0，说明绝对遮挡
                if (sd < 0.0) {
                    res = 0.0; // 彻底遮挡
                    break;
                }
                
                // --- 软阴影公式：重点在于 st 的全局一致性 ---
                // 这里的 st 诚实地记录了从 shadow_o 到当前采样点的距离
                res = min(res, 12.0 * sd / max(st, 0.05));
                
                // --- 核心优化 B：自适应步进 ---
                // 影子步进可以稍微大胆一点，用 0.95 的折扣，加快穿越空旷区的速度
                st += max(sd * 0.95, 0.02);
                
                // 如果这个物体已经把光贡献得差不多了，跳出这个物体的 march
                if (res < 0.01 || shadow_steps > 48u) { break; }
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
    view_inv: mat4x4<f32>,       // 64
    proj_inv: mat4x4<f32>,       // 64
    prev_view_proj: mat4x4<f32>, // 64
    
    cam_pos: vec4<f32>,          // 16
    light_dir: vec4<f32>,        // 16
    
    // 逻辑数据包 A
    entity_count: u32,
    debug_mode: u32,
    time: f32,
    frame_index: u32,            // 这四个加起来刚好 16 字节
    
    // 逻辑数据包 B
    is_moving: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,                  // 这四个也加起来 16 字节
    
    _final_padding: array<vec4<u32>, 8>, // 128 字节
};

struct Entity {
    model_matrix: mat4x4<f32>,
    inv_model_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    sdf_index: u32,
    instance_scale: f32,
    _align_pad: vec2<u32>, // 同步 Rust 端的 8 字节补丁
    screen_rect: array<f32, 4>,
    flags: u32,
    _pad: array<u32, 11>,
};

struct Scene { entities: array<Entity, 64>, };

struct Voxel {
    dist: f32,
    color: vec3<f32>,
    normal: vec3<f32>,
    modifier: f32,
    stretch: vec3<f32>,
    tangent: vec3<f32>,
    emissive: f32,
    ao: f32,
    entity_id: u32,
    skip_hint: f32, // 新增：安全跳跃暗示
};

fn decode_oct_custom(packed: u32, bits_per_axis: u32) -> vec3<f32> {
    let mask = (1u << bits_per_axis) - 1u;
    let ux = f32((packed >> bits_per_axis) & mask) / f32(mask);
    let uy = f32(packed & mask) / f32(mask);
    var v = vec2<f32>(ux * 2.0 - 1.0, uy * 2.0 - 1.0);
    var n = vec3<f32>(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    if (n.z < 0.0) {
        let sign_v = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        v = (1.0 - abs(v.yx)) * sign_v;
        n = vec3<f32>(v.x, v.y, n.z);
    }
    return normalize(n);
}

// 专为影子设计的“单载荷”解包，带宽消耗直接减半
fn unpack_cem4_dist_only(pos: vec3<i32>, z_offset: i32) -> f32 {
    let raw_0 = textureLoad(t_cem_data, vec3<i32>(pos.x, pos.y, pos.z * 2 + z_offset), 0);
    let d_bits = (raw_0.x >> 12u) & 0xFFFFFu;
    var d_i32 = i32(d_bits);
    if (d_bits >= 524288u) { d_i32 -= 1048576i; }
    return f32(d_i32) / 524287.0;
}

fn unpack_cem4(pos: vec3<i32>, z_offset: i32) -> Voxel {
    let raw_0: vec4<u32> = textureLoad(t_cem_data, vec3<i32>(pos.x, pos.y, pos.z * 2 + z_offset), 0);
    let raw_1: vec4<u32> = textureLoad(t_cem_data, vec3<i32>(pos.x, pos.y, pos.z * 2 + 1 + z_offset), 0);

    var v: Voxel;

    let d_bits = (raw_0.x >> 12u) & 0xFFFFFu;
    var d_i32 = i32(d_bits);
    if (d_bits >= 524288u) { d_i32 -= 1048576i; }
    v.dist = f32(d_i32) / 524287.0;
    v.color = palette[raw_0.x & 0xFFFu].rgb;

    v.normal = decode_oct_custom(raw_0.y >> 12u, 10u);
    v.modifier = f32((raw_0.y >> 4u) & 0xFFu) / 255.0;

    let s0 = f32((raw_0.z >> 22u) & 0x3FFu) / 1023.0;
    let s1 = f32((raw_0.z >> 12u) & 0x3FFu) / 1023.0;
    let s2 = f32((raw_0.z >> 2u) & 0x3FFu) / 1023.0;
    v.stretch = vec3<f32>(s0, s1, s2);

    v.tangent = decode_oct_custom(raw_0.w >> 18u, 7u);
    v.emissive = f32((raw_0.w >> 10u) & 0xFFu) / 255.0;
    v.ao = f32((raw_0.w >> 2u) & 0xFFu) / 255.0;

    // 从 raw_1.z (即 res_word6) 提取跳跃暗示
    v.skip_hint = f32(raw_1.z & 0xFFu) / 255.0;

    return v;
}

fn sample_dna_dist_only(uvw: vec3<f32>, sdf_index: u32) -> f32 {
    let tex_coord = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)) * 63.0;
    let i0 = vec3<i32>(floor(tex_coord));
    let i1 = min(i0 + vec3<i32>(1), vec3<i32>(63));
    let f = fract(tex_coord);
    let z_off = i32(sdf_index) * 128;

    let d000 = unpack_cem4_dist_only(i0, z_off);
    let d100 = unpack_cem4_dist_only(vec3<i32>(i1.x, i0.y, i0.z), z_off);
    let d010 = unpack_cem4_dist_only(vec3<i32>(i0.x, i1.y, i0.z), z_off);
    let d110 = unpack_cem4_dist_only(vec3<i32>(i1.x, i1.y, i0.z), z_off);
    let d001 = unpack_cem4_dist_only(vec3<i32>(i0.x, i0.y, i1.z), z_off);
    let d101 = unpack_cem4_dist_only(vec3<i32>(i1.x, i0.y, i1.z), z_off);
    let d011 = unpack_cem4_dist_only(vec3<i32>(i0.x, i1.y, i1.z), z_off);
    let d111 = unpack_cem4_dist_only(i1, z_off);

    return mix(mix(mix(d000, d100, f.x), mix(d010, d110, f.x), f.y),
               mix(mix(d001, d101, f.x), mix(d011, d111, f.x), f.y), f.z);
}

fn sample_dna_smooth(uvw: vec3<f32>, sdf_index: u32) -> Voxel {
    let tex_coord = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)) * 63.0;
    let i0 = vec3<i32>(floor(tex_coord));
    let i1 = min(i0 + vec3<i32>(1), vec3<i32>(63));
    let f = fract(tex_coord);
    
    let z_off = i32(sdf_index) * 128;

    let v000 = unpack_cem4(i0, z_off);
    let v100 = unpack_cem4(vec3<i32>(i1.x, i0.y, i0.z), z_off);
    let v010 = unpack_cem4(vec3<i32>(i0.x, i1.y, i0.z), z_off);
    let v110 = unpack_cem4(vec3<i32>(i1.x, i1.y, i0.z), z_off);
    let v001 = unpack_cem4(vec3<i32>(i0.x, i0.y, i1.z), z_off);
    let v101 = unpack_cem4(vec3<i32>(i1.x, i0.y, i1.z), z_off);
    let v011 = unpack_cem4(vec3<i32>(i0.x, i1.y, i1.z), z_off);
    let v111 = unpack_cem4(i1, z_off);

    var res: Voxel;
    res.dist = mix(mix(mix(v000.dist, v100.dist, f.x), mix(v010.dist, v110.dist, f.x), f.y),
                   mix(mix(v001.dist, v101.dist, f.x), mix(v011.dist, v111.dist, f.x), f.y), f.z);
    
    res.normal = normalize(mix(mix(mix(v000.normal, v100.normal, f.x), mix(v010.normal, v110.normal, f.x), f.y),
                               mix(mix(v001.normal, v101.normal, f.x), mix(v011.normal, v111.normal, f.x), f.y), f.z));

    res.stretch = mix(mix(mix(v000.stretch, v100.stretch, f.x), mix(v010.stretch, v110.stretch, f.x), f.y),
                      mix(mix(v001.stretch, v101.stretch, f.x), mix(v011.stretch, v111.stretch, f.x), f.y), f.z);
    
    res.color = v000.color;
    res.modifier = v000.modifier;
    res.tangent = v000.tangent;
    res.emissive = v000.emissive;
    res.ao = v000.ao;
    res.entity_id = v000.entity_id;
    res.skip_hint = v000.skip_hint;
    return res;
}

// 快速步进采样函数：只获取距离和跳跃暗示，节省带宽
fn sample_dna_fast_step(uvw: vec3<f32>, sdf_index: u32, model_scale: f32) -> Voxel {
    // 1. 极简采样：只拿距离和跳跃暗示，不解压颜色和法线（节省带宽）
    let z_off = i32(sdf_index) * 128;
    let tex_coord = vec3<i32>(uvw * 63.0);
    
    // 关键：只读第一个 word 拿距离，读 word6 拿 skip_hint
    let raw_0 = textureLoad(t_cem_data, vec3<i32>(tex_coord.x, tex_coord.y, tex_coord.z * 2 + z_off), 0).x;
    let raw_1_z = textureLoad(t_cem_data, vec3<i32>(tex_coord.x, tex_coord.y, tex_coord.z * 2 + 1 + z_off), 0).z;
    
    var v: Voxel;
    // 解包距离
    let d_bits = (raw_0 >> 12u) & 0xFFFFFu;
    var d_i32 = i32(d_bits);
    if (d_bits >= 524288u) { d_i32 -= 1048576i; }
    v.dist = f32(d_i32) / 524287.0;
    
    // 提取跳跃暗示 (0.0 - 1.0)
    v.skip_hint = f32(raw_1_z & 0xFFu) / 255.0;
    
    // 只有在非常接近表面时，我们才给它加上各向异性修正
    if (v.dist < 0.05) {
        // 使用完整的各向异性采样
        v = sample_dna_anisotropic(uvw, sdf_index, model_scale);
    } else {
        v.dist = v.dist * model_scale;
    }
    
    return v;
}

// 全量解包函数：获取完整的体素信息
fn unpack_cem4_full(tex_coord: vec3<i32>, z_off: i32) -> Voxel {
    let raw_0 = textureLoad(t_cem_data, vec3<i32>(tex_coord.x, tex_coord.y, tex_coord.z * 2 + z_off), 0);
    let raw_1 = textureLoad(t_cem_data, vec3<i32>(tex_coord.x, tex_coord.y, tex_coord.z * 2 + 1 + z_off), 0);
    
    var v: Voxel;
    // 解包距离
    let d_bits = (raw_0.x >> 12u) & 0xFFFFFu;
    var d_i32 = i32(d_bits);
    if (d_bits >= 524288u) { d_i32 -= 1048576i; }
    v.dist = f32(d_i32) / 524287.0;
    
    // 解包颜色 - 使用与原始函数相同的方式
    v.color = palette[raw_0.x & 0xFFFu].rgb;
    
    // 解包法线
    v.normal = decode_oct_custom(raw_0.y >> 12u, 10u);
    v.modifier = f32((raw_0.y >> 4u) & 0xFFu) / 255.0;
    
    // 解包其他属性
    let s0 = f32((raw_0.z >> 22u) & 0x3FFu) / 1023.0;
    let s1 = f32((raw_0.z >> 12u) & 0x3FFu) / 1023.0;
    let s2 = f32((raw_0.z >> 2u) & 0x3FFu) / 1023.0;
    v.stretch = vec3<f32>(s0, s1, s2);
    
    v.tangent = decode_oct_custom(raw_0.w >> 18u, 7u);
    v.emissive = f32((raw_0.w >> 10u) & 0xFFu) / 255.0;
    v.ao = f32((raw_0.w >> 2u) & 0xFFu) / 255.0;
    
    // 从 raw_1.z (即 res_word6) 提取跳跃暗示
    v.skip_hint = f32(raw_1.z & 0xFFu) / 255.0;
    
    return v;
}

// --- 2. 增强型 Voxel 采样：引入各向异性修正 ---
fn sample_dna_anisotropic(uvw: vec3<f32>, sdf_index: u32, model_scale: f32) -> Voxel {
    var v = sample_dna_smooth(uvw, sdf_index);
    
    // --- 核心修复 1：注入"生命膨胀量" ---
    // 减去 0.008 相当于把所有表面向外推 0.008 个体素单位
    // 这对于细杆来说是救命的，防止它在插值中被"消融"
    let swelling = 0.005;
    v.dist = v.dist - swelling;

    // --- 核心修复 2：动态张力 ---
    // 细长的杆子需要更强的张力补偿，把 0.45 提到 0.6
    let anisotropy = 1.0 + v.stretch.x * 0.1;
    v.dist = v.dist * model_scale * anisotropy;
    
    return v;
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

// 1. 核心噪声与哈希（保持稳定）
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn noise_stable(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash22(i + vec2<f32>(0.0, 0.0)).x;
    let b = hash22(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(i + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(i + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 2. 积云专属：带有“撕裂感”的沃洛诺伊
fn cumulus_voronoi(p: vec2<f32>, t: f32) -> f32 {
    // 关键重构：用噪声去扭曲 UV，产生撕裂的边缘
    let distortion = noise_stable(p * 0.5 + t * 0.05);
    let distorted_p = p + distortion * 0.8;
    
    let i = floor(distorted_p);
    let f = fract(distorted_p);
    var min_dist = 1.0;
    
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let anchor = hash22(i + neighbor);
            let diff = neighbor + anchor - f;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    // 反转距离：让中心变厚
    return 1.0 - min_dist;
}



// 2. 基础平滑噪声
fn noise_hash(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash22(i + vec2<f32>(0.0, 0.0)).x;
    let b = hash22(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(i + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(i + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 3. 核心：领域扭曲分形噪声 (产生积聚感和岛屿感)
fn cloud_fbm_warped(p: vec2<f32>, t: f32) -> f32 {
    // 整体飘移
    let drift = vec2<f32>(t * 1.03, t * 0.515);
    var uv = p + drift;
    
    let warp_strength = 3.1;
    let final_warp = 0.6;
    
    let q = vec2<f32>(
        noise_hash(uv + vec2<f32>(t * 3.8, 0.0)),
        noise_hash(uv + vec2<f32>(0.0, t * 3.65))
    );
    
    let r = vec2<f32>(
        noise_hash(uv + q * warp_strength + vec2<f32>(t * 0.1, t * 0.05)),
        noise_hash(uv + q * (warp_strength * 0.7) + vec2<f32>(t * 5.02, t * 5.08))
    );
    
    var f = noise_hash(uv + r * final_warp);
    f = smoothstep(0.4, 0.8, f);
    return f;
}

// 4. 大尺度云团分组（新增）
fn cloud_group_mask(p: vec2<f32>, t: f32) -> f32 {
    // 极低频率噪声，决定云团的大范围分布
    let group_uv = p * 0.15 + vec2<f32>(t * 0.08, t * 0.04);
    
    // 多层叠加制造自然分组
    let g1 = noise_hash(group_uv);
    let g2 = noise_hash(group_uv * 2.5 + 1.2) * 0.5;
    let g3 = noise_hash(group_uv * 5.0 - 0.8) * 0.25;
    
    var group = (g1 + g2 + g3) * 0.55;  // 归一化到 0-1 左右
    
    // 锐化分组边界，让云团更明显
    group = smoothstep(0.35, 0.7, group);
    
    return group;
}

// --- 5. 极速天空与雾效 + 流体云系统 ---
fn get_sky_and_fog(ray_dir: vec3<f32>, color: vec3<f32>, t: f32) -> vec3<f32> {
    let zenith = vec3<f32>(0.12, 0.3, 0.6);
    let horizon = vec3<f32>(0.65, 0.8, 0.95);
    let haze_color = vec3<f32>(0.85, 0.9, 0.95);
    let cloud_top = vec3<f32>(1.0, 1.0, 1.0);
    let cloud_bottom = vec3<f32>(0.75, 0.82, 0.9);
    
    let L = normalize(params.light_dir.xyz);
    
    // 1. 基础梯度
    var sky = mix(horizon, zenith, clamp(ray_dir.y * 1.3, 0.0, 1.0));
    
    // 2. 地平线厚度感
    let haze_factor = pow(1.0 - max(ray_dir.y, 0.0), 5.0);
    sky = mix(sky, haze_color, haze_factor * 0.4);
    
    // 3. 流体云渲染
    if (ray_dir.y > 0.0) {
        let cloud_height = 120.0;
        let t_cloud = cloud_height / (ray_dir.y + 0.001);
        
        let world_pos = (params.cam_pos.xyz + ray_dir * t_cloud).xz * vec2<f32>(0.015, 0.02);
        let time = params.time * 0.08;
        
        // 获取小尺度云朵密度
        let small_cloud = cloud_fbm_warped(world_pos, time);
        
        // 获取大尺度云团分组遮罩
        let group_mask = cloud_group_mask(world_pos, time);
        
        // 组合：大分组决定哪里有大片云，小云朵在分组内填充细节
        let density = small_cloud * group_mask;
        
        if (density > 0.01) {
            var final_cloud = mix(cloud_bottom, cloud_top, pow(density, 0.5));
            
            let sun_highlight = pow(max(dot(ray_dir, L), 0.0), 12.0) * 0.3 * (1.0 - density);
            final_cloud += sun_highlight;
            
            let dist_fade = (1.0 - haze_factor) * smoothstep(0.0, 0.15, ray_dir.y);
            sky = mix(sky, final_cloud, density * 0.85 * dist_fade);
        }
    }
    
    // 4. 太阳
    let sun_dot = max(dot(ray_dir, L), 0.0);
    sky += vec3<f32>(1.0, 0.95, 0.8) * pow(sun_dot, 128.0) * 0.6;

    // 5. 远景消隐
    let fog_factor = clamp(t / FOG_DIST, 0.0, 1.0);
    let fog_intensity = fog_factor * fog_factor;
    let final_fog_color = mix(horizon, haze_color, 0.5);
    
    if (t >= FOG_DIST) { return sky; }
    return mix(color, final_fog_color, fog_intensity);
}

fn sdPlane(p: vec3<f32>) -> f32 {
    return p.y - GROUND_Y;
}

// --- 4. 大地棋盘格着色 ---
fn get_ground_info(p: vec3<f32>) -> vec3<f32> {
    let check = (i32(floor(p.x)) + i32(floor(p.z))) & 1;
    if (check == 0) { return vec3<f32>(0.15, 0.15, 0.18); }
    return vec3<f32>(0.25, 0.25, 0.28);
}

// 快速判定：当前像素是否可能存在实体
fn get_retina_mask(uv: vec2<f32>, tile_base: u32, count: u32, ray_dir: vec3<f32>) -> u32 {
    // 1. 如果射线看向地面 (y < -2)，必有实体
    if (ray_dir.y < -0.01) { return 1u; }
    
    // 2. 检查当前 Tile 里的物体的 Mini-Tile 矩形
    for (var i = 0u; i < count; i++) {
        let e = scene.entities[tile_data[tile_base + 4u + i]];
        let r = e.screen_rect;
        // 矩形判定
        if (uv.x >= r[0] && uv.x <= r[2] && uv.y >= r[1] && uv.y <= r[3]) {
            return 1u;
        }
    }
    return 0u; // 绝对的虚空（天空）
}

// 使用共享内存加速 Retina 决策
var<workgroup> shared_needs_full_res: bool;

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
        
        // --- 核心重构：各向异性修正 (The Tension Law) ---
        let anisotropy_factor = 1.0 + voxel.stretch.x * 0.5; // 先保守一点
        let d_dna = voxel.dist * model_max_dim * e.instance_scale * anisotropy_factor;

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

// 1. 定义工作组共享变量 (Workgroup Shared Memory)
// 这块内存位于 GPU 核心内部，访问速度是显存的 100 倍
var<workgroup> block_has_content: bool;

// --- 5. 核心：Retina-DNA 步进器 ---
@compute @workgroup_size(8, 8)
fn cs_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let screen_size = textureDimensions(output_texture);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // --- [核心：运动自适应决策] ---
    let is_even_frame = (params.frame_index % 2u) == 0u;
    let is_block_active = ((wg_id.x + wg_id.y) % 2u == 0u) == is_even_frame;

    // 状态 A：相机在动 -> 强制全屏 1:1 渲染，不留虚影
    // 状态 B：块处于活跃状态 -> 必须渲染
    var needs_render = is_block_active || (params.is_moving != 0u);

    if (!needs_render) {
        // [只有静止时的不活跃块才走重投影]
        let history = sample_history_reprojected(id.xy, screen_size);
        if (history.a > 0.0) {
            textureStore(output_texture, id.xy, history);
            return;
        }
    }

    // --- 第一步：全组初始化决策 ---
    // 只有组内第 0 号线程执行初始化
    if (all(local_id.xy == vec2<u32>(0u))) {
        block_has_content = false;
    }
    
    // 同步：确保所有线程都看到了 block_has_content 被设为 false
    workgroupBarrier();

    // --- 第二步：快速投票 (Quick Vote) ---
    // 每个线程算一下自己是不是看向天空
    let screen_uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);
    let uv = vec2<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0);
    let target_pos = params.proj_inv * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(target_pos.xyz / target_pos.w), 0.0)).xyz);
    let ray_p = params.cam_pos.xyz;

    let tiles_x = (screen_size.x + 15u) / 16u;
    let base_addr = ((id.y / 16u) * tiles_x + (id.x / 16u)) * TILE_STRIDE;
    let entity_count = tile_data[base_addr];

    // 检查自己是否处于“实体区”
    var my_pixel_has_content = false;
    if (ray_dir.y < -0.01) {
        my_pixel_has_content = true;
    } else {
        // 快速遍历 Mini-Tile
        for (var i = 0u; i < entity_count; i++) {
            let e = scene.entities[tile_data[base_addr + 4u + i]];
            let r = e.screen_rect;
            if (screen_uv.x >= r[0] && screen_uv.x <= r[2] && screen_uv.y >= r[1] && screen_uv.y <= r[3]) {
                my_pixel_has_content = true;
                break;
            }
        }
    }

    // 如果当前像素探测到了实体，修改共享变量通知全组
    if (my_pixel_has_content) {
        block_has_content = true;
    }

    // 关键同步：等待组内所有 64 个像素投票完成
    workgroupBarrier();

    // --- 第三步：分流执行 (Retina Early Exit) ---
    if (!block_has_content) {
        // [神迹时刻]：全组 64 个线程同时进入这个分支
        // 没有任何线程会去跑复杂的 DNA 步进循环，指令流水线极其干净
        let sky = get_sky_and_fog(ray_dir, vec3<f32>(0.0), 1000.0);
        textureStore(output_texture, id.xy, vec4<f32>(sky, 0.0));
        return;
    }

    // --- 第四步：昂贵的步进逻辑 (仅在有内容的块运行) ---
    var t_hit = 1000.0;
    var hit_entity = 0xFFFFFFFFu;
    var t_ground = -1.0;
    if (ray_dir.y < -0.001) { t_ground = (GROUND_Y - ray_p.y) / ray_dir.y; }

    var final_v: Voxel;
    var hit = false;

    for (var i = 0u; i < entity_count; i++) {
        let e_idx = tile_data[base_addr + 4u + i];
        if (hit && t_hit < 1000.0) { break; }
        
        let e = scene.entities[e_idx];
        let r = intersect_aabb_robust((e.inv_model_matrix * vec4<f32>(ray_p, 1.0)).xyz,
                                     (e.inv_model_matrix * vec4<f32>(ray_dir, 0.0)).xyz,
                                      e.aabb_min.xyz, e.aabb_max.xyz);
        
        if (r.y > 0.0 && r.x < t_hit && r.x < r.y) {
            var t_march = max(r.x, 0.0);
            let t_limit = min(r.y, t_hit);
            var steps = 0u;
            
            // 预计算本地射线原点和方向，减少循环内计算
            let local_ray_o = (e.inv_model_matrix * vec4<f32>(ray_p, 1.0)).xyz;
            let local_ray_d = (e.inv_model_matrix * vec4<f32>(ray_dir, 0.0)).xyz;
            
            while (t_march < t_limit) {
                steps++;
                let p_local = local_ray_o + local_ray_d * t_march;
                if (any(p_local < e.aabb_min.xyz - 0.1) || any(p_local > e.aabb_max.xyz + 0.1)) { break; }
                
                let dims = e.aabb_max.xyz - e.aabb_min.xyz;
                let uvw = clamp((p_local - e.aabb_min.xyz) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
                
                // --- 核心优化：双速步进 ---
                let v = sample_dna_fast_step(uvw, e.sdf_index, max(dims.x, max(dims.y, dims.z)) * e.instance_scale);
                
                if (v.dist < 0.001) {
                    // 撞击！这时才去执行最沉重的“全量解包”，拿颜色和法线
                    final_v = unpack_cem4_full(vec3<i32>(uvw * 63.0), i32(e.sdf_index) * 128);
                    t_hit = t_march; 
                    hit = true; 
                    hit_entity = e_idx;
                    break;
                }
                
                // 利用 skip_hint 增加步进信心
                // 如果 skip_hint 大，说明周围很空，可以大步跨越
                let leap = max(v.dist * 0.8, v.skip_hint * 0.05);
                t_march += max(leap, 0.006);
                
                if (steps > 64u) { break; } // 将原本的 100 步砍到 64 步，靠精准度补齐
            }
        }
    }

    // --- 第五步：最终着色 ---
    var final_rgb: vec3<f32>;
    let L = normalize(params.light_dir.xyz);

    if (params.debug_mode == 1u) {
        // 调试模式：显示深度
        if (hit && (t_ground < 0.0 || t_hit < t_ground)) {
            final_rgb = vec3<f32>(t_hit / 50.0);
        } else if (t_ground > 0.0) {
            final_rgb = vec3<f32>(t_ground / 50.0);
        } else {
            final_rgb = vec3<f32>(1.0);
        }
        textureStore(output_texture, id.xy, vec4<f32>(final_rgb, 1.0));
    } else if (params.debug_mode == 2u) {
        // 调试模式：显示法线
        if (hit && (t_ground < 0.0 || t_hit < t_ground)) {
            let e = scene.entities[hit_entity];
            let world_normal = normalize((e.model_matrix * vec4<f32>(final_v.normal, 0.0)).xyz);
            final_rgb = (world_normal + 1.0) * 0.5;
        } else if (t_ground > 0.0) {
            final_rgb = vec3<f32>(0.0, 1.0, 0.0);
        } else {
            final_rgb = vec3<f32>(1.0);
        }
        textureStore(output_texture, id.xy, vec4<f32>(final_rgb, 1.0));
    } else if (params.debug_mode == 3u) {
        // 调试模式：显示阴影
        if (hit && (t_ground < 0.0 || t_hit < t_ground)) {
            let p = ray_p + ray_dir * t_hit;
            let e = scene.entities[hit_entity];
            let world_normal = normalize((e.model_matrix * vec4<f32>(final_v.normal, 0.0)).xyz);
            let luoer = calculate_luoer_tile_optimized(p, world_normal, L, id.xy, base_addr, entity_count, hit_entity);
            final_rgb = vec3<f32>(luoer);
        } else if (t_ground > 0.0) {
            let p = ray_p + ray_dir * t_ground;
            let luoer = calculate_luoer_tile_optimized(p, vec3<f32>(0.0, 1.0, 0.0), L, id.xy, base_addr, entity_count, 0xFFFFFFFFu);
            final_rgb = vec3<f32>(luoer);
        } else {
            final_rgb = vec3<f32>(1.0);
        }
        textureStore(output_texture, id.xy, vec4<f32>(final_rgb, 1.0));
    } else if (params.debug_mode == 4u) {
        // 调试模式：显示 AO
        final_rgb = vec3<f32>(0.5, 0.5, 0.5);
        textureStore(output_texture, id.xy, vec4<f32>(final_rgb, 1.0));
    } else if (params.debug_mode == 5u) {
        // 调试模式：显示帧率/统计信息
        final_rgb = vec3<f32>(0.0, 0.0, 0.0);
        textureStore(output_texture, id.xy, vec4<f32>(final_rgb, 1.0));
    } else {
        // 正常渲染模式
        if (hit && (t_ground < 0.0 || t_hit < t_ground)) {
            let p = ray_p + ray_dir * t_hit;
            let e = scene.entities[hit_entity];
            let world_normal = normalize((e.model_matrix * vec4<f32>(final_v.normal, 0.0)).xyz);
            let luoer = calculate_luoer_tile_optimized(p, world_normal, L, id.xy, base_addr, entity_count, hit_entity);
            let lighting = mix(0.2, 1.0, luoer * max(0.0, dot(world_normal, L)));
            final_rgb = get_sky_and_fog(ray_dir, final_v.color * final_v.modifier * lighting, t_hit);
            // 关键：Alpha 通道存入真实的物理距离 t_hit
            textureStore(output_texture, id.xy, vec4<f32>(final_rgb, t_hit));
        } else if (t_ground > 0.0) {
            let p = ray_p + ray_dir * t_ground;
            let ground_col = get_ground_info(p);
            let luoer = calculate_luoer_tile_optimized(p, vec3<f32>(0.0, 1.0, 0.0), L, id.xy, base_addr, entity_count, 0xFFFFFFFFu);
            let lighting = mix(0.2, 1.0, luoer * max(0.0, dot(vec3<f32>(0.0, 1.0, 0.0), L)));
            final_rgb = get_sky_and_fog(ray_dir, ground_col * lighting, t_ground);
            // 关键：Alpha 通道存入真实的物理距离 t_ground
            textureStore(output_texture, id.xy, vec4<f32>(final_rgb, t_ground));
        } else {
            let sky = get_sky_and_fog(ray_dir, vec3<f32>(0.0), 1000.0);
            // 天空的深度设为0.0
            textureStore(output_texture, id.xy, vec4<f32>(sky, 0.0));
        }
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