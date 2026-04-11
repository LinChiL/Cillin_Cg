@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> scene: Scene;
@group(0) @binding(2) var t_cem_data: texture_3d<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> palette: array<vec4<f32>>;

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
    modifier: f32,
    entity_id: u32, // 记录所属实体ID，用于法线隔离
};

fn unpack_voxel_safe(data: u32) -> Voxel {
    var v: Voxel;
    // 1. 距离还原：显式处理 12bit 补码
    let d_bits = (data >> 20u) & 0xFFFu;
    var d_f32: f32;
    if (d_bits >= 2048u) {
        d_f32 = f32(i32(d_bits) - 4096i);
    } else {
        d_f32 = f32(d_bits);
    }
    // 关键：还原到 -1.0 到 1.0 的标准范围
    v.dist = d_f32 / 2047.0;

    // 2. 颜色还原：撞库
    let id = (data >> 10u) & 0x3FFu;
    v.color = palette[id].rgb;
    
    // 3. 微调值
    v.modifier = f32((data >> 2u) & 0xFFu) / 255.0;
    return v;
}

fn sample_dna_smooth(uvw: vec3<f32>, sdf_index: u32) -> Voxel {
    // 这里的 63.0 是为了保证索引不越界
    let tex_coord = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)) * 63.0;
    let i0 = vec3<i32>(floor(tex_coord));
    let i1 = min(i0 + vec3<i32>(1), vec3<i32>(63));
    let f = fract(tex_coord);
    let z_off = i32(sdf_index) * 64;

    // 采样周围 8 个点
    let v000 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i0.x, i0.y, i0.z + z_off), 0).r);
    let v100 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i1.x, i0.y, i0.z + z_off), 0).r);
    let v010 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i0.x, i1.y, i0.z + z_off), 0).r);
    let v110 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i1.x, i1.y, i0.z + z_off), 0).r);
    let v001 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i0.x, i0.y, i1.z + z_off), 0).r);
    let v101 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i1.x, i0.y, i1.z + z_off), 0).r);
    let v011 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i0.x, i1.y, i1.z + z_off), 0).r);
    let v111 = unpack_voxel_safe(textureLoad(t_cem_data, vec3<i32>(i1.x, i1.y, i1.z + z_off), 0).r);

    var res: Voxel;
    // 距离和颜色全插值
    res.dist = mix(mix(mix(v000.dist, v100.dist, f.x), mix(v010.dist, v110.dist, f.x), f.y),
                   mix(mix(v001.dist, v101.dist, f.x), mix(v011.dist, v111.dist, f.x), f.y), f.z);
    res.color = mix(mix(mix(v000.color, v100.color, f.x), mix(v010.color, v110.color, f.x), f.y),
                    mix(mix(v001.color, v101.color, f.x), mix(v011.color, v111.color, f.x), f.y), f.z);
    res.modifier = mix(mix(mix(v000.modifier, v100.modifier, f.x), mix(v010.modifier, v110.modifier, f.x), f.y),
                       mix(mix(v001.modifier, v101.modifier, f.x), mix(v011.modifier, v111.modifier, f.x), f.y), f.z);
    return res;
}

fn sdAABB(p: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> f32 {
    let center = (b_min + b_max) * 0.5;
    let half_extents = (b_max - b_min) * 0.5;
    let q = abs(p - center) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn get_scene_sdf_full(world_p: vec3<f32>) -> Voxel {
    var min_v: Voxel;
    min_v.dist = 100.0;
    min_v.color = vec3<f32>(0.2, 0.2, 0.2); // 默认背景色

    for (var i: u32 = 0u; i < params.entity_count; i++) {
        let e = scene.entities[i];
        let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
        
        let dims = e.aabb_max.xyz - e.aabb_min.xyz;
        let model_world_size = max(dims.x, max(dims.y, dims.z));
        
        // 核心：计算 AABB 的物理距离（带 instance_scale）
        let d_box = sdAABB(local_p, e.aabb_min.xyz, e.aabb_max.xyz) * e.instance_scale;
        
        // 只有当射线足够靠近 AABB 时，才进入 DNA 采样
        if (d_box < min_v.dist) {
            let uvw = (local_p - e.aabb_min.xyz) / dims;
            
            // 判定是否在盒子内或非常接近边缘
            if (all(uvw >= vec3<f32>(-0.01)) && all(uvw <= vec3<f32>(1.01))) {
                let voxel = sample_dna_smooth(uvw, e.sdf_index);
                // 还原 DNA 的绝对世界距离
                let d_dna = voxel.dist * model_world_size * e.instance_scale;
                
                // 联合判断：必须真的比 AABB 距离更近才更新
                if (d_dna < min_v.dist) {
                    min_v = voxel;
                    min_v.dist = d_dna;
                    min_v.entity_id = i;
                }
            } else {
                // 盒子外，作为占位符更新距离，方便跳跃
                if (d_box < min_v.dist) {
                    min_v.dist = d_box;
                    min_v.entity_id = i;
                }
            }
        }
    }
    return min_v;
}

fn get_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.015; // 采样步长
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * get_scene_sdf_full(p + k.xyy * h).dist +
        k.yyx * get_scene_sdf_full(p + k.yyx * h).dist +
        k.yxy * get_scene_sdf_full(p + k.yxy * h).dist +
        k.xxx * get_scene_sdf_full(p + k.xxx * h).dist
    );
}

fn get_normal_custom(p: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * get_scene_sdf_full(p + k.xyy * h).dist +
        k.yyx * get_scene_sdf_full(p + k.yyx * h).dist +
        k.yxy * get_scene_sdf_full(p + k.yxy * h).dist +
        k.xxx * get_scene_sdf_full(p + k.xxx * h).dist
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
fn get_normal_isolated(p: vec3<f32>, entity_idx: u32) -> vec3<f32> {
    let e = scene.entities[entity_idx];
    
    // 关键：h 必须随缩放自适应！
    // 确保 h 在局部空间始终占用大约 0.5 个体素的宽度
    let h = 0.01 * e.instance_scale;
    
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * get_entity_sdf_only(p + k.xyy * h, entity_idx) +
        k.yyx * get_entity_sdf_only(p + k.yyx * h, entity_idx) +
        k.yxy * get_entity_sdf_only(p + k.yxy * h, entity_idx) +
        k.xxx * get_entity_sdf_only(p + k.xxx * h, entity_idx)
    );
}

fn calculate_luoer(p: vec3<f32>, normal: vec3<f32>, L: vec3<f32>) -> f32 {
    var t: f32 = 0.25; // 起跳偏置，防止自遮挡
    var res: f32 = 1.0;
    let k: f32 = 8.0;   // 软阴影硬度
    
    for (var i = 0; i < 24; i++) {
        let d = get_scene_sdf_full(p + normal * 0.1 + L * t).dist;
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
        let d = get_scene_sdf_full(sample_p).dist;
        
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
    
    // 关键修正 1：大幅降低初始偏置
    // 只有这样，薄如纸片的模型才能被射线捕捉到
    var t: f32 = 0.03 + 0.02 * dither;
    
    var res: f32 = 1.0;
    let k: f32 = 12.0;   // 稍微降低硬度，增加阴影的包容度
    
    for (var i = 0; i < 48; i++) { // 增加步进次数，保证精度
        // 关键修正 2：法线偏置减小到 0.01
        let sample_p = p + normal * 0.01 + L * t;
        let d = get_scene_sdf_full(sample_p).dist;
        
        // 关键修正 3：负距离判定 (内部命中)
        // 在三线性插值下，薄片中心可能是负值。一旦发现负值，说明绝对撞上了。
        if (d < 0.0) { return 0.0; }
        
        // 判定命中
        if (d < 0.0005) { return 0.0; }
        
        // 软阴影公式
        res = min(res, k * d / t);
        
        // 关键修正 4：微步进 (Micro-stepping)
        // 将强制步进从 0.02 降低到 0.005
        // 这样射线就像显微镜一样，不会跳过任何薄片
        t += max(d, 0.005);
        
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

    // 射线初始化
    let screen_uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);
    let uv = vec2<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0);
    let target_pos = params.proj_inv * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
    let ray_dir = normalize((params.view_inv * vec4<f32>(normalize(target_pos.xyz / target_pos.w), 0.0)).xyz);
    let ray_p = params.cam_pos.xyz;

    var t = 0.1;
    var hit = false;
    var final_v: Voxel;
    
    for (var i = 0; i < 128; i++) {
        final_v = get_scene_sdf_full(ray_p + ray_dir * t);
        
        // 关键：微小的命中阈值
        if (final_v.dist < 0.001) { hit = true; break; }
        
        // 关键：增加安全步长上限，防止射线在表面反复横跳
        t += max(final_v.dist * 0.75, 0.0005); 
        if (t > 800.0) { break; }
    }

    if (hit) {
        let hit_p = ray_p + ray_dir * t;
        let normal = get_normal_isolated(hit_p, final_v.entity_id);
        let L = normalize(params.light_dir.xyz);

        // 曦罗混合
        let luoer = calculate_luoer_precision(hit_p, normal, L, id.xy);
        
        // 关键修复：最终颜色对齐
        let base_color = final_v.color * final_v.modifier;
        let dot_nl = max(0.0, dot(normal, L));
        
        // 调试模式
        if (params.debug_mode == 1u) {
            // 模式 1：距离场可视化（红正绿负）
            let dist_color = debug_color(final_v.dist * 10.0);
            textureStore(output_texture, id.xy, vec4<f32>(dist_color, 1.0));
        } else if (params.debug_mode == 2u) {
            // 模式 2：UVW 映射可视化
            // 计算 UVW 坐标
            let e = scene.entities[final_v.entity_id];
            let local_p = (e.inv_model_matrix * vec4<f32>(hit_p, 1.0)).xyz;
            let dims = e.aabb_max.xyz - e.aabb_min.xyz;
            let uvw = (local_p - e.aabb_min.xyz) / dims;
            textureStore(output_texture, id.xy, vec4<f32>(uvw, 1.0));
        } else if (params.debug_mode == 3u) {
            // 模式 3：法线可视化
            let normal_color = (normal + 1.0) * 0.5; // 将法线从 [-1,1] 映射到 [0,1]
            textureStore(output_texture, id.xy, vec4<f32>(normal_color, 1.0));
        } else if (params.debug_mode == 4u) {
            // 模式 4：原始颜色
            textureStore(output_texture, id.xy, vec4<f32>(final_v.color, 1.0));
        } else {
            // 正常渲染
            let lighting = mix(0.35, 1.0, luoer * smoothstep(0.0, 0.05, dot_nl));
            textureStore(output_texture, id.xy, vec4<f32>(base_color * lighting, 1.0));
        }
    } else {
        // 背景色
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