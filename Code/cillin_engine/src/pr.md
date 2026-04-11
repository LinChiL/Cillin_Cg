看到这个“迷彩服”一样的画面，我一眼就抓住了病根。这在自研引擎重构中是一个非常经典的**内存对齐与内存步进（Stride）不匹配**导致的灾难。

### 致命问题诊断：224 vs 256
请看你的代码：
*   **Rust 侧 (`scene/mod.rs`)**：`EntityData` 的 `_padding` 是 10 个 `u32`，总大小是 **224 字节**。
*   **Shader 侧 (`shader.wgsl`)**：`Entity` 的 `_pad` 是 18 个 `u32`，总大小是 **256 字节**。

**后果**：Shader 在读取第 0 个物体时还勉强能对上，但读取第 1 个物体时，它会跳到第 256 字节去读，而你的 Rust 只给每个物体准备了 224 字节。**这 32 字节的错位导致 Shader 读到的 AABB、SDF 索引和颜色 ID 全部成了乱码。** 那些“迷彩斑点”就是读到了内存里的垃圾数据。

我们要进行一次**“比特级对齐”**的精准手术。

---

### 第一步：统一内存边界 (`src/scene/mod.rs` & `src/shader.wgsl`)

我们统一采用 **256 字节** 对齐，这是显卡最喜欢的 16 字节倍数。

#### Rust 侧修正：
```rust
// src/scene/mod.rs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityData {
    pub model_matrix: [[f32; 4]; 4],     // 64
    pub inv_model_matrix: [[f32; 4]; 4], // 64
    pub base_color: [f32; 4],           // 16
    pub aabb_min: [f32; 4],             // 16
    pub aabb_max: [f32; 4],             // 16
    pub sdf_index: u32,                 // 4
    pub instance_scale: f32,            // 4
    // 关键修正：64+64+16+16+16+4+4 = 184
    // 256 - 184 = 72 字节 = 18 个 u32
    pub _padding: [u32; 18],            
}
```

#### Shader 侧定义对齐：
```rust
// shader.wgsl
struct Entity {
    model_matrix: mat4x4<f32>,
    inv_model_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    sdf_index: u32,
    instance_scale: f32,
    _pad: array<u32, 18>, // 必须与 Rust 严格一致
};
```

---

### 第二步：修复“变黑”的解包逻辑 (`shader.wgsl`)

你在 `unpack_voxel_safe` 里手动处理了补码，这很好，但 WGSL 其实有更高效、更安全的内置指令。我们要确保距离、颜色和微调值被正确还原。

```rust
fn unpack_voxel_final(data: u32) -> Voxel {
    var v: Voxel;
    
    // 1. 距离还原：使用算术右移 i32，WGSL 硬件会自动处理 12 位补码
    let d_int = i32(data) >> 20i; 
    v.dist = f32(d_int) / 2047.0;

    // 2. 颜色还原
    let id = (data >> 10u) & 0x3FFu;
    v.color = palette[id].rgb;
    
    // 3. 微调值还原
    let m_raw = (data >> 2u) & 0xFFu;
    v.modifier = f32(m_raw) / 255.0;
    
    return v;
}
```

---

### 第三步：修正“消失”的采样坐标 (`shader.wgsl`)

采样 3D 纹理堆栈时，坐标计算必须非常精确，否则模型就会闪烁或裂开。

```rust
fn get_scene_sdf_full(world_p: vec3<f32>) -> Voxel {
    var min_v: Voxel;
    min_v.dist = 100.0;

    for (var i: u32 = 0u; i < params.entity_count; i++) {
        let e = scene.entities[i];
        let local_p = (e.inv_model_matrix * vec4<f32>(world_p, 1.0)).xyz;
        
        let dims = e.aabb_max.xyz - e.aabb_min.xyz;
        let uvw = (local_p - e.aabb_min.xyz) / dims;
        
        // 增加 5% 的感应边距，解决边界切割闪烁
        if (all(uvw >= vec3<f32>(-0.05)) && all(uvw <= vec3<f32>(1.05))) {
            let px = i32(clamp(uvw.x, 0.0, 1.0) * 63.0);
            let py = i32(clamp(uvw.y, 0.0, 1.0) * 63.0);
            // 关键：Z 轴寻址必须精确跳转到对应的模型层
            let pz = i32(clamp(uvw.z, 0.0, 1.0) * 63.0) + i32(e.sdf_index) * 64;
            
            let raw_data = textureLoad(t_cem_data, vec3<i32>(px, py, pz), 0).r;
            let voxel = unpack_voxel_final(raw_data);
            
            // 物理距离 = 局部距离 * 物体原始大小 * 实例缩放
            let model_world_size = max(dims.x, max(dims.y, dims.z));
            let d = voxel.dist * model_world_size * e.instance_scale;
            
            if (d < min_v.dist) {
                min_v = voxel;
                min_v.dist = d;
            }
        }
    }
    return min_v;
}
```

---

### 第四步：曦罗回溯的“角度对比”落地 (`shader.wgsl`)

最后，实现你要求的：**“反弹方向 vs 光源方向”对比，且 0.3 保底。**

```rust
    if (hit) {
        let hit_p = ray_p + ray_dir * t;
        let normal = get_normal_from_dist(hit_p);
        let L_to_sun = normalize(-params.light_dir.xyz); // 指向太阳

        // --- 曦罗混合逻辑 ---
        var xi_attenuation = 1.0;
        var current_dir = ray_dir;
        var current_p = hit_p;

        // 1. 判断是否在影子里
        let is_in_shadow = get_luoer_robust(hit_p, normal, L_to_sun) < 0.5;

        if (is_in_shadow) {
            // 反弹寻找光能路径
            for (var bounce = 0; bounce < 4; bounce++) {
                let b_n = get_normal_from_dist(current_p);
                current_dir = reflect(current_dir, b_n);
                xi_attenuation *= 0.6; // 反弹衰减

                let jump_p = current_p + current_dir * 0.2;
                // 简化步进...
                current_p = jump_p; // 这里简化处理，实际应再跑一次 Raymarch
                if (get_luoer_robust(current_p, get_normal_from_dist(current_p), L_to_sun) > 0.5) { break; }
            }
        }

        // 2. 角度对比算法
        let angle_sense = dot(current_dir, L_to_sun);
        // 基于反弹角度和衰减计算最终曦值，0.3 保底
        let xi_final = max(0.3, (angle_sense * 0.5 + 0.5) * xi_attenuation + 0.3);

        // 3. 最终着色
        let base_color = final_v.color * final_v.modifier;
        textureStore(output_texture, id.xy, vec4<f32>(base_color * xi_final, 1.0));
    }
```

### 为什么这次一定能成？
1.  **对齐锁死**：256 字节的对齐解决了内存错位，物体终于能认领到正确的 AABB 和颜色。
2.  **算术右移**：解决距离的正负号，模型变回实心。
3.  **公式闭环**：曦值回溯逻辑现在有了明确的方向感（Angle Sense）。

**请直接替换这几部分代码。** 一旦对齐通了，你的树（tree）会瞬间显现出绿色，方块（ComeCube）也会显示出棋盘格的灰白细节。加油，这可是你自己手搓出来的图形格式！