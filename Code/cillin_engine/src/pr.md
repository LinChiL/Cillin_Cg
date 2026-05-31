**好的，我已经完整阅读了所有关键代码。**

代码确实比较乱（多次重构留下的历史残留：动态创建 bind group、多种 pool fill 路径并存、resize 不彻底、sun matrix 不稳定等）。我现在给你**一套干净、可靠的修复方案**。

---

### **第一步：替换 `CremModule`（核心清理）**

**完整替换 `src/crem/crem_main.rs` 中的 `CremModule` 实现**（重点部分）：

```rust
// src/crem/crem_main.rs
impl CremModule {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // ... 保持前面所有纹理创建代码不变（直到 sun_pool_id_view / sun_pool_depth_view 创建完）...

        // === 关键修复：统一在 new() 末尾创建所有 BindGroup ===
        let sun_pool_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sun Pool Bind Group"),
            layout: &sun_pool_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tri_id_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&sun_pool_id_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sun_pool_depth_view) },
            ],
        });

        let bind_group = device.create_bind_group(/* ... 保持你原来的 bind_group 创建 ... */);

        Self {
            // ... 其他字段保持不变 ...
            sun_pool_bind_group: Some(sun_pool_bind_group),
            // ...
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // ... 保持前面所有纹理重新创建代码 ...

        // === 必须重新创建 BindGroup ===
        self.sun_pool_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sun Pool Bind Group (Resized)"),
            layout: &self.sun_pool_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.tri_id_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.sun_pool_id_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.sun_pool_depth_view) },
            ],
        }));

        // 同时重建主 bind_group（因为 view 变了）
        self.bind_group = Some(/* ... 你原来的重建代码 ... */);
    }

    // 使用最可靠的 Compute 版本
    pub fn execute_pool_fill_simple(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let Some(raster_bg) = &self.raster_bind_group else { return; };
        let Some(pool_bg) = &self.sun_pool_bind_group else { 
            println!("Sun Pool BindGroup 未创建");
            return; 
        };

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sun Pool Fill"),
            ..Default::default()
        });

        cpass.set_pipeline(&self.sun_pool_pipeline);
        cpass.set_bind_group(0, raster_bg, &[]);
        cpass.set_bind_group(1, pool_bg, &[]);

        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
```

---

### **第二步：清理 `CoreState::render()`**

替换 `render()` 中 CREM 相关部分为：

```rust
pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    self.debug_mode = 0; // 先用3看阴影效果，确认后再切7

    let light_dir = glam::Vec3::new(0.5, 1.0, 0.5).normalize();
    let view_proj = self.render_context.current_proj * self.render_context.current_view;

    // 每次都更新太阳矩阵
    self.crem.update_sun_matrix(light_dir, self.camera.eye);

    // 更新 params（保持你原来的逻辑）...
    // ...

    let mut encoder = self.render_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let crem_entities: Vec<crate::scene::EntityData> = /* ... 保持原来 ... */;
    self.crem.update_crem_entities(&self.render_context.queue, &crem_entities);

    let count = crem_entities.len() as u32;

    // 阶段1：主相机 G-Buffer
    self.crem.execute_raster_instanced(&mut encoder, count);

    // 阶段2：阳光池填充（关键修复）
    self.crem.execute_pool_fill_simple(
        &mut encoder,
        &self.params_buffer,
        self.render_context.size.width,
        self.render_context.size.height,
    );

    // 阶段3：Compute 主渲染
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(self.render_context.compute_pipeline.as_ref().unwrap());
        cpass.set_bind_group(0, &self.compute_bind_group, &[]);
        cpass.set_bind_group(1, self.crem.bind_group.as_ref().unwrap(), &[]);
        cpass.dispatch_workgroups((self.render_context.size.width + 7) / 8, (self.render_context.size.height + 7) / 8, 1);
    }

    // ... 后面的 blit、present 等保持不变 ...
}
```

---

### **第三步：Shader 修复（`crem_shader.wgsl`）**

在 `fill_sun_pool` 中**增加安全检查**：

```wgsl
@compute @workgroup_size(8, 8)
fn fill_sun_pool(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(crem_tri_id_tex);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let tri_id = textureLoad(crem_tri_id_tex, vec2<i32>(id.xy), 0).r;
    if (tri_id == 0u) { return; }

    let depth = textureLoad(crem_depth_tex, vec2<i32>(id.xy), 0).r;

    let uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = params.proj_inv * vec4<f32>(ndc, depth, 1.0);
    let world_p = (params.view_inv * (clip / clip.w)).xyz;

    let sun_clip = params.sun_view_proj * vec4<f32>(world_p, 1.0);
    let sun_ndc = sun_clip.xyz / sun_clip.w;
    let sun_uv = vec2<f32>(sun_ndc.x * 0.5 + 0.5, 1.0 - (sun_ndc.y * 0.5 + 0.5));

    let pool_size = vec2<u32>(textureDimensions(sun_pool_id_tex));
    let write_coord = vec2<i32>(clamp(sun_uv * vec2<f32>(pool_size), vec2<f32>(0.0), vec2<f32>(pool_size - vec2<f32>(1.0))));

    textureStore(sun_pool_id_tex, write_coord, vec4<u32>(tri_id, 0u, 0u, 0u));
}
```

---

**现在请按顺序操作：**

1. 替换 `crem_main.rs` 中的 `new()`、`resize()` 和 `execute_pool_fill_simple`。
2. 修改 `CoreState::render()` 中的调用顺序。
3. 更新 shader 中的 `fill_sun_pool`。
4. 编译运行，保持 `debug_mode = 7`。

**告诉我运行结果**（尤其是启动时是否有绿色、resize 后是否稳定、移动时是否还有贪吃蛇）。

如果还有问题，我会继续给你**第2轮清理**（进一步简化 `update_sun_matrix` 和阴影采样）。

我们这次把这块彻底弄干净。开始吧！