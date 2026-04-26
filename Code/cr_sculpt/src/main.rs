use std::borrow::Cow;
use std::sync::Arc;

use glam::{Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use gltf;
use rfd;

mod math;
use math::{Params, Primitive, Anchor};

impl Primitive {
    pub fn new_sphere(pos: glam::Vec3, radius: f32) -> Self {
        Self {
            // 关键：逆矩阵用于将世界坐标转回局部球心坐标
            inv_model_matrix: glam::Mat4::from_translation(pos).inverse().to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            params: [radius, 0.0, 0.5, 0.0], // [半径, 0, 平滑度, 类型0=球]
        }
    }

    pub fn new_box(pos: glam::Vec3, size: f32) -> Self {
        Self {
            inv_model_matrix: glam::Mat4::from_translation(pos).inverse().to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            params: [size, 0.1, 0.5, 1.0], // [半长, 圆角半径, 平滑度, 类型1=方]
        }
    }
}

struct App<'a> {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout, // 新增
    render_bind_group_layout: wgpu::BindGroupLayout,  // 新增
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    scaffold_render_pipeline: wgpu::RenderPipeline,
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
    primitives_buffer: wgpu::Buffer,
    anchor_buffer: wgpu::Buffer,
    grid_buffer: wgpu::Buffer,
    scaffold_buffer: wgpu::Buffer,
    params: math::Params,
    camera: math::Camera,
    is_mmb_pressed: bool,
    is_shift_pressed: bool,
    last_mouse_pos: [f32; 2],
    // WASD 键盘状态
    is_w_pressed: bool,
    is_a_pressed: bool,
    is_s_pressed: bool,
    is_d_pressed: bool,
    // FPS 跟踪
    last_frame_time: std::time::Instant,
    fps: f32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    primitives: Vec<math::Primitive>,
    scaffold_vertices: Vec<glam::Vec3>,
    anchors: Vec<math::Anchor>,
}

impl<'a> App<'a> {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sculpt Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sculpt.wgsl"))),
        });

        let params = Params {
            view_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: Vec4::new(0.0, 1.0, -5.0, 1.0).to_array(),
            light_dir: Vec4::new(0.0, 1.0, 0.0, 0.0).to_array(),
            prim_count: 0,
            anchor_count: 0,
            scaffold_count: 0,
            is_moving: 0,
            grid_origin: [-2.0, -2.0, -2.0, 4.0 / 16.0], // 网格参数
            time: 0.0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _final_padding: [[0.0; 4]; 4],
            model_center: [0.0, 0.0, 0.0, 1.0],
        };

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<math::Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 初始写入参数
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        // --- [1. 预分配足够大的仓库，解决 Size is zero 错误] ---
        // 预设最大支持 128 个几何体
        let primitives_max_size = (128 * std::mem::size_of::<Primitive>()) as u64;
        let primitives_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitives Buffer"),
            size: primitives_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 预设最大支持 10 万个锚点
        let anchor_max_size = (100_000 * 32) as u64; // Anchor 是 32 字节
        let anchor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Anchor Buffer"),
            size: anchor_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 32x32x32 网格缓冲区 (32768个格子 * 8字节 = 256KB)
        let grid_max_size = (32768 * 8) as u64; // GridCell 是 8 字节
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Buffer"),
            size: grid_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 预设最大支持 10 万个脚手架点
        let scaffold_max_size = (100_000 * 16) as u64; // vec4 是 16 字节
        let scaffold_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scaffold Buffer"),
            size: scaffold_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- [2. 修正布局，匹配 Storage Texture] ---
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: primitives_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: anchor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scaffold_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        let scaffold_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scaffold Render Pipeline"),
            layout: Some(&compute_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_scaffold",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_scaffold",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
        });

        // 1. 先在外部创建 Context
        let egui_ctx = egui::Context::default();

        // 2. 加载中文字体
        let mut fonts = egui::FontDefinitions::default();
        // 尝试加载指定的中文字体
        if let Ok(font_data) = std::fs::read("f:\\Cillin_CG\\Cillin_Cg\\Asset\\Font\\GlowSansSC-Normal-Regular.otf") {
            fonts.font_data.insert("glow_sans".to_owned(), egui::FontData::from_owned(font_data));
            fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap().insert(0, "glow_sans".to_owned());
            fonts.families.get_mut(&egui::FontFamily::Monospace).unwrap().push("glow_sans".to_owned());
            egui_ctx.set_fonts(fonts);
        }

        // 3. 使用刚才创建的局部变量 egui_ctx 来初始化 egui_state
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),           // 注意这里直接传局部变量
            egui::ViewportId::ROOT,     // 修正：使用 ROOT 比较标准
            &window,
            None,
            None,
        );

        // 1. 在 device 还没被移交进 Self 之前，先用它初始化 egui_renderer
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // 2. 最后再统一构造 Self
        Self {
            window,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            compute_pipeline,
            compute_bind_group_layout,
            render_bind_group_layout,
            compute_bind_group,
            render_bind_group,
            scaffold_render_pipeline,
            output_texture,
            output_texture_view,
            params_buffer,
            primitives_buffer,
            anchor_buffer,
            grid_buffer,
            scaffold_buffer,
            params,
            camera: math::Camera::new(glam::Vec3::new(0.0, 1.0, -5.0), 0.0, 0.0),
            is_mmb_pressed: false,
            is_shift_pressed: false,
            last_mouse_pos: [0.0, 0.0],
            // WASD 键盘状态
            is_w_pressed: false,
            is_a_pressed: false,
            is_s_pressed: false,
            is_d_pressed: false,
            // FPS 跟踪
            last_frame_time: std::time::Instant::now(),
            fps: 0.0,
            egui_ctx,
            egui_state,
            egui_renderer,
            primitives: Vec::new(),
            scaffold_vertices: Vec::new(),
            anchors: Vec::new(),
        }
    }

    fn update(&mut self, delta_time: f32) {
        // WASD 移动逻辑
        let move_speed = 2.0 * delta_time;
        if self.is_w_pressed {
            self.camera.eye += self.camera.get_forward() * move_speed;
        }
        if self.is_s_pressed {
            self.camera.eye -= self.camera.get_forward() * move_speed;
        }
        if self.is_a_pressed {
            self.camera.eye -= self.camera.get_right() * move_speed;
        }
        if self.is_d_pressed {
            self.camera.eye += self.camera.get_right() * move_speed;
        }

        // 使用新的相机类生成矩阵
        let view_matrix = self.camera.get_view_matrix();
        self.params.view_inv = view_matrix.inverse().to_cols_array_2d();

        let aspect_ratio = self.config.width as f32 / self.config.height as f32;
        let proj_matrix = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 1000.0);
        self.params.proj_inv = proj_matrix.inverse().to_cols_array_2d();

        self.params.cam_pos = self.camera.eye.extend(1.0).to_array();
        self.params.time += delta_time;
        
        // 计算视图投影矩阵用于点云渲染
        let view_proj = proj_matrix * view_matrix;
        self.params.prev_view_proj = view_proj.to_cols_array_2d();

        // 计算 FPS
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_frame_time).as_secs_f32();
        if elapsed > 0.0 {
            self.fps = 1.0 / elapsed;
        }
        self.last_frame_time = now;
        
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    fn render(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);

        let mut import_clicked = false;
        let mut bake_clicked = false;
        let mut add_sphere = false;
        let mut add_box = false;

        // FPS 显示 (右上角)
        egui::Window::new("FPS")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .fixed_pos(egui::Pos2::new(self.config.width as f32 - 150.0, 10.0))
            .show(&self.egui_ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("FPS: {:.1}", self.fps));
                    ui.label(format!("点数: {}", self.params.scaffold_count));
                });
            });

        egui::SidePanel::left("panel").show(&self.egui_ctx, |ui| {
            ui.heading("CrSculpt 创作台");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("➕ 球体").clicked() {
                    add_sphere = true;
                }
                if ui.button("➕ 立方体").clicked() {
                    add_box = true;
                }
            });

            ui.separator();
            ui.label("几何体列表:");

            let mut changed = false;
            let mut to_remove = None;

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (i, prim) in self.primitives.iter_mut().enumerate() {
                    ui.push_id(i, |ui| {
                        ui.collapsing(format!("几何体 #{}", i), |ui| {
                            // 1. 位置控制 (提取位移)
                            let mut pos = glam::Mat4::from_cols_array_2d(&prim.inv_model_matrix).inverse().transform_point3(glam::Vec3::ZERO);
                            ui.horizontal(|ui| {
                                ui.label("位置:");
                                if ui.add(egui::DragValue::new(&mut pos.x).speed(0.1)).changed() ||
                                   ui.add(egui::DragValue::new(&mut pos.y).speed(0.1)).changed() ||
                                   ui.add(egui::DragValue::new(&mut pos.z).speed(0.1)).changed() {
                                    prim.inv_model_matrix = glam::Mat4::from_translation(pos).inverse().to_cols_array_2d();
                                    changed = true;
                                }
                            });

                            // 2. 尺寸控制
                            if ui.add(egui::Slider::new(&mut prim.params[0], 0.1..=5.0).text("尺寸/半径")).changed() { changed = true; }
                            
                            // 3. 平滑度
                            if ui.add(egui::Slider::new(&mut prim.params[2], 0.01..=2.0).text("平滑融合")).changed() { changed = true; }

                            // 4. 材质颜色
                            if ui.color_edit_button_rgba_unmultiplied(&mut prim.color).changed() { changed = true; }

                            if ui.button("🗑 删除").clicked() { to_remove = Some(i); changed = true; }
                        });
                    });
                }
            });

            if let Some(idx) = to_remove { self.primitives.remove(idx); }

            if changed {
                // 更新 Primitives 缓冲区
                self.queue.write_buffer(&self.primitives_buffer, 0, bytemuck::cast_slice(&self.primitives));
                self.params.prim_count = self.primitives.len() as u32;
            }

            ui.separator();
            if ui.button("📂 导入 GLB").clicked() { import_clicked = true; }
            if ui.button("🔥 一键烘焙 (Bake)").clicked() { bake_clicked = true; }
        });

        // 处理按钮点击事件
        let mut primitives_changed = false;
        if add_sphere {
            let pos = self.camera.eye + self.camera.get_forward() * 5.0;
            self.primitives.push(Primitive::new_sphere(pos, 1.0));
            primitives_changed = true;
        }
        if add_box {
            let pos = self.camera.eye + self.camera.get_forward() * 5.0;
            self.primitives.push(Primitive::new_box(pos, 1.0));
            primitives_changed = true;
        }
        if primitives_changed {
            // 更新 Primitives 缓冲区
            self.queue.write_buffer(&self.primitives_buffer, 0, bytemuck::cast_slice(&self.primitives));
            self.params.prim_count = self.primitives.len() as u32;
        }
        if import_clicked {
            self.import_scaffold();
        }
        if bake_clicked {
            self.bake_anchors();
        }

        let full_output = self.egui_ctx.end_frame();
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        // 核心修复 1：处理 UI 纹理更新 (字体、图标)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }



        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        
        // --- [Pass B: Render 画出点云脚手架] ---
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Draw Scaffold"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // 核心：在模型之上叠加
                        store: wgpu::StoreOp::Store
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            rpass.set_pipeline(&self.scaffold_render_pipeline);
            rpass.set_bind_group(0, &self.compute_bind_group, &[]); // 复用含 scaffold 的绑定组
            
            // 一次性画出所有点！即便有 10 万个点，对显卡来说也只是 0.1ms 的事
            if self.params.scaffold_count > 0 {
                rpass.draw(0..self.params.scaffold_count, 0..1);
            }
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_descriptor);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer.render(&mut rpass, &paint_jobs, &screen_descriptor);
        }

        // 核心修复 2：处理纹理释放
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // 1. 重建输出纹理
            self.output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Output Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.output_texture_view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 2. 关键修复：重新创建 BindGroup，否则它们引用的还是旧视图
            self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.compute_bind_group_layout, // 需要把 Layout 存进 struct
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.output_texture_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.primitives_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.anchor_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.grid_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.scaffold_buffer.as_entire_binding() },
                ],
                label: None,
            });

            self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.output_texture_view) },
                ],
                label: None,
            });
        }
    }

    // 导入 GLB 顶点
    fn load_glb(&self, path: &str) -> Vec<Vec3> {
        let (document, buffers, _) = gltf::import(path).unwrap();
        let mut unique_verts = Vec::new();
        
        // 使用哈希集来过滤重复坐标
        // 关键：由于浮点数存在精度误差，我们将坐标乘以 1000 转为整数进行匹配
        let mut seen_positions = std::collections::HashSet::new();

        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                if let Some(pos_iter) = reader.read_positions() {
                    for p in pos_iter {
                        // 生成坐标指纹 (x,y,z 放大一千倍并取整)
                        let fingerprint = (
                            (p[0] * 1000.0) as i32,
                            (p[1] * 1000.0) as i32,
                            (p[2] * 1000.0) as i32,
                        );

                        if seen_positions.insert(fingerprint) {
                            unique_verts.push(Vec3::from(p));
                        }
                    }
                }
            }
        }
        
        println!("GLB 原始点数: {}, 物理去重后: {}", seen_positions.len(), unique_verts.len());
        unique_verts
    }

    // 计算点到基础几何体的最小 SDF 距离
    fn calculate_min_base_sdf(&self, p: Vec3) -> f32 {
        // 核心修正：显式标注 f32
        let mut min_d: f32 = 1000.0;

        // 遍历目前场景中摆放的所有几何体
        for prim in &self.primitives {
            let inv_mat = glam::Mat4::from_cols_array_2d(&prim.inv_model_matrix);
            // 将脚手架顶点 p 转换到该几何体的局部空间
            let local_p = inv_mat.transform_point3(p);

            let type_id = prim.params[3] as u32;
            let d: f32 = match type_id {
                0 => { // 球体
                    local_p.length() - prim.params[0]
                }
                1 => { // 胶囊体/立方体（根据你的 params 定义扩展）
                    let q = local_p.abs() - glam::Vec3::splat(prim.params[0]);
                    q.max(glam::Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0) - prim.params[1]
                }
                _ => 1000.0,
            };
            
            // 我们需要找到离这个顶点最近的那个基座，计算距离
            min_d = min_d.min(d);
        }
        
        // 如果一个基座都没有，默认给个大的距离
        if self.primitives.is_empty() { return 1000.0; }
        
        min_d
    }

    fn bake_anchors(&mut self) {
        if self.scaffold_vertices.is_empty() || self.primitives.is_empty() {
            println!("警告：没有脚手架或基座，无法烘焙！");
            return;
        }

        // 1. 计算几何中心 (Center of Mass)
        let mut center = glam::Vec3::ZERO;
        for v in &self.scaffold_vertices {
            center += *v;
        }
        center /= self.scaffold_vertices.len() as f32;
        self.params.model_center = center.extend(1.0).to_array();

        // 2. 基础位移烘焙 (使用法线投影)
        let mut raw_anchors = Vec::new();
        for &v in self.scaffold_vertices.iter().take(20000) {
            // 核心修正：不但算距离，还要算基座在该点的法线方向
            let d_base = self.calculate_min_base_sdf(v);

            raw_anchors.push(Anchor {
                pos: [v.x, v.y, v.z, 0.2], // 半径缩回 0.2，提高局部精度
                offset_attr: [-d_base, 0.0, 1.0, 0.0], // SDF偏移，烘焙时暂存
            });
        }

        // 3. --- 核心：构建 32x32x32 空间网格 ---
        let grid_size = 32usize;
        let world_size = 4.0f32; // 建模区域 4x4x4
        let cell_len = world_size / grid_size as f32;
        let origin = glam::Vec3::splat(-2.0); // 网格起点在 -2.0

        let mut buckets: Vec<Vec<Anchor>> = vec![Vec::new(); grid_size.pow(3)];

        // 多重注入：将锚点同时注入到相邻格子（防止边界断裂）
        for a in raw_anchors {
            let p = glam::Vec3::from_slice(&a.pos[0..3]);
            let c = ((p - origin) / cell_len).floor();
            let ix = c.x as i32; let iy = c.y as i32; let iz = c.z as i32;

            // 注入到当前格子和相邻的 7 个格子（共 8 格）
            for dx in 0..2 {
                for dy in 0..2 {
                    for dz in 0..2 {
                        let nx = ix + dx - 1;
                        let ny = iy + dy - 1;
                        let nz = iz + dz - 1;

                        if nx >= 0 && nx < 32 && ny >= 0 && ny < 32 && nz >= 0 && nz < 32 {
                            let idx = (nz as usize * 1024 + ny as usize * 32 + nx as usize);
                            buckets[idx].push(a);
                        }
                    }
                }
            }
        }

        // 4. 将桶装数据压扁为排序后的数组
        let mut sorted_anchors = Vec::new();
        let mut grid_cells = vec![math::GridCell { offset: 0, count: 0 }; grid_size.pow(3)];

        for (i, bucket) in buckets.iter().enumerate() {
            grid_cells[i].offset = sorted_anchors.len() as u32;
            grid_cells[i].count = bucket.len() as u32;
            sorted_anchors.extend(bucket);
        }

        // 5. 上传到 GPU
        self.queue.write_buffer(&self.anchor_buffer, 0, bytemuck::cast_slice(&sorted_anchors));
        self.queue.write_buffer(&self.grid_buffer, 0, bytemuck::cast_slice(&grid_cells));

        self.anchors = sorted_anchors;
        self.params.anchor_count = sorted_anchors.len() as u32;
        self.params.grid_origin = [origin.x, origin.y, origin.z, cell_len];
        
        println!("烘焙完成：生成了 {} 个细节锚点", self.params.anchor_count);
    }

    // 打开文件对话框并导入 GLB
    fn import_scaffold(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GLB Files", &["glb"])
            .add_filter("GLTF Files", &["gltf"])
            .pick_file() {
            
            let verts = self.load_glb(path.to_str().unwrap());
            self.scaffold_vertices = verts;

            // 核心修复 1：将原始顶点转化为 vec4 (x, y, z, 1.0) 传给显存
            let scaffold_data: Vec<glam::Vec4> = self.scaffold_vertices.iter()
                .map(|v| v.extend(1.0))
                .collect();

            self.queue.write_buffer(&self.scaffold_buffer, 0, bytemuck::cast_slice(&scaffold_data));
            
            // 更新参数
            self.params.scaffold_count = scaffold_data.len() as u32;
            
            println!("脚手架点云上传成功：{} 个点", self.params.scaffold_count);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(winit::window::Window::new(&event_loop).unwrap());
    window.set_title("CrSculpt");

    let mut app = pollster::block_on(App::new(window.clone()));

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == app.window.id() => {
                // 让 egui 优先处理（如果点在 UI 上，不触发相机操作）
                if app.egui_state.on_window_event(&app.window, event).consumed { return; }

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => app.resize(*size),
                    
                    // --- 键盘监听：Shift 和 WASD ---
                    WindowEvent::KeyboardInput { event: kb_event, .. } => {
                        match &kb_event.logical_key {
                            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift) => {
                                app.is_shift_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            winit::keyboard::Key::Character(c) if c == "w" => {
                                app.is_w_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            winit::keyboard::Key::Character(c) if c == "a" => {
                                app.is_a_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            winit::keyboard::Key::Character(c) if c == "s" => {
                                app.is_s_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            winit::keyboard::Key::Character(c) if c == "d" => {
                                app.is_d_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            _ => {}
                        }
                    }

                    // --- 鼠标点击监听：中键 ---
                    WindowEvent::MouseInput { button, state, .. } => {
                        if *button == winit::event::MouseButton::Middle {
                            app.is_mmb_pressed = *state == winit::event::ElementState::Pressed;
                        }
                    }

                    // --- 鼠标移动监听：旋转与平移 ---
                    WindowEvent::CursorMoved { position, .. } => {
                        let dx = position.x as f32 - app.last_mouse_pos[0];
                        let dy = position.y as f32 - app.last_mouse_pos[1];

                        if app.is_mmb_pressed {
                            if app.is_shift_pressed {
                                app.camera.pan(dx, dy); // Shift + MMB = Pan
                            } else {
                                app.camera.rotate(dx, dy); // MMB = Orbit
                            }
                        }

                        app.last_mouse_pos = [position.x as f32, position.y as f32];
                    }

                    // --- 滚轮监听：缩放 ---
                    WindowEvent::MouseWheel { delta, .. } => {
                        if let winit::event::MouseScrollDelta::LineDelta(_, y) = delta {
                            app.camera.zoom(*y);
                        }
                    }
                    _ => {}
                }
            },
            Event::AboutToWait => {
                let now = std::time::Instant::now();
                let delta_time = (now - last_time).as_secs_f32();
                last_time = now;

                app.update(delta_time);
                app.render();
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}