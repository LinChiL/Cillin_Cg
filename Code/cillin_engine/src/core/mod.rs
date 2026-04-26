use std::sync::Arc;
use wgpu::{QuerySet, Buffer};
use winit::window::Window;

use crate::{rendering::RenderContext, resources::AssetManager, camera::{Camera, CameraController}, input::InputState, scene::{Entity, TileSystem}, console::Console, editor::EditorState};
use crate::editor;

const QUERY_BUFFER_COUNT: usize = 3; // 三倍缓冲
pub const MAP_STATE_IDLE: u8 = 0;
pub const MAP_STATE_PENDING: u8 = 1;
pub const MAP_STATE_READY: u8 = 2;

pub struct CoreState<'a> {
    pub window: Arc<Window>,
    pub render_context: RenderContext<'a>,
    
    pub assets: AssetManager,
    
    pub camera: Camera,
    pub controller: CameraController,
    pub input_state: InputState,
    
    pub entities: Vec<Entity>,
    pub instance_counters: std::collections::HashMap<u32, usize>,
    
    pub console: Console,
    
    pub editor: EditorState,
    
    pub shadows_enabled: bool,
    pub ao_enabled: bool,
    pub debug_mode: u32,
    
    pub output_texture: wgpu::Texture,
    pub output_texture_view: wgpu::TextureView,
    pub history_texture: wgpu::Texture,
    pub history_texture_view: wgpu::TextureView,
    pub compute_bind_group: wgpu::BindGroup,
    pub blit_bind_group: wgpu::BindGroup,
    pub params_buffer: wgpu::Buffer,
    pub entity_list_buffer: wgpu::Buffer,
    pub compute_bind_group_layout: wgpu::BindGroupLayout,
    pub blit_bind_group_layout: wgpu::BindGroupLayout,
    
    pub tile_system: TileSystem,
    
    pub query_set: QuerySet,
    pub query_buffer: Buffer,
    pub mapped_buffers: Vec<Buffer>,
    pub active_query_index: usize,
    pub buffer_ready_flags: Vec<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    pub is_perf_mode: bool,
    pub last_compute_time: f32,
    pub last_render_time: f32,
    pub frame_count: u32,
    pub last_perf_print: std::time::Instant,
    pub start_time: std::time::Instant,
    pub prev_view_proj: glam::Mat4,
    pub frame_index: u32,
    pub is_moving: bool,
}

impl<'a> CoreState<'a> {
    pub async fn new(window: Arc<Window>) -> Self {
        use wgpu::util::DeviceExt;
        
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 4,
            ty: wgpu::QueryType::Timestamp,
            label: Some("Perf Query Set"),
        });
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // 创建三倍缓冲池
        let mut mapped_buffers = Vec::new();
        let mut buffer_ready_flags = Vec::new();

        for i in 0..QUERY_BUFFER_COUNT {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Mapped Query Buffer {}", i)),
                size: 32, // 4个u64
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            mapped_buffers.push(buffer);
            buffer_ready_flags.push(std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAP_STATE_IDLE)));
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .find(|f| **f == wgpu::TextureFormat::Bgra8UnormSrgb)
            .copied()
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let assets = crate::resources::AssetManager::new(&device, &queue).await;

        let entities: Vec<Entity> = std::fs::read_to_string(crate::resources::SCENE_FILE)
            .map(|raw| {
                let scene_data: crate::scene::SceneData = serde_json::from_str(&raw).unwrap();
                scene_data.entities.into_iter().map(|mut entity| {
                    if entity.code.is_empty() {
                        let instance_index = 0usize;
                        entity.code = format!("{}s{:06}", entity.model_id, instance_index);
                    }
                    entity
                }).collect()
            })
            .unwrap_or_default();

        let mut instance_counters: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for entity in &entities {
            *instance_counters.entry(entity.model_id).or_insert(0) += 1;
        }

        let camera_pos = glam::Vec3::new(3.0, 3.0, 3.0);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), size.width as f32 / size.height as f32, 0.1, 10000.0);
        let camera_matrix = proj * view;
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera"),
            contents: bytemuck::cast_slice(camera_matrix.as_ref()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
            label: None,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
            label: None,
        });

        let light_uniform = crate::rendering::LightUniform {
            direction: [-0.5, -0.5, -1.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0, 1.0],
        };
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
            label: None,
        });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: light_buffer.as_entire_binding() }],
            label: None,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Brain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Eye Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Xi-Luoer Brain Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Display Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_blit",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_blit",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 创建历史纹理，用于时域积累
        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("History Texture"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_texture_view = history_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let entity_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entity List Buffer"),
            size: (2048 * 256) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = crate::rendering::Params {
            view_inv: view.inverse().to_cols_array_2d(),
            proj_inv: proj.inverse().to_cols_array_2d(),
            prev_view_proj: proj.to_cols_array_2d(),
            cam_pos: camera_pos.extend(1.0).to_array(),
            light_dir: glam::Vec3::new(0.5, 1.0, 0.5).normalize().extend(0.0).to_array(),
            entity_count: entities.len() as u32,
            debug_mode: 0,
            time: 0.0,
            frame_index: 0,
            is_moving: 0,
            _unused_pad: [0, 0, 0],
            _final_padding: [0; 32],
        };
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: 384, // 强制指定 384 字节
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        let tile_system = TileSystem::new(&device);

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: entity_list_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&assets.global_sdf_view) },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: assets.palette_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: tile_system.tile_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&history_texture_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            ],
            label: Some("Compute Bind Group"),
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_texture_view) },
            ],
            label: Some("Blit Bind Group"),
        });

        let camera = Camera::new(
            glam::Vec3::new(-23.289192, 25.579927, -51.22953),
            -287.58f32.to_radians(),
            -6.18f32.to_radians(),
        );
        let controller = CameraController::new(15.0);
        let input_state = InputState::new();

        let view = camera.get_view_matrix();
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), size.width as f32 / size.height as f32, 0.1, 10000.0);

        let render_context = RenderContext::new(
            surface,
            device,
            queue,
            render_pipeline,
            Some(compute_pipeline),
            camera_bind_group,
            light_bind_group,
            None,
            depth_texture,
            depth_view,
            size,
            format,
        );

        // 设置初始的 view 和 proj 矩阵
        let mut render_context = render_context;
        render_context.current_view = view;
        render_context.current_proj = proj;

        Self {
            window,
            render_context,
            assets,
            camera,
            controller,
            input_state,
            entities,
            instance_counters,
            console: Console::new(),
            editor: EditorState::new(),
            shadows_enabled: true,
            ao_enabled: true,
            debug_mode: 0,
            output_texture,
            output_texture_view,
            history_texture,
            history_texture_view,
            compute_bind_group,
            blit_bind_group,
            params_buffer,
            entity_list_buffer,
            compute_bind_group_layout,
            blit_bind_group_layout,
            tile_system,
            query_set,
            query_buffer,
            mapped_buffers,
            active_query_index: 0,
            buffer_ready_flags,
            is_perf_mode: false,
            last_compute_time: 0.0,
            last_render_time: 0.0,
            frame_count: 0,
            last_perf_print: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            prev_view_proj: proj, // 初始化为当前的 proj * view
            frame_index: 0,
            is_moving: false,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.render_context.resize(new_size);

            let device = &self.render_context.device;

            let new_output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let new_output_texture_view = new_output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 更新历史纹理
        let new_history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("History Texture"),
            size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let new_history_texture_view = new_history_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let new_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&new_output_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.entity_list_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.assets.global_sdf_view) },
                wgpu::BindGroupEntry { binding: 4, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.assets.palette_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.tile_system.tile_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&new_history_texture_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            ],
            label: Some("Compute Bind Group"),
        });

            let new_blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&new_output_texture_view) },
                ],
                label: Some("Blit Bind Group"),
            });

            self.output_texture = new_output_texture;
            self.output_texture_view = new_output_texture_view;
            self.history_texture = new_history_texture;
            self.history_texture_view = new_history_texture_view;
            self.compute_bind_group = new_compute_bind_group;
            self.blit_bind_group = new_blit_bind_group;
        }
    }

    pub fn save_scene(&self) {
        let scene_data = crate::scene::SceneData {
            entities: self.entities.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&scene_data) {
            if std::fs::write(crate::resources::SCENE_FILE, json).is_ok() {
                println!("场景已保存");
            }
        }
    }

    pub fn get_click_ground_pos(&self) -> glam::Vec3 {
        editor::get_click_ground_pos(
            self.input_state.mouse_pos,
            self.render_context.size.width,
            self.render_context.size.height,
            self.render_context.current_view,
            self.render_context.current_proj,
            self.camera.eye
        )
    }

    pub fn get_clicked_entity(&self) -> Option<&Entity> {
        let ray_dir = editor::get_ray_from_screen(
            self.input_state.mouse_pos,
            self.render_context.size.width,
            self.render_context.size.height,
            self.render_context.current_view,
            self.render_context.current_proj,
            self.camera.eye
        );
        
        editor::get_clicked_entity(&self.entities, self.camera.eye, ray_dir)
    }

    pub fn undo(&mut self) {
        self.editor.undo(&mut self.entities);
    }

    pub fn redo(&mut self) {
        self.editor.redo(&mut self.entities);
    }

    pub fn execute_command(&mut self) {
        let command = self.console.execute();
        if command.is_empty() {
            self.console.is_open = false;
            return;
        }

        if command == "mem" || command == "vram" {
            let (allocated, _total) = self.assets.get_vram_usage(
                self.render_context.size.width,
                self.render_context.size.height
            );
            let tile_vram = self.tile_system.get_vram_usage();
            
            println!("┌─────────────────────────────────────────┐");
            println!("│           VRAM USAGE REPORT             │");
            println!("├─────────────────────────────────────────┤");
            println!("│ Asset VRAM:      {:>10.2} MB       │", allocated as f64 / 1024.0 / 1024.0);
            println!("│ Tile System:     {:>10.2} MB       │", tile_vram as f64 / 1024.0 / 1024.0);
            println!("│ Estimated Total: {:>10.2} MB       │", (allocated + tile_vram) as f64 / 1024.0 / 1024.0);
            println!("└─────────────────────────────────────────┘");

            self.console.is_open = false;
            return;
        }

        if command == "stat" {
            self.is_perf_mode = !self.is_perf_mode;
            // 重置一下计数器，确保立即能看到输出
            self.frame_count = 0; 
            println!(">>> 性能分析系统已{}", if self.is_perf_mode { "启动" } else { "停止" });
            self.console.is_open = false;
            return;
        }

        if command == "modelinfo" {
            let total_entities = self.entities.len();
            let mut model_counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for e in &self.entities {
                *model_counts.entry(e.model_id).or_insert(0) += 1;
            }

            let mut sorted: Vec<(&u32, &u32)> = model_counts.iter().collect();
            sorted.sort_by_key(|(_, c)| *c);

            println!("┌─────────────────────────────────────────┐");
            println!("│           MODEL INFO REPORT              │");
            println!("├─────────────────────────────────────────┤");
            println!("│ Total Entities:    {:>10}         │", total_entities);
            println!("├─────────────────────────────────────────┤");
            for (model_id, count) in sorted {
                println!("│ Model {:>3}:           {:>10}         │", model_id, count);
            }
            println!("└─────────────────────────────────────────┘");

            self.console.is_open = false;
            return;
        }

        println!("执行命令: {}", command);
        let parts: Vec<&str> = command.split_whitespace().collect();
        if let Some(cmd) = parts.first() {
            match *cmd {
                "help" => println!("可用命令: help, exit, spawn <model_id>, serspawn <model_id>, cmr <x> <y> <z>, cmr now, shadow, ao, debug <0|1|2|3|4|5>, mem, stat, modelinfo"),
                "exit" => println!("控制台已关闭"),
                "spawn" => {
                    if parts.len() >= 2 {
                        if let Ok(model_id) = parts[1].parse::<u32>() {
                            let pos = self.get_click_ground_pos();
                            let instance_index = *self.instance_counters.get(&model_id).unwrap_or(&0);
                            let entity = Entity::new(
                                self.entities.len(),
                                model_id,
                                instance_index,
                                pos,
                                glam::Vec3::ZERO,
                                glam::Vec3::ONE,
                                [1.0, 1.0, 1.0, 1.0],
                            );
                            self.entities.push(entity);
                            *self.instance_counters.entry(model_id).or_insert(0) += 1;
                            println!("已在位置 {:?} 生成模型 {}", pos, model_id);
                        }
                    }
                }
                "serspawn" => {
                    if parts.len() >= 2 {
                        if let Ok(model_id) = parts[1].parse::<u32>() {
                            self.editor.enable_ser_spawn_mode(model_id);
                            println!("已开启连续生成模式，模型 ID: {}", model_id);
                        }
                    }
                }
                "cmr" => {
                    if parts.len() >= 2 && parts[1] == "now" {
                        println!("摄像机位置: ({}, {}, {})", self.camera.eye.x, self.camera.eye.y, self.camera.eye.z);
                        println!("摄像机角度: yaw={:.2}°, pitch={:.2}°", self.camera.yaw.to_degrees(), self.camera.pitch.to_degrees());
                    } else if parts.len() >= 4 {
                        if let (Ok(x), Ok(y), Ok(z)) = (parts[1].parse::<f32>(), parts[2].parse::<f32>(), parts[3].parse::<f32>()) {
                            self.camera.eye = glam::Vec3::new(x, y, z);
                        }
                    }
                }
                "shadow" => {
                    self.shadows_enabled = !self.shadows_enabled;
                    println!("影子已{}", if self.shadows_enabled { "开启" } else { "关闭" });
                }
                "ao" => {
                    self.ao_enabled = !self.ao_enabled;
                    println!("环境光遮蔽已{}", if self.ao_enabled { "开启" } else { "关闭" });
                }
                "debug" => {
                    if parts.len() >= 2 {
                        if let Ok(mode) = parts[1].parse::<u32>() {
                            self.debug_mode = mode;
                        }
                    }
                }
                _ => println!("未知命令: {}", cmd),
            }
        }
        self.console.is_open = false;
    }

    pub fn update(&mut self) {
        // 使用真实时间
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_perf_print).as_secs_f32();
        
        self.controller.is_up_pressed = self.input_state.is_w_pressed;
        self.controller.is_down_pressed = self.input_state.is_s_pressed;
        self.controller.is_left_pressed = self.input_state.is_a_pressed;
        self.controller.is_right_pressed = self.input_state.is_d_pressed;
        self.controller.is_q_pressed = self.input_state.is_q_pressed;
        self.controller.is_e_pressed = self.input_state.is_e_pressed;

        // 3. 更新相机物理位置
        let old_pos = self.camera.eye;
        self.controller.update(&mut self.camera, dt);
        
        // 4. 计算位移速度（用于 Shader 决策）
        let movement_speed = (self.camera.eye - old_pos).length();
        self.is_moving = movement_speed > 0.001;
        
        // 5. 更新矩阵
        let view = self.camera.get_view_matrix();
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), self.render_context.size.width as f32 / self.render_context.size.height as f32, 0.1, 10000.0);
        
        self.render_context.current_view = view;
        self.render_context.current_proj = proj;
        
        // 更新计时基准
        self.last_perf_print = now;

        if self.input_state.is_delete_pressed && self.editor.selected_idx.is_some() {
            let idx = self.editor.selected_idx.unwrap();
            if idx < self.entities.len() {
                println!("已删除模型: ID={}", self.entities[idx].id);
                self.entities.remove(idx);
                for (i, entity) in self.entities.iter_mut().enumerate() {
                    entity.id = i;
                }
                self.editor.select_entity(None);
            }
        }

        if self.editor.is_ser_spawn_mode && self.input_state.is_left_mouse_pressed {
            let pos = self.get_click_ground_pos();
            let model_id = self.editor.ser_spawn_model_id;
            let instance_index = *self.instance_counters.get(&model_id).unwrap_or(&0);
            let entity = Entity::new(
                self.entities.len(),
                model_id,
                instance_index,
                pos,
                glam::Vec3::ZERO,
                glam::Vec3::ONE,
                [1.0, 1.0, 1.0, 1.0],
            );
            self.entities.push(entity);
            *self.instance_counters.entry(model_id).or_insert(0) += 1;
        }

        self.tile_system.resize_if_needed(&self.render_context.device, self.render_context.size.width, self.render_context.size.height);

        // 更新 Tile 系统并获取实体数据
        let entity_data = self.tile_system.update_tile_buffer(
            &self.entities,
            &self.assets.models,
            self.camera.eye,
            self.render_context.current_view,
            self.render_context.current_proj,
            &self.render_context.queue,
            self.render_context.size.width,
            self.render_context.size.height,
        );

        // 更新实体列表缓冲区
        if !entity_data.is_empty() {
            self.render_context.queue.write_buffer(&self.entity_list_buffer, 0, bytemuck::cast_slice(&entity_data));
        }

        // 重置输入状态
        self.input_state.reset();
    }

    pub fn update_instance_buffer(&mut self, model_id: u32) {
        let model_asset = match self.assets.models.get(&model_id) {
            Some(m) => m,
            None => return,
        };

        let instances: Vec<crate::scene::InstanceRaw> = self.entities.iter()
            .filter(|e| e.model_id == model_id)
            .map(|e| e.to_raw())
            .collect();

        let needed_capacity = instances.len() * std::mem::size_of::<crate::scene::InstanceRaw>();
        if model_asset.instance_buffer.size() < needed_capacity as u64 {
        }

        self.render_context.queue.write_buffer(
            &model_asset.instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );
    }

    pub fn print_perf_stats(&self) {
        // --- 本机实际帧率 ---
        let total_frame_time = self.last_compute_time + self.last_render_time;
        let native_fps = if total_frame_time > 0.0 { 1000.0 / total_frame_time } else { 0.0 };

        // --- 性能投影计算 ---
        // 假设当前显卡与 2010 年显卡的代际算力差距为 70 倍（保守估计）
        let generation_gap = 70.0;
        let projected_legacy_brain = self.last_compute_time * generation_gap;
        let projected_fps = 1000.0 / (projected_legacy_brain + self.last_render_time + 1.0); // +1.0 为驱动开销

        println!("--- Cillin Engine Perf Stats (Analysis) ---");
        println!("GPU Brain (Now):  {:.3} ms", self.last_compute_time);
        println!("Native FPS:       {:.1} FPS", native_fps);

        // 关键：输出老机器的预测数据
        println!("-------------------------------------------");
        println!("Projected for Legacy Device (2010):");
        println!("Estimated Time:   {:.2} ms / frame", projected_legacy_brain);

        if projected_fps < 15.0 {
            println!("Estimated FPS:    {:.1} FPS (❌ UNPLAYABLE)", projected_fps);
        } else if projected_fps < 30.0 {
            println!("Estimated FPS:    {:.1} FPS (⚠️ STRUGGLING)", projected_fps);
        } else {
            println!("Estimated FPS:    {:.1} FPS (✅ OK)", projected_fps);
        }
        println!("-------------------------------------------");
    }

    pub fn process_query_results_async(&mut self) {
        // 始终 poll 以推进回调
        self.render_context.device.poll(wgpu::Maintain::Poll);
        
        if !self.is_perf_mode { return; }
        
        let mut has_data = false;
        for i in 0..QUERY_BUFFER_COUNT {
            let state = self.buffer_ready_flags[i].load(std::sync::atomic::Ordering::Acquire);
            if state == MAP_STATE_READY {
                let buffer_slice = self.mapped_buffers[i].slice(..);
                {
                    let data = buffer_slice.get_mapped_range();
                    let ts: &[u64] = bytemuck::cast_slice(&data);
                    if ts.len() >= 4 {
                        self.last_compute_time = ts[1].saturating_sub(ts[0]) as f32 / 1_000_000.0;
                        self.last_render_time = ts[3].saturating_sub(ts[2]) as f32 / 1_000_000.0;
                        self.frame_count += 1;
                        has_data = true;
                    }
                    drop(data);
                }
                // 解除映射并重置状态为 IDLE
                self.mapped_buffers[i].unmap();
                self.buffer_ready_flags[i].store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
            }
        }
        
        if has_data && self.frame_count % 20 == 0 {
            self.print_perf_stats();
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 1. 更新参数 Buffer
        let time = self.start_time.elapsed().as_secs_f32();
        let view_proj = self.render_context.current_proj * self.render_context.current_view;
        
        // 第一次运行如果没有 prev_vp，就设为 current_vp
        if self.frame_index == 0 {
            self.prev_view_proj = view_proj;
        }
        
        let params = crate::rendering::Params {
            view_inv: self.render_context.current_view.inverse().to_cols_array_2d(),
            proj_inv: self.render_context.current_proj.inverse().to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            cam_pos: self.camera.eye.extend(1.0).to_array(),
            light_dir: glam::Vec3::new(0.5, 1.0, 0.5).normalize().extend(0.0).to_array(),
            entity_count: self.entities.len() as u32,
            debug_mode: self.debug_mode,
            time: 0.0, // 暂时没用到可设为0
            frame_index: self.frame_index,
            is_moving: if self.is_moving { 1u32 } else { 0u32 },
            _unused_pad: [0, 0, 0],
            _final_padding: [0; 32],
        };
        self.render_context.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // 2. 创建 Command Encoder
        let mut encoder = self.render_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // 3. 执行 Compute Pass 并记录时间戳
        encoder.write_timestamp(&self.query_set, 0);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(self.render_context.compute_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.render_context.size.width + 7) / 8, (self.render_context.size.height + 7) / 8, 1);
        }
        encoder.write_timestamp(&self.query_set, 1);

        // 4. 执行 Render Pass 并记录时间戳
        encoder.write_timestamp(&self.query_set, 2);
        let output = self.render_context.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rpass.set_pipeline(&self.render_context.render_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        encoder.write_timestamp(&self.query_set, 3);

        // 将当前帧的输出复制到历史纹理
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.history_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.render_context.size.width,
                height: self.render_context.size.height,
                depth_or_array_layers: 1,
            },
        );

        // --- 性能数据抓取逻辑 ---
        if self.is_perf_mode {
            let current_idx = self.active_query_index;
            let state = self.buffer_ready_flags[current_idx].load(std::sync::atomic::Ordering::Acquire);
            
            // 只有 IDLE 状态才能发起新的查询
            if state == MAP_STATE_IDLE {
                // 尝试原子地改为 PENDING
                match self.buffer_ready_flags[current_idx].compare_exchange(
                    MAP_STATE_IDLE,
                    MAP_STATE_PENDING,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // 解析查询结果并拷贝到缓冲区
                        encoder.resolve_query_set(&self.query_set, 0..4, &self.query_buffer, 0);
                        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.mapped_buffers[current_idx], 0, 32);
                        self.render_context.queue.submit(std::iter::once(encoder.finish()));
                        
                        // 发起异步映射
                        let ready_flag = std::sync::Arc::clone(&self.buffer_ready_flags[current_idx]);
                        self.mapped_buffers[current_idx].slice(..).map_async(wgpu::MapMode::Read, move |res| {
                            if res.is_ok() {
                                ready_flag.store(MAP_STATE_READY, std::sync::atomic::Ordering::Release);
                            } else {
                                // 映射失败，回退到 IDLE
                                ready_flag.store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
                            }
                        });
                        
                        // 轮换到下一个缓冲区
                        self.active_query_index = (current_idx + 1) % QUERY_BUFFER_COUNT;
                    },
                    Err(_) => {
                        // 状态不是 IDLE，跳过本帧的性能采集，直接提交普通渲染命令
                        self.render_context.queue.submit(std::iter::once(encoder.finish()));
                    }
                }
            } else {
                // 缓冲区正忙，跳过性能采集，仅提交渲染
                self.render_context.queue.submit(std::iter::once(encoder.finish()));
            }
        } else {
            // 性能模式关闭，正常提交
            self.render_context.queue.submit(std::iter::once(encoder.finish()));
        }

        output.present();
        
        // 每帧都尝试处理（它内部有 is_perf_mode 判断）
        self.process_query_results_async();
        
        // 更新上一帧的 view_proj 和帧计数器
        self.prev_view_proj = view_proj;
        self.frame_index += 1;

        Ok(())
    }
}
