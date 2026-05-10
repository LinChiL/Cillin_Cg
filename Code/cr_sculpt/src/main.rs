use std::borrow::Cow;
use std::sync::Arc;

use glam::{Vec3, Vec4, Vec3Swizzles, Vec4Swizzles};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use gltf;
use rfd;

mod math;
use math::{Params, Primitive, MeshSample, Triangle};

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
    depth_blit_bind_group_layout: wgpu::BindGroupLayout, // 新增
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    scaffold_render_pipeline: wgpu::RenderPipeline,
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
    primitives_buffer: wgpu::Buffer,
    grid_buffer: wgpu::Buffer,
    scaffold_buffer: wgpu::Buffer,
    triangle_buffer: wgpu::Buffer,
    voronoi_texture: wgpu::Texture,
    voronoi_texture_view: wgpu::TextureView,
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
    scaffold_path: Option<String>,
    triangles: Vec<math::Triangle>,
    
    // 新增：唯一顶点 + 邻接信息（用于背面剔除）
    vertex_positions: Vec<glam::Vec3>,     // 唯一顶点
    vertex_triangles: Vec<Vec<u32>>,       // 每个顶点连接的三角形索引列表
    
    show_scaffold: bool, // 控制是否显示点云
    show_voronoi_debug: bool,   // 新增：圆盘调试模式开关
    show_sphere_debug: bool,    // 新增：圆球调试模式开关
    visible_vertices: Vec<glam::Vec4>,   // 屏幕空间 + 深度 (x,y,depth,1.0)
    
    // 深度图调试
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    depth_texture_view_for_render: wgpu::TextureView,
    tri_id_texture: wgpu::Texture,
    tri_id_texture_view: wgpu::TextureView,
    tri_id_texture_view_for_render: wgpu::TextureView,
    depth_bind_group_layout: wgpu::BindGroupLayout,
    depth_bind_group: wgpu::BindGroup,
    depth_render_pipeline: wgpu::RenderPipeline,
    depth_blit_pipeline: wgpu::RenderPipeline,
    depth_blit_bind_group: wgpu::BindGroup,
    show_depth_debug: bool,
    show_normal_debug: bool,
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

        let params = Params::default();

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

        // 三角形缓冲区 (最多 10 万个三角形 * 48 字节)
        let triangle_max_size = (100_000 * 48) as u64;
        let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triangle Buffer"),
            size: triangle_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建 Voronoi 纹理（屏幕大小，存储每个像素最近的三角形 ID）
        let voronoi_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voronoi Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint, // 存储 u32 三角形 ID
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let voronoi_texture_view = voronoi_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let depth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Debug Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture_view_for_render = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let tri_id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Triangle ID Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let tri_id_texture_view = tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let tri_id_texture_view_for_render = tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                    resource: triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scaffold_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&tri_id_texture_view),
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

        // === 轻量深度图管线 ===
        let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Pipeline Layout"),
            bind_group_layouts: &[&depth_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Bind Group"),
            layout: &depth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triangle_buffer.as_entire_binding(),
                },
            ],
        });

        let depth_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Render Pipeline"),
            layout: Some(&depth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_depth",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_depth",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let depth_blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Blit Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        let depth_blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Blit Pipeline Layout"),
            bind_group_layouts: &[&depth_blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Blit Pipeline"),
            layout: Some(&depth_blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_depth_blit",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let depth_blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Blit Bind Group"),
            layout: &depth_blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_view),
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
            depth_blit_bind_group_layout,
            compute_bind_group,
            render_bind_group,
            scaffold_render_pipeline,
            output_texture,
            output_texture_view,
            params_buffer,
            primitives_buffer,
            grid_buffer,
            scaffold_buffer,
            triangle_buffer,
            voronoi_texture,
            voronoi_texture_view,
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
            scaffold_path: None,
            triangles: Vec::new(),
            vertex_positions: Vec::new(),     // 初始化唯一顶点
            vertex_triangles: Vec::new(),     // 初始化邻接信息
            show_scaffold: false, // 默认关闭点云，显示三角形表面
            show_voronoi_debug: false,
            show_sphere_debug: false,
            visible_vertices: Vec::new(),
            depth_texture,
            depth_texture_view,
            depth_texture_view_for_render,
            tri_id_texture,
            tri_id_texture_view,
            tri_id_texture_view_for_render,
            depth_bind_group_layout,
            depth_bind_group,
            depth_render_pipeline,
            depth_blit_pipeline,
            depth_blit_bind_group,
            show_depth_debug: false,
            show_normal_debug: false,
        }
    }

    fn update_visible_vertices(&mut self) {
        if self.vertex_positions.is_empty() {
            self.visible_vertices.clear();
            self.params.scaffold_count = 0;
            return;
        }

        let cam_pos = self.camera.eye;
        let view_proj = {
            let view_matrix = self.camera.get_view_matrix();
            let aspect_ratio = self.config.width as f32 / self.config.height as f32;
            let proj_matrix = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 1000.0);
            proj_matrix * view_matrix
        };

        self.visible_vertices.clear();

        let mut visible_count = 0;
        let mut total_count = 0;

        for (vidx, &world_pos) in self.vertex_positions.iter().enumerate() {
            total_count += 1;
            let mut is_visible = false;

            // 检查该顶点关联的所有三角形，只要有一个面向相机就保留
            for &tri_idx in &self.vertex_triangles[vidx] {
                let tri = &self.triangles[tri_idx as usize];

                let p0 = Vec3::from_slice(&tri.v0[0..3]);
                let p1 = Vec3::from_slice(&tri.v1[0..3]);
                let p2 = Vec3::from_slice(&tri.v2[0..3]);

                let center = (p0 + p1 + p2) * (1.0 / 3.0);
                let to_camera = (cam_pos - center).normalize();

                let normal = (p1 - p0).cross(p2 - p0).normalize();

                // 传统背面剔除：dot > 0 表示面向相机
                if normal.dot(to_camera) > -0.02 {   // 轻微放宽阈值，防止边缘抖动
                    is_visible = true;
                    break;
                }
            }

            if is_visible {
                // 投影到屏幕
                let clip = view_proj * world_pos.extend(1.0);
                if clip.w > 0.01 {
                    let ndc = clip / clip.w;
                    let screen_x = (ndc.x * 0.5 + 0.5) * self.config.width as f32;
                    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * self.config.height as f32;
                    let depth = ndc.z * 0.5 + 0.5;

                    self.visible_vertices.push(glam::Vec4::new(screen_x, screen_y, depth, 1.0));
                    visible_count += 1;
                }
            }
        }

        println!("Visible vertices: {}/{}   (cam_pos: {:?})", visible_count, total_count, cam_pos);

        // 上传 GPU
        let upload_data = self.visible_vertices.clone();
        self.queue.write_buffer(&self.scaffold_buffer, 0, bytemuck::cast_slice(&upload_data));
        self.params.scaffold_count = upload_data.len() as u32;
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
        
        // 计算视图投影矩阵用于点云渲染和实时沃洛诺伊更新
        let view_proj = proj_matrix * view_matrix;
        self.params.prev_view_proj = view_proj.to_cols_array_2d(); // 用于点云渲染和实时沃洛诺伊

        // 计算 FPS
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_frame_time).as_secs_f32();
        if elapsed > 0.0 {
            self.fps = 1.0 / elapsed;
        }
        self.last_frame_time = now;
        
        // === 精确计算包围球屏幕空间投影圆盘 ===
        let model_center = glam::Vec3::from_slice(&self.params.model_center[0..3]);
        let base_radius = if !self.primitives.is_empty() && self.primitives[0].params[3] == 0.0 {
            self.primitives[0].params[0]
        } else {
            1.0
        };

        let view_proj = proj_matrix * view_matrix;
        
        // 1. 计算球心在屏幕空间的位置
        let center_clip = view_proj * model_center.extend(1.0);
        let (center_screen, radius_screen) = if center_clip.w <= 0.0 {
            // 球心在相机后面，fallback
            (
                glam::Vec2::new(self.config.width as f32 * 0.5, self.config.height as f32 * 0.5),
                300.0
            )
        } else {
            let center_ndc = center_clip / center_clip.w;
            let center_screen = glam::Vec2::new(
                (center_ndc.x * 0.5 + 0.5) * self.config.width as f32,
                (1.0 - (center_ndc.y * 0.5 + 0.5)) * self.config.height as f32,
            );

            // 2. 使用切线投影计算屏幕空间半径
            let cam_to_center = model_center - self.camera.eye;
            let dist = cam_to_center.length().max(0.001);

            let sphere_screen_radius = if dist > base_radius {
                // 简化版：使用视角和角大小计算
                let half_fov = (45.0f32.to_radians() * 0.5);
                let angular_size = (base_radius / dist).atan();
                (angular_size / half_fov) * (self.config.height as f32 * 0.5)
            } else {
                // 相机在球内部，特殊处理
                800.0 // 很大，基本覆盖全屏
            };

            (center_screen, sphere_screen_radius * 1.0) // 经验放大系数
        };
        
        self.params.disk_center = [center_screen.x, center_screen.y, 0.0, 0.0];
        self.params.disk_radius = radius_screen;
        self.params.base_radius = base_radius;
        
        // 更新可见顶点投影
        self.update_visible_vertices();
        
        // 设置调试模式：0=正常, 1=圆盘调试, 2=深度图模式, 3=圆球调试, 4=法线调试
        if self.show_depth_debug {
            // 注意：我们现在用传统光栅化渲染深度图，不用 compute shader 的 debug_mode 2
            self.params.debug_mode = 0;
        } else if self.show_voronoi_debug {
            self.params.debug_mode = 1;
        } else if self.show_sphere_debug {
            self.params.debug_mode = 3;
        } else if self.show_normal_debug {
            self.params.debug_mode = 4;
        } else {
            self.params.debug_mode = 0;
        }
        
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    fn render(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);

        let mut import_clicked = false;
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
            ui.checkbox(&mut self.show_scaffold, "显示点云 (调试用)");
            ui.checkbox(&mut self.show_voronoi_debug, "显示圆盘调试 (Voronoi Disk)");
            ui.checkbox(&mut self.show_depth_debug, "显示深度图 (Depth Map)");
            ui.checkbox(&mut self.show_normal_debug, "显示法线调试 (Normal Debug)");
            ui.checkbox(&mut self.show_sphere_debug, "显示圆球调试 (Vs Sphere)");
            ui.separator();
            if ui.button("📂 导入 GLB").clicked() { import_clicked = true; }
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

        // --- [Pass C: 深度图渲染 Pass] - 始终执行，为 cs_main 提供三角ID和深度数据 ---
        if self.params.anchor_count > 0 {
            let mut dpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth + ID MRT Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tri_id_texture_view_for_render,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view_for_render,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            dpass.set_pipeline(&self.depth_render_pipeline);
            dpass.set_bind_group(0, &self.depth_bind_group, &[]);
            dpass.draw(0..(self.params.anchor_count * 3), 0..1);
            drop(dpass);
        }

        // --- [Pass D: Compute Pass] - 始终执行，使用深度图数据 ---
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
            drop(cpass);
        }

        // --- [Pass E: 显示 Pass] ---
        if self.show_depth_debug && self.params.anchor_count > 0 {
            // 深度图调试模式：直接 blit 深度图
            let mut bpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            bpass.set_pipeline(&self.depth_blit_pipeline);
            bpass.set_bind_group(0, &self.depth_blit_bind_group, &[]);
            bpass.draw(0..4, 0..1);
            drop(bpass);
        } else {
            // 正常渲染模式：使用 compute 着色器结果
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
            drop(rpass);
        }
            
        // --- [Pass F: 画出点云脚手架] ---
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Draw Scaffold"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            rpass.set_pipeline(&self.scaffold_render_pipeline);
            rpass.set_bind_group(0, &self.compute_bind_group, &[]);
            
            if self.show_scaffold && self.params.scaffold_count > 0 {
                rpass.draw(0..self.params.scaffold_count, 0..1);
            }
            drop(rpass);
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

            // 2. 重新创建 Voronoi 纹理
            self.voronoi_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Voronoi Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.voronoi_texture_view = self.voronoi_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 3. 重新创建深度图纹理
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Debug Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_texture_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_texture_view_for_render = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.tri_id_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Triangle ID Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            self.tri_id_texture_view = self.tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.tri_id_texture_view_for_render = self.tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 4. 关键修复：重新创建 BindGroup，否则它们引用的还是旧视图
            self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.output_texture_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.primitives_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.triangle_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.scaffold_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.depth_texture_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.tri_id_texture_view) },
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

            self.depth_blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.depth_blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_texture_view) },
                ],
                label: None,
            });

            self.depth_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.depth_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: self.params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.triangle_buffer.as_entire_binding() },
                ],
                label: None,
            });
        }
    }

    // 在 Rust 里实现和 Shader 完全一致的 smin 基座采样
    fn get_base_sdf_rust(&self, p: Vec3, primitives: &[Primitive]) -> f32 {
        if primitives.is_empty() {
            return 1000.0;
        }
        
        let mut min_d = 1000.0;
        
        for (idx, prim) in primitives.iter().enumerate() {
            let inv = glam::Mat4::from_cols_array_2d(&prim.inv_model_matrix);
            let local_p = inv.transform_point3(p);
            
            let type_id = prim.params[3] as u32;
            let d = match type_id {
                0 => { // Sphere
                    local_p.length() - prim.params[0]
                },
                1 => { // Box
                    let q = local_p.abs() - glam::Vec3::splat(prim.params[0]);
                    q.max(glam::Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0) - prim.params[1]
                },
                _ => 1000.0,
            };
            
            if idx == 0 {
                min_d = d;
            } else {
                let k = prim.params[2];
                let h = (0.5 + 0.5 * (min_d - d) / k).clamp(0.0, 1.0);
                min_d = min_d * h + d * (1.0 - h) - k * h * (1.0 - h);
            }
        }
        
        min_d
    }

    // Anchor 聚类
    fn cluster_samples(
        &self,
        samples: &[MeshSample],
        cell_size: f32,
    ) -> Vec<MeshSample>
    {
        use std::collections::HashMap;

        let mut cells:
            HashMap<(i32,i32,i32), Vec<MeshSample>>
            = HashMap::new();

        for s in samples {

            let key = (
                (s.pos.x / cell_size).floor() as i32,
                (s.pos.y / cell_size).floor() as i32,
                (s.pos.z / cell_size).floor() as i32,
            );

            cells.entry(key)
                .or_default()
                .push(*s);
        }

        let mut result = Vec::new();

        for (_, group) in cells {

            let mut pos = Vec3::ZERO;
            let mut normal = Vec3::ZERO;

            for s in &group {
                pos += s.pos;
                normal += s.normal;
            }

            let inv =
                1.0 / group.len() as f32;

            result.push(MeshSample {
                pos: pos * inv,
                normal: normal.normalize(),
            });
        }

        result
    }

    // primitive surface projection
    fn project_to_primitive_surface(
        &self,
        p: Vec3,
    ) -> Option<(Vec3, Vec3)>
    {
        let mut best_dist = 999999.0;

        let mut best_proj = None;
        let mut best_normal = None;

        for prim in &self.primitives {

            let inv =
                glam::Mat4::from_cols_array_2d(
                    &prim.inv_model_matrix
                );

            let local_p =
                inv.transform_point3(p);

            let type_id =
                prim.params[3] as u32;

            match type_id {

                // sphere
                0 => {

                    let radius = prim.params[0];

                    let len = local_p.length();

                    if len < 0.0001 {
                        continue;
                    }

                    let local_normal =
                        local_p.normalize();

                    let local_proj =
                        local_normal * radius;

                    let world =
                        inv.inverse()
                        .transform_point3(local_proj);

                    let world_normal =
                        inv.inverse()
                        .transform_vector3(local_normal)
                        .normalize();

                    let d =
                        (world - p).length();

                    if d < best_dist {

                        best_dist = d;

                        best_proj = Some(world);

                        best_normal = Some(world_normal);
                    }
                }

                _ => {}
            }
        }

        if let (Some(p), Some(n)) =
            (best_proj, best_normal)
        {
            Some((p,n))
        }
        else {
            None
        }
    }

    // 打开文件对话框并导入 GLB
    fn import_scaffold(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GLB Files", &["glb"])
            .add_filter("GLTF Files", &["gltf"])
            .pick_file() {
            
            // 保存路径供后续烘焙使用
            self.scaffold_path = Some(path.to_str().unwrap().to_string());
            
            // 加载三角形和顶点（使用原始索引 + 邻接信息）
            let (document, buffers, _) = gltf::import(&path).unwrap();
            
            let mut positions: Vec<Vec3> = Vec::new();
            let mut vertex_triangles: Vec<Vec<u32>> = Vec::new();
            let mut triangles = Vec::new();
            let mut verts = Vec::new();
            
            for mesh in document.meshes() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|b| Some(&buffers[b.index()]));
                    
                    let pos_iter: Vec<[f32;3]> = reader.read_positions().unwrap().collect();
                    let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                    
                    // 先收集所有唯一顶点
                    let mut max_idx = 0u32;
                    for &idx in &indices {
                        max_idx = max_idx.max(idx);
                    }
                    
                    // 初始化邻接表
                    let vert_count = (max_idx + 1) as usize;
                    if positions.len() < vert_count {
                        positions.resize(vert_count, Vec3::ZERO);
                        vertex_triangles.resize(vert_count, Vec::new());
                    }
                    
                    for chunk in indices.chunks_exact(3) {
                        let i0 = chunk[0] as usize;
                        let i1 = chunk[1] as usize;
                        let i2 = chunk[2] as usize;
                        
                        let p0 = pos_iter[i0];
                        let p1 = pos_iter[i1];
                        let p2 = pos_iter[i2];
                        
                        // 存储唯一顶点（按索引去重）
                        positions[i0] = Vec3::from(p0);
                        positions[i1] = Vec3::from(p1);
                        positions[i2] = Vec3::from(p2);
                        
                        let tri_idx = triangles.len() as u32;
                        
                        triangles.push(math::Triangle {
                            v0: [p0[0], p0[1], p0[2], 1.0],
                            v1: [p1[0], p1[1], p1[2], 1.0],
                            v2: [p2[0], p2[1], p2[2], 1.0],
                        });
                        
                        // 记录邻接关系
                        vertex_triangles[i0].push(tri_idx);
                        vertex_triangles[i1].push(tri_idx);
                        vertex_triangles[i2].push(tri_idx);
                    }
                    
                    // 收集所有顶点用于点云显示（保留原始顺序）
                    for p in &pos_iter {
                        verts.push(Vec3::from(*p));
                    }
                }
            }
            
            self.vertex_positions = positions;
            self.vertex_triangles = vertex_triangles;
            self.scaffold_vertices = verts;
            self.triangles = triangles;
            
            println!("唯一顶点: {}, 三角形: {}, 原始顶点: {}", 
                self.vertex_positions.len(), 
                self.triangles.len(),
                self.scaffold_vertices.len());
            
            // ========== 自动生成包围球 ==========
            // 1. 计算所有顶点的中心点
            let mut center = glam::Vec3::ZERO;
            for tri in &self.triangles {
                center += Vec3::new(tri.v0[0], tri.v0[1], tri.v0[2]);
                center += Vec3::new(tri.v1[0], tri.v1[1], tri.v1[2]);
                center += Vec3::new(tri.v2[0], tri.v2[1], tri.v2[2]);
            }
            center /= (self.triangles.len() * 3) as f32;
            
            // 2. 计算最大半径
            let mut max_r = 0.0f32;
            for tri in &self.triangles {
                let d0 = (Vec3::new(tri.v0[0], tri.v0[1], tri.v0[2]) - center).length();
                let d1 = (Vec3::new(tri.v1[0], tri.v1[1], tri.v1[2]) - center).length();
                let d2 = (Vec3::new(tri.v2[0], tri.v2[1], tri.v2[2]) - center).length();
                max_r = max_r.max(d0).max(d1).max(d2);
            }
            
            // 3. 自动创建一个包围球 Primitive
            let base_sphere = math::Primitive::new_sphere(center, max_r * 1.1);
            
            if self.primitives.is_empty() {
                self.primitives.push(base_sphere);
            } else {
                self.primitives[0] = base_sphere;
            }
            
            // 更新 Primitive Buffer
            self.queue.write_buffer(&self.primitives_buffer, 0, bytemuck::cast_slice(&self.primitives));
            self.params.prim_count = self.primitives.len() as u32;
            
            println!("自动生成包围球：中心 {:?}, 半径 {:.2}", center, max_r * 1.1);
            
            // 更新 model_center
            self.params.model_center = [center.x, center.y, center.z, 1.0];
            
            // 上传到 GPU
            self.queue.write_buffer(&self.triangle_buffer, 0, bytemuck::cast_slice(&self.triangles));
            
            // 核心修复 1：将原始顶点转化为 vec4 (x, y, z, 1.0) 传给显存
            let scaffold_data: Vec<glam::Vec4> = self.scaffold_vertices.iter()
                .map(|v| v.extend(1.0))
                .collect();

            self.queue.write_buffer(&self.scaffold_buffer, 0, bytemuck::cast_slice(&scaffold_data));
            
            // 更新参数 - 使用 anchor_count 存储三角形数量
            self.params.scaffold_count = scaffold_data.len() as u32;
            self.params.anchor_count = self.triangles.len() as u32;
            
            println!("脚手架上传成功：{} 个点，{} 个三角形", self.params.scaffold_count, self.params.anchor_count);
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



