use std::{collections::HashMap, fs, sync::Arc};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use serde::{Serialize, Deserialize};
use glam::{Vec3, Mat4, Vec4Swizzles};

fn load_cpal_to_vec4(file_path: &str) -> Vec<[f32; 4]> {
    let data = fs::read(file_path).expect("读取色板失败");
    let mut colors = Vec::with_capacity(1024);
    
    // 1. 跳过 8 字节头 (CPAL魔数 4 字节 + Count 4 字节)
    let color_data = &data[8..];
    
    // 2. 每 3 个字节 (RGB) 转为一个 [f32; 4] (RGBA)
    for chunk in color_data.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        colors.push([r, g, b, 1.0]);
    }
    
    // 补齐到 1024 个
    while colors.len() < 1024 {
        colors.push([0.0, 0.0, 0.0, 1.0]);
    }
    colors
}



// --- 导入模块 ---
use cillin_engine::{camera, console, input, models, rendering, scene, undo};

// --- 配置路径（请确保路径正确） ---
const MODEL_DIR: &str = "../../Asset/cemModel"; 
const SCENE_FILE: &str = "scene_data.json";

// --- 数据结构定义 ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 20, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    direction: [f32; 3],
    _padding: u32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderUniforms {
    shadows_enabled: u32,
    ao_enabled: u32,
    debug_mode: u32, // 0: 正常渲染, 1: SDF 场可视化, 2: UVW 映射可视化
    _padding: [u32; 5], // 确保大小为 32 字节（4 * 8）
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    view_inv: [[f32; 4]; 4], // 64
    proj_inv: [[f32; 4]; 4], // 64
    cam_pos: [f32; 4],       // 16
    light_dir: [f32; 4],     // 16
    entity_count: u32,       // 4
    debug_mode: u32,         // 4
    _pad1: [f32; 2],         // 8 (vec2<f32>)
    _pad2: [[f32; 4]; 3],    // 48 (3个vec4<f32>)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelEntry {
    id: u32,
    name: String,
    file: String,
    default_scale: [f32; 3],
}

#[derive(Serialize, Deserialize, Debug)]
struct AssetManifest {
    models: Vec<ModelEntry>,
}

#[derive(PartialEq)]
enum EditMode { Idle, Grab, Rotate, Scale }

#[derive(Debug, Clone, Copy, PartialEq)]
enum Axis {
    None,
    X,
    Y,
    Z,
}

// --- 核心状态 ---

struct State<'a> {
    window: Arc<Window>,
    render_context: rendering::RenderContext<'a>,
    
    manifest: HashMap<u32, ModelEntry>,
    models: HashMap<u32, models::ModelAsset>,
    entities: Vec<scene::Entity>,
    instance_counters: std::collections::HashMap<u32, usize>,
    
    camera: camera::Camera,
    controller: camera::CameraController,
    input_state: input::InputState,
    camera_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    render_uniforms_buffer: wgpu::Buffer,
    
    edit_mode: EditMode,
    selected_idx: Option<usize>,
    active_model_id: u32,
    selected_axis: Axis,
    
    console: console::Console,
    undo_manager: undo::UndoManager,
    
    edit_start_state: Option<undo::EntityAction>,
    
    // 全局 SDF 相关
    global_sdf_texture: wgpu::Texture,
    global_sdf_view: wgpu::TextureView,
    global_sdf_sampler: wgpu::Sampler,
    global_sdf_bind_group: wgpu::BindGroup,
    
    // 渲染控制标志
    shadows_enabled: bool,
    ao_enabled: bool,
    debug_mode: u32,
    
    // 曦罗混合相关
    entity_list_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    palette_buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,
    blit_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    output_texture: wgpu::Texture,
}



impl<'a> State<'a> {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

        let caps = surface.get_capabilities(&adapter);
        // 选择支持的 sRGB 格式
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

        // 1. 加载清单与模型
        let manifest_raw = fs::read_to_string(format!("{}/manifest.json", MODEL_DIR)).expect("Manifest.json 没找到");
        let manifest_data: AssetManifest = serde_json::from_str(&manifest_raw).unwrap();
        let mut manifest_map = HashMap::new();
        let mut loaded_models = HashMap::new();

        // 创建 BindGroupLayout（只包含纹理和采样器）
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Uint }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
            ],
            label: Some("Texture Layout"),
        });

        for entry in manifest_data.models {
            manifest_map.insert(entry.id, entry.clone());
            
            let file_path = format!("{}/{}", MODEL_DIR, entry.file);
            
            // 尝试加载 CEM 文件，如果文件不存在则跳过
            if let Ok((voxel_data, aabb_min, aabb_max)) = std::fs::File::open(&file_path).map(|_| {
                models::load_cem_data(&file_path)
            }) {
                // 创建一个空的 ModelAsset（后续会更新为真正的 .cem 加载逻辑）
                let empty_model = models::ModelAsset {
                    vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Empty Vertex Buffer"),
                        size: 0,
                        usage: wgpu::BufferUsages::VERTEX,
                        mapped_at_creation: false,
                    }),
                    index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Empty Index Buffer"),
                        size: 0,
                        usage: wgpu::BufferUsages::INDEX,
                        mapped_at_creation: false,
                    }),
                    num_indices: 0,
                    bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("Empty Texture"),
                                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                                view_formats: &[],
                            }).create_view(&wgpu::TextureViewDescriptor::default())) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor::default())) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("Empty SDF Texture"),
                                size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D3,
                                format: wgpu::TextureFormat::R32Uint,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                                view_formats: &[],
                            }).create_view(&wgpu::TextureViewDescriptor::default())) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor::default())) },
                        ],
                        label: Some("Empty Bind Group"),
                    }),
                    instance_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Empty Instance Buffer"),
                        size: 0,
                        usage: wgpu::BufferUsages::VERTEX,
                        mapped_at_creation: false,
                    }),
                    instance_capacity: 0,
                    sdf_view: device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Empty SDF View"),
                        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R32Uint,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }).create_view(&wgpu::TextureViewDescriptor::default()),
                    dna_data: voxel_data,
                    albedo_data: None,
                    aabb_min,
                    aabb_max,
                };
                
                loaded_models.insert(entry.id, empty_model);
                println!("Loaded CEM model: {} (id: {})", entry.name, entry.id);
            } else {
                println!("Skipping missing CEM model: {} (id: {})", entry.name, entry.id);
            }
        }

        // 2. 加载场景
        let mut instance_counters: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let entities: Vec<scene::Entity> = fs::read_to_string(SCENE_FILE)
            .map(|raw| {
                let scene_data: scene::SceneData = serde_json::from_str(&raw).unwrap();
                scene_data.entities.into_iter().map(|mut entity| {
                    if entity.code.is_empty() {
                        let instance_index = *instance_counters.get(&entity.model_id).unwrap_or(&0);
                        entity.code = format!("{}s{:06}", entity.model_id, instance_index);
                        *instance_counters.entry(entity.model_id).or_insert(0) += 1;
                    }
                    entity
                }).collect()
            })
            .unwrap_or_default();

        // 3. 相机与管线
        let camera_pos = Vec3::new(3.0, 3.0, 3.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0f32.to_radians(), size.width as f32 / size.height as f32, 0.1, 100.0);
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

        // 灯光设置
        let light_uniform = LightUniform {
            direction: [-0.5, -0.5, -1.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0, 1.0],
        };
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // 创建渲染控制 uniforms buffer
        let render_uniforms = RenderUniforms {
            shadows_enabled: 1,
            ao_enabled: 1,
            debug_mode: 0,
            _padding: [0; 5],
        };
        let render_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Uniforms"),
            contents: bytemuck::cast_slice(&[render_uniforms]),
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

        // 全局 SDF bind group layout
        let global_sdf_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
            ],
            label: None,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // --- 1. 大脑布局 (Compute 专用) ---
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Brain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // --- 2. 眼睛布局 (Render 专用) ---
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

        // --- 创建计算管线布局 ---
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        // --- 创建渲染管线布局 ---
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        // 1. 创建计算管线 (大脑)
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Xi-Luoer Brain Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        // 2. 创建 Blit 管线 (眼睛)
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
                    format: config.format,
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
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 1. 计算需要盖多少层楼
        let model_count = loaded_models.len() as u32;
        let sdf_res = 64;
        let atlas_depth = sdf_res * model_count;

        // 2. 创建一个超深的 3D 纹理堆栈
        let global_sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CEM Holographic Data"),
            size: wgpu::Extent3d {
                width: sdf_res,
                height: sdf_res,
                depth_or_array_layers: atlas_depth, // 关键：深度是 64 * 模型数
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 3. 将每个模型的 SDF 数据搬进对应的"楼层"
        for (id, model) in &loaded_models {
            // 假设 id 是从 1 开始的，我们映射到 index 0, 1, 2...
            let index = *id - 1;
            
            // 安全检查：防止 Source buffer of size 0
            if model.dna_data.is_empty() {
                println!("跳过模型 {} 的纹理上传，数据为空", id);
                continue;
            }

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &global_sdf_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: index * sdf_res }, // 关键：纵向偏移
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&model.dna_data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * sdf_res),
                    rows_per_image: Some(sdf_res),
                },
                wgpu::Extent3d { width: sdf_res, height: sdf_res, depth_or_array_layers: sdf_res },
            );
        }

        let global_sdf_view = global_sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 1. 创建纹理阵列（假设贴图都是 512x512）
        let tex_size = 512;
        let albedo_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Texture Array"),
            size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: model_count },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2, // 2D 数组
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 2. 将每个模型的贴图写进去
        for (id, model) in &loaded_models {
            // 假设 id 是从 1 开始的，我们映射到 index 0, 1, 2...
            let index = *id - 1;
            if let Some(texture_data) = &model.albedo_data {
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &albedo_array,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: index },
                        aspect: wgpu::TextureAspect::All,
                    },
                    texture_data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex_size),
                        rows_per_image: Some(tex_size),
                    },
                    wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
                );
            }
        }

        let albedo_array_view = albedo_array.create_view(&wgpu::TextureViewDescriptor::default());

        // 创建全局 SDF 采样器
        let global_sdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Global SDF Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 创建贴图采样器
        let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Albedo Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 创建全局 SDF bind group
        let global_sdf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_sdf_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: render_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&global_sdf_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&global_sdf_sampler) },
            ],
            label: None,
        });

        // 创建实体清单缓冲区 - 预分配1024个物体的空间
        let entity_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entity List Buffer"),
            // 预留 1024 个物体的空间 (每个 256 字节)
            size: (1024 * 256) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建参数缓冲区
        let params = Params {
            view_inv: view.inverse().to_cols_array_2d(),
            proj_inv: proj.inverse().to_cols_array_2d(),
            cam_pos: camera_pos.extend(1.0).to_array(),
            light_dir: Vec3::new(0.5, 1.0, 0.5).normalize().extend(0.0).to_array(), // 从上往下斜着照射
            entity_count: entities.len() as u32,
            debug_mode: 0, // 默认关闭调试模式
            _pad1: [0.0; 2],
            _pad2: [[0.0; 4]; 3],
        };
        
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 加载全局色板
        let palette_data = load_cpal_to_vec4("../../Asset/Global/master.cpal");
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Palette Buffer"),
            contents: bytemuck::cast_slice(&palette_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 创建输出纹理视图
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // 关键：必须包含这三个用途
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 创建主绑定组 - 使用第一个模型的SDF
        let sdf_view = if let Some(model) = loaded_models.get(&1) {
            &model.sdf_view
        } else {
            &global_sdf_view
        };
        
        // --- 创建对应的绑定组 ---
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: entity_list_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&global_sdf_view) },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: palette_buffer.as_entire_binding() },
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

        // 创建渲染上下文
        let render_context = rendering::RenderContext::new(
            surface,
            device,
            queue,
            render_pipeline,
            None,
            camera_bind_group,
            light_bind_group,
            None,
            depth_texture,
            depth_view,
            size,
            format,
        );

        // 摄像机从 (1.5, 1.5, 1.5) 看向原点 (0, 0, 0)，距离更近以增强透视效果
        let camera = camera::Camera::new(
            Vec3::new(1.5, 1.5, 1.5),
            -135.0f32.to_radians(), // 朝向原点的水平角度
            -35.0f32.to_radians(), // 向下看的垂直角度
        );
        
        let controller = camera::CameraController::new(0.25); // 降低移动速度
        
        let input_state = input::InputState::new();

        let mut state = Self {
            window,
            render_context,
            manifest: manifest_map,
            models: loaded_models,
            entities,
            instance_counters,
            camera,
            controller,
            input_state,
            camera_buffer,
            light_buffer,
            render_uniforms_buffer,
            edit_mode: EditMode::Idle,
            selected_idx: None,
            active_model_id: 1,
            selected_axis: Axis::None,
            console: console::Console::new(),
            undo_manager: undo::UndoManager::new(50),
            edit_start_state: None,
            
            // --- 补充下面这四个缺失的字段 ---
            global_sdf_texture,
            global_sdf_view,
            global_sdf_sampler,
            global_sdf_bind_group,
            
            // 初始化渲染控制标志
            shadows_enabled: true,
            ao_enabled: true,
            debug_mode: 0,
            
            // 曦罗混合相关
            entity_list_buffer,
            params_buffer,
            palette_buffer,
            compute_bind_group,
            blit_bind_group,
            compute_pipeline,
            output_texture,
        };

        // 关键：启动时先计算一次场景 SDF
        // state.update_global_sdf(); // 暂时停用，因为 C++ 还没支持 DNA 合并

        state
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.render_context.resize(new_size);
        }
    }

    fn save_scene(&self) {
        let scene_manager = scene::SceneManager {
            entities: self.entities.clone(),
            next_entity_id: self.entities.len(),
            models: HashMap::new(),
            instance_counters: HashMap::new(),
        };
        
        if scene_manager.save_scene(SCENE_FILE).is_ok() {
            println!("场景已存至 {}", SCENE_FILE);
        } else {
            println!("保存失败");
        }
    }



    fn get_click_ground_pos(&self) -> Vec3 {
        let (x, y) = self.input_state.mouse_pos;
        let ndc = glam::vec4(
            (x as f32 / self.render_context.size.width as f32) * 2.0 - 1.0,
            1.0 - (y as f32 / self.render_context.size.height as f32) * 2.0,
            0.0, 1.0
        );
        let inv_vp = (self.render_context.current_proj * self.render_context.current_view).inverse();
        let world_pos = inv_vp * ndc;
        let world_pos = world_pos.xyz() / world_pos.w;
        let ray_dir = (world_pos - self.camera.eye).normalize();
        
        let t = -self.camera.eye.y / ray_dir.y;
        self.camera.eye + ray_dir * t
    }

    fn get_clicked_entity(&self) -> Option<&scene::Entity> {
        let (x, y) = self.input_state.mouse_pos;
        let ndc = glam::vec4(
            (x as f32 / self.render_context.size.width as f32) * 2.0 - 1.0,
            1.0 - (y as f32 / self.render_context.size.height as f32) * 2.0,
            0.0, 1.0
        );
        let inv_vp = (self.render_context.current_proj * self.render_context.current_view).inverse();
        let world_pos = inv_vp * ndc;
        let world_pos = world_pos.xyz() / world_pos.w;
        let ray_dir = (world_pos - self.camera.eye).normalize();

        let mut closest_entity: Option<&scene::Entity> = None;
        let mut closest_t = f32::MAX;

        for entity in &self.entities {
            let model_matrix = Mat4::from_cols_array_2d(&entity.get_model_matrix());
            let entity_center = model_matrix.transform_point3(Vec3::ZERO);
            let entity_scale = entity.scale.max_element();
            
            let offset = entity_center - self.camera.eye;
            let b = offset.dot(ray_dir);
            let c = offset.dot(offset) - entity_scale * entity_scale * 0.5;
            
            let discriminant = b * b - c;
            if discriminant< 0.0 {
                continue;
            }
            
            let t = b - discriminant.sqrt();
            if t >0.0 && t< closest_t {
                closest_t = t;
                closest_entity = Some(entity);
            }
        }

        closest_entity
    }

    fn undo(&mut self) {
        self.undo_manager.undo(&mut self.entities);
    }

    fn redo(&mut self) {
        self.undo_manager.redo(&mut self.entities);
    }

    fn execute_command(&mut self) {
        let command = self.console.execute();
        if command.is_empty() {
            self.console.is_open = false;
            self.edit_mode = EditMode::Idle;
            return;
        }

        println!("执行命令: {}", command);
        
        let parts: Vec<&str> = command.split_whitespace().collect();
        if let Some(cmd) = parts.first() {
            match *cmd {
                "help" => println!("可用命令: help, exit, spawn <model_id>, cmr <x> <y> <z>, cmr now, sdf, shadow, ao, debug <0|1|2>"),
                "exit" => println!("控制台已关闭"),
                "spawn" => {
                    if parts.len() >= 2 {
                        if let Ok(model_id) = parts[1].parse::<u32>() {
                            let pos = self.get_click_ground_pos();
                            let instance_index = *self.instance_counters.get(&model_id).unwrap_or(&0);
                            let entity = scene::Entity::new(
                                self.entities.len(),
                                model_id,
                                instance_index,
                                pos,
                                Vec3::ZERO,
                                Vec3::ONE,
                                [1.0, 1.0, 1.0, 1.0],
                            );
                            self.entities.push(entity);
                            *self.instance_counters.entry(model_id).or_insert(0) += 1;
                            println!("已在位置 {:?} 生成模型 {}", pos, model_id);
                        } else {
                            println!("错误: 无效的模型 ID");
                        }
                    } else {
                        println!("错误: 请提供模型 ID");
                    }
                }
                "cmr" => {
                    if parts.len() >= 2 && parts[1] == "now" {
                        println!("摄像机当前位置: ({}, {}, {})", 
                            self.camera.eye.x, 
                            self.camera.eye.y, 
                            self.camera.eye.z);
                    } else if parts.len() >= 4 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[1].parse::<f32>(),
                            parts[2].parse::<f32>(),
                            parts[3].parse::<f32>(),
                        ) {
                            self.camera.eye = Vec3::new(x, y, z);
                            println!("摄像机已移动到位置 ({}, {}, {})", x, y, z);
                        } else {
                            println!("错误: 无效的坐标");
                        }
                    } else {
                        println!("错误: 使用方法: cmr <x> <y> <z> 或 cmr now");
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
                            println!("调试模式已切换为: {}", mode);
                        }
                    } else {
                        println!("用法: debug <0|1|2> (0:关, 1:距离场, 2:坐标映射)");
                    }
                }
                _ => println!("未知命令: {}", cmd),
            }
        }
        self.console.is_open = false;
        self.edit_mode = EditMode::Idle;
    }

    fn update(&mut self) {
        // 从输入状态更新控制器状态
        self.controller.is_up_pressed = self.input_state.is_w_pressed;
        self.controller.is_down_pressed = self.input_state.is_s_pressed;
        self.controller.is_left_pressed = self.input_state.is_a_pressed;
        self.controller.is_right_pressed = self.input_state.is_d_pressed;
        self.controller.is_q_pressed = self.input_state.is_q_pressed;
        self.controller.is_e_pressed = self.input_state.is_e_pressed;
        
        // 更新相机位置
        self.controller.update(&mut self.camera, 1.0);
        
        // 更新视图和投影矩阵
        self.render_context.current_view = self.camera.get_view_matrix();
        self.render_context.current_proj = Mat4::perspective_rh(
            45.0f32.to_radians(), 
            self.render_context.size.width as f32 / self.render_context.size.height as f32, 
            0.1, 
            100.0
        );
        
        // 更新相机缓冲区
        let camera_matrix = self.render_context.current_proj * self.render_context.current_view;
        self.render_context.queue.write_buffer(
            &self.camera_buffer, 
            0, 
            bytemuck::cast_slice(camera_matrix.as_ref())
        );
    }

    fn update_instance_buffer(&mut self, model_id: u32) {
        // 1. 收集当前模型的所有实例数据
        let instances: Vec<scene::InstanceRaw> = self.entities
            .iter()
            .filter(|e| e.model_id == model_id)
            .map(|e| e.to_raw())
            .collect();
            
        if instances.is_empty() {
            return;
        }

        let model_asset = self.models.get_mut(&model_id).unwrap();
        let instance_count = instances.len();

        // 2. 检查容量：如果现有的 Buffer 装不下了，就扩容
        if instance_count > model_asset.instance_capacity {
            // 策略：扩容到当前需求的 2 倍，防止频繁重分配
            model_asset.instance_capacity = instance_count * 2;
            
            model_asset.instance_buffer = self.render_context.device.create_buffer(&wgpu::BufferDescriptor { 
                label: Some(&format!("Model {} Instance Buffer", model_id)), 
                size: (model_asset.instance_capacity * std::mem::size_of::<scene::InstanceRaw>()) as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, 
                mapped_at_creation: false, 
            }); 
            println!("Model {} 显存扩容至: {}", model_id, model_asset.instance_capacity); 
        }

        // 3. 高效同步数据：只把改变的数据写入显存
        self.render_context.queue.write_buffer( 
            &model_asset.instance_buffer, 
            0, 
            bytemuck::cast_slice(&instances), 
        ); 
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 更新实体清单缓冲区
        let entity_data: Vec<scene::EntityData> = self.entities.iter()
            .map(|e| {
                let model = self.models.get(&e.model_id).unwrap();
                // 关键修复：这里的 0 必须改为对应的层索引 (ID - 1)
                let layer_index = e.model_id - 1;
                e.to_entity_data(layer_index, model.aabb_min, model.aabb_max)
            })
            .collect();
        
        self.render_context.queue.write_buffer(
            &self.entity_list_buffer,
            0,
            bytemuck::cast_slice(&entity_data),
        );
        
        // 更新参数缓冲区
        let params = Params {
            view_inv: self.render_context.current_view.inverse().to_cols_array_2d(),
            proj_inv: self.render_context.current_proj.inverse().to_cols_array_2d(),
            cam_pos: self.camera.eye.extend(1.0).to_array(),
            light_dir: Vec3::new(0.5, 1.0, 0.5).normalize().extend(0.0).to_array(), // 从上往下斜着照射
            entity_count: self.entities.len() as u32,
            debug_mode: self.debug_mode,
            _pad1: [0.0; 2],
            _pad2: [[0.0; 4]; 3],
        };
        
        self.render_context.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
        
        let mut encoder = self.render_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // 第一步：计算大脑
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]); // 使用计算专用组
            cpass.dispatch_workgroups((self.render_context.size.width + 7) / 8, (self.render_context.size.height + 7) / 8, 1);
        }

        // 第二步：眼睛展示
        let output = self.render_context.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rpass.set_pipeline(&self.render_context.render_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]); // 使用展示专用组
            rpass.draw(0..3, 0..1); // 画一个全屏三角形
        }

        self.render_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

// --- 程序入口 ---

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("Developer Mode Engine").build(&event_loop).unwrap());

    let mut state: State<'_> = pollster::block_on(State::new(window.clone()));

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { ref event, window_id } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => state.resize(*s),
            WindowEvent::MouseInput { button, state: button_state, .. } => {
                if button == &MouseButton::Middle {
                    state.input_state.is_middle_mouse_pressed = *button_state == ElementState::Pressed;
                    if state.input_state.is_middle_mouse_pressed {
                        state.input_state.last_mouse_pos = Some(state.input_state.mouse_pos);
                    } else {
                        state.input_state.last_mouse_pos = None;
                    }
                } else if button == &MouseButton::Left {
                    if *button_state == ElementState::Pressed {
                        if let Some(entity) = state.get_clicked_entity() {
                            println!("{}", entity.model_id);
                            println!("{}", entity.code);
                            state.selected_idx = Some(state.entities.iter().position(|e| e.id == entity.id).unwrap());
                            state.edit_mode = EditMode::Idle;
                        } else if state.selected_idx.is_some() {
                            println!("cancel");
                            if let Some(mut action) = state.edit_start_state.take() {
                                if let Some(idx) = state.selected_idx {
                                    action.new_position = state.entities[idx].position;
                                    action.new_rotation = state.entities[idx].rotation;
                                    action.new_scale = state.entities[idx].scale;
                                    state.undo_manager.push(action);
                                }
                            }
                            state.selected_idx = None;
                            state.edit_mode = EditMode::Idle;
                            state.selected_axis = Axis::None;
                        }
                    } else if state.edit_mode != EditMode::Idle {
                        if let Some(mut action) = state.edit_start_state.take() {
                            if let Some(idx) = state.selected_idx {
                                action.new_position = state.entities[idx].position;
                                action.new_rotation = state.entities[idx].rotation;
                                action.new_scale = state.entities[idx].scale;
                                state.undo_manager.push(action);
                            }
                        }
                        state.edit_mode = EditMode::Idle;
                        state.selected_axis = Axis::None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                if let Some(last_pos) = state.input_state.last_mouse_pos {
                    let dx = -(new_pos.0 - last_pos.0) as f32;
                    let dy = (new_pos.1 - last_pos.1) as f32;
                    
                    state.camera.yaw -= dx * 0.001;
                    state.camera.pitch -= dy * 0.001;
                    
                    state.camera.pitch = state.camera.pitch.max(-std::f32::consts::FRAC_PI_2).min(std::f32::consts::FRAC_PI_2);
                } else if state.selected_idx.is_some() && state.edit_mode != EditMode::Idle {
                    match state.edit_mode {
                        EditMode::Grab => {
                            if let Some(idx) = state.selected_idx {
                                let dx = (new_pos.0 - state.input_state.mouse_pos.0) as f32 * 0.01;
                                let dy = (state.input_state.mouse_pos.1 - new_pos.1) as f32 * 0.01;
                                
                                match state.selected_axis {
                                    Axis::X => {
                                        state.entities[idx].position.x += dx;
                                    }
                                    Axis::Y => {
                                        state.entities[idx].position.y += dy;
                                    }
                                    Axis::Z => {
                                        state.entities[idx].position.z += dy;
                                    }
                                    Axis::None => {}
                                }
                                

                            }
                        }
                        EditMode::Scale => {
                            if let Some(idx) = state.selected_idx {
                                let dy = (new_pos.1 - state.input_state.mouse_pos.1) as f32 * 0.005;
                                let scale_factor = 1.0 + dy;
                                
                                state.entities[idx].scale *= scale_factor;
                                

                            }
                        }
                        EditMode::Rotate => {
                            if let Some(idx) = state.selected_idx {
                                let dx = (new_pos.0 - state.input_state.mouse_pos.0) as f32 * 0.01;
                                let dy = (new_pos.1 - state.input_state.mouse_pos.1) as f32 * 0.01;
                                
                                state.entities[idx].rotation.y += dx;
                                state.entities[idx].rotation.x += dy;
                                

                            }
                        }
                        EditMode::Idle => {}
                    }
                };
                state.input_state.last_mouse_pos = if state.input_state.is_middle_mouse_pressed {
                    Some(new_pos)
                } else {
                    None
                };
                state.input_state.mouse_pos = new_pos;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if state.console.is_open {
                    if event.state == ElementState::Pressed {
                        if matches!(event.logical_key, winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace)) {
                            state.console.remove_char();
                        } else if let Some(text) = &event.text {
                            if text == "\r" {
                                state.execute_command();
                            } else if text != "\x08" { // 跳过退格符
                                state.console.add_char(text.chars().next().unwrap());
                            }
                        } else if matches!(event.logical_key, winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape)) {
                            state.console.is_open = false;
                            state.edit_mode = EditMode::Idle;
                        }
                    }
                    return;
                }
                
                // 使用输入模块处理按键
                state.input_state.process_key_event(event);
                
                // 处理特殊按键
                if let winit::keyboard::Key::Character(c) = &event.logical_key {
                    if event.state == ElementState::Pressed {
                        match c.as_str() {
                            "p" => state.save_scene(),
                            "/" => state.console.is_open = true,
                            "g" | "G" => {
                                if let Some(idx) = state.selected_idx {
                                    state.edit_mode = EditMode::Grab;
                                    state.selected_axis = Axis::None;
                                    state.edit_start_state = Some(undo::EntityAction {
                                        entity_id: idx,
                                        old_position: state.entities[idx].position,
                                        old_rotation: state.entities[idx].rotation,
                                        old_scale: state.entities[idx].scale,
                                        new_position: state.entities[idx].position,
                                        new_rotation: state.entities[idx].rotation,
                                        new_scale: state.entities[idx].scale,
                                    });
                                }
                            }
                            "x" | "X" => {
                                if state.edit_mode == EditMode::Grab {
                                    state.selected_axis = Axis::X;
                                }
                            }
                            "y" | "Y" => {
                                if state.edit_mode == EditMode::Grab {
                                    state.selected_axis = Axis::Y;
                                } else {
                                    state.redo();
                                }
                            }
                            "z" | "Z" => {
                                if state.edit_mode == EditMode::Grab {
                                    state.selected_axis = Axis::Z;
                                }
                            }
                            "s" | "S" => {
                                if let Some(idx) = state.selected_idx {
                                    state.edit_mode = EditMode::Scale;
                                    state.edit_start_state = Some(undo::EntityAction {
                                        entity_id: idx,
                                        old_position: state.entities[idx].position,
                                        old_rotation: state.entities[idx].rotation,
                                        old_scale: state.entities[idx].scale,
                                        new_position: state.entities[idx].position,
                                        new_rotation: state.entities[idx].rotation,
                                        new_scale: state.entities[idx].scale,
                                    });
                                }
                            }
                            "r" | "R" => {
                                if let Some(idx) = state.selected_idx {
                                    state.edit_mode = EditMode::Rotate;
                                    state.edit_start_state = Some(undo::EntityAction {
                                        entity_id: idx,
                                        old_position: state.entities[idx].position,
                                        old_rotation: state.entities[idx].rotation,
                                        old_scale: state.entities[idx].scale,
                                        new_position: state.entities[idx].position,
                                        new_rotation: state.entities[idx].rotation,
                                        new_scale: state.entities[idx].scale,
                                    });
                                }
                            }
                            "u" | "U" => {
                                state.undo();
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta: _, .. } => {
                state.input_state.process_event(event);
                // 使用滚轮调整相机位置
                state.camera.eye += state.camera.get_forward() * state.input_state.scroll_delta * 0.1;
                state.input_state.reset_scroll();
            }
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.render_context.size),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        },
        Event::AboutToWait => state.window.request_redraw(),
        _ => {}
    }).unwrap();
}
