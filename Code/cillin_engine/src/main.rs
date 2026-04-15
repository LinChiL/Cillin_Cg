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
    compute_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    blit_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    output_texture: wgpu::Texture,
    
    // 性能监控相关
    query_set: wgpu::QuerySet,
    query_buffer: wgpu::Buffer,
    mapped_buffer: wgpu::Buffer,
    is_perf_mode: bool,
    last_compute_time: f32,
    last_render_time: f32,
    frame_count: u32,
    last_perf_print: std::time::Instant,
    
    // 连续生成模式
    is_ser_spawn_mode: bool,
    ser_spawn_model_id: u32,
    
    // Tile 加速相关
    tile_buffer: wgpu::Buffer,
    tile_map_cache: Vec<scene::TileData>,
}



impl<'a> State<'a> {
    // 辅助函数：世界点转像素点
    fn project_to_pixel(&self, world_p: glam::Vec3, vp: glam::Mat4) -> Option<glam::Vec2> {
        let clip = vp * world_p.extend(1.0);
        if clip.w <= 0.0 { return None; }
        let ndc = clip.xyz() / clip.w;
        Some(glam::vec2(
            (ndc.x + 1.0) * 0.5 * self.render_context.size.width as f32,
            (1.0 - ndc.y) * 0.5 * self.render_context.size.height as f32,
        ))
    }

    // 将 AABB 的 8 个顶点投影到屏幕坐标
    fn project_aabb_to_screen(&self, entity: &scene::Entity, model: &models::ModelAsset) -> Option<(glam::Vec2, glam::Vec2)> {
        let mvp = self.render_context.current_proj * self.render_context.current_view * glam::Mat4::from_cols_array_2d(&entity.get_model_matrix());
        
        let min = model.aabb_min;
        let max = model.aabb_max;
        
        // AABB 的 8 个顶点坐标
        let corners = [
            glam::vec4(min[0], min[1], min[2], 1.0),
            glam::vec4(min[0], min[1], max[2], 1.0),
            glam::vec4(min[0], max[1], min[2], 1.0),
            glam::vec4(min[0], max[1], max[2], 1.0),
            glam::vec4(max[0], min[1], min[2], 1.0),
            glam::vec4(max[0], min[1], max[2], 1.0),
            glam::vec4(max[0], max[1], min[2], 1.0),
            glam::vec4(max[0], max[1], max[2], 1.0),
        ];

        let mut screen_min = glam::vec2(f32::MAX, f32::MAX);
        let mut screen_max = glam::vec2(f32::MIN, f32::MIN);
        let mut any_in_front = false;

        for corner in corners {
            let mut clip_pos = mvp * corner;
            
            // 简单的裁剪判定：如果顶点在相机后面，这一步比较复杂，我们简单处理
            if clip_pos.w > 0.0 {
                any_in_front = true;
            } else {
                continue;
            }

            // 归一化设备坐标 (NDC)
            let ndc = clip_pos.xyz() / clip_pos.w;
            
            // 映射到屏幕像素坐标
            let pixel_x = (ndc.x + 1.0) * 0.5 * self.render_context.size.width as f32;
            let pixel_y = (1.0 - ndc.y) * 0.5 * self.render_context.size.height as f32;

            screen_min = screen_min.min(glam::vec2(pixel_x, pixel_y));
            screen_max = screen_max.max(glam::vec2(pixel_x, pixel_y));
        }

        if !any_in_front { return None; }
        
        Some((screen_min, screen_max))
    }

    // 每帧更新 Tile 缓冲区
    fn update_tile_buffer(&mut self) {
        let width = self.render_context.size.width;
        let height = self.render_context.size.height;
        let tiles_x = (width + 15) / 16;
        let tiles_y = (height + 15) / 16;
        let total_tiles = (tiles_x * tiles_y) as usize;

        if self.tile_map_cache.len() != total_tiles {
            self.tile_map_cache.resize(total_tiles, scene::TileData { count: 0, _padding: [0; 3], entity_indices: [0; 128] });
        }
        self.tile_map_cache.fill(scene::TileData { count: 0, _padding: [0; 3], entity_indices: [0; 128] });

        // 1. 【生命感逻辑】：按距离排序，确保“挡路”的物体永远排在名单前列
        let mut sorted_indices: Vec<usize> = (0..self.entities.len()).collect();
        let cam_pos = self.camera.eye;
        sorted_indices.sort_by(|&a, &b| {
            let da = (self.entities[a].position - cam_pos).length();
            let db = (self.entities[b].position - cam_pos).length();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. 获取光源方向
        let light_dir = glam::vec3(0.5, 1.0, 0.5).normalize();
        let shadow_ext = -light_dir * 50.0; // 影子最长 50 米
        let vp = self.render_context.current_proj * self.render_context.current_view;

        for idx in sorted_indices {
            let entity = &self.entities[idx];
            let model = &self.models[&entity.model_id];
            let model_mat = glam::Mat4::from_cols_array_2d(&entity.get_model_matrix());
            
            let mut screen_min = glam::vec2(f32::MAX, f32::MAX);
            let mut screen_max = glam::vec2(f32::MIN, f32::MIN);
            let mut is_visible = false;

            // AABB 8 个顶点
            let min = model.aabb_min;
            let max = model.aabb_max;
            let corners = [
                glam::vec3(min[0], min[1], min[2]),
                glam::vec3(min[0], min[1], max[2]),
                glam::vec3(min[0], max[1], min[2]),
                glam::vec3(min[0], max[1], max[2]),
                glam::vec3(max[0], min[1], min[2]),
                glam::vec3(max[0], min[1], max[2]),
                glam::vec3(max[0], max[1], min[2]),
                glam::vec3(max[0], max[1], max[2]),
            ];

            for corner in corners {
                let world_pos = model_mat.transform_point3(corner);
                
                // 投影本体和影子
                let points = [world_pos, world_pos + shadow_ext];
                for p in points {
                    if let Some(pixel) = self.project_to_pixel(p, vp) {
                        // 关键防御：过滤掉无效的投影结果
                        if pixel.x.is_finite() && pixel.y.is_finite() {
                            screen_min = screen_min.min(pixel);
                            screen_max = screen_max.max(pixel);
                            is_visible = true;
                        }
                    }
                }
            }

            // 如果该物体及其影子完全在相机背后，跳过
            if !is_visible { continue; }

            // 3. 关键防御：计算 Tile 索引时必须进行极度严格的范围锁定
            // 使用 i32 进行初步计算，防止减 1 时的下溢出
            let ts_x = ((screen_min.x / 16.0).floor() as i32 - 1).clamp(0, tiles_x as i32 - 1);
            let te_x = ((screen_max.x / 16.0).ceil() as i32 + 1).clamp(0, tiles_x as i32 - 1);
            let ts_y = ((screen_min.y / 16.0).floor() as i32 - 1).clamp(0, tiles_y as i32 - 1);
            let te_y = ((screen_max.y / 16.0).ceil() as i32 + 1).clamp(0, tiles_y as i32 - 1);

            for ty in ts_y..=te_y {
                let row_offset = (ty as usize).wrapping_mul(tiles_x as usize);
                for tx in ts_x..=te_x {
                    let t_idx = row_offset.wrapping_add(tx as usize);
                    
                    // 4. 最终防线：检查数组边界
                    if t_idx < total_tiles {
                        let count = self.tile_map_cache[t_idx].count as usize;
                        if count < 128 {
                            self.tile_map_cache[t_idx].entity_indices[count] = idx as u32;
                            self.tile_map_cache[t_idx].count += 1;
                        }
                    }
                }
            }
        }

        self.render_context.queue.write_buffer(&self.tile_buffer, 0, bytemuck::cast_slice(&self.tile_map_cache));
    }

    async fn new(window: Arc<Window>) -> Self {
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
                // 开启时间戳查询特性
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();

        // 创建一个查询集，预留 4 个位置（Compute开始/结束，Render开始/结束）
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 4,
            ty: wgpu::QueryType::Timestamp,
            label: Some("Perf Query Set"),
        });

        // 还需要一个 Buffer 用来把 GPU 里的计时结果读回 CPU
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 32, // 4 * 8 字节
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 创建一个用于映射读取的缓冲区
        let mapped_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mapped Query Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // 创建 Tile Buffer (支持到 4K 分辨率的预留空间)
        let max_tiles_x = (3840 + 15) / 16;
        let max_tiles_y = (2160 + 15) / 16;
        let tile_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Storage Buffer"),
            size: (max_tiles_x * max_tiles_y * std::mem::size_of::<scene::TileData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });



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
                                format: wgpu::TextureFormat::Rg32Uint,
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
                        format: wgpu::TextureFormat::Rg32Uint,
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
        let proj = Mat4::perspective_rh(45.0f32.to_radians(), size.width as f32 / size.height as f32, 0.1, 10000.0);
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
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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
            format: wgpu::TextureFormat::Rg32Uint,
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
                    bytes_per_row: Some(8 * sdf_res),
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
                wgpu::BindGroupEntry { binding: 6, resource: tile_buffer.as_entire_binding() },
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
            compute_bind_group_layout,
            blit_bind_group_layout,
            compute_bind_group,
            blit_bind_group,
            compute_pipeline,
            output_texture,
            
            // 性能监控相关
            query_set,
            query_buffer,
            mapped_buffer,
            is_perf_mode: false,
            last_compute_time: 0.0,
            last_render_time: 0.0,
            frame_count: 0,
            last_perf_print: std::time::Instant::now(),
            
            // 连续生成模式
            is_ser_spawn_mode: false,
            ser_spawn_model_id: 0,
            
            // Tile 加速相关
            tile_buffer,
            tile_map_cache: Vec::new(),
        };

        // 关键：启动时先计算一次场景 SDF
        // state.update_global_sdf(); // 暂时停用，因为 C++ 还没支持 DNA 合并

        state
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.render_context.resize(new_size);
            
            // 更新输出纹理
            let device = &self.render_context.device;
            
            // 创建新的输出纹理
            let new_output_texture = device.create_texture(&wgpu::TextureDescriptor {
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
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let new_output_texture_view = new_output_texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            // 更新绑定组
            let new_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&new_output_texture_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.entity_list_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.global_sdf_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.palette_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.tile_buffer.as_entire_binding() },
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
            
            // 更新状态
            self.output_texture = new_output_texture;
            self.compute_bind_group = new_compute_bind_group;
            self.blit_bind_group = new_blit_bind_group;
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
        
        // 如果 ray_dir.y 趋近于 0 或大于 0 (看向天空)，t 会变成无限大或负数
        if ray_dir.y.abs() < 0.001 {
            return self.camera.eye + ray_dir * 10.0; // 兜底：放在相机前 10 米
        }
        
        let t = -self.camera.eye.y / ray_dir.y;
        if t < 0.0 {
            return self.camera.eye + ray_dir * 10.0; // 兜底
        }
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

    // 获取 wgpu 设备显存使用情况（估算）
    fn get_vram_usage(&self) -> (u64, u64) {
        // wgpu 没有直接提供显存查询 API，但我们可以：
        // 1. 手动累加已知的显存分配
        let mut total_allocated = 0u64;
        
        // 全局 SDF 纹理
        let sdf_size = self.global_sdf_texture.size();
        total_allocated += (sdf_size.width * sdf_size.height * sdf_size.depth_or_array_layers) as u64 * 8; // Rg32Uint = 8 bytes
        
        // 输出纹理
        let output_size = self.output_texture.size();
        total_allocated += (output_size.width * output_size.height) as u64 * 4; // RGBA8 = 4 bytes
        
        // 实体列表缓冲区
        total_allocated += self.entity_list_buffer.size();
        
        // 参数缓冲区
        total_allocated += self.params_buffer.size();
        
        // 颜色库缓冲区
        total_allocated += self.palette_buffer.size();
        
        // 每个模型的实例缓冲区和顶点缓冲区（虽然你没用到顶点，但为了完整）
        for model in self.models.values() {
            total_allocated += model.instance_buffer.size();
            total_allocated += model.vertex_buffer.size();
            total_allocated += model.index_buffer.size();
        }
        
        // 粗略估算驱动/交换链开销（通常是实际分配量的 20-30%）
        let driver_overhead = total_allocated / 4;
        
        (total_allocated, total_allocated + driver_overhead)
    }
    
    // 获取系统内存使用（RSS，驻留集大小）
    fn get_ram_usage(&self) -> usize {
        #[cfg(target_os = "windows")]
        {
            use winapi::um::psapi::GetProcessMemoryInfo;
            use winapi::um::processthreadsapi::GetCurrentProcess;
            use winapi::um::psapi::PROCESS_MEMORY_COUNTERS;
            use std::mem::zeroed;
            
            unsafe {
                let mut pmc: PROCESS_MEMORY_COUNTERS = zeroed();
                pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
                if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                    return pmc.WorkingSetSize as usize;
                }
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: usize = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            use libc::{task_info, task_t, task_basic_info, task_basic_info_t, mach_task_self_, TASK_BASIC_INFO, natural_t, integer_t};
            unsafe {
                let mut info = std::mem::zeroed::<task_basic_info>();
                let mut count = (std::mem::size_of::<task_basic_info>() / std::mem::size_of::<natural_t>()) as natural_t;
                let ret = task_info(
                    mach_task_self_,
                    TASK_BASIC_INFO,
                    &mut info as *mut _ as *mut integer_t,
                    &mut count,
                );
                if ret == 0 {
                    return info.resident_size;
                }
            }
        }
        
        // Fallback: 估算（不准确但总比没有好）
        std::mem::size_of::<Self>()
    }

    fn execute_command(&mut self) {
        let command = self.console.execute();
        if command.is_empty() {
            self.console.is_open = false;
            self.edit_mode = EditMode::Idle;
            return;
        }

        if command == "mem" || command == "vram" {
            // --- 1. 计算核心 DNA 堆栈 ---
            let model_count = self.models.len();
            let sdf_res = 64u64;
            // Rg32Uint = 8 bytes per voxel
            let dna_per_model = sdf_res * sdf_res * sdf_res * 8;
            let total_dna_vram = dna_per_model * model_count as u64;

            // --- 2. 计算 Tile 瓦片系统 (4K 预留空间) ---
            // Stride 132 u32 = 528 bytes
            let tile_stride = 132 * 4;
            let max_tiles = ((3840 + 15) / 16) * ((2160 + 15) / 16);
            let total_tile_vram = max_tiles as u64 * tile_stride as u64;

            // --- 3. 计算输出画布 ---
            let (w, h) = (self.render_context.size.width as u64, self.render_context.size.height as u64);
            let canvas_vram = w * h * 4; // Rgba8

            // --- 4. 计算实体清单 ---
            let entity_list_vram = 1024 * 256; // 1024 slots * 256 bytes

            // --- 打印报告 ---
            println!("┌──────────────────────────────────────────────────────────┐");
            println!("│              CILLIN ENGINE VRAM REPORT (CEM3)            │");
            println!("├──────────────────────────────────────────────────────────┤");
            println!("│ [Holographic DNA Stack]                                  │");
            println!("│  - Per Model:   {:>8.2} MB (64^3 * 8B)               │", dna_per_model as f64 / 1024.0 / 1024.0);
            println!("│  - Model Count: {:>8}                                  │", model_count);
            println!("│  - Total DNA:   {:>8.2} MB                               │", total_dna_vram as f64 / 1024.0 / 1024.0);
            println!("│                                                          │");
            println!("│ [Adaptive Tile System]                                   │");
            println!("│  - Stride:      {:>8} Bytes (128 Indices)            │", tile_stride);
            println!("│  - Buffer Size: {:>8.2} MB (Fixed for 4K)              │", total_tile_vram as f64 / 1024.0 / 1024.0);
            println!("│                                                          │");
            println!("│ [Render Canvas & Pipeline]                               │");
            println!("│  - Resolution:  {}x{}                               │", w, h);
            println!("│  - Framebuffer: {:>8.2} MB                               │", canvas_vram as f64 / 1024.0 / 1024.0);
            println!("│  - Entity List: {:>8.2} KB                               │", entity_list_vram as f64 / 1024.0);
            println!("├──────────────────────────────────────────────────────────┤");
            let total_vram = total_dna_vram + total_tile_vram + canvas_vram + entity_list_vram as u64;
            println!("│ TOTAL ALLOCATED VRAM: {:>10.2} MB                      │", total_vram as f64 / 1024.0 / 1024.0);
            println!("└──────────────────────────────────────────────────────────┘");

            self.console.is_open = false;
            self.edit_mode = EditMode::Idle;
            return;
        }

        // 处理 stat 命令
        if command == "stat" {
            self.is_perf_mode = !self.is_perf_mode;
            println!("性能监控已{}", if self.is_perf_mode { "开启" } else { "关闭" });
            self.console.is_open = false;
            self.edit_mode = EditMode::Idle;
            return;
        }

        println!("执行命令: {}", command);
        
        let parts: Vec<&str> = command.split_whitespace().collect();
        if let Some(cmd) = parts.first() {
            match *cmd {
                "help" => println!("可用命令: help, exit, spawn <model_id>, serspawn <model_id>, cmr <x> <y> <z>, cmr now, sdf, shadow, ao, debug <0|1|2|3|4|5>, mem, stat, modelinfo"),
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
                "serspawn" => {
                    if parts.len() >= 2 {
                        if let Ok(model_id) = parts[1].parse::<u32>() {
                            self.is_ser_spawn_mode = true;
                            self.ser_spawn_model_id = model_id;
                            println!("已开启连续生成模式，模型 ID: {}", model_id);
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
                        println!("用法: debug <0|1|2|3|4|5> (0:关, 1:距离场(红正绿负), 2:坐标映射, 3:法线, 4:原始颜色, 5:距离场扫描(红内蓝外))");
                    }
                }
                "modelinfo" => {
                    println!("========== 模型信息 ==========");
                    for entity in &self.entities {
                        let model_name = self.manifest.get(&entity.model_id).map(|m| m.name.as_str()).unwrap_or("Unknown");
                        println!("实体 ID: {}, 模型 ID: {}, 模型名称: {}", entity.id, entity.model_id, model_name);
                    }
                    println!("==============================");
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
        
        // 处理 delete 键删除模型
        if self.input_state.is_delete_pressed && self.selected_idx.is_some() {
            let idx = self.selected_idx.unwrap();
            if idx < self.entities.len() {
                let entity = &self.entities[idx];
                println!("已删除模型: ID={}, ModelID={}", entity.id, entity.model_id);
                self.entities.remove(idx);
                // 重新分配 ID
                for (i, entity) in self.entities.iter_mut().enumerate() {
                    entity.id = i;
                }
                self.selected_idx = None;
            }
        }
        
        // 处理连续生成模式
        if self.is_ser_spawn_mode && self.input_state.is_left_mouse_pressed {
            let pos = self.get_click_ground_pos();
            let model_id = self.ser_spawn_model_id;
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
        }
        
        // 更新相机位置
        self.controller.update(&mut self.camera, 1.0);
        
        // 更新视图和投影矩阵
        self.render_context.current_view = self.camera.get_view_matrix();
        self.render_context.current_proj = Mat4::perspective_rh(
            45.0f32.to_radians(), 
            self.render_context.size.width as f32 / self.render_context.size.height as f32, 
            0.1, 
            10000.0
        );
        
        // 更新相机缓冲区
        let camera_matrix = self.render_context.current_proj * self.render_context.current_view;
        self.render_context.queue.write_buffer(
            &self.camera_buffer, 
            0, 
            bytemuck::cast_slice(camera_matrix.as_ref())
        );
        
        // 重置输入状态
        self.input_state.reset();
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

    fn print_perf_stats(&self) {
        // 1. 计算显存 (VRAM)
        let sdf_size = (64 * 64 * 64 * self.models.len() * 4) as f64 / 1024.0 / 1024.0;
        let out_tex_size = (self.render_context.size.width * self.render_context.size.height * 4) as f64 / 1024.0 / 1024.0;
        let total_vram = sdf_size + out_tex_size + 2.0; // 2MB 是 Buffers 的估算

        // 2. 打印精美的控制台报表
        println!("--- Cillin Engine Perf Stats ---");
        println!("VRAM Usage:  {:.2} MB (SDF Data: {:.2} MB)", total_vram, sdf_size);
        println!("GPU Brain:   {:.3} ms (Xi-Luoer Compute)", self.last_compute_time);
        println!("GPU Eye:     {:.3} ms (Blit & UI)", self.last_render_time);
        println!("Total Frame: {:.2} FPS", 1000.0 / (self.last_compute_time + self.last_render_time));
        println!("-------------------------------");
    }

    fn process_query_results(&mut self) {
        let device = &self.render_context.device;
        let queue = &self.render_context.queue;

        // 创建一个新的命令编码器来复制查询结果
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.resolve_query_set(&self.query_set, 0..4, &self.query_buffer, 0);
        // 将查询结果复制到可映射的缓冲区
        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.mapped_buffer, 0, 32);
        queue.submit(std::iter::once(encoder.finish()));

        // 映射缓冲区以读取结果
        let buffer_slice = self.mapped_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            if timestamps.len() >= 4 {
                let compute_start = timestamps[0];
                let compute_end = timestamps[1];
                let render_start = timestamps[2];
                let render_end = timestamps[3];

                // 计算时间差（纳秒）
                let compute_time_ns = compute_end - compute_start;
                let render_time_ns = render_end - render_start;

                // 转换为毫秒
                self.last_compute_time = compute_time_ns as f32 / 1_000_000.0;
                self.last_render_time = render_time_ns as f32 / 1_000_000.0;

                // 每 10 帧打印一次性能统计
                self.frame_count += 1;
                if self.is_perf_mode && self.frame_count % 10 == 0 {
                    self.print_perf_stats();
                }
            }
        }

        // 取消映射
        self.mapped_buffer.unmap();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 性能革命：先更新 Tile 名单
        self.update_tile_buffer();
        
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

        // --- 1. 监控大脑 (Xi-Luoer 混合算法耗时) ---
        encoder.write_timestamp(&self.query_set, 0); // 记下开始时间
        // 第一步：计算大脑
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]); // 使用计算专用组
            cpass.dispatch_workgroups((self.render_context.size.width + 7) / 8, (self.render_context.size.height + 7) / 8, 1);
        }
        encoder.write_timestamp(&self.query_set, 1); // 记下结束时间

        // --- 2. 监控眼睛 (最终画面呈现耗时) ---
        encoder.write_timestamp(&self.query_set, 2);
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
        encoder.write_timestamp(&self.query_set, 3);

        self.render_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // 处理查询结果
        self.process_query_results();

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
                // 关键修复 1: 无论如何，先让 input_state 记录下点击状态
                state.input_state.process_event(event);

                // 如果控制台开着，不处理 3D 场景点击
                if state.console.is_open { return; }

                if button == &MouseButton::Middle {
                    if *button_state == ElementState::Pressed {
                        state.input_state.last_mouse_pos = Some(state.input_state.mouse_pos);
                    } else {
                        state.input_state.last_mouse_pos = None;
                    }
                } else if button == &MouseButton::Left {
                    if *button_state == ElementState::Pressed {
                        // 如果处于连续生成模式，点击时不应该触发“选中/取消选中”逻辑，避免冲突
                        if state.is_ser_spawn_mode {
                            // 这里可以留空，因为生成逻辑在 update() 里面通过 is_left_mouse_pressed 触发
                            return;
                        }

                        // 原有的选中逻辑
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
            WindowEvent::CursorMoved { .. } => {
                state.input_state.process_event(event);
            }
            WindowEvent::MouseWheel { delta: _, .. } => {
                state.input_state.process_event(event);
                // 使用滚轮调整相机位置
                state.camera.eye += state.camera.get_forward() * state.input_state.scroll_delta * 0.1;
                state.input_state.reset_scroll();
            }
            WindowEvent::KeyboardInput { .. } => {
                state.input_state.process_event(event);
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
