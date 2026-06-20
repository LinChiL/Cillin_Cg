use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use glam::{Vec3, Vec4, Vec3Swizzles, Vec4Swizzles};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use gltf;
use rfd;

#[derive(Hash, Copy, Clone, PartialEq, Eq)]
struct VertexKey {
    x: i32,
    y: i32,
    z: i32,
}

impl VertexKey {
    fn from_pos(p: [f32; 3]) -> Self {
        Self {
            x: (p[0] * 1000.0) as i32,
            y: (p[1] * 1000.0) as i32,
            z: (p[2] * 1000.0) as i32,
        }
    }
}

fn compute_smoothed_triangles(
    positions: &[[f32; 3]],
    indices: &[u32],
    uvs: &[[f32; 2]],
    threshold_deg: f32,
) -> (Vec<math::Triangle>, usize) {
    let threshold_cos = threshold_deg.to_radians().cos();
    let mut tris = Vec::new();

    let mut face_normals = Vec::new();
    for chunk in indices.chunks_exact(3) {
        let p0 = Vec3::from_slice(&positions[chunk[0] as usize]);
        let p1 = Vec3::from_slice(&positions[chunk[1] as usize]);
        let p2 = Vec3::from_slice(&positions[chunk[2] as usize]);
        let n = if (p1 - p0).cross(p2 - p0).length_squared() > 0.0001 {
            (p1 - p0).cross(p2 - p0).normalize()
        } else {
            Vec3::Y
        };
        face_normals.push(n);
    }

    let mut pos_to_faces: HashMap<VertexKey, Vec<usize>> = HashMap::new();
    for (face_idx, chunk) in indices.chunks_exact(3).enumerate() {
        for &v_idx in chunk {
            let p = positions[v_idx as usize];
            let key = VertexKey::from_pos(p);
            pos_to_faces.entry(key).or_default().push(face_idx);
        }
    }
    
    let unique_vertex_count = pos_to_faces.len();

    for (face_idx, chunk) in indices.chunks_exact(3).enumerate() {
        let current_face_normal = face_normals[face_idx];
        let mut vertex_smoothed_normals = [Vec3::ZERO; 3];

        for i in 0..3 {
            let v_idx = chunk[i] as usize;
            let p = positions[v_idx];
            let key = VertexKey::from_pos(p);

            let mut sum_n = Vec3::ZERO;
            if let Some(adjacent_faces) = pos_to_faces.get(&key) {
                for &adj_face_idx in adjacent_faces {
                    let adj_normal = face_normals[adj_face_idx];
                    if current_face_normal.dot(adj_normal) >= threshold_cos {
                        sum_n += adj_normal;
                    }
                }
            }
            vertex_smoothed_normals[i] = if sum_n.length_squared() > 0.0001 {
                sum_n.normalize()
            } else {
                current_face_normal
            };
        }

        let p0 = positions[chunk[0] as usize];
        let p1 = positions[chunk[1] as usize];
        let p2 = positions[chunk[2] as usize];
        let uv0 = uvs[chunk[0] as usize];
        let uv1 = uvs[chunk[1] as usize];
        let uv2 = uvs[chunk[2] as usize];

        tris.push(math::Triangle {
            v0: [p0[0], p0[1], p0[2], 1.0],
            v1: [p1[0], p1[1], p1[2], 1.0],
            v2: [p2[0], p2[1], p2[2], 1.0],
            n0: vertex_smoothed_normals[0].extend(0.0).to_array(),
            n1: vertex_smoothed_normals[1].extend(0.0).to_array(),
            n2: vertex_smoothed_normals[2].extend(0.0).to_array(),
            uv01: [uv0[0], uv0[1], uv1[0], uv1[1]],
            uv2: [uv2[0], uv2[1], 0.0, 0.0],
        });
    }

    (tris, unique_vertex_count)
}

mod math;
use math::{Params, MeshSample, Triangle};

fn pixel_buffer_size<T>(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64) * (std::mem::size_of::<T>() as u64)
}

// 编辑模式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditMode {
    None,
    Grab,
    Rotate,
    Scale,
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
    scaffold_buffer: wgpu::Buffer,
    triangle_buffer: wgpu::Buffer,
    voronoi_texture: wgpu::Texture,
    voronoi_texture_view: wgpu::TextureView,
    params: math::Params,
    camera: math::Camera,
    is_mmb_pressed: bool,
    is_shift_pressed: bool,
    is_lmb_pressed: bool, // 左键点击状态
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
    scaffold_vertices: Vec<glam::Vec3>,
    scaffold_path: Option<String>,
    triangles: Vec<math::Triangle>,
    
    // 新增：唯一顶点 + 邻接信息（用于背面剔除）
    vertex_positions: Vec<glam::Vec3>,     // 唯一顶点
    vertex_triangles: Vec<Vec<u32>>,       // 每个顶点连接的三角形索引列表
    
    show_scaffold: bool, // 控制是否显示点云
    visible_vertices: Vec<glam::Vec4>,   // 屏幕空间 + 深度 (x,y,depth,1.0)
    
    // 深度图调试
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    depth_texture_view_for_render: wgpu::TextureView,
    tri_id_texture: wgpu::Texture,
    tri_id_texture_view: wgpu::TextureView,
    tri_id_texture_view_for_render: wgpu::TextureView,
    warp_buffer: wgpu::Buffer,
    sdf_buffer: wgpu::Buffer,
    ap2_pipeline: wgpu::ComputePipeline,
    ap3_pipeline: wgpu::ComputePipeline,
    depth_bind_group_layout: wgpu::BindGroupLayout,
    depth_bind_group: wgpu::BindGroup,
    depth_render_pipeline: wgpu::RenderPipeline,
    depth_blit_pipeline: wgpu::RenderPipeline,
    depth_blit_bind_group: wgpu::BindGroup,
    show_depth_debug: bool,
    show_normal_debug: bool,
    debug_mode: u32,
    
    // UV G-Buffer (存世界坐标)
    uv_texture: wgpu::Texture,
    uv_texture_view: wgpu::TextureView,
    uv_texture_view_for_render: wgpu::TextureView,
    
    // 模型 UV 纹理 (存模型的 UV 坐标)
    model_uv_texture: wgpu::Texture,
    model_uv_texture_view: wgpu::TextureView,
    model_uv_texture_view_for_render: wgpu::TextureView,

    // 世界法线纹理
    normal_texture: wgpu::Texture,
    normal_texture_view: wgpu::TextureView,
    normal_texture_view_for_render: wgpu::TextureView,

    // Albedo 贴图
    albedo_texture: wgpu::Texture,
    albedo_texture_view: wgpu::TextureView,
    albedo_sampler: wgpu::Sampler,

    // 模型库管理
    model_registry: std::collections::HashMap<u32, math::ModelRegistryItem>,
    material_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_groups: std::collections::HashMap<u32, wgpu::BindGroup>,
    mesh_render_pipeline: wgpu::RenderPipeline,

    // 场景中的实例列表
    instances: Vec<math::InstanceData>,
    instance_buffer: wgpu::Buffer,

    // 交互控制
    command_input: String,
    active_spawn_id: Option<u32>,
    
    // 编辑模式相关
    selected_instance: Option<usize>,
    edit_mode: EditMode,
    initial_pos: glam::Vec3,           // 变换开始时物体的位置
    initial_rot: glam::Quat,           // 变换开始时物体的旋转
    initial_scale: glam::Vec3,         // 变换开始时物体的缩放
    initial_mouse_pos: glam::Vec2,     // 变换开始时的鼠标位置
}

impl<'a> App<'a> {
    fn load_all_models(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_layout: &wgpu::BindGroupLayout,
        albedo_sampler: &wgpu::Sampler,
    ) -> (
        std::collections::HashMap<u32, math::ModelRegistryItem>,
        std::collections::HashMap<u32, wgpu::BindGroup>,
        Vec<math::Triangle>,
    ) {
        let manifest_path = "cremModel/manifest.json";
        let manifest_str = std::fs::read_to_string(manifest_path).expect("无法读取 manifest.json");
        let manifest: math::ModelManifest = serde_json::from_str(&manifest_str).expect("解析 JSON 失败");

        let mut registry = std::collections::HashMap::new();
        let mut material_bind_groups = std::collections::HashMap::new();
        let mut all_triangles = Vec::new();

        for item in manifest.models {
            let file_path = format!("cremModel/{}", item.file);
            println!("正在预加载模型: {}", file_path);

            let tri_start = all_triangles.len() as u32;
            let (tris, texture_view) = Self::load_glb_with_texture(device, queue, &file_path);
            let tri_count = tris.len() as u32;
            all_triangles.append(&mut tris.clone());

            let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Material Bind Group {}", item.id)),
                layout: material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(albedo_sampler),
                    },
                ],
            });

            material_bind_groups.insert(item.id, material_bind_group);

            registry.insert(item.id, math::ModelRegistryItem {
                info: item.clone(),
                tri_start,
                tri_count,
                material_id: item.id,
            });
        }

        (registry, material_bind_groups, all_triangles)
    }

    fn load_glb_with_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> (Vec<math::Triangle>, wgpu::TextureView) {
        let (document, buffers, images) = gltf::import(path).expect("加载 GLB 失败");
        let mut triangles = Vec::new();

        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let (mut smoothed_tris, _) = compute_smoothed_triangles(&positions, &indices, &uvs, 45.0);
                triangles.append(&mut smoothed_tris);
            }
        }

        let texture_view = if let Some(image) = images.first() {
            let width = image.width;
            let height = image.height;
            let pixels = image.pixels.clone();
            let bytes_per_pixel = pixels.len() / (width * height) as usize;

            let rgba_pixels: Vec<u8> = if bytes_per_pixel == 3 {
                pixels.chunks(3)
                    .flat_map(|rgb| vec![rgb[0], rgb[1], rgb[2], 255u8])
                    .collect()
            } else {
                pixels.to_vec()
            };

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Model Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba_pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            texture.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Fallback Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default())
        };

        (triangles, texture_view)
    }

    fn load_glb_triangles_only(path: &str) -> Vec<math::Triangle> {
        let (document, buffers, _) = gltf::import(path).expect("加载 GLB 失败");
        let mut triangles = Vec::new();
        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let (mut smoothed_tris, _) = compute_smoothed_triangles(&positions, &indices, &uvs, 45.0);
                triangles.append(&mut smoothed_tris);
            }
        }
        triangles
    }

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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sculpt Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sculpt.wgsl"))),
        });

        let mut params = Params::default();

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<math::Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 初始写入参数
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        // 预设最大支持 10 万个脚手架点（初始值，会动态扩容）
        let scaffold_max_size = (100_000 * 16) as u64; // vec4 是 16 字节
        let scaffold_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scaffold Buffer"),
            size: scaffold_max_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 三角形缓冲区 (最多 10 万个三角形 * 80 字节，包含 UV 数据)（初始值，会动态扩容）
        let triangle_max_size = (100_000 * 80) as u64;
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
                // Binding 0: Output Texture
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
                // Binding 1: Params (Uniform Buffer)
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
                // Binding 2: Triangles (Storage Buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Scaffold (Storage Buffer)
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
                // Binding 4: Depth Texture (Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 5: Tri ID Texture (Uint)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 6: UV Texture (Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 7: Albedo Texture (Float, filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 8: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 9: Instances (Storage Buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 模型 UV 纹理 (binding 12)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 世界法线纹理 (binding 13)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // 从 3 改成 2，对应 WGSL 里的 triangles
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, // 增加 FRAGMENT 可见性
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 新增：Binding 9 (instances)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, // 增加 FRAGMENT 可见性
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let tri_id_texture_view = tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let tri_id_texture_view_for_render = tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let warp_buffer_size = pixel_buffer_size::<math::WarpPixel>(size.width, size.height);
        let warp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Warp Buffer"),
            size: warp_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sdf_buffer_size = (size.width * size.height * 4) as u64;
        let sdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Buffer"),
            size: sdf_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // UV G-Buffer 纹理 (存世界坐标)
        let uv_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UV G-Buffer"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let uv_texture_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_texture_view_for_render = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 模型 UV 纹理 (存模型的 UV 坐标)
        let model_uv_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Model UV Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let model_uv_texture_view = model_uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let model_uv_texture_view_for_render = model_uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 世界法线纹理 (存世界空间法线)
        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_texture_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_texture_view_for_render = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 默认 Albedo 纹理（棋盘格）
        let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Albedo Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 创建默认棋盘格纹理
        let albedo_size = 256u32;
        let mut albedo_data = Vec::with_capacity((albedo_size * albedo_size * 4) as usize);
        for y in 0..albedo_size {
            for x in 0..albedo_size {
                let color = if ((x / 32) + (y / 32)) % 2 == 0 {
                    [0.8, 0.8, 0.8, 1.0]
                } else {
                    [0.4, 0.4, 0.4, 1.0]
                };
                albedo_data.extend_from_slice(&color);
            }
        }

        let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Texture"),
            size: wgpu::Extent3d {
                width: albedo_size,
                height: albedo_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &albedo_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&albedo_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(albedo_size * 4),
                rows_per_image: Some(albedo_size),
            },
            wgpu::Extent3d {
                width: albedo_size,
                height: albedo_size,
                depth_or_array_layers: 1,
            },
        );
        
        let albedo_texture_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());



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

        let ap2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AP2 Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_ap2",
        });

        let ap3_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AP3 Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_ap3",
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

        // === 轻量深度图管线 ===
        let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Pipeline Layout"),
            bind_group_layouts: &[&depth_bind_group_layout],
            push_constant_ranges: &[],
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
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
                depth_compare: wgpu::CompareFunction::LessEqual,
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

        // === 创建材质绑定组布局（用于多材质支持）===
        let material_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 创建网格渲染管线布局（基础绑定组 + 材质绑定组）
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[&depth_bind_group_layout, &material_bind_group_layout],
            push_constant_ranges: &[],
        });

        // 创建网格渲染管线
        let mesh_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Render Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_mesh",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // === 加载所有模型到模型库 ===
        let (model_registry, material_bind_groups, all_triangles) = Self::load_all_models(
            &device,
            &queue,
            &material_bind_group_layout,
            &albedo_sampler,
        );

        // 上传所有三角形到 GPU 缓冲区
        let tri_count = all_triangles.len();
        let tri_size = (tri_count * std::mem::size_of::<math::Triangle>()) as u64;
        let triangle_buffer_to_use = if triangle_buffer.size() < tri_size {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Triangle Buffer (Resized)"),
                size: (tri_size as f32 * 1.5) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            triangle_buffer
        };
        queue.write_buffer(&triangle_buffer_to_use, 0, bytemuck::cast_slice(&all_triangles));

        // 将所有三角形存储到 self.triangles（用于现有渲染逻辑）
        let triangles = all_triangles.clone();
        
        // 更新 anchor_count 为所有模型三角形的总和
        params.anchor_count = all_triangles.len() as u32;
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        // 创建实例缓冲区（初始容量 1024 个实例）
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (1024 * std::mem::size_of::<math::InstanceData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 使用统一的 BindGroup 创建方法
        let (compute_bind_group, render_bind_group, depth_bind_group, depth_blit_bind_group) = Self::create_all_bind_groups(
            &device,
            &compute_bind_group_layout,
            &render_bind_group_layout,
            &depth_bind_group_layout,
            &depth_blit_bind_group_layout,
            &output_texture_view,
            &params_buffer,
            &triangle_buffer_to_use,
            &scaffold_buffer,
            &depth_texture_view,
            &tri_id_texture_view,
            &uv_texture_view,
            &model_uv_texture_view,
            &normal_texture_view,
            &albedo_texture_view,
            &albedo_sampler,
            &instance_buffer,
            &warp_buffer,
            &sdf_buffer,
        );

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
            scaffold_buffer,
            triangle_buffer: triangle_buffer_to_use,
            voronoi_texture,
            voronoi_texture_view,
            params,
            camera: math::Camera::new(glam::Vec3::new(0.0, 1.0, -5.0), 0.0, 0.0),
            is_mmb_pressed: false,
            is_shift_pressed: false,
            is_lmb_pressed: false,
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
            scaffold_vertices: Vec::new(),
            scaffold_path: None,
            triangles,
            vertex_positions: Vec::new(),     // 初始化唯一顶点
            vertex_triangles: Vec::new(),     // 初始化邻接信息
            show_scaffold: false, // 默认关闭点云，显示三角形表面
            visible_vertices: Vec::new(),
            depth_texture,
            depth_texture_view,
            depth_texture_view_for_render,
            tri_id_texture,
            tri_id_texture_view,
            tri_id_texture_view_for_render,
            warp_buffer,
            sdf_buffer,
            ap2_pipeline,
            ap3_pipeline,
            depth_bind_group_layout,
            depth_bind_group,
            depth_render_pipeline,
            depth_blit_pipeline,
            depth_blit_bind_group,
            show_depth_debug: false,
            show_normal_debug: false,
            debug_mode: 0u32,
            uv_texture,
            uv_texture_view,
            uv_texture_view_for_render,
            model_uv_texture,
            model_uv_texture_view,
            model_uv_texture_view_for_render,
            normal_texture,
            normal_texture_view,
            normal_texture_view_for_render,
            albedo_texture,
            albedo_texture_view,
            albedo_sampler,
            model_registry,
            material_bind_group_layout,
            material_bind_groups,
            mesh_render_pipeline,
            instances: Vec::new(),
            instance_buffer,
            command_input: String::new(),
            active_spawn_id: None,
            
            // 编辑模式初始化
            selected_instance: None,
            edit_mode: EditMode::None,
            initial_pos: glam::Vec3::ZERO,
            initial_rot: glam::Quat::IDENTITY,
            initial_scale: glam::Vec3::ONE,
            initial_mouse_pos: glam::Vec2::ZERO,
        }
    }

    // 统一组装所有的 BindGroup
    fn create_all_bind_groups(
        device: &wgpu::Device,
        compute_layout: &wgpu::BindGroupLayout,
        render_layout: &wgpu::BindGroupLayout,
        depth_layout: &wgpu::BindGroupLayout,
        depth_blit_layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        triangle_buffer: &wgpu::Buffer,
        scaffold_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
        tri_id_view: &wgpu::TextureView,
        uv_view: &wgpu::TextureView,
        model_uv_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        albedo_view: &wgpu::TextureView,
        albedo_sampler: &wgpu::Sampler,
        instance_buffer: &wgpu::Buffer,
        warp_buffer: &wgpu::Buffer,
        sdf_buffer: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: compute_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(output_view) },
                wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: triangle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scaffold_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(tri_id_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(uv_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(albedo_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(albedo_sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: warp_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: sdf_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(model_uv_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(normal_view) },
            ],
        });

        let render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(output_view) },
            ],
        });

        let depth = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Bind Group"),
            layout: depth_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: triangle_buffer.as_entire_binding() }, // 从 3 改成 2
                wgpu::BindGroupEntry { binding: 9, resource: instance_buffer.as_entire_binding() },
            ],
        });

        let depth_blit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Blit Bind Group"),
            layout: depth_blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            ],
        });

        (compute, render, depth, depth_blit)
    }

    fn ensure_buffer_size(
        device: &wgpu::Device,
        buffer: &mut wgpu::Buffer,
        required_size: u64,
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> bool {
        if buffer.size() < required_size {
            let max_buffer_size = 128 * 1024 * 1024; // 128MB - wgpu 绑定限制
            let new_size = std::cmp::min((required_size as f32 * 1.5) as u64, max_buffer_size);
            *buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_size,
                usage,
                mapped_at_creation: false,
            });
            true
        } else {
            false
        }
    }

    // 从深度图中拾取实例
    fn pick_instance(&mut self, mouse_pos: [f32; 2]) {
        let x = mouse_pos[0] as u32;
        let y = mouse_pos[1] as u32;

        // 创建 4 字节的暂存 Buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        // 拷贝点击位置的那 1 个像素
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.tri_id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        // 读取数据
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait); // 等待 GPU 完成

        let data = buffer_slice.get_mapped_range();
        let result = u32::from_ne_bytes(data[0..4].try_into().unwrap());
        
        if result > 0 {
            // 解码：直接存储 instance_id + 1
            let instance_idx = (result - 1) as usize;
            if instance_idx < self.instances.len() {
                self.selected_instance = Some(instance_idx);
                self.params.selected_instance_id = instance_idx as u32;
                println!("选中了实例: {}", instance_idx);
                
                // 保存当前变换状态
                if let Some(inst) = self.instances.get(instance_idx) {
                    let mat = glam::Mat4::from_cols_array_2d(&inst.model_matrix);
                    let (scale, rot, pos) = mat.to_scale_rotation_translation();
                    self.initial_pos = pos;
                    self.initial_rot = rot;
                    self.initial_scale = scale;
                }
            } else {
                self.selected_instance = None;
                self.params.selected_instance_id = u32::MAX;
            }
        } else {
            self.selected_instance = None;
            self.params.selected_instance_id = u32::MAX;
        }
        
        // 同步 params 到 GPU
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
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

    fn run_depth_pass(&self, encoder: &mut wgpu::CommandEncoder, instances_to_draw: &[math::InstanceData]) {
        if instances_to_draw.is_empty() && (self.params.anchor_count == 0 || self.triangles.is_empty()) {
            return;
        }

        let mut dpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth + ID + UV MRT Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.tri_id_texture_view_for_render,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.uv_texture_view_for_render,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.model_uv_texture_view_for_render,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.normal_texture_view_for_render,
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

        if !instances_to_draw.is_empty() {
            for (i, instance) in instances_to_draw.iter().enumerate() {
                if let Some(reg) = self.model_registry.get(&instance.model_id) {
                    let vertex_start = reg.tri_start * 3;
                    let vertex_end = (reg.tri_start + reg.tri_count) * 3;
                    dpass.draw(vertex_start..vertex_end, (i as u32)..(i as u32 + 1));
                }
            }
        } else if self.params.anchor_count > 0 {
            dpass.draw(0..(self.triangles.len() as u32 * 3), 0..1);
        }
    }

    fn run_forward_pass(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, instances_to_draw: &[math::InstanceData]) {
        if instances_to_draw.is_empty() {
            return;
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Forward Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_pipeline(&self.mesh_render_pipeline);
        rpass.set_bind_group(0, &self.depth_bind_group, &[]);

        for (i, instance) in instances_to_draw.iter().enumerate() {
            if let Some(reg) = self.model_registry.get(&instance.model_id) {
                if let Some(mat_bg) = self.material_bind_groups.get(&reg.material_id) {
                    rpass.set_bind_group(1, mat_bg, &[]);
                    let vertex_start = reg.tri_start * 3;
                    let vertex_end = (reg.tri_start + reg.tri_count) * 3;
                    rpass.draw(vertex_start..vertex_end, (i as u32)..(i as u32 + 1));
                }
            }
        }
    }

    fn run_compute_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_bind_group, &[]);
        cpass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
    }

    fn clear_warp_buffer(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.warp_buffer, 0, None);
    }

    fn run_ap2_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("AP2 Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.ap2_pipeline);
        cpass.set_bind_group(0, &self.compute_bind_group, &[]);
        cpass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
    }

    fn run_ap3_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("AP3 Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.ap3_pipeline);
        cpass.set_bind_group(0, &self.compute_bind_group, &[]);
        cpass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
    }

    fn run_draw_pass(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        if self.show_depth_debug && self.params.anchor_count > 0 {
            let mut bpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
        } else {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.5, a: 1.0 }),
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
    }

    fn run_scaffold_pass(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Draw Scaffold"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
        rpass.set_pipeline(&self.scaffold_render_pipeline);
        rpass.set_bind_group(0, &self.compute_bind_group, &[]);
        if self.show_scaffold && self.params.scaffold_count > 0 {
            rpass.draw(0..self.params.scaffold_count, 0..1);
        }
    }

    fn run_ui_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        paint_jobs: &[egui::ClippedPrimitive],
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Egui Main Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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

        self.egui_renderer.render(&mut rpass, paint_jobs, screen_descriptor);
    }

    fn update(&mut self, delta_time: f32) {
        // WASD 移动逻辑
        let move_speed = 2.0 * delta_time;
        if self.is_w_pressed {
            self.camera.eye += self.camera.get_forward() * move_speed;
        }
        if self.is_s_pressed && self.edit_mode == EditMode::None {
            self.camera.eye -= self.camera.get_forward() * move_speed;
        }
        if self.is_a_pressed {
            self.camera.eye -= self.camera.get_right() * move_speed;
        }
        if self.is_d_pressed {
            self.camera.eye += self.camera.get_right() * move_speed;
        }

        // 使用新的相机类生成矩阵
        self.params.update_matrices(&self.camera, self.config.width, self.config.height);
        self.params.time += delta_time;
        self.params.debug_mode = self.debug_mode;
        
        // 计算 FPS
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_frame_time).as_secs_f32();
        if elapsed > 0.0 {
            self.fps = 1.0 / elapsed;
        }
        self.last_frame_time = now;
        
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));

        // 放置预览逻辑：只有在放置模式下才运行
        if let Some(model_id) = self.active_spawn_id {
            let (ray_o, ray_dir) = self.camera.get_ray(
                self.last_mouse_pos[0],
                self.last_mouse_pos[1],
                self.config.width,
                self.config.height
            );

            if ray_dir.y.abs() > 0.001 {
                let t = -ray_o.y / ray_dir.y;
                if t > 0.0 {
                    let intersect_pos = ray_o + ray_dir * t;

                    // --- 虚影逻辑 ---
                    // 获取模型的默认缩放
                    let (scale, tri_start) = if let Some(model_info) = self.model_registry.get(&model_id) {
                        (glam::Vec3::from_slice(&model_info.info.default_scale), model_info.tri_start)
                    } else {
                        (glam::Vec3::ONE, 0)
                    };

                    // 创建预览实例
                    let preview_instance = math::InstanceData {
                        model_matrix: glam::Mat4::from_scale_rotation_translation(
                            scale,
                            glam::Quat::IDENTITY,
                            intersect_pos
                        ).to_cols_array_2d(),
                        model_id,
                        instance_id: 9999, // 标记为预览
                        tri_start,
                        bvh_start: 0,
                        extra_data: [0.0, 0.0],
                        _pad: [0; 10],
                    };

                    // 临时推入显示，下一帧会被 update 覆盖
                    let mut display_instances = self.instances.clone();
                    display_instances.push(preview_instance);
                    self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&display_instances));
                }
            }
        }

        // G/R/S 编辑模式处理
        if let Some(idx) = self.selected_instance {
            match self.edit_mode {
                EditMode::Grab => {
                    // 计算鼠标位移
                    let mouse_delta = glam::Vec2::new(
                        self.last_mouse_pos[0] - self.initial_mouse_pos.x,
                        self.last_mouse_pos[1] - self.initial_mouse_pos.y
                    );

                    // 获取射线与物体所在水平面的交点
                    let (ray_o, ray_dir) = self.camera.get_ray(
                        self.last_mouse_pos[0],
                        self.last_mouse_pos[1],
                        self.config.width,
                        self.config.height
                    );

                    if ray_dir.y.abs() > 0.001 {
                        let t = (self.initial_pos.y - ray_o.y) / ray_dir.y;
                        let new_pos = ray_o + ray_dir * t;

                        // 更新矩阵
                        let new_mat = glam::Mat4::from_scale_rotation_translation(
                            self.initial_scale,
                            self.initial_rot,
                            new_pos
                        );
                        self.instances[idx].model_matrix = new_mat.to_cols_array_2d();
                        
                        // 实时同步 GPU
                        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
                    }
                }
                EditMode::Rotate => {
                    // 计算鼠标旋转量
                    let mouse_delta = glam::Vec2::new(
                        self.last_mouse_pos[0] - self.initial_mouse_pos.x,
                        self.last_mouse_pos[1] - self.initial_mouse_pos.y
                    );

                    // 根据鼠标位移计算旋转角度
                    let rotate_speed = 0.005;
                    let rot_x = mouse_delta.y * rotate_speed;
                    let rot_y = mouse_delta.x * rotate_speed;

                    // 创建旋转四元数
                    let new_rot = glam::Quat::from_euler(
                        glam::EulerRot::XYZ,
                        rot_x,
                        rot_y,
                        0.0
                    ) * self.initial_rot;

                    // 更新矩阵
                    let new_mat = glam::Mat4::from_scale_rotation_translation(
                        self.initial_scale,
                        new_rot,
                        self.initial_pos
                    );
                    self.instances[idx].model_matrix = new_mat.to_cols_array_2d();
                    
                    // 实时同步 GPU
                    self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
                }
                EditMode::Scale => {
                    // 计算鼠标位移（只使用 Y 轴）
                    let mouse_delta = self.last_mouse_pos[1] - self.initial_mouse_pos.y;

                    // 根据鼠标位移计算缩放因子
                    let scale_speed = 0.002;
                    let scale_factor = 1.0 + mouse_delta * scale_speed;
                    
                    // 限制缩放范围
                    let new_scale = self.initial_scale * scale_factor.max(0.1).min(10.0);

                    // 更新矩阵
                    let new_mat = glam::Mat4::from_scale_rotation_translation(
                        new_scale,
                        self.initial_rot,
                        self.initial_pos
                    );
                    self.instances[idx].model_matrix = new_mat.to_cols_array_2d();
                    
                    // 实时同步 GPU
                    self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
                }
                EditMode::None => {}
            }
        }
    }

    fn render(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);

        let mut import_clicked = false;

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

            ui.separator();
            ui.label("几何体列表:");

            ui.separator();
            ui.checkbox(&mut self.show_scaffold, "显示点云 (调试用)");
            ui.checkbox(&mut self.show_depth_debug, "显示深度图 (Depth Map)");
            ui.checkbox(&mut self.show_normal_debug, "显示法线调试 (Normal Debug)");
            ui.separator();
            ui.label("调试模式 (Ap 可视化):");
            ui.add(egui::Slider::new(&mut self.debug_mode, 0..=5).text("debug_mode"));
            let mode_label = match self.debug_mode {
                0 => "0: 正常渲染",
                1 => "1: Ap1 - ID 图 (绿色=有三角面)",
                2 => "2: Ap2 - 位移点 (红色=已映射)",
                3 => "3: Ap3 - 补洞 (蓝色=补出来的)",
                4 => "4: 候选面热力图 (黑=0, 绿=1-2, 黄=3+)",
                5 => "5: 寻根诊断 (红=无候选, 蓝=未相交, 黄=边界外, 绿=成功)",
                _ => "未知模式",
            };
            ui.label(mode_label);
            ui.separator();
            ui.label("扭曲参数 (Ap2/Ap3):");
            ui.add(egui::Slider::new(&mut self.params.distort_strength, 0.0..=1.0).text("强度"));
            ui.add(egui::Slider::new(&mut self.params.distort_frequency, 0.0..=5.0).text("频率"));
            ui.separator();
            if ui.button("📂 导入 GLB").clicked() { import_clicked = true; }

            ui.separator();
            ui.label("命令 (输入 spawn <id>):");
            ui.text_edit_singleline(&mut self.command_input);
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let parts: Vec<&str> = self.command_input.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "spawn" {
                    if let Ok(id) = parts[1].parse::<u32>() {
                        if self.model_registry.contains_key(&id) {
                            self.active_spawn_id = Some(id);
                            println!("进入放置模式，模型 ID: {}", id);
                        } else {
                            println!("模型 ID {} 不存在", id);
                        }
                    }
                }
                self.command_input.clear();
            }

            if self.active_spawn_id.is_some() {
                ui.label(format!("🎯 正在放置模型 ID: {}", self.active_spawn_id.unwrap()));
                if ui.button("取消放置").clicked() {
                    self.active_spawn_id = None;
                }
            }
        });

        // 处理按钮点击事件
        if import_clicked {
            self.import_scaffold();
        }

        let full_output = self.egui_ctx.end_frame();
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        // 核心修复 1：处理 UI 纹理更新 (字体、图标)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        // 构建待渲染的实例列表（包含预览物体）
        let mut instances_to_draw = self.instances.clone();

        if let Some(model_id) = self.active_spawn_id {
            let (ray_o, ray_dir) = self.camera.get_ray(
                self.last_mouse_pos[0],
                self.last_mouse_pos[1],
                self.config.width,
                self.config.height,
            );
            if ray_dir.y.abs() > 0.001 {
                let t = -ray_o.y / ray_dir.y;
                if t > 0.0 {
                    let pos = ray_o + ray_dir * t;
                    let tri_start = self.model_registry.get(&model_id).map(|m| m.tri_start).unwrap_or(0);
                    let preview = math::InstanceData {
                        model_matrix: glam::Mat4::from_translation(pos).to_cols_array_2d(),
                        model_id,
                        instance_id: 9999,
                        tri_start,
                        bvh_start: 0,
                        extra_data: [0.0, 0.0],
                        _pad: [0; 10],
                    };
                    instances_to_draw.push(preview);
                }
            }
        }

        // 核心修复：强制确保有实例数据，并写入 GPU
        if instances_to_draw.is_empty() {
            instances_to_draw.push(math::InstanceData {
                model_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
                model_id: 0,
                instance_id: 0,
                tri_start: 0,
                bvh_start: 0,
                extra_data: [0.0, 0.0],
                _pad: [0; 10],
            });
        }
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances_to_draw));

        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // 执行各个渲染 Pass
        self.clear_warp_buffer(&mut encoder);
        self.run_depth_pass(&mut encoder, &instances_to_draw);
        self.run_compute_pass(&mut encoder);
        self.run_ap2_pass(&mut encoder);
        self.run_ap3_pass(&mut encoder);
        // self.run_forward_pass(&mut encoder, &self.output_texture_view, &instances_to_draw);
        self.run_draw_pass(&mut encoder, &view);
        self.run_scaffold_pass(&mut encoder, &view);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        // 更新 egui 缓冲区
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_descriptor);
        self.run_ui_pass(&mut encoder, &view, &paint_jobs, &screen_descriptor);

        // 处理纹理释放
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
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.tri_id_texture_view = self.tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.tri_id_texture_view_for_render = self.tri_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.warp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Warp Buffer"),
                size: pixel_buffer_size::<math::WarpPixel>(new_size.width, new_size.height),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.sdf_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SDF Buffer"),
                size: (new_size.width * new_size.height * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // 重新创建 UV G-Buffer 纹理
            self.uv_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("UV G-Buffer"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.uv_texture_view = self.uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.uv_texture_view_for_render = self.uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 重新创建模型 UV 纹理
            self.model_uv_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Model UV Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.model_uv_texture_view = self.model_uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.model_uv_texture_view_for_render = self.model_uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 重新创建世界法线纹理
            self.normal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Normal Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.normal_texture_view = self.normal_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.normal_texture_view_for_render = self.normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // 4. 关键修复：重新创建 BindGroup，否则它们引用的还是旧视图
            let (compute, render, depth, depth_blit) = Self::create_all_bind_groups(
                &self.device,
                &self.compute_bind_group_layout,
                &self.render_bind_group_layout,
                &self.depth_bind_group_layout,
                &self.depth_blit_bind_group_layout,
                &self.output_texture_view,
                &self.params_buffer,
                &self.triangle_buffer,
                &self.scaffold_buffer,
                &self.depth_texture_view,
                &self.tri_id_texture_view,
                &self.uv_texture_view,
                &self.model_uv_texture_view,
                &self.normal_texture_view,
                &self.albedo_texture_view,
                &self.albedo_sampler,
                &self.instance_buffer,
                &self.warp_buffer,
                &self.sdf_buffer,
            );
            self.compute_bind_group = compute;
            self.render_bind_group = render;
            self.depth_bind_group = depth;
            self.depth_blit_bind_group = depth_blit;
        }
    }

    // 在 Rust 里实现和 Shader 完全一致的 smin 基座采样
    // 打开文件对话框并导入 GLB
    fn import_scaffold(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GLB Files", &["glb"])
            .add_filter("GLTF Files", &["gltf"])
            .pick_file() {
            
            // 保存路径供后续烘焙使用
            self.scaffold_path = Some(path.to_str().unwrap().to_string());
            
            // 加载三角形、顶点和贴图
            let (document, buffers, images) = gltf::import(&path).unwrap();
            
            let mut triangles = Vec::new();
            let mut scaffold_vertices = Vec::new(); // 存储顶点位置和法线
            
            for mesh in document.meshes() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|b| Some(&buffers[b.index()]));
                    
                    let pos_iter: Vec<[f32;3]> = reader.read_positions().unwrap().collect();
                    let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                    
                    // 读取 UV 坐标（如果存在）
                    let uv_iter: Vec<[f32; 2]> = reader.read_tex_coords(0)
                        .map(|it| it.into_f32().collect())
                        .unwrap_or_else(|| vec![[0.0, 0.0]; pos_iter.len()]);
                    
                    // 读取法线（如果存在，不存在则计算）
                    let normal_iter: Vec<[f32; 3]> = reader.read_normals()
                        .map(|it| it.collect())
                        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; pos_iter.len()]);
                    
                    for chunk in indices.chunks_exact(3) {
                        let i0 = chunk[0] as usize;
                        let i1 = chunk[1] as usize;
                        let i2 = chunk[2] as usize;
                        
                        let p0 = pos_iter[i0];
                        let p1 = pos_iter[i1];
                        let p2 = pos_iter[i2];
                        
                        let uv0 = uv_iter[i0];
                        let uv1 = uv_iter[i1];
                        let uv2 = uv_iter[i2];
                        
                        let n0 = normal_iter[i0];
                        let n1 = normal_iter[i1];
                        let n2 = normal_iter[i2];
                        
                        triangles.push(math::Triangle {
                            v0: [p0[0], p0[1], p0[2], 1.0],
                            v1: [p1[0], p1[1], p1[2], 1.0],
                            v2: [p2[0], p2[1], p2[2], 1.0],
                            n0: [n0[0], n0[1], n0[2], 0.0],
                            n1: [n1[0], n1[1], n1[2], 0.0],
                            n2: [n2[0], n2[1], n2[2], 0.0],
                            uv01: [uv0[0], uv0[1], uv1[0], uv1[1]],
                            uv2: [uv2[0], uv2[1], 0.0, 0.0],
                        });
                    }
                    
                    // 收集所有顶点（位置 + 法线）用于点云显示
                    // 格式：[pos.x, pos.y, pos.z, packed_normal]
                    for (i, p) in pos_iter.iter().enumerate() {
                        let n = normal_iter[i];
                        // 将法线打包到 w 分量：将 [-1,1] 映射到 [0,1] 然后乘以 255 转为整数
                        let packed_normal = ((n[0] * 0.5 + 0.5) * 65535.0).round() as u16;
                        let packed_normal_high = ((n[1] * 0.5 + 0.5) * 65535.0).round() as u16;
                        // 简单处理：只存储 x 分量用于背面剔除判断
                        let packed = ((n[0] * 0.5 + 0.5) * 255.0).round() as f32 / 255.0;
                        scaffold_vertices.push(glam::Vec4::new(p[0], p[1], p[2], packed));
                    }
                }
            }
            
            self.scaffold_vertices.clear(); // 保持向后兼容，不再使用
            self.triangles = triangles;
            
            // 统计唯一顶点数
            let (document, buffers, _) = gltf::import(&path).unwrap();
            let mut unique_vertices: HashMap<VertexKey, ()> = HashMap::new();
            for mesh in document.meshes() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|b| Some(&buffers[b.index()]));
                    let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                    let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                    for chunk in indices.chunks_exact(3) {
                        for &v_idx in chunk {
                            let p = positions[v_idx as usize];
                            unique_vertices.insert(VertexKey::from_pos(p), ());
                        }
                    }
                }
            }
            
            println!("三角形: {}, 原始顶点数: {}, 合并后顶点数: {}", 
                self.triangles.len(),
                scaffold_vertices.len(),
                unique_vertices.len());
            
            // ========== 动态缓冲区扩容检查 ==========
            let triangle_size = (self.triangles.len() * std::mem::size_of::<math::Triangle>()) as u64;
            let scaffold_size = (scaffold_vertices.len() * std::mem::size_of::<glam::Vec4>()) as u64;
            
            let triangle_resized = Self::ensure_buffer_size(
                &self.device,
                &mut self.triangle_buffer, 
                triangle_size, 
                "Triangle Buffer", 
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            );
            
            let scaffold_resized = Self::ensure_buffer_size(
                &self.device,
                &mut self.scaffold_buffer, 
                scaffold_size, 
                "Scaffold Buffer", 
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            );
            
            // ========== 更新模型参数 ==========
            // 1. 计算所有顶点的中心点
            let mut center = glam::Vec3::ZERO;
            for tri in &self.triangles {
                center += Vec3::new(tri.v0[0], tri.v0[1], tri.v0[2]);
                center += Vec3::new(tri.v1[0], tri.v1[1], tri.v1[2]);
                center += Vec3::new(tri.v2[0], tri.v2[1], tri.v2[2]);
            }
            center /= (self.triangles.len() * 3) as f32;
            
            // 更新模型中心参数
            self.params.model_center = center.extend(1.0).to_array();
            // 设置默认基础颜色为灰色
            self.params.base_color = [0.8, 0.8, 0.8, 1.0];
            
            // 计算包围球半径
            let mut max_radius: f32 = 0.0;
            for tri in &self.triangles {
                let p0 = Vec3::new(tri.v0[0], tri.v0[1], tri.v0[2]);
                let p1 = Vec3::new(tri.v1[0], tri.v1[1], tri.v1[2]);
                let p2 = Vec3::new(tri.v2[0], tri.v2[1], tri.v2[2]);
                let dist0 = (p0 - center).length();
                let dist1 = (p1 - center).length();
                let dist2 = (p2 - center).length();
                max_radius = max_radius.max(dist0).max(dist1).max(dist2);
            }
            self.params.base_radius = max_radius * 1.1; // 增加 10% 余量
            
            println!("模型中心: {:?}, 包围球半径: {}", center, self.params.base_radius);
            
            // 上传到 GPU
            self.queue.write_buffer(&self.triangle_buffer, 0, bytemuck::cast_slice(&self.triangles));
            
            // 上传顶点数据（包含法线信息）到 GPU
            // 格式：[pos.x, pos.y, pos.z, packed_normal]
            self.queue.write_buffer(&self.scaffold_buffer, 0, bytemuck::cast_slice(&scaffold_vertices));
            
            // 更新参数 - 使用 anchor_count 存储三角形数量
            self.params.scaffold_count = scaffold_vertices.len() as u32;
            self.params.anchor_count = self.triangles.len() as u32;
            
            // 【关键修复】：立即将更新后的参数写入 GPU
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
            
            println!("脚手架上传成功：{} 个点，{} 个三角形", self.params.scaffold_count, self.params.anchor_count);
            
            // ========== 加载 GLB 贴图并更新 BindGroup ==========
            for image in images {
                // 获取贴图数据
                let width = image.width;
                let height = image.height;
                let pixels = image.pixels;
                
                // 检查通道数：GLB 贴图可能是 RGB 或 RGBA
                let bytes_per_pixel = pixels.len() / (width * height) as usize;
                println!("贴图原始格式：{}x{}，通道数：{}", width, height, bytes_per_pixel);
                
                // 如果是 RGB（3 通道），转换为 RGBA（4 通道）
                let rgba_pixels: Vec<u8> = if bytes_per_pixel == 3 {
                    pixels.chunks(3)
                        .flat_map(|rgb| vec![rgb[0], rgb[1], rgb[2], 255u8])
                        .collect()
                } else {
                    pixels
                };
                
                // 创建新的 albedo 纹理
                let albedo_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Loaded Albedo Texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb, // GLB 贴图通常是 sRGB
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                
                // 写入纹理数据
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &albedo_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba_pixels,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                
                self.albedo_texture = albedo_texture;
                self.albedo_texture_view = self.albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
                
                println!("加载贴图成功：{}x{} (已转换为 RGBA)", width, height);
                break; // 只加载第一个贴图
            }
            
            // ========== 使用统一方法重新创建 BindGroup ==========
            let (compute, render, depth, depth_blit) = Self::create_all_bind_groups(
                &self.device,
                &self.compute_bind_group_layout,
                &self.render_bind_group_layout,
                &self.depth_bind_group_layout,
                &self.depth_blit_bind_group_layout,
                &self.output_texture_view,
                &self.params_buffer,
                &self.triangle_buffer,
                &self.scaffold_buffer,
                &self.depth_texture_view,
                &self.tri_id_texture_view,
                &self.uv_texture_view,
                &self.model_uv_texture_view,
                &self.normal_texture_view,
                &self.albedo_texture_view,
                &self.albedo_sampler,
                &self.instance_buffer,
                &self.warp_buffer,
                &self.sdf_buffer,
            );
            self.compute_bind_group = compute;
            self.render_bind_group = render;
            self.depth_bind_group = depth;
            self.depth_blit_bind_group = depth_blit;

            // ========== 【关键修复】：更新模型注册表 ==========
            // 当手动导入 GLB 时，必须同步更新 model_registry 和 material_bind_groups
            let imported_tri_count = self.triangles.len() as u32;
            
            // 更新或插入模型注册表条目（使用 ID 0 作为导入模型的 ID）
            self.model_registry.insert(0, math::ModelRegistryItem {
                info: math::ModelManifestItem {
                    id: 0,
                    name: "Imported".to_string(),
                    file: self.scaffold_path.clone().unwrap_or_default(),
                    default_scale: [1.0, 1.0, 1.0],
                },
                tri_start: 0,
                tri_count: imported_tri_count,
                material_id: 0,
            });
            
            // 更新材质绑定组（使用刚创建的 albedo_texture_view）
            let imported_mat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Imported Material Bind Group"),
                layout: &self.material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.albedo_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.albedo_sampler),
                    },
                ],
            });
            self.material_bind_groups.insert(0, imported_mat_bg);

            // 【关键修复】：导入模型后自动创建一个实例
            let tri_start = self.model_registry.get(&0).map(|m| m.tri_start).unwrap_or(0);
            let first_instance = math::InstanceData {
                model_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
                model_id: 0,
                instance_id: 0,
                tri_start,
                bvh_start: 0,
                extra_data: [0.0, 0.0],
                _pad: [0; 10],
            };
            self.instances.clear();
            self.instances.push(first_instance);
            self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
            println!("已创建默认实例，实例数量：{}", self.instances.len());
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
                            winit::keyboard::Key::Character(c) if c == "d" => {
                                app.is_d_pressed = kb_event.state == winit::event::ElementState::Pressed;
                            }
                            // S 键：优先检查编辑模式
                            winit::keyboard::Key::Character(c) if c == "s" => {
                                if kb_event.state == winit::event::ElementState::Pressed {
                                    // 如果有物体选中，S 键触发缩放模式
                                    if app.selected_instance.is_some() {
                                        app.edit_mode = EditMode::Scale;
                                        app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                    } else {
                                        // 否则触发相机后退
                                        app.is_s_pressed = true;
                                    }
                                } else {
                                    // 松开 S 键时，关闭相机后退
                                    if app.edit_mode == EditMode::None {
                                        app.is_s_pressed = false;
                                    }
                                }
                            }
                            // G/R 编辑模式快捷键
                            winit::keyboard::Key::Character(c) if c == "g" => {
                                if kb_event.state == winit::event::ElementState::Pressed {
                                    app.edit_mode = EditMode::Grab;
                                    app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                }
                            }
                            winit::keyboard::Key::Character(c) if c == "r" => {
                                if kb_event.state == winit::event::ElementState::Pressed {
                                    app.edit_mode = EditMode::Rotate;
                                    app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                }
                            }
                            // Esc 退出编辑模式
                            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                                if kb_event.state == winit::event::ElementState::Pressed {
                                    app.edit_mode = EditMode::None;
                                }
                            }
                            _ => {}
                        }
                    }

                    // --- 鼠标点击监听：中键 ---
                    WindowEvent::MouseInput { button, state, .. } => {
                        if *button == winit::event::MouseButton::Middle {
                            app.is_mmb_pressed = *state == winit::event::ElementState::Pressed;
                        }
                        if *button == winit::event::MouseButton::Left {
                            app.is_lmb_pressed = *state == winit::event::ElementState::Pressed;
                            
                            // 左键按下时，如果不是在放置模式且不在编辑模式，则尝试拾取实例
                            if *state == winit::event::ElementState::Pressed && app.active_spawn_id.is_none() && app.edit_mode == EditMode::None {
                                app.pick_instance(app.last_mouse_pos);
                            }
                            
                            // 编辑模式下，左键按下表示确认变换
                            if *state == winit::event::ElementState::Pressed && app.edit_mode != EditMode::None {
                                app.edit_mode = EditMode::None;
                                println!("编辑模式已退出");
                            }
                            
                            // 左键按下时处理实例放置
                            if *state == winit::event::ElementState::Pressed && app.active_spawn_id.is_some() {
                                let model_id = app.active_spawn_id.unwrap();
                                let (ray_o, ray_dir) = app.camera.get_ray(
                                    app.last_mouse_pos[0],
                                    app.last_mouse_pos[1],
                                    app.config.width,
                                    app.config.height
                                );
                                
                                // 射线与地平面 (Y=0) 求交
                                if ray_dir.y.abs() > 0.001 {
                                    let t = -ray_o.y / ray_dir.y;
                                    if t > 0.0 {
                                        let intersect_pos = ray_o + ray_dir * t;
                                        
                                        if let Some(model_info) = app.model_registry.get(&model_id) {
                                            let scale = model_info.info.default_scale;
                                            
                                            let new_instance = math::InstanceData {
                                                model_matrix: glam::Mat4::from_scale_rotation_translation(
                                                    glam::Vec3::from_slice(&scale),
                                                    glam::Quat::IDENTITY,
                                                    intersect_pos
                                                ).to_cols_array_2d(),
                                                model_id,
                                                instance_id: app.instances.len() as u32,
                                                tri_start: model_info.tri_start,
                                                bvh_start: 0,
                                                extra_data: [0.0, 0.0],
                                                _pad: [0; 10],
                                            };
                                            
                                            app.instances.push(new_instance);
                                            app.queue.write_buffer(&app.instance_buffer, 0, bytemuck::cast_slice(&app.instances));
                                            println!("放置实例: 模型 ID {} 在位置 {:?}", model_id, intersect_pos);
                                            app.active_spawn_id = None; // 退出放置模式
                                        }
                                    }
                                }
                            }
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



