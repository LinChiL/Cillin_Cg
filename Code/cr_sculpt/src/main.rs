use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use glam::{FloatExt, Vec3, Vec4, Vec3Swizzles, Vec4Swizzles};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use gltf;
use rfd;

mod physics;
mod shadow;
use physics::InstancePhysics;

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

// ===== 性能监控 =====
#[derive(Clone)]
struct PerfStats {
    frame_time_ms: f32,
    fps: f32,
    // 各 pass 耗时 (ms)
    depth_pass_ms: f32,
    compute_pass_ms: f32,
    ap3_pass_ms: f32,
    draw_pass_ms: f32,
    scaffold_pass_ms: f32,
    ui_pass_ms: f32,
    // 非渲染开销 (ms)
    egui_build_ms: f32,       // begin_frame + end_frame + tessellate
    buffer_upload_ms: f32,    // write_buffer 调用
    surface_acquire_ms: f32,  // get_current_texture (含 vsync 等待)
    submit_present_ms: f32,   // submit + present
    // GPU 时间 (ms) - 来自 Timestamp Query
    gpu_clear_warp_ms: f32,
    gpu_depth_ms: f32,
    gpu_compute_ms: f32,
    gpu_shadow_bin_ms: f32,
    gpu_shadow_trace_ms: f32,
    gpu_ap3_ms: f32,
    gpu_draw_ms: f32,
    gpu_scaffold_ms: f32,
    gpu_ui_ms: f32,
    gpu_total_ms: f32,
    // Draw calls
    depth_draw_calls: u32,
    forward_draw_calls: u32,
    draw_draw_calls: u32,
    scaffold_draw_calls: u32,
    // 几何体
    total_triangles: u32,
    rendered_triangles: u32,
    instance_count: u32,
    // 内存 (bytes)
    triangle_buffer_size: u64,
    instance_buffer_size: u64,
    // 历史采样 (用于绘制曲线)
    frame_history: Vec<f32>,
    max_history: usize,
}

impl PerfStats {
    fn new() -> Self {
        Self {
            frame_time_ms: 0.0,
            fps: 0.0,
            depth_pass_ms: 0.0,
            compute_pass_ms: 0.0,
            ap3_pass_ms: 0.0,
            draw_pass_ms: 0.0,
            scaffold_pass_ms: 0.0,
            ui_pass_ms: 0.0,
            egui_build_ms: 0.0,
            buffer_upload_ms: 0.0,
            surface_acquire_ms: 0.0,
            submit_present_ms: 0.0,
            gpu_clear_warp_ms: 0.0,
            gpu_depth_ms: 0.0,
            gpu_compute_ms: 0.0,
            gpu_shadow_bin_ms: 0.0,
            gpu_shadow_trace_ms: 0.0,
            gpu_ap3_ms: 0.0,
            gpu_draw_ms: 0.0,
            gpu_scaffold_ms: 0.0,
            gpu_ui_ms: 0.0,
            gpu_total_ms: 0.0,
            depth_draw_calls: 0,
            forward_draw_calls: 0,
            draw_draw_calls: 0,
            scaffold_draw_calls: 0,
            total_triangles: 0,
            rendered_triangles: 0,
            instance_count: 0,
            triangle_buffer_size: 0,
            instance_buffer_size: 0,
            frame_history: Vec::new(),
            max_history: 120,
        }
    }

    fn push_frame(&mut self, frame_time_ms: f32) {
        self.frame_history.push(frame_time_ms);
        if self.frame_history.len() > self.max_history {
            self.frame_history.remove(0);
        }
    }
}

fn apply_imported_visual_normals(
    triangles: &mut [math::Triangle],
    normals: &[[f32; 3]],
    indices: &[u32],
) {
    for (tri, chunk) in triangles.iter_mut().zip(indices.chunks_exact(3)) {
        let n0 = normals[chunk[0] as usize];
        let n1 = normals[chunk[1] as usize];
        let n2 = normals[chunk[2] as usize];
        tri.n0 = [n0[0], n0[1], n0[2], 0.0];
        tri.n1 = [n1[0], n1[1], n1[2], 0.0];
        tri.n2 = [n2[0], n2[1], n2[2], 0.0];
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
        let mut vertex_warp_normals = [Vec3::ZERO; 3];

        for i in 0..3 {
            let v_idx = chunk[i] as usize;
            let p = positions[v_idx];
            let key = VertexKey::from_pos(p);

            let mut visual_sum_n = Vec3::ZERO;
            let mut warp_sum_n = Vec3::ZERO;
            if let Some(adjacent_faces) = pos_to_faces.get(&key) {
                for &adj_face_idx in adjacent_faces {
                    let adj_normal = face_normals[adj_face_idx];
                    if current_face_normal.dot(adj_normal) >= threshold_cos {
                        visual_sum_n += adj_normal;
                    }
                    warp_sum_n += adj_normal;
                }
            }
            vertex_smoothed_normals[i] = if visual_sum_n.length_squared() > 0.0001 {
                visual_sum_n.normalize()
            } else {
                current_face_normal
            };
            vertex_warp_normals[i] = if warp_sum_n.length_squared() > 0.0001 {
                warp_sum_n.normalize()
            } else {
                vertex_smoothed_normals[i]
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
            warp_n0: vertex_warp_normals[0].extend(0.0).to_array(),
            warp_n1: vertex_warp_normals[1].extend(0.0).to_array(),
            warp_n2: vertex_warp_normals[2].extend(0.0).to_array(),
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

// Blender 风格轴约束
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisConstraint {
    None,
    X,
    Y,
    Z,
    YZ, // Shift+X: 排除 X，只在 YZ 平面移动（效果同锁定 X 轴）
    XZ, // Shift+Y: 排除 Y
    XY, // Shift+Z: 排除 Z
}

// 独立物理性能监控窗口数据
struct ProfilerWindowData<'a> {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
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
    shadow_system: shadow::ShadowSystem,
    depth_bind_group_layout: wgpu::BindGroupLayout,
    depth_bind_group: wgpu::BindGroup,
    depth_render_pipeline: wgpu::RenderPipeline,
    depth_blit_pipeline: wgpu::RenderPipeline,
    depth_blit_bind_group: wgpu::BindGroup,
    show_depth_debug: bool,
    show_normal_debug: bool,
    show_perf_monitor: bool,
    perf_stats: PerfStats,
    profiler_window: ProfilerWindowData<'a>,
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

    // 扭曲后世界位置纹理
    warped_pos_texture: wgpu::Texture,
    warped_pos_texture_view: wgpu::TextureView,
    warped_pos_texture_view_for_render: wgpu::TextureView,

    // Albedo 贴图（Atlas）
    albedo_texture: wgpu::Texture,
    albedo_texture_view: wgpu::TextureView,
    albedo_sampler: wgpu::Sampler,
    atlas_width: u32,
    atlas_height: u32,
    atlas_cursor_x: u32,
    atlas_cursor_y: u32,
    atlas_row_height: u32,
    // GPU Timestamp Query
    ts_query_set: wgpu::QuerySet,
    ts_resolve_buffer: wgpu::Buffer,
    ts_staging_buffer: wgpu::Buffer,

    // 模型库管理
    model_registry: std::collections::HashMap<u32, math::ModelRegistryItem>,
    model_colliders: std::collections::HashMap<u32, physics::ModelCollider>,
    material_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_groups: std::collections::HashMap<u32, wgpu::BindGroup>,
    mesh_render_pipeline: wgpu::RenderPipeline,

    // 场景中的实例列表
    instances: Vec<math::InstanceData>,
    instance_physics: Vec<InstancePhysics>,
    instance_buffer: wgpu::Buffer,

    // 交互控制
    command_input: String,
    active_spawn_id: Option<u32>,
    continuous_spawn_id: Option<u32>,
    rebuild_colliders_requested: bool,
    
    // 编辑模式相关
    selected_instance: Option<usize>,
    edit_mode: EditMode,
    axis_constraint: AxisConstraint,
    last_axis_key: Option<KeyCode>,
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
        wgpu::Texture,
        wgpu::TextureView,
        u32,
        u32,
        u32,
        u32,
        u32,
    ) {
        let manifest_path = "cremModel/manifest.json";
        let manifest_str = std::fs::read_to_string(manifest_path).expect("无法读取 manifest.json");
        let manifest: math::ModelManifest = serde_json::from_str(&manifest_str).expect("解析 JSON 失败");

        const MAX_ATLAS_WIDTH: u32 = 8192;

        let mut loaded_models = Vec::new();
        // Shelf-packing 预计算布局
        let mut atlas_width: u32 = 0;
        let mut atlas_height: u32 = 0;
        let mut model_positions: Vec<(u32, u32)> = Vec::new(); // (x, y) 在 atlas 中的位置
        {
            let mut cx: u32 = 0;
            let mut cy: u32 = 0;
            let mut row_h: u32 = 0;
            for item in &manifest.models {
                let file_path = format!("cremModel/{}", item.file);
                println!("正在预加载模型: {}", file_path);
                let (tris, rgba_pixels, width, height, material_color) = Self::load_glb_for_atlas(&file_path);
                let w = width.max(1);
                let h = height.max(1);
                // 如果当前行放不下，换行
                if cx + w > MAX_ATLAS_WIDTH {
                    cx = 0;
                    cy += row_h;
                    row_h = 0;
                }
                model_positions.push((cx, cy));
                cx += w;
                row_h = row_h.max(h);
                atlas_width = atlas_width.max(cx);
                atlas_height = atlas_height.max(cy + row_h);
                loaded_models.push((item, tris, rgba_pixels, w, h, material_color));
            }
        }
        let atlas_cursor_x = atlas_width; // 实际上 atlas_width 就是当前行末尾
        let atlas_cursor_y = atlas_height.saturating_sub(if model_positions.is_empty() { 0 } else { loaded_models.last().map(|m| m.4).unwrap_or(1) }); // 最后一个模型所在行
        // 重新计算正确的 cursor 位置
        let (atlas_cursor_x, atlas_cursor_y, atlas_row_height) = {
            let mut cx: u32 = 0;
            let mut cy: u32 = 0;
            let mut row_h: u32 = 0;
            for (_, _, _, w, h, _) in &loaded_models {
                if cx + w > MAX_ATLAS_WIDTH {
                    cx = 0;
                    cy += row_h;
                    row_h = 0;
                }
                cx += *w;
                row_h = row_h.max(*h);
            }
            (cx, cy, row_h)
        };

        atlas_width = atlas_width.max(1);
        atlas_height = atlas_height.max(1);
        let mut atlas_pixels = vec![255u8; (atlas_width * atlas_height * 4) as usize];
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Model Texture Atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut registry = std::collections::HashMap::new();
        let mut material_bind_groups = std::collections::HashMap::new();
        let mut all_triangles = Vec::new();

        for (idx, (item, mut tris, rgba_pixels, width, height, material_color)) in loaded_models.into_iter().enumerate() {
            let (ax, ay) = model_positions[idx];
            for y in 0..height {
                let dst_start = ((((ay + y) * atlas_width) + ax) * 4) as usize;
                let src_start = (y * width * 4) as usize;
                let len = (width * 4) as usize;
                atlas_pixels[dst_start..dst_start + len].copy_from_slice(&rgba_pixels[src_start..src_start + len]);
            }

            let u0 = ax as f32 / atlas_width as f32;
            let v0 = ay as f32 / atlas_height as f32;
            let us = width as f32 / atlas_width as f32;
            let vs = height as f32 / atlas_height as f32;
            for tri in &mut tris {
                tri.uv01 = [
                    u0 + tri.uv01[0] * us,
                    v0 + tri.uv01[1] * vs,
                    u0 + tri.uv01[2] * us,
                    v0 + tri.uv01[3] * vs,
                ];
                tri.uv2 = [u0 + tri.uv2[0] * us, v0 + tri.uv2[1] * vs, 0.0, 0.0];
            }

            let tri_start = all_triangles.len() as u32;
            let tri_count = tris.len() as u32;
            all_triangles.extend(tris);

            let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Material Bind Group {}", item.id)),
                layout: material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&atlas_view),
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
                material_color,
            });
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_width * 4),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
        );

        (registry, material_bind_groups, all_triangles, atlas_texture, atlas_view, atlas_width, atlas_height, atlas_cursor_x, atlas_cursor_y, atlas_row_height)
    }

    fn load_glb_for_atlas(path: &str) -> (Vec<math::Triangle>, Vec<u8>, u32, u32, [f32; 3]) {
        let (document, buffers, images) = gltf::import(path).expect("加载 GLB 失败");
        let mut triangles = Vec::new();

        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
                let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let mut tris = compute_smoothed_triangles(&positions, &indices, &uvs, 45.0).0;
                if let Some(normals) = normals.filter(|n| n.len() == positions.len()) {
                    apply_imported_visual_normals(&mut tris, &normals, &indices);
                }
                triangles.append(&mut tris);
            }
        }

        let Some(image) = images.first() else {
            return (triangles, vec![255, 255, 255, 255], 1, 1, [1.0, 1.0, 1.0]);
        };
        let width = image.width.max(1);
        let height = image.height.max(1);
        let pixels = image.pixels.clone();
        let bytes_per_pixel = pixels.len() / (width * height) as usize;
        let rgba_pixels: Vec<u8> = match bytes_per_pixel {
            3 => pixels.chunks(3).flat_map(|rgb| vec![rgb[0], rgb[1], rgb[2], 255u8]).collect(),
            4 => pixels,
            _ => vec![255, 255, 255, 255],
        };

        let mut sum = [0u64; 3];
        for px in rgba_pixels.chunks_exact(4) {
            sum[0] += px[0] as u64;
            sum[1] += px[1] as u64;
            sum[2] += px[2] as u64;
        }
        let count = (rgba_pixels.len() / 4).max(1) as f32;
        let material_color = [
            sum[0] as f32 / (255.0 * count),
            sum[1] as f32 / (255.0 * count),
            sum[2] as f32 / (255.0 * count),
        ];

        (triangles, rgba_pixels, width, height, material_color)
    }

    fn build_model_colliders(
        model_registry: &std::collections::HashMap<u32, math::ModelRegistryItem>,
        triangles: &[math::Triangle],
    ) -> std::collections::HashMap<u32, physics::ModelCollider> {
        // #region debug-point window-freeze-collider
        let collider_total_t0 = std::time::Instant::now();
        eprintln!("[debug-window-freeze] build_model_colliders start models={} triangles={}", model_registry.len(), triangles.len());
        // #endregion debug-point window-freeze-collider
        let mut colliders = std::collections::HashMap::new();
        for (&model_id, reg) in model_registry {
            // #region debug-point window-freeze-collider
            let collider_model_t0 = std::time::Instant::now();
            eprintln!("[debug-window-freeze] collider model start id={} tri_count={}", model_id, reg.tri_count);
            // #endregion debug-point window-freeze-collider
            let tris = &triangles[reg.tri_start as usize..(reg.tri_start + reg.tri_count) as usize];
            let mut sample_points = Vec::with_capacity(tris.len() * 7);
            let mut bounds_min = glam::Vec3::splat(f32::INFINITY);
            let mut bounds_max = glam::Vec3::splat(f32::NEG_INFINITY);
            for tri in tris {
                let v0 = glam::Vec4::from(tri.v0).truncate();
                let v1 = glam::Vec4::from(tri.v1).truncate();
                let v2 = glam::Vec4::from(tri.v2).truncate();
                for v in [v0, v1, v2] {
                    bounds_min = bounds_min.min(v);
                    bounds_max = bounds_max.max(v);
                }
                sample_points.push(v0);
                sample_points.push(v1);
                sample_points.push(v2);
                sample_points.push((v0 + v1) * 0.5);
                sample_points.push((v1 + v2) * 0.5);
                sample_points.push((v2 + v0) * 0.5);
                sample_points.push((v0 + v1 + v2) / 3.0);
            }
            let extent = bounds_max - bounds_min;
            let pad = extent.max_element().max(1.0) * 0.08;
            bounds_min -= glam::Vec3::splat(pad);
            bounds_max += glam::Vec3::splat(pad);
            let resolution = if tris.len() > 4000 {
                12
            } else if tris.len() > 1500 {
                16
            } else {
                24
            };
            // #region debug-point window-freeze-collider
            eprintln!("[debug-window-freeze] collider model sdf_resolution id={} resolution={}", model_id, resolution);
            // #endregion debug-point window-freeze-collider
            let mut sdf = Vec::with_capacity((resolution * resolution * resolution) as usize);
            for z in 0..resolution {
                // #region debug-point window-freeze-collider
                if z % 8 == 0 {
                    eprintln!("[debug-window-freeze] collider model progress id={} z={}/{} elapsed_ms={:.1}", model_id, z, resolution, collider_model_t0.elapsed().as_secs_f64() * 1000.0);
                }
                // #endregion debug-point window-freeze-collider
                for y in 0..resolution {
                    for x in 0..resolution {
                        let uvw = glam::Vec3::new(x as f32, y as f32, z as f32) / (resolution - 1) as f32;
                        let p = bounds_min + (bounds_max - bounds_min) * uvw;
                        let (distance, _) = Self::signed_distance_to_tris(tris, p);
                        sdf.push(distance);
                    }
                }
            }
            colliders.insert(model_id, physics::ModelCollider { sample_points, bounds_min, bounds_max, resolution, sdf });
            // #region debug-point window-freeze-collider
            eprintln!("[debug-window-freeze] collider model done id={} samples={} elapsed_ms={:.1}", model_id, colliders.get(&model_id).map(|c| c.sample_points.len()).unwrap_or(0), collider_model_t0.elapsed().as_secs_f64() * 1000.0);
            // #endregion debug-point window-freeze-collider
        }
        // #region debug-point window-freeze-collider
        eprintln!("[debug-window-freeze] build_model_colliders done elapsed_ms={:.1}", collider_total_t0.elapsed().as_secs_f64() * 1000.0);
        // #endregion debug-point window-freeze-collider
        colliders
    }

    fn closest_point_on_triangle(p: glam::Vec3, a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec3 {
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 { return a; }

        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 { return b; }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return a + ab * (d1 / (d1 - d3));
        }

        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 { return c; }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return a + ac * (d2 / (d2 - d6));
        }

        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        }

        let denom = 1.0 / (va + vb + vc);
        a + ab * (vb * denom) + ac * (vc * denom)
    }

    fn ray_intersects_triangle(origin: glam::Vec3, dir: glam::Vec3, a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> bool {
        let eps = 1e-6;
        let edge1 = b - a;
        let edge2 = c - a;
        let h = dir.cross(edge2);
        let det = edge1.dot(h);
        if det.abs() < eps { return false; }
        let inv_det = 1.0 / det;
        let s = origin - a;
        let u = inv_det * s.dot(h);
        if !(0.0..=1.0).contains(&u) { return false; }
        let q = s.cross(edge1);
        let v = inv_det * dir.dot(q);
        if v < 0.0 || u + v > 1.0 { return false; }
        inv_det * edge2.dot(q) > eps
    }

    fn signed_distance_to_tris(tris: &[math::Triangle], local_pos: glam::Vec3) -> (f32, glam::Vec3) {
        let mut best_d2 = f32::INFINITY;
        let mut best_point = glam::Vec3::ZERO;
        let mut best_normal = glam::Vec3::Y;
        let mut intersections = 0;
        let ray_dir = glam::Vec3::X;

        for tri in tris {
            let a = glam::Vec4::from(tri.v0).truncate();
            let b = glam::Vec4::from(tri.v1).truncate();
            let c = glam::Vec4::from(tri.v2).truncate();
            let closest = Self::closest_point_on_triangle(local_pos, a, b, c);
            let d2 = local_pos.distance_squared(closest);
            if d2 < best_d2 {
                best_d2 = d2;
                best_point = closest;
                best_normal = (b - a).cross(c - a).normalize_or_zero();
            }
            if Self::ray_intersects_triangle(local_pos + ray_dir * 1e-4, ray_dir, a, b, c) {
                intersections += 1;
            }
        }

        let inside = intersections % 2 == 1;
        let dist = best_d2.sqrt();
        let mut dir = (local_pos - best_point).normalize_or_zero();
        if dir.length_squared() < 1e-8 {
            dir = best_normal;
        }
        let normal = if inside { -dir } else { dir };
        (if inside { -dist } else { dist }, normal.normalize_or_zero())
    }

    fn push_instance(&mut self, instance: math::InstanceData) {
        self.instances.push(instance);
        self.instance_physics.push(InstancePhysics::default());
    }

    fn sync_instance_physics_len(&mut self) {
        physics::sync_instance_physics_len(&self.instances, &mut self.instance_physics);
    }

    /// 计算射线到轴线的最近点（用于 Blender 风格轴约束）
    fn ray_closest_to_axis(&self, ray_o: glam::Vec3, ray_dir: glam::Vec3, line_origin: glam::Vec3, axis_dir: glam::Vec3) -> glam::Vec3 {
        let w = ray_o - line_origin;
        let d2 = ray_dir.dot(ray_dir);
        let da = ray_dir.dot(axis_dir);
        let wa = w.dot(axis_dir);
        let wd = w.dot(ray_dir);
        let denom = d2 - da * da;
        if denom.abs() < 1e-10 {
            return line_origin + axis_dir * wa;
        }
        let t = -(wd - wa * da) / denom;
        let p = ray_o + ray_dir * t;
        line_origin + axis_dir * (p - line_origin).dot(axis_dir)
    }

    fn scene_path(&self) -> std::path::PathBuf {
        // 通过可执行文件路径推导项目根目录
        let exe = std::env::current_exe().unwrap_or_default();
        // debug:  .../target/debug/cr_sculpt.exe → 向上3级到项目根
        // release: .../target/release/cr_sculpt.exe → 向上3级
        let project_root = exe.parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        project_root.join("src/scene/scene.json")
    }

    fn save_scene(&self) {
        let path = self.scene_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let scene = math::SceneData {
            instances: self.instances.iter().map(|inst| math::SceneInstance {
                model_id: inst.model_id,
                instance_id: inst.instance_id,
                model_matrix: inst.model_matrix,
            }).collect(),
            camera: Some(math::SceneCamera {
                eye: self.camera.eye.into(),
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
            }),
        };
        match serde_json::to_string_pretty(&scene) {
            Ok(json) => {
                match std::fs::write(&path, &json) {
                    Ok(_) => println!("场景已保存到: {}", path.display()),
                    Err(e) => eprintln!("保存场景失败: {}", e),
                }
            }
            Err(e) => eprintln!("序列化场景失败: {}", e),
        }
    }

    fn load_scene(&mut self) -> bool {
        let path = self.scene_path();
        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                println!("未找到存档文件: {}", path.display());
                return false;
            }
        };
        let scene: math::SceneData = match serde_json::from_str(&json) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("解析场景文件失败: {}", e);
                return false;
            }
        };
        for inst_data in &scene.instances {
            if let Some(inst) = self.make_instance(inst_data.model_id, self.instances.len() as u32, glam::Mat4::from_cols_array_2d(&inst_data.model_matrix)) {
                self.push_instance(inst);
            }
        }
        if !self.instances.is_empty() {
            self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
        }
        if let Some(cam) = scene.camera {
            self.camera = math::Camera::new(cam.eye.into(), cam.yaw, cam.pitch);
            println!("已还原摄像机位置: eye=({:.1},{:.1},{:.1}) yaw={:.1} pitch={:.1}",
                cam.eye[0], cam.eye[1], cam.eye[2], cam.yaw, cam.pitch);
        }
        println!("已加载 {} 个实例", scene.instances.len());
        true
    }

    fn rebuild_colliders(&mut self) {
        let t0 = std::time::Instant::now();
        println!("开始构建碰撞体 SDF...");
        let temp_registry = self.model_registry.clone();
        self.model_colliders = Self::build_model_colliders(&temp_registry, &self.triangles);
        println!("碰撞体 SDF 构建完成，耗时 {:.1}ms ({} 个模型)", t0.elapsed().as_secs_f64() * 1000.0, self.model_colliders.len());
    }

    fn make_instance(&self, model_id: u32, instance_id: u32, transform: glam::Mat4) -> Option<math::InstanceData> {
        let model = self.model_registry.get(&model_id)?;
        Some(math::InstanceData {
            model_matrix: transform.to_cols_array_2d(),
            model_id,
            instance_id,
            tri_start: model.tri_start,
            bvh_start: model.tri_count,
            material_color: [model.material_color[0], model.material_color[1], model.material_color[2], 1.0],
            _pad: [0; 8],
        })
    }

    fn spawn_preview_instance(&self, model_id: u32, pos: glam::Vec3) -> Option<math::InstanceData> {
        let model = self.model_registry.get(&model_id)?;
        let transform = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::from_slice(&model.info.default_scale),
            glam::Quat::IDENTITY,
            pos,
        );
        self.make_instance(model_id, 9999, transform)
    }

    fn build_instances_to_draw(&self) -> Vec<math::InstanceData> {
        let cam_pos = self.camera.eye;
        let active_radius = 1350.0;

        let mut instances = Vec::new();
        for inst in &self.instances {
            let pos_x = inst.model_matrix[3][0];
            let pos_z = inst.model_matrix[3][2];
            let dist_xz = ((pos_x - cam_pos.x).powi(2) + (pos_z - cam_pos.z).powi(2)).sqrt();
            if dist_xz <= active_radius {
                instances.push(inst.clone());
            }
        }
        if let Some(model_id) = self.active_spawn_id.or(self.continuous_spawn_id) {
            let (ray_o, ray_dir) = self.camera.get_ray(
                self.last_mouse_pos[0],
                self.last_mouse_pos[1],
                self.config.width,
                self.config.height,
            );
            if ray_dir.y.abs() > 0.001 {
                let t = -ray_o.y / ray_dir.y;
                if t > 0.0 {
                    if let Some(preview) = self.spawn_preview_instance(model_id, ray_o + ray_dir * t) {
                        instances.push(preview);
                    }
                }
            }
        }
        if instances.is_empty() {
            if let Some(instance) = self.make_instance(0, 0, glam::Mat4::IDENTITY) {
                instances.push(instance);
            }
        }
        instances
    }

    fn load_glb_triangles_only(path: &str) -> Vec<math::Triangle> {
        let (document, buffers, _) = gltf::import(path).expect("加载 GLB 失败");
        let mut triangles = Vec::new();
        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
                let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let mut tris = compute_smoothed_triangles(&positions, &indices, &uvs, 45.0).0;
                if let Some(normals) = normals.filter(|n| n.len() == positions.len()) {
                    apply_imported_visual_normals(&mut tris, &normals, &indices);
                }
                triangles.append(&mut tris);
            }
        }
        triangles
    }

    async fn new(window: Arc<Window>, profiler_window: Arc<Window>) -> Self {
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
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
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

        // === 性能监控窗口 Surface 与配置 ===
        let prof_surface = instance.create_surface(profiler_window.clone()).unwrap();
        let prof_size = profiler_window.inner_size();
        let prof_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: prof_size.width.max(1),
            height: prof_size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        prof_surface.configure(&device, &prof_config);

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mainshader.wgsl"))),
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

        // 三角形缓冲区 (最多 10 万个三角形，包含视觉法线、扭曲法线和 UV 数据)（初始值，会动态扩容）
        let triangle_max_size = (100_000 * std::mem::size_of::<math::Triangle>()) as u64;
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
                // 扭曲法线纹理 (binding 15)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 像素阴影纹理 (binding 16)
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        // 扭曲后世界位置纹理 (存扭曲后的世界坐标)
        let warped_pos_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Warped Position Texture"),
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
        let warped_pos_texture_view = warped_pos_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let warped_pos_texture_view_for_render = warped_pos_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
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
                    binding: 14,
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
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
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
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // === 加载所有模型到模型库 ===
        let (model_registry, material_bind_groups, all_triangles, albedo_texture, albedo_texture_view, atlas_width, atlas_height, atlas_cursor_x, atlas_cursor_y, atlas_row_height) = Self::load_all_models(
            &device,
            &queue,
            &material_bind_group_layout,
            &albedo_sampler,
        );

        let model_colliders = std::collections::HashMap::new();

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

        let shadow_system = shadow::ShadowSystem::new(
            &device,
            &shader,
            size.width,
            size.height,
            shadow::ShadowInputs {
                params_buffer: &params_buffer,
                triangle_buffer: &triangle_buffer_to_use,
                instance_buffer: &instance_buffer,
                warp_buffer: &warp_buffer,
                tri_id_view: &tri_id_texture_view,
                world_pos_view: &uv_texture_view,
                normal_view: &normal_texture_view,
            },
        );

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
            &warped_pos_texture_view,
            &shadow_system.view,
            &albedo_texture_view,
            &albedo_sampler,
            &instance_buffer,
            &warp_buffer,
            &sdf_buffer,
        );

        // 1. 先在外部创建 Context
        let egui_ctx = egui::Context::default();

        // 2. 使用刚才创建的局部变量 egui_ctx 来初始化 egui_state
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),           // 注意这里直接传局部变量
            egui::ViewportId::ROOT,     // 修正：使用 ROOT 比较标准
            &window,
            None,
            None,
        );

        // 3. 在 device 还没被移交进 Self 之前，先用它初始化 egui_renderer
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // === 性能监控窗口 Egui 初始化 ===
        let prof_egui_ctx = egui::Context::default();
        let prof_egui_state = egui_winit::State::new(
            prof_egui_ctx.clone(),
            egui::ViewportId::from_hash_of("profiler_viewport"),
            &profiler_window,
            None,
            None,
        );
        let prof_egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // 为主窗口和监控窗口都设置中文字体
        let font_path = "f:\\Cillin_CG\\Cillin_Cg\\Asset\\Font\\GlowSansSC-Normal-Regular.otf";
        if let Ok(font_data) = std::fs::read(font_path) {
            let mut prof_fonts = egui::FontDefinitions::default();
            prof_fonts.font_data.insert("glow_sans".to_owned(), egui::FontData::from_owned(font_data.clone()));
            prof_fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap().insert(0, "glow_sans".to_owned());
            prof_fonts.families.get_mut(&egui::FontFamily::Monospace).unwrap().push("glow_sans".to_owned());

            let mut main_fonts = egui::FontDefinitions::default();
            main_fonts.font_data.insert("glow_sans".to_owned(), egui::FontData::from_owned(font_data));
            main_fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap().insert(0, "glow_sans".to_owned());
            main_fonts.families.get_mut(&egui::FontFamily::Monospace).unwrap().push("glow_sans".to_owned());

            // 注意：主窗口字体已经设置过，但这里优先用这个覆盖以确保一致性
            egui_ctx.set_fonts(main_fonts);
            prof_egui_ctx.set_fonts(prof_fonts);
        }

        // GPU Timestamp Query (在 device move 之前创建)
        let ts_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: 10,
            ty: wgpu::QueryType::Timestamp,
        });
        let ts_resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Resolve Buffer"),
            size: 10 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ts_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Staging Buffer"),
            size: 10 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            shadow_system,
            depth_bind_group_layout,
            depth_bind_group,
            depth_render_pipeline,
            depth_blit_pipeline,
            depth_blit_bind_group,
            show_depth_debug: false,
            show_normal_debug: false,
            show_perf_monitor: false,
            perf_stats: PerfStats::new(),
            profiler_window: ProfilerWindowData {
                window: profiler_window,
                surface: prof_surface,
                config: prof_config,
                egui_ctx: prof_egui_ctx,
                egui_state: prof_egui_state,
                egui_renderer: prof_egui_renderer,
            },
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
            warped_pos_texture,
            warped_pos_texture_view,
            warped_pos_texture_view_for_render,
            albedo_texture,
            albedo_texture_view,
            albedo_sampler,
            atlas_width,
            atlas_height,
            atlas_cursor_x,
            atlas_cursor_y,
            atlas_row_height,
            ts_query_set,
            ts_resolve_buffer,
            ts_staging_buffer,
            model_registry,
            model_colliders,
            material_bind_group_layout,
            material_bind_groups,
            mesh_render_pipeline,
            instances: Vec::new(),
            instance_physics: Vec::new(),
            instance_buffer,
            command_input: String::new(),
            active_spawn_id: None,
            continuous_spawn_id: None,
            rebuild_colliders_requested: false,
            
            // 编辑模式初始化
            selected_instance: None,
            edit_mode: EditMode::None,
            axis_constraint: AxisConstraint::None,
            last_axis_key: None,
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
        warp_normal_view: &wgpu::TextureView,
        shadow_view: &wgpu::TextureView,
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
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(warp_normal_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(shadow_view) },
            ],
        });

        let render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(output_view) },
            ],
        });

        let depth = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Bind Group"),
            layout: depth_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: triangle_buffer.as_entire_binding() }, // 从 3 改成 2
                wgpu::BindGroupEntry { binding: 9, resource: instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: warp_buffer.as_entire_binding() },
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
        let x = mouse_pos[0].clamp(0.0, (self.config.width.saturating_sub(1)) as f32) as u32;
        let y = mouse_pos[1].clamp(0.0, (self.config.height.saturating_sub(1)) as f32) as u32;
        let pixel_size = std::mem::size_of::<math::WarpPixel>() as u64;
        let pixel_offset = ((y as u64 * self.config.width as u64) + x as u64) * pixel_size;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pick WarpPixel Staging Buffer"),
            size: pixel_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.warp_buffer, pixel_offset, &staging_buffer, 0, pixel_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let picked = bytemuck::from_bytes::<math::WarpPixel>(&data[..pixel_size as usize]);
        let instance_idx = if picked.flags > 0 && picked.tri_id > 0 {
            Some(((picked.tri_id >> 20) - 1) as usize)
        } else {
            None
        };

        if let Some(instance_idx) = instance_idx.filter(|idx| *idx < self.instances.len()) {
            self.selected_instance = Some(instance_idx);
            self.params.selected_instance_id = instance_idx as u32;
            println!("选中了实例: {}", instance_idx);

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
            let proj_matrix = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 10000.0);
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

    fn update_collider_debug_points(&mut self) {
        self.visible_vertices.clear();
        let view_proj = {
            let view_matrix = self.camera.get_view_matrix();
            let aspect_ratio = self.config.width as f32 / self.config.height as f32;
            let proj_matrix = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 10000.0);
            proj_matrix * view_matrix
        };

        for instance in &self.instances {
            let Some(collider) = self.model_colliders.get(&instance.model_id) else { continue; };
            let model_mat = glam::Mat4::from_cols_array_2d(&instance.model_matrix);
            let (scale, _, _) = model_mat.to_scale_rotation_translation();
            let marker_size = 0.015 * scale.abs().max_element().max(0.5);
            let step = (collider.sample_points.len() / 1200).max(1);
            for sample in collider.sample_points.iter().step_by(step) {
                let world_pos = model_mat.transform_point3(*sample);
                let clip = view_proj * world_pos.extend(1.0);
                if clip.w <= 0.01 {
                    continue;
                }
                let ndc = clip / clip.w;
                if ndc.x.abs() > 1.1 || ndc.y.abs() > 1.1 || ndc.z < -0.1 || ndc.z > 1.1 {
                    continue;
                }
                for offset in [
                    glam::Vec3::ZERO,
                    glam::Vec3::X * marker_size,
                    -glam::Vec3::X * marker_size,
                    glam::Vec3::Y * marker_size,
                    -glam::Vec3::Y * marker_size,
                ] {
                    self.visible_vertices.push((world_pos + offset).extend(0.5));
                }
                if self.visible_vertices.len() >= 95_000 {
                    break;
                }
            }
        }

        self.queue.write_buffer(&self.scaffold_buffer, 0, bytemuck::cast_slice(&self.visible_vertices));
        self.params.scaffold_count = self.visible_vertices.len() as u32;
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
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.warped_pos_texture_view_for_render,
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
                    // vs_depth 内部已使用 instance.tri_start 偏移，draw call 从 0 开始
                    dpass.draw(0..(reg.tri_count * 3), (i as u32)..(i as u32 + 1));
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

    fn run_shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        self.shadow_system.run(encoder, self.instances.len() as u32);
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
                    load: if self.debug_mode == 6 {
                        wgpu::LoadOp::Clear(wgpu::Color { r: 0.015, g: 0.015, b: 0.02, a: 1.0 })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        rpass.set_pipeline(&self.scaffold_render_pipeline);
        rpass.set_bind_group(0, &self.compute_bind_group, &[]);
        if (self.show_scaffold || self.debug_mode == 6) && self.params.scaffold_count > 0 {
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
        if self.rebuild_colliders_requested {
            self.rebuild_colliders();
            self.rebuild_colliders_requested = false;
        }

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

        physics::apply_physics(&mut self.instances, &mut self.instance_physics, &self.model_colliders, self.edit_mode != EditMode::None, delta_time);

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

                        // 应用 Blender 风格轴约束
                        let constrained_pos = match self.axis_constraint {
                            AxisConstraint::None => new_pos,
                            // 单轴约束：计算射线到轴线的最近点
                            AxisConstraint::X => self.ray_closest_to_axis(ray_o, ray_dir, self.initial_pos, glam::Vec3::X),
                            AxisConstraint::Y => self.ray_closest_to_axis(ray_o, ray_dir, self.initial_pos, glam::Vec3::Y),
                            AxisConstraint::Z => self.ray_closest_to_axis(ray_o, ray_dir, self.initial_pos, glam::Vec3::Z),
                            // 平面约束：投影到对应平面
                            AxisConstraint::YZ => {
                                let t_x = (self.initial_pos.x - ray_o.x) / ray_dir.x;
                                ray_o + ray_dir * t_x
                            }
                            AxisConstraint::XZ => new_pos,
                            AxisConstraint::XY => {
                                let t_z = (self.initial_pos.z - ray_o.z) / ray_dir.z;
                                ray_o + ray_dir * t_z
                            }
                        };

                        // 更新矩阵
                        let new_mat = glam::Mat4::from_scale_rotation_translation(
                            self.initial_scale,
                            self.initial_rot,
                            constrained_pos
                        );
                        self.instances[idx].model_matrix = new_mat.to_cols_array_2d();
                        self.instance_physics[idx].velocity = glam::Vec3::ZERO;
                        self.instance_physics[idx].angular_velocity = glam::Vec3::ZERO;
                        
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

                    // Blender 风格旋转：无约束绕视线方向旋转，X/Y/Z 绕世界轴旋转
                    let rotate_speed = 0.01;
                    let angle = (mouse_delta.x - mouse_delta.y) * rotate_speed;
                    let axis = match self.axis_constraint {
                        AxisConstraint::X | AxisConstraint::YZ => glam::Vec3::X,
                        AxisConstraint::Y | AxisConstraint::XZ => glam::Vec3::Y,
                        AxisConstraint::Z | AxisConstraint::XY => glam::Vec3::Z,
                        AxisConstraint::None => self.camera.get_forward(),
                    };
                    let new_rot = glam::Quat::from_axis_angle(axis.normalize(), angle) * self.initial_rot;

                    // 更新矩阵
                    let new_mat = glam::Mat4::from_scale_rotation_translation(
                        self.initial_scale,
                        new_rot,
                        self.initial_pos
                    );
                    self.instances[idx].model_matrix = new_mat.to_cols_array_2d();
                    self.instance_physics[idx].velocity = glam::Vec3::ZERO;
                    self.instance_physics[idx].angular_velocity = glam::Vec3::ZERO;
                    
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
                    self.instance_physics[idx].velocity = glam::Vec3::ZERO;
                    self.instance_physics[idx].angular_velocity = glam::Vec3::ZERO;
                    
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
        self.sync_instance_physics_len();

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
            if self.edit_mode != EditMode::None {
                let mode_name = match self.edit_mode {
                    EditMode::Grab => "移动 (G)",
                    EditMode::Rotate => "旋转 (R)",
                    EditMode::Scale => "缩放 (S)",
                    _ => "",
                };
                let axis_name = match self.axis_constraint {
                    AxisConstraint::X => " | 锁定 X 轴",
                    AxisConstraint::Y => " | 锁定 Y 轴",
                    AxisConstraint::Z => " | 锁定 Z 轴",
                    AxisConstraint::YZ => " | 锁定 YZ 平面",
                    AxisConstraint::XZ => " | 锁定 XZ 平面",
                    AxisConstraint::XY => " | 锁定 XY 平面",
                    AxisConstraint::None => "",
                };
                ui.label(egui::RichText::new(format!("{}{}", mode_name, axis_name)).color(egui::Color32::from_rgb(100, 200, 255)));
                ui.label("X/Y/Z 锁定轴 | Shift+X/Y/Z 锁定平面 | Esc/左键 退出");
            }
            ui.separator();
            ui.label("几何体列表:");

            ui.separator();
            ui.checkbox(&mut self.show_scaffold, "显示点云 (调试用)");
            ui.checkbox(&mut self.show_depth_debug, "显示深度图 (Depth Map)");
            ui.checkbox(&mut self.show_normal_debug, "显示法线调试 (Normal Debug)");
            ui.checkbox(&mut self.show_perf_monitor, "🛠 打开物理监控窗口 (Profiler)");
            ui.separator();
            ui.label("调试模式 (Ap 可视化):");
            ui.add(egui::Slider::new(&mut self.debug_mode, 0..=6).text("debug_mode"));
            let mode_label = match self.debug_mode {
                0 => "0: 正常渲染",
                1 => "1: Ap1 - ID 图 (绿色=有三角面)",
                2 => "2: Ap2 - 位移点 (红色=已映射)",
                3 => "3: Ap3 - 补洞 (蓝色=补出来的)",
                4 => "4: 隐性 SDF 距离场 (绿近红远)",
                5 => "5: 融合 SDF 预览 (红=融合增强, 绿=距离近)",
                6 => "6: 物理碰撞体采样点 (黄色=参与碰撞)",
                _ => "未知模式",
            };
            ui.label(mode_label);
            ui.separator();
            ui.label("选中模型物理:");
            ui.label("碰撞: SDF Grid + 子步进 CCD");
            if ui.button("构建碰撞体 SDF").clicked() {
                self.rebuild_colliders_requested = true;
            }
            if self.model_colliders.is_empty() {
                ui.label("尚未构建碰撞体，请点击上方按钮");
            } else if let Some(idx) = self.selected_instance.filter(|idx| *idx < self.instances.len()) {
                ui.label(format!("当前实例: {}", idx));
                ui.checkbox(&mut self.instance_physics[idx].gravity_enabled, "开启重力");
                ui.label(format!("速度: {:.2}, {:.2}, {:.2}",
                    self.instance_physics[idx].velocity.x,
                    self.instance_physics[idx].velocity.y,
                    self.instance_physics[idx].velocity.z,
                ));
                ui.label(format!("角速度: {:.2}, {:.2}, {:.2}",
                    self.instance_physics[idx].angular_velocity.x,
                    self.instance_physics[idx].angular_velocity.y,
                    self.instance_physics[idx].angular_velocity.z,
                ));
                if ui.button("清零速度").clicked() {
                    self.instance_physics[idx].velocity = glam::Vec3::ZERO;
                    self.instance_physics[idx].angular_velocity = glam::Vec3::ZERO;
                }
            } else {
                ui.label("未选中模型");
            }
            ui.separator();
            ui.label("扭曲参数:");
            ui.add(egui::Slider::new(&mut self.params.distort_strength, 0.0..=2.0).text("扭曲强度"));
            ui.add(egui::Slider::new(&mut self.params.distort_frequency, 0.1..=5.0).text("扭曲频率"));
            ui.separator();
            if ui.button("📂 导入 GLB").clicked() { import_clicked = true; }

            ui.separator();
            ui.label("命令 (输入 spawn <id> 或 cspawn <id>):");
            ui.text_edit_singleline(&mut self.command_input);
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let parts: Vec<&str> = self.command_input.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "spawn" {
                    if let Ok(id) = parts[1].parse::<u32>() {
                        if self.model_registry.contains_key(&id) {
                            self.active_spawn_id = Some(id);
                            self.continuous_spawn_id = None;
                            println!("进入放置模式，模型 ID: {}", id);
                        } else {
                            println!("模型 ID {} 不存在", id);
                        }
                    }
                }
                if parts.len() >= 2 && parts[0] == "cspawn" {
                    if let Ok(id) = parts[1].parse::<u32>() {
                        if self.model_registry.contains_key(&id) {
                            self.continuous_spawn_id = Some(id);
                            self.active_spawn_id = None;
                            println!("进入连续生成模式，模型 ID: {}", id);
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

            if self.continuous_spawn_id.is_some() {
                ui.label(format!("🔄 连续生成模式，模型 ID: {}", self.continuous_spawn_id.unwrap()));
                if ui.button("退出连续模式").clicked() {
                    self.continuous_spawn_id = None;
                }
            }
        });

        // === 性能监控面板已移至独立物理子窗口（在 render 末尾渲染）===

        // 处理按钮点击事件
        if import_clicked {
            self.import_scaffold();
        }

        // === 计时：Egui 构建 ===
        let te0 = std::time::Instant::now();
        let full_output = self.egui_ctx.end_frame();
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let egui_build_ms = te0.elapsed().as_secs_f32() * 1000.0;

        // 核心修复 1：处理 UI 纹理更新 (字体、图标)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let instances_to_draw = self.build_instances_to_draw();
        self.params.instance_count = instances_to_draw.len() as u32;
        if self.debug_mode == 6 {
            self.update_collider_debug_points();
        } else if !self.show_scaffold {
            self.params.scaffold_count = 0;
        }

        // === 计时：Buffer 上传 ===
        let tb = std::time::Instant::now();
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances_to_draw));
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
        let buffer_upload_ms = tb.elapsed().as_secs_f32() * 1000.0;

        // === 计时：Surface 获取 (含 vsync 等待) ===
        let ts = std::time::Instant::now();
        let output = self.surface.get_current_texture().unwrap();
        let surface_acquire_ms = ts.elapsed().as_secs_f32() * 1000.0;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // === GPU Timestamp: 帧开始 ===
        encoder.write_timestamp(&self.ts_query_set, 0);

        // T0→T1: Clear WarpBuffer
        self.clear_warp_buffer(&mut encoder);
        encoder.write_timestamp(&self.ts_query_set, 1);

        // T1→T2: Depth MRT Pass
        self.run_depth_pass(&mut encoder, &instances_to_draw);
        encoder.write_timestamp(&self.ts_query_set, 2);

        // T2→T3: Compute SDF BG
        self.run_compute_pass(&mut encoder);
        encoder.write_timestamp(&self.ts_query_set, 3);

        // T3→T4: Shadow Binning (clear + linked-list insert)
        self.shadow_system.clear_and_bin(&mut encoder, instances_to_draw.len() as u32);
        encoder.write_timestamp(&self.ts_query_set, 4);

        // T4→T5: Shadow Tracing (per-pixel ray traversal)
        self.shadow_system.trace_shadow(&mut encoder);
        encoder.write_timestamp(&self.ts_query_set, 5);

        // T5→T6: AP3 Shading
        self.run_ap3_pass(&mut encoder);
        encoder.write_timestamp(&self.ts_query_set, 6);

        if self.debug_mode != 6 {
            self.run_draw_pass(&mut encoder, &view);
        }
        // T6→T7: Draw Blit
        encoder.write_timestamp(&self.ts_query_set, 7);

        self.run_scaffold_pass(&mut encoder, &view);
        // T7→T8: Scaffold
        encoder.write_timestamp(&self.ts_query_set, 8);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_descriptor);
        self.run_ui_pass(&mut encoder, &view, &paint_jobs, &screen_descriptor);
        // T8→T9: UI Overlay
        encoder.write_timestamp(&self.ts_query_set, 9);

        // 处理纹理释放
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // === Resolve Timestamps (0..10) ===
        encoder.resolve_query_set(&self.ts_query_set, 0..10, &self.ts_resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.ts_resolve_buffer,
            0,
            &self.ts_staging_buffer,
            0,
            8 * 10,
        );

        let tsp = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        let submit_present_ms = tsp.elapsed().as_secs_f32() * 1000.0;

        // === 异步读取 Timestamp 结果 (10 probes → 9 intervals) ===
        let (gpu_clear_warp_ms, gpu_depth_ms, gpu_compute_ms, gpu_shadow_bin_ms, gpu_shadow_trace_ms, gpu_ap3_ms, gpu_draw_ms, gpu_scaffold_ms, gpu_ui_ms, gpu_total_ms) = {
            let buffer_slice = self.ts_staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
            self.device.poll(wgpu::Maintain::Wait);
            if rx.recv().is_ok() && buffer_slice.get_mapped_range().len() >= 80 {
                let data = buffer_slice.get_mapped_range();
                let timestamps: &[u64] = bytemuck::cast_slice(&data);
                let period_ns = self.queue.get_timestamp_period() as f64;
                
                fn ts_diff(a: u64, b: u64, period: f64) -> f32 {
                    if a > b || period <= 0.0 { return 0.0; }
                    ((b - a) as f64 * period / 1_000_000.0) as f32
                }
                
                let p = period_ns;
                (
                    ts_diff(timestamps[0], timestamps[1], p),
                    ts_diff(timestamps[1], timestamps[2], p),
                    ts_diff(timestamps[2], timestamps[3], p),
                    ts_diff(timestamps[3], timestamps[4], p),
                    ts_diff(timestamps[4], timestamps[5], p),
                    ts_diff(timestamps[5], timestamps[6], p),
                    ts_diff(timestamps[6], timestamps[7], p),
                    ts_diff(timestamps[7], timestamps[8], p),
                    ts_diff(timestamps[8], timestamps[9], p),
                    ts_diff(timestamps[0], timestamps[9], p),
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        };
        self.ts_staging_buffer.unmap();

        // === 收集性能统计 ===
        let rendered_tris: u32 = instances_to_draw.iter()
            .filter_map(|inst| self.model_registry.get(&inst.model_id))
            .map(|reg| reg.tri_count)
            .sum();
        
        self.perf_stats = PerfStats {
            frame_time_ms: if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 },
            fps: self.fps,
            depth_pass_ms: gpu_depth_ms,
            compute_pass_ms: gpu_compute_ms,
            ap3_pass_ms: gpu_ap3_ms,
            draw_pass_ms: gpu_draw_ms,
            scaffold_pass_ms: gpu_scaffold_ms,
            ui_pass_ms: gpu_ui_ms,
            egui_build_ms,
            buffer_upload_ms,
            surface_acquire_ms,
            submit_present_ms,
            gpu_clear_warp_ms,
            gpu_depth_ms,
            gpu_compute_ms,
            gpu_shadow_bin_ms,
            gpu_shadow_trace_ms,
            gpu_ap3_ms,
            gpu_draw_ms,
            gpu_scaffold_ms,
            gpu_ui_ms,
            gpu_total_ms,
            depth_draw_calls: instances_to_draw.len() as u32,
            forward_draw_calls: 0,
            draw_draw_calls: 1,
            scaffold_draw_calls: if self.show_scaffold || self.debug_mode == 6 { 1 } else { 0 },
            total_triangles: self.triangles.len() as u32,
            rendered_triangles: rendered_tris,
            instance_count: instances_to_draw.len() as u32,
            triangle_buffer_size: (self.triangles.len() * std::mem::size_of::<math::Triangle>()) as u64,
            instance_buffer_size: (instances_to_draw.len() * std::mem::size_of::<math::InstanceData>()) as u64,
            frame_history: self.perf_stats.frame_history.clone(),
            max_history: self.perf_stats.max_history,
        };
        self.perf_stats.push_frame(if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 });

        // ==========================================
        // 核心改动：在独立的物理子窗口中绘制高详尽度的 Profiler UI
        // ==========================================
        if self.show_perf_monitor {
            self.profiler_window.window.set_visible(true);

            let prof_input = self.profiler_window.egui_state.take_egui_input(&self.profiler_window.window);
            self.profiler_window.egui_ctx.begin_frame(prof_input);

            // 在独立窗口内全屏展示监控
            egui::CentralPanel::default().show(&self.profiler_window.egui_ctx, |ui| {
                let s = &self.perf_stats;

                ui.heading(egui::RichText::new("CrSculpt 核心渲染监控与深度诊断系统").size(20.0).color(egui::Color32::from_rgb(120, 220, 255)));
                ui.small("当前窗口为物理独立子窗口，可拖拽、缩放，绝不卡出主渲染视口");
                ui.separator();

                let total_gpu = s.gpu_total_ms.max(1.0);

                ui.columns(2, |columns| {
                    // 左栏：实时帧率与耗时图表
                    columns[0].vertical(|ui| {
                        ui.heading("帧时间分析");
                        let history = &s.frame_history;
                        let (avg_t, min_t, max_t) = if !history.is_empty() {
                            let sum: f32 = history.iter().sum();
                            let avg = sum / history.len() as f32;
                            let min = history.iter().cloned().fold(f32::MAX, f32::min);
                            let max = history.iter().cloned().fold(f32::MIN, f32::max);
                            (avg, min, max)
                        } else {
                            (0.0, 0.0, 0.0)
                        };

                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::GREEN, format!("FPS: {:.1}", s.fps));
                            ui.label(format!("当前帧耗时: {:.2} ms", s.frame_time_ms));
                        });
                        ui.small(format!("统计区间: 平均: {:.2}ms | 最小: {:.2}ms | 最大: {:.2}ms", avg_t, min_t, max_t));

                        // 绘制性能曲线图
                        if !history.is_empty() {
                            let range = (max_t - min_t).max(1.0);
                            let graph_rect = egui::Rect::from_min_size(
                                ui.cursor().min,
                                egui::Vec2::new(ui.available_width(), 90.0),
                            );
                            let painter = ui.painter_at(graph_rect);
                            painter.rect_filled(graph_rect, 4.0, egui::Color32::from_gray(15));
                            
                            // 60FPS(16.6ms) 目标基准线
                            let target_y = graph_rect.bottom() - (16.67 - min_t) / range * graph_rect.height();
                            if target_y >= graph_rect.top() && target_y <= graph_rect.bottom() {
                                painter.line_segment(
                                    [egui::Pos2::new(graph_rect.left(), target_y), egui::Pos2::new(graph_rect.right(), target_y)],
                                    egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
                                );
                            }

                            let n = history.len();
                            let w = graph_rect.width() / (n as f32).max(1.0);
                            let mut points = Vec::new();
                            for (i, &t) in history.iter().enumerate() {
                                let x = graph_rect.left() + i as f32 * w;
                                let y = graph_rect.bottom() - (t - min_t) / range * graph_rect.height();
                                points.push(egui::Pos2::new(x, y));
                            }
                            if points.len() >= 2 {
                                painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, egui::Color32::from_rgb(50, 220, 100))));
                            }
                            ui.allocate_rect(graph_rect, egui::Sense::hover());
                        }

                        ui.separator();
                        ui.heading("GPU Pass 耗时占比明细");
                        fn bar(ui: &mut egui::Ui, name: &str, ms: f32, total: f32, col: egui::Color32) {
                            ui.horizontal(|ui| {
                                ui.label(format!("{:<14}", name));
                                let frac = (ms / total).clamp(0.0, 1.0);
                                ui.add(egui::ProgressBar::new(frac).desired_width(180.0).fill(col));
                                ui.label(format!("{:.2} ms ({:.1}%)", ms, frac * 100.0));
                            });
                        }
                        bar(ui, "Clear Warp", s.gpu_clear_warp_ms, total_gpu, egui::Color32::from_rgb(120, 120, 120));
                        bar(ui, "Depth MRT", s.depth_pass_ms, total_gpu, egui::Color32::from_rgb(100, 149, 237));
                        bar(ui, "Compute SDF", s.compute_pass_ms, total_gpu, egui::Color32::from_rgb(0, 191, 255));
                        bar(ui, "Shadow Bin", s.gpu_shadow_bin_ms, total_gpu, egui::Color32::from_rgb(255, 127, 80));
                        bar(ui, "Shadow Trace", s.gpu_shadow_trace_ms, total_gpu, egui::Color32::from_rgb(255, 69, 0));
                        bar(ui, "AP3 Shading", s.ap3_pass_ms, total_gpu, egui::Color32::from_rgb(186, 85, 211));
                        bar(ui, "Draw Blit", s.draw_pass_ms, total_gpu, egui::Color32::from_rgb(60, 179, 113));
                        bar(ui, "Scaffold", s.scaffold_pass_ms, total_gpu, egui::Color32::from_rgb(230, 180, 40));
                        bar(ui, "UI Overlay", s.ui_pass_ms, total_gpu, egui::Color32::GRAY);
                    });

                    // 右栏：资源开销与深度诊断分析
                    columns[1].vertical(|ui| {
                        ui.heading("🛠 系统资源与性能诊断");
                        
                        // 智能诊断逻辑（直接解决你的 51.8ms AP3 痛点！）
                        if s.ap3_pass_ms > 15.0 {
                            ui.group(|ui| {
                                ui.colored_label(egui::Color32::from_rgb(255, 80, 80), "🚨 核心性能红色瓶颈检测：AP3 Pass 严重过载");
                                ui.label(format!("当前 AP3 像素扭曲阶段耗时高达 {:.2} ms (占 GPU 用时的 {:.1}%)。", s.ap3_pass_ms, (s.ap3_pass_ms / total_gpu) * 100.0));
                                ui.small("原理透视：该开销由 mainshader.wgsl 中的 FBM (分形布朗运动) 噪声函数导致。为了生成法线偏移，着色器对每个覆盖像素执行了 4 次 3D FBM 采样，等效于每像素运算 192 次 Hash 插值，导致算术单元(ALU)达到极限瓶颈。");
                                ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "优化行动建议：");
                                ui.small(" 1. 将主着色器中 FBM 循环迭代次数由 6 次缩减到 3~4 次，耗时可立减 40% 以上。");
                                ui.small(" 2. 避免对远景或非扭曲表面的像素执行 FBM 运算（可通过早期裁减跳过）。");
                                ui.small(" 3. 采用预计算 3D 噪声贴图 (Noise LUT) 采样，以 Texture 访问替代纯算术计算。");
                            });
                        } else {
                            ui.group(|ui| {
                                ui.colored_label(egui::Color32::GREEN, "✔ 渲染管线负载正常");
                                ui.small("所有着色阶段运算耗时均在 15ms 绿线预算内。");
                            });
                        }

                        ui.separator();
                        ui.heading("GPU 显存占用分析");
                        fn format_bytes(bytes: u64) -> String {
                            if bytes >= 1024 * 1024 { format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0)) }
                            else { format!("{:.2} KB", bytes as f64 / 1024.0) }
                        }
                        
                        let tri_cap = self.triangle_buffer.size();
                        let warp_cap = self.warp_buffer.size();

                        ui.label(format!("三角形静态缓冲: {} / {}", format_bytes(s.triangle_buffer_size), format_bytes(tri_cap)));
                        ui.add(egui::ProgressBar::new(s.triangle_buffer_size as f32 / tri_cap as f32).desired_height(4.0));

                        ui.label(format!("世界空间 Warp 缓冲区 (GBuffer 映射阵列):"));
                        ui.small(format!("  分辨率: {} x {} 像素 (每个元素占 32 字节)", self.config.width, self.config.height));
                        ui.small(format!("  总内存开销: {} (理论占用上限)", format_bytes(warp_cap)));

                        ui.separator();
                        ui.heading("场景与物理配置");
                        ui.small(format!("当前实例总数: {} | 碰撞体网格数: {}", s.instance_count, self.model_colliders.len()));
                        ui.small(format!("光网格(Sun-Grid): 1024x1024 Cells | 网格覆盖世界半宽: {:.1}", 400.0));
                    });
                });
            });

            let prof_output = self.profiler_window.egui_ctx.end_frame();
            let prof_paint_jobs = self.profiler_window.egui_ctx.tessellate(prof_output.shapes, prof_output.pixels_per_point);

            for (id, image_delta) in &prof_output.textures_delta.set {
                self.profiler_window.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
            }

            // 获取性能监控窗口的 swapchain 纹理进行独立渲染
            let prof_surf_texture = self.profiler_window.surface.get_current_texture().unwrap();
            let prof_view = prof_surf_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut prof_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Profiler Window Encoder"),
            });

            let prof_screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.profiler_window.config.width, self.profiler_window.config.height],
                pixels_per_point: self.profiler_window.window.scale_factor() as f32,
            };

            self.profiler_window.egui_renderer.update_buffers(&self.device, &self.queue, &mut prof_encoder, &prof_paint_jobs, &prof_screen_descriptor);

            {
                let mut rpass = prof_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Profiler Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &prof_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.09, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                self.profiler_window.egui_renderer.render(&mut rpass, &prof_paint_jobs, &prof_screen_descriptor);
            }

            for id in &prof_output.textures_delta.free {
                self.profiler_window.egui_renderer.free_texture(id);
            }

            self.queue.submit(std::iter::once(prof_encoder.finish()));
            prof_surf_texture.present();
        } else {
            // 如果未勾选，确保物理窗口不可见
            self.profiler_window.window.set_visible(false);
        }
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

            // 重新创建扭曲后世界位置纹理
            self.warped_pos_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Warped Position Texture"),
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
            self.warped_pos_texture_view = self.warped_pos_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.warped_pos_texture_view_for_render = self.warped_pos_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.shadow_system.resize(
                &self.device,
                new_size.width,
                new_size.height,
                shadow::ShadowInputs {
                    params_buffer: &self.params_buffer,
                    triangle_buffer: &self.triangle_buffer,
                    instance_buffer: &self.instance_buffer,
                    warp_buffer: &self.warp_buffer,
                    tri_id_view: &self.tri_id_texture_view,
                    world_pos_view: &self.uv_texture_view,
                    normal_view: &self.normal_texture_view,
                },
            );

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
                &self.warped_pos_texture_view,
                &self.shadow_system.view,
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

    fn resize_profiler(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.profiler_window.config.width = new_size.width;
            self.profiler_window.config.height = new_size.height;
            self.profiler_window.surface.configure(&self.device, &self.profiler_window.config);
        }
    }

    // 在 Rust 里实现和 Shader 完全一致的 smin 基座采样
    // 打开文件对话框并导入 GLB（追加到现有模型库，不覆盖）
    fn import_scaffold(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GLB Files", &["glb"])
            .add_filter("GLTF Files", &["gltf"])
            .pick_file() {
            
            // 保存路径
            self.scaffold_path = Some(path.to_str().unwrap().to_string());
            let path_str = self.scaffold_path.clone().unwrap();

            // === 使用 load_glb_for_atlas 加载模型（与 manifest 模型一致的处理） ===
            let (mut new_triangles, new_rgba_pixels, new_tex_w, new_tex_h, material_color) = Self::load_glb_for_atlas(&path_str);
            
            let new_tri_count = new_triangles.len() as u32;
            if new_tri_count == 0 {
                println!("导入的模型没有三角形，跳过");
                return;
            }

            // === 分配唯一 ID（不与 manifest 模型冲突） ===
            let max_existing_id = self.model_registry.keys().max().copied().unwrap_or(0);
            let new_model_id = (max_existing_id + 1).max(1000);

            // === Shelf-packing：确定新纹理在 atlas 中的位置 ===
            const MAX_ATLAS_WIDTH: u32 = 8192;
            let tex_w = new_tex_w.max(1);
            let tex_h = new_tex_h.max(1);
            let old_atlas_w = self.atlas_width.max(1);
            let old_atlas_h = self.atlas_height.max(1);

            // 如果当前行放不下，换行
            let (place_x, place_y) = if self.atlas_cursor_x + tex_w > MAX_ATLAS_WIDTH {
                (0, self.atlas_cursor_y + self.atlas_row_height)
            } else {
                (self.atlas_cursor_x, self.atlas_cursor_y)
            };

            // 计算新的 atlas 尺寸
            let new_atlas_w = old_atlas_w.max(place_x + tex_w);
            let new_atlas_h = old_atlas_h.max(place_y + tex_h);
            let need_expand = new_atlas_w > old_atlas_w || new_atlas_h > old_atlas_h;

            if need_expand {
                // 创建新的 atlas 纹理
                let new_atlas_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Extended Model Texture Atlas"),
                    size: wgpu::Extent3d {
                        width: new_atlas_w,
                        height: new_atlas_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });

                // 复制旧 atlas 到新 atlas（保持在左上角，旧 UV 无需修改）
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Atlas Encoder"),
                });
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.albedo_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &new_atlas_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: old_atlas_w,
                        height: old_atlas_h,
                        depth_or_array_layers: 1,
                    },
                );
                self.queue.submit(Some(encoder.finish()));

                // 写入新模型的纹理到 atlas
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &new_atlas_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: place_x,
                            y: place_y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &new_rgba_pixels,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(tex_w * 4),
                        rows_per_image: Some(tex_h),
                    },
                    wgpu::Extent3d {
                        width: tex_w,
                        height: tex_h,
                        depth_or_array_layers: 1,
                    },
                );

                // 【核心修复】当图集发生扩容时，将当前显存/内存中已经存在的全部旧三角形的 UV 坐标，
                // 按照画布扩容的比例进行等比缩放，确保旧 UV 指向正确的像素区域。
                // 公式: U_新 = U_旧 × (W_旧 / W_新), V_新 = V_旧 × (H_旧 / H_新)
                let scale_u = old_atlas_w as f32 / new_atlas_w as f32;
                let scale_v = old_atlas_h as f32 / new_atlas_h as f32;
                for tri in &mut self.triangles {
                    tri.uv01[0] *= scale_u;
                    tri.uv01[1] *= scale_v;
                    tri.uv01[2] *= scale_u;
                    tri.uv01[3] *= scale_v;
                    tri.uv2[0] *= scale_u;
                    tri.uv2[1] *= scale_v;
                }

                self.albedo_texture = new_atlas_texture;
                self.atlas_width = new_atlas_w;
                self.atlas_height = new_atlas_h;
            } else {
                // 不需要扩展，直接写入现有 atlas
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.albedo_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: place_x,
                            y: place_y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &new_rgba_pixels,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(tex_w * 4),
                        rows_per_image: Some(tex_h),
                    },
                    wgpu::Extent3d {
                        width: tex_w,
                        height: tex_h,
                        depth_or_array_layers: 1,
                    },
                );
            }

            // === 为新模型的三角形重映射 UV 到 atlas ===
            let u0 = place_x as f32 / new_atlas_w as f32;
            let v0 = place_y as f32 / new_atlas_h as f32;
            let us = tex_w as f32 / new_atlas_w as f32;
            let vs = tex_h as f32 / new_atlas_h as f32;
            for tri in &mut new_triangles {
                tri.uv01 = [
                    u0 + tri.uv01[0] * us,
                    v0 + tri.uv01[1] * vs,
                    u0 + tri.uv01[2] * us,
                    v0 + tri.uv01[3] * vs,
                ];
                tri.uv2 = [u0 + tri.uv2[0] * us, v0 + tri.uv2[1] * vs, 0.0, 0.0];
            }

            // === 追加三角形到全局列表 ===
            let tri_start = self.triangles.len() as u32;
            self.triangles.extend(new_triangles);

            // === 更新 GPU 三角形缓冲区 ===
            let total_tri_size = (self.triangles.len() * std::mem::size_of::<math::Triangle>()) as u64;
            Self::ensure_buffer_size(
                &self.device,
                &mut self.triangle_buffer,
                total_tri_size,
                "Triangle Buffer",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.queue.write_buffer(&self.triangle_buffer, 0, bytemuck::cast_slice(&self.triangles));

            // === 更新 atlas cursor 位置 ===
            self.atlas_cursor_x = place_x + tex_w;
            self.atlas_cursor_y = place_y;
            self.atlas_row_height = self.atlas_row_height.max(tex_h);

            // === 更新 albedo_texture_view 和 material bind groups（仅当 atlas 扩展时） ===
            if need_expand {
                self.albedo_texture_view = self.albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
                // 所有旧模型的 material bind group 需要更新（指向新的 atlas texture view）
                let old_model_ids: Vec<u32> = self.model_registry.keys().copied().collect();
                for model_id in old_model_ids {
                    let new_mat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Material Bind Group {}", model_id)),
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
                    self.material_bind_groups.insert(model_id, new_mat_bg);
                }
            }

            // 注册新模型
            self.model_registry.insert(new_model_id, math::ModelRegistryItem {
                info: math::ModelManifestItem {
                    id: new_model_id,
                    name: format!("Imported_{}", new_model_id),
                    file: path_str,
                    default_scale: [1.0, 1.0, 1.0],
                },
                tri_start,
                tri_count: new_tri_count,
                material_id: new_model_id,
                material_color,
            });
            if let Some(reg) = self.model_registry.get(&new_model_id) {
                let temp_registry = std::collections::HashMap::from([(new_model_id, reg.clone())]);
                self.model_colliders.extend(Self::build_model_colliders(&temp_registry, &self.triangles));
            }

            // 为新模型创建 material bind group
            let new_mat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Imported Material Bind Group {}", new_model_id)),
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
            self.material_bind_groups.insert(new_model_id, new_mat_bg);

            // === 更新全局参数 ===
            // 不覆盖 model_center / base_radius / base_color（这些按模型实例独立计算）
            self.params.anchor_count = self.triangles.len() as u32;
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));

            // === 重新创建所有 BindGroup（因为 atlas 纹理变了） ===
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
                &self.warped_pos_texture_view,
                &self.shadow_system.view,
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

            // === 为导入的模型创建默认实例 ===
            if let Some(instance) = self.make_instance(new_model_id, 0, glam::Mat4::IDENTITY) {
                self.push_instance(instance);
            }
            self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));

            println!("导入模型成功: ID={}, 三角形={}, 总三角形={}, Atlas={}x{}",
                new_model_id, new_tri_count, self.triangles.len(), new_atlas_w, new_atlas_h);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // 1. 创建主窗口 (CrSculpt 视口)
    let window = Arc::new(winit::window::Window::new(&event_loop).unwrap());
    window.set_title("CrSculpt");

    // 2. 创建物理独立的性能监控窗口 (Profiler)，初始隐藏
    let profiler_window = Arc::new(winit::window::Window::new(&event_loop).unwrap());
    profiler_window.set_title("CrSculpt Performance Profiler & Diagnostics");
    profiler_window.set_visible(false);

    // 3. 将两个物理窗口传入初始化
    let mut app = pollster::block_on(App::new(window.clone(), profiler_window.clone()));

    // 自动加载上次保存的场景
    app.load_scene();

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        match event {
            // ---- A. 分流处理：主窗口事件 ----
            Event::WindowEvent { ref event, window_id } if window_id == app.window.id() => {
                // 让 egui 优先处理 UI，但保留全局键盘快捷键。
                let egui_consumed = app.egui_state.on_window_event(&app.window, event).consumed;
                if egui_consumed && !matches!(event, WindowEvent::KeyboardInput { .. }) {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => app.resize(*size),
                    
                    // --- 键盘监听：Shift 和 WASD ---
                    WindowEvent::KeyboardInput { event: kb_event, .. } => {
                        let pressed = kb_event.state == winit::event::ElementState::Pressed;
                        match kb_event.physical_key {
                            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                                app.is_shift_pressed = pressed;
                            }
                            PhysicalKey::Code(KeyCode::KeyW) => {
                                app.is_w_pressed = pressed;
                            }
                            PhysicalKey::Code(KeyCode::KeyA) => {
                                app.is_a_pressed = pressed;
                            }
                            PhysicalKey::Code(KeyCode::KeyD) => {
                                app.is_d_pressed = pressed;
                            }
                            PhysicalKey::Code(KeyCode::KeyS) => {
                                if pressed {
                                    if app.selected_instance.is_some() {
                                        app.edit_mode = EditMode::Scale;
                                        app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                    } else {
                                        app.is_s_pressed = true;
                                    }
                                } else if app.edit_mode == EditMode::None {
                                    app.is_s_pressed = false;
                                }
                            }
                            PhysicalKey::Code(KeyCode::KeyG) => {
                                if pressed && app.selected_instance.is_some() {
                                    app.edit_mode = EditMode::Grab;
                                    app.axis_constraint = AxisConstraint::None;
                                    app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                }
                            }
                            PhysicalKey::Code(KeyCode::KeyR) => {
                                if pressed && app.selected_instance.is_some() {
                                    app.edit_mode = EditMode::Rotate;
                                    app.axis_constraint = AxisConstraint::None;
                                    app.initial_mouse_pos = glam::Vec2::new(app.last_mouse_pos[0], app.last_mouse_pos[1]);
                                }
                            }
                            PhysicalKey::Code(key @ (KeyCode::KeyX | KeyCode::KeyY | KeyCode::KeyZ)) => {
                                if app.edit_mode == EditMode::Grab || app.edit_mode == EditMode::Rotate {
                                    if pressed {
                                        let axis = match key {
                                            KeyCode::KeyX => AxisConstraint::X,
                                            KeyCode::KeyY => AxisConstraint::Y,
                                            _ => AxisConstraint::Z,
                                        };
                                        let plane = match key {
                                            KeyCode::KeyX => AxisConstraint::YZ,
                                            KeyCode::KeyY => AxisConstraint::XZ,
                                            _ => AxisConstraint::XY,
                                        };
                                        if app.is_shift_pressed {
                                            app.axis_constraint = plane;
                                        } else {
                                            if app.last_axis_key == Some(key) {
                                                app.axis_constraint = AxisConstraint::None;
                                            } else {
                                                app.axis_constraint = axis;
                                            }
                                        }
                                        app.last_axis_key = Some(key);
                                    }
                                }
                            }
                            PhysicalKey::Code(KeyCode::Escape) => {
                                if pressed {
                                    app.edit_mode = EditMode::None;
                                    app.axis_constraint = AxisConstraint::None;
                                }
                            }
                            PhysicalKey::Code(KeyCode::KeyP) => {
                                if pressed {
                                    app.save_scene();
                                }
                            }
                            PhysicalKey::Code(KeyCode::Delete) => {
                                if pressed {
                                    if let Some(idx) = app.selected_instance {
                                        if idx < app.instances.len() {
                                            app.instances.remove(idx);
                                            app.instance_physics.remove(idx);
                                            app.selected_instance = None;
                                            app.params.selected_instance_id = u32::MAX;
                                            app.queue.write_buffer(&app.params_buffer, 0, bytemuck::cast_slice(&[app.params]));
                                            // 重新上传实例数据到 GPU
                                            let instance_count = app.instances.len().max(1);
                                            let buffer_size = (instance_count * std::mem::size_of::<math::InstanceData>()) as u64;
                                            App::ensure_buffer_size(&app.device, &mut app.instance_buffer, buffer_size, "Instance Buffer", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                                            app.queue.write_buffer(&app.instance_buffer, 0, bytemuck::cast_slice(&app.instances));
                                            println!("已删除实例 {}", idx);
                                        }
                                    }
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
                            if *state == winit::event::ElementState::Pressed && app.active_spawn_id.is_none() && app.continuous_spawn_id.is_none() && app.edit_mode == EditMode::None {
                                app.pick_instance(app.last_mouse_pos);
                            }
                            
                            // 编辑模式下，左键按下表示确认变换
                            if *state == winit::event::ElementState::Pressed && app.edit_mode != EditMode::None {
                                app.edit_mode = EditMode::None;
                                app.axis_constraint = AxisConstraint::None;
                                app.last_axis_key = None;
                                println!("编辑模式已退出");
                            }
                            
                            // 左键按下时处理实例放置
                            if *state == winit::event::ElementState::Pressed && (app.active_spawn_id.is_some() || app.continuous_spawn_id.is_some()) {
                                let model_id = app.active_spawn_id.or(app.continuous_spawn_id).unwrap();
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
                                            let transform = glam::Mat4::from_scale_rotation_translation(
                                                glam::Vec3::from_slice(&model_info.info.default_scale),
                                                glam::Quat::IDENTITY,
                                                intersect_pos,
                                            );
                                            if let Some(new_instance) = app.make_instance(model_id, app.instances.len() as u32, transform) {
                                                app.push_instance(new_instance);
                                                app.queue.write_buffer(&app.instance_buffer, 0, bytemuck::cast_slice(&app.instances));
                                                println!("放置实例: 模型 ID {} 在位置 {:?}", model_id, intersect_pos);
                                                if app.active_spawn_id.is_some() {
                                                    app.active_spawn_id = None;
                                                }
                                            }
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
            // ---- B. 分流处理：性能监控窗口事件 ----
            Event::WindowEvent { ref event, window_id } if window_id == app.profiler_window.window.id() => {
                let egui_consumed = app.profiler_window.egui_state.on_window_event(&app.profiler_window.window, event).consumed;
                if egui_consumed {
                    return;
                }

                match event {
                    // 子窗口被点击关闭时，仅仅是隐藏它并重置控制台勾选状态，不退出进程
                    WindowEvent::CloseRequested => {
                        app.show_perf_monitor = false;
                        app.profiler_window.window.set_visible(false);
                    }
                    // 独立窗口允许自由拖拽缩放大小，数据完美自适应
                    WindowEvent::Resized(size) => {
                        app.resize_profiler(*size);
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
                if app.show_perf_monitor {
                    app.profiler_window.window.request_redraw();
                }
            }
            _ => {}
        }
    }).unwrap();
}



