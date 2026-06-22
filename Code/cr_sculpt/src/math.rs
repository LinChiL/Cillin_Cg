

use serde::{Deserialize, Serialize};
use glam::Vec4Swizzles;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridCell {
    pub offset: u32, // 该格子在排序后的锚点数组中的起点
    pub count: u32,  // 该格子里有多少个锚点
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub v0: [f32; 4], // xyz = 顶点0, w = 属性
    pub v1: [f32; 4], // xyz = 顶点1, w = 属性
    pub v2: [f32; 4], // xyz = 顶点2, w = 属性
    pub n0: [f32; 4], // xyz = 视觉法线0, w = 填充
    pub n1: [f32; 4], // xyz = 视觉法线1, w = 填充
    pub n2: [f32; 4], // xyz = 视觉法线2, w = 填充
    pub warp_n0: [f32; 4], // xyz = 扭曲法线0, w = 填充
    pub warp_n1: [f32; 4], // xyz = 扭曲法线1, w = 填充
    pub warp_n2: [f32; 4], // xyz = 扭曲法线2, w = 填充
    pub uv01: [f32; 4], // [u0, v0, u1, v1]
    pub uv2: [f32; 4],  // [u2, v2, 0.0, 0.0] - 补齐到 vec4 保证对齐
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub view_inv: [[f32; 4]; 4],       // 64
    pub proj_inv: [[f32; 4]; 4],       // 64
    pub prev_view_proj: [[f32; 4]; 4], // 64
    pub cam_pos: [f32; 4],             // 16
    pub light_dir: [f32; 4],           // 16

    pub anchor_count: u32,             // 4
    pub scaffold_count: u32,           // 4
    pub time: f32,                     // 4
    pub selected_instance_id: u32,     // 4 (选中的实例ID)

    pub model_center: [f32; 4],        // 16
    pub base_color: [f32; 4],          // 16

    pub base_radius: f32,              // 4
    pub debug_mode: u32,               // 4
    pub screen_width: u32,              // 4
    pub screen_height: u32,             // 4

    // Ap1 包络字段
    pub envelope_displacement: f32,     // 4 — 滑条控制偏移量
    pub show_envelope: u32,             // 4 — 1 显示, 0 关闭
    pub envelope_vertex_count: u32,     // 4 — envelope 顶点数 (3 × 三角形数)
    pub _pad: u32,                      // 4 (对齐填充)

    // Ap2 扭曲字段
    pub distort_strength: f32,          // 4 — 扭曲强度
    pub distort_frequency: f32,         // 4 — 扭曲频率
    pub ap2_iteration: u32,             // 4 — 反向传播迭代次数
    pub instance_count: u32,            // 4 — 当前帧实例数量
} // 总计 320 + 16 = 336 字节

impl Default for Params {
    fn default() -> Self {
        Self {
            view_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0, 1.0, -5.0, 1.0],
            light_dir: [0.0, 1.0, 0.0, 0.0],
            anchor_count: 0,
            scaffold_count: 0,
            time: 0.0,
            selected_instance_id: 9999, // 默认未选中
            model_center: [0.0, 0.0, 0.0, 1.0],
            base_color: [0.8, 0.8, 0.8, 1.0],
            base_radius: 1.0,
            debug_mode: 0u32,
            screen_width: 0u32,
            screen_height: 0u32,
            envelope_displacement: 0.5,
            show_envelope: 0,
            envelope_vertex_count: 0,
            _pad: 0u32,
            // Ap2 默认值
            distort_strength: 0.3,
            distort_frequency: 1.0,
            ap2_iteration: 4,
            instance_count: 1u32,
        }
    }
}

impl Params {
    pub fn update_matrices(&mut self, camera: &Camera, width: u32, height: u32) {
        let view = camera.get_view_matrix();
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), width as f32 / height as f32, 0.1, 1000.0);
        
        self.view_inv = view.inverse().to_cols_array_2d();
        self.proj_inv = proj.inverse().to_cols_array_2d();
        self.prev_view_proj = (proj * view).to_cols_array_2d();
        self.cam_pos = camera.eye.extend(1.0).to_array();
        self.screen_width = width;
        self.screen_height = height;
    }
}

#[derive(Clone, Copy)]
pub struct MeshSample {
    pub pos: glam::Vec3,
    pub normal: glam::Vec3,
}

pub struct Camera {
    pub eye: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new(eye: glam::Vec3, yaw: f32, pitch: f32) -> Self {
        Self { eye, yaw, pitch }
    }

    pub fn get_forward(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    pub fn get_right(&self) -> glam::Vec3 {
        glam::Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    pub fn get_up(&self) -> glam::Vec3 {
        self.get_right().cross(self.get_forward()).normalize()
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_to_rh(self.eye, self.get_forward(), glam::Vec3::Y)
    }

    // Blender 旋转：中键
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.002; // 降低灵敏度，使旋转更平滑
        self.pitch -= dy * 0.002; // 降低灵敏度，使旋转更平滑
        self.pitch = self.pitch.clamp(-1.5, 1.5); // 防止翻转
    }

    // Blender 平移：Shift + 中键
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.01;
        self.eye -= self.get_right() * dx * sensitivity;
        self.eye += self.get_up() * dy * sensitivity;
    }

    // Blender 缩放：滚轮
    pub fn zoom(&mut self, delta: f32) {
        self.eye += self.get_forward() * delta * 0.5;
    }

    // 根据鼠标屏幕坐标生成世界空间射线
    pub fn get_ray(&self, mouse_x: f32, mouse_y: f32, width: u32, height: u32) -> (glam::Vec3, glam::Vec3) {
        let aspect_ratio = width as f32 / height as f32;
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 1000.0);
        let view = self.get_view_matrix();
        let inv_vp = (proj * view).inverse();

        // NDC 坐标
        let x = (2.0 * mouse_x / width as f32) - 1.0;
        let y = 1.0 - (2.0 * mouse_y / height as f32);

        // 射线起点和方向
        let near_point = inv_vp * glam::Vec4::new(x, y, 0.0, 1.0);
        let far_point = inv_vp * glam::Vec4::new(x, y, 1.0, 1.0);

        let ray_origin = near_point.xyz() / near_point.w;
        let ray_direction = (far_point.xyz() / far_point.w - ray_origin).normalize();

        (ray_origin, ray_direction)
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ModelManifestItem {
    pub id: u32,
    pub name: String,
    pub file: String,
    pub default_scale: [f32; 3],
}

#[derive(Deserialize, Debug)]
pub struct ModelManifest {
    pub models: Vec<ModelManifestItem>,
}

#[derive(Clone)]
pub struct ModelRegistryItem {
    pub info: ModelManifestItem,
    pub tri_start: u32,
    pub tri_count: u32,
    pub material_id: u32,
    pub material_color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4], // 64 字节
    pub model_id: u32,               // 4 字节
    pub instance_id: u32,            // 4 字节
    pub tri_start: u32,              // 4 字节 【新增：模型三角形起始位置】
    pub bvh_start: u32,              // 4 字节 【新增：模型 BVH 起始位置】
    pub material_color: [f32; 4],    // 16 字节，xyz = 近似材质色
    pub _pad: [u32; 8],              // 32 字节填充。总计 128 字节
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpPixel {
    pub src_x: u32,
    pub src_y: u32,
    pub tri_id: u32,
    pub flags: u32,
    pub barycentric: [f32; 4],
}

/// Ap1 包络顶点：每个三角形顶点展开为 position + normal
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvelopeVertex {
    pub pos: [f32; 4],   // xyz = 位置, w = padding
    pub normal: [f32; 4], // xyz = 法线, w = padding
}

/// BVH 节点结构
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BVHNode {
    pub aabb_min: [f32; 4], // xyz = min bounds, w = left_child (internal) or tri_start (leaf)
    pub aabb_max: [f32; 4], // xyz = max bounds, w = right_child (internal) or negative tri_count (leaf)
}

/// 场景序列化：一个实例
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SceneInstance {
    pub model_id: u32,
    pub instance_id: u32,
    pub model_matrix: [[f32; 4]; 4],
}

/// 场景序列化：整个场景
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SceneData {
    pub instances: Vec<SceneInstance>,
}