

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub inv_model_matrix: [[f32; 4]; 4],
    pub color: [f32; 4],                
    pub params: [f32; 4], // [x:半径/尺寸, y:高度/圆角, z:平滑度, w:类型ID]
}

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
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub view_inv: [[f32; 4]; 4],      // 64 bytes
    pub proj_inv: [[f32; 4]; 4],      // 64 bytes
    pub prev_view_proj: [[f32; 4]; 4], // 64 bytes
    pub cam_pos: [f32; 4],            // 16 bytes
    pub light_dir: [f32; 4],          // 16 bytes
    
    // 数据包 A (16 bytes)
    pub prim_count: u32,      // 4
    pub anchor_count: u32,    // 4
    pub scaffold_count: u32,  // 4
    pub is_moving: u32,       // 4
    
    pub grid_origin: [f32; 4],  // 16 bytes
    
    // 数据包 B (16 bytes)
    pub time: f32,    // 4
    pub _pad1: u32,   // 4
    pub _pad2: u32,   // 4
    pub _pad3: u32,   // 4

    pub model_center: [f32; 4], // 16 bytes
    
    pub disk_center: [f32; 4],  // 16 bytes
    pub disk_radius: f32,       // 4
    pub base_radius: f32,       // 4
    pub debug_mode: u32,        // 4 - 0=正常, 1=圆盘调试, 2=圆球调试
    pub _padding: u32,          // 4 ← 确保总大小为 16 的倍数（400字节）
}

impl Default for Params {
    fn default() -> Self {
        Self {
            view_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0, 1.0, -5.0, 1.0],
            light_dir: [0.0, 1.0, 0.0, 0.0],
            prim_count: 0,
            anchor_count: 0,
            scaffold_count: 0,
            is_moving: 0,
            grid_origin: [-2.0, -2.0, -2.0, 4.0 / 16.0],
            time: 0.0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            model_center: [0.0, 0.0, 0.0, 1.0],
            disk_center: [0.0, 0.0, 0.0, 0.0],
            disk_radius: 0.0,
            base_radius: 1.0,
            debug_mode: 0,
            _padding: 0,
        }
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
}