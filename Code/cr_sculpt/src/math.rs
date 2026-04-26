

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub inv_model_matrix: [[f32; 4]; 4],
    pub color: [f32; 4],                
    pub params: [f32; 4], // [x:半径/尺寸, y:高度/圆角, z:平滑度, w:类型ID]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Anchor {
    pub pos: [f32; 4],          // [x, y, z, 影响半径]
    pub offset_attr: [f32; 4],   // [SDF偏移值, R, G, B]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridCell {
    pub offset: u32, // 该格子在排序后的锚点数组中的起点
    pub count: u32,  // 该格子里有多少个锚点
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub view_inv: [[f32; 4]; 4],        // 64
    pub proj_inv: [[f32; 4]; 4],        // 64
    pub prev_view_proj: [[f32; 4]; 4],   // 64 (重投影矩阵)
    pub cam_pos: [f32; 4],              // 16
    pub light_dir: [f32; 4],            // 16
    
    // 数据包 A
    pub prim_count: u32,
    pub anchor_count: u32,
    pub scaffold_count: u32,
    pub is_moving: u32,                 // 16
    
    // --- 新增：空间网格参数 ---
    pub grid_origin: [f32; 4], // 网格左下角起点 [x, y, z, cell_size]
    
    pub time: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,                     // 16

    // 最终对齐补丁：补齐到 320 字节 (还需要 64 字节)
    pub _final_padding: [[f32; 4]; 4],  // 64

    // 模型几何中心 (用于径向位移场)
    pub model_center: [f32; 4],
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