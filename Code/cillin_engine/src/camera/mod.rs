use glam::{Vec3, Mat4};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: Vec3,
    pub yaw: f32,   // 左右看 (弧度)
    pub pitch: f32, // 上下看 (弧度)
}

impl Camera {
    pub fn new(eye: Vec3, yaw: f32, pitch: f32) -> Self {
        Self { eye, yaw, pitch }
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let forward = self.get_forward();
        Mat4::look_to_rh(self.eye, forward, Vec3::Y)
    }

    pub fn get_forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    pub fn get_right(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }
}

#[derive(Debug)]
pub struct CameraController {
    pub speed: f32,
    pub is_up_pressed: bool,
    pub is_down_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
    pub is_q_pressed: bool,
    pub is_e_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_q_pressed: false,
            is_e_pressed: false,
        }
    }

    pub fn update(&mut self, camera: &mut Camera, delta_time: f32) {
        let forward = camera.get_forward();
        let right = camera.get_right();
        let speed = self.speed * delta_time;

        if self.is_up_pressed {
            camera.eye += forward * speed;
        }
        if self.is_down_pressed {
            camera.eye -= forward * speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * speed;
        }
        if self.is_right_pressed {
            camera.eye += right * speed;
        }
        if self.is_q_pressed {
            camera.eye.y -= speed;
        }
        if self.is_e_pressed {
            camera.eye.y += speed;
        }
    }
}
