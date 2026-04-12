#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("cpp/sdf/sdf_logic.hpp");

        // 新增支持 AABB 的烘焙函数
        fn generate_sdf_baked_aabb(
            vertices: &[f32],
            indices: &[u32],
            res: i32,
            min_x: f32, min_y: f32, min_z: f32,
            max_x: f32, max_y: f32, max_z: f32,
        ) -> Vec<f32>;

        fn merge_global_sdf(
            entities: &[EntitySdfInfo],
            global_res: i32,
            world_box_size: f32
        ) -> Vec<f32>;
    }

    #[derive(Debug)]
    pub struct EntitySdfInfo {
        pub inv_matrix: [f32; 16],
        pub sdf_ptr: *const u32,
        pub res: i32,
    }
}

pub mod camera;
pub mod console;
pub mod input;
pub mod models;
pub mod rendering;
pub mod scene;
pub mod undo;
