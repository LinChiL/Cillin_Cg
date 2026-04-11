use glam::{Vec3, Quat, Mat4};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityData {
    pub model_matrix: [[f32; 4]; 4],     // 64
    pub inv_model_matrix: [[f32; 4]; 4], // 64
    pub base_color: [f32; 4],           // 16
    pub aabb_min: [f32; 4],             // 16
    pub aabb_max: [f32; 4],             // 16
    pub sdf_index: u32,                 // 4
    pub instance_scale: f32,            // 4
    // 修改：补齐到 256 字节 (256 - 184 = 72字节 = 18个u32)
    pub _padding: [u32; 18],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: usize,
    pub model_id: u32,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub base_color: [f32; 4],
}

impl Entity {
    pub fn new(id: usize, model_id: u32, instance_index: usize, position: Vec3, rotation: Vec3, scale: Vec3, base_color: [f32; 4]) -> Self {
        let code = format!("{}s{:06}", model_id, instance_index);
        Self {
            id,
            model_id,
            position,
            rotation,
            scale,
            code,
            base_color,
        }
    }

    pub fn get_model_matrix(&self) -> [[f32; 4]; 4] {
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, 
            self.rotation.y, 
            self.rotation.x, 
            self.rotation.z
        );
        
        let matrix = Mat4::from_scale_rotation_translation(
            self.scale,
            rotation,
            self.position
        );
        
        matrix.to_cols_array_2d()
    }

    pub fn get_inv_model_matrix(&self) -> [[f32; 4]; 4] {
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, 
            self.rotation.y, 
            self.rotation.x, 
            self.rotation.z
        );
        
        let matrix = Mat4::from_scale_rotation_translation(
            self.scale,
            rotation,
            self.position
        );
        
        matrix.inverse().to_cols_array_2d()
    }

    pub fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model_matrix: self.get_model_matrix(),
        }
    }

    pub fn to_entity_data(&self, sdf_index: u32, aabb_min: [f32; 4], aabb_max: [f32; 4]) -> EntityData {
        let model_matrix = self.get_model_matrix();
        let inv_model_matrix = self.get_inv_model_matrix();

        EntityData {
            model_matrix,
            inv_model_matrix,
            base_color: self.base_color,
            aabb_min,
            aabb_max,
            sdf_index,
            instance_scale: self.scale.max_element(), // 缩放补偿系数
            _padding: [0; 18],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneData {
    pub entities: Vec<Entity>,
}

pub struct SceneManager {
    pub entities: Vec<Entity>,
    pub next_entity_id: usize,
    pub models: HashMap<String, super::models::ModelAsset>,
    pub instance_counters: HashMap<u32, usize>,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            next_entity_id: 0,
            models: HashMap::new(),
            instance_counters: HashMap::new(),
        }
    }

    pub fn add_entity(&mut self, model_id: u32, position: Vec3, rotation: Vec3, scale: Vec3, base_color: [f32; 4]) -> Entity {
        let instance_index = *self.instance_counters.get(&model_id).unwrap_or(&0);
        let entity = Entity::new(self.next_entity_id, model_id, instance_index, position, rotation, scale, base_color);
        self.entities.push(entity.clone());
        self.next_entity_id += 1;
        *self.instance_counters.entry(model_id).or_insert(0) += 1;
        entity
    }

    pub fn save_scene(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let scene_data = SceneData {
            entities: self.entities.clone(),
        };
        
        let json = serde_json::to_string_pretty(&scene_data)?;
        std::fs::write(file_path, json)?;
        
        Ok(())
    }

    pub fn load_scene(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(file_path)?;
        let scene_data: SceneData = serde_json::from_str(&json)?;
        
        // 重置实例计数器
        self.instance_counters.clear();
        
        // 为每个实体生成 code
        self.entities = scene_data.entities.into_iter().map(|mut entity| {
            if entity.code.is_empty() {
                let instance_index = *self.instance_counters.get(&entity.model_id).unwrap_or(&0);
                entity.code = format!("{}s{:06}", entity.model_id, instance_index);
                *self.instance_counters.entry(entity.model_id).or_insert(0) += 1;
            }
            entity
        }).collect();
        
        self.next_entity_id = self.entities.len();
        
        Ok(())
    }
}
