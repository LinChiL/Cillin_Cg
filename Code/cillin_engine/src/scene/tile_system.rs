use wgpu::{Device, Queue, Buffer};
use glam::{Vec3, Mat4, Vec4Swizzles};
use crate::scene::{Entity, TileData, MAX_ENTITIES_PER_TILE, EntityData}; // 这里加入了 EntityData
use crate::models::ModelAsset;

pub struct TileSystem {
    pub tile_buffer: Buffer,
    tile_map_cache: Vec<TileData>,
    max_tiles_x: u32,
    max_tiles_y: u32,
    last_width: u32,
    last_height: u32,
}

impl TileSystem {
    pub fn new(device: &Device) -> Self {
        let max_tiles_x: u32 = (3840 + 15) / 16;
        let max_tiles_y: u32 = (2160 + 15) / 16;
        let tile_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Storage Buffer"),
            size: (max_tiles_x as u64) * (max_tiles_y as u64) * (std::mem::size_of::<TileData>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            tile_buffer,
            tile_map_cache: Vec::new(),
            max_tiles_x,
            max_tiles_y,
            last_width: 0,
            last_height: 0,
        }
    }

    pub fn update_tile_buffer(
        &mut self,
        entities: &[Entity],
        models: &std::collections::HashMap<u32, ModelAsset>,
        cam_pos: Vec3,
        view: Mat4,
        proj: Mat4,
        queue: &Queue,
        screen_width: u32,
        screen_height: u32,
    ) -> Vec<EntityData> {
        let tiles_x = (screen_width + 15) / 16;
        let tiles_y = (screen_height + 15) / 16;
        let total_tiles = (tiles_x * tiles_y) as usize;

        if self.last_width != screen_width || self.last_height != screen_height {
            self.tile_map_cache.resize(total_tiles, TileData { count: 0, _padding: [0; 3], entity_indices: [0; MAX_ENTITIES_PER_TILE] });
            self.last_width = screen_width;
            self.last_height = screen_height;
        } else {
            if self.tile_map_cache.len() != total_tiles {
                self.tile_map_cache.resize(total_tiles, TileData { count: 0, _padding: [0; 3], entity_indices: [0; MAX_ENTITIES_PER_TILE] });
            }
            self.tile_map_cache.fill(TileData { count: 0, _padding: [0; 3], entity_indices: [0; MAX_ENTITIES_PER_TILE] });
        }

        let vp = proj * view;
        
        // --- 核心修复步骤 1：先计算所有实体的渲染数据 ---
        // 我们必须保持这个顺序，因为 GPU 端的存储 Buffer 是按这个索引读的
        let mut entity_datas = Vec::with_capacity(entities.len());
        for entity in entities {
            let model = match models.get(&entity.model_id) {
                Some(m) => m,
                None => {
                    // 如果模型不存在，创建一个默认的 EntityData
                    entity_datas.push(entity.to_entity_data(entity.model_id, [0.0; 4], [0.0; 4], [0.0; 4], 0));
                    continue;
                }
            };
            let model_mat = Mat4::from_cols_array_2d(&entity.get_model_matrix());
            
            let mut tight_min = Vec3::new(f32::MAX, f32::MAX, 0.0);
            let mut tight_max = Vec3::new(f32::MIN, f32::MIN, 0.0);
            let mut is_any_corner_in_front = false;

            let corners = [
                Vec3::new(model.aabb_min[0], model.aabb_min[1], model.aabb_min[2]),
                Vec3::new(model.aabb_min[0], model.aabb_min[1], model.aabb_max[2]),
                Vec3::new(model.aabb_min[0], model.aabb_max[1], model.aabb_min[2]),
                Vec3::new(model.aabb_min[0], model.aabb_max[1], model.aabb_max[2]),
                Vec3::new(model.aabb_max[0], model.aabb_min[1], model.aabb_min[2]),
                Vec3::new(model.aabb_max[0], model.aabb_min[1], model.aabb_max[2]),
                Vec3::new(model.aabb_max[0], model.aabb_max[1], model.aabb_min[2]),
                Vec3::new(model.aabb_max[0], model.aabb_max[1], model.aabb_max[2]),
            ];

            for corner in corners {
                let p_world = model_mat.transform_point3(corner);
                if let Some(pixel) = self.project_to_pixel(p_world, vp, screen_width, screen_height) {
                    tight_min = tight_min.min(pixel);
                    tight_max = tight_max.max(pixel);
                    is_any_corner_in_front = true;
                }
            }

            // 投影保护：如果物体完全在相机后，给一个无效但安全的矩形
            let screen_rect = if is_any_corner_in_front {
                [
                    (tight_min.x / screen_width as f32).clamp(0.0, 1.0),
                    (tight_min.y / screen_height as f32).clamp(0.0, 1.0),
                    (tight_max.x / screen_width as f32).clamp(0.0, 1.0),
                    (tight_max.y / screen_height as f32).clamp(0.0, 1.0),
                ]
            } else {
                [0.0, 0.0, 0.0, 0.0] // 彻底不显示
            };
            
            let area = (tight_max.x - tight_min.x) * (tight_max.y - tight_min.y);
            let flags = if area < 400.0 { 1u32 } else { 0u32 };

            entity_datas.push(entity.to_entity_data(entity.model_id, model.aabb_min, model.aabb_max, screen_rect, flags));
        }

        // --- 核心修复步骤 2：对索引进行深度排序 ---
        // 我们需要知道哪些物体离摄像机更近
        let mut sorted_indices: Vec<usize> = (0..entities.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            let da = (entities[a].position - cam_pos).length_squared();
            let db = (entities[b].position - cam_pos).length_squared();
            // 注意：从小到大排（近的在前）
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- 核心修复步骤 3：按排序后的顺序填充 Tile ---
        let light_dir = Vec3::new(0.5, 1.0, 0.5).normalize();
        let shadow_ext = -light_dir * 50.0;

        for &idx in &sorted_indices {
            let entity = &entities[idx];
            let model = match models.get(&entity.model_id) {
                Some(m) => m,
                None => continue,
            };
            let model_mat = Mat4::from_cols_array_2d(&entity.get_model_matrix());
            
            let mut shadow_min = Vec3::new(f32::MAX, f32::MAX, 0.0);
            let mut shadow_max = Vec3::new(f32::MIN, f32::MIN, 0.0);
            let mut is_visible = false;

            let corners = [
                Vec3::new(model.aabb_min[0], model.aabb_min[1], model.aabb_min[2]),
                Vec3::new(model.aabb_min[0], model.aabb_min[1], model.aabb_max[2]),
                Vec3::new(model.aabb_min[0], model.aabb_max[1], model.aabb_min[2]),
                Vec3::new(model.aabb_min[0], model.aabb_max[1], model.aabb_max[2]),
                Vec3::new(model.aabb_max[0], model.aabb_min[1], model.aabb_min[2]),
                Vec3::new(model.aabb_max[0], model.aabb_min[1], model.aabb_max[2]),
                Vec3::new(model.aabb_max[0], model.aabb_max[1], model.aabb_min[2]),
                Vec3::new(model.aabb_max[0], model.aabb_max[1], model.aabb_max[2]),
            ];

            for corner in corners {
                let p_world = model_mat.transform_point3(corner);
                // 本体投影
                if let Some(pixel) = self.project_to_pixel(p_world, vp, screen_width, screen_height) {
                    shadow_min = shadow_min.min(pixel);
                    shadow_max = shadow_max.max(pixel);
                    is_visible = true;
                }
                // 影子投影
                if let Some(pixel) = self.project_to_pixel(p_world + shadow_ext, vp, screen_width, screen_height) {
                    shadow_min = shadow_min.min(pixel);
                    shadow_max = shadow_max.max(pixel);
                }
            }

            if !is_visible { continue; }

            let sw = screen_width as f32;
            let sh = screen_height as f32;
            let ts_x = ((shadow_min.x.clamp(0.0, sw) / 16.0).floor() as i32).clamp(0, tiles_x as i32 - 1);
            let te_x = ((shadow_max.x.clamp(0.0, sw) / 16.0).ceil() as i32).clamp(0, tiles_x as i32 - 1);
            let ts_y = ((shadow_min.y.clamp(0.0, sh) / 16.0).floor() as i32).clamp(0, tiles_y as i32 - 1);
            let te_y = ((shadow_max.y.clamp(0.0, sh) / 16.0).ceil() as i32).clamp(0, tiles_y as i32 - 1);

            for ty in ts_y..=te_y {
                let row_offset = (ty as usize).wrapping_mul(tiles_x as usize);
                for tx in ts_x..=te_x {
                    let t_idx = row_offset.wrapping_add(tx as usize);
                    if t_idx < total_tiles {
                        let count = self.tile_map_cache[t_idx].count as usize;
                        if count < MAX_ENTITIES_PER_TILE {
                            // 存入的是物体在 entities 原数组中的原始索引
                            self.tile_map_cache[t_idx].entity_indices[count] = idx as u32;
                            self.tile_map_cache[t_idx].count += 1;
                        }
                    }
                }
            }
        }

        queue.write_buffer(&self.tile_buffer, 0, bytemuck::cast_slice(&self.tile_map_cache));
        entity_datas
    }

    fn project_to_pixel(&self, world_p: Vec3, vp: Mat4, width: u32, height: u32) -> Option<Vec3> {
        let clip = vp * world_p.extend(1.0);
        if clip.w <= 0.0 { return None; }
        let ndc = clip.xyz() / clip.w as f32;
        Some(Vec3::new(
            (ndc.x + 1.0) * 0.5 * width as f32,
            (1.0 - ndc.y) * 0.5 * height as f32,
            0.0,
        ))
    }

    pub fn resize_if_needed(&mut self, device: &Device, new_width: u32, new_height: u32) -> bool {
        let new_tiles_x = (new_width + 15) / 16;
        let new_tiles_y = (new_height + 15) / 16;
        
        if new_tiles_x > self.max_tiles_x || new_tiles_y > self.max_tiles_y {
            self.max_tiles_x = new_tiles_x.max(self.max_tiles_x);
            self.max_tiles_y = new_tiles_y.max(self.max_tiles_y);
            let size = (self.max_tiles_x as u64) * (self.max_tiles_y as u64) * (std::mem::size_of::<TileData>() as u64);
            self.tile_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tile Storage Buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            return true;
        }
        false
    }

    pub fn get_vram_usage(&self) -> u64 {
        let tile_stride = 132u64 * 4;
        let total_tiles = self.max_tiles_x as u64 * self.max_tiles_y as u64;
        total_tiles * tile_stride
    }
}