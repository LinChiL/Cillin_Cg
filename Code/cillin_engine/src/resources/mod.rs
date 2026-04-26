use std::{collections::HashMap, fs};
use wgpu::{Device, Queue, Texture, TextureView, Sampler};
use wgpu::util::DeviceExt;
use serde::{Serialize, Deserialize};
use crate::models::{self, ModelAsset, VoxelCEM4};

pub const MODEL_DIR: &str = "../../Asset/cemModel";
pub const SCENE_FILE: &str = "scene_data.json";
const CPAL_PATH: &str = "../../Asset/Global/master.cpal";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelEntry {
    pub id: u32,
    pub name: String,
    pub file: String,
    pub default_scale: [f32; 3],
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssetManifest {
    pub models: Vec<ModelEntry>,
}

pub struct AssetManager {
    pub manifest: HashMap<u32, ModelEntry>,
    pub models: HashMap<u32, ModelAsset>,
    pub global_sdf_texture: Texture,
    pub global_sdf_view: TextureView,
    pub global_sdf_sampler: Sampler,
    pub albedo_array: Texture,
    pub albedo_array_view: TextureView,
    pub albedo_sampler: Sampler,
    pub palette_buffer: wgpu::Buffer,
    pub model_count: u32,
}

impl AssetManager {
    pub async fn new(device: &Device, queue: &Queue) -> Self {
        let manifest_raw = fs::read_to_string(format!("{}/manifest.json", MODEL_DIR))
            .expect("Manifest.json 没找到");
        let manifest_data: AssetManifest = serde_json::from_str(&manifest_raw).unwrap();
        let mut manifest = HashMap::new();
        let mut loaded_models = HashMap::new();

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Uint }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
            ],
            label: Some("Texture Layout"),
        });

        for entry in manifest_data.models.clone() {
            manifest.insert(entry.id, entry.clone());
            
            let file_path = format!("{}/{}", MODEL_DIR, entry.file);
            
            if let Ok((voxel_data, aabb_min, aabb_max)) = std::fs::File::open(&file_path).map(|_| {
                models::load_cem_data(&file_path)
            }) {
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
                                format: wgpu::TextureFormat::Rgba32Uint,
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
                        format: wgpu::TextureFormat::Rgba32Uint,
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

        let model_count = loaded_models.len() as u32;
        let sdf_res = 64;

        let global_sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CEM4 Holographic Data"),
            size: wgpu::Extent3d {
                width: sdf_res,
                height: sdf_res,
                depth_or_array_layers: (manifest_data.models.len() as u32) * sdf_res * 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for entry in &manifest_data.models {
            if let Some(model) = loaded_models.get(&entry.id) {
                let index = entry.id;
                if model.dna_data.is_empty() {
                    println!("跳过模型 {} 的纹理上传，数据为空", entry.id);
                    continue;
                }

                let buffer_3d = Self::prepare_texture_data(&model.dna_data, sdf_res);
                
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &global_sdf_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: index * sdf_res * 2 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&buffer_3d),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(16 * sdf_res),
                        rows_per_image: Some(sdf_res),
                    },
                    wgpu::Extent3d { width: sdf_res, height: sdf_res, depth_or_array_layers: sdf_res * 2 },
                );
            }
        }

        let global_sdf_view = global_sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let global_sdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Global SDF Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let tex_size = 512;
        let albedo_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Texture Array"),
            size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: model_count },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for entry in &manifest_data.models {
            if let Some(model) = loaded_models.get(&entry.id) {
                let index = entry.id;
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
        }

        let albedo_array_view = albedo_array.create_view(&wgpu::TextureViewDescriptor::default());
        let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Albedo Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let palette_data = load_cpal_to_vec4(CPAL_PATH);
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Palette Buffer"),
            contents: bytemuck::cast_slice(&palette_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            manifest,
            models: loaded_models,
            global_sdf_texture,
            global_sdf_view,
            global_sdf_sampler,
            albedo_array,
            albedo_array_view,
            albedo_sampler,
            palette_buffer,
            model_count,
        }
    }

    pub fn prepare_texture_data(dna_data: &[VoxelCEM4], sdf_res: u32) -> Vec<[u32; 4]> {
        let mut buffer_3d = vec![[0u32; 4]; (sdf_res * sdf_res * sdf_res * 2) as usize];
        
        for z in 0..sdf_res {
            for y in 0..sdf_res {
                for x in 0..sdf_res {
                    let voxel_idx = x + y * sdf_res + z * sdf_res * sdf_res;
                    let v = &dna_data[voxel_idx as usize];
                    
                    buffer_3d[(x + y * sdf_res + (z * 2) * sdf_res * sdf_res) as usize] = 
                        [v.word0, v.word1, v.word2, v.word3];
                    buffer_3d[(x + y * sdf_res + (z * 2 + 1) * sdf_res * sdf_res) as usize] = 
                        [v.child_ptr, v.tetra_info, v.reserved[0], v.reserved[1]];
                }
            }
        }
        
        buffer_3d
    }

    pub fn get_vram_usage(&self, output_width: u32, output_height: u32) -> (u64, u64) {
        let mut total_allocated = 0u64;

        let dna_real_size = self.model_count as u64 * 64 * 64 * 128 * 16;
        total_allocated += dna_real_size;
        
        let albedo_size = self.albedo_array.size();
        total_allocated += (albedo_size.width * albedo_size.height * albedo_size.depth_or_array_layers) as u64 * 4;
        
        let output_size = (output_width as u64) * (output_height as u64) * 4;
        total_allocated += output_size;
        
        total_allocated += self.palette_buffer.size();
        
        let driver_overhead = total_allocated / 4;
        
        (total_allocated, total_allocated + driver_overhead)
    }
}

pub fn load_cpal_to_vec4(file_path: &str) -> Vec<[f32; 4]> {
    let data = fs::read(file_path).expect("读取色板失败");
    let mut colors = Vec::with_capacity(1024);
    
    let color_data = &data[8..];
    
    for chunk in color_data.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        colors.push([r, g, b, 1.0]);
    }
    
    while colors.len() < 1024 {
        colors.push([0.0, 0.0, 0.0, 1.0]);
    }
    colors
}