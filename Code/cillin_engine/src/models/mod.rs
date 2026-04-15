use wgpu::{Device, Queue, BindGroup, BindGroupLayout, Buffer};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::error::Error;
use std::fs::File;
use std::io::Read;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ModelInfo {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
}



pub struct ModelAsset {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub num_indices: u32,
    pub bind_group: BindGroup,
    
    // 持久化实例数据
    pub instance_buffer: Buffer,
    pub instance_capacity: usize,
    
    // SDF 3D 贴图
    pub sdf_view: wgpu::TextureView,
    
    // 关键改变：统一使用 u32 存储 DNA 数据
    pub dna_data: Vec<u32>,
    
    // 原始贴图数据（用于贴图阵列）
    pub albedo_data: Option<Vec<u8>>,
    
    // AABB 信息
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
}

pub fn load_glb(
    device: &Device,
    queue: &Queue,
    path: &str,
    layout: &BindGroupLayout,
) -> Result<ModelAsset, Box<dyn Error>> {
    // --- 核心逻辑：加载 .cem DNA ---
    let cem_path = path.replace(".glb", ".cem");
    let (dna_data, aabb_min, aabb_max) = if std::path::Path::new(&cem_path).exists() {
        println!("检测到全息 DNA 资产，正在加载: {}", cem_path);
        load_cem_data(&cem_path)
    } else {
        // 如果没有 .cem，我们降级生成一个空的
        println!("警告: 未找到 .cem 文件，该模型将无法在 DNA 模式下渲染!");
        (vec![0u32; 64*64*64], [0.0; 4], [0.0; 4])
    };

    // 创建默认纹理
    let texture_size = wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Default Texture"), 
        size: texture_size, 
        mip_level_count: 1, 
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2, 
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 
        view_formats: &[],
    });
    
    let mut pixels = vec![0u8; (256 * 256 * 4) as usize];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 4;
            if (x / 32 + y / 32) % 2 == 0 {
                pixels[idx] = 255;
                pixels[idx + 1] = 255;
                pixels[idx + 2] = 255;
                pixels[idx + 3] = 255;
            } else {
                pixels[idx] = 128;
                pixels[idx + 1] = 128;
                pixels[idx + 2] = 128;
                pixels[idx + 3] = 255;
            }
        }
    }
    
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * 256),
            rows_per_image: Some(256),
        },
        texture_size,
    );
    
    let diffuse_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Default Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // 创建一个占位的纹理视图
    let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1, 
        sample_count: 1, 
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let sdf_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("SDF Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&diffuse_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&sdf_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sdf_sampler) },
        ],
        label: None,
    });

    // 创建空的顶点和索引缓冲区
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Empty Vertex Buffer"),
        contents: &[],
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Empty Index Buffer"),
        contents: &[],
        usage: wgpu::BufferUsages::INDEX,
    });

    // 初始化实例缓冲区（初始容量为 16）
    let initial_capacity = 16;
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Instance Buffer"),
        size: (initial_capacity * std::mem::size_of::<super::scene::InstanceRaw>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Ok(ModelAsset {
        vertex_buffer,
        index_buffer,
        num_indices: 0,
        bind_group,
        instance_buffer,
        instance_capacity: initial_capacity,
        sdf_view,
        dna_data: Vec::new(), // 释放 CPU 端的 DNA 数据，已经上传到显存
        albedo_data: Some(pixels),
        aabb_min,
        aabb_max,
    })
}

pub fn load_cem_data(path: &str) -> (Vec<u32>, [f32; 4], [f32; 4]) {
    let mut file = File::open(path).expect("找不到 CEM 模型文件");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("读取文件失败");

    // 1. 验证魔数
    let magic = &buffer[0..4];
    let is_64bit = magic == b"CEM3";
    if !is_64bit && magic != b"CEM2" {
        panic!("模型格式不匹配！期望 CEM2 或 CEM3，得到 {:?}", magic);
    }

    // 2. 提取 AABB (12字节 min + 12字节 max)
    let min_x = f32::from_le_bytes(buffer[8..12].try_into().unwrap());
    let min_y = f32::from_le_bytes(buffer[12..16].try_into().unwrap());
    let min_z = f32::from_le_bytes(buffer[16..20].try_into().unwrap());
    
    let max_x = f32::from_le_bytes(buffer[20..24].try_into().unwrap());
    let max_y = f32::from_le_bytes(buffer[24..28].try_into().unwrap());
    let max_z = f32::from_le_bytes(buffer[28..32].try_into().unwrap());

    // 3. 提取 Voxel 数据
    let voxel_data_u32: Vec<u32> = if is_64bit {
        // CEM3 格式：64-bit 体素 (8 字节 per voxel)
        buffer[32..].chunks_exact(8)
            .flat_map(|chunk| {
                let r = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let g = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                [r, g].into_iter()
            })
            .collect()
    } else {
        // CEM2 格式：32-bit 体素 (4 字节 per voxel)
        buffer[32..].chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    };

    println!("已成功加载cem资产: {} | 数据量: {} | 格式: {}", path, voxel_data_u32.len(), if is_64bit { "CEM3 (64-bit)" } else { "CEM2 (32-bit)" });

    let min_b: [f32; 4] = [min_x, min_y, min_z, 1.0];
    let max_b: [f32; 4] = [max_x, max_y, max_z, 1.0];

    (voxel_data_u32, min_b, max_b)
}
