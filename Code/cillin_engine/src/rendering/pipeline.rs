use wgpu::{Device, Queue, Buffer, ComputePipeline, RenderPipeline, BindGroup, BindGroupLayout, Texture, TextureView};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::scene::EntityData;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Params {
    pub view_inv: [[f32; 4]; 4],       // 64
    pub proj_inv: [[f32; 4]; 4],       // 64
    pub prev_view_proj: [[f32; 4]; 4],  // 64
    
    pub cam_pos: [f32; 4],             // 16
    pub light_dir: [f32; 4],           // 16
    
    // --- 逻辑数据包 A ---
    pub entity_count: u32,             // 4
    pub debug_mode: u32,               // 4
    pub time: f32,                     // 4
    pub frame_index: u32,              // 4
    
    // --- 逻辑数据包 B ---
    pub is_moving: u32,                // 4
    pub _unused_pad: [u32; 3],         // 12 (补齐到 16)
    
    // --- 最终填充：将结构体推到 384 字节 (64的倍数，最稳) ---
    pub _final_padding: [u32; 32],     // 128
}

pub struct VoxelRenderer {
    pub compute_pipeline: ComputePipeline,
    pub render_pipeline: RenderPipeline,
    pub compute_bind_group_layout: BindGroupLayout,
    pub blit_bind_group_layout: BindGroupLayout,
    pub compute_bind_group: BindGroup,
    pub blit_bind_group: BindGroup,
    pub output_texture: Texture,
    pub output_texture_view: TextureView,
    pub entity_list_buffer: Buffer,
    pub params_buffer: Buffer,
    pub tile_buffer: Buffer,
    pub query_set: wgpu::QuerySet,
    pub query_buffer: Buffer,
    pub mapped_buffer: Buffer,
    pub last_compute_time: f32,
    pub last_render_time: f32,
    pub frame_count: u32,
    pub is_perf_mode: bool,
    pub start_time: std::time::Instant,
}

impl VoxelRenderer {
    pub fn new(
        device: &Device,
        _queue: &Queue,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        palette_buffer: &Buffer,
    ) -> Self {
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Brain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Eye Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Xi-Luoer Brain Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: shader,
            entry_point: "cs_main",
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Display Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_blit",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_blit",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let entity_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entity List Buffer"),
            size: (1024 * 256) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Buffer"),
            size: ((3840 + 15) / 16 * (2160 + 15) / 16 * 132 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 4,
            ty: wgpu::QueryType::Timestamp,
            label: Some("Perf Query Set"),
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mapped_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mapped Query Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: entity_list_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: palette_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: tile_buffer.as_entire_binding() },
            ],
            label: Some("Compute Bind Group"),
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_texture_view) },
            ],
            label: Some("Blit Bind Group"),
        });

        Self {
            compute_pipeline,
            render_pipeline,
            compute_bind_group_layout,
            blit_bind_group_layout,
            compute_bind_group,
            blit_bind_group,
            output_texture,
            output_texture_view,
            entity_list_buffer,
            params_buffer,
            tile_buffer,
            query_set,
            query_buffer,
            mapped_buffer,
            last_compute_time: 0.0,
            last_render_time: 0.0,
            frame_count: 0,
            is_perf_mode: false,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn resize_output(&mut self, device: &Device, width: u32, height: u32) {
        self.output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.output_texture_view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.output_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.entity_list_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.params_buffer.as_entire_binding() },
            ],
            label: Some("Compute Bind Group"),
        });

        self.blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.output_texture_view) },
            ],
            label: Some("Blit Bind Group"),
        });
    }

    pub fn update_entity_buffer(&self, queue: &Queue, entities: &[EntityData]) {
        queue.write_buffer(&self.entity_list_buffer, 0, bytemuck::cast_slice(entities));
    }

    pub fn update_params(&self, queue: &Queue, view_inv: &[[f32; 4]; 4], proj_inv: &[[f32; 4]; 4], cam_pos: Vec3, light_dir: Vec3, entity_count: u32, debug_mode: u32, time: f32, frame_index: u32, prev_view_proj: &[[f32; 4]; 4], is_moving: u32) {
        let params = Params {
            view_inv: *view_inv,
            proj_inv: *proj_inv,
            prev_view_proj: *prev_view_proj,
            cam_pos: cam_pos.extend(1.0).to_array(),
            light_dir: light_dir.normalize().extend(0.0).to_array(),
            entity_count,
            debug_mode,
            time,
            frame_index,
            is_moving,
            _unused_pad: [0, 0, 0],
            _final_padding: [0; 32],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    pub fn process_query_results(&mut self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.resolve_query_set(&self.query_set, 0..4, &self.query_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.mapped_buffer, 0, 32);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.mapped_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            if timestamps.len() >= 4 {
                let compute_time_ns = timestamps[1] - timestamps[0];
                let render_time_ns = timestamps[3] - timestamps[2];
                self.last_compute_time = compute_time_ns as f32 / 1_000_000.0;
                self.last_render_time = render_time_ns as f32 / 1_000_000.0;
                self.frame_count += 1;
            }
        }

        self.mapped_buffer.unmap();
    }

    pub fn print_perf_stats(&self) {
        if self.frame_count % 10 == 0 {
            println!("--- Cillin Engine Perf Stats ---");
            println!("GPU Brain:   {:.3} ms (Xi-Luoer Compute)", self.last_compute_time);
            println!("GPU Eye:     {:.3} ms (Blit & UI)", self.last_render_time);
            println!("Total Frame: {:.2} FPS", 1000.0 / (self.last_compute_time + self.last_render_time));
            println!("-------------------------------");
        }
    }
}