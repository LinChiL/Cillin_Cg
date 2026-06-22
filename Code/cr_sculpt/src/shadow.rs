pub struct ShadowSystem {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    width: u32,
    height: u32,
}

pub struct ShadowInputs<'a> {
    pub params_buffer: &'a wgpu::Buffer,
    pub triangle_buffer: &'a wgpu::Buffer,
    pub instance_buffer: &'a wgpu::Buffer,
    pub warp_buffer: &'a wgpu::Buffer,
    pub tri_id_view: &'a wgpu::TextureView,
    pub world_pos_view: &'a wgpu::TextureView,
    pub normal_view: &'a wgpu::TextureView,
}

impl ShadowSystem {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        width: u32,
        height: u32,
        inputs: ShadowInputs<'_>,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pixel Shadow Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pixel Shadow Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pixel Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pixel Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "cs_shadow_trace",
        });

        let bind_group = Self::create_bind_group(device, &bind_group_layout, &view, inputs);

        Self {
            texture,
            view,
            bind_group_layout,
            bind_group,
            pipeline,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, inputs: ShadowInputs<'_>) {
        self.width = width;
        self.height = height;
        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pixel Shadow Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group = Self::create_bind_group(device, &self.bind_group_layout, &self.view, inputs);
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Pixel Shadow Trace"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        shadow_view: &wgpu::TextureView,
        inputs: ShadowInputs<'_>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pixel Shadow Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: inputs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inputs.triangle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(inputs.tri_id_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(inputs.world_pos_view) },
                wgpu::BindGroupEntry { binding: 9, resource: inputs.instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: inputs.warp_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(inputs.normal_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(shadow_view) },
            ],
        })
    }
}
