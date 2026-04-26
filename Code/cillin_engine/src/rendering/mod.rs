use wgpu::{Device, Queue, Surface, TextureView, CommandEncoder, RenderPass, BindGroup, RenderPipeline, ComputePipeline};
use glam::Mat4;
use bytemuck::{Pod, Zeroable};

pub mod pipeline;
pub use pipeline::{VoxelRenderer, Params};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct LightUniform {
    pub direction: [f32; 3],
    pub _padding: u32,
    pub color: [f32; 4],
}

pub struct RenderContext<'a> {
    pub surface: Surface<'a>,
    pub device: Device,
    pub queue: Queue,
    pub render_pipeline: RenderPipeline,
    pub compute_pipeline: Option<ComputePipeline>,
    pub camera_bind_group: BindGroup,
    pub light_bind_group: BindGroup,
    pub main_bind_group: Option<BindGroup>,
    pub depth_texture: wgpu::Texture,
    pub depth_view: TextureView,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub format: wgpu::TextureFormat,
    pub current_view: Mat4,
    pub current_proj: Mat4,
}

impl<'a> RenderContext<'a> {
    pub fn new(
        surface: Surface<'a>,
        device: Device,
        queue: Queue,
        render_pipeline: RenderPipeline,
        compute_pipeline: Option<ComputePipeline>,
        camera_bind_group: BindGroup,
        light_bind_group: BindGroup,
        main_bind_group: Option<BindGroup>,
        depth_texture: wgpu::Texture,
        depth_view: TextureView,
        size: winit::dpi::PhysicalSize<u32>,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            render_pipeline,
            compute_pipeline,
            camera_bind_group,
            light_bind_group,
            main_bind_group,
            depth_texture,
            depth_view,
            size,
            format,
            current_view: Mat4::from_cols_array_2d(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            current_proj: Mat4::from_cols_array_2d(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.surface.configure(&self.device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: new_size.width,
            height: new_size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: new_size.width,
                height: new_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_texture = depth_texture;
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub fn begin_render_pass<'b>(
        &'b self,
        encoder: &'b mut CommandEncoder,
        view: &'b TextureView,
    ) -> RenderPass<'b> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }
}
