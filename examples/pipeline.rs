#![allow(unused, deprecated)]

use std::ffi::CString;
use std::time::Duration;
use std::time::Instant;

use ash::vk;
use glam::vec4;
use vka::*;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new()?;
    let window = event_loop.create_window(
        winit::window::WindowAttributes::default()
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600)),
    )?;
    let rd = RenderingDevice::new(&RenderingDeviceDesc::with_window(&window).with_gpu_validation())?;

    let rpass = rd.render_pass_create(&RenderPassDesc {
        attachments: &[Attachment {
            format: vk::Format::B8G8R8A8_UNORM,
            samples: 1,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: Some(vk::ImageLayout::PRESENT_SRC_KHR),
            ops: Operations::Color {
                load: LoadOp::Clear(vec4(1.0, 1.0, 0.0, 1.0)),
                store: StoreOp::Store,
            },
        }],
        subpasses: &[Subpass {
            colors: &[(0, None)],
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        }],
    });

    let layout = rd.pipeline_layout_create(&[])?;

    let vert_spv = include_bytes!("../shaders/triangle.vert.spv");
    let frag_spv = include_bytes!("../shaders/triangle.frag.spv");

    let vert_module = rd.shader_module_create(vert_spv)?;
    let frag_module = rd.shader_module_create(frag_spv)?;

    #[repr(C)]
    struct Vertex {
        pos: [f32; 4],
        color: [f32; 4],
    }
    let vertices = [
        Vertex { pos: [0.0, -0.5, 0.0, 1.0], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { pos: [0.5, 0.5, 0.0, 1.0], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { pos: [-0.5, 0.5, 0.0, 1.0], color: [0.0, 0.0, 1.0, 1.0] },
    ];
    let vertex_buf = rd.buffer_create(&BufferDesc::vertex((std::mem::size_of::<Vertex>() * 3) as u64))?;
    rd.write_buffer(&vertex_buf, &vertices, 0);
    rd.submit_wait()?;

    let attributes = vertex_attributes! {
        0 => vk::Format::R32G32B32A32_SFLOAT,
        1 => vk::Format::R32G32B32A32_SFLOAT
    };

    let vertex_input = &[VertexInputLayout {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        rate: vk::VertexInputRate::VERTEX,
        attributes: &attributes,
    }];

    let main_name = CString::new("main").unwrap();

    let pipeline = rd.graphics_pipeline_create(&GraphicsPipelineDesc {
        layout: &layout,
        stages: &[
            ShaderStage {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module.value,
                name: main_name.clone(),
            },
            ShaderStage {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_module.value,
                name: main_name,
            },
        ],
        vertex_input,
        input_assembly: InputAssemblyState {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
        },
        rasterization: RasterizationState::default(),
        multisample: MultisampleState::default(),
        depth_stencil: None,
        color_blend: ColorBlendState {
            attachments: &[COLOR_BLEND_ALPHA],
            ..Default::default()
        },
        render_pass: Some(&rpass),
        subpass: 0,
    })?;

    let mut fps_timer = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    event_loop.run(|event, event_loop| match event {
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::RedrawRequested => {
                let frame = rd.acquire_swapchain_image().unwrap();
                rd.record(|dev, cmd| unsafe {
                    let extent = rd.get_swapchain_extent();

                    rd.begin_render_pass(
                        cmd,
                        &rpass,
                        &[rd.image_full_view(&frame)],
                        vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent,
                        },
                    );
                    dev.cmd_set_viewport(cmd, 0, &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: extent.width as f32,
                        height: extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }]);
                    dev.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    }]);

                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    dev.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buf.handle], &[0]);
                    dev.cmd_draw(cmd, 3, 1, 0, 0);
                    rd.end_render_pass(cmd, &rpass);
                });
                rd.submit();
                rd.present();

                frame_count += 1;
                let elapsed = fps_timer.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    fps = frame_count as f64 / elapsed.as_secs_f64();
                    println!("FPS: {:.2}", fps);
                    frame_count = 0;
                    fps_timer = Instant::now();
                }

                window.request_redraw();
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        },
        _ => (),
    });

    Ok(())
}
