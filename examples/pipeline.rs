#![allow(unused, deprecated)]

use std::ffi::CString;
use std::io::Cursor;
use std::time::Duration;
use std::time::Instant;

use ash::vk;
use vka::*;

pub fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = event_loop
        .create_window(winit::window::WindowAttributes::default().with_inner_size(winit::dpi::PhysicalSize::new(800, 600)))
        .unwrap();
    window.set_resizable(false);
    let rd = RenderingDevice::new(&RenderingDeviceDesc::with_window(&window)).unwrap();
    let mut color_image = rd.new_image(
        &ImageDesc::new_2d(vk::Format::B8G8R8A8_UNORM, 800, 600)
            .samples(4)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT),
    );

    let rpass = rd.new_render_pass(&RenderPassDesc {
        attachments: &[
            Attachment {
                format: color_image.format,
                samples: color_image.samples.as_raw(),
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ops: Operations::Color {
                    load: LoadOp::Clear(vka::color32(0.0, 1.0, 1.0, 0.5)),
                    store: StoreOp::Discard,
                },
            },
            Attachment {
                format: vk::Format::B8G8R8A8_UNORM,
                samples: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ops: Operations::Color {
                    load: LoadOp::Discard,
                    store: StoreOp::Store,
                },
            },
        ],
        subpasses: &[Subpass {
            colors: &[(0, Some(1))],
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        }],
    });

    let layout = rd.new_pipeline_layout(&[]);

    let vert_spv = include_bytes!("../shaders/triangle.vert.spv");
    let frag_spv = include_bytes!("../shaders/triangle.frag.spv");

    let vert_module = rd.new_shader(&ash::util::read_spv(&mut Cursor::new(vert_spv)).unwrap());
    let frag_module = rd.new_shader(&ash::util::read_spv(&mut Cursor::new(frag_spv)).unwrap());

    #[repr(C)]
    struct Vertex {
        pos: [f32; 4],
        color: [f32; 4],
    }
    let vertices = [
        Vertex {
            pos: [0.0, -0.5, 0.0, 1.0],
            color: [1.0, 0.0, 0.0, 0.5],
        },
        Vertex {
            pos: [0.5, 0.5, 0.0, 1.0],
            color: [0.0, 1.0, 0.0, 0.5],
        },
        Vertex {
            pos: [-0.5, 0.5, 0.0, 1.0],
            color: [0.0, 0.0, 1.0, 0.5],
        },
    ];
    let vertex_buf = rd.new_buffer(&BufferDesc::vertex((std::mem::size_of::<Vertex>() * 3) as u64));
    rd.write_buffer(&vertex_buf, &vertices, 0);
    rd.submit();
    rd.wait_queue();

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

    let pipeline = rd.new_render_pipeline(&vka::RenderPipelineDesc {
        layout: &layout,
        stages: &[
            ShaderStage {
                stage: vk::ShaderStageFlags::VERTEX,
                module: &vert_module,
                name: &main_name,
            },
            ShaderStage {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: &frag_module,
                name: &main_name,
            },
        ],
        vertex_input,
        input_assembly: InputAssemblyState {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
        },
        rasterization: RasterizationState::default(),
        multisample: MultisampleState {
            sample_count: color_image.samples.as_raw(),
            ..Default::default()
        },
        depth_stencil: None,
        color_blend: ColorBlendState {
            attachments: &[COLOR_BLEND_REPLACE],
            ..Default::default()
        },
        render_pass: Some(&rpass),
        subpass: 0,
    });

    let mut fps_timer = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    let mut surface = rd.new_surface(&window, vka::SurfaceConfig::default());

    event_loop.run(|event, event_loop| match event {
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::RedrawRequested => {
                let extent = surface.swapchain.extent;
                rd.record_frame(&mut surface, |cmd, image| {
                    let views = [color_image.full_view(), image.inner.full_view()];
                    cmd.begin_render_pass(
                        &rpass,
                        &views,
                        vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent,
                        },
                    );
                    cmd.set_viewport(
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: extent.width as f32,
                            height: extent.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );
                    cmd.set_scissor(
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent,
                        }],
                    );
                    cmd.bind_pipeline(&pipeline);
                    cmd.bind_vertex_buffers(0, &[vertex_buf.raw], &[0]);
                    cmd.draw(3, 1, 0, 0);
                    cmd.end_render_pass();
                });
                rd.submit();
                surface.present();

                frame_count += 1;
                let elapsed = fps_timer.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    fps = frame_count as f64 / elapsed.as_secs_f64();
                    log::info!("FPS: {:.2}", fps);
                    frame_count = 0;
                    fps_timer = Instant::now();
                }

                window.request_redraw();
            }
            winit::event::WindowEvent::Resized(s) => {
                log::info!("Resized to {}x{}", s.width, s.height);
                surface.configure(vka::SurfaceConfig {
                    width: s.width,
                    height: s.height,
                    vsync: false,
                    frame_latency: 2,
                });
                color_image = rd.new_image(
                    &ImageDesc::new_2d(vk::Format::B8G8R8A8_UNORM, s.width, s.height)
                        .samples(4)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT),
                );
                window.request_redraw();
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        },
        _ => (),
    });
}
