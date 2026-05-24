#![allow(unused, deprecated)]
use std::time::Duration;
use std::time::Instant;

use ash::vk;

pub use vka::*;
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::event::WindowEvent;
use winit::event_loop;
use winit::event_loop::EventLoop;
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::window::WindowAttributes;

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = event_loop.create_window(WindowAttributes::default().with_inner_size(PhysicalSize::new(800, 600))).unwrap();
    let rd = RenderingDevice::new(&RenderingDeviceDesc::with_window(&window).with_gpu_validation()).unwrap();
    let rpass = rd.new_render_pass(&vka::RenderPassDesc {
        attachments: &[vka::Attachment {
            format: vk::Format::B8G8R8A8_UNORM,
            samples: 1,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ops: vka::Operations::Color {
                load: vka::LoadOp::Clear(vka::color32(1.0, 1.0, 0.0, 1.0)),
                store: vka::StoreOp::Store,
            },
        }],
        subpasses: &[vka::Subpass {
            colors: &[(0, None)],
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        }],
    });

    let mut fps_timer = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    let mut surface = rd.new_surface(&window, SurfaceConfig::default());

    event_loop.run(|event, event_loop| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                rd.record_frame(&mut surface, |cmd, image| {
                    cmd.begin_render_pass(
                        &rpass,
                        &[image.inner.full_view()],
                        vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D {
                                width: image.inner.extent.width,
                                height: image.inner.extent.height,
                            },
                        },
                    );
                    cmd.end_render_pass();
                });
                rd.submit();
                surface.present();
                rd.advance_frame();

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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        },
        _ => (),
    });
}
