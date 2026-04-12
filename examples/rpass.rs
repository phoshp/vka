#![allow(unused, deprecated)]
use std::time::Duration;
use std::time::Instant;

use ash::vk;
use glam::vec4;

pub use vka::*;
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::event::WindowEvent;
use winit::event_loop;
use winit::event_loop::EventLoop;
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::window::WindowAttributes;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = event_loop.create_window(WindowAttributes::default().with_inner_size(PhysicalSize::new(800, 600)))?;
    let rd = RenderingDevice::new(&RenderingDeviceDesc::with_window(&window).with_gpu_validation())?;
    let color_image = rd.image_create(&ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 128, 128).usage(vk::ImageUsageFlags::COLOR_ATTACHMENT))?;
    let rpass = rd.render_pass_create(&vka::RenderPassDesc {
        attachments: &[vka::Attachment {
            format: vk::Format::B8G8R8A8_UNORM,
            samples: 1,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: Some(vk::ImageLayout::PRESENT_SRC_KHR),
            ops: vka::Operations::Color {
                load: vka::LoadOp::Clear(vec4(1.0, 1.0, 0.0, 1.0)),
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

    event_loop.run(|event, event_loop| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                let frame = rd.acquire_swapchain_image().unwrap();
                rd.record(|dev, cmd| {
                    rd.begin_render_pass(
                        cmd,
                        &rpass,
                        &[rd.image_full_view(&frame)],
                        vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D { width: 800, height: 600 },
                        },
                    );
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        },
        _ => (),
    });

    //
    //
    //
    //
    // // read the rendered image back.
    // let mut image_raw_data = [0u8; 128 * 128 * 4];
    // rd.read_image(&color_image, &mut image_raw_data, 4)?;
    //
    // let mut img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(128, 128, image_raw_data.as_mut()).unwrap();
    // img.save("examples/rpass.png")?;
    // println!("Saved rpass.png");
    //
    Ok(())
}
