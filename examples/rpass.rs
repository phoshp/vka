#![allow(unused)]
use ash::vk;
use glam::vec4;
pub use vka::*;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let rd = RenderingDevice::new(&RenderingDeviceInfo::default().with_gpu_validation())?;
    let color_image = rd.image_create(&ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 128, 128).usage(vk::ImageUsageFlags::COLOR_ATTACHMENT))?;

    let rpass = rd.render_pass_create(&vka::RenderPassDesc {
        extent: vk::Extent2D { width: 128, height: 128 },
        attachments: &[vka::Attachment {
            image: &color_image,
            layout: None, // let it infer
            ops: vka::Operations::Color {
                load: vka::LoadOp::Clear(vec4(1.0, 0.0, 0.0, 1.0)),
                store: vka::StoreOp::Store,
            },
        }],
        subpasses: &[vka::SubpassDesc {
            colors: &[(0, None)],
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        }],
    });

    rd.record(|dev, cmd| {
        rd.begin_render_pass(cmd, &rpass, vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width: 128, height: 128 },
        });

        // Hmmm...

        rd.end_render_pass(cmd, &rpass);
    });
    rd.submit_wait()?;

    // read the rendered image back.
    let mut image_raw_data = [0u8; 128 * 128 * 4];
    rd.read_image(&color_image, &mut image_raw_data)?;

    let mut img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(128, 128, image_raw_data.as_mut()).unwrap();
    img.save("examples/rpass.png")?;
    println!("Saved rpass.png");

    Ok(())
}
