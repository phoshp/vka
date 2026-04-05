#![allow(unused)]
use ash::vk;
pub use vka::*;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let rd = RenderingDevice::new(&RenderingDeviceInfo::default().with_gpu_validation())?;
    let color_image = rd.image_create(&ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 1024, 1024).usage(vk::ImageUsageFlags::COLOR_ATTACHMENT))?;

    let rpass = rd.render_pass_create(&vka::RenderPassDesc {
        extent: vk::Extent2D { width: 1024, height: 1024 },
        attachments: &[vka::Attachment {
            image: &color_image,
            ops: vka::Operations::Color {
                load: vka::LoadOp::Load,
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
        rd.begin_render_pass(cmd, &vka::RenderPassBeginDesc {
            pass: &rpass,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 1024,
                    height: 1024,
                }
            },
            clear_values: &[]
        });

        // Hmmm...

        rd.end_render_pass(cmd, &rpass);
    });

    Ok(())
}
