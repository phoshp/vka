#![allow(unused)]

use ash::vk;
use vka;
use vka::RenderingDeviceInfo;

pub fn main() -> vka::VkaResult<()> {
    env_logger::init();
    let rd = vka::RenderingDevice::new(&RenderingDeviceInfo::default().with_gpu_validation())?;

    let buffer = rd.buffer_uniform(16 * 1024)?;
    let image = rd.image_2d(vk::Format::R8G8B8A8_UNORM, 256, 256, 1, 1, vk::SampleCountFlags::TYPE_1, vk::ImageUsageFlags::SAMPLED)?;
    let img_view = rd.image_full_view(&image);
    let sampler = rd.sampler_nearest(vk::SamplerAddressMode::REPEAT);

    rd.acquire_swapchain_image();
    rd.record(|dev, cmd, frame| unsafe {
        rd.barrier_image(cmd, &image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        dev.cmd_clear_color_image(
            cmd,
            image.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] },
            &[image.full_range()],
        );
        rd.barrier_image(cmd, &image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    });
    rd.submit()?;
    rd.present()
}
