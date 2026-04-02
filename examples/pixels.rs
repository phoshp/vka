#![allow(unused)]
use ash::vk;
pub use vka::*;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let rd = RenderingDevice::new(&RenderingDeviceInfo::default().with_gpu_validation())?;
    let pixels = rd.image_create(
        &ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 1024, 1024)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .location(MemoryLocation::CpuToGpu)
    )?;
    pixels.set_name("pixels buffer image");

    rd.clear_color_image(&pixels, vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] }, pixels.full_range());
    rd.submit_wait()?;

    let mut data = pixels.mapped_mut().unwrap();
    let mut img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(1024, 1024, data.as_mut()).unwrap();
    img.save("examples/pixels.png")?;
    println!("Saved pixels.png");

    Ok(())
}
