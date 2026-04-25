#![allow(unused)]
use ash::vk;
pub use vka::*;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let rd = RenderingDevice::new(&RenderingDeviceDesc::default().with_gpu_validation())?;
    let pixels = rd.image_create(
        &ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 1024, 1024)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .location(MemoryLocation::GpuOnly),
    )?;
    pixels.set_name("pixels buffer image");

    rd.clear_color_image(&pixels, vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] }, pixels.full_range());
    rd.submit_wait()?;

    let mut data = vec![0u8; 32 * 32 * 4];
    rd.read_image(
        &pixels,
        &mut data,
        vk::Offset3D::default(),
        vk::Extent3D { width: 32, height: 32, depth: 1 },
        4,
        vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
    );
    let mut native_img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(32, 32, data.as_mut()).unwrap();
    native_img.save("examples/pixels.png")?;
    println!("Saved pixels.png");
    Ok(())
}
