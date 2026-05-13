use ash::vk;
use vka;
use vka::RenderingDeviceDesc;

pub fn main() {
    env_logger::init();
    let rd = vka::RenderingDevice::new(&RenderingDeviceDesc::default().with_gpu_validation()).unwrap();

    let buffer = rd.new_buffer(&vka::BufferDesc::uniform(4 * 1024));
    let image = rd.new_image(&vka::ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 256, 256));

    rd.record(|cmd| {
        cmd.image_barrier_begin(&image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        cmd.clear_color_image(
            &image,
            vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] },
            &[image.full_range()],
        );
        cmd.image_barrier_end(&image);
    });
    rd.submit();

    rd.write_buffer(&buffer, &[1u8, 1u8, 1u8, 0u8, 0, 0], 0);
    rd.wait_queue();

    let mut read_data = [0; 6];
    rd.read_buffer(&buffer, &mut read_data, 0); // this is implicitly immediate.
    assert_eq!(read_data, [1, 1, 1, 0, 0, 0]);
    println!("Demo succeeded!");
}
