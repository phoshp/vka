#![allow(unused)]

use ash::vk;
use vka;
use vka::RenderingDeviceDesc;

pub fn main() -> vka::Result<()> {
    env_logger::init();
    let rd = vka::RenderingDevice::new(&RenderingDeviceDesc::default().with_gpu_validation())?;

    let buffer = rd.buffer_create(&vka::BufferDesc::uniform(4 * 1024))?;
    let image = rd.image_create(&vka::ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, 256, 256))?;
    let sampler = rd.sampler_nearest(vk::SamplerAddressMode::REPEAT);

    let set_layout = rd.descriptor_set_layout_create(&[
        vka::DescriptorSetLayoutEntry {
            binding: 0,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            count: 1,
            flags: None,
        },
        vka::DescriptorSetLayoutEntry {
            binding: 1,
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            count: 1,
            flags: None,
        },
        vka::DescriptorSetLayoutEntry {
            binding: 2,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
            flags: None,
        },
    ])?;
    let set = rd.descriptor_set_create(&set_layout)?;
    rd.descriptor_set_write(&set, &[
        vka::WriteDescriptor::Buffer { binding: 0, array_element: 0, infos: &[buffer.descriptor(0, vk::WHOLE_SIZE)] },
        vka::WriteDescriptor::Image { binding: 1, array_element: 0, infos: &[rd.image_full_view(&image).descriptor()] },
        vka::WriteDescriptor::Image { binding: 2, array_element: 0, infos: &[vk::DescriptorImageInfo::default().sampler(sampler.value)] },
    ]);
    // TODO: test descriptors in a pipeline.
    Ok(())
}
