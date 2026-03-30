use ash::vk;
use ash::vk::BaseOutStructure;
use ash::vk::TaggedStructure;

pub trait AsExtent3D {
    fn as_extent3d(&self, depth: u32) -> vk::Extent3D;
}

impl AsExtent3D for vk::Extent2D {
    fn as_extent3d(&self, depth: u32) -> vk::Extent3D {
        vk::Extent3D { width: self.width, height: self.height, depth }
    }
}

pub fn vulkan_version_str(version: u32) -> String {
    let major = vk::api_version_major(version);
    let minor = vk::api_version_minor(version);
    let patch = vk::api_version_patch(version);
    format!("{}.{}.{}", major, minor, patch)
}

pub fn device_type_to_str(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "IntegratedGpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "DiscreteGpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "VirtualGpu",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "Other",
    }
}

pub fn device_full_name(props: &vk::PhysicalDeviceProperties) -> String {
    format!("{} [{}]", props.device_name_as_c_str().unwrap().to_str().unwrap(), device_type_to_str(props.device_type))
}

pub fn find_sample_count(props: &vk::PhysicalDeviceProperties, requested: vk::SampleCountFlags) -> vk::SampleCountFlags {
    let count = props.limits.framebuffer_color_sample_counts & requested;
    if count & vk::SampleCountFlags::TYPE_8 != vk::SampleCountFlags::empty() {
        return vk::SampleCountFlags::TYPE_8;
    } else if count & vk::SampleCountFlags::TYPE_4 != vk::SampleCountFlags::empty() {
        return vk::SampleCountFlags::TYPE_4;
    } else if count & vk::SampleCountFlags::TYPE_2 != vk::SampleCountFlags::empty() {
        return vk::SampleCountFlags::TYPE_2;
    }
    return vk::SampleCountFlags::TYPE_1;
}

#[inline]
pub fn bytes_of<T: ?Sized>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, size_of_val(value)) }
}

#[inline]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

pub const fn clear_color(r: f32, g: f32, b: f32, a: f32) -> vk::ClearValue {
    vk::ClearValue {
        color: vk::ClearColorValue { float32: [r, g, b, a] },
    }
}

pub const fn clear_depth(depth: f32, stencil: u32) -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
    }
}
