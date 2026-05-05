use std::sync::atomic::{AtomicU64, Ordering};
use std::ffi::CStr;
use std::hash::DefaultHasher;
use std::hash::Hasher;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use ash::vk;

mod belt;
mod buffer;
mod command;
mod descriptor;
mod device;
mod image;
mod pass;
mod pipeline;
mod sampler;
mod semaphore;
mod shader;
mod surface;

pub use buffer::*;
pub use command::*;
pub use descriptor::*;
pub use device::*;
pub use image::*;
pub use pass::*;
pub use pipeline::*;
pub use sampler::*;
pub use semaphore::*;
pub use shader::*;
pub use surface::*;

pub use gpu_allocator::MemoryLocation;
pub use gpu_allocator::vulkan::Allocation;
pub use gpu_allocator::vulkan::AllocationScheme;

pub use ash;
pub use ash_window;
pub use gpu_allocator;
pub use raw_window_handle;

/// Vulkan handles are not unique.
/// Give each resource a unique ID for debugging and internal caching purposes.
pub fn next_resource_id() -> u64 {
    static RES_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
    RES_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub struct RenderingDeviceDesc<'a> {
    pub app_name: &'a CStr,
    pub gpu_validation: bool,
    pub n_frames: usize,
    pub pick_device: Option<usize>,                             // index for picking a specific device
    pub surface: Option<(RawDisplayHandle, RawWindowHandle)>, // None for headless setup
}

impl RenderingDeviceDesc<'_> {
    pub fn with_window(win: &(impl HasDisplayHandle + HasWindowHandle)) -> Self {
        let rdh = win.display_handle().unwrap().as_raw();
        let rwh = win.window_handle().unwrap().as_raw();
        Self {
            surface: Some((rdh, rwh)),
            ..Default::default()
        }
    }

    pub fn with_surface(display: RawDisplayHandle, window: RawWindowHandle) -> Self {
        Self {
            surface: Some((display, window)),
            ..Default::default()
        }
    }

    pub fn with_gpu_validation(self) -> Self {
        Self {
            gpu_validation: true,
            ..self
        }
    }

    pub fn with_frames(mut self, frames: usize) -> Self {
        self.n_frames = frames;
        self
    }
}

impl Default for RenderingDeviceDesc<'_> {
    fn default() -> Self {
        Self {
            app_name: c"vka app",
            gpu_validation: false,
            n_frames: 2,
            pick_device: None,
            surface: None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
}

impl BufferDesc {
    pub fn new(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn uniform(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn storage(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn index(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn vertex(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn indirect(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::INDIRECT_BUFFER,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn location(mut self, location: MemoryLocation) -> Self {
        self.location = location;
        self
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ImageDesc {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,

    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,

    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
    pub location: MemoryLocation,
}

impl ImageDesc {
    pub fn new_2d(format: vk::Format, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            samples: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn new_3d(format: vk::Format, width: u32, height: u32, depth: u32) -> Self {
        Self {
            format,
            width,
            height,
            depth,
            mip_levels: 1,
            array_layers: 1,
            samples: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn new_cube(format: vk::Format, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 6,
            samples: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            location: MemoryLocation::GpuOnly,
        }
    }

    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn array_layers(mut self, array_layers: u32) -> Self {
        self.array_layers = array_layers;
        self
    }

    pub fn samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    pub fn tiling(mut self, tiling: vk::ImageTiling) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn flags(mut self, flags: vk::ImageCreateFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn location(mut self, location: MemoryLocation) -> Self {
        self.location = location;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color32 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

pub const fn color32(r: f32, g: f32, b: f32, a: f32) -> Color32 {
    Color32 { r, g, b, a }
}

pub trait AsExtent3D {
    fn as_extent3d(&self, depth: u32) -> vk::Extent3D;
}

impl AsExtent3D for vk::Extent2D {
    fn as_extent3d(&self, depth: u32) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth,
        }
    }
}

pub fn hash_struct<T: Sized>(s: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes_of(s));
    hasher.finish()
}

pub fn vulkan_version_str(version: u32) -> String {
    let major = vk::api_version_major(version);
    let minor = vk::api_version_minor(version);
    let patch = vk::api_version_patch(version);
    format!("{}.{}.{}", major, minor, patch)
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
