use std::ffi::CStr;

use ash::vk;
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

pub struct RenderingDeviceDesc<'a> {
    pub app_name: &'a CStr,
    pub gpu_validation: bool,
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
}

impl Default for RenderingDeviceDesc<'_> {
    fn default() -> Self {
        Self {
            app_name: c"vka app",
            gpu_validation: false,
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
