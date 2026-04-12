use ash::vk;
use raw_window_handle::RawDisplayHandle;
use raw_window_handle::RawWindowHandle;

use crate::ENTRY;
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            vsync: false,
        }
    }
}

pub struct Surface {
    pub instance: ash::khr::surface::Instance,
    pub handle: vk::SurfaceKHR,
}

pub fn make_surface(instance: &ash::Instance, rdh: RawDisplayHandle, rwh: RawWindowHandle) -> Result<Surface> {
    unsafe {
        log::info!("Creating the surface");
        let surface_khr = ash::khr::surface::Instance::new(&ENTRY, &instance);
        let surface = ash_window::create_surface(&ENTRY, &instance, rdh, rwh, None)?;
        Result::Ok(Surface { instance: surface_khr, handle: surface })
    }
}
