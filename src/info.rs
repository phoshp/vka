use std::ffi::CStr;

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct RenderingDeviceInfo<'a> {
    pub app_name: &'a CStr,
    pub gpu_validation: bool,
    pub pick_device: Option<usize>,                             // index for picking a specific device
    pub surface: Option<(RawDisplayHandle, RawWindowHandle)>, // None for headless setup
}

impl RenderingDeviceInfo<'_> {
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

impl Default for RenderingDeviceInfo<'_> {
    fn default() -> Self {
        Self {
            app_name: c"vka app",
            gpu_validation: false,
            pick_device: None,
            surface: None,
        }
    }
}
