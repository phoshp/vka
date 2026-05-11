use std::ops::Deref;
use std::sync::Arc;

use ash::vk;

use crate::RenderingDevice;
use crate::SharedDevice;

#[derive(Clone)]
#[repr(transparent)]
pub struct ShaderModule(Arc<ShaderModuleImpl>);

impl Deref for ShaderModule {
    type Target = Arc<ShaderModuleImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ShaderModuleImpl {
    pub raw: vk::ShaderModule,
    pub id: u64,
    device: Arc<SharedDevice>,
}

impl Drop for ShaderModuleImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_shader_module(self.raw, None);
        }
    }
}
pub use ash::util::read_spv as read_spv;

impl RenderingDevice {
    pub fn new_shader(&self, code: &[u32]) -> ShaderModule {
        let raw = unsafe { self.raw.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).expect("Failed to create shader module") };
        let inner = ShaderModuleImpl {
            raw,
            id: crate::next_resource_id(),
            device: self.shared.clone(),
        };
        ShaderModule(Arc::new(inner))
    }
}
