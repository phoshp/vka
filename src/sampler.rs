use std::ops::Deref;
use std::sync::Arc;

use ash::vk;

use crate::RenderingDevice;
use crate::SharedDevice;
use crate::next_resource_id;

#[derive(Clone)]
#[repr(transparent)]
pub struct Sampler(Arc<SamplerImpl>);

pub struct SamplerImpl {
    pub raw: vk::Sampler,
    pub id: u64,
    device: Arc<SharedDevice>,
}

impl Deref for Sampler {
    type Target = Arc<SamplerImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for SamplerImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_sampler(self.raw, None);
        }
    }
}

impl RenderingDevice {
    pub fn new_sampler(&self, desc: &vk::SamplerCreateInfo) -> Sampler {
        let raw = unsafe { self.raw.create_sampler(desc, None).expect("Failed to create sampler") };
        let id = next_resource_id();
        Sampler(Arc::new(SamplerImpl { raw, id, device: self.shared.clone() }))
    }

    /// Creates a Sampler with nearest filtering for both min and mag.
    pub fn new_sampler_nearest(&self, wrap_mode: vk::SamplerAddressMode) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with nearest mag and linear min filtering.
    pub fn new_sampler_nearest_linear(&self, wrap_mode: vk::SamplerAddressMode) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with bilinear filtering (linear min/mag, nearest mipmap).
    pub fn new_sampler_bilinear(&self, wrap_mode: vk::SamplerAddressMode) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with trilinear filtering (linear min/mag/mipmap).
    pub fn new_sampler_trilinear(&self, wrap_mode: vk::SamplerAddressMode) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with anisotropic filtering enabled.
    pub fn new_sampler_anisotropic(&self, wrap_mode: vk::SamplerAddressMode, max_anisotropy: f32) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode)
                .anisotropy_enable(true)
                .max_anisotropy(max_anisotropy),
        )
    }

    /// Creates a Sampler configured for shadow mapping (linear filtering, compare enable).
    pub fn new_sampler_shadow(&self, wrap_mode: vk::SamplerAddressMode) -> Sampler {
        self.new_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL),
        )
    }
}
