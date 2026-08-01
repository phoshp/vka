use std::slice;

use ash::prelude::VkResult;
use ash::vk;
use ash::vk::SurfaceFormatKHR;
use itertools::Itertools;
use raw_window_handle::HasDisplayHandle;
use raw_window_handle::HasWindowHandle;
use raw_window_handle::RawDisplayHandle;
use raw_window_handle::RawWindowHandle;

use crate::AsExtent3D;
use crate::Image;
use crate::RenderingDevice;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
    pub frame_latency: u32,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            vsync: false,
            frame_latency: 2,
        }
    }
}

pub struct Surface {
    pub raw: vk::SurfaceKHR,
    pub instance: ash::khr::surface::Instance,
    pub swapchain: Swapchain,
    pub config: SurfaceConfig,
    device: RenderingDevice,

    pub acquire_semaphores: Vec<vk::Semaphore>,
    pub present_semaphores: Vec<vk::Semaphore>,
    pub current_image: Option<SurfaceImage>,
    pub fence: vk::Fence,
}

#[derive(Clone)]
pub struct SurfaceImage {
    pub inner: Image,
    pub index: u32,
    pub frame_index: usize,
}

impl Surface {
    pub unsafe fn acquire_next_image_raw(&mut self, frame_index: usize) -> Option<SurfaceImage> {
        if let Some(i) = &self.current_image {
            if i.frame_index == frame_index {
                return Some(i.clone());
            }
        }
        let res = unsafe {
            self.swapchain.device.acquire_next_image(self.swapchain.raw, u64::MAX, self.acquire_semaphores[frame_index], vk::Fence::null())
        };
        match res {
            Ok((index, _)) => {
                let image = SurfaceImage {
                    inner: self.swapchain.images[index as usize].clone(),
                    index,
                    frame_index,
                };
                self.current_image = Some(image);
                self.current_image.clone()
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::ERROR_SURFACE_LOST_KHR) | Err(vk::Result::NOT_READY) => None,
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        }
    }

    pub fn get_current_image(&self) -> Option<SurfaceImage> {
        self.current_image.clone()
    }

    pub fn present(&mut self) -> bool {
        let image_index = match self.current_image.take() {
            Some(i) => i.index,
            None => {
                log::error!("No image acquired for presentation!");
                return false;
            }
        };
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&self.present_semaphores[image_index as usize]))
            .swapchains(slice::from_ref(&self.swapchain.raw))
            .image_indices(slice::from_ref(&image_index));
        let res = unsafe { self.swapchain.device.queue_present(self.device.present_queue, &present_info) };
        match res {
            Ok(suboptimal) => suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR) => {
                log::error!("Presentation failed with {:?}, recreating swapchain", res);
                self.recreate_swapchain();
                false
            }
            Err(e) => panic!("Failed to present image: {:?}", e),
        }
    }

    pub fn configure(&mut self, config: SurfaceConfig) {
        self.config = config;
        self.recreate_swapchain();
    }

    pub fn recreate_swapchain(&mut self) {
        self.device.submit();
        self.device.wait_idle();
        self.device.reset_frames();

        self.current_image = None;
        self.acquire_semaphores.drain(..).for_each(|s| unsafe { self.device.raw.destroy_semaphore(s, None) });
        self.present_semaphores.drain(..).for_each(|s| unsafe { self.device.raw.destroy_semaphore(s, None) });

        let old_swapchain = self.swapchain.raw;
        self.swapchain = make_swapchain(&self.device, self.raw, self.config, Some(old_swapchain)).expect("Failed to recreate swapchain");

        self.acquire_semaphores = (0..self.device.n_frames())
            .map(|_| unsafe { self.device.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() })
            .collect_vec();
        self.present_semaphores = (0..self.swapchain.images.len())
            .map(|_| unsafe { self.device.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() })
            .collect_vec();
        unsafe {
            self.device.raw.reset_fences(&[self.fence]).unwrap();
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle();
            self.device.raw.destroy_fence(self.fence, None);

            for sem in &self.acquire_semaphores {
                self.device.raw.destroy_semaphore(*sem, None);
            }
            for sem in &self.present_semaphores {
                self.device.raw.destroy_semaphore(*sem, None);
            }
            self.swapchain.device.destroy_swapchain(self.swapchain.raw, None);
            self.instance.destroy_surface(self.raw, None);
        }
    }
}

impl RenderingDevice {
    pub fn new_surface(&self, window: &(impl HasWindowHandle + HasDisplayHandle), config: SurfaceConfig) -> Surface {
        self.new_surface_raw(window.display_handle().unwrap().as_raw(), window.window_handle().unwrap().as_raw(), config)
    }

    pub fn new_surface_raw(&self, rdh: RawDisplayHandle, rwh: RawWindowHandle, config: SurfaceConfig) -> Surface {
        let surface_khr = ash::khr::surface::Instance::new(&self.shared.entry, &self.shared.instance);
        let surface = unsafe { ash_window::create_surface(&self.shared.entry, &self.shared.instance, rdh, rwh, None).expect("Failed to create surface") };
        let swapchain = make_swapchain(&self, surface, config, None).expect("Failed to create swapchain");

        let acquire_semaphores = (0..self.n_frames())
            .map(|_| unsafe { self.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() })
            .collect_vec();
        let present_semaphores = (0..swapchain.images.len())
            .map(|_| unsafe { self.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() })
            .collect_vec();
        let fence = unsafe { self.raw.create_fence(&vk::FenceCreateInfo::default(), None).unwrap() };

        Surface {
            raw: surface,
            instance: surface_khr,
            swapchain,
            config,
            device: self.clone(),

            acquire_semaphores,
            present_semaphores,
            current_image: None,
            fence,
        }
    }
}

/// Encapsulates the Vulkan swapchain and its associated images for presentation to a window surface.
pub struct Swapchain {
    pub raw: vk::SwapchainKHR,
    pub device: ash::khr::swapchain::Device,
    pub images: Vec<Image>,

    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
}

/// Factory function to create or recreate a swapchain based on the current surface config
pub fn make_swapchain(rd: &RenderingDevice, surface: vk::SurfaceKHR, config: SurfaceConfig, old_swapchain: Option<vk::SwapchainKHR>) -> VkResult<Swapchain> {
    unsafe {
        let device = &rd.raw;
        let instance = &rd.shared.instance;

        let device = ash::khr::swapchain::Device::new(instance, device);
        let surface_inst = ash::khr::surface::Instance::new(&rd.shared.entry, instance);
        let present_modes = surface_inst.get_physical_device_surface_present_modes(rd.phy_device, surface)?;
        let caps = surface_inst.get_physical_device_surface_capabilities(rd.phy_device, surface)?;
        let formats = surface_inst.get_physical_device_surface_formats(rd.phy_device, surface)?;

        let SurfaceFormatKHR { format, color_space } = formats
            .iter()
            .find_map(|&f| {
                if f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                    Some(f)
                } else {
                    None
                }
            })
            .unwrap_or(formats[0]);

        let present_mode = if config.vsync {
            vk::PresentModeKHR::FIFO
        } else if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            present_modes[0]
        };

        let image_count = (config.frame_latency + 1).max(caps.min_image_count + 1);
        let extent = vk::Extent2D {
            width: if caps.max_image_extent.width > 0 {
                config.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width)
            } else {
                config.width
            },
            height: if caps.max_image_extent.height > 0 {
                config.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height)
            } else {
                config.height
            },
        };
        log::info!("Creating swapchain:");
        log::info!("Available present modes: {:?}", present_modes);
        log::info!("Selected present mode: {:?}", present_mode);
        log::info!("Surface Format: {:?} {:?}", format, color_space);
        log::info!("Framebuffer Size: {}x{}", extent.width, extent.height);

        let swapchain = device.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(format)
                .image_color_space(color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // TODO: Support other composite alpha modes
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null())),
            None,
        )?;
        if let Some(old) = old_swapchain {
            device.destroy_swapchain(old, None);
        }

        let images = device
            .get_swapchain_images(swapchain)?
            .iter()
            .map(|&image| {
                let img = rd.new_image_raw(image, format, extent.as_extent3d(1), vk::SampleCountFlags::TYPE_1, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, None);
                img
            })
            .collect_vec();

        rd.record(|encoder| {
            for img in images.iter() {
                encoder.image_barrier(img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR);
            }
        });
        Ok(Swapchain {
            raw: swapchain,
            device,
            images,
            extent,
            format,
            color_space,
            present_mode,
        })
    }
}
