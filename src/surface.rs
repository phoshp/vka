use std::slice;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ash::prelude::VkResult;
use ash::vk;
use ash::vk::SurfaceFormatKHR;
use itertools::Itertools;
use parking_lot::Mutex;
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

    pub acquire_semaphores: Vec<Arc<Mutex<AcquireSemaphore>>>,
    pub present_semaphores: Vec<Arc<Mutex<PresentSemaphores>>>,
    pub acquire_index: usize,
    pub fence: vk::Fence,
}

pub struct AcquireSemaphore {
    pub acquire: vk::Semaphore,
    pub should_wait: bool,
    pub last_used_submission: u64,
}

pub struct PresentSemaphores {
    pub present: Vec<vk::Semaphore>,
    pub count: u32,
}

impl PresentSemaphores {
    pub fn get_waits(&self) -> &[vk::Semaphore] {
        &self.present[..self.count as usize]
    }

    pub fn signal(&mut self, device: &ash::Device) -> vk::Semaphore {
        let sem = match self.present.get(self.count as usize) {
            Some(sem) => *sem,
            None => {
                let sem = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() };
                self.present.push(sem);
                sem
            }
        };
        self.count += 1;
        sem
    }
}

#[derive(Clone)]
pub struct SurfaceImage {
    pub image: Image,
    pub index: u32,

    pub(crate) acquire: Arc<Mutex<AcquireSemaphore>>,
    pub(crate) present: Arc<Mutex<PresentSemaphores>>,
}

impl Surface {
    pub fn acquire(&mut self) -> SurfaceImage {
        let acquire_sem = self.acquire_semaphores[self.acquire_index].lock();
        self.device.wait_submission(acquire_sem.last_used_submission);

        let res = unsafe {
            self.swapchain
                .device
                .acquire_next_image(self.swapchain.raw, u64::MAX, acquire_sem.acquire, self.fence)
        };
        drop(acquire_sem);

        match res {
            Ok((index, _)) => {
                unsafe {
                    self.device.raw.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
                    self.device.raw.reset_fences(&[self.fence]).unwrap();
                }

                let image = SurfaceImage {
                    image: self.swapchain.images[index as usize].clone(),
                    index,
                    acquire: self.acquire_semaphores[self.acquire_index as usize].clone(),
                    present: self.present_semaphores[index as usize].clone(),
                };

                {
                    // Transition image to GENERAL for use.
                    let mut cmd = self.device.new_command_buffer();
                    // let old_layout = if image.image.initialized.load(Ordering::Acquire) { vk::ImageLayout::PRESENT_SRC_KHR } else { vk::ImageLayout::UNDEFINED };
                    cmd.image_barrier_raw(image.image.raw, image.image.aspect, vk::ImageLayout::PRESENT_SRC_KHR, image.image.optimal_layout);
                    self.device.submit([cmd], Some(&image));
                }

                self.acquire_index = (self.acquire_index + 1) % self.acquire_semaphores.len();
                image
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::ERROR_SURFACE_LOST_KHR) | Err(vk::Result::NOT_READY) => {
                log::error!("Failed to acquire next image: {:?}, recreating swapchain", res);
                self.recreate_swapchain();
                self.acquire()
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        }
    }

    pub fn present(&mut self, image: &SurfaceImage) -> bool {
        {
            // Transition image to PRESENT_SRC_KHR for presentation.
            let mut cmd = self.device.new_command_buffer();
            cmd.image_barrier_raw(image.image.raw, image.image.aspect, image.image.optimal_layout, vk::ImageLayout::PRESENT_SRC_KHR);
            self.device.submit([cmd], Some(&image));
        }
        let mut present_sem = image.present.lock();
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(present_sem.get_waits())
            .swapchains(slice::from_ref(&self.swapchain.raw))
            .image_indices(slice::from_ref(&image.index));

        let _idx = self.device.submit_mutex.lock();
        let res = unsafe { self.swapchain.device.queue_present(self.device.present_queue, &present_info) };
        drop(_idx);

        match res {
            Ok(suboptimal) => {
                present_sem.count = 0;
                image.acquire.lock().should_wait = true;
                suboptimal
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::ERROR_SURFACE_LOST_KHR) | Err(vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR) => {
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
        self.device.wait_idle();

        self.acquire_semaphores
            .drain(..)
            .for_each(|s| unsafe { self.device.raw.destroy_semaphore(s.lock().acquire, None) });
        self.present_semaphores
            .drain(..)
            .for_each(|s| s.lock().present.iter().for_each(|x| unsafe { self.device.raw.destroy_semaphore(*x, None) }));

        let old_swapchain = self.swapchain.raw;
        self.swapchain = make_swapchain(&self.device, self.raw, self.config, Some(old_swapchain)).expect("Failed to recreate swapchain");

        self.acquire_semaphores = (0..self.swapchain.images.len())
            .map(|_| unsafe {
                let acquire = self.device.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                AcquireSemaphore {
                    acquire,
                    should_wait: true,
                    last_used_submission: 0,
                }
            })
            .map(Mutex::new)
            .map(Arc::new)
            .collect_vec();
        self.present_semaphores = (0..self.swapchain.images.len())
            .map(|_| PresentSemaphores { present: Vec::new(), count: 0 })
            .map(Mutex::new)
            .map(Arc::new)
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
                self.device.raw.destroy_semaphore(sem.lock().acquire, None);
            }
            for sems in &self.present_semaphores {
                sems.lock().present.iter().for_each(|x| {
                    self.device.raw.destroy_semaphore(*x, None);
                });
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

        let acquire_semaphores = (0..swapchain.images.len())
            .map(|_| unsafe {
                let acquire = self.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                AcquireSemaphore {
                    acquire,
                    should_wait: true,
                    last_used_submission: 0,
                }
            })
            .map(Mutex::new)
            .map(Arc::new)
            .collect_vec();
        let present_semaphores = (0..swapchain.images.len())
            .map(|_| PresentSemaphores { present: Vec::new(), count: 0 })
            .map(Mutex::new)
            .map(Arc::new)
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
            acquire_index: 0,
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
        log::info!("Available present modes: {}", present_modes.iter().map(|&m| format!("{:?}", m)).join(","));
        log::info!("Selected present mode: {:?}", present_mode);
        log::info!("Surface Format: {:?}", format);
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
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
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
                let img = rd.new_image_raw(
                    image,
                    format,
                    extent.as_extent3d(1),
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    None,
                );
                img
            })
            .collect_vec();

        {
            let mut cmd = rd.new_command_buffer();
            for image in &images {
                cmd.image_barrier_raw(image.raw, image.aspect, vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR);
            }
            rd.submit([cmd], None);
        }

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
