use ash::vk;
use ash::vk::SurfaceFormatKHR;
use itertools::Itertools;

use crate::AsExtent3D;
use crate::Image;
use crate::RenderingDevice;
use crate::Result;

/// Encapsulates the Vulkan swapchain and its associated images for presentation to a window surface.
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub device: ash::khr::swapchain::Device,
    pub images: Vec<Image>,

    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,

    pub present_semaphores: Vec<vk::Semaphore>,
}

fn present_mode_str(mode: vk::PresentModeKHR) -> &'static str {
    match mode {
        vk::PresentModeKHR::IMMEDIATE => "Immediate",
        vk::PresentModeKHR::MAILBOX => "Mailbox",
        vk::PresentModeKHR::FIFO => "Fifo",
        vk::PresentModeKHR::FIFO_RELAXED => "Fifo Relaxed",
        _ => "Unknown",
    }
}

/// Factory function to create or recreate a swapchain based on the current surface configuration.
pub fn make_swapchain(rd: &RenderingDevice, old_swapchain: Option<vk::SwapchainKHR>) -> Result<Swapchain> {
    unsafe {
        let device = &rd.device;
        let instance = &rd.instance;
        let surface = rd.surface.as_ref().unwrap();
        let config = rd.surface_config.get();

        let device = ash::khr::swapchain::Device::new(instance, device);
        let present_modes = surface.instance.get_physical_device_surface_present_modes(rd.phy_device, surface.handle)?;
        let caps = surface.instance.get_physical_device_surface_capabilities(rd.phy_device, surface.handle)?;
        let formats = surface.instance.get_physical_device_surface_formats(rd.phy_device, surface.handle)?;

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

        let image_count = caps.min_image_count + 1;
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
        log::info!("Available present modes: {}", present_modes.iter().map(|&m| present_mode_str(m)).join(","));
        log::info!("Selected present mode: {}", present_mode_str(present_mode));
        log::info!("Surface Format: {}", format.as_raw());
        log::info!("Framebuffer Size: {}x{}", extent.width, extent.height);

        let swapchain = device.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface.handle)
                .min_image_count(image_count)
                .image_format(format)
                .image_color_space(color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null())),
            None,
        )?;
        let images = device
            .get_swapchain_images(swapchain)?
            .iter()
            .enumerate()
            .map(|(j, &image)| {
                let img = rd.image_from_raw(
                    image,
                    format,
                    extent.as_extent3d(1),
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    None,
                );
                img.layout.set(vk::ImageLayout::PRESENT_SRC_KHR); // initial state for swapchain images
                img.set_name(format!("Swapchain Image {}", j));
                img
            })
            .collect_vec();

        let present_semaphores = (0..image_count)
            .map(|i| rd.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect_vec();

        Ok(Swapchain {
            handle: swapchain,
            device,
            images,
            extent,
            format,
            color_space,
            present_mode,
            present_semaphores
        })
    }
}
