#![allow(unused)]

use std::borrow::Cow;
use std::cell::{Cell, Ref, RefMut};
use std::ops::DerefMut;
use std::{cell::RefCell, ffi::CStr, mem::ManuallyDrop, ops::Deref, rc::Rc, sync::LazyLock};

use anyhow::{anyhow};
use ash::ext::debug_utils;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use itertools::Itertools;
use std::sync::Mutex as StdMutex;

mod belt;
mod buffer;
mod desc;
mod descriptor;
mod image;
mod mutex;
mod pass;
mod pipeline;
mod resource;
mod surface;
mod swapchain;
mod utils;

pub use buffer::*;
pub use desc::*;
pub use descriptor::*;
pub use image::*;
pub use mutex::*;
pub use pass::*;
pub use pipeline::*;
pub use resource::*;
pub use surface::*;
pub use swapchain::*;
pub use utils::*;

pub use gpu_allocator::MemoryLocation;
pub use gpu_allocator::vulkan::Allocation;
pub use gpu_allocator::vulkan::AllocationScheme;

use crate::belt::StagingBelt;

pub type Result<T> = anyhow::Result<T>;

pub static ENTRY: LazyLock<ash::Entry> = LazyLock::new(|| unsafe { ash::Entry::load().expect("Failed to load Vulkan library") });

/// Holds indices for the different Vulkan queue families used by the device.
#[derive(Debug, Clone, Copy)]
pub struct QueueFamilies {
    pub present: u32,
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
}

impl Default for QueueFamilies {
    fn default() -> Self {
        Self {
            present: vk::QUEUE_FAMILY_IGNORED,
            graphics: vk::QUEUE_FAMILY_IGNORED,
            compute: vk::QUEUE_FAMILY_IGNORED,
            transfer: vk::QUEUE_FAMILY_IGNORED,
        }
    }
}

/// Represents resources required for rendering a single frame in flight.
pub struct Frame {
    pub cmd_pool: vk::CommandPool,
    front_cmd: Cell<vk::CommandBuffer>,
    back_cmd: Cell<vk::CommandBuffer>,
    idle: Cell<bool>,

    pub image_semaphore: vk::Semaphore,
    pub fence: vk::Fence,

    pub belt: RefCell<StagingBelt>,
}

pub struct Presentation {
    pub surface: Surface,
    pub surface_config: Cell<SurfaceConfig>,
    pub swapchain: RefCell<Swapchain>,

    pub semaphores: Vec<vk::Semaphore>,
    pub image_index: Cell<usize>,
    pub image_acquired: Cell<bool>,
    pub suboptimal_swapchain: Cell<bool>,
}

/// The inner state of a Vulkan rendering device containing the instance, physical device, logical device, and other core resources.
pub struct RenderingDeviceImpl {
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub phy_device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub mem_properties: vk::PhysicalDeviceMemoryProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub features11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub features12: vk::PhysicalDeviceVulkan12Features<'static>,
    pub features13: vk::PhysicalDeviceVulkan13Features<'static>,

    pub enabled_extensions: Vec<&'static CStr>,
    pub enabled_layers: Vec<&'static CStr>,
    pub enabled_instance_exts: Vec<&'static CStr>,

    pub allocator: ManuallyDrop<StdMutex<Allocator>>,

    debug_utils: Option<DebugUtils>,
    presentation: Option<Presentation>,

    pub queue_families: QueueFamilies,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub frames: Vec<Frame>,
    pub frame_index: Cell<usize>,
}

impl Into<RenderingDevice> for Rc<RenderingDeviceImpl> {
    fn into(self) -> RenderingDevice {
        RenderingDevice(self)
    }
}

/// A reference-counted wrapper around `RenderingDeviceImpl`, providing convenient access to Vulkan operations.
#[derive(Clone)]
#[repr(transparent)]
pub struct RenderingDevice(Rc<RenderingDeviceImpl>);

impl Deref for RenderingDevice {
    type Target = RenderingDeviceImpl;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RenderingDevice {
    /// Initializes a new Vulkan rendering device, instances, and necessary Queues/Allocators according to `RenderingDeviceDesc`.
    pub fn new(desc: &RenderingDeviceDesc) -> Result<Self> {
        unsafe {
            let vulkan_version = ENTRY.try_enumerate_instance_version()?.unwrap_or(vk::API_VERSION_1_0);
            let enum_layer_props = ENTRY.enumerate_instance_layer_properties()?;
            let enum_ext_props = ENTRY.enumerate_instance_extension_properties(None)?;

            let available_layers = enum_layer_props.iter().map(|x| x.layer_name_as_c_str().unwrap()).collect_vec();
            let available_exts = enum_ext_props.iter().map(|x| x.extension_name_as_c_str().unwrap()).collect_vec();

            let app_info = vk::ApplicationInfo::default()
                .engine_name(desc.app_name)
                .application_name(desc.app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vulkan_version);

            let mut enabled_layers = Vec::new();
            let mut enabled_instance_exts = vec![vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME];

            if let Some(surface) = desc.surface {
                enabled_instance_exts.extend(ash_window::enumerate_required_extensions(surface.0).unwrap().iter().map(|&x| CStr::from_ptr(x)));
            }
            let validation_layers_enabled = desc.gpu_validation && available_layers.contains(&c"VK_LAYER_KHRONOS_validation") && available_exts.contains(&vk::EXT_DEBUG_UTILS_NAME);

            if validation_layers_enabled {
                enabled_layers.push(c"VK_LAYER_KHRONOS_validation");
                enabled_instance_exts.push(vk::EXT_DEBUG_UTILS_NAME);
            }
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                enabled_instance_exts.push(vk::KHR_PORTABILITY_ENUMERATION_NAME)
            }

            let inst_layers_ptr = enabled_layers.iter().map(|x| x.as_ptr()).collect_vec();
            let inst_exts_ptr = enabled_instance_exts.iter().map(|x| x.as_ptr()).collect_vec();
            let mut instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&inst_layers_ptr)
                .enabled_extension_names(&inst_exts_ptr);
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                instance_info.flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }

            log::info!("Creating vulkan instance:");
            log::info!("App name: {}", desc.app_name.to_str()?);
            log::info!("Vulkan Version: {}", vulkan_version_str(vulkan_version));
            log::info!("Extensions: {}", enabled_instance_exts.iter().map(|&v| v.to_str().unwrap()).join(", "));
            log::info!("Layers: {}", enabled_layers.iter().map(|&v| v.to_str().unwrap()).join(", "));
            let instance = ENTRY.create_instance(&instance_info, None)?;

            let mut enabled_device_exts = vec![vk::KHR_DYNAMIC_RENDERING_NAME];
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                enabled_device_exts.push(vk::KHR_PORTABILITY_SUBSET_NAME);
            }

            let found_devices = instance
                .enumerate_physical_devices()?
                .iter()
                .map(|&pd| (pd, instance.get_physical_device_properties(pd)))
                .collect_vec();
            log::info!("Found devices: {}", found_devices.clone().iter().map(|f| utils::device_full_name(&f.1)).join(", "));

            let (mut phy_device, mut properties, _) = found_devices
                .iter()
                .map(|&(pd, props)| {
                    // TODO: better scoring system
                    let score = match props.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 5,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
                        vk::PhysicalDeviceType::CPU => 2,
                        _ => 1,
                    };
                    (pd, props, score)
                })
                .sorted_by(|a, b| Ord::cmp(&b.2, &a.2))
                .next()
                .ok_or(vk::Result::ERROR_UNKNOWN)?;

            if let Some(idx) = desc.pick_device {
                if let Some((pd, props)) = found_devices.get(idx) {
                    log::info!("Picking device at specified index {}", idx);
                    phy_device = *pd;
                    properties = *props;
                } else {
                    log::warn!("Specified device index {} is out of bounds, ignoring", idx);
                }
            }

            let mut features = vk::PhysicalDeviceFeatures2::default();
            let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();

            features = features.push_next(&mut features11).push_next(&mut features12).push_next(&mut features13);
            instance.get_physical_device_features2(phy_device, &mut features);

            features.features.robust_buffer_access &= desc.gpu_validation as u32;

            let surface = if let Some((rdh, rwh)) = desc.surface {
                enabled_device_exts.push(vk::KHR_SWAPCHAIN_NAME);
                Some(make_surface(&instance, rdh, rwh)?)
            } else {
                None
            };

            let queue_props = instance.get_physical_device_queue_family_properties(phy_device);
            let mut queue_families = QueueFamilies::default();
            for (i, props) in queue_props.iter().enumerate() {
                let i = i as u32;
                if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) && queue_families.graphics == vk::QUEUE_FAMILY_IGNORED {
                    queue_families.graphics = i;
                }
                if let Some(surface) = &surface {
                    if surface.instance.get_physical_device_surface_support(phy_device, i, surface.handle).unwrap_or(false) {
                        queue_families.present = i;
                    }
                }
                if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    queue_families.compute = i;
                }
                if props.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    queue_families.transfer = i;
                }
            }

            if queue_families.present == vk::QUEUE_FAMILY_IGNORED {
                queue_families.present = queue_families.graphics;
                log::warn!("No present queue found, falling back to graphics queue");
            }

            let features10 = features.features;
            log::info!("Creating logical device");
            log::info!("Picked device: {}", utils::device_full_name(&properties));
            log::info!("Enabled Extensions: {}", &enabled_device_exts.iter().map(|x| x.to_str().unwrap()).join(","));

            let queue_create_infos = [queue_families.present, queue_families.graphics, queue_families.compute, queue_families.transfer]
                .iter()
                .unique()
                .filter(|&&i| i != vk::QUEUE_FAMILY_IGNORED)
                .map(|&i| vk::DeviceQueueCreateInfo::default().queue_family_index(i).queue_priorities(&[1.0]))
                .collect_vec();
            let device = instance.create_device(
                phy_device,
                &vk::DeviceCreateInfo::default()
                    .enabled_extension_names(&enabled_device_exts.iter().map(|x| x.as_ptr()).collect_vec())
                    .queue_create_infos(&queue_create_infos)
                    .push_next(&mut features),
                None,
            )?;
            let mem_properties = instance.get_physical_device_memory_properties(phy_device);
            let debug_utils = if validation_layers_enabled { Some(make_debug_utils(&instance, &device)?) } else { None };

            let graphics_queue = device.get_device_queue(queue_families.graphics, 0);
            let present_queue = device.get_device_queue(queue_families.present, 0);
            // TODO: more on that later
            let allocator = StdMutex::new(
                Allocator::new(&AllocatorCreateDesc {
                    instance: instance.clone(),
                    device: device.clone(),
                    physical_device: phy_device,
                    debug_settings: Default::default(),
                    buffer_device_address: features12.buffer_device_address != 0 || enabled_device_exts.contains(&vk::EXT_BUFFER_DEVICE_ADDRESS_NAME),
                    allocation_sizes: Default::default(),
                })
                .unwrap(),
            );

            let n_frames = 2; // TODO: make it configurable
            let frames = (0..n_frames)
                .map(|_| unsafe {
                    let cmd_pool = device
                        .create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .queue_family_index(queue_families.graphics)
                                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                            None,
                        )
                        .unwrap();
                    let cmds = device
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_buffer_count(2)
                                .command_pool(cmd_pool)
                                .level(vk::CommandBufferLevel::PRIMARY),
                        )
                        .unwrap();
                    device
                        .begin_command_buffer(cmds[0], &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                        .unwrap();
                    device
                        .begin_command_buffer(cmds[1], &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                        .unwrap();

                    let image_semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                    let fence = device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap();
                    Frame {
                        cmd_pool,
                        front_cmd: Cell::new(cmds[0]),
                        back_cmd: Cell::new(cmds[1]),
                        idle: Cell::new(false),
                        image_semaphore,
                        fence,
                        belt: RefCell::new(StagingBelt::new(4 * 1024 * 1024)), // 4 MB per chunk
                    }
                })
                .collect_vec();

            let rd = RenderingDevice(Rc::new(RenderingDeviceImpl {
                instance,
                device,
                phy_device,
                properties,
                mem_properties,
                features: features10,
                features11,
                features12,
                features13,

                enabled_extensions: enabled_device_exts,
                enabled_layers,
                enabled_instance_exts,

                allocator: ManuallyDrop::new(allocator),

                debug_utils,
                presentation: None,

                queue_families,
                graphics_queue,
                present_queue,

                frames,
                frame_index: Cell::new(0),
            }));

            rd.wait_queue()?;

            let presentation = surface.map(|s| {
                let config = SurfaceConfig::default();
                let swapchain = swapchain::make_swapchain(&rd, &s, config, None).unwrap();
                let semaphores = (0..swapchain.images.len())
                    .map(|_| rd.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
                    .collect_vec();
                Presentation {
                    surface: s,
                    surface_config: Cell::new(config),
                    swapchain: RefCell::new(swapchain),

                    semaphores,
                    image_index: Cell::new(0),
                    image_acquired: Cell::new(false),
                    suboptimal_swapchain: Cell::new(false),
                }
            });
            unsafe {
                let ptr = Rc::as_ptr(&rd.0) as *mut RenderingDeviceImpl;
                (*ptr).presentation = presentation;
            }

            Result::Ok(rd)
        }
    }

    /// Updates the swapchain surface configuration and recreates the swapchain.
    pub fn reconfigure_surface(&self, config: SurfaceConfig) {
        self.presentation.as_ref().inspect(|p| p.surface_config.set(config));
        self.recreate_swapchain();
    }

    pub fn recreate_swapchain(&self) {
        unsafe {
            if let Some(p) = &self.presentation {
                self.device.device_wait_idle();
                p.suboptimal_swapchain.set(false);

                p.swapchain.replace_with(|old| unsafe {
                    let new_swapchain = swapchain::make_swapchain(self, &p.surface, p.surface_config.get(), Some(old.handle)).unwrap();
                    old.device.destroy_swapchain(old.handle, None);
                    new_swapchain
                });
            }
        }
    }

    /// Gets the resources for the current frame.
    pub fn frame(&self) -> &Frame {
        &self.frames[self.frame_index.get()]
    }

    pub fn frame_wait_idle(&self, frame: &Frame) -> Result<()> {
        if !frame.idle.get() {
            unsafe {
                self.device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
                self.device.reset_fences(&[frame.fence])?;
            }
            frame.idle.set(true);
        }
        Ok(())
    }

    /// Acquires the next available image from the swapchain for rendering. Recreates swapchain if suboptimal.
    pub fn acquire_swapchain_image(&self) -> Option<Image> {
        let p = match self.presentation.as_ref() {
            Some(p) => p,
            None => return None,
        };
        if p.suboptimal_swapchain.get() {
            log::info!("Swapchain is suboptimal, recreating");
            self.recreate_swapchain();
        }
        let frame = self.frame();
        self.frame_wait_idle(frame);
        let swapchain = p.swapchain.borrow();
        let (image_index, suboptimal) = unsafe {
            swapchain
                .device
                .acquire_next_image(swapchain.handle, u64::MAX, frame.image_semaphore, vk::Fence::null())
                .unwrap()
        };
        p.suboptimal_swapchain.set(p.suboptimal_swapchain.get() || suboptimal);
        p.image_acquired.set(true);
        p.image_index.set(image_index as usize);

        Some(swapchain.images[image_index as usize].clone())
    }

    pub fn debug_utils(&self) -> Option<&DebugUtils> {
        self.debug_utils.as_ref()
    }

    pub fn presentation(&self) -> Option<&Presentation> {
        self.presentation.as_ref()
    }

    pub fn is_image_acquired(&self) -> bool {
        self.presentation().map(|p| p.image_acquired.get()).unwrap_or(false)
    }

    pub fn get_swapchain_extent(&self) -> vk::Extent2D {
        self.presentation().map(|p| p.swapchain.borrow().extent).unwrap_or_default()
    }

    /// Gets the current frame's command buffer where rendering commands should be recorded.
    pub fn get_cmd_buffer(&self) -> vk::CommandBuffer {
        self.frame().front_cmd.get()
    }

    /// Records commands to the current frame's command buffer via a closure.
    pub fn record(&self, record_fn: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        let cmd = self.get_cmd_buffer();
        record_fn(&self.device, cmd);
    }

    /// Submits the current frame's command buffer to the graphics queue and advances to the next frame.
    pub fn submit(&self) -> Result<()> {
        unsafe {
            let frame = self.frame();

            self.frame_wait_idle(frame);
            frame.belt.borrow_mut().reset();

            self.device.end_command_buffer(frame.front_cmd.get()).unwrap();

            let cmd_buffers = [frame.front_cmd.get()];
            let wait_semaphores = self.is_image_acquired().then(|| frame.image_semaphore);
            let signal_semaphores = self.is_image_acquired().then(|| self.presentation().map(|p| p.semaphores[p.image_index.get()]).unwrap());

            let mut submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_buffers)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                .wait_semaphores(wait_semaphores.as_slice())
                .signal_semaphores(signal_semaphores.as_slice());

            self.device.queue_submit(self.graphics_queue, &[submit_info], frame.fence)?;

            self.device.reset_command_buffer(frame.back_cmd.get(), vk::CommandBufferResetFlags::empty())?;
            frame.front_cmd.swap(&frame.back_cmd);
            frame.idle.set(false);
            self.device.begin_command_buffer(
                frame.front_cmd.get(),
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.frame_index.set((self.frame_index.get() + 1) % self.frames.len());
            Result::Ok(())
        }
    }

    /// Blocks until the graphics queue goes idle.
    pub fn wait_queue(&self) -> Result<()> {
        unsafe { Ok(self.device.queue_wait_idle(self.graphics_queue)?) }
    }

    /// Submits the current frame and waits for it to complete.
    pub fn submit_wait(&self) -> Result<()> {
        self.submit()?;
        self.wait_queue()
    }

    /// Presents the last rendered frame to the swapchain.
    pub fn present(&self) -> Result<()> {
        if let Some(p) = &self.presentation {
            let swapchain = p.swapchain.borrow();
            let image_index = p.image_index.get();

            let suboptimal = unsafe {
                swapchain.device.queue_present(
                    self.present_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[p.semaphores[image_index]])
                        .swapchains(&[swapchain.handle])
                        .image_indices(&[image_index as u32]),
                )?
            };
            p.image_acquired.set(false);
            p.suboptimal_swapchain.set(p.suboptimal_swapchain.get() || suboptimal);
        }
        Result::Ok(())
    }

    pub fn read_buffer(&self, buffer: &Buffer, data: &mut [u8], offset: u64) -> Result<()> {
        let (staging_buffer, ptr) = self.frame().belt.borrow_mut().download(self, data.len() as u64)?;
        self.copy_buffer(
            buffer,
            &staging_buffer,
            &[vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size: data.len() as u64,
            }],
        );
        self.submit_wait()?;
        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        data.copy_from_slice(read);
        Result::Ok(())
    }

    pub fn read_image(
        &self,
        image: &Image,
        data: &mut [u8],
        offset: vk::Offset3D,
        extent: vk::Extent3D,
        bytes_per_pixel: u64,
        subresource: vk::ImageSubresourceLayers,
    ) -> Result<()> {
        let size = extent.width as u64 * extent.height as u64 * extent.depth as u64 * bytes_per_pixel * subresource.layer_count as u64;
        if size != data.len() as u64 {
            return Err(anyhow!("Data buffer size does not match image region size"));
        }
        let (staging_buffer, ptr) = self.frame().belt.borrow_mut().download(self, size)?;

        self.record(|dev, cmd| unsafe {
            let prev = self.barrier_image(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            dev.cmd_copy_image_to_buffer(
                cmd,
                image.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buffer.handle,
                &[vk::BufferImageCopy::default()
                    .image_offset(offset)
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: extent.depth,
                    })
                    .image_subresource(subresource)],
            );
            self.barrier_image(cmd, image, prev);
        });
        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        self.submit_wait()?;
        data.copy_from_slice(read);
        Result::Ok(())
    }

    pub fn write_buffer<T>(&self, buffer: &Buffer, data: &[T], offset: u64) -> Result<()> {
        let (staging_buf, cursor, size) = self.frame().belt.borrow_mut().upload(self, bytes_of(data))?;
        self.record(|dev, cmd| unsafe {
            dev.cmd_copy_buffer(
                cmd,
                staging_buf.handle,
                buffer.handle,
                &[vk::BufferCopy::default().src_offset(cursor).dst_offset(offset).size(size)],
            );
            self.barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
        Result::Ok(())
    }

    pub fn write_image<T>(
        &self,
        image: &Image,
        data: &[T],
        offset: vk::Offset3D,
        extent: vk::Extent3D,
        subresource: vk::ImageSubresourceLayers,
        new_layout: Option<vk::ImageLayout>,
    ) -> Result<()> {
        let (staging_buf, cursor, size) = self.frame().belt.borrow_mut().upload(self, bytes_of(data))?;
        self.record(|dev, cmd| unsafe {
            let prev = self.barrier_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            dev.cmd_copy_buffer_to_image(
                cmd,
                staging_buf.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(cursor)
                    .image_subresource(subresource)
                    .image_offset(offset)
                    .image_extent(extent)],
            );
            self.barrier_image(cmd, image, new_layout.unwrap_or(prev));
        });
        Result::Ok(())
    }

    pub fn init_image<T>(&self, image: &Image, data: &[T]) {
        self.write_image(
            image,
            data,
            vk::Offset3D::default(),
            image.extent,
            vk::ImageSubresourceLayers::default().aspect_mask(image.aspect).layer_count(1),
            Some(if image.usage.contains(vk::ImageUsageFlags::STORAGE) {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            }),
        );
    }

    pub fn copy_buffer(&self, src_buf: &Buffer, dst_buf: &Buffer, regions: &[vk::BufferCopy]) {
        self.record(|dev, cmd| unsafe {
            self.barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            dev.cmd_copy_buffer(cmd, src_buf.handle, dst_buf.handle, regions);
            self.barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
    }

    pub fn copy_image(&self, src_img: &Image, dst_img: &Image, regions: &[vk::ImageCopy]) {
        self.record(|dev, cmd| unsafe {
            let prev1 = self.barrier_image(cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let prev2 = self.barrier_image(cmd, dst_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

            dev.cmd_copy_image(
                cmd,
                src_img.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_img.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );

            self.barrier_image(cmd, src_img, prev1);
            self.barrier_image(cmd, dst_img, prev2);
        });
    }

    pub fn copy_buffer_image(&self, src_buf: &Buffer, dst_img: &Image, regions: &[vk::BufferImageCopy]) {
        self.record(|dev, cmd| unsafe {
            self.barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            let prev = self.barrier_image(cmd, dst_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            dev.cmd_copy_buffer_to_image(cmd, src_buf.handle, dst_img.handle, vk::ImageLayout::TRANSFER_DST_OPTIMAL, regions);
            self.barrier_image(cmd, dst_img, prev);
            self.barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
    }

    pub fn copy_image_buffer(&self, src_img: &Image, dst_buf: &Buffer, regions: &[vk::BufferImageCopy]) {
        self.record(|dev, cmd| unsafe {
            self.barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            let prev = self.barrier_image(cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            dev.cmd_copy_image_to_buffer(cmd, src_img.handle, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst_buf.handle, regions);
            self.barrier_image(cmd, src_img, prev);
            self.barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
    }

    pub fn fill_buffer(&self, buffer: &Buffer, clear_value: u32, offset: u64, size: u64) {
        self.record(|dev, cmd| unsafe {
            self.barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            dev.cmd_fill_buffer(cmd, buffer.handle, offset, size, clear_value);
            self.barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
    }

    pub fn clear_color_image(&self, image: &Image, color: vk::ClearColorValue, range: vk::ImageSubresourceRange) {
        self.record(|dev, cmd| unsafe {
            let prev = self.barrier_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            dev.cmd_clear_color_image(cmd, image.handle, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, &[range]);
            self.barrier_image(cmd, image, prev);
        });
    }
}

impl Drop for RenderingDeviceImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            log::info!("Destroying device");
            if let Some(debug_utils) = &self.debug_utils {
                debug_utils.instance.destroy_debug_utils_messenger(debug_utils.messenger, None);
            }

            for frame in self.frames.iter() {
                self.device.destroy_fence(frame.fence, None);
                self.device.destroy_semaphore(frame.image_semaphore, None);
                self.device.destroy_command_pool(frame.cmd_pool, None);

                for chunk in frame.belt.borrow_mut().active_chunks.iter_mut() {
                    Rc::get_mut(&mut chunk.buffer).unwrap().destroy(self);
                }
                if let Some(mut readback) = frame.belt.borrow_mut().readback_buffer.take() {
                    Rc::get_mut(&mut readback).unwrap().destroy(self);
                }
            }
            if let Some(p) = &self.presentation {
                for &sem in p.semaphores.iter() {
                    self.device.destroy_semaphore(sem, None);
                }
                let swapchain = p.swapchain.borrow();
                for image in swapchain.images.iter() {
                    for view in image.views.borrow().values() {
                        self.device.destroy_image_view(view.handle, None);
                    }
                }
                swapchain.device.destroy_swapchain(swapchain.handle, None);
                p.surface.instance.destroy_surface(p.surface.handle, None);
            }
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Handles Vulkan debug callbacks and messenger.
pub struct DebugUtils {
    pub instance: debug_utils::Instance,
    pub device: debug_utils::Device,
    pub messenger: vk::DebugUtilsMessengerEXT,
}

pub fn make_debug_utils(instance: &ash::Instance, device: &ash::Device) -> Result<DebugUtils> {
    unsafe {
        let debug_inst = debug_utils::Instance::new(&ENTRY, instance);
        let debug_dev = debug_utils::Device::new(instance, device);
        let messenger = debug_inst.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::INFO)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                .pfn_user_callback(Some(self::vulkan_debug_callback)),
            None,
        )?;
        Result::Ok(DebugUtils {
            instance: debug_inst,
            device: debug_dev,
            messenger,
        })
    }
}

pub extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    log::error!("[{:?} {:?}] {:?}", message_severity, message_types, message);
    vk::FALSE
}
