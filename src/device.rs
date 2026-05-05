use std::{ffi::CStr, mem::ManuallyDrop, ops::Deref};

use ash::{ext::debug_utils, prelude::VkResult};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use itertools::Itertools;
use parking_lot::Mutex;
use std::sync::{Arc, Mutex as StdMutex};

use crate::{Buffer, CommandBuffer, Image, RelaySemaphores, RenderingDeviceDesc, SurfaceImage, TimelineFence, belt::StagingBelt};

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

#[derive(Default)]
struct TempSubmitInfo {
    pub wait_stages: Vec<vk::PipelineStageFlags>,
    pub wait: Vec<vk::Semaphore>,
    pub signal: Vec<vk::Semaphore>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
}

impl TempSubmitInfo {
    pub fn clear(&mut self) {
        self.wait_stages.clear();
        self.wait.clear();
        self.signal.clear();
        self.cmd_buffers.clear();
    }
}

#[allow(dead_code)]
struct DeviceExtensions {
    pub debug_utils: Option<DebugUtils>,
    pub mesh_shader: Option<ash::ext::mesh_shader::Device>,
    pub acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    pub buffer_device_address: Option<ash::khr::buffer_device_address::Device>,
}

pub struct SharedDevice {
    pub raw: ash::Device,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub allocator: ManuallyDrop<StdMutex<Allocator>>,
}

impl Drop for SharedDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            ManuallyDrop::drop(&mut self.allocator);
            self.raw.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The inner state of a Vulkan rendering device containing the instance, physical device, logical device, and other core resources.
pub struct RenderingDeviceImpl {
    pub raw: ash::Device,
    pub shared: Arc<SharedDevice>,
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

    extensions: DeviceExtensions,
    pub queue_families: QueueFamilies,
    pub main_queue: vk::Queue,
    pub present_queue: vk::Queue,

    cmd_buffers: Mutex<Vec<CommandBuffer>>,
    pending_cmd_buffers: Mutex<Vec<(u64, CommandBuffer)>>,

    relay_semaphores: Mutex<RelaySemaphores>,
    temp_submit_info: Mutex<TempSubmitInfo>,
    pub(crate) fence: Mutex<TimelineFence>,
    pub staging_belt: Mutex<StagingBelt>,

    pub submit_mutex: Mutex<u64>,
    pub device_mutex: Mutex<()>,
}

/// A reference-counted wrapper around `RenderingDeviceImpl`, providing convenient access to Vulkan operations.
#[derive(Clone)]
#[repr(transparent)]
pub struct RenderingDevice(pub(crate) Arc<RenderingDeviceImpl>);

impl Deref for RenderingDevice {
    type Target = RenderingDeviceImpl;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RenderingDevice {
    /// Initializes a new Vulkan rendering device, instances, and necessary Queues/Allocators according to `RenderingDeviceDesc`.
    pub fn new(desc: &RenderingDeviceDesc) -> VkResult<RenderingDevice> {
        unsafe {
            let entry = ash::Entry::load().expect("Failed to load Vulkan library");
            let vulkan_version = entry.try_enumerate_instance_version()?.unwrap_or(vk::API_VERSION_1_0);
            let enum_layer_props = entry.enumerate_instance_layer_properties()?;
            let enum_ext_props = entry.enumerate_instance_extension_properties(None)?;

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
            log::info!("App name: {}", desc.app_name.to_str().unwrap());
            log::info!("Vulkan Version: {}", crate::vulkan_version_str(vulkan_version));
            log::info!("Extensions: {}", enabled_instance_exts.iter().map(|&v| v.to_str().unwrap()).join(", "));
            log::info!("Layers: {}", enabled_layers.iter().map(|&v| v.to_str().unwrap()).join(", "));
            let instance = entry.create_instance(&instance_info, None)?;

            let mut enabled_device_exts = vec![vk::KHR_DYNAMIC_RENDERING_NAME];
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                enabled_device_exts.push(vk::KHR_PORTABILITY_SUBSET_NAME);
            }

            let found_devices = instance
                .enumerate_physical_devices()?
                .iter()
                .map(|&pd| (pd, instance.get_physical_device_properties(pd)))
                .collect_vec();
            log::info!("Found devices: {}", found_devices.clone().iter().map(|f| format!("{:?}[{:?}]", &f.1.device_name_as_c_str().unwrap(), f.1.device_type)).join(", "));

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
                .next().expect("No Vulkan-compatible devices found!");

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
                let surface = ash_window::create_surface(&entry, &instance, rdh, rwh, None)?;
                Some((surface, ash::khr::surface::Instance::new(&entry, &instance)))
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
                    if surface.1.get_physical_device_surface_support(phy_device, i, surface.0).unwrap_or(false) {
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
            log::info!("Picked device: {:?}[{:?}]", properties.device_name_as_c_str().unwrap(), properties.device_type);
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
            let debug_utils = if validation_layers_enabled { Some(make_debug_utils(&entry, &instance, &device)?) } else { None };

            let main_queue = device.get_device_queue(queue_families.graphics, 0);
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
            let extensions = DeviceExtensions {
                debug_utils,
                mesh_shader: enabled_device_exts
                    .contains(&ash::ext::mesh_shader::NAME)
                    .then(|| ash::ext::mesh_shader::Device::new(&instance, &device)),
                acceleration_structure: enabled_device_exts
                    .contains(&ash::khr::acceleration_structure::NAME)
                    .then(|| ash::khr::acceleration_structure::Device::new(&instance, &device)),
                buffer_device_address: (features12.buffer_device_address == 1).then(|| ash::khr::buffer_device_address::Device::new(&instance, &device)),
            };

            let relay_semaphores = Mutex::new(RelaySemaphores::new(&device));
            let temp_submit_info = Mutex::new(TempSubmitInfo::default());
            let fence = Mutex::new(TimelineFence::default());

            let shared = Arc::new(SharedDevice {
                raw: device.clone(),
                entry,
                instance,
                allocator: ManuallyDrop::new(allocator)
            });

            if let Some((surface, surface_inst)) = &surface {
                surface_inst.destroy_surface(*surface, None);
            }

            let rd = RenderingDevice(Arc::new(RenderingDeviceImpl {
                raw: device,
                shared,
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

                extensions,

                queue_families,
                main_queue,
                present_queue,

                cmd_buffers: Mutex::new(Vec::new()),
                pending_cmd_buffers: Mutex::new(Vec::new()),

                relay_semaphores,
                temp_submit_info,
                fence,
                staging_belt: Mutex::new(StagingBelt::new(4 * 1024 * 1024)),

                submit_mutex: Mutex::new(1),
                device_mutex: Mutex::new(()),
            }));
            rd.wait_idle();
            Result::Ok(rd)
        }
    }

    pub fn new_command_buffer(&self) -> CommandBuffer {
        let mut buffers = self.cmd_buffers.lock();
        let mut cmd = if let Some(cmd) = buffers.pop() {
            cmd
        } else {
            CommandBuffer::new(self, self.queue_families.graphics).unwrap()
        };
        cmd.begin();
        cmd
    }

    pub fn release_command_buffer(&self, mut buffer: CommandBuffer) {
        buffer.reset();
        self.cmd_buffers.lock().push(buffer);
    }


    fn maintain(&self) {
        let mut fence = self.fence.lock();
        fence.maintain(&self.raw);

        self.staging_belt.lock().maintain(fence.last_completed);

        let mut pending_buffers = self.pending_cmd_buffers.lock();
        let last_pending = pending_buffers.iter().enumerate().rev().find(|(_, val)| val.0 <= fence.last_completed);

        if let Some((index, _)) = last_pending {
            pending_buffers.drain(..=index).for_each(|(_, buf)| self.release_command_buffer(buf));
        }
    }

    pub fn submit<T: IntoIterator<Item = CommandBuffer>>(&self, cmd_buffers: T, image: Option<&SurfaceImage>) -> u64 {
        let mut temp = self.temp_submit_info.lock();
        temp.clear();

        let mut submit_lock = self.submit_mutex.lock();
        let submit_index = *submit_lock;

        for cmd in cmd_buffers.into_iter() {
            temp.cmd_buffers.push(cmd.raw);
            self.pending_cmd_buffers.lock().push((submit_index, cmd));
        }

        let relay = self.relay_semaphores.lock().advance(&self.raw);
        if let Some(wait) = relay.wait {
            temp.wait.push(wait);
            temp.wait_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
        }
        temp.signal.push(relay.signal);

        if let Some(image) = &image {
            let mut acquire = image.acquire.lock();
            acquire.last_used_submission = submit_index;
            if acquire.should_wait {
                temp.wait.push(acquire.acquire);
                temp.wait_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
                acquire.should_wait = false;
            }
            temp.signal.push(image.present.lock().signal(&self.raw));
        }

        self.maintain();

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&temp.cmd_buffers)
            .wait_dst_stage_mask(&temp.wait_stages)
            .wait_semaphores(&temp.wait)
            .signal_semaphores(&temp.signal);

        let fence_raw = self.fence.lock().add(&self.raw, submit_index);
        unsafe {
            self.raw.queue_submit(self.main_queue, &[submit_info], fence_raw).unwrap();
        }
        *submit_lock += 1;
        submit_index
    }

    pub fn wait_submission(&self, submission: u64) {
        self.fence.lock().wait_for(&self.raw, submission);
        self.maintain();
    }

    /// Blocks until the main queue goes idle.
    pub fn wait_queue(&self) {
        unsafe { let _ = self.raw.queue_wait_idle(self.main_queue); }
    }

    /// Blocks until the device goes idle
    pub fn wait_idle(&self) {
        unsafe { let _ = self.raw.device_wait_idle(); }
    }

    pub fn read_buffer(&self, buffer: &Buffer, data: &mut [u8], offset: u64) {
        let (staging_buffer, ptr) = self.staging_belt.lock().download(self, data.len() as u64);
        let mut cmd = self.new_command_buffer();
        cmd.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
        cmd.copy_buffer(
            buffer,
            &staging_buffer,
            &[vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size: data.len() as u64,
            }],
        );
        cmd.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        let id = self.submit([cmd.finish()], None);
        self.wait_submission(id);

        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        data.copy_from_slice(read);
    }

    pub fn read_image(
        &self,
        image: &Image,
        data: &mut [u8],
        offset: vk::Offset3D,
        extent: vk::Extent3D,
        bytes_per_pixel: u64,
        subresource: vk::ImageSubresourceLayers,
    ) {
        let size = extent.width as u64 * extent.height as u64 * extent.depth as u64 * bytes_per_pixel * subresource.layer_count as u64;
        assert!(size == data.len() as u64, "Data buffer size does not match image region size");

        let (staging_buffer, ptr) = self.staging_belt.lock().download(self, size);

        let mut cmd = self.new_command_buffer();
        cmd.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
        cmd.copy_image_to_buffer(
            image,
            &staging_buffer,
            &[vk::BufferImageCopy::default()
                .image_offset(offset)
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: extent.depth,
                })
                .image_subresource(subresource)],
        );
        cmd.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        let id = self.submit([cmd.finish()], None);
        self.wait_submission(id);

        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        data.copy_from_slice(read);
    }

    pub fn write_buffer<T>(&self, buffer: &Buffer, data: &[T], offset: u64) {
        let (staging_buf, cursor, size) = self.staging_belt.lock().upload(self, crate::bytes_of(data));
        let mut cmd = self.new_command_buffer();
        cmd.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
        cmd.copy_buffer(
            &staging_buf,
            buffer,
            &[vk::BufferCopy::default().src_offset(cursor).dst_offset(offset).size(size)],
        );
        cmd.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        self.submit([cmd.finish()], None);
    }

    pub fn write_image<T>(
        &self,
        image: &Image,
        data: &[T],
        offset: vk::Offset3D,
        extent: vk::Extent3D,
        subresource: vk::ImageSubresourceLayers,
    ) {
        let (staging_buf, cursor, _) = self.staging_belt.lock().upload(self, crate::bytes_of(data));
        let mut cmd = self.new_command_buffer();
        cmd.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
        cmd.copy_buffer_to_image(
            &staging_buf,
            image,
            &[vk::BufferImageCopy::default()
                .buffer_offset(cursor)
                .image_subresource(subresource)
                .image_offset(offset)
                .image_extent(extent)],
        );
        cmd.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        self.submit([cmd.finish()], None);
    }

    pub fn init_image<T>(&self, image: &Image, data: &[T]) {
        self.write_image(
            image,
            data,
            vk::Offset3D::default(),
            image.extent,
            vk::ImageSubresourceLayers::default().aspect_mask(image.aspect).layer_count(1),
        );
    }
}

impl Drop for RenderingDeviceImpl {
    fn drop(&mut self) {
        unsafe {
            log::info!("Destroying device");
            let _ = self.raw.device_wait_idle();
            if let Some(debug_utils) = &self.extensions.debug_utils {
                debug_utils.instance.destroy_debug_utils_messenger(debug_utils.messenger, None);
            }
            self.relay_semaphores.lock().destroy(&self.raw);
            self.fence.lock().destroy(&self.raw);
        }
    }
}

/// Handles Vulkan debug callbacks and messenger.
pub struct DebugUtils {
    pub instance: debug_utils::Instance,
    pub device: debug_utils::Device,
    pub messenger: vk::DebugUtilsMessengerEXT,
}

pub fn make_debug_utils(entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> VkResult<DebugUtils> {
    unsafe {
        let debug_inst = debug_utils::Instance::new(&entry, instance);
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
    log::error!("[VULKAN {:?} {:?}] {:?}", message_severity, message_types, message);
    vk::FALSE
}
