use std::{ffi::CStr, mem::ManuallyDrop, ops::Deref};

use ash::vk;
use ash::{ext::debug_utils, prelude::VkResult};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use itertools::Itertools;
use parking_lot::lock_api::RawMutex;
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, Mutex as StdMutex};

use crate::Surface;
use crate::{Buffer, CommandEncoder, Image, RenderingDeviceDesc, SurfaceImage, belt::StagingBelt};

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

#[allow(dead_code)]
struct DeviceExtensions {
    pub debug_utils: Option<DebugUtils>,
    pub mesh_shader: Option<ash::ext::mesh_shader::Device>,
    pub ray_tracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    pub buffer_device_address: Option<ash::khr::buffer_device_address::Device>,
}

#[derive(Debug, Clone, Default)]
pub struct Features {
    pub core10: vk::PhysicalDeviceFeatures,
    pub core11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub core12: vk::PhysicalDeviceVulkan12Features<'static>,
    pub core13: vk::PhysicalDeviceVulkan13Features<'static>,

    pub mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
    pub ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
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

    pub features: Features,

    pub enabled_extensions: Vec<&'static CStr>,
    pub enabled_layers: Vec<&'static CStr>,
    pub enabled_instance_exts: Vec<&'static CStr>,

    extensions: DeviceExtensions,
    pub queue_families: QueueFamilies,
    pub main_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub device_mutex: parking_lot::RawMutex,

    staging_belt: Mutex<StagingBelt>,
    pub frames: Vec<Mutex<Frame>>,
    pub frame_counter: RwLock<(u64, usize)>,
}

pub struct EncoderInFlight {
    pub inner: Box<CommandEncoder>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
    pub pending: bool,
}

pub struct Frame {
    pub wait_semaphore: Option<vk::Semaphore>,
    pub signal_semaphore: Option<vk::Semaphore>,

    pub encoders: Vec<EncoderInFlight>,
    pub all_cmd_buffers: Vec<vk::CommandBuffer>,

    pub fence: vk::Fence,
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
            log::info!(
                "Found devices: {}",
                found_devices
                    .clone()
                    .iter()
                    .map(|f| format!("{:?}[{:?}]", &f.1.device_name_as_c_str().unwrap(), f.1.device_type))
                    .join(", ")
            );

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
                .expect("No Vulkan-compatible devices found!");

            if let Some(idx) = desc.pick_device {
                if let Some((pd, props)) = found_devices.get(idx) {
                    log::info!("Picking device at specified index {}", idx);
                    phy_device = *pd;
                    properties = *props;
                } else {
                    log::warn!("Specified device index {} is out of bounds, ignoring", idx);
                }
            }

            let mut features10 = vk::PhysicalDeviceFeatures2::default();
            let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();

            let mut features_mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut features_ray_tracing_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
            let mut features_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut features_acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();

            features10 = features10
                .push_next(&mut features11)
                .push_next(&mut features12)
                .push_next(&mut features13)
                .push_next(&mut features_mesh_shader)
                .push_next(&mut features_ray_tracing_pipeline)
                .push_next(&mut features_ray_query)
                .push_next(&mut features_acceleration_structure);
            instance.get_physical_device_features2(phy_device, &mut features10);
            features10.features.robust_buffer_access &= desc.gpu_validation as u32;

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
                    .push_next(&mut features10),
                None,
            )?;
            let mem_properties = instance.get_physical_device_memory_properties(phy_device);
            let debug_utils = if validation_layers_enabled {
                Some(make_debug_utils(&entry, &instance, &device)?)
            } else {
                None
            };

            let main_queue = device.get_device_queue(queue_families.graphics, 0);
            let present_queue = device.get_device_queue(queue_families.present, 0);

            let mut features = Features::default();
            features.core10 = features10.features;
            features.core11 = features11;
            features.core12 = features12;
            features.core13 = features13;
            features.mesh_shader = features_mesh_shader;
            features.ray_tracing_pipeline = features_ray_tracing_pipeline;
            features.ray_query = features_ray_query;
            features.acceleration_structure = features_acceleration_structure;

            let extensions = DeviceExtensions {
                debug_utils,
                mesh_shader: enabled_device_exts
                    .contains(&ash::ext::mesh_shader::NAME)
                    .then(|| ash::ext::mesh_shader::Device::new(&instance, &device)),
                ray_tracing_pipeline: enabled_device_exts
                    .contains(&ash::khr::ray_tracing_pipeline::NAME)
                    .then(|| ash::khr::ray_tracing_pipeline::Device::new(&instance, &device)),
                acceleration_structure: enabled_device_exts
                    .contains(&ash::khr::acceleration_structure::NAME)
                    .then(|| ash::khr::acceleration_structure::Device::new(&instance, &device)),
                buffer_device_address: (features.core12.buffer_device_address == 1).then(|| ash::khr::buffer_device_address::Device::new(&instance, &device)),
            };

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

            let shared = Arc::new(SharedDevice {
                raw: device.clone(),
                entry,
                instance,
                allocator: ManuallyDrop::new(allocator),
            });

            if let Some((surface, surface_inst)) = &surface {
                surface_inst.destroy_surface(*surface, None);
            }

            let frames = (0..desc.n_frames)
                .map(|_| {
                    Mutex::new(Frame {
                        wait_semaphore: None,
                        signal_semaphore: None,
                        encoders: Vec::new(),
                        all_cmd_buffers: Vec::new(),

                        fence: device
                            .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                            .expect("Failed to create frame fence"),
                    })
                })
                .collect_vec();
            let frame_counter = RwLock::new((0, 0));

            let rd = RenderingDevice(Arc::new(RenderingDeviceImpl {
                raw: device,
                shared,
                phy_device,
                properties,
                mem_properties,
                features,

                enabled_extensions: enabled_device_exts,
                enabled_layers,
                enabled_instance_exts,

                extensions,

                queue_families,
                main_queue,
                present_queue,

                device_mutex: RawMutex::INIT,

                staging_belt: Mutex::new(StagingBelt::new(4 * 1024 * 1024)),
                frames,
                frame_counter,
            }));
            rd.wait_idle();
            Result::Ok(rd)
        }
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn reset_frames(&self) {
        *self.frame_counter.write() = (0, 0);
        for frame in &self.frames {
            let mut frame = frame.lock();
            unsafe {
                self.raw.destroy_fence(frame.fence, None);
                frame.fence = self
                    .raw
                    .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                    .expect("Failed to create frame fence");
            }
            for mut encoder in frame.encoders.drain(..) {
                encoder.inner.reset(&encoder.cmd_buffers);
                encoder.cmd_buffers.clear();
                encoder.pending = false;
            }
            frame.all_cmd_buffers.clear();
            frame.wait_semaphore = None;
            frame.signal_semaphore = None;
        }
    }

    pub fn wait_internal_frame(&self, frame: &mut Frame) {
        unsafe {
            self.raw.wait_for_fences(&[frame.fence], true, u64::MAX).unwrap();
        }
        for encoder in frame.encoders.iter_mut() {
            if encoder.pending {
                encoder.inner.reset(&encoder.cmd_buffers);
                encoder.cmd_buffers.clear();
                encoder.pending = false;
            }
        }
    }

    pub fn record_frame(&self, surface: &mut Surface, record_fn: impl FnOnce(&mut CommandEncoder, SurfaceImage)) {
        let frame_idx = self.frame_counter.read();
        let mut frame = self.frames[frame_idx.1].lock();
        self.wait_internal_frame(&mut frame);

        if let Some(image) = unsafe { surface.acquire_next_image_raw(frame_idx.1) } {
            frame.wait_semaphore = Some(surface.acquire_semaphores[frame_idx.1]);

            let mut encoder = self.pick_encoder(&mut frame);
            drop(frame);
            encoder.inner.begin_encoding();

            encoder
                .inner
                .image_barrier_raw(image.inner.raw, image.inner.aspect, vk::ImageLayout::PRESENT_SRC_KHR, image.inner.optimal_layout);
            record_fn(&mut encoder.inner, image.clone());
            encoder
                .inner
                .image_barrier_raw(image.inner.raw, image.inner.aspect, image.inner.optimal_layout, vk::ImageLayout::PRESENT_SRC_KHR);

            let cmd = encoder.inner.end_encoding();
            encoder.cmd_buffers.push(cmd);

            let mut frame = self.frames[frame_idx.1].lock();
            frame.signal_semaphore = Some(surface.present_semaphores[image.index as usize]);
            frame.all_cmd_buffers.push(cmd);
            frame.encoders.push(encoder);
        } else {
            drop(frame);
            drop(frame_idx);
            log::error!("Failed to acquire next image, recreating swapchain");
            surface.recreate_swapchain();
        }
    }

    pub fn record(&self, record_fn: impl FnOnce(&mut CommandEncoder)) {
        let frame_idx = self.frame_counter.read();
        let mut frame = self.frames[frame_idx.1].lock();
        self.wait_internal_frame(&mut frame);

        let mut encoder = self.pick_encoder(&mut frame);
        drop(frame);

        encoder.inner.begin_encoding();
        record_fn(&mut encoder.inner);
        let cmd = encoder.inner.end_encoding();
        encoder.cmd_buffers.push(cmd);

        let mut frame = self.frames[frame_idx.1].lock();
        frame.all_cmd_buffers.push(cmd);
        frame.encoders.push(encoder);
    }

    fn pick_encoder(&self, frame: &mut Frame) -> EncoderInFlight {
        match frame.encoders.pop() {
            Some(e) => e,
            None => {
                log::info!("Creating new encoder");
                let inner = Box::new(CommandEncoder::new(&self.raw, self.queue_families.graphics).unwrap());
                EncoderInFlight { inner, cmd_buffers: Vec::new(), pending: false }
            }
        }
    }

    pub fn submit(&self) -> u64 {
        let frame_idx = self.frame_counter.read();
        let mut frame = self.frames[frame_idx.1].lock();

        self.wait_internal_frame(&mut frame);
        if frame.all_cmd_buffers.is_empty() {
            return frame_idx.0;
        }
        for encoder in frame.encoders.iter_mut() {
            encoder.pending = true;
        }

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let mut wait_semaphores = [vk::Semaphore::null()];
        let mut wait_count = 0;

        if let Some(wait) = frame.wait_semaphore.take() {
            wait_semaphores[0] = wait;
            wait_count = 1;
        }
        let mut signal_semaphores = [vk::Semaphore::null()];
        let mut signal_count = 0;

        if let Some(signal) = frame.signal_semaphore.take() {
            signal_semaphores[0] = signal;
            signal_count = 1;
        }
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores[..wait_count])
            .wait_dst_stage_mask(&wait_stages[..wait_count])
            .command_buffers(&frame.all_cmd_buffers)
            .signal_semaphores(&signal_semaphores[..signal_count]);

        unsafe {
            self.device_mutex.lock();
            self.raw.reset_fences(&[frame.fence]).unwrap();
            self.raw.queue_submit(self.main_queue, &[submit_info], frame.fence).unwrap();
            self.device_mutex.unlock();
        }
        frame.all_cmd_buffers.clear();
        frame_idx.0
    }

    pub fn advance_frame(&self) {
        let mut frame_idx = self.frame_counter.write();
        let completed_submission = frame_idx.0.saturating_sub(self.frames.len() as u64);
        self.staging_belt.lock().maintain(completed_submission);

        frame_idx.0 += 1;
        frame_idx.1 = frame_idx.0 as usize % self.frames.len();
    }

    pub fn wait_for(&self, frame_num: u64) {
        let frame_idx = self.frame_counter.read();
        if frame_num >= frame_idx.0.saturating_sub(self.n_frames() as u64) {
            let target = frame_num % self.frames.len() as u64;
            self.wait_internal_frame(&mut self.frames[target as usize].lock());
        }
    }

    /// Blocks until the main queue goes idle.
    pub fn wait_queue(&self) {
        unsafe {
            let _ = self.raw.queue_wait_idle(self.main_queue);
        }
    }

    /// Blocks until the device goes idle
    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
        }
    }

    pub fn read_buffer(&self, buffer: &Buffer, data: &mut [u8], offset: u64) {
        let (staging_buffer, ptr) = self.staging_belt.lock().download(self, data.len() as u64);
        self.record(|encoder| {
            encoder.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            encoder.copy_buffer(
                buffer,
                &staging_buffer,
                &[vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: 0,
                    size: data.len() as u64,
                }],
            );
            encoder.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
        self.submit();
        self.wait_queue();

        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        data.copy_from_slice(read);
    }

    pub fn read_image(&self, image: &Image, data: &mut [u8], offset: vk::Offset3D, extent: vk::Extent3D, bytes_per_pixel: u64, subresource: vk::ImageSubresourceLayers) {
        let size = extent.width as u64 * extent.height as u64 * extent.depth as u64 * bytes_per_pixel * subresource.layer_count as u64;
        assert!(size == data.len() as u64, "Data buffer size does not match image region size");

        let (staging_buffer, ptr) = self.staging_belt.lock().download(self, size);

        self.record(|encoder| {
            encoder.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            encoder.copy_image_to_buffer(
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
            encoder.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
        self.submit();
        self.wait_queue();

        let read = unsafe { std::slice::from_raw_parts(ptr, data.len()) };
        data.copy_from_slice(read);
    }

    pub fn write_buffer<T>(&self, buffer: &Buffer, data: &[T], offset: u64) {
        let (staging_buf, cursor, size) = self.staging_belt.lock().upload(self, crate::bytes_of(data));
        self.record(|cmd| {
            cmd.barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            cmd.copy_buffer(&staging_buf, buffer, &[vk::BufferCopy::default().src_offset(cursor).dst_offset(offset).size(size)]);
            cmd.barrier(vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS);
        });
    }

    pub fn write_image<T>(&self, image: &Image, data: &[T], offset: vk::Offset3D, extent: vk::Extent3D, subresource: vk::ImageSubresourceLayers) {
        let (staging_buf, cursor, _) = self.staging_belt.lock().upload(self, crate::bytes_of(data));
        self.record(|cmd| {
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
        });
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
            for frame in &self.frames {
                let frame = frame.lock();
                self.raw.destroy_fence(frame.fence, None);
            }
            self.frames.clear();
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
