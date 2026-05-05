use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;
use parking_lot::Mutex;

use crate::BufferDesc;
use crate::RenderingDevice;
use crate::SharedDevice;
use crate::next_resource_id;

/// A reference-counted wrapper around a Vulkan buffer resource.
#[derive(Clone)]
#[repr(transparent)]
pub struct Buffer(Arc<BufferImpl>);

pub struct BufferImpl {
    pub raw: vk::Buffer,
    pub id: u64,
    pub alloc: Allocation,
    device: Arc<SharedDevice>,

    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    views: Mutex<HashMap<u64, vk::BufferView>>,
}

impl Drop for BufferImpl {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.device.raw;
            for view in self.views.lock().values() {
                dev.destroy_buffer_view(*view, None);
            }
            let alloc = std::mem::take(&mut self.alloc);
            if !alloc.is_null() {
                dev.destroy_buffer(self.raw, None);
                self.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }
}

impl Buffer {
    pub fn descriptor(&self, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo { buffer: self.raw, offset, range }
    }
}

impl Deref for Buffer {
    type Target = Arc<BufferImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RenderingDevice {
    /// Creates a buffer based on the provided description, allocating memory and binding it.
    pub fn new_buffer(&self, desc: &BufferDesc) -> Buffer {
        self.new_buffer_info(vk::BufferCreateInfo::default().size(desc.size).usage(desc.usage).sharing_mode(vk::SharingMode::EXCLUSIVE), desc.location)
    }

    pub fn new_buffer_info(&self, mut info: vk::BufferCreateInfo, location: MemoryLocation) -> Buffer {
        unsafe {
            info.usage |= vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = self.raw.create_buffer(&info, None).expect("Failed to create buffer");
            let mem_reqs = self.raw.get_buffer_memory_requirements(buffer);
            let alloc = self
                .shared
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "vka_buf",
                    requirements: mem_reqs,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged
                })
                .unwrap();
            self.raw.bind_buffer_memory(buffer, alloc.memory(), alloc.offset()).expect("Failed to bind buffer memory");
            self.new_buffer_raw(buffer, info.size, info.usage, Some(alloc))
        }
    }

    pub fn new_buffer_raw(&self, buffer: vk::Buffer, size: u64, usage: vk::BufferUsageFlags, alloc: Option<Allocation>) -> Buffer {
        let inner = BufferImpl {
            id: next_resource_id(),
            raw: buffer,
            alloc: alloc.unwrap_or_default(),
            device: self.shared.clone(),

            size,
            usage,
            views: Mutex::new(HashMap::new()),
        };
        Buffer(Arc::new(inner))
    }

    /// Creates a buffer view for a specific region of the given buffer.
    /// Views are cached internally by the buffer to avoid redundant creations.
    pub fn new_buffer_view(&self, buffer: &Buffer, format: vk::Format, offset: u64, range: u64) -> vk::BufferView {
        self.new_buffer_view_with(
            buffer,
            &vk::BufferViewCreateInfo::default()
                .buffer(buffer.raw)
                .format(format)
                .offset(crate::align_up(offset, self.properties.limits.min_texel_buffer_offset_alignment))
                .range(range),
        )
    }

    pub fn new_buffer_view_with(&self, buffer: &Buffer, info: &vk::BufferViewCreateInfo) -> vk::BufferView {
        let hash = crate::hash_struct(info);
        if let Some(view) = buffer.views.lock().get(&hash) {
            return *view;
        }
        let view = unsafe { self.raw.create_buffer_view(info, None).expect("Failed to create buffer view") };
        buffer.views.lock().insert(hash, view);
        view
    }
}
