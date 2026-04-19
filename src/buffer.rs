use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::BufferDesc;
use crate::Handle;
use crate::RenderingDevice;
use crate::Resource;
use crate::Result;
use crate::bytes_of;
use crate::utils;

/// A reference-counted wrapper around a Vulkan buffer resource.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Buffer(Handle<BufferImpl>);

impl Buffer {
    pub fn descriptor(&self, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo { buffer: self.handle, offset, range }
    }
}

impl Deref for Buffer {
    type Target = Handle<BufferImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Inner state of a Vulkan buffer, managing its handle, size, usage, and cached views.
#[derive(Debug)]
pub struct BufferImpl {
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub views: RefCell<HashMap<u64, vk::BufferView>>,
}

impl RenderingDevice {
    /// Creates a buffer based on the provided description, allocating memory and binding it.
    pub fn buffer_create(&self, desc: &BufferDesc) -> Result<Buffer> {
        self.buffer_from_info(vk::BufferCreateInfo::default().size(desc.size).usage(desc.usage).sharing_mode(vk::SharingMode::EXCLUSIVE), desc.location)
    }

    pub fn buffer_from_info(&self, mut info: vk::BufferCreateInfo, location: MemoryLocation) -> Result<Buffer> {
        unsafe {
            let alignment = if info.usage.contains(vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER) {
                self.properties.limits.min_texel_buffer_offset_alignment
            } else if info.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                self.properties.limits.min_uniform_buffer_offset_alignment
            } else if info.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                self.properties.limits.min_storage_buffer_offset_alignment
            } else {
                4
            };
            info.size = utils::align_up(info.size, alignment);
            info.usage |= vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = self.device.create_buffer(&info, None)?;
            let mem_reqs = self.device.get_buffer_memory_requirements(buffer);
            let alloc = self
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
            self.device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
            Result::Ok(self.buffer_from_raw(buffer, info.size, info.usage, Some(alloc)))
        }
    }

    pub fn buffer_from_raw(&self, buffer: vk::Buffer, size: u64, usage: vk::BufferUsageFlags, alloc: Option<Allocation>) -> Buffer {
        Buffer(Resource::new(
            self,
            BufferImpl {
                handle: buffer,
                size,
                usage,
                views: RefCell::new(HashMap::new()),
            },
            alloc,
            |res, rd| unsafe {
                for view in res.views.borrow().values() {
                    rd.device.destroy_buffer_view(*view, None);
                }
                rd.device.destroy_buffer(res.value.handle, None);
            },
        ))
    }

    /// Creates a buffer view for a specific region of the given buffer.
    /// Views are cached internally by the buffer to avoid redundant creations.
    pub fn buffer_view(&self, buffer: &Buffer, format: vk::Format, offset: u64, range: u64) -> Result<vk::BufferView> {
        self.buffer_view_with(
            buffer,
            &vk::BufferViewCreateInfo::default()
                .buffer(buffer.handle)
                .format(format)
                .offset(utils::align_up(offset, self.properties.limits.min_texel_buffer_offset_alignment))
                .range(range),
        )
    }

    pub fn buffer_view_with(&self, buffer: &Buffer, info: &vk::BufferViewCreateInfo) -> Result<vk::BufferView> {
        let hash = utils::hash_struct(info);
        if let Some(view) = buffer.views.borrow().get(&hash) {
            return Result::Ok(*view);
        }
        let view = unsafe { self.device.create_buffer_view(info, None)? };
        buffer.views.borrow_mut().insert(hash, view);
        Result::Ok(view)
    }
}
