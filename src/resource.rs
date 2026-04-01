use std::cell::OnceCell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::rc::Rc;
use std::rc::Weak;

use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::RenderingDevice;
use crate::RenderingDeviceImpl;
use crate::image::Image;

/// A ref-counted handle to a GPU resource, such as a buffer or an image.
/// It holds the resource value, its allocation (if any), and a weak reference to the rendering device for cleanup purposes.
pub type Handle<T> = Rc<Resource<T>>;

pub struct Resource<T> {
    pub value: T,
    pub alloc: Allocation,
    pub device: Weak<RenderingDeviceImpl>,
    name: OnceCell<String>,
    dtor: Option<Box<dyn FnMut(&mut Self, &RenderingDeviceImpl)>>,
}

impl<T> Deref for Resource<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> Resource<T> {
    /// Spawns a new ref-counted handling mapping `value`, an allocation, and its destructor closure.
    pub fn new(rd: &RenderingDevice, value: T, alloc: Option<Allocation>, dtor: impl FnMut(&mut Resource<T>, &RenderingDeviceImpl) + 'static) -> Handle<T> {
        let device = Rc::downgrade(&rd.0);
        Rc::new(Self {
            value,
            alloc: alloc.unwrap_or(Allocation::default()),
            device,
            name: OnceCell::new(),
            dtor: Some(Box::new(dtor)),
        })
    }

    pub fn internal(rd: &RenderingDevice, value: T) -> Handle<T> {
        Self::new(rd, value, None, |_, _| {})
    }

    pub fn get_name(&self) -> &str {
        self.name.get().map(|s| s.as_str()).unwrap_or("unnamed")
    }

    pub fn set_name(&self, name: impl Into<String>) {
        self.name.set(name.into());
    }

    pub fn rendering_device(&self) -> Option<RenderingDevice> {
        self.device.upgrade().map(RenderingDevice)
    }

    pub fn alloc(&self) -> &Allocation {
        &self.alloc
    }

    /// Returns a mutable slice to the mapped memory if the resource is host-visible, or None otherwise.
    pub fn mapped_mut(&self) -> Option<&mut [u8]> {
        self.alloc
            .mapped_ptr()
            .map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.cast().as_ptr(), self.alloc.size() as usize) })
    }

    /// Immediately invokes the destructor and frees the resource and memory.
    pub fn destroy(&mut self, rd: &RenderingDeviceImpl) {
        let res_name = self.name.get().map_or(std::any::type_name::<T>(), |s| s.as_str()).to_string();
        log::debug!("Dropping resource {}", res_name);
        // hmm, not the best way, maybe we can use deferred cleanup.
        unsafe {
            rd.device.queue_wait_idle(rd.graphics_queue).unwrap();
        }

        let alloc = std::mem::take(&mut self.alloc);
        if let Some(mut dtor) = self.dtor.take() {
            dtor(self, &rd);
        }
        if !alloc.is_null() {
            rd.allocator.lock().unwrap().free(alloc);
        }
    }
}

impl<T> Drop for Resource<T> {
    fn drop(&mut self) {
        if let Some(rd) = self.device.upgrade() {
            self.destroy(&rd);
        }
    }
}

impl<T: Debug> fmt::Debug for Resource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resource").field("handle", &self.value).field("name", &self.get_name()).finish()
    }
}

impl RenderingDevice {
    /// Records a generic global memory pipeline barrier.
    pub fn barrier(&self, cmd: vk::CommandBuffer, src_stages: vk::PipelineStageFlags, dst_stages: vk::PipelineStageFlags) {
        unsafe {
            // most of the driver implementations filter access flags based on the pipeline stages.
            // so specifying MEMORY_READ/MEMORY_WRITE is sufficient.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            self.device
                .cmd_pipeline_barrier(cmd, src_stages, dst_stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
    }

    /// Transitions an image to a new layout and records the corresponding image memory pipeline barrier.
    pub fn barrier_image(&self, cmd: vk::CommandBuffer, image: &Image, mut new_layout: vk::ImageLayout) -> vk::ImageLayout {
        // Transitioning to undefined is not valid, so we treat it as general layout
        // also might be triggered by the first image write operations due to how we handle it.
        if new_layout == vk::ImageLayout::UNDEFINED || new_layout == vk::ImageLayout::PREINITIALIZED {
            if image.usage.contains(vk::ImageUsageFlags::STORAGE) {
                new_layout = vk::ImageLayout::GENERAL // since this is a must for storage images.
            } else {
                new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            }
        }
        let old_layout = image.layout.get();
        self.barrier_image_from(cmd, image.handle, image.aspect, old_layout, new_layout);
        image.layout.set(new_layout);
        old_layout
    }

    pub fn barrier_image_from(&self, cmd: vk::CommandBuffer, image: vk::Image, aspect_mask: vk::ImageAspectFlags, old_layout: vk::ImageLayout, mut new_layout: vk::ImageLayout) -> vk::ImageLayout {
        unsafe {
            if new_layout == vk::ImageLayout::UNDEFINED || new_layout == vk::ImageLayout::PREINITIALIZED {
                new_layout = vk::ImageLayout::GENERAL; // we have no choice
            }
            let (src_stages, src_access) = match old_layout {
                vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
                vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::SHADER_READ | vk::AccessFlags::INPUT_ATTACHMENT_READ),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE),
                vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::empty()),
                _ => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE),
            };

            let (dst_stages, dst_access) = match new_layout {
                vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::SHADER_READ | vk::AccessFlags::INPUT_ATTACHMENT_READ),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE),
                vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::empty()),
                _ => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
            };

            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            self.device
                .cmd_pipeline_barrier(cmd, src_stages, dst_stages, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
            new_layout
        }
    }
}
