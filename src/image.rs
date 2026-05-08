use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::Weak;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;
use parking_lot::Mutex;

use crate::ImageDesc;
use crate::RenderingDevice;
use crate::SharedDevice;
use crate::hash_struct;
use crate::next_resource_id;

/// A wrapper around a Vulkan image resource, providing additional metadata and caching for image views.
///
/// The `Image` struct holds the Vulkan image handle, its format, extent, usage, aspect mask, sample count.
/// It also caches the image views to avoid redundant Vulkan calls.
#[derive(Clone)]
#[repr(transparent)]
pub struct Image(Arc<ImageImpl>);

impl Deref for Image {
    type Target = Arc<ImageImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageImpl {
    pub raw: vk::Image,
    pub id: u64,
    pub alloc: Allocation,
    device: Arc<SharedDevice>,

    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,
    pub optimal_layout: vk::ImageLayout,

    full_view: OnceLock<ImageView>,
    pub(crate) views: Mutex<HashMap<u64, ImageView>>,
}

impl Drop for ImageImpl {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.device.raw;
            for view in self.views.lock().values() {
                dev.destroy_image_view(view.raw, None);
            }
            let alloc = std::mem::take(&mut self.alloc);
            if !alloc.is_null() {
                dev.destroy_image(self.raw, None);
                self.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }
}

impl Image {
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(self.aspect)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
    }

    /// Gets or creates a view encompassing the entire image (all mips and layers).
    pub fn full_view(&self) -> &ImageView {
        self.full_view.get_or_init(move || self.view_range(self.full_range()))
    }

    /// Gets or creates a view for a specific mip level and array layer.
    #[inline]
    pub fn view(&self, aspect: vk::ImageAspectFlags, mip_level: u32, layer: u32) -> ImageView {
        self.view_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(mip_level)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1),
        )
    }

    /// Gets or creates a view covering a custom `vk::ImageSubresourceRange`.
    pub fn view_range(&self, range: vk::ImageSubresourceRange) -> ImageView {
        self.view_raw(
            &vk::ImageViewCreateInfo::default()
                .image(self.raw)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.format)
                .subresource_range(range),
        )
    }

    pub fn view_raw(&self, info: &vk::ImageViewCreateInfo) -> ImageView {
        let hash = hash_struct(info);
        let mut views = self.views.lock();
        if let Some(view) = views.get(&hash) {
            return view.clone();
        }
        let raw = unsafe { self.device.raw.create_image_view(info, None).unwrap() };
        let view = ImageView {
            raw,
            id: next_resource_id(),
            image: Arc::downgrade(&self.0),
        };
        views.insert(hash, view.clone());
        view
    }
}

#[derive(Clone)]
pub struct ImageView {
    pub raw: vk::ImageView,
    pub id: u64,
    image: Weak<ImageImpl>,
}

impl ImageView {
    pub fn image(&self) -> Option<Image> {
        self.image.upgrade().map(Image)
    }

    pub fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: self.raw, image_layout: vk::ImageLayout::GENERAL }
    }
}

fn conv_format_to_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub fn find_optimal_layout(usage: vk::ImageUsageFlags) -> vk::ImageLayout {
    if usage.contains(vk::ImageUsageFlags::STORAGE) {
        vk::ImageLayout::GENERAL
    } else if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}


impl RenderingDevice {
    /// Creates an image based on the provided description, allocating memory and binding it.
    pub fn new_image(&self, desc: &ImageDesc) -> Image {
        self.new_image_info(
            vk::ImageCreateInfo::default()
                .image_type(if desc.depth == 1 { vk::ImageType::TYPE_2D } else { vk::ImageType::TYPE_3D })
                .format(desc.format)
                .extent(vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: desc.depth,
                })
                .mip_levels(desc.mip_levels)
                .array_layers(desc.array_layers)
                .samples(vk::SampleCountFlags::from_raw(desc.samples))
                .tiling(desc.tiling)
                .usage(desc.usage)
                .flags(desc.flags),
            desc.location,
        )
    }

    /// Creates an image and allocate memory for it from a `vk::ImageCreateInfo` and `MemoryLocation`.
    pub fn new_image_info(&self, mut info: vk::ImageCreateInfo, location: MemoryLocation) -> Image {
        unsafe {
            info.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
            let image = self.raw.create_image(&info, None).expect("Failed to create image");
            let mem_reqs = self.raw.get_image_memory_requirements(image);
            let alloc = self
                .shared
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "vka_image",
                    requirements: mem_reqs,
                    location,
                    linear: info.tiling == vk::ImageTiling::LINEAR,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap();
            self.raw.bind_image_memory(image, alloc.memory(), alloc.offset()).expect("Failed to bind image memory");
            let res = self.new_image_raw(image, info.format, info.extent, info.samples, info.usage, Some(alloc));
            {
                let mut cmd = self.new_command_buffer();
                cmd.image_barrier_raw(res.raw, res.aspect, vk::ImageLayout::UNDEFINED, res.optimal_layout);
                self.submit([cmd], None);
            }
            res
        }
    }

    pub fn new_image_raw(
        &self,
        image: vk::Image,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        alloc: Option<Allocation>,
    ) -> Image {
        let aspect = conv_format_to_aspect_mask(format);
        let optimal_layout = find_optimal_layout(usage);
        let inner = ImageImpl {
            raw: image,
            id: next_resource_id(),
            alloc: alloc.unwrap_or_default(),
            device: self.shared.clone(),

            format,
            extent,
            usage,
            aspect,
            samples,
            optimal_layout,

            full_view: OnceLock::new(),
            views: Mutex::new(HashMap::new()),
        };
        Image(Arc::new(inner))
    }

}
