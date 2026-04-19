use std::cell::Cell;
use std::cell::OnceCell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;
use std::rc::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::Handle;
use crate::ImageDesc;
use crate::RenderingDevice;
use crate::Resource;
use crate::Result;
use crate::WeakHandle;
use crate::bytes_of;
use crate::utils;

/// A wrapper around a Vulkan image resource, providing additional metadata and caching for image views.
///
/// The `Image` struct holds the Vulkan image handle, its format, extent, usage, aspect mask, sample count, and layout.
/// It also caches the image views to avoid redundant Vulkan calls.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Image(Handle<ImageImpl>);

impl Deref for Image {
    type Target = Handle<ImageImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// The internal implementation of the `Image` resource, containing the Vulkan image handle and related metadata.
#[derive(Debug)]
pub struct ImageImpl {
    pub handle: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,

    full_view: OnceCell<ImageView>,
    pub(crate) views: RefCell<HashMap<u64, ImageView>>,

    pub layout: Cell<vk::ImageLayout>,
}

impl ImageImpl {
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(self.aspect)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
    }

    pub fn assume_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }
}

static VIEW_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct ImageView {
    pub handle: vk::ImageView,
    pub id: u64,
    image: WeakHandle<ImageImpl>,
}

impl ImageView {
    pub fn get_image(&self) -> Option<Image> {
        self.image.upgrade().map(Image)
    }

    pub fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: self.handle, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }
    }
}

pub fn conv_format_to_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

impl RenderingDevice {
    /// Creates an image based on the provided description, allocating memory and binding it.
    pub fn image_create(&self, desc: &ImageDesc) -> Result<Image> {
        self.image_from_info(
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
    pub fn image_from_info(&self, mut info: vk::ImageCreateInfo, location: MemoryLocation) -> Result<Image> {
        unsafe {
            info.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
            let image = self.device.create_image(&info, None)?;
            let mem_reqs = self.device.get_image_memory_requirements(image);
            let alloc = self
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
            self.device.bind_image_memory(image, alloc.memory(), alloc.offset())?;
            Result::Ok(self.image_from_raw(image, info.format, info.extent, info.samples, info.usage, Some(alloc)))
        }
    }

    pub fn image_from_raw(
        &self,
        image: vk::Image,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        alloc: Option<Allocation>,
    ) -> Image {
        let aspect = conv_format_to_aspect_mask(format);
        Image(Resource::new(
            self,
            ImageImpl {
                handle: image,
                format,
                extent,
                usage,
                aspect,
                samples,
                full_view: OnceCell::new(),
                views: RefCell::new(HashMap::new()),
                layout: Cell::new(vk::ImageLayout::UNDEFINED),
            },
            alloc,
            |res, rd| unsafe {
                for view in res.views.borrow().values() {
                    rd.device.destroy_image_view(view.handle, None);
                }
                rd.device.destroy_image(res.handle, None);
            },
        ))
    }

    /// Gets or creates a view encompassing the entire image (all mips and layers).
    pub fn image_full_view<'a>(&self, image: &'a Image) -> &'a ImageView {
        image.full_view.get_or_init(|| {
            self.image_view_range(
                image,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image.aspect)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
            )
        })
    }

    /// Gets or creates a view for a specific mip level and array layer.
    #[inline]
    pub fn image_view(&self, image: &Image, aspect: vk::ImageAspectFlags, mip_level: u32, layer: u32) -> ImageView {
        self.image_view_range(
            image,
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(mip_level)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1),
        )
    }

    /// Gets or creates a view covering a custom `vk::ImageSubresourceRange`.
    pub fn image_view_range(&self, image: &Image, range: vk::ImageSubresourceRange) -> ImageView {
        self.image_view_create(
            image,
            &vk::ImageViewCreateInfo::default()
                .image(image.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image.format)
                .subresource_range(range),
        )
    }

    pub fn image_view_create(&self, image: &Image, info: &vk::ImageViewCreateInfo) -> ImageView {
        let hash = utils::hash_struct(info);
        if let Some(view) = image.views.borrow().get(&hash) {
            return view.clone();
        }
        let raw = unsafe { self.device.create_image_view(info, None).unwrap() };
        let view = ImageView {
            handle: raw,
            id: VIEW_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            image: Rc::downgrade(image),
        };
        image.views.borrow_mut().insert(hash, view.clone());
        view
    }

    /// Creates a generic Vulkan sampler.
    pub fn sampler_create(&self, info: vk::SamplerCreateInfo) -> Handle<vk::Sampler> {
        let value = unsafe { self.device.create_sampler(&info, None).unwrap() };
        Resource::new(self, value, None, |res, rd| unsafe {
            rd.device.destroy_sampler(res.value, None);
        })
    }

    /// Creates a Sampler with nearest filtering for both min and mag.
    pub fn sampler_nearest(&self, wrap_mode: vk::SamplerAddressMode) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with nearest mag and linear min filtering.
    pub fn sampler_nearest_linear(&self, wrap_mode: vk::SamplerAddressMode) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with bilinear filtering (linear min/mag, nearest mipmap).
    pub fn sampler_bilinear(&self, wrap_mode: vk::SamplerAddressMode) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with trilinear filtering (linear min/mag/mipmap).
    pub fn sampler_trilinear(&self, wrap_mode: vk::SamplerAddressMode) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode),
        )
    }

    /// Creates a Sampler with anisotropic filtering enabled.
    pub fn sampler_anisotropic(&self, wrap_mode: vk::SamplerAddressMode, max_anisotropy: f32) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode)
                .anisotropy_enable(true)
                .max_anisotropy(max_anisotropy),
        )
    }

    /// Creates a Sampler configured for shadow mapping (linear filtering, compare enable).
    pub fn sampler_shadow(&self, wrap_mode: vk::SamplerAddressMode) -> Handle<vk::Sampler> {
        self.sampler_create(
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(wrap_mode)
                .address_mode_v(wrap_mode)
                .address_mode_w(wrap_mode)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL),
        )
    }
}
