use std::cell::Cell;
use std::cell::OnceCell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::Handle;
use crate::RenderingDevice;
use crate::Resource;
use crate::VkaResult;
use crate::bytes_of;
use crate::utils;

/// A wrapper around a Vulkan image resource, providing additional metadata and caching for image views.
///
/// The `Image` struct holds the Vulkan image handle, its format, extent, usage, aspect mask, sample count, and layout.
/// It also caches the image views to avoid redundant Vulkan calls.
#[derive(Clone)]
#[repr(transparent)]
pub struct Image(Handle<ImageImpl>);

impl Deref for Image {
    type Target = Handle<ImageImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The internal implementation of the `Image` resource, containing the Vulkan image handle and related metadata.
pub struct ImageImpl {
    pub handle: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,

    full_view: OnceCell<vk::ImageView>,
    views: RefCell<HashMap<u64, vk::ImageView>>,

    pub layout: Cell<vk::ImageLayout>,
}

impl ImageImpl {
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(self.aspect)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
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
    #[inline]
    pub fn image_2d(&self, format: vk::Format, width: u32, height: u32, levels: u32, layers: u32, samples: vk::SampleCountFlags, usage: vk::ImageUsageFlags) -> VkaResult<Image> {
        self.image_from_info(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D::default().width(width).height(height).depth(1))
                .mip_levels(levels)
                .array_layers(layers)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )
    }

    #[inline]
    pub fn image_3d(&self, format: vk::Format, width: u32, height: u32, depth: u32, levels: u32, samples: vk::SampleCountFlags, usage: vk::ImageUsageFlags) -> VkaResult<Image> {
        self.image_from_info(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_3D)
                .format(format)
                .extent(vk::Extent3D::default().width(width).height(height).depth(depth))
                .mip_levels(levels)
                .array_layers(1)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )
    }

    #[inline]
    pub fn image_cube(&self, format: vk::Format, width: u32, height: u32, levels: u32, samples: vk::SampleCountFlags, usage: vk::ImageUsageFlags) -> VkaResult<Image> {
        self.image_from_info(
            vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D::default().width(width).height(height).depth(1))
                .mip_levels(levels)
                .array_layers(6)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )
    }

    pub fn image_from_info(&self, mut info: vk::ImageCreateInfo) -> VkaResult<Image> {
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
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::DedicatedImage(image),
                })
                .unwrap();
            self.device.bind_image_memory(image, alloc.memory(), alloc.offset())?;
            VkaResult::Ok(self.image_from_raw(image, info.format, info.extent, info.samples, info.usage, Some(alloc)))
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
                    rd.device.destroy_image_view(*view, None);
                }
                rd.device.destroy_image(res.handle, None);
            },
        ))
    }

    pub fn image_full_view(&self, image: &Image) -> vk::ImageView {
        *image.full_view.get_or_init(|| {
            self.image_view_range(
                image,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image.aspect)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
            )
        })
    }

    #[inline]
    pub fn image_view(&self, image: &Image, mip_level: u32, layer: u32) -> vk::ImageView {
        self.image_view_range(
            image,
            vk::ImageSubresourceRange::default()
                .aspect_mask(image.aspect)
                .base_mip_level(mip_level)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1),
        )
    }

    pub fn image_view_range(&self, image: &Image, range: vk::ImageSubresourceRange) -> vk::ImageView {
        self.image_view_create(
            image,
            &vk::ImageViewCreateInfo::default()
                .image(image.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image.format)
                .subresource_range(range),
        )
    }

    pub fn image_view_create(&self, image: &Image, info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        let hash = utils::hash_struct(info);
        if let Some(view) = image.views.borrow().get(&hash) {
            return *view;
        }
        let view = unsafe { self.device.create_image_view(info, None).unwrap() };
        image.views.borrow_mut().insert(hash, view);
        view
    }

    pub fn sampler_create(&self, info: vk::SamplerCreateInfo) -> Handle<vk::Sampler> {
        let value = unsafe { self.device.create_sampler(&info, None).unwrap() };
        Resource::new(self, value, None, |res, rd| unsafe {
            rd.device.destroy_sampler(res.value, None);
        })
    }

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
