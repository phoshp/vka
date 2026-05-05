use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use ash::vk;
use itertools::Itertools;

use crate::RenderingDevice;
use crate::SharedDevice;

#[derive(Clone, Copy)]
pub struct DescriptorSetLayoutEntry {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub count: u32,
    pub flags: Option<vk::DescriptorBindingFlags>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct DescriptorSetLayout(Arc<DescriptorSetLayoutImpl>);

impl Deref for DescriptorSetLayout {
    type Target = Arc<DescriptorSetLayoutImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct DescriptorSetLayoutImpl {
    pub raw: vk::DescriptorSetLayout,
    pub bindings: HashMap<u32, DescriptorSetLayoutEntry>,
    device: Arc<SharedDevice>,
}

impl Drop for DescriptorSetLayoutImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct DescriptorSet(Arc<DescriptorSetImpl>);

impl Deref for DescriptorSet {
    type Target = Arc<DescriptorSetImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct DescriptorSetImpl {
    pub raw: vk::DescriptorSet,
    pub pool: vk::DescriptorPool,
    pub layout: DescriptorSetLayout,
    device: Arc<SharedDevice>,
}

impl Drop for DescriptorSetImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_descriptor_pool(self.pool, None);
        }
    }
}

pub enum WriteDescriptor<'a> {
    Buffer {
        binding: u32,
        array_element: u32,
        infos: &'a [vk::DescriptorBufferInfo],
    },
    Image {
        binding: u32,
        array_element: u32,
        infos: &'a [vk::DescriptorImageInfo],
    },
    TexelBuffer {
        binding: u32,
        array_element: u32,
        views: &'a [vk::BufferView],
    },
    InlineUniform {
        binding: u32,
        data: &'a [u8],
    },
    AccelerationStructure {
        binding: u32,
        array_element: u32,
        structures: &'a [vk::AccelerationStructureKHR],
    },
}

impl RenderingDevice {
    pub fn new_descriptor_set_layout(&self, entries: &[DescriptorSetLayoutEntry]) -> DescriptorSetLayout {
        let bindings = entries
            .iter()
            .map(|e| vk::DescriptorSetLayoutBinding::default().binding(e.binding).descriptor_type(e.ty).descriptor_count(e.count))
            .collect::<Vec<_>>();
        let flags = entries.iter().map(|e| e.flags.unwrap_or_default()).collect::<Vec<_>>();
        let update_after_bind = flags.iter().any(|&f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND));

        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(if update_after_bind { vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL } else { vk::DescriptorSetLayoutCreateFlags::empty() })
            .bindings(&bindings)
            .push_next(&mut binding_flags_info);
        let raw = unsafe { self.raw.create_descriptor_set_layout(&layout_info, None).expect("Failed to create descriptor set layout") };
        let inner = DescriptorSetLayoutImpl {
            raw,
            bindings: HashMap::from_iter(entries.iter().map(|&e| (e.binding, e))),
            device: self.shared.clone(),
        };
        DescriptorSetLayout(Arc::new(inner))
    }

    pub fn new_descriptor_set(&self, layout: &DescriptorSetLayout) -> DescriptorSet {
        let mut pool_size_map = HashMap::new();
        for entry in layout.bindings.values() {
            pool_size_map.entry(entry.ty).and_modify(|c| *c += entry.count).or_insert(entry.count);
        }
        let mut inline_uniform_ext = vk::DescriptorPoolInlineUniformBlockCreateInfo::default()
            .max_inline_uniform_block_bindings(*pool_size_map.get(&vk::DescriptorType::INLINE_UNIFORM_BLOCK).unwrap_or(&0));
        let pool_sizes = pool_size_map
            .into_iter()
            .map(|(ty, count)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(count))
            .collect_vec();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .push_next(&mut inline_uniform_ext);
        let pool = unsafe { self.raw.create_descriptor_pool(&pool_info, None).expect("Failed to create descriptor pool") };
        let raw = unsafe {
            self.raw
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[layout.raw])).expect("Failed to allocate descriptor sets")
        }[0];
        let inner = DescriptorSetImpl {
            raw,
            pool,
            layout: layout.clone(),
            device: self.shared.clone(),
        };
        DescriptorSet(Arc::new(inner))
    }

    pub fn write_descriptors(&self, set: &DescriptorSet, writes: &[WriteDescriptor]) {
        let mut inline_uniform_exts = Vec::with_capacity(writes.len());
        let mut acceleration_exts = Vec::with_capacity(writes.len());
        let mut vk_writes = Vec::new();

        for write in writes {
            let mut vk_write = vk::WriteDescriptorSet::default().dst_set(set.raw);
            let mut inline_uniform = None;
            let mut acceleration = None;

            match write {
                WriteDescriptor::Buffer { binding, array_element, infos } => {
                    vk_write = vk_write.dst_binding(*binding).dst_array_element(*array_element).buffer_info(infos);
                }
                WriteDescriptor::Image { binding, array_element, infos } => {
                    vk_write = vk_write.dst_binding(*binding).dst_array_element(*array_element).image_info(infos);
                }
                WriteDescriptor::TexelBuffer { binding, array_element, views } => {
                    vk_write = vk_write.dst_binding(*binding).dst_array_element(*array_element).texel_buffer_view(views);
                }
                WriteDescriptor::InlineUniform { binding, data } => {
                    vk_write = vk_write.dst_binding(*binding).descriptor_count(1);
                    inline_uniform = Some(vk::WriteDescriptorSetInlineUniformBlock::default().data(data));
                }
                WriteDescriptor::AccelerationStructure {
                    binding,
                    array_element,
                    structures,
                } => {
                    vk_write = vk_write.dst_binding(*binding).dst_array_element(*array_element);
                    acceleration = Some(vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(structures));
                }
            }
            vk_write.descriptor_type = set.layout.bindings.get(&vk_write.dst_binding).expect("invalid binding index").ty;
            inline_uniform_exts.push(inline_uniform);
            acceleration_exts.push(acceleration);
            vk_writes.push(vk_write);
        }

        for (i, ext) in inline_uniform_exts.iter_mut().enumerate() {
            if let Some(ext) = ext {
                let vk_write = &mut vk_writes[i];
                *vk_write = vk_write.push_next(ext);
            }
        }
        for (i, ext) in acceleration_exts.iter_mut().enumerate() {
            if let Some(ext) = ext {
                let vk_write = &mut vk_writes[i];
                *vk_write = vk_write.push_next(ext);
            }
        }
        unsafe {
            let _idx = self.submit_mutex.lock();
            self.wait_queue();
            self.raw.update_descriptor_sets(&vk_writes, &[]);
        }
    }
}
