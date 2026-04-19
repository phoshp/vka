use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;

use ash::vk;
use itertools::Itertools;

use crate::Handle;
use crate::RenderingDevice;
use crate::Resource;

#[derive(Clone, Copy)]
pub struct DescriptorSetLayoutEntry {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub count: u32,
    pub flags: Option<vk::DescriptorBindingFlags>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct DescriptorSetLayout(Handle<DescriptorSetLayoutImpl>);

impl Deref for DescriptorSetLayout {
    type Target = Handle<DescriptorSetLayoutImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct DescriptorSetLayoutImpl {
    pub handle: vk::DescriptorSetLayout,
    pub bindings: HashMap<u32, DescriptorSetLayoutEntry>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct DescriptorSet(Handle<DescriptorSetImpl>);

impl Deref for DescriptorSet {
    type Target = Handle<DescriptorSetImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct DescriptorSetImpl {
    pub handle: vk::DescriptorSet,
    pub pool: vk::DescriptorPool,
    pub layout: DescriptorSetLayout,
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
    pub fn descriptor_set_layout_create(&self, entries: &[DescriptorSetLayoutEntry]) -> super::Result<DescriptorSetLayout> {
        let bindings = entries
            .iter()
            .map(|e| vk::DescriptorSetLayoutBinding::default().binding(e.binding).descriptor_type(e.ty).descriptor_count(e.count))
            .collect::<Vec<_>>();
        let flags = entries.iter().map(|e| e.flags.unwrap_or_default()).collect::<Vec<_>>();
        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut binding_flags_info);
        let raw = unsafe { self.device.create_descriptor_set_layout(&layout_info, None)? };
        let handle = Resource::new(
            self,
            DescriptorSetLayoutImpl {
                handle: raw,
                bindings: HashMap::from_iter(entries.iter().map(|&e| (e.binding, e))),
            },
            None,
            |res, rd| unsafe {
                rd.device.destroy_descriptor_set_layout(res.handle, None);
            },
        );
        Ok(DescriptorSetLayout(handle))
    }

    pub fn descriptor_set_create(&self, layout: &DescriptorSetLayout) -> super::Result<DescriptorSet> {
        let mut pool_size_map = HashMap::new();
        for entry in layout.bindings.values() {
            pool_size_map.entry(entry.ty).and_modify(|c| *c += entry.count).or_insert(entry.count);
        }
        let pool_sizes = pool_size_map
            .into_iter()
            .map(|(ty, count)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(count))
            .collect_vec();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };
        let set = unsafe {
            self.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[layout.handle]))?[0]
        };
        let handle = Resource::new(
            self,
            DescriptorSetImpl {
                handle: set,
                pool,
                layout: layout.clone(),
            },
            None,
            |res, rd| unsafe {
                rd.device.destroy_descriptor_pool(res.pool, None);
            },
        );
        Ok(DescriptorSet(handle))
    }

    pub fn descriptor_set_write(&self, set: &DescriptorSet, writes: &[WriteDescriptor]) {
        let mut inline_uniform_exts = Vec::with_capacity(writes.len());
        let mut acceleration_exts = Vec::with_capacity(writes.len());
        let mut vk_writes = Vec::new();

        for mut write in writes {
            let mut vk_write = vk::WriteDescriptorSet::default().dst_set(set.handle);
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
                WriteDescriptor::AccelerationStructure { binding, array_element, structures } => {
                    vk_write = vk_write.dst_binding(*binding).dst_array_element(*array_element);
                    acceleration = Some(vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(structures));
                }
            }
            vk_write.descriptor_type = set.layout.bindings.get(&vk_write.dst_binding).expect("invalid binding index").ty;
            inline_uniform_exts.push(inline_uniform);
            acceleration_exts.push(acceleration);
            vk_writes.push(vk_write);
        }
        for (i, vk_write) in vk_writes.iter_mut().enumerate() {
            if let Some(inline_uniform) = &mut inline_uniform_exts[i] {
                vk_write.push_next(inline_uniform);
            }
            if let Some(acceleration) = &mut acceleration_exts[i] {
                vk_write.push_next(acceleration);
            }
        }
        unsafe {
            self.device.update_descriptor_sets(&vk_writes, &[]);
        }
    }
}
