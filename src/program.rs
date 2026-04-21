use std::collections::HashMap;
use std::ops::Deref;

use ash::vk;
use itertools::Itertools;

use crate::Handle;
use crate::RenderingDevice;
use crate::Resource;

#[derive(Clone)]
#[repr(transparent)]
pub struct Program(Handle<ProgramImpl>);

impl Deref for Program {
    type Target = Handle<ProgramImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ProgramImpl {
    pub pipeline_layout: vk::PipelineLayout,
    pub set_layouts: Vec<super::DescriptorSetLayout>,
}

impl RenderingDevice {
    pub fn program_create(&self, set_layouts: &[super::DescriptorSetLayout]) -> super::Result<Program> {
        let set_layouts_vk = set_layouts.iter().map(|l| l.handle).collect_vec();
        let push_constant_ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::ALL).offset(0).size(128)];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_vk)
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None)? };
        let inner = ProgramImpl {
            pipeline_layout,
            set_layouts: Vec::from(set_layouts),
        };
        Ok(Program(Resource::new(self, inner, None, |res, rd| {
            unsafe { rd.device.destroy_pipeline_layout(res.pipeline_layout, None) };
        })))
    }
}
