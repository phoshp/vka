use std::ffi::CString;
use std::ops::Deref;

use ash::vk;
use itertools::Itertools;

use crate::Handle;
use crate::RenderPass;
use crate::RenderingDevice;
use crate::Resource;

#[derive(Clone)]
#[repr(transparent)]
pub struct PipelineLayout(Handle<PipelineLayoutImpl>);

impl Deref for PipelineLayout {
    type Target = Handle<PipelineLayoutImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct PipelineLayoutImpl {
    pub handle: vk::PipelineLayout,
    pub set_layouts: Vec<super::DescriptorSetLayout>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct GraphicsPipeline(Handle<GraphicsPipelineImpl>);

impl Deref for GraphicsPipeline {
    type Target = Handle<GraphicsPipelineImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct GraphicsPipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: PipelineLayout,
    pub pass: Option<vk::RenderPass>,
    pub subpass: u32,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ComputePipeline(Handle<ComputePipelineImpl>);

impl Deref for ComputePipeline {
    type Target = Handle<ComputePipelineImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ComputePipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: PipelineLayout,
}

pub struct ShaderStage {
    pub stage: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    pub name: CString,
}

pub struct VertexAttribute {
    pub location: u32,
    pub format: vk::Format,
    pub offset: u32,
}

pub struct VertexInputLayout<'a> {
    pub binding: u32,
    pub stride: u32,
    pub rate: vk::VertexInputRate,
    pub attributes: &'a [VertexAttribute],
}

#[macro_export]
macro_rules! vertex_attributes {
    ($($location:expr => $format:expr),* $(,)?) => {
        {
            let mut attributes = Vec::new();
            let mut offset = 0;
            $(
                let format = $format;
                attributes.push(VertexAttribute {
                    location: $location,
                    format,
                    offset,
                });
                let size = match format {
                    vk::Format::R32_SFLOAT => 4,
                    vk::Format::R32G32_SFLOAT => 8,
                    vk::Format::R32G32B32_SFLOAT => 12,
                    vk::Format::R32G32B32A32_SFLOAT => 16,

                    vk::Format::R32_SINT => 4,
                    vk::Format::R32G32_SINT => 8,
                    vk::Format::R32G32B32_SINT => 12,
                    vk::Format::R32G32B32A32_SINT => 16,

                    vk::Format::R32_UINT => 4,
                    vk::Format::R32G32_UINT => 8,
                    vk::Format::R32G32B32_UINT => 12,
                    vk::Format::R32G32B32A32_UINT => 16,

                    _ => panic!("Unsupported format"),
                };
                offset += size;
            )*
            attributes
        }
    };
}

pub struct InputAssemblyState {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool,
}

pub struct RasterizationState {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

pub struct MultisampleState<'a> {
    pub sample_count: u32,
    pub sample_shading_enable: bool,
    pub min_sample_shading: f32,
    pub sample_mask: Option<&'a [vk::SampleMask]>,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: vk::StencilOpState,
    pub back: vk::StencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

pub struct ColorBlendState<'a> {
    pub logic_op_enable: bool,
    pub logic_op: vk::LogicOp,
    pub attachments: &'a [vk::PipelineColorBlendAttachmentState],
    pub blend_constants: [f32; 4],
}

pub const fn make_blend_state(
    blend_enable: bool,
    src_color: vk::BlendFactor,
    dst_color: vk::BlendFactor,
    color_op: vk::BlendOp,
    src_alpha: vk::BlendFactor,
    dst_alpha: vk::BlendFactor,
    alpha_op: vk::BlendOp,
) -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState {
        blend_enable: if blend_enable { vk::TRUE } else { vk::FALSE },
        src_color_blend_factor: src_color,
        dst_color_blend_factor: dst_color,
        color_blend_op: color_op,
        src_alpha_blend_factor: src_alpha,
        dst_alpha_blend_factor: dst_alpha,
        alpha_blend_op: alpha_op,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }
}

pub const COLOR_BLEND_REPLACE: vk::PipelineColorBlendAttachmentState = make_blend_state(
    false,
    vk::BlendFactor::ONE,
    vk::BlendFactor::ZERO,
    vk::BlendOp::ADD,
    vk::BlendFactor::ONE,
    vk::BlendFactor::ZERO,
    vk::BlendOp::ADD,
);
pub const COLOR_BLEND_ALPHA: vk::PipelineColorBlendAttachmentState = make_blend_state(
    true,
    vk::BlendFactor::SRC_ALPHA,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendOp::ADD,
    vk::BlendFactor::ONE,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendOp::ADD,
);
pub const COLOR_BLEND_PREMULTIPLIED: vk::PipelineColorBlendAttachmentState = make_blend_state(
    true,
    vk::BlendFactor::ONE,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendOp::ADD,
    vk::BlendFactor::ONE,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendOp::ADD,
);

pub struct GraphicsPipelineDesc<'a> {
    pub layout: &'a PipelineLayout,
    pub stages: &'a [ShaderStage],
    pub vertex_input: &'a [VertexInputLayout<'a>],
    pub input_assembly: InputAssemblyState,
    pub rasterization: RasterizationState,
    pub multisample: MultisampleState<'a>,
    pub depth_stencil: Option<DepthStencilState>,
    pub color_blend: ColorBlendState<'a>,
    pub render_pass: Option<&'a RenderPass>,
    pub subpass: u32,
}

pub struct ComputePipelineDesc<'a> {
    pub layout: &'a PipelineLayout,
    pub stage: ShaderStage,
}

impl Default for InputAssemblyState {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
        }
    }
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        }
    }
}

impl Default for MultisampleState<'_> {
    fn default() -> Self {
        Self {
            sample_count: 1,
            sample_shading_enable: false,
            min_sample_shading: 0.0,
            sample_mask: None,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
        }
    }
}

impl Default for ColorBlendState<'_> {
    fn default() -> Self {
        Self {
            logic_op_enable: false,
            logic_op: vk::LogicOp::default(),
            attachments: &[COLOR_BLEND_REPLACE],
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

impl RenderingDevice {
    pub fn shader_module_create(&self, spv: &[u8]) -> super::Result<Handle<vk::ShaderModule>> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
        let info = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { self.device.create_shader_module(&info, None)? };
        Ok(Resource::new(self, module, None, |res, rd| {
            unsafe { rd.device.destroy_shader_module(res.value, None) };
        }))
    }

    pub fn pipeline_layout_create(&self, set_layouts: &[super::DescriptorSetLayout]) -> super::Result<PipelineLayout> {
        let set_layouts_vk = set_layouts.iter().map(|l| l.handle).collect_vec();
        let push_constant_ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::ALL).offset(0).size(128)];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_vk)
            .push_constant_ranges(&push_constant_ranges);
        let raw = unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None)? };
        let inner = PipelineLayoutImpl {
            handle: raw,
            set_layouts: Vec::from(set_layouts),
        };
        Ok(PipelineLayout(Resource::new(self, inner, None, |res, rd| {
            unsafe { rd.device.destroy_pipeline_layout(res.handle, None) };
        })))
    }

    pub fn graphics_pipeline_create(&self, desc: &GraphicsPipelineDesc) -> super::Result<GraphicsPipeline> {
        let stages = desc
            .stages
            .iter()
            .map(|s| vk::PipelineShaderStageCreateInfo::default().stage(s.stage).module(s.module).name(&s.name))
            .collect::<Vec<_>>();

        let mut vertex_input_bindings = Vec::new();
        let mut vertex_input_attributes = Vec::new();

        for layout in desc.vertex_input {
            vertex_input_bindings.push(
                vk::VertexInputBindingDescription::default()
                    .binding(layout.binding)
                    .stride(layout.stride)
                    .input_rate(layout.rate),
            );
            for attr in layout.attributes {
                vertex_input_attributes.push(
                    vk::VertexInputAttributeDescription::default()
                        .location(attr.location)
                        .binding(layout.binding)
                        .format(attr.format)
                        .offset(attr.offset),
                );
            }
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_input_bindings)
            .vertex_attribute_descriptions(&vertex_input_attributes);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(desc.input_assembly.topology)
            .primitive_restart_enable(desc.input_assembly.primitive_restart_enable);

        let viewports = [vk::Viewport::default().width(32.0).height(32.0).min_depth(0.0).max_depth(1.0)];
        let scissors = [vk::Rect2D::default().extent(vk::Extent2D { width: 32, height: 32 })];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewports(&viewports).scissors(&scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(desc.rasterization.depth_clamp_enable)
            .rasterizer_discard_enable(desc.rasterization.rasterizer_discard_enable)
            .polygon_mode(desc.rasterization.polygon_mode)
            .cull_mode(desc.rasterization.cull_mode)
            .front_face(desc.rasterization.front_face)
            .depth_bias_enable(desc.rasterization.depth_bias_enable)
            .depth_bias_constant_factor(desc.rasterization.depth_bias_constant_factor)
            .depth_bias_clamp(desc.rasterization.depth_bias_clamp)
            .depth_bias_slope_factor(desc.rasterization.depth_bias_slope_factor)
            .line_width(desc.rasterization.line_width);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::from_raw(desc.multisample.sample_count))
            .sample_shading_enable(desc.multisample.sample_shading_enable)
            .min_sample_shading(desc.multisample.min_sample_shading)
            .sample_mask(desc.multisample.sample_mask.as_deref().unwrap_or(&[]))
            .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enable)
            .alpha_to_one_enable(desc.multisample.alpha_to_one_enable);

        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();
        if let Some(ds) = &desc.depth_stencil {
            depth_stencil_state = depth_stencil_state
                .depth_test_enable(ds.depth_test_enable)
                .depth_write_enable(ds.depth_write_enable)
                .depth_compare_op(ds.depth_compare_op)
                .depth_bounds_test_enable(ds.depth_bounds_test_enable)
                .stencil_test_enable(ds.stencil_test_enable)
                .front(ds.front)
                .back(ds.back)
                .min_depth_bounds(ds.min_depth_bounds)
                .max_depth_bounds(ds.max_depth_bounds);
        }

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(desc.color_blend.logic_op_enable)
            .logic_op(desc.color_blend.logic_op)
            .attachments(&desc.color_blend.attachments)
            .blend_constants(desc.color_blend.blend_constants);

        let pass = desc.render_pass.as_ref().map(|p| p.handle);
        let subpass = desc.subpass;
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::DEPTH_BOUNDS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
        ];

        let result = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .depth_stencil_state(&depth_stencil_state)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states))
                    .layout(desc.layout.handle)
                    .render_pass(pass.unwrap_or(vk::RenderPass::null()))
                    .subpass(subpass)],
                None,
            )
        };
        let raw = match result {
            Ok(pipelines) => pipelines.into_iter().next().unwrap(),
            Err((_, err)) => return Err(err.into()),
        };

        let pipeline = GraphicsPipelineImpl {
            handle: raw,
            layout: desc.layout.clone(),
            pass,
            subpass,
        };

        Ok(GraphicsPipeline(Resource::new(self, pipeline, None, |res, rd| unsafe {
            rd.device.destroy_pipeline(res.handle, None);
        })))
    }

    pub fn compute_pipeline_create(&self, desc: &ComputePipelineDesc) -> super::Result<ComputePipeline> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(desc.stage.stage)
            .module(desc.stage.module)
            .name(&desc.stage.name);

        let result = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(desc.layout.handle).base_pipeline_index(-1)],
                None,
            )
        };
        let raw = match result {
            Ok(pipelines) => pipelines.into_iter().next().unwrap(),
            Err((_, err)) => return Err(err.into()),
        };

        let pipeline = ComputePipelineImpl {
            handle: raw,
            layout: desc.layout.clone(),
        };
        Ok(ComputePipeline(Resource::new(self, pipeline, None, |res, rd| unsafe {
            rd.device.destroy_pipeline(res.handle, None);
        })))
    }
}
