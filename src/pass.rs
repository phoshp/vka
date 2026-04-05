use std::collections::HashMap;
use std::ops::Deref;

use ash::vk;
use glam::Vec4;
use itertools::Itertools;

use crate::Handle;
use crate::Image;
use crate::RenderingDevice;
use crate::Resource;

#[derive(Debug, Clone)]
pub struct RenderPass(Handle<RenderPassImpl>);

#[derive(Debug)]
pub struct RenderPassImpl {
    pub handle: vk::RenderPass,

    pub framebuffer: vk::Framebuffer,
    pub images: Vec<Image>,
    pub initial_layouts: Vec<vk::ImageLayout>,
    pub final_layouts: Vec<vk::ImageLayout>
}

impl Deref for RenderPass {
    type Target = Handle<RenderPassImpl>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub enum LoadOp<V> {
    Load,
    Clear(V),
    Discard,
}

pub enum StoreOp {
    Store,
    Discard,
}

pub struct Attachment<'a> {
    pub image: &'a Image,
    pub ops: Operations,
}

pub enum Operations {
    Input {
        load: LoadOp<Vec4>,
        store: StoreOp,
    },
    Color {
        load: LoadOp<Vec4>,
        store: StoreOp,
    },
    DepthStencil {
        load: LoadOp<f32>,
        store: StoreOp,
        stencil_load: LoadOp<u32>,
        stencil_store: StoreOp,
    },
}

#[derive(Debug, Default)]
pub struct SubpassDesc<'a> {
    pub inputs: &'a [u32],
    pub colors: &'a [(u32, Option<u32>)],
    pub depth_stencil: Option<u32>,
    pub bind_point: vk::PipelineBindPoint,
}

pub struct RenderPassDesc<'a> {
    pub extent: vk::Extent2D,
    pub attachments: &'a [Attachment<'a>],
    pub subpasses: &'a [SubpassDesc<'a>],
}

fn conv_usage_to_layout(usage: vk::ImageUsageFlags) -> vk::ImageLayout {
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

pub struct RenderPassBeginDesc<'a> {
    pub pass: &'a RenderPass,
    pub render_area: vk::Rect2D,
    pub clear_values: &'a [vk::ClearValue],
}

impl RenderingDevice {
    pub fn render_pass_create(&self, desc: &RenderPassDesc) -> RenderPass {
        fn conv_load_op<V>(op: &LoadOp<V>) -> vk::AttachmentLoadOp {
            match op {
                LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                LoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
            }
        }
        fn conv_store_op(op: &StoreOp) -> vk::AttachmentStoreOp {
            match op {
                StoreOp::Store => vk::AttachmentStoreOp::STORE,
                StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
            }
        }

        let attachments = desc
            .attachments
            .iter()
            .map(|a| {
                let ops = match &a.ops {
                    Operations::Input { load, store } => (conv_load_op(&load), conv_store_op(&store), vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE),
                    Operations::Color { load, store } => (conv_load_op(&load), conv_store_op(&store), vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE),
                    Operations::DepthStencil {
                        load,
                        store,
                        stencil_load,
                        stencil_store,
                    } => (conv_load_op(&load), conv_store_op(&store), conv_load_op(&stencil_load), conv_store_op(&stencil_store)),
                };
                let layout = match a.ops {
                    Operations::Input { .. } => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    Operations::Color { .. } => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    Operations::DepthStencil { .. } => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                };
                vk::AttachmentDescription::default()
                    .format(a.image.format)
                    .samples(a.image.samples)
                    .load_op(ops.0)
                    .store_op(ops.1)
                    .stencil_load_op(ops.2)
                    .stencil_store_op(ops.3)
                    .initial_layout(layout)
                    .final_layout(layout)
            })
            .collect::<Vec<_>>();

        let mut subpasses = Vec::new();
        let refs = attachments
            .iter()
            .enumerate()
            .map(|(i, a)| vk::AttachmentReference::default().attachment(i as u32).layout(a.initial_layout))
            .collect::<Vec<_>>();
        let mut subpass_inputs = Vec::new();
        let mut subpass_colors = Vec::new();
        let mut subpass_resolves = Vec::new();
        let mut subpass_depth_stencil = Vec::new();

        for s in desc.subpasses.iter() {
            subpass_inputs.push(Vec::from_iter(s.inputs.iter().map(|&i| refs[i as usize])));
            subpass_colors.push(Vec::from_iter(s.colors.iter().map(|(i, _)| refs[*i as usize])));
            subpass_resolves.push(Vec::from_iter(
                s.colors
                    .iter()
                    .map(|(_, resolve)| {
                        resolve
                            .as_ref()
                            .map(|r| refs[*r as usize])
                            .unwrap_or(vk::AttachmentReference::default().attachment(vk::ATTACHMENT_UNUSED))
                    })
                    .collect::<Vec<_>>(),
            ));
            subpass_depth_stencil.push(
                s.depth_stencil
                    .map(|i| refs[i as usize])
                    .unwrap_or(vk::AttachmentReference::default().attachment(vk::ATTACHMENT_UNUSED)),
            );
        }

        let mut dependencies = Vec::new();
        let mut prev_subpass = vk::SUBPASS_EXTERNAL;

        for (i, s) in desc.subpasses.iter().enumerate() {
            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(s.bind_point)
                .input_attachments(&subpass_inputs[i])
                .color_attachments(&subpass_colors[i])
                .resolve_attachments(&subpass_resolves[i])
                .depth_stencil_attachment(&subpass_depth_stencil[i]);
            subpasses.push(subpass);

            dependencies.push(
                vk::SubpassDependency::default()
                    .src_subpass(prev_subpass)
                    .dst_subpass(i as u32)
                    .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
            );
            prev_subpass = i as u32;
        }
        dependencies.push(
            vk::SubpassDependency::default()
                .src_subpass(prev_subpass)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
        );

        let raw = unsafe {
            self.device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses)
                        .dependencies(&dependencies),
                    None,
                )
                .unwrap()
        };

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(raw)
                        .attachments(&desc.attachments.iter().map(|t| self.image_full_view(t.image)).collect_vec())
                        .width(desc.extent.width)
                        .height(desc.extent.height)
                        .layers(1),
                    None,
                )
                .unwrap()
        };
        let initial_layouts = attachments.iter().map(|a| a.initial_layout).collect_vec();
        let final_layouts = attachments.iter().map(|a| a.final_layout).collect_vec();
        let rpass = RenderPassImpl {
            handle: raw,
            framebuffer,
            images: desc.attachments.iter().map(|a| a.image.clone()).collect_vec(),
            initial_layouts,
            final_layouts,
        };

        RenderPass(Resource::new(self, rpass, None, |res, rd| unsafe {
            rd.device.destroy_render_pass(res.handle, None);
            rd.device.destroy_framebuffer(res.framebuffer, None);
        }))
    }

    pub fn begin_render_pass(&self, cmd: vk::CommandBuffer, desc: &RenderPassBeginDesc) {
        unsafe {
            self.device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(desc.pass.handle)
                    .framebuffer(desc.pass.framebuffer)
                    .render_area(desc.render_area)
                    .clear_values(desc.clear_values),
                vk::SubpassContents::INLINE,
            );
        }
        for (i, t) in desc.pass.images.iter().enumerate() {
            t.assume_layout(desc.pass.initial_layouts[i]);
        }
    }

    pub fn end_render_pass(&self, cmd: vk::CommandBuffer, rpass: &RenderPass) {
        unsafe {
            self.device.cmd_end_render_pass(cmd);
        }
        for (i, t) in rpass.images.iter().enumerate() {
            t.assume_layout(rpass.final_layouts[i]);
        }
    }
}
