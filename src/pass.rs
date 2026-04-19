use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use ash::vk;
use ash::vk::Extent2D;
use glam::Vec4;
use itertools::Itertools;

use crate::Handle;
use crate::Image;
use crate::ImageView;
use crate::RenderingDevice;
use crate::Resource;

#[derive(Clone)]
#[repr(transparent)]
pub struct RenderPass(Handle<RenderPassImpl>);

pub struct RenderPassImpl {
    pub handle: vk::RenderPass,

    framebuffers: RefCell<HashMap<u64, vk::Framebuffer>>,
    images: RefCell<Vec<Image>>,

    pub clear_values: Vec<vk::ClearValue>,
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

pub struct Attachment {
    pub format: vk::Format,
    pub samples: u32,
    pub layout: vk::ImageLayout,
    pub final_layout: Option<vk::ImageLayout>,
    pub ops: Operations,
}

pub enum Operations {
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
pub struct Subpass<'a> {
    pub inputs: &'a [u32],
    pub colors: &'a [(u32, Option<u32>)],
    pub depth_stencil: Option<u32>,
    pub preserve: &'a [u32],
    pub bind_point: vk::PipelineBindPoint,
}

pub struct RenderPassDesc<'a> {
    pub attachments: &'a [Attachment],
    pub subpasses: &'a [Subpass<'a>],
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

impl RenderingDevice {
    pub fn render_pass_create(&self, desc: &RenderPassDesc) -> RenderPass {
        let attachments = desc
            .attachments
            .iter()
            .map(|a| {
                let ops = match &a.ops {
                    Operations::Color { load, store } => (conv_load_op(&load), conv_store_op(&store), vk::AttachmentLoadOp::DONT_CARE, vk::AttachmentStoreOp::DONT_CARE),
                    Operations::DepthStencil {
                        load,
                        store,
                        stencil_load,
                        stencil_store,
                    } => (conv_load_op(&load), conv_store_op(&store), conv_load_op(&stencil_load), conv_store_op(&stencil_store)),
                };
                vk::AttachmentDescription::default()
                    .format(a.format)
                    .samples(vk::SampleCountFlags::from_raw(a.samples))
                    .load_op(ops.0)
                    .store_op(ops.1)
                    .stencil_load_op(ops.2)
                    .stencil_store_op(ops.3)
                    .initial_layout(a.layout)
                    .final_layout(a.final_layout.unwrap_or(a.layout))
            })
            .collect::<Vec<_>>();

        let clear_values = desc.attachments.iter().map(|a| {
            match &a.ops {
                Operations::Color { load, .. } => match load {
                    LoadOp::Clear(c) => vk::ClearValue {
                        color: vk::ClearColorValue { float32: [c.x, c.y, c.z, c.w] },
                    },
                    _ => vk::ClearValue::default(),
                },
                Operations::DepthStencil { load, stencil_load, .. } => {
                    let depth = match load {
                        LoadOp::Clear(c) => *c,
                        _ => 1.0,
                    };
                    let stencil = match stencil_load {
                        LoadOp::Clear(c) => *c,
                        _ => 0,
                    };
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                    }
                }
            }
        }).collect::<Vec<_>>();

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
        let mut subpass_reserve = Vec::new();

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
            subpass_reserve.push(Vec::from_iter(s.preserve.iter().map(|&i| i)));
        }

        let mut dependencies = Vec::new();
        let mut prev_subpass = vk::SUBPASS_EXTERNAL;

        for (i, s) in desc.subpasses.iter().enumerate() {
            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(s.bind_point)
                .input_attachments(&subpass_inputs[i])
                .color_attachments(&subpass_colors[i])
                .resolve_attachments(&subpass_resolves[i])
                .depth_stencil_attachment(&subpass_depth_stencil[i])
                .preserve_attachments(&subpass_reserve[i]);
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

        let initial_layouts = attachments.iter().map(|a| a.initial_layout).collect_vec();
        let final_layouts = attachments.iter().map(|a| a.final_layout).collect_vec();
        let rpass = RenderPassImpl {
            handle: raw,
            framebuffers: Default::default(),

            images: RefCell::new(Vec::with_capacity(attachments.len())),
            clear_values,
            initial_layouts,
            final_layouts,
        };

        RenderPass(Resource::new(self, rpass, None, |res, rd| unsafe {
            for fb in res.framebuffers.borrow().values() {
                rd.device.destroy_framebuffer(*fb, None);
            }
            rd.device.destroy_render_pass(res.handle, None);
        }))
    }

    pub fn begin_render_pass(&self, cmd: vk::CommandBuffer, rpass: &RenderPass, views: &[&ImageView], area: vk::Rect2D) {
        unsafe {
            let mut images = rpass.images.borrow_mut();
            images.clear();
            images.extend(views.iter().map(|i| i.get_image().expect("attachment image must be valid")));

            for (i, t) in images.iter().enumerate() {
                let init_layout = rpass.initial_layouts[i];
                if t.layout.get() != init_layout && init_layout != vk::ImageLayout::UNDEFINED {
                    self.barrier_image(cmd, t, init_layout);
                }
            }

            let extent = vk::Extent2D {
                width: area.extent.width + area.offset.x.max(0) as u32,
                height: area.extent.height + area.offset.y.max(0) as u32,
            };
            let mut hasher = DefaultHasher::new();
            extent.hash(&mut hasher);
            views.iter().for_each(|t| t.id.hash(&mut hasher));
            let framebuffer_key = hasher.finish();

            let framebuffer = *rpass.framebuffers.borrow_mut().entry(framebuffer_key).or_insert_with(|| unsafe {
                self.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(rpass.handle)
                            .attachments(&images.iter().map(|t| self.image_full_view(t).handle).collect_vec())
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .unwrap()
            });

            self.device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(rpass.handle)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(&rpass.clear_values),
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn next_subpass(&self, cmd: vk::CommandBuffer, rpass: &RenderPass) {
        for (i, t) in rpass.images.borrow().iter().enumerate() {
            // we assumed image layouts of attachments remain same during whole pass, so guarantee this inside next subpass
            let required_layout = rpass.initial_layouts[i];
            if t.layout.get() != required_layout {
                self.barrier_image(cmd, t, required_layout);
            }
        }
        unsafe {
            self.device.cmd_next_subpass(cmd, vk::SubpassContents::INLINE);
        }
    }

    pub fn end_render_pass(&self, cmd: vk::CommandBuffer, rpass: &RenderPass) {
        unsafe {
            self.device.cmd_end_render_pass(cmd);
        }
        for (i, t) in rpass.images.borrow().iter().enumerate() {
            t.assume_layout(rpass.final_layouts[i]);
        }
    }
}
