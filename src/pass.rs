use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use ash::vk;
use parking_lot::Mutex;

use crate::Color32;
use crate::RenderingDevice;
use crate::SharedDevice;

#[derive(Clone)]
#[repr(transparent)]
pub struct RenderPass(Arc<RenderPassImpl>);

pub struct RenderPassImpl {
    pub raw: vk::RenderPass,
    pub id: u64,
    pub clear_values: Vec<vk::ClearValue>,

    device: Arc<SharedDevice>,
    pub(crate) framebuffers: Mutex<HashMap<u64, vk::Framebuffer>>,
}

impl Drop for RenderPassImpl {
    fn drop(&mut self) {
        unsafe {
            // TODO: lifetime of images?
            for fb in self.framebuffers.lock().values() {
                self.device.raw.destroy_framebuffer(*fb, None);
            }
            self.device.raw.destroy_render_pass(self.raw, None);
        }
    }
}

impl Deref for RenderPass {
    type Target = Arc<RenderPassImpl>;
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
    pub usage: vk::ImageUsageFlags,
    pub ops: Operations,
}

pub enum Operations {
    Color {
        load: LoadOp<Color32>,
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
    pub fn new_render_pass(&self, desc: &RenderPassDesc) -> RenderPass {
        let attachments = desc
            .attachments
            .iter()
            .map(|a| {
                let ops = match &a.ops {
                    Operations::Color { load, store } => (
                        conv_load_op(&load),
                        conv_store_op(&store),
                        vk::AttachmentLoadOp::DONT_CARE,
                        vk::AttachmentStoreOp::DONT_CARE,
                    ),
                    Operations::DepthStencil {
                        load,
                        store,
                        stencil_load,
                        stencil_store,
                    } => (conv_load_op(&load), conv_store_op(&store), conv_load_op(&stencil_load), conv_store_op(&stencil_store)),
                };
                let layout = crate::find_optimal_layout(a.usage);
                vk::AttachmentDescription::default()
                    .format(a.format)
                    .samples(vk::SampleCountFlags::from_raw(a.samples))
                    .load_op(ops.0)
                    .store_op(ops.1)
                    .stencil_load_op(ops.2)
                    .stencil_store_op(ops.3)
                    .initial_layout(layout)
                    .final_layout(layout)
            })
            .collect::<Vec<_>>();

        let clear_values = desc
            .attachments
            .iter()
            .map(|a| match &a.ops {
                Operations::Color { load, .. } => match load {
                    LoadOp::Clear(c) => vk::ClearValue {
                        color: vk::ClearColorValue { float32: [c.r, c.g, c.b, c.a] },
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

        let raw = unsafe {
            self.raw
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses)
                        .dependencies(&dependencies),
                    None,
                )
                .expect("Failed to create render pass")
        };
        let inner = RenderPassImpl {
            raw,
            id: crate::next_resource_id(),
            clear_values,
            device: self.shared.clone(),
            framebuffers: Default::default(),
        };
        RenderPass(Arc::new(inner))
    }
}
