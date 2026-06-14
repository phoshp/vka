use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

use ash::prelude::VkResult;
use ash::vk;
use itertools::Itertools;

use crate::Buffer;
use crate::Image;
use crate::ImageView;
use crate::Pipeline;
use crate::PipelineLayout;
use crate::RenderPass;

pub struct CommandEncoder {
    pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    device: ash::Device,

    active: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    render_pass_views: Vec<(ImageView, vk::ImageLayout)>,
    prev_image_layout: vk::ImageLayout,
    hasher: DefaultHasher,
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

const BIND_POINT_NONE: vk::PipelineBindPoint = vk::PipelineBindPoint::from_raw(!0);

const CMD_ALLOC_GRANULARITY: u32 = 8;

impl CommandEncoder {
    pub fn new(device: &ash::Device, family: u32) -> VkResult<Self> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::TRANSIENT).queue_family_index(family),
                None,
            )?
        };
        let free = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(CMD_ALLOC_GRANULARITY),
                )
                .unwrap()
        };
        Ok(CommandEncoder {
            pool,
            free,
            device: device.clone(),

            active: vk::CommandBuffer::null(),
            bind_point: BIND_POINT_NONE,
            render_pass_views: Vec::new(),
            prev_image_layout: vk::ImageLayout::UNDEFINED,
            hasher: DefaultHasher::new(),
        })
    }

    pub(crate) fn begin_encoding(&mut self) {
        unsafe {
            if let Some(cmd) = self.free.pop() {
                self.active = cmd;
            } else {
                self.free.extend(
                    self.device
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_pool(self.pool)
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_buffer_count(CMD_ALLOC_GRANULARITY),
                        )
                        .unwrap(),
                );
                self.active = self.free.pop().unwrap();
            }

            self.device
                .begin_command_buffer(self.active, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .unwrap();
            self.bind_point = BIND_POINT_NONE;
        }
    }

    pub(crate) fn end_encoding(&mut self) -> vk::CommandBuffer {
        unsafe {
            self.device.end_command_buffer(self.active).unwrap();
        }
        self.bind_point = BIND_POINT_NONE;
        self.active
    }

    pub(crate) fn reset(&mut self, cmd_buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::default()).unwrap();
        }
        self.bind_point = BIND_POINT_NONE;
        self.free.extend(cmd_buffers);
    }

    #[inline]
    fn check_bind_point(&self, expected: &[vk::PipelineBindPoint]) {
        debug_assert!(
            expected.iter().any(|&x| self.bind_point == x),
            "Expected bind point {:?}, but got {:?}",
            expected,
            self.bind_point
        );
    }
}

/// Base Pass
impl CommandEncoder {
    pub fn bind_pipeline(&mut self, pipeline: &Pipeline) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS, vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device.cmd_bind_pipeline(self.active, self.bind_point, pipeline.raw);
        }
    }

    pub fn bind_descriptor_sets(&mut self, layout: vk::PipelineLayout, first_set: u32, descriptor_sets: &[vk::DescriptorSet], dynamic_offsets: &[u32]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS, vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device
                .cmd_bind_descriptor_sets(self.active, self.bind_point, layout, first_set, descriptor_sets, dynamic_offsets);
        }
    }

    pub fn push_constants(&mut self, pipeline_layout: &PipelineLayout, offset: u32, data: &[u8]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS, vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device.cmd_push_constants(self.active, pipeline_layout.raw, vk::ShaderStageFlags::ALL, offset, data);
        }
    }
}

/// Render Pass
impl CommandEncoder {
    pub fn begin_render_pass(&mut self, rpass: &RenderPass, views: &[&ImageView], area: vk::Rect2D) {
        self.check_bind_point(&[BIND_POINT_NONE]);

        let extent = vk::Extent2D {
            width: area.extent.width + area.offset.x.max(0) as u32,
            height: area.extent.height + area.offset.y.max(0) as u32,
        };
        extent.hash(&mut self.hasher);
        views.iter().for_each(|t| t.id.hash(&mut self.hasher));
        let framebuffer_key = self.hasher.finish();

        let framebuffer = *rpass.framebuffers.lock().entry(framebuffer_key).or_insert_with(|| unsafe {
            self.device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(rpass.raw)
                        .attachments(&views.iter().map(|t| t.raw).collect_vec())
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
                .unwrap()
        });

        self.render_pass_views.clear();
        for (i, view) in views.iter().enumerate() {
            let img = view.image().expect("ImageView must have a valid image");
            if img.optimal_layout != rpass.layouts[i] {
                self.image_barrier(&img, img.optimal_layout, rpass.layouts[i]);
            }
            self.render_pass_views.push(((*view).clone(), rpass.layouts[i]));
        }
        self.bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe {
            self.device.cmd_begin_render_pass(
                self.active,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(rpass.raw)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(&rpass.clear_values),
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn clear_attachments(&mut self, attachments: &[vk::ClearAttachment], rects: &[vk::ClearRect]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_clear_attachments(self.active, attachments, rects);
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: &Buffer, offset: vk::DeviceSize, index_type: vk::IndexType) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_bind_index_buffer(self.active, buffer.raw, offset, index_type);
        }
    }

    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_bind_vertex_buffers(self.active, first_binding, buffers, offsets);
        }
    }

    pub fn set_viewport(&mut self, first_viewport: u32, viewports: &[vk::Viewport]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_viewport(self.active, first_viewport, viewports);
        }
    }

    pub fn set_scissor(&mut self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_scissor(self.active, first_scissor, scissors);
        }
    }

    pub fn set_line_width(&mut self, line_width: f32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_line_width(self.active, line_width);
        }
    }

    pub fn set_depth_bias(&mut self, depth_bias_constant_factor: f32, depth_bias_clamp: f32, depth_bias_slope_factor: f32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device
                .cmd_set_depth_bias(self.active, depth_bias_constant_factor, depth_bias_clamp, depth_bias_slope_factor);
        }
    }

    pub fn set_depth_bounds(&mut self, min_depth_bounds: f32, max_depth_bounds: f32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_depth_bounds(self.active, min_depth_bounds, max_depth_bounds);
        }
    }

    pub fn set_blend_constants(&mut self, blend_constants: &[f32; 4]) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_blend_constants(self.active, blend_constants);
        }
    }

    pub fn set_stencil_compare_mask(&mut self, face_mask: vk::StencilFaceFlags, stencil_compare_mask: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_stencil_compare_mask(self.active, face_mask, stencil_compare_mask);
        }
    }

    pub fn set_stencil_write_mask(&mut self, face_mask: vk::StencilFaceFlags, stencil_write_mask: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_stencil_write_mask(self.active, face_mask, stencil_write_mask);
        }
    }

    pub fn set_stencil_reference(&mut self, face_mask: vk::StencilFaceFlags, stencil_reference: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_set_stencil_reference(self.active, face_mask, stencil_reference);
        }
    }

    pub fn begin_rendering(&mut self, rendering_info: &vk::RenderingInfo) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        self.bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe {
            self.device.cmd_begin_rendering(self.active, rendering_info);
        }
    }

    pub fn next_subpass(&mut self) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_next_subpass(self.active, vk::SubpassContents::INLINE);
        }
    }

    pub fn end_render_pass(&mut self) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        self.bind_point = BIND_POINT_NONE;
        unsafe {
            self.device.cmd_end_render_pass(self.active);
        }
        for (view, layout) in self.render_pass_views.iter() {
            let img = view.image().expect("ImageView must have a valid image");
            if img.optimal_layout != *layout {
                self.image_barrier(&img, *layout, img.optimal_layout);
            }
        }
    }

    pub fn end_rendering(&mut self) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        self.bind_point = BIND_POINT_NONE;
        unsafe {
            self.device.cmd_end_rendering(self.active);
        }
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_draw(self.active, vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device
                .cmd_draw_indexed(self.active, index_count, instance_count, first_index, vertex_offset, first_instance);
        }
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_draw_indirect(self.active, buffer.raw, offset, draw_count, stride);
        }
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device.cmd_draw_indexed_indirect(self.active, buffer.raw, offset, draw_count, stride);
        }
    }

    // draw_indirect_count

    pub fn draw_indirect_count(&mut self, buffer: &Buffer, offset: vk::DeviceSize, count_buffer: &Buffer, count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device
                .cmd_draw_indirect_count(self.active, buffer.raw, offset, count_buffer.raw, count_offset, max_draw_count, stride);
        }
    }

    pub fn draw_indexed_indirect_count(&mut self, buffer: &Buffer, offset: vk::DeviceSize, count_buffer: &Buffer, count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::GRAPHICS]);
        unsafe {
            self.device
                .cmd_draw_indexed_indirect_count(self.active, buffer.raw, offset, count_buffer.raw, count_offset, max_draw_count, stride);
        }
    }
}

/// Memory Barriers
impl CommandEncoder {
    pub fn barrier(&self, src_stages: vk::PipelineStageFlags, dst_stages: vk::PipelineStageFlags) {
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            self.device
                .cmd_pipeline_barrier(self.active, src_stages, dst_stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
    }

    pub fn image_barrier(&self, image: &Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) {
        self.image_barrier_raw(image.raw, image.aspect, old_layout, new_layout);
    }

    pub fn image_barrier_begin(&mut self, image: &Image, new_layout: vk::ImageLayout) {
        self.image_barrier_raw(image.raw, image.aspect, image.optimal_layout, new_layout);
        self.prev_image_layout = new_layout;
    }

    pub fn image_barrier_end(&mut self, image: &Image) {
        self.image_barrier_raw(image.raw, image.aspect, self.prev_image_layout, image.optimal_layout);
    }

    pub fn image_barrier_raw(&self, image: vk::Image, aspect_mask: vk::ImageAspectFlags, old_layout: vk::ImageLayout, mut new_layout: vk::ImageLayout) {
        self.check_bind_point(&[vk::PipelineBindPoint::COMPUTE, BIND_POINT_NONE]);
        unsafe {
            if new_layout == vk::ImageLayout::UNDEFINED || new_layout == vk::ImageLayout::PREINITIALIZED {
                new_layout = old_layout;
            }
            let (src_stages, src_access) = match old_layout {
                vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
                vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::SHADER_READ | vk::AccessFlags::INPUT_ATTACHMENT_READ),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE),
                vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
                _ => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE),
            };

            let (dst_stages, dst_access) = match new_layout {
                vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::SHADER_READ | vk::AccessFlags::INPUT_ATTACHMENT_READ),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE),
                vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::empty()),
                _ => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
            };

            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            self.device
                .cmd_pipeline_barrier(self.active, src_stages, dst_stages, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
        }
    }
}

/// Compute Pass
impl CommandEncoder {
    pub fn begin_compute_pass(&mut self) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        self.bind_point = vk::PipelineBindPoint::COMPUTE;
    }

    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device.cmd_dispatch(self.active, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn dispatch_indirect(&self, buffer: &Buffer, offset: vk::DeviceSize) {
        self.check_bind_point(&[vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device.cmd_dispatch_indirect(self.active, buffer.raw, offset);
        }
    }

    pub fn dispatch_base(&self, base_group_x: u32, base_group_y: u32, base_group_z: u32, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.check_bind_point(&[vk::PipelineBindPoint::COMPUTE]);
        unsafe {
            self.device
                .cmd_dispatch_base(self.active, base_group_x, base_group_y, base_group_z, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn end_compute_pass(&mut self) {
        self.check_bind_point(&[vk::PipelineBindPoint::COMPUTE]);
        self.bind_point = BIND_POINT_NONE;
    }
}

/// Transfer
impl CommandEncoder {
    pub fn copy_buffer(&mut self, src_buffer: &Buffer, dst_buffer: &Buffer, regions: &[vk::BufferCopy]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device.cmd_copy_buffer(self.active, src_buffer.raw, dst_buffer.raw, regions);
        }
    }

    pub fn copy_image(&mut self, src_image: &Image, dst_image: &Image, regions: &[vk::ImageCopy]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device.cmd_copy_image(
                self.active,
                src_image.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );
        }
    }

    pub fn copy_buffer_to_image(&mut self, src_buffer: &Buffer, dst_image: &Image, regions: &[vk::BufferImageCopy]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device
                .cmd_copy_buffer_to_image(self.active, src_buffer.raw, dst_image.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, regions);
        }
    }

    pub fn copy_image_to_buffer(&mut self, src_image: &Image, dst_buffer: &Buffer, regions: &[vk::BufferImageCopy]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device
                .cmd_copy_image_to_buffer(self.active, src_image.raw, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst_buffer.raw, regions);
        }
    }

    pub fn blit_image(&mut self, src_image: &Image, dst_image: &Image, regions: &[vk::ImageBlit], filter: vk::Filter) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device.cmd_blit_image(
                self.active,
                src_image.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
                filter,
            );
        }
    }

    pub fn fill_buffer(&mut self, dst_buffer: &Buffer, dst_offset: vk::DeviceSize, size: vk::DeviceSize, data: u32) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device.cmd_fill_buffer(self.active, dst_buffer.raw, dst_offset, size, data);
        }
    }

    pub fn update_buffer(&mut self, dst_buffer: &Buffer, dst_offset: vk::DeviceSize, data: &[u8]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device.cmd_update_buffer(self.active, dst_buffer.raw, dst_offset, data);
        }
    }

    pub fn clear_color_image(&mut self, image: &Image, color: vk::ClearColorValue, ranges: &[vk::ImageSubresourceRange]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device
                .cmd_clear_color_image(self.active, image.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, ranges);
        }
    }

    pub fn clear_depth_stencil_image(&mut self, image: &Image, depth_stencil: vk::ClearDepthStencilValue, ranges: &[vk::ImageSubresourceRange]) {
        self.check_bind_point(&[BIND_POINT_NONE]);
        unsafe {
            self.device
                .cmd_clear_depth_stencil_image(self.active, image.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &depth_stencil, ranges);
        }
    }
}

/// Queries
impl CommandEncoder {
    pub fn begin_query(&mut self, query_pool: vk::QueryPool, query: u32, flags: vk::QueryControlFlags) {
        unsafe {
            self.device.cmd_begin_query(self.active, query_pool, query, flags);
        }
    }

    pub fn end_query(&mut self, query_pool: vk::QueryPool, query: u32) {
        unsafe {
            self.device.cmd_end_query(self.active, query_pool, query);
        }
    }

    pub fn reset_query_pool(&mut self, query_pool: vk::QueryPool, first_query: u32, query_count: u32) {
        unsafe {
            self.device.cmd_reset_query_pool(self.active, query_pool, first_query, query_count);
        }
    }

    pub fn copy_query_pool_results(
        &mut self,
        query_pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: &Buffer,
        dst_offset: vk::DeviceSize,
        stride: vk::DeviceSize,
        flags: vk::QueryResultFlags,
    ) {
        unsafe {
            self.device
                .cmd_copy_query_pool_results(self.active, query_pool, first_query, query_count, dst_buffer.raw, dst_offset, stride, flags);
        }
    }

    pub fn write_timestamp(&mut self, pipeline_stage: vk::PipelineStageFlags, query_pool: vk::QueryPool, query: u32) {
        unsafe {
            self.device.cmd_write_timestamp(self.active, pipeline_stage, query_pool, query);
        }
    }

    pub fn execute_commands(&mut self, command_buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.device.cmd_execute_commands(self.active, command_buffers);
        }
    }

    pub fn set_device_mask(&self, device_mask: u32) {
        unsafe {
            self.device.cmd_set_device_mask(self.active, device_mask);
        }
    }
}
