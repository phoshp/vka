use std::cell::RefCell;
use std::ptr;

use anyhow::Ok;
use anyhow::anyhow;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::Buffer;
use crate::BufferDesc;
use crate::Image;
use crate::RenderingDevice;
use crate::Result;
use crate::utils;

pub const COPY_ALIGNMENT: u64 = 4;

pub struct StagingBelt {
    chunk_size: u64,
    pub(crate) active_chunks: Vec<StagingChunk>,
    pub(crate) readback_buffer: Option<Buffer>,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            chunk_size,
            active_chunks: Vec::new(),
            readback_buffer: None,
        }
    }

    pub fn ensure_readback_buffer(&mut self, rd: &RenderingDevice, size: u64) -> Result<()> {
        if self.readback_buffer.as_ref().map_or(true, |b| b.size < size) {
            let buf = rd.buffer_create(&BufferDesc::new(size).location(MemoryLocation::GpuToCpu))?;
            buf.set_name("staging readback buffer");
            self.readback_buffer = Some(buf);
        }
        Ok(())
    }

    pub fn read_image(&mut self, rd: &RenderingDevice, image: &Image, bytes_per_pixel: u64, region: &vk::BufferImageCopy) -> Result<*mut u8> {
        let size = region.image_extent.width as u64 * region.image_extent.height as u64 * region.image_extent.depth as u64 * bytes_per_pixel;
        if size <= 0 {
            return Err(anyhow!("Tried to read zero bytes from staging buffer"));
        }
        self.ensure_readback_buffer(rd, size)?;
        let staging_buffer = self.readback_buffer.as_ref().unwrap();
        rd.record(|dev, cmd| unsafe {
            let prev = rd.barrier_image(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            dev.cmd_copy_image_to_buffer(cmd, image.handle, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, staging_buffer.handle, &[region.clone()]);
            rd.barrier_image(cmd, image, prev);
        });

        Ok(staging_buffer.alloc().mapped_ptr().unwrap().as_ptr() as *mut u8)
    }

    pub fn read_buffer(&mut self, rd: &RenderingDevice, buffer: &Buffer, offset: u64, size: u64) -> Result<*mut u8> {
        if size <= 0 {
            return Err(anyhow!("Tried to read zero bytes from staging buffer"));
        }
        self.ensure_readback_buffer(rd, size)?;
        let staging_buffer = self.readback_buffer.as_ref().unwrap();
        rd.copy_buffer(
            buffer,
            staging_buffer,
            &[vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size,
            }],
        );

        Ok(staging_buffer.alloc().mapped_ptr().unwrap().as_ptr() as *mut u8)
    }

    pub fn write(&mut self, rd: &RenderingDevice, data: &[u8]) -> Option<(Buffer, u64, u64)> {
        let size = size_of_val(data) as u64;
        if size <= 0 {
            log::warn!("Tried to write zero bytes to staging buffer");
            return None;
        }

        let index = if let Some(i) = self.active_chunks.iter().position(|c| c.can_allocate(size)) {
            i
        } else {
            assert!(size < self.chunk_size);
            let buffer = rd
                .buffer_create(
                    &BufferDesc::new(self.chunk_size.max(size))
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .location(MemoryLocation::CpuToGpu),
                )
                .unwrap();
            buffer.set_name(format!("staging chunk {}", self.active_chunks.len()));
            self.active_chunks.push(StagingChunk { buffer, cursor: 0 });
            self.active_chunks.len() - 1
        };

        let chunk = &mut self.active_chunks[index];
        let offset = chunk.allocate(size);
        let ptr = chunk.buffer.alloc().mapped_ptr().ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED).unwrap().as_ptr() as *mut u8;
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), size as usize);
        }

        Some((chunk.buffer.clone(), offset, size))
    }

    pub fn reset(&mut self) {
        self.active_chunks.iter_mut().for_each(|c| {
            c.cursor = 0;
        });
    }
}

pub struct StagingChunk {
    pub buffer: Buffer,
    pub cursor: u64,
}

impl StagingChunk {
    pub fn can_allocate(&self, size: u64) -> bool {
        let end = utils::align_up(self.cursor + size, COPY_ALIGNMENT);
        end <= self.buffer.size
    }

    pub fn allocate(&mut self, size: u64) -> u64 {
        assert!(self.can_allocate(size));
        let offset = self.cursor;
        self.cursor = utils::align_up(self.cursor + size, COPY_ALIGNMENT);
        offset
    }
}
