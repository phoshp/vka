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

    pub fn download(&mut self, rd: &RenderingDevice, size: u64) -> Result<(Buffer, *mut u8)> {
        if size <= 0 {
            return Err(anyhow!("Tried to read zero bytes from buffer"));
        }
        if self.readback_buffer.as_ref().map_or(true, |b| b.size < size) {
            let buf = rd.buffer_create(&BufferDesc::new(size).location(MemoryLocation::GpuToCpu))?;
            buf.set_name("staging readback buffer");
            self.readback_buffer = Some(buf);
        }
        let staging_buffer = self.readback_buffer.as_ref().unwrap();
        let ptr = staging_buffer.alloc().mapped_ptr().ok_or(anyhow!("Failed to map staging buffer"))?.as_ptr() as *mut u8;
        Ok((staging_buffer.clone(), ptr))
    }

    pub fn upload(&mut self, rd: &RenderingDevice, data: &[u8]) -> Result<(Buffer, u64, u64)> {
        let size = size_of_val(data) as u64;
        if size <= 0 {
            return Err(anyhow!("Tried to write zero bytes to staging buffer"));
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
        Ok((chunk.buffer.clone(), offset, size))
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
