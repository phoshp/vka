use std::ptr;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::Buffer;
use crate::RenderingDevice;
use crate::utils;

pub const COPY_ALIGNMENT: u64 = 4;

pub struct StagingBelt {
    chunk_size: u64,
    active_chunks: Vec<StagingChunk>,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self { chunk_size, active_chunks: Vec::new() }
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
            let buffer = rd.buffer_create(self.chunk_size.max(size), vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu).unwrap();
            self.active_chunks.push(StagingChunk { buffer, cursor: 0 });
            self.active_chunks.len() - 1
        };

        let chunk = &mut self.active_chunks[index];
        let offset = chunk.allocate(size);
        let ptr = chunk.buffer.alloc().unwrap().mapped_ptr().ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED).unwrap().as_ptr() as *mut u8;
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
