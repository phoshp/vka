use std::ptr;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::Buffer;
use crate::BufferDesc;
use crate::RenderingDevice;

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

    pub fn download(&mut self, rd: &RenderingDevice, size: u64) -> (Buffer, *mut u8) {
        assert!(size > 0, "Tried to download zero bytes from buffer");
        if self.readback_buffer.as_ref().map_or(true, |b| b.size < size) {
            let buf = rd.new_buffer(&BufferDesc::new(size).location(MemoryLocation::GpuToCpu));
            self.readback_buffer = Some(buf);
        }
        let staging_buffer = self.readback_buffer.as_ref().unwrap();
        let ptr = staging_buffer.alloc.mapped_ptr().unwrap().as_ptr() as *mut u8;
        (staging_buffer.clone(), ptr)
    }

    pub fn upload(&mut self, rd: &RenderingDevice, data: &[u8]) -> (Buffer, u64, u64) {
        let size = size_of_val(data) as u64;
        assert!(size > 0, "Tried to upload zero bytes to buffer");
        let submission = rd.frame_counter.read();
        let index = if let Some(i) = self.active_chunks.iter().position(|c| c.can_allocate(size)) {
            i
        } else {
            let buffer = rd.new_buffer(
                &BufferDesc::new(self.chunk_size.max(size))
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .location(MemoryLocation::CpuToGpu),
            );
            self.active_chunks.push(StagingChunk {
                buffer,
                cursor: 0,
                last_used_submission: submission.0,
            });
            self.active_chunks.len() - 1
        };

        let chunk = &mut self.active_chunks[index];
        chunk.last_used_submission = submission.0;
        let offset = chunk.allocate(size);
        let ptr = chunk.buffer.alloc.mapped_ptr().unwrap().as_ptr() as *mut u8;
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), size as usize);
        }
        (chunk.buffer.clone(), offset, size)
    }

    pub fn maintain(&mut self, completed: u64) {
        self.active_chunks.iter_mut().for_each(|c| {
            if c.last_used_submission <= completed {
                c.cursor = 0;
            }
        });
    }
}

pub struct StagingChunk {
    pub buffer: Buffer,
    pub cursor: u64,
    pub last_used_submission: u64,
}

impl StagingChunk {
    pub fn can_allocate(&self, size: u64) -> bool {
        let end = crate::align_up(self.cursor + size, COPY_ALIGNMENT);
        end <= self.buffer.size
    }

    pub fn allocate(&mut self, size: u64) -> u64 {
        assert!(self.can_allocate(size));
        let offset = self.cursor;
        self.cursor = crate::align_up(self.cursor + size, COPY_ALIGNMENT);
        offset
    }
}
