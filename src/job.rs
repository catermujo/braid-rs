//! Per-job packet storage and cancellation/status types.

use crate::error::{BraidError, BraidResult};
use crate::pipeline::{BufferData, BufferSlot, ElementKind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Default)]
/// Cooperative cancellation flag shared with backend stage execution.
pub struct CancelFlag {
    inner: Arc<AtomicBool>,
}

impl CancelFlag {
    /// Request cancellation.
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::Release);
    }

    /// Return whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Coarse lifecycle state for a stack-local job.
pub enum JobStatus {
    /// Job is queued but has not started running.
    Queued,
    /// Job is actively encoding, staging, or decoding.
    Running,
    /// Job finished successfully and results can be collected.
    Completed,
    /// Job failed with an error.
    Failed,
    /// Job was cancelled cooperatively.
    Cancelled,
}

#[derive(Debug)]
/// One slot payload inside a [`JobPacket`].
pub struct PacketBuffer {
    /// Buffer slot id.
    pub slot: BufferSlot,
    /// Backing storage for the slot.
    pub data: BufferData,
}

#[derive(Debug, Default)]
/// Reusable per-job mutable buffer set shared between planner and backend.
pub struct JobPacket {
    query_count: usize,
    buffers: Vec<PacketBuffer>,
}

impl JobPacket {
    /// Clear logical contents while keeping allocations for reuse.
    pub fn clear_for_reuse(&mut self) {
        self.query_count = 0;
        for buffer in &mut self.buffers {
            buffer.data.clear();
        }
    }

    /// Return the number of queries encoded into this packet.
    pub fn query_count(&self) -> usize {
        self.query_count
    }

    /// Set the number of queries encoded into this packet.
    pub fn set_query_count(&mut self, query_count: usize) {
        self.query_count = query_count;
    }

    fn ensure_slot(&mut self, slot: BufferSlot, expected: ElementKind) -> usize {
        for (idx, buffer) in self.buffers.iter_mut().enumerate() {
            if buffer.slot != slot {
                continue;
            }
            if buffer.data.kind() != expected {
                buffer.data = match expected {
                    ElementKind::U32 => BufferData::U32(Vec::new()),
                    ElementKind::U64 => BufferData::U64(Vec::new()),
                    ElementKind::F32 => BufferData::F32(Vec::new()),
                };
            }
            return idx;
        }

        self.buffers.push(PacketBuffer {
            slot,
            data: match expected {
                ElementKind::U32 => BufferData::U32(Vec::new()),
                ElementKind::U64 => BufferData::U64(Vec::new()),
                ElementKind::F32 => BufferData::F32(Vec::new()),
            },
        });
        self.buffers.len() - 1
    }

    /// Ensure a `u32` buffer exists at `slot` and resize it to `len`.
    pub fn ensure_u32(&mut self, slot: BufferSlot, len: usize) -> &mut Vec<u32> {
        let idx = self.ensure_slot(slot, ElementKind::U32);
        match &mut self.buffers[idx].data {
            BufferData::U32(vals) => {
                vals.resize(len, 0);
                vals
            }
            _ => unreachable!(),
        }
    }

    /// Ensure a `u64` buffer exists at `slot` and resize it to `len`.
    pub fn ensure_u64(&mut self, slot: BufferSlot, len: usize) -> &mut Vec<u64> {
        let idx = self.ensure_slot(slot, ElementKind::U64);
        match &mut self.buffers[idx].data {
            BufferData::U64(vals) => {
                vals.resize(len, 0);
                vals
            }
            _ => unreachable!(),
        }
    }

    /// Ensure an `f32` buffer exists at `slot` and resize it to `len`.
    pub fn ensure_f32(&mut self, slot: BufferSlot, len: usize) -> &mut Vec<f32> {
        let idx = self.ensure_slot(slot, ElementKind::F32);
        match &mut self.buffers[idx].data {
            BufferData::F32(vals) => {
                vals.resize(len, 0.0);
                vals
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn load_static_buffer(&mut self, slot: BufferSlot, data: &BufferData) {
        match data {
            BufferData::U32(values) => {
                self.ensure_u32(slot, values.len()).copy_from_slice(values);
            }
            BufferData::U64(values) => {
                self.ensure_u64(slot, values.len()).copy_from_slice(values);
            }
            BufferData::F32(values) => {
                self.ensure_f32(slot, values.len()).copy_from_slice(values);
            }
        }
    }

    pub(crate) fn buffer_descriptors(
        &self,
    ) -> impl Iterator<Item = (BufferSlot, ElementKind, usize)> + '_ {
        self.buffers
            .iter()
            .map(|buffer| (buffer.slot, buffer.data.kind(), buffer.data.len()))
    }

    /// Read-only typed `u32` view for one slot.
    pub fn u32(&self, slot: BufferSlot) -> BraidResult<&[u32]> {
        self.view(slot, ElementKind::U32)
            .and_then(|buffer| match buffer {
                BufferData::U32(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    /// Mutable typed `u32` view for one slot.
    pub fn u32_mut(&mut self, slot: BufferSlot) -> BraidResult<&mut [u32]> {
        self.view_mut(slot, ElementKind::U32)
            .and_then(|buffer| match buffer {
                BufferData::U32(vals) => Ok(vals.as_mut_slice()),
                _ => unreachable!(),
            })
    }

    /// Read-only typed `u64` view for one slot.
    pub fn u64(&self, slot: BufferSlot) -> BraidResult<&[u64]> {
        self.view(slot, ElementKind::U64)
            .and_then(|buffer| match buffer {
                BufferData::U64(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    /// Read-only typed `f32` view for one slot.
    pub fn f32(&self, slot: BufferSlot) -> BraidResult<&[f32]> {
        self.view(slot, ElementKind::F32)
            .and_then(|buffer| match buffer {
                BufferData::F32(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    /// Mutable typed `f32` view for one slot.
    pub fn f32_mut(&mut self, slot: BufferSlot) -> BraidResult<&mut [f32]> {
        self.view_mut(slot, ElementKind::F32)
            .and_then(|buffer| match buffer {
                BufferData::F32(vals) => Ok(vals.as_mut_slice()),
                _ => unreachable!(),
            })
    }

    /// Borrow several distinct `f32` buffers at once.
    ///
    /// This is useful for kernels that need multiple mutable `f32` slices without copying.
    pub fn with_f32_buffers<R>(
        &mut self,
        slots: &[BufferSlot],
        f: impl FnOnce(Vec<&mut [f32]>) -> BraidResult<R>,
    ) -> BraidResult<R> {
        let mut indices = Vec::with_capacity(slots.len());
        for slot in slots {
            let Some(index) = self.buffers.iter().position(|buffer| buffer.slot == *slot) else {
                return Err(BraidError::MissingBuffer(*slot));
            };
            if indices.contains(&index) {
                return Err(BraidError::from("duplicate f32 buffer slot request"));
            }
            let buffer = &self.buffers[index];
            if buffer.data.kind() != ElementKind::F32 {
                return Err(BraidError::InvalidBufferType {
                    slot: *slot,
                    expected: ElementKind::F32,
                });
            }
            indices.push(index);
        }

        let mut ptrs = Vec::with_capacity(indices.len());
        for index in indices {
            let buffer = &mut self.buffers[index];
            match &mut buffer.data {
                BufferData::F32(vals) => ptrs.push(vals.as_mut_slice() as *mut [f32]),
                _ => unreachable!(),
            }
        }

        let mut views = Vec::with_capacity(ptrs.len());
        for ptr in ptrs {
            // The requested slots are unique, so these mutable views do not alias.
            unsafe {
                views.push(&mut *ptr);
            }
        }
        f(views)
    }

    fn view(&self, slot: BufferSlot, expected: ElementKind) -> BraidResult<&BufferData> {
        let buffer = self
            .buffers
            .iter()
            .find(|buffer| buffer.slot == slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        if buffer.data.kind() != expected {
            return Err(BraidError::InvalidBufferType { slot, expected });
        }
        Ok(&buffer.data)
    }

    fn view_mut(
        &mut self,
        slot: BufferSlot,
        expected: ElementKind,
    ) -> BraidResult<&mut BufferData> {
        let buffer = self
            .buffers
            .iter_mut()
            .find(|buffer| buffer.slot == slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        if buffer.data.kind() != expected {
            return Err(BraidError::InvalidBufferType { slot, expected });
        }
        Ok(&mut buffer.data)
    }
}
