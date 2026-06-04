mod buffer_pool;
mod compute;
mod cpu;
mod error;
mod executor;
mod job;
mod pipeline;
mod planner;
mod scratch;
mod slot_table;
mod stack;
mod version;

pub use compute::ComputeBackend;
pub use cpu::{CpuComputeBackend, CpuKernel, CpuKernelFactory};
pub use error::{BraidError, BraidResult};
pub use executor::{BackendConfig, BackendHandle, BraidExecutor};
pub use job::{CancelFlag, JobPacket, JobStatus};
pub use pipeline::{
    BufferAccess, BufferBinding, BufferLayout, BufferSlot, BufferSpec, CompiledPlan, DispatchHint,
    ElementKind, JobId, KernelKind, KernelSpec, PipelineShape, PlanBuilder, StageSpec,
    StaticBuffer, StaticBufferSet, VersionId,
};
pub use planner::PlannerBackend;
pub use scratch::{BatchScratch, ComputeScratch, PlannerScratch};
pub use slot_table::{SlotKey, SlotTable};
pub use stack::Stack;

#[cfg(test)]
mod tests;
