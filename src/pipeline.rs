//! Generic compiled pipeline types shared between planners and backends.
//!
//! Planners emit these shapes. Backends consume them. The types here intentionally avoid
//! planner-specific meaning.

use crate::error::{BraidError, BraidResult};
use crate::job::JobPacket;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

/// Opaque stack-local job identifier.
pub type JobId = u64;
/// Monotonic identifier for frozen compiled stack versions.
pub type VersionId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Primitive element types supported by packet buffers.
pub enum ElementKind {
    /// Unsigned 32-bit integers.
    U32,
    /// Unsigned 64-bit integers.
    U64,
    /// 32-bit floating-point values.
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Declared layout contract for one pipeline buffer slot.
pub enum BufferLayout {
    /// Exactly one element per query.
    PerQueryScalar,
    /// A fixed-width vector per query.
    PerQueryVector {
        /// Element count for each query.
        width: usize,
    },
    /// Planner/backend-managed variable-length buffer.
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Access mode declared for one kernel binding.
pub enum BufferAccess {
    /// Read-only access.
    Read,
    /// Write-only access.
    Write,
    /// Read-write access.
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Scheduling hint for how one kernel prefers to run.
pub enum DispatchHint {
    /// Run one kernel invocation across the whole batch.
    WholeBatch,
    /// Split the query batch across shards when backend supports it.
    QuerySharded,
    /// Run strictly serially.
    Serial,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Opaque numeric slot identifier for packet and static buffers.
pub struct BufferSlot(u16);

impl BufferSlot {
    /// Create a slot from its raw numeric value.
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the raw numeric slot value.
    pub const fn raw(self) -> u16 {
        self.0
    }
}

impl Display for BufferSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u16> for BufferSlot {
    fn from(value: u16) -> Self {
        Self::new(value)
    }
}

impl From<BufferSlot> for u16 {
    fn from(value: BufferSlot) -> Self {
        value.raw()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Opaque numeric identifier for backend kernel implementations.
pub struct KernelKind(u32);

impl KernelKind {
    /// Create a kernel kind from its raw numeric value.
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Return the raw numeric kernel kind value.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Display for KernelKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for KernelKind {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<KernelKind> for u32 {
    fn from(value: KernelKind) -> Self {
        value.raw()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Declares one buffer slot used by a compiled pipeline.
pub struct BufferSpec {
    /// Slot id used to address the buffer in packets and bindings.
    pub slot: BufferSlot,
    /// Element type stored in this slot.
    pub element_kind: ElementKind,
    /// Declared logical layout of the buffer.
    pub layout: BufferLayout,
}

impl BufferSpec {
    /// Create a custom buffer declaration.
    pub const fn new(slot: BufferSlot, element_kind: ElementKind, layout: BufferLayout) -> Self {
        Self {
            slot,
            element_kind,
            layout,
        }
    }

    /// Declare a scalar-per-query buffer.
    pub const fn per_query_scalar(slot: BufferSlot, element_kind: ElementKind) -> Self {
        Self::new(slot, element_kind, BufferLayout::PerQueryScalar)
    }

    /// Declare a fixed-width vector-per-query buffer.
    pub const fn per_query_vector(
        slot: BufferSlot,
        element_kind: ElementKind,
        width: usize,
    ) -> Self {
        Self::new(slot, element_kind, BufferLayout::PerQueryVector { width })
    }

    /// Declare a planner/backend-managed dynamic buffer.
    pub const fn dynamic(slot: BufferSlot, element_kind: ElementKind) -> Self {
        Self::new(slot, element_kind, BufferLayout::Dynamic)
    }

    fn validate_len(&self, query_count: usize, len: usize) -> BraidResult<()> {
        let expected_len = match self.layout {
            BufferLayout::PerQueryScalar => Some(query_count),
            BufferLayout::PerQueryVector { width } => query_count.checked_mul(width),
            BufferLayout::Dynamic => return Ok(()),
        };

        let Some(expected_len) = expected_len else {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} length overflow for declared layout",
                self.slot
            )));
        };

        if len != expected_len {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} has length {} but declared layout expects {}",
                self.slot, len, expected_len
            )));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One kernel's view of one pipeline buffer slot.
pub struct BufferBinding {
    /// Slot referenced by the kernel.
    pub slot: BufferSlot,
    /// Declared access mode.
    pub access: BufferAccess,
}

impl BufferBinding {
    /// Create a generic buffer binding.
    pub const fn new(slot: BufferSlot, access: BufferAccess) -> Self {
        Self { slot, access }
    }

    /// Read-only binding helper.
    pub const fn read(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::Read)
    }

    /// Write-only binding helper.
    pub const fn write(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::Write)
    }

    /// Read-write binding helper.
    pub const fn read_write(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::ReadWrite)
    }
}

#[derive(Clone, Debug)]
/// One compiled kernel invocation inside a stage.
pub struct KernelSpec {
    /// Backend kernel kind to instantiate.
    pub kind_id: KernelKind,
    /// Planner-defined opaque payload for backend preparation.
    pub payload: Arc<[u8]>,
    /// Buffers this kernel reads or writes.
    pub bindings: Vec<BufferBinding>,
    /// Scheduler hint for batch execution.
    pub dispatch: DispatchHint,
}

impl KernelSpec {
    /// Create one kernel spec with payload bytes.
    pub fn new(kind_id: KernelKind, payload: impl Into<Arc<[u8]>>) -> Self {
        Self {
            kind_id,
            payload: payload.into(),
            bindings: Vec::new(),
            dispatch: DispatchHint::WholeBatch,
        }
    }

    /// Create one kernel spec with no payload.
    pub fn empty(kind_id: KernelKind) -> Self {
        Self::new(kind_id, Vec::<u8>::new())
    }

    /// Attach buffer bindings to the kernel.
    pub fn with_bindings(mut self, bindings: impl IntoIterator<Item = BufferBinding>) -> Self {
        self.bindings.extend(bindings);
        self
    }

    /// Override the kernel dispatch hint.
    pub const fn with_dispatch(mut self, dispatch: DispatchHint) -> Self {
        self.dispatch = dispatch;
        self
    }
}

#[derive(Clone, Debug, Default)]
/// Barrier-separated group of kernels.
pub struct StageSpec {
    /// Kernels executed within this stage.
    pub kernels: Vec<KernelSpec>,
}

impl StageSpec {
    /// Helper for a one-kernel stage.
    pub fn single(kernel: KernelSpec) -> Self {
        Self {
            kernels: vec![kernel],
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Full buffer and stage layout for a compiled plan.
pub struct PipelineShape {
    /// Declared packet/static buffer slots used by the pipeline.
    pub buffers: Vec<BufferSpec>,
    /// Ordered stage list.
    pub stages: Vec<StageSpec>,
}

#[derive(Clone, Debug)]
/// Type-erased buffer storage for packet and static buffers.
pub enum BufferData {
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
}

impl BufferData {
    /// Return the element type of this buffer.
    pub fn kind(&self) -> ElementKind {
        match self {
            Self::U32(_) => ElementKind::U32,
            Self::U64(_) => ElementKind::U64,
            Self::F32(_) => ElementKind::F32,
        }
    }

    /// Clear logical contents while keeping capacity for reuse.
    pub fn clear(&mut self) {
        match self {
            Self::U32(vals) => vals.clear(),
            Self::U64(vals) => vals.clear(),
            Self::F32(vals) => vals.clear(),
        }
    }

    /// Return the logical element count.
    pub fn len(&self) -> usize {
        match self {
            Self::U32(vals) => vals.len(),
            Self::U64(vals) => vals.len(),
            Self::F32(vals) => vals.len(),
        }
    }
}

#[derive(Clone, Debug)]
/// Immutable static buffer loaded into packets before stage execution.
pub struct StaticBuffer {
    /// Slot addressed by the static buffer.
    pub slot: BufferSlot,
    /// Static data stored in that slot.
    pub data: BufferData,
}

impl StaticBuffer {
    /// Create one static buffer.
    pub fn new(slot: BufferSlot, data: BufferData) -> Self {
        Self { slot, data }
    }
}

/// Collection of static buffers attached to a compiled plan.
pub type StaticBufferSet = Vec<StaticBuffer>;

#[derive(Clone, Debug)]
/// Planner output consumed by `Stack` creation, recompile, and backend prepare.
pub struct CompiledPlan<M> {
    /// Generic pipeline layout.
    pub pipeline: PipelineShape,
    /// Planner-provided immutable static buffers.
    pub static_buffers: StaticBufferSet,
    /// Planner-specific metadata preserved for encode/decode.
    pub planner_meta: M,
}

impl<M> CompiledPlan<M> {
    /// Start building a compiled plan with planner metadata.
    pub fn builder(planner_meta: M) -> PlanBuilder<M> {
        PlanBuilder::new(planner_meta)
    }

    /// Validate slot declarations, static buffers, and kernel bindings.
    pub fn validate(&self) -> BraidResult<()> {
        let specs = self.specs_by_slot()?;
        let mut static_slots = HashMap::with_capacity(self.static_buffers.len());
        for buffer in &self.static_buffers {
            if static_slots.insert(buffer.slot, ()).is_some() {
                return Err(BraidError::InvalidSpec(format!(
                    "duplicate static buffer slot {}",
                    buffer.slot
                )));
            }
            let Some(spec) = specs.get(&buffer.slot) else {
                return Err(BraidError::InvalidSpec(format!(
                    "static buffer slot {} is not declared in pipeline",
                    buffer.slot
                )));
            };
            if spec.element_kind != buffer.data.kind() {
                return Err(BraidError::InvalidSpec(format!(
                    "static buffer slot {} has wrong element kind",
                    buffer.slot
                )));
            }
        }

        for (stage_index, stage) in self.pipeline.stages.iter().enumerate() {
            for (kernel_index, kernel) in stage.kernels.iter().enumerate() {
                for binding in &kernel.bindings {
                    if !specs.contains_key(&binding.slot) {
                        return Err(BraidError::InvalidSpec(format!(
                            "stage {} kernel {} references undeclared buffer slot {}",
                            stage_index, kernel_index, binding.slot
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn validate_packet(&self, packet: &JobPacket) -> BraidResult<()> {
        let specs = self.specs_by_slot()?;
        for (slot, kind, len) in packet.buffer_descriptors() {
            let Some(spec) = specs.get(&slot) else {
                if len == 0 {
                    continue;
                }
                return Err(BraidError::InvalidSpec(format!(
                    "packet contains undeclared buffer slot {}",
                    slot
                )));
            };
            if spec.element_kind != kind {
                return Err(BraidError::InvalidBufferType {
                    slot,
                    expected: spec.element_kind,
                });
            }
            spec.validate_len(packet.query_count(), len)?;
        }

        Ok(())
    }

    fn specs_by_slot(&self) -> BraidResult<HashMap<BufferSlot, &BufferSpec>> {
        let mut specs = HashMap::with_capacity(self.pipeline.buffers.len());
        for spec in &self.pipeline.buffers {
            if specs.insert(spec.slot, spec).is_some() {
                return Err(BraidError::InvalidSpec(format!(
                    "duplicate buffer slot {} in pipeline",
                    spec.slot
                )));
            }
        }
        Ok(specs)
    }
}

#[derive(Clone, Debug)]
/// Convenience builder for [`CompiledPlan`].
pub struct PlanBuilder<M> {
    pipeline: PipelineShape,
    static_buffers: StaticBufferSet,
    planner_meta: M,
}

impl<M> PlanBuilder<M> {
    /// Create an empty plan builder with planner metadata.
    pub fn new(planner_meta: M) -> Self {
        Self {
            pipeline: PipelineShape::default(),
            static_buffers: Vec::new(),
            planner_meta,
        }
    }

    /// Append one buffer declaration.
    pub fn buffer(&mut self, spec: BufferSpec) -> &mut Self {
        self.pipeline.buffers.push(spec);
        self
    }

    /// Append one stage.
    pub fn stage(&mut self, stage: StageSpec) -> &mut Self {
        self.pipeline.stages.push(stage);
        self
    }

    /// Append one static buffer.
    pub fn static_buffer(&mut self, buffer: StaticBuffer) -> &mut Self {
        self.static_buffers.push(buffer);
        self
    }

    /// Build without validation.
    pub fn build(self) -> CompiledPlan<M> {
        CompiledPlan {
            pipeline: self.pipeline,
            static_buffers: self.static_buffers,
            planner_meta: self.planner_meta,
        }
    }

    /// Build and validate the final plan.
    pub fn build_checked(self) -> BraidResult<CompiledPlan<M>> {
        let plan = self.build();
        plan.validate()?;
        Ok(plan)
    }
}
