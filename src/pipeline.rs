use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub type JobId = u64;
pub type VersionId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    U32,
    U64,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLayout {
    PerQueryScalar,
    PerQueryVector { width: usize },
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchHint {
    WholeBatch,
    QuerySharded,
    Serial,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BufferSlot(u16);

impl BufferSlot {
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

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
pub struct KernelKind(u32);

impl KernelKind {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

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
pub struct BufferSpec {
    pub slot: BufferSlot,
    pub element_kind: ElementKind,
    pub layout: BufferLayout,
}

impl BufferSpec {
    pub const fn new(slot: BufferSlot, element_kind: ElementKind, layout: BufferLayout) -> Self {
        Self {
            slot,
            element_kind,
            layout,
        }
    }

    pub const fn per_query_scalar(slot: BufferSlot, element_kind: ElementKind) -> Self {
        Self::new(slot, element_kind, BufferLayout::PerQueryScalar)
    }

    pub const fn per_query_vector(
        slot: BufferSlot,
        element_kind: ElementKind,
        width: usize,
    ) -> Self {
        Self::new(slot, element_kind, BufferLayout::PerQueryVector { width })
    }

    pub const fn dynamic(slot: BufferSlot, element_kind: ElementKind) -> Self {
        Self::new(slot, element_kind, BufferLayout::Dynamic)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub slot: BufferSlot,
    pub access: BufferAccess,
}

impl BufferBinding {
    pub const fn new(slot: BufferSlot, access: BufferAccess) -> Self {
        Self { slot, access }
    }

    pub const fn read(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::Read)
    }

    pub const fn write(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::Write)
    }

    pub const fn read_write(slot: BufferSlot) -> Self {
        Self::new(slot, BufferAccess::ReadWrite)
    }
}

#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub kind_id: KernelKind,
    pub payload: Arc<[u8]>,
    pub bindings: Vec<BufferBinding>,
    pub dispatch: DispatchHint,
}

impl KernelSpec {
    pub fn new(kind_id: KernelKind, payload: impl Into<Arc<[u8]>>) -> Self {
        Self {
            kind_id,
            payload: payload.into(),
            bindings: Vec::new(),
            dispatch: DispatchHint::WholeBatch,
        }
    }

    pub fn empty(kind_id: KernelKind) -> Self {
        Self::new(kind_id, Vec::<u8>::new())
    }

    pub fn with_bindings(mut self, bindings: impl IntoIterator<Item = BufferBinding>) -> Self {
        self.bindings.extend(bindings);
        self
    }

    pub const fn with_dispatch(mut self, dispatch: DispatchHint) -> Self {
        self.dispatch = dispatch;
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct StageSpec {
    pub kernels: Vec<KernelSpec>,
}

impl StageSpec {
    pub fn single(kernel: KernelSpec) -> Self {
        Self {
            kernels: vec![kernel],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PipelineShape {
    pub buffers: Vec<BufferSpec>,
    pub stages: Vec<StageSpec>,
}

#[derive(Clone, Debug)]
pub enum BufferData {
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
}

impl BufferData {
    pub fn kind(&self) -> ElementKind {
        match self {
            Self::U32(_) => ElementKind::U32,
            Self::U64(_) => ElementKind::U64,
            Self::F32(_) => ElementKind::F32,
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::U32(vals) => vals.clear(),
            Self::U64(vals) => vals.clear(),
            Self::F32(vals) => vals.clear(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U32(vals) => vals.len(),
            Self::U64(vals) => vals.len(),
            Self::F32(vals) => vals.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StaticBuffer {
    pub slot: BufferSlot,
    pub data: BufferData,
}

impl StaticBuffer {
    pub fn new(slot: BufferSlot, data: BufferData) -> Self {
        Self { slot, data }
    }
}

pub type StaticBufferSet = Vec<StaticBuffer>;

#[derive(Clone, Debug)]
pub struct CompiledPlan<M> {
    pub pipeline: PipelineShape,
    pub static_buffers: StaticBufferSet,
    pub planner_meta: M,
}

impl<M> CompiledPlan<M> {
    pub fn builder(planner_meta: M) -> PlanBuilder<M> {
        PlanBuilder::new(planner_meta)
    }
}

#[derive(Clone, Debug)]
pub struct PlanBuilder<M> {
    pipeline: PipelineShape,
    static_buffers: StaticBufferSet,
    planner_meta: M,
}

impl<M> PlanBuilder<M> {
    pub fn new(planner_meta: M) -> Self {
        Self {
            pipeline: PipelineShape::default(),
            static_buffers: Vec::new(),
            planner_meta,
        }
    }

    pub fn buffer(&mut self, spec: BufferSpec) -> &mut Self {
        self.pipeline.buffers.push(spec);
        self
    }

    pub fn stage(&mut self, stage: StageSpec) -> &mut Self {
        self.pipeline.stages.push(stage);
        self
    }

    pub fn static_buffer(&mut self, buffer: StaticBuffer) -> &mut Self {
        self.static_buffers.push(buffer);
        self
    }

    pub fn build(self) -> CompiledPlan<M> {
        CompiledPlan {
            pipeline: self.pipeline,
            static_buffers: self.static_buffers,
            planner_meta: self.planner_meta,
        }
    }
}
