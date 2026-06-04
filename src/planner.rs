use crate::error::BraidResult;
use crate::job::JobPacket;
use crate::pipeline::CompiledPlan;
use crate::scratch::{BatchScratch, PlannerScratch};

/// Planner-side interface for turning domain data into an executable pipeline.
///
/// A planner owns all domain meaning:
///
/// - how authored specs become mutable planner state
/// - how state changes are applied
/// - how state compiles into a generic [`CompiledPlan`]
/// - how queries encode into a [`JobPacket`]
/// - how backend output decodes into user-facing resolutions
pub trait PlannerBackend: Send + Sync + 'static {
    /// Initial authored input used to build planner state.
    type Spec: Send + 'static;
    /// Mutable planner-owned state kept by a [`crate::Stack`].
    type State: Send + 'static;
    /// Incremental change description applied to planner state.
    type Change: Send + 'static;
    /// Per-dispatch query item.
    type Query: Send + Sync + 'static;
    /// Per-query decoded output returned by [`crate::Stack::collect`].
    type Resolution: Send + 'static;
    /// Planner-specific metadata stored inside [`CompiledPlan`].
    type PlannerMeta: Send + Sync + 'static;

    /// Build initial mutable state from a spec.
    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State>;
    /// Reset an existing state object from a fresh spec, preferably reusing storage.
    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()>;
    /// Apply changes in place to the current mutable state.
    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()>;
    /// Build the next state from the current state plus changes without mutating the old one.
    ///
    /// This supports transactional update flow: compile the new state first, then swap it in only
    /// if compile succeeds.
    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State>;
    /// Compile planner state into a generic pipeline plus planner metadata.
    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>>;
    /// Encode a batch of planner queries into a reusable packet buffer.
    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()>;
    /// Decode backend output from a packet into user-facing query resolutions.
    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>>;
}
