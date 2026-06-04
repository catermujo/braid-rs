//! Core error types used across planner, backend, and executor code.

use crate::pipeline::{BufferSlot, ElementKind, KernelKind};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Standard result alias used by `braid`.
pub type BraidResult<T> = Result<T, BraidError>;

#[derive(Clone, Debug)]
/// Error type for core stack, planner, backend, and packet operations.
pub enum BraidError {
    /// Job was cancelled cooperatively.
    Cancelled,
    /// A stack-local job id was not found.
    UnknownJob,
    /// Executor or backend runtime is shutting down.
    ExecutorShutdown,
    /// Backend did not know how to prepare the requested kernel kind.
    BackendRejectedKernel(KernelKind),
    /// Requested packet buffer slot was missing.
    MissingBuffer(BufferSlot),
    /// Packet buffer slot existed with the wrong element type.
    InvalidBufferType {
        /// Slot that had the wrong type.
        slot: BufferSlot,
        /// Expected element kind for that slot.
        expected: ElementKind,
    },
    /// Planner encountered a duplicate identifier.
    DuplicateId {
        /// Identifier kind label.
        kind: &'static str,
        /// Duplicate identifier value.
        id: String,
    },
    /// Planner encountered a missing referenced identifier.
    MissingReference {
        /// Identifier kind label.
        kind: &'static str,
        /// Owner identifier value.
        id: String,
        /// Missing referenced identifier.
        reference: String,
    },
    /// Planner encountered an empty required scope.
    EmptyScope {
        /// Identifier kind label.
        kind: &'static str,
        /// Identifier value with empty scope.
        id: String,
    },
    /// Generic invalid compiled-plan or pipeline-layout error.
    InvalidSpec(String),
    /// Generic message error.
    Message(String),
    /// Shared synchronization primitive was poisoned.
    Poisoned(&'static str),
}

impl BraidError {
    /// Helper for creating a poisoned shared-state error.
    pub fn poisoned(name: &'static str) -> Self {
        Self::Poisoned(name)
    }
}

impl Display for BraidError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "job cancelled"),
            Self::UnknownJob => write!(f, "unknown job"),
            Self::ExecutorShutdown => write!(f, "executor is shut down"),
            Self::BackendRejectedKernel(kind) => {
                write!(f, "backend rejected kernel kind {}", kind)
            }
            Self::MissingBuffer(slot) => write!(f, "missing buffer at slot {}", slot),
            Self::InvalidBufferType { slot, expected } => {
                write!(
                    f,
                    "invalid buffer type at slot {} expected {:?}",
                    slot, expected
                )
            }
            Self::DuplicateId { kind, id } => write!(f, "duplicate {} id '{}'", kind, id),
            Self::MissingReference {
                kind,
                id,
                reference,
            } => write!(f, "{} '{}' references missing '{}'", kind, id, reference),
            Self::EmptyScope { kind, id } => write!(f, "{} '{}' has empty scope", kind, id),
            Self::InvalidSpec(msg) => write!(f, "invalid spec: {}", msg),
            Self::Message(msg) => write!(f, "{msg}"),
            Self::Poisoned(name) => write!(f, "shared state '{}' was poisoned", name),
        }
    }
}

impl Error for BraidError {}

impl From<String> for BraidError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

impl From<&str> for BraidError {
    fn from(value: &str) -> Self {
        Self::Message(value.to_owned())
    }
}
