//! Re-exports [`VectorIndex`] from [`crate::vector`] for backwards compatibility.
//!
//! The canonical definition lives in `vector::mod` where it is co-located with
//! the backend implementations. This module is kept so that existing import paths
//! (`use crate::vector_index::VectorIndex`) continue to resolve without changes.

/// Re-export so that `use crate::vector_index::VectorIndex` keeps resolving.
pub use crate::vector::VectorIndex;
