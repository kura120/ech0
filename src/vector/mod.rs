//! Vector index backends for ech0.
//!
//! This module owns the [`VectorIndex`] trait and all backend implementations.
//! The concrete backend is selected at compile time via feature flags.
//! Callers hold `Arc<dyn VectorIndex>` — never a concrete backend type directly.
//!
//! ## Feature flags
//!
//! | Feature | Backend |
//! |---|---|
//! | `backend-hora` | [`HoraVectorLayer`] — pure Rust, default |
//! | `backend-usearch` | [`UsearchVectorLayer`] — high-performance, requires C++ toolchain |
//!
//! When both flags are active, `backend-hora` takes precedence for `DefaultVectorLayer`.

use uuid::Uuid;

use crate::error::EchoError;

// ---------------------------------------------------------------------------
// Submodule declarations — feature-gated per backend
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-usearch")]
pub mod usearch;

#[cfg(feature = "backend-hora")]
pub mod hora;

// ---------------------------------------------------------------------------
// Concrete type re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-usearch")]
pub use usearch::UsearchVectorLayer;

#[cfg(feature = "backend-hora")]
pub use hora::HoraVectorLayer;

// ---------------------------------------------------------------------------
// DefaultVectorLayer — resolved at compile time
//
// HoraVectorLayer takes precedence when both features are active.
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-hora")]
pub use hora::HoraVectorLayer as DefaultVectorLayer;

#[cfg(all(feature = "backend-usearch", not(feature = "backend-hora")))]
pub use usearch::UsearchVectorLayer as DefaultVectorLayer;

// ---------------------------------------------------------------------------
// VectorIndex trait
// ---------------------------------------------------------------------------

/// Abstraction over vector index backends.
///
/// All implementations must be `Send + Sync` and handle their own internal
/// synchronization. `Store` holds `Arc<dyn VectorIndex>` and never imports a
/// specific backend directly.
pub trait VectorIndex: Send + Sync {
    /// Add a single embedding. Returns the `u64` label assigned in the index.
    fn add(&self, node_id: Uuid, embedding: &[f32]) -> Result<u64, EchoError>;

    /// Add a batch of embeddings. Validates all dimensions before inserting any.
    /// Returns `(Uuid, label)` mappings for all inserted vectors.
    fn add_batch(&self, embeddings: &[(Uuid, Vec<f32>)]) -> Result<Vec<(Uuid, u64)>, EchoError>;

    /// Search for the `limit` nearest neighbors to the query embedding.
    /// Returns `(Uuid, similarity)` pairs sorted by descending similarity.
    fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(Uuid, f32)>, EchoError>;

    /// Remove a vector by node `Uuid`. No-op if not present.
    fn remove(&self, node_id: Uuid) -> Result<(), EchoError>;

    /// Returns `true` if the node has a vector in the index.
    fn contains(&self, node_id: Uuid) -> Result<bool, EchoError>;

    /// Returns the `u64` label for the given node `Uuid`, if present.
    fn get_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError>;

    /// Save index state to disk.
    fn save(&self) -> Result<(), EchoError>;

    /// Restore in-memory label↔uuid mappings from stored `(Uuid, label)` pairs.
    /// Called during cold start after loading the index file from disk.
    fn restore_mappings(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError>;

    /// Rebuild the entire index from scratch using provided embeddings.
    /// Returns `(Uuid, label)` mappings for all inserted vectors.
    fn rebuild_from_embeddings(
        &self,
        entries: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError>;

    /// Configured dimensionality of this index.
    fn dimensions(&self) -> usize;

    /// Returns `true` if the index contains no vectors.
    fn is_empty(&self) -> bool;

    /// Returns the number of vectors in the index.
    fn len(&self) -> usize;

    /// Returns `true` if the index file exists on disk.
    fn index_file_exists(&self) -> bool;
}
