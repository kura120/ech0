//! Abstraction trait for vector index backends.
//!
//! `VectorIndex` decouples ech0's storage layer from the concrete usearch implementation,
//! making the backend swappable and enabling compilation on targets where usearch is
//! unavailable.

use uuid::Uuid;

use crate::error::EchoError;

/// Abstraction over vector index backends.
///
/// Implementations must be Send + Sync. All methods that mutate state must
/// handle their own internal synchronization.
pub trait VectorIndex: Send + Sync {
    /// Add a single embedding. Returns the u64 label assigned in the index.
    fn add(&self, node_id: Uuid, embedding: &[f32]) -> Result<u64, EchoError>;

    /// Add a batch of embeddings. Validates all dimensions before inserting any.
    /// Returns (Uuid, label) mappings for all inserted vectors.
    fn add_batch(&self, embeddings: &[(Uuid, Vec<f32>)]) -> Result<Vec<(Uuid, u64)>, EchoError>;

    /// Search for the limit nearest neighbors to the query embedding.
    /// Returns (Uuid, similarity) pairs sorted by descending similarity.
    fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(Uuid, f32)>, EchoError>;

    /// Remove a vector by node Uuid. No-op if not present.
    fn remove(&self, node_id: Uuid) -> Result<(), EchoError>;

    /// Returns true if the node has a vector in the index.
    fn contains(&self, node_id: Uuid) -> Result<bool, EchoError>;

    /// Returns the u64 label for the given node Uuid, if present.
    fn get_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError>;

    /// Save index state to disk.
    fn save(&self) -> Result<(), EchoError>;

    /// Restore in-memory label↔uuid mappings from stored (Uuid, label) pairs.
    /// Called during cold start after loading the index file from disk.
    fn restore_mappings(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError>;

    /// Rebuild the entire index from scratch using provided embeddings.
    /// Returns (Uuid, label) mappings for all inserted vectors.
    fn rebuild_from_embeddings(
        &self,
        entries: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError>;

    /// Configured dimensionality of this index.
    fn dimensions(&self) -> usize;

    /// Returns true if the index contains no vectors.
    fn is_empty(&self) -> bool;

    /// Returns the number of vectors in the index.
    fn len(&self) -> usize;

    /// Returns true if the index file exists on disk.
    fn index_file_exists(&self) -> bool;
}
