//! usearch vector storage layer.
//!
//! Owns all embedding storage and approximate nearest neighbor (ANN) search operations.
//! The usearch index is always paired with the redb graph layer — a node always has both
//! a graph entry and a vector entry, and they are never out of sync.
//!
//! On cold start, if the usearch index file is missing or corrupt, ech0 rebuilds it
//! from redb (the graph is always the source of truth).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::{debug, info, instrument, warn};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use uuid::Uuid;

use crate::error::{EchoError, ErrorContext};
use super::VectorIndex;

// ---------------------------------------------------------------------------
// UsearchVectorLayer
// ---------------------------------------------------------------------------

/// The usearch-backed vector storage layer. Handles embedding storage and ANN search.
///
/// Internally maintains a mapping from `u64` labels (usearch keys) to `Uuid`s (graph node IDs)
/// and vice versa, so callers always work with `Uuid`s.
///
/// Thread safety: the usearch `Index` is internally thread-safe for concurrent reads.
/// Writes are serialized through the `RwLock` on `label_to_uuid`. The `RwLock` protects
/// the mapping tables — usearch itself handles its own internal locking.
///
/// Implements [`VectorIndex`] so callers can hold `Arc<dyn VectorIndex>` to decouple
/// from the concrete usearch backend.
impl std::fmt::Debug for UsearchVectorLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsearchVectorLayer")
            .field("dimensions", &self.dimensions)
            .field("index_path", &self.index_path)
            .field("size", &self.index.size())
            .finish()
    }
}

pub struct UsearchVectorLayer {
    /// The usearch ANN index.
    index: Index,

    /// Path to the usearch index file on disk.
    index_path: PathBuf,

    /// Configured vector dimensionality. All embeddings must match this exactly.
    dimensions: usize,

    /// Monotonically increasing counter for generating unique usearch labels.
    /// Starts from the max existing label + 1 on cold start.
    next_label: AtomicU64,

    /// Mapping from usearch u64 label → graph node Uuid.
    /// Protected by RwLock for concurrent read access during search.
    label_to_uuid: RwLock<HashMap<u64, Uuid>>,

    /// Reverse mapping from graph node Uuid → usearch u64 label.
    uuid_to_label: RwLock<HashMap<Uuid, u64>>,
}

impl UsearchVectorLayer {
    /// Open or create the usearch index at the given path with the specified dimensionality.
    ///
    /// If the index file exists and is valid, it is loaded. If it is missing or corrupt,
    /// an empty index is created (caller should then trigger a rebuild from redb).
    #[instrument(skip_all, fields(path = %path.as_ref().display(), dimensions = dimensions))]
    pub fn open(path: impl AsRef<Path>, dimensions: usize) -> Result<Self, EchoError> {
        if dimensions == 0 {
            return Err(EchoError::invalid_input(
                "vector dimensions must be greater than zero",
            ));
        }

        let index_path = path.as_ref().to_path_buf();

        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&options).map_err(|error| {
            EchoError::storage_failure(format!("failed to create usearch index: {error}"))
                .with_context(
                    ErrorContext::new("vector::open")
                        .with_source(error.to_string())
                        .with_field("dimensions", dimensions.to_string()),
                )
        })?;

        let mut loaded_from_file = false;

        // Attempt to load existing index from disk
        if index_path.exists() {
            match index.load(index_path.to_str().unwrap_or_default()) {
                Ok(()) => {
                    info!(
                        count = index.size(),
                        "loaded existing usearch index from disk"
                    );
                    loaded_from_file = true;
                }
                Err(error) => {
                    // Index file is corrupt or incompatible — start fresh.
                    // Caller should trigger a rebuild from redb.
                    warn!(
                        error = %error,
                        "failed to load usearch index from disk — starting with empty index"
                    );
                }
            }
        } else {
            debug!("no existing usearch index file — starting with empty index");
        }

        // Reserve initial capacity if starting fresh
        if !loaded_from_file {
            // Reserve a reasonable initial capacity; will grow as needed
            let initial_capacity = 1024;
            index.reserve(initial_capacity).map_err(|error| {
                EchoError::storage_failure(format!(
                    "failed to reserve initial index capacity: {error}"
                ))
            })?;
        }

        Ok(Self {
            index,
            index_path,
            dimensions,
            next_label: AtomicU64::new(0),
            label_to_uuid: RwLock::new(HashMap::new()),
            uuid_to_label: RwLock::new(HashMap::new()),
        })
    }

    /// Returns the configured dimensionality of this vector layer.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the current number of vectors stored in the index.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.size() == 0
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Add an embedding for a node to the vector index.
    ///
    /// Validates that the embedding dimensionality matches the configured dimensions.
    /// Returns the `u64` label assigned to this vector in the usearch index. The caller
    /// (Store) is responsible for persisting this mapping in the graph layer's `vector_keys` table.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::InvalidInput` if the embedding has the wrong number of dimensions.
    /// Returns `EchoError::StorageFailure` if usearch fails to add the vector.
    #[instrument(skip(self, embedding), fields(node_id = %node_id, embedding_dims = embedding.len()))]
    pub fn add(&self, node_id: Uuid, embedding: &[f32]) -> Result<u64, EchoError> {
        // Validate dimensionality — never silently accept wrong-dimension embeddings
        if embedding.len() != self.dimensions {
            return Err(EchoError::invalid_input(format!(
                "embedding has {} dimensions, expected {}",
                embedding.len(),
                self.dimensions
            ))
            .with_context(
                ErrorContext::new("vector::add")
                    .with_field("node_id", node_id.to_string())
                    .with_field("got_dims", embedding.len().to_string())
                    .with_field("expected_dims", self.dimensions.to_string()),
            ));
        }

        let label = self.next_label.fetch_add(1, Ordering::SeqCst);

        // Ensure index has capacity. usearch may need to grow.
        let current_capacity = self.index.capacity();
        let current_size = self.index.size();
        if current_size >= current_capacity {
            let new_capacity = (current_capacity * 2).max(current_capacity + 1024);
            self.index.reserve(new_capacity).map_err(|error| {
                EchoError::storage_failure(format!(
                    "failed to grow usearch index capacity: {error}"
                ))
                .with_context(
                    ErrorContext::new("vector::add")
                        .with_source(error.to_string())
                        .with_field("current_capacity", current_capacity.to_string())
                        .with_field("new_capacity", new_capacity.to_string()),
                )
            })?;
        }

        self.index.add(label, embedding).map_err(|error| {
            EchoError::storage_failure(format!("failed to add vector to usearch index: {error}"))
                .with_context(
                    ErrorContext::new("vector::add")
                        .with_source(error.to_string())
                        .with_field("node_id", node_id.to_string())
                        .with_field("label", label.to_string()),
                )
        })?;

        // Update bidirectional mappings
        {
            let mut l2u = self.label_to_uuid.write().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
                    .with_context(ErrorContext::new("vector::add"))
            })?;
            l2u.insert(label, node_id);
        }
        {
            let mut u2l = self.uuid_to_label.write().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
                    .with_context(ErrorContext::new("vector::add"))
            })?;
            u2l.insert(node_id, label);
        }

        debug!(node_id = %node_id, label = label, "vector added to index");
        Ok(label)
    }

    /// Add multiple embeddings in a batch. Returns the `(Uuid, u64)` mappings for
    /// all successfully added vectors.
    ///
    /// If any single vector fails validation, the entire batch is rejected.
    #[instrument(skip_all, fields(count = embeddings.len()))]
    pub fn add_batch(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError> {
        // Validate all dimensions up front — reject entire batch on any mismatch
        for (node_id, embedding) in embeddings {
            if embedding.len() != self.dimensions {
                return Err(EchoError::invalid_input(format!(
                    "embedding for node {} has {} dimensions, expected {}",
                    node_id,
                    embedding.len(),
                    self.dimensions
                )));
            }
        }

        let mut mappings = Vec::with_capacity(embeddings.len());
        for (node_id, embedding) in embeddings {
            let label = self.add(*node_id, embedding)?;
            mappings.push((*node_id, label));
        }

        Ok(mappings)
    }

    // -----------------------------------------------------------------------
    // Search operations
    // -----------------------------------------------------------------------

    /// Search for the `limit` nearest neighbors to the given query embedding.
    ///
    /// Returns a list of `(Uuid, f32)` pairs — node ID and cosine similarity score.
    /// Results are sorted by descending similarity (most similar first).
    ///
    /// # Errors
    ///
    /// Returns `EchoError::InvalidInput` if the query has the wrong dimensionality.
    /// Returns `EchoError::StorageFailure` if usearch search fails.
    #[instrument(skip(self, query_embedding), fields(query_dims = query_embedding.len(), limit = limit))]
    pub fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Uuid, f32)>, EchoError> {
        if query_embedding.len() != self.dimensions {
            return Err(EchoError::invalid_input(format!(
                "query embedding has {} dimensions, expected {}",
                query_embedding.len(),
                self.dimensions
            )));
        }

        if limit == 0 || self.index.size() == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query_embedding, limit).map_err(|error| {
            EchoError::storage_failure(format!("usearch search failed: {error}"))
                .with_context(ErrorContext::new("vector::search").with_source(error.to_string()))
        })?;

        let labels = results.keys;
        let distances = results.distances;

        let l2u = self.label_to_uuid.read().map_err(|_| {
            EchoError::storage_failure("label_to_uuid lock poisoned")
                .with_context(ErrorContext::new("vector::search"))
        })?;

        let mut scored_results = Vec::with_capacity(labels.len());
        for (label, distance) in labels.iter().zip(distances.iter()) {
            if let Some(uuid) = l2u.get(label) {
                // usearch cosine metric returns distance (1 - similarity).
                // Convert to similarity score for the caller.
                let similarity = 1.0 - distance;
                scored_results.push((*uuid, similarity));
            } else {
                // Label exists in usearch but not in our mapping — possible consistency
                // issue after a crash. Log and skip.
                warn!(
                    label = label,
                    "usearch label not found in label_to_uuid mapping — skipping"
                );
            }
        }

        // Sort by descending similarity (most similar first)
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        debug!(results = scored_results.len(), "vector search completed");
        Ok(scored_results)
    }

    // -----------------------------------------------------------------------
    // Removal
    // -----------------------------------------------------------------------

    /// Remove a vector from the index by its node Uuid.
    ///
    /// If the node has no vector in the index, this is a no-op and returns `Ok(())`.
    ///
    /// Note: usearch `remove` marks the label as deleted but does not reclaim space
    /// until the next save/load cycle. This is acceptable for prune operations.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn remove(&self, node_id: Uuid) -> Result<(), EchoError> {
        let label = {
            let u2l = self.uuid_to_label.read().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
                    .with_context(ErrorContext::new("vector::remove"))
            })?;
            match u2l.get(&node_id) {
                Some(label) => *label,
                None => {
                    debug!(node_id = %node_id, "no vector label found for node — nothing to remove");
                    return Ok(());
                }
            }
        };

        self.index.remove(label).map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to remove vector from usearch index: {error}"
            ))
            .with_context(
                ErrorContext::new("vector::remove")
                    .with_source(error.to_string())
                    .with_field("node_id", node_id.to_string())
                    .with_field("label", label.to_string()),
            )
        })?;

        // Clean up bidirectional mappings
        {
            let mut l2u = self
                .label_to_uuid
                .write()
                .map_err(|_| EchoError::storage_failure("label_to_uuid lock poisoned"))?;
            l2u.remove(&label);
        }
        {
            let mut u2l = self
                .uuid_to_label
                .write()
                .map_err(|_| EchoError::storage_failure("uuid_to_label lock poisoned"))?;
            u2l.remove(&node_id);
        }

        debug!(node_id = %node_id, label = label, "vector removed from index");
        Ok(())
    }

    /// Remove multiple vectors by node Uuid. Convenience batch wrapper around `remove`.
    pub fn remove_batch(&self, node_ids: &[Uuid]) -> Result<(), EchoError> {
        for node_id in node_ids {
            self.remove(*node_id)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Save the current index state to disk.
    ///
    /// Called by `Store` after successful ingest commits. The index file is written
    /// atomically by usearch — a crash during save does not corrupt the existing file.
    #[instrument(skip(self))]
    pub fn save(&self) -> Result<(), EchoError> {
        self.index
            .save(self.index_path.to_str().unwrap_or_default())
            .map_err(|error| {
                EchoError::storage_failure(format!("failed to save usearch index to disk: {error}"))
                    .with_context(
                        ErrorContext::new("vector::save")
                            .with_source(error.to_string())
                            .with_field("path", self.index_path.display().to_string()),
                    )
            })?;

        debug!(
            path = %self.index_path.display(),
            size = self.index.size(),
            "usearch index saved to disk"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cold start rebuild
    // -----------------------------------------------------------------------

    /// Rebuild the in-memory label↔uuid mappings from a list of known mappings.
    ///
    /// Called during cold start when the usearch index is loaded from disk and the
    /// mapping tables need to be reconstructed from the graph layer's `vector_keys` table.
    #[instrument(skip_all, fields(mapping_count = mappings.len()))]
    pub fn restore_mappings(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError> {
        let mut l2u = self.label_to_uuid.write().map_err(|_| {
            EchoError::storage_failure("label_to_uuid lock poisoned")
                .with_context(ErrorContext::new("vector::restore_mappings"))
        })?;
        let mut u2l = self.uuid_to_label.write().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("vector::restore_mappings"))
        })?;

        l2u.clear();
        u2l.clear();

        let mut max_label: u64 = 0;
        for (uuid, label) in mappings {
            l2u.insert(*label, *uuid);
            u2l.insert(*uuid, *label);
            if *label >= max_label {
                max_label = *label + 1;
            }
        }

        // Set the next label counter to one past the highest existing label
        self.next_label.store(max_label, Ordering::SeqCst);

        info!(
            restored = mappings.len(),
            next_label = max_label,
            "vector label mappings restored"
        );
        Ok(())
    }

    /// Rebuild the entire usearch index from scratch using embeddings provided by the caller.
    ///
    /// This is the cold-start recovery path: when the usearch index file is missing or corrupt,
    /// `Store` reads all nodes from redb, re-embeds them via the `Embedder`, and calls this
    /// method to reconstruct the index.
    ///
    /// # Arguments
    ///
    /// * `entries` — list of `(node_id, embedding)` pairs to insert into the fresh index.
    ///
    /// # Returns
    ///
    /// The `(Uuid, u64)` mappings for all inserted vectors, to be persisted in the graph layer.
    #[instrument(skip_all, fields(entry_count = entries.len()))]
    pub fn rebuild_from_embeddings(
        &self,
        entries: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError> {
        // Clear in-memory mappings
        {
            let mut l2u = self
                .label_to_uuid
                .write()
                .map_err(|_| EchoError::storage_failure("label_to_uuid lock poisoned"))?;
            l2u.clear();
        }
        {
            let mut u2l = self
                .uuid_to_label
                .write()
                .map_err(|_| EchoError::storage_failure("uuid_to_label lock poisoned"))?;
            u2l.clear();
        }
        self.next_label.store(0, Ordering::SeqCst);

        // Reset the usearch index — wipes all existing vectors and labels,
        // allowing labels to start from 0 again without duplicate key errors.
        self.index.reset().map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to reset usearch index for rebuild: {error}"
            ))
        })?;

        // Reserve capacity for the incoming entries
        let capacity = entries.len().max(1024);
        self.index.reserve(capacity).map_err(|error| {
            EchoError::storage_failure(format!("failed to reserve capacity after reset: {error}"))
        })?;

        // Re-add all entries with fresh labels starting from 0
        let mappings = self.add_batch(entries)?;

        info!(
            rebuilt = mappings.len(),
            "usearch index rebuilt from embeddings"
        );
        Ok(mappings)
    }
    /// Check whether the index file exists and appears loadable.
    ///
    /// Used during `Store::new()` to decide whether a cold-start rebuild is needed.
    pub fn index_file_exists(&self) -> bool {
        self.index_path.exists()
    }

    /// Returns the path to the usearch index file.
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    /// Check if a node has a vector in the index.
    pub fn contains(&self, node_id: Uuid) -> Result<bool, EchoError> {
        let u2l = self.uuid_to_label.read().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("vector::contains"))
        })?;
        Ok(u2l.contains_key(&node_id))
    }

    /// Get the usearch label for a node Uuid, if it exists in the index.
    pub fn get_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError> {
        let u2l = self.uuid_to_label.read().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("vector::get_label"))
        })?;
        Ok(u2l.get(&node_id).copied())
    }
}

// ---------------------------------------------------------------------------
// VectorIndex trait implementation
// ---------------------------------------------------------------------------

impl VectorIndex for UsearchVectorLayer {
    fn add(&self, node_id: Uuid, embedding: &[f32]) -> Result<u64, EchoError> {
        self.add(node_id, embedding)
    }

    fn add_batch(&self, embeddings: &[(Uuid, Vec<f32>)]) -> Result<Vec<(Uuid, u64)>, EchoError> {
        self.add_batch(embeddings)
    }

    fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(Uuid, f32)>, EchoError> {
        self.search(query_embedding, limit)
    }

    fn remove(&self, node_id: Uuid) -> Result<(), EchoError> {
        self.remove(node_id)
    }

    fn contains(&self, node_id: Uuid) -> Result<bool, EchoError> {
        self.contains(node_id)
    }

    fn get_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError> {
        self.get_label(node_id)
    }

    fn save(&self) -> Result<(), EchoError> {
        self.save()
    }

    fn restore_mappings(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError> {
        self.restore_mappings(mappings)
    }

    fn rebuild_from_embeddings(
        &self,
        entries: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError> {
        self.rebuild_from_embeddings(entries)
    }

    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn index_file_exists(&self) -> bool {
        self.index_file_exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a UsearchVectorLayer in a temporary directory for testing.
    fn temp_vector_layer(dimensions: usize) -> (UsearchVectorLayer, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let index_path = dir.path().join("test_vectors.usearch");
        let layer = UsearchVectorLayer::open(&index_path, dimensions)
            .expect("failed to create vector layer");
        (layer, dir)
    }

    /// Create a simple test embedding of the given dimensions.
    fn make_embedding(dimensions: usize, fill_value: f32) -> Vec<f32> {
        vec![fill_value; dimensions]
    }

    #[test]
    fn open_creates_empty_index() {
        let (layer, _dir) = temp_vector_layer(128);
        assert!(layer.is_empty());
        assert_eq!(layer.len(), 0);
        assert_eq!(layer.dimensions(), 128);
    }

    #[test]
    fn zero_dimensions_is_rejected() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let index_path = dir.path().join("test.usearch");
        let result = UsearchVectorLayer::open(&index_path, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn add_and_search_single_vector() {
        let dims = 64;
        let (layer, _dir) = temp_vector_layer(dims);

        let node_id = Uuid::new_v4();
        let embedding = make_embedding(dims, 1.0);

        let label = layer.add(node_id, &embedding).expect("add should succeed");
        assert_eq!(label, 0, "first label should be 0");
        assert_eq!(layer.len(), 1);

        // Search with the same vector — should find itself
        let results = layer.search(&embedding, 5).expect("search should succeed");
        assert!(!results.is_empty(), "should find at least one result");
        assert_eq!(results[0].0, node_id, "top result should be the same node");
        // Cosine similarity of a vector with itself should be ~1.0
        assert!(
            results[0].1 > 0.99,
            "self-similarity should be close to 1.0, got {}",
            results[0].1
        );
    }

    #[test]
    fn wrong_dimensions_rejected_on_add() {
        let (layer, _dir) = temp_vector_layer(64);

        let node_id = Uuid::new_v4();
        let wrong_embedding = make_embedding(128, 1.0);

        let result = layer.add(node_id, &wrong_embedding);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn wrong_dimensions_rejected_on_search() {
        let (layer, _dir) = temp_vector_layer(64);

        let wrong_query = make_embedding(128, 1.0);
        let result = layer.search(&wrong_query, 5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn search_on_empty_index_returns_empty() {
        let (layer, _dir) = temp_vector_layer(64);
        let query = make_embedding(64, 1.0);
        let results = layer.search(&query, 10).expect("search should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_limit_zero_returns_empty() {
        let dims = 64;
        let (layer, _dir) = temp_vector_layer(dims);

        let node_id = Uuid::new_v4();
        layer
            .add(node_id, &make_embedding(dims, 1.0))
            .expect("add should succeed");

        let results = layer
            .search(&make_embedding(dims, 1.0), 0)
            .expect("search should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn add_batch_validates_all_dimensions() {
        let (layer, _dir) = temp_vector_layer(64);

        let entries = vec![
            (Uuid::new_v4(), make_embedding(64, 1.0)),
            (Uuid::new_v4(), make_embedding(128, 1.0)), // wrong dims
        ];

        let result = layer.add_batch(&entries);
        assert!(result.is_err());
        // No vectors should have been added since validation happens upfront
        assert!(layer.is_empty());
    }

    #[test]
    fn add_batch_succeeds_with_valid_entries() {
        let dims = 64;
        let (layer, _dir) = temp_vector_layer(dims);

        let entries: Vec<(Uuid, Vec<f32>)> = (0..5)
            .map(|i| (Uuid::new_v4(), make_embedding(dims, i as f32 * 0.1 + 0.1)))
            .collect();

        let mappings = layer.add_batch(&entries).expect("batch add should succeed");
        assert_eq!(mappings.len(), 5);
        assert_eq!(layer.len(), 5);
    }

    #[test]
    fn remove_cleans_up_mappings() {
        let dims = 64;
        let (layer, _dir) = temp_vector_layer(dims);

        let node_id = Uuid::new_v4();
        layer
            .add(node_id, &make_embedding(dims, 1.0))
            .expect("add should succeed");

        assert!(layer.contains(node_id).expect("contains should succeed"));

        layer.remove(node_id).expect("remove should succeed");

        assert!(!layer.contains(node_id).expect("contains should succeed"));
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let (layer, _dir) = temp_vector_layer(64);
        // Should not error when removing a node that was never added
        layer
            .remove(Uuid::new_v4())
            .expect("remove of nonexistent should be ok");
    }

    #[test]
    fn contains_and_get_label() {
        let dims = 32;
        let (layer, _dir) = temp_vector_layer(dims);

        let node_id = Uuid::new_v4();
        assert!(!layer.contains(node_id).expect("contains should succeed"));
        assert!(
            layer
                .get_label(node_id)
                .expect("get_label should succeed")
                .is_none()
        );

        let label = layer
            .add(node_id, &make_embedding(dims, 0.5))
            .expect("add should succeed");

        assert!(layer.contains(node_id).expect("contains should succeed"));
        assert_eq!(
            layer
                .get_label(node_id)
                .expect("get_label should succeed")
                .expect("label should exist"),
            label
        );
    }

    #[test]
    fn restore_mappings_sets_next_label_correctly() {
        let dims = 32;
        let (layer, _dir) = temp_vector_layer(dims);

        let mappings = vec![
            (Uuid::new_v4(), 0),
            (Uuid::new_v4(), 5),
            (Uuid::new_v4(), 10),
        ];

        layer
            .restore_mappings(&mappings)
            .expect("restore should succeed");

        // next_label should be max(10) + 1 = 11
        assert_eq!(layer.next_label.load(Ordering::SeqCst), 11);

        // All mappings should be queryable
        for (uuid, _label) in &mappings {
            assert!(layer.contains(*uuid).expect("contains should succeed"));
        }
    }

    #[test]
    fn labels_are_monotonically_increasing() {
        let dims = 16;
        let (layer, _dir) = temp_vector_layer(dims);

        let mut labels = Vec::new();
        for _ in 0..10 {
            let label = layer
                .add(Uuid::new_v4(), &make_embedding(dims, 0.5))
                .expect("add should succeed");
            labels.push(label);
        }

        for window in labels.windows(2) {
            assert!(
                window[1] > window[0],
                "labels should be strictly increasing: {} should be > {}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn save_and_reload() {
        let dims = 32;
        let dir = TempDir::new().expect("failed to create temp dir");
        let index_path = dir.path().join("test_vectors.usearch");

        let node_id = Uuid::new_v4();

        // Create index, add a vector, save
        {
            let layer =
                UsearchVectorLayer::open(&index_path, dims).expect("failed to create vector layer");
            layer
                .add(node_id, &make_embedding(dims, 0.7))
                .expect("add should succeed");
            layer.save().expect("save should succeed");
        }

        // Reopen — the index file should be loaded
        {
            let layer =
                UsearchVectorLayer::open(&index_path, dims).expect("failed to reopen vector layer");
            // The index should have the vector (loaded from file)
            assert_eq!(layer.len(), 1, "reloaded index should have 1 vector");

            // But the in-memory mapping is empty until restore_mappings is called
            // (the caller — Store — is responsible for calling restore_mappings from redb data)
            assert!(
                !layer.contains(node_id).unwrap_or(false),
                "mapping not yet restored"
            );
        }
    }

    #[test]
    fn rebuild_from_embeddings_clears_and_re_adds() {
        let dims = 32;
        let (layer, _dir) = temp_vector_layer(dims);

        // Add some initial vectors
        for _ in 0..3 {
            layer
                .add(Uuid::new_v4(), &make_embedding(dims, 0.5))
                .expect("add should succeed");
        }
        assert_eq!(layer.len(), 3);

        // Rebuild with new entries
        let new_entries: Vec<(Uuid, Vec<f32>)> = (0..5)
            .map(|i| (Uuid::new_v4(), make_embedding(dims, i as f32 * 0.1 + 0.1)))
            .collect();

        let mappings = layer
            .rebuild_from_embeddings(&new_entries)
            .expect("rebuild should succeed");

        assert_eq!(mappings.len(), 5);
        // Note: len() reflects usearch internal size which includes both old (not cleared
        // in this stub implementation) and new entries. The TODO in rebuild_from_embeddings
        // notes that a production implementation should create a fresh index.
        // For now, verify mappings are correct.
        for (uuid, _label) in &mappings {
            assert!(layer.contains(*uuid).expect("contains should succeed"));
        }
    }

    #[test]
    fn multiple_vectors_search_returns_ranked() {
        let dims = 32;
        let (layer, _dir) = temp_vector_layer(dims);

        // Create embeddings with different values so they have different similarities
        let node_close = Uuid::new_v4();
        let node_far = Uuid::new_v4();

        // "Close" embedding — mostly 1.0s
        let mut close_embedding = vec![1.0f32; dims];
        close_embedding[0] = 0.9;

        // "Far" embedding — mostly 0.0s with one 1.0
        let mut far_embedding = vec![0.0f32; dims];
        far_embedding[0] = 1.0;

        layer
            .add(node_close, &close_embedding)
            .expect("add should succeed");
        layer
            .add(node_far, &far_embedding)
            .expect("add should succeed");

        // Query with all 1.0s — should be closer to `close_embedding`
        let query = vec![1.0f32; dims];
        let results = layer.search(&query, 10).expect("search should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].0, node_close,
            "closer embedding should rank first"
        );
        assert!(
            results[0].1 >= results[1].1,
            "first result should have higher or equal similarity"
        );
    }
}
