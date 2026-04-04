//! hora-backed vector storage layer.
//!
//! `HoraVectorLayer` implements [`VectorIndex`] using hora's `BruteForceIndex<f32, usize>`.
//! It is the default backend when `feature = "backend-hora"` is active (pure Rust,
//! no C++ toolchain required, Windows-safe).
//!
//! # Persistence
//!
//! hora provides native binary serialization via its `SerializableIndex` trait
//! (`dump` / `load`). We use that for the index itself and write a companion
//! sidecar file (`<index_path>.meta`) containing the `label ↔ Uuid` mappings as
//! JSON. Both files are written on every `save()` call.
//!
//! On `open()` we attempt to load both files. If either is missing or corrupt
//! we start fresh (empty index, empty maps) and log a warning. The Store will
//! then trigger a cold-start rebuild from redb.
//!
//! # Similarity metric
//!
//! hora's `Metric::CosineSimilarity` implementation has a known bug: its
//! `dot_product` primitive returns `-(actual dot product)`, so
//! `sqrt(dot(v,v)) = sqrt(-|v|²) = NaN`. hora's `Neighbor::cmp` then panics
//! on `partial_cmp(NaN).unwrap()`.
//!
//! To work around this we:
//! 1. **Normalise** every embedding to unit length before handing it to hora.
//! 2. Use `Metric::Euclidean` (hora actually computes *squared* Euclidean
//!    distance — no `sqrt`, so no NaN risk).
//! 3. For unit vectors, squared Euclidean satisfies
//!    `‖a - b‖² = 2 − 2·cos(θ)`, so we recover cosine similarity via
//!    `cos(θ) = 1 − dist / 2`.
//!
//! This gives callers the same `[0.0, 1.0]` similarity range as
//! `UsearchVectorLayer`.
//!
//! # Removal
//!
//! hora's `BruteForceIndex` has no per-item removal. We use soft deletion:
//! the vector stays in the hora index; we remove it from the `label_to_uuid`
//! and `uuid_to_label` maps. Search results whose label is not in `label_to_uuid`
//! are silently skipped. Soft-deleted entries are evicted on the next
//! `rebuild_from_embeddings()` call.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use hora::core::ann_index::{ANNIndex, SerializableIndex};
use hora::core::metrics::Metric;
use hora::index::bruteforce_idx::BruteForceIndex;
use hora::index::bruteforce_params::BruteForceParams;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::error::{EchoError, ErrorContext};
use super::VectorIndex;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Normalise `embedding` to unit length.
///
/// Returns `EchoError::InvalidInput` when the vector has zero magnitude,
/// which would produce NaN after division.
fn normalise_embedding(embedding: &[f32], context: &'static str) -> Result<Vec<f32>, EchoError> {
    let norm_sq: f32 = embedding.iter().map(|x| x * x).sum();
    if norm_sq == 0.0 {
        return Err(EchoError::invalid_input(
            "embedding has zero magnitude — cannot normalise to unit vector",
        )
        .with_context(ErrorContext::new(context)));
    }
    let norm = norm_sq.sqrt();
    Ok(embedding.iter().map(|x| x / norm).collect())
}

/// Convert hora's squared Euclidean distance to cosine similarity.
///
/// For unit-normalised vectors: `‖a - b‖² = 2 − 2·cos(θ)`, so
/// `cos(θ) = 1 − dist / 2`. Clamped to `[0.0, 1.0]` to absorb float errors.
#[inline]
fn squared_euclidean_to_cosine(dist: f32) -> f32 {
    (1.0 - dist / 2.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Sidecar metadata — persisted alongside the hora index file
// ---------------------------------------------------------------------------

/// JSON sidecar written next to the hora index file.
///
/// Contains the `label ↔ Uuid` mappings that hora does not know about,
/// plus the next-label counter so we never reuse a label after reload.
#[derive(Serialize, Deserialize)]
struct HoraMeta {
    /// All live `(label, uuid_string)` pairs at the time of the last `save()`.
    mappings: Vec<(u64, String)>,
    /// The value of `next_label` at the time of the last `save()`.
    next_label: u64,
}

// ---------------------------------------------------------------------------
// HoraVectorLayer
// ---------------------------------------------------------------------------

/// hora-backed vector storage layer. Handles embedding storage and ANN search.
///
/// Wraps a `BruteForceIndex<f32, usize>` in an `RwLock` because hora's index
/// is not internally thread-safe — all mutations need exclusive access while
/// concurrent reads can proceed with shared access.
///
/// Implements [`VectorIndex`] so the `Store` can hold `Arc<dyn VectorIndex>`.
pub struct HoraVectorLayer {
    /// The hora brute-force ANN index.
    ///
    /// `BruteForceIndex` does not require a `build()` call before search —
    /// `built()` always returns `true`. We call `build(Metric::Euclidean)`
    /// once in `open()` to register the metric. See the module-level doc for
    /// why `Euclidean` is used instead of `CosineSimilarity`.
    index: RwLock<BruteForceIndex<f32, usize>>,

    /// Path to the hora binary index file on disk.
    index_path: PathBuf,

    /// Path to the JSON sidecar file that stores `label ↔ Uuid` mappings.
    meta_path: PathBuf,

    /// Configured vector dimensionality. All embeddings must match this exactly.
    dimensions: usize,

    /// Monotonically increasing counter for generating unique hora labels.
    next_label: AtomicU64,

    /// Mapping from hora `usize` label (stored as `u64`) → graph node `Uuid`.
    label_to_uuid: RwLock<HashMap<u64, Uuid>>,

    /// Reverse mapping from graph node `Uuid` → hora label.
    uuid_to_label: RwLock<HashMap<Uuid, u64>>,
}

impl std::fmt::Debug for HoraVectorLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoraVectorLayer")
            .field("dimensions", &self.dimensions)
            .field("index_path", &self.index_path)
            .finish()
    }
}

impl HoraVectorLayer {
    /// Returns the sidecar meta path for a given index path.
    fn meta_path_for(index_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.meta", index_path.display()))
    }

    /// Open or create the hora index at `path` with the given dimensionality.
    ///
    /// Attempts to load both the hora binary index and the JSON sidecar. If
    /// either is missing or fails to deserialize, starts with an empty index
    /// (the caller should trigger a cold-start rebuild from redb).
    #[instrument(skip_all, fields(path = %path.as_ref().display(), dimensions = dimensions))]
    pub fn open(path: impl AsRef<Path>, dimensions: usize) -> Result<Self, EchoError> {
        if dimensions == 0 {
            return Err(EchoError::invalid_input(
                "vector dimensions must be greater than zero",
            ));
        }

        let index_path = path.as_ref().to_path_buf();
        let meta_path = Self::meta_path_for(&index_path);

        // Build a fresh index as the fallback.
        let fresh_index = || -> BruteForceIndex<f32, usize> {
            let mut idx = BruteForceIndex::<f32, usize>::new(
                dimensions,
                &BruteForceParams::default(),
            );
            // Register the metric. BruteForceIndex::built() always returns true,
            // but build() must be called to set `self.mt` before search.
            // We use Euclidean (squared Euclidean in hora) with pre-normalised
            // unit vectors — see the module doc for the full explanation.
            // Unwrap is safe — Euclidean is a known-good metric.
            idx.build(Metric::Euclidean)
                .expect("BruteForce build with Euclidean must not fail");
            idx
        };

        let mut label_to_uuid: HashMap<u64, Uuid> = HashMap::new();
        let mut uuid_to_label: HashMap<Uuid, u64> = HashMap::new();
        let mut next_label: u64 = 0;

        // Attempt to load both the index and the sidecar together.
        if index_path.exists() && meta_path.exists() {
            let index_path_str = index_path.to_str().unwrap_or_default();
            let load_result: Result<BruteForceIndex<f32, usize>, _> =
                BruteForceIndex::<f32, usize>::load(index_path_str);

            match load_result {
                Ok(loaded_index) => {
                    // Index loaded — now load the sidecar.
                    match std::fs::read_to_string(&meta_path) {
                        Ok(meta_json) => {
                            match serde_json::from_str::<HoraMeta>(&meta_json) {
                                Ok(meta) => {
                                    for (label, uuid_str) in meta.mappings {
                                        match uuid_str.parse::<Uuid>() {
                                            Ok(uuid) => {
                                                label_to_uuid.insert(label, uuid);
                                                uuid_to_label.insert(uuid, label);
                                            }
                                            Err(error) => {
                                                warn!(
                                                    label = label,
                                                    error = %error,
                                                    "invalid UUID in hora meta — skipping entry"
                                                );
                                            }
                                        }
                                    }
                                    next_label = meta.next_label;
                                    info!(
                                        vectors = label_to_uuid.len(),
                                        next_label = next_label,
                                        "loaded hora index and sidecar from disk"
                                    );
                                    return Ok(Self {
                                        index: RwLock::new(loaded_index),
                                        index_path,
                                        meta_path,
                                        dimensions,
                                        next_label: AtomicU64::new(next_label),
                                        label_to_uuid: RwLock::new(label_to_uuid),
                                        uuid_to_label: RwLock::new(uuid_to_label),
                                    });
                                }
                                Err(error) => {
                                    warn!(
                                        error = %error,
                                        "failed to parse hora meta sidecar — starting with empty index"
                                    );
                                }
                            }
                        }
                        Err(error) => {
                            warn!(
                                error = %error,
                                "failed to read hora meta sidecar — starting with empty index"
                            );
                        }
                    }
                }
                Err(error) => {
                    warn!(
                        error = error,
                        "failed to load hora index from disk — starting with empty index"
                    );
                }
            }
        } else if index_path.exists() || meta_path.exists() {
            debug!("only one of hora index / sidecar exists — starting with empty index");
        } else {
            debug!("no existing hora index file — starting with empty index");
        }

        Ok(Self {
            index: RwLock::new(fresh_index()),
            index_path,
            meta_path,
            dimensions,
            // next_label is always 0 on fresh start; use the variable to avoid
            // an unused-assignment warning on the failure paths through open().
            next_label: AtomicU64::new(next_label),
            label_to_uuid: RwLock::new(HashMap::new()),
            uuid_to_label: RwLock::new(HashMap::new()),
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Write-lock the index and call the provided closure.
    fn with_index_write<F, T>(&self, context: &'static str, f: F) -> Result<T, EchoError>
    where
        F: FnOnce(&mut BruteForceIndex<f32, usize>) -> Result<T, EchoError>,
    {
        let mut guard = self.index.write().map_err(|_| {
            EchoError::storage_failure("hora index write lock poisoned")
                .with_context(ErrorContext::new(context))
        })?;
        f(&mut guard)
    }

    /// Read-lock the index and call the provided closure.
    fn with_index_read<F, T>(&self, context: &'static str, f: F) -> Result<T, EchoError>
    where
        F: FnOnce(&BruteForceIndex<f32, usize>) -> Result<T, EchoError>,
    {
        let guard = self.index.read().map_err(|_| {
            EchoError::storage_failure("hora index read lock poisoned")
                .with_context(ErrorContext::new(context))
        })?;
        f(&guard)
    }
}

// ---------------------------------------------------------------------------
// Public inherent methods (mirror VectorIndex signatures)
// ---------------------------------------------------------------------------

impl HoraVectorLayer {
    /// Returns the configured dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the number of live (non-soft-deleted) vectors.
    pub fn len(&self) -> usize {
        self.label_to_uuid
            .read()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    /// Returns `true` if there are no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the index file exists on disk.
    pub fn index_file_exists(&self) -> bool {
        self.index_path.exists()
    }

    /// Add an embedding for a node.
    ///
    /// The embedding is normalised to unit length before being stored so that
    /// hora's squared Euclidean distance equals `2 − 2·cos(θ)`, from which we
    /// can recover cosine similarity in [`search()`]. Returns the `u64` label
    /// assigned in the hora index.
    #[instrument(skip(self, embedding), fields(node_id = %node_id, embedding_dims = embedding.len()))]
    pub fn add(&self, node_id: Uuid, embedding: &[f32]) -> Result<u64, EchoError> {
        if embedding.len() != self.dimensions {
            return Err(EchoError::invalid_input(format!(
                "embedding has {} dimensions, expected {}",
                embedding.len(),
                self.dimensions
            ))
            .with_context(
                ErrorContext::new("hora::add")
                    .with_field("node_id", node_id.to_string())
                    .with_field("got_dims", embedding.len().to_string())
                    .with_field("expected_dims", self.dimensions.to_string()),
            ));
        }

        let unit_embedding = normalise_embedding(embedding, "hora::add")?;

        let label = self.next_label.fetch_add(1, Ordering::SeqCst);
        let hora_label = label as usize;

        self.with_index_write("hora::add", |idx| {
            idx.add(&unit_embedding, hora_label).map_err(|error| {
                EchoError::storage_failure(format!(
                    "hora failed to add vector: {error}"
                ))
                .with_context(
                    ErrorContext::new("hora::add")
                        .with_source(error)
                        .with_field("node_id", node_id.to_string())
                        .with_field("label", label.to_string()),
                )
            })
        })?;

        {
            let mut l2u = self.label_to_uuid.write().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
                    .with_context(ErrorContext::new("hora::add"))
            })?;
            l2u.insert(label, node_id);
        }
        {
            let mut u2l = self.uuid_to_label.write().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
                    .with_context(ErrorContext::new("hora::add"))
            })?;
            u2l.insert(node_id, label);
        }

        debug!(node_id = %node_id, label = label, "vector added to hora index");
        Ok(label)
    }

    /// Add a batch of embeddings. Validates all dimensions before inserting any.
    #[instrument(skip_all, fields(count = embeddings.len()))]
    pub fn add_batch(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError> {
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

    /// Search for the `limit` nearest neighbors.
    ///
    /// Returns `(Uuid, similarity)` pairs sorted by descending similarity
    /// (`1.0` = identical, `0.0` = orthogonal), matching `UsearchVectorLayer`'s
    /// output format.
    ///
    /// The query embedding is normalised to unit length before the search so
    /// that hora's squared Euclidean distance can be converted to cosine
    /// similarity via `cos(θ) = 1 − dist / 2`.
    ///
    /// Internally searches for `limit + 64` candidates to account for any
    /// soft-deleted entries that hora still holds but are no longer in the maps.
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

        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        // Search for extra candidates to absorb soft-deleted entries.
        let search_k = limit.saturating_add(64);

        let unit_query = normalise_embedding(query_embedding, "hora::search")?;

        let hora_results = self.with_index_read("hora::search", |idx| {
            Ok(idx.search_nodes(&unit_query, search_k))
        })?;

        let l2u = self.label_to_uuid.read().map_err(|_| {
            EchoError::storage_failure("label_to_uuid lock poisoned")
                .with_context(ErrorContext::new("hora::search"))
        })?;

        let mut scored: Vec<(Uuid, f32)> = Vec::with_capacity(hora_results.len());
        for (node, metric_value) in &hora_results {
            if let Some(hora_label) = node.idx() {
                let label = *hora_label as u64;
                if let Some(uuid) = l2u.get(&label) {
                    // hora Euclidean metric returns squared Euclidean distance.
                    // For unit vectors: dist = 2 - 2*cos(θ), so cos(θ) = 1 - dist/2.
                    let similarity = squared_euclidean_to_cosine(*metric_value);
                    scored.push((*uuid, similarity));
                }
                // If label is absent from the map, the entry was soft-deleted — skip it.
            }
        }

        // hora returns results closest-first (smallest squared distance first),
        // which corresponds to highest similarity first after conversion.
        // Sort explicitly to be robust against floating-point edge cases.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        debug!(results = scored.len(), "hora search completed");
        Ok(scored)
    }

    /// Remove a vector by its node `Uuid` (soft delete).
    ///
    /// hora's `BruteForceIndex` has no per-item removal. We remove the node from
    /// both maps so subsequent searches skip the old entry. The vector stays in
    /// hora's storage until the next `rebuild_from_embeddings()` call.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn remove(&self, node_id: Uuid) -> Result<(), EchoError> {
        let label = {
            let u2l = self.uuid_to_label.read().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
                    .with_context(ErrorContext::new("hora::remove"))
            })?;
            match u2l.get(&node_id) {
                Some(label) => *label,
                None => {
                    debug!(node_id = %node_id, "no vector label for node — nothing to remove");
                    return Ok(());
                }
            }
        };

        {
            let mut l2u = self.label_to_uuid.write().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
                    .with_context(ErrorContext::new("hora::remove"))
            })?;
            l2u.remove(&label);
        }
        {
            let mut u2l = self.uuid_to_label.write().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
                    .with_context(ErrorContext::new("hora::remove"))
            })?;
            u2l.remove(&node_id);
        }

        debug!(node_id = %node_id, label = label, "vector soft-deleted from hora index");
        Ok(())
    }

    /// Check if a node has a live vector in the index.
    pub fn contains(&self, node_id: Uuid) -> Result<bool, EchoError> {
        let u2l = self.uuid_to_label.read().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("hora::contains"))
        })?;
        Ok(u2l.contains_key(&node_id))
    }

    /// Get the hora label for a node `Uuid`, if present.
    pub fn get_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError> {
        let u2l = self.uuid_to_label.read().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("hora::get_label"))
        })?;
        Ok(u2l.get(&node_id).copied())
    }

    /// Persist the hora index and sidecar to disk.
    ///
    /// Uses hora's native `SerializableIndex::dump()` for the binary index and
    /// writes the `label ↔ Uuid` mappings as JSON to the companion `.meta` file.
    #[instrument(skip(self))]
    pub fn save(&self) -> Result<(), EchoError> {
        let index_path_str = self.index_path.to_str().unwrap_or_default().to_owned();

        // hora's dump() requires &mut self — acquire write lock.
        self.with_index_write("hora::save", |idx| {
            idx.dump(&index_path_str).map_err(|error| {
                EchoError::storage_failure(format!("hora dump failed: {error}"))
                    .with_context(
                        ErrorContext::new("hora::save")
                            .with_source(error)
                            .with_field("path", index_path_str.clone()),
                    )
            })
        })?;

        // Build the sidecar from the current live mappings.
        let (mappings_snapshot, next_label_snapshot) = {
            let l2u = self.label_to_uuid.read().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
                    .with_context(ErrorContext::new("hora::save"))
            })?;
            let snapshot: Vec<(u64, String)> = l2u
                .iter()
                .map(|(label, uuid)| (*label, uuid.to_string()))
                .collect();
            (snapshot, self.next_label.load(Ordering::SeqCst))
        };

        let meta = HoraMeta {
            mappings: mappings_snapshot,
            next_label: next_label_snapshot,
        };
        let meta_json = serde_json::to_string(&meta).map_err(|error| {
            EchoError::storage_failure(format!("failed to serialize hora meta: {error}"))
                .with_context(ErrorContext::new("hora::save"))
        })?;
        std::fs::write(&self.meta_path, meta_json.as_bytes()).map_err(|error| {
            EchoError::storage_failure(format!("failed to write hora meta sidecar: {error}"))
                .with_context(
                    ErrorContext::new("hora::save")
                        .with_field("meta_path", self.meta_path.display().to_string()),
                )
        })?;

        debug!(
            path = %self.index_path.display(),
            vectors = meta.next_label,
            "hora index and sidecar saved to disk"
        );
        Ok(())
    }

    /// Restore `label ↔ Uuid` mappings from provided pairs.
    ///
    /// When the hora index was loaded from disk (`open()` succeeded), the mappings
    /// are already embedded in the sidecar and restored during `open()`. In that
    /// case this method is a deliberate no-op — the sidecar data is authoritative.
    ///
    /// When starting fresh (index file absent or corrupt), `label_to_uuid` is
    /// empty and the provided `mappings` from redb are applied normally.
    #[instrument(skip_all, fields(mapping_count = mappings.len()))]
    pub fn restore_mappings(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError> {
        let already_loaded = {
            let l2u = self.label_to_uuid.read().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
                    .with_context(ErrorContext::new("hora::restore_mappings"))
            })?;
            !l2u.is_empty()
        };

        if already_loaded {
            // Mappings were restored from the on-disk sidecar during open().
            // Redb and the sidecar should be in sync — skip redundant work.
            debug!(
                mapping_count = mappings.len(),
                "hora sidecar already loaded — restore_mappings is a no-op"
            );
            return Ok(());
        }

        let mut l2u = self.label_to_uuid.write().map_err(|_| {
            EchoError::storage_failure("label_to_uuid lock poisoned")
                .with_context(ErrorContext::new("hora::restore_mappings"))
        })?;
        let mut u2l = self.uuid_to_label.write().map_err(|_| {
            EchoError::storage_failure("uuid_to_label lock poisoned")
                .with_context(ErrorContext::new("hora::restore_mappings"))
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

        self.next_label.store(max_label, Ordering::SeqCst);

        info!(
            restored = mappings.len(),
            next_label = max_label,
            "hora label mappings restored from provided pairs"
        );
        Ok(())
    }

    /// Rebuild the entire index from scratch using provided embeddings.
    ///
    /// Creates a fresh `BruteForceIndex`, clears all maps, resets `next_label`
    /// to 0, and re-adds every entry. Soft-deleted vectors are evicted.
    #[instrument(skip_all, fields(entry_count = entries.len()))]
    pub fn rebuild_from_embeddings(
        &self,
        entries: &[(Uuid, Vec<f32>)],
    ) -> Result<Vec<(Uuid, u64)>, EchoError> {
        // Replace the hora index with a brand-new empty instance.
        {
            let mut guard = self.index.write().map_err(|_| {
                EchoError::storage_failure("hora index write lock poisoned")
                    .with_context(ErrorContext::new("hora::rebuild_from_embeddings"))
            })?;
            let mut new_idx = BruteForceIndex::<f32, usize>::new(
                self.dimensions,
                &BruteForceParams::default(),
            );
            new_idx
                .build(Metric::CosineSimilarity)
                .expect("BruteForce build with CosineSimilarity must not fail");
            *guard = new_idx;
        }

        // Reset in-memory maps and label counter.
        {
            let mut l2u = self.label_to_uuid.write().map_err(|_| {
                EchoError::storage_failure("label_to_uuid lock poisoned")
            })?;
            l2u.clear();
        }
        {
            let mut u2l = self.uuid_to_label.write().map_err(|_| {
                EchoError::storage_failure("uuid_to_label lock poisoned")
            })?;
            u2l.clear();
        }
        self.next_label.store(0, Ordering::SeqCst);

        let mappings = self.add_batch(entries)?;

        info!(rebuilt = mappings.len(), "hora index rebuilt from embeddings");
        Ok(mappings)
    }
}

// ---------------------------------------------------------------------------
// VectorIndex trait implementation
// ---------------------------------------------------------------------------

impl VectorIndex for HoraVectorLayer {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_hora_layer(dimensions: usize) -> (HoraVectorLayer, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let index_path = dir.path().join("test_vectors.hora");
        let layer =
            HoraVectorLayer::open(&index_path, dimensions).expect("failed to create hora layer");
        (layer, dir)
    }

    fn make_embedding(dimensions: usize, fill_value: f32) -> Vec<f32> {
        vec![fill_value; dimensions]
    }

    #[test]
    fn open_creates_empty_index() {
        let (layer, _dir) = temp_hora_layer(128);
        assert!(layer.is_empty());
        assert_eq!(layer.len(), 0);
        assert_eq!(layer.dimensions(), 128);
    }

    #[test]
    fn zero_dimensions_is_rejected() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let index_path = dir.path().join("test.hora");
        let result = HoraVectorLayer::open(&index_path, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn add_and_search_single_vector() {
        let dims = 64;
        let (layer, _dir) = temp_hora_layer(dims);

        let node_id = Uuid::new_v4();
        let embedding = make_embedding(dims, 1.0);

        let label = layer.add(node_id, &embedding).expect("add should succeed");
        assert_eq!(label, 0, "first label should be 0");
        assert_eq!(layer.len(), 1);

        let results = layer.search(&embedding, 5).expect("search should succeed");
        assert!(!results.is_empty(), "should find at least one result");
        assert_eq!(results[0].0, node_id, "top result should be the same node");
        assert!(
            results[0].1 > 0.99,
            "self-similarity should be close to 1.0, got {}",
            results[0].1
        );
    }

    #[test]
    fn wrong_dimensions_rejected_on_add() {
        let (layer, _dir) = temp_hora_layer(64);
        let result = layer.add(Uuid::new_v4(), &make_embedding(128, 1.0));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn wrong_dimensions_rejected_on_search() {
        let (layer, _dir) = temp_hora_layer(64);
        let result = layer.search(&make_embedding(128, 1.0), 5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().code,
            crate::error::ErrorCode::InvalidInput
        ));
    }

    #[test]
    fn search_on_empty_index_returns_empty() {
        let (layer, _dir) = temp_hora_layer(64);
        let results = layer
            .search(&make_embedding(64, 1.0), 10)
            .expect("search should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn add_batch_validates_all_dimensions() {
        let (layer, _dir) = temp_hora_layer(64);
        let entries = vec![
            (Uuid::new_v4(), make_embedding(64, 1.0)),
            (Uuid::new_v4(), make_embedding(128, 1.0)), // wrong dims
        ];
        let result = layer.add_batch(&entries);
        assert!(result.is_err());
        assert!(layer.is_empty(), "no vectors should be added on batch failure");
    }

    #[test]
    fn add_batch_succeeds_with_valid_entries() {
        let dims = 64;
        let (layer, _dir) = temp_hora_layer(dims);
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
        let (layer, _dir) = temp_hora_layer(dims);
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
        let (layer, _dir) = temp_hora_layer(64);
        layer
            .remove(Uuid::new_v4())
            .expect("remove of nonexistent should be ok");
    }

    #[test]
    fn contains_and_get_label() {
        let dims = 32;
        let (layer, _dir) = temp_hora_layer(dims);
        let node_id = Uuid::new_v4();
        assert!(!layer.contains(node_id).expect("contains should succeed"));
        assert!(layer
            .get_label(node_id)
            .expect("get_label should succeed")
            .is_none());

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
        let (layer, _dir) = temp_hora_layer(dims);
        let mappings = vec![
            (Uuid::new_v4(), 0u64),
            (Uuid::new_v4(), 5u64),
            (Uuid::new_v4(), 10u64),
        ];
        layer
            .restore_mappings(&mappings)
            .expect("restore should succeed");
        // next_label should be max(10) + 1 = 11
        assert_eq!(layer.next_label.load(Ordering::SeqCst), 11);
        for (uuid, _label) in &mappings {
            assert!(layer.contains(*uuid).expect("contains should succeed"));
        }
    }

    #[test]
    fn labels_are_monotonically_increasing() {
        let dims = 16;
        let (layer, _dir) = temp_hora_layer(dims);
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
        let index_path = dir.path().join("test_vectors.hora");
        let node_id = Uuid::new_v4();

        // Create, add, save.
        {
            let layer =
                HoraVectorLayer::open(&index_path, dims).expect("failed to create hora layer");
            layer
                .add(node_id, &make_embedding(dims, 0.7))
                .expect("add should succeed");
            layer.save().expect("save should succeed");
        }

        // Reload — both index and sidecar should be loaded.
        {
            let layer =
                HoraVectorLayer::open(&index_path, dims).expect("failed to reopen hora layer");
            assert_eq!(layer.len(), 1, "reloaded layer should have 1 vector");
            // For hora, mappings are embedded in the sidecar loaded during open(),
            // so contains() returns true immediately without calling restore_mappings().
            assert!(
                layer.contains(node_id).unwrap_or(false),
                "mapping should be restored from sidecar on open"
            );
        }
    }

    #[test]
    fn multiple_vectors_search_returns_ranked() {
        let dims = 32;
        let (layer, _dir) = temp_hora_layer(dims);

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

        // Query with all 1.0s — closer to close_embedding
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
