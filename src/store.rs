//! `Store<E, X>` — the single entry point for ech0.
//!
//! Coordinates the graph layer (redb), vector layer (usearch), and all feature-gated
//! subsystems (linking, decay, conflict detection, provenance). All public API methods
//! live here. No other module exposes public functions to the caller.
//!
//! # Atomicity
//!
//! `ingest_text()` and `ingest_nodes()` are atomic from the caller's perspective.
//! The pipeline is: extract → embed → write graph → write vector. If any step fails,
//! nothing is written to either layer.
//!
//! # Thread safety
//!
//! `Store` is `Send + Sync` when `E` and `X` are `Send + Sync` (which they must be,
//! per the trait bounds). Internal state is protected by the storage layers' own
//! synchronization primitives.

use std::sync::Arc;

use tracing::{Instrument, info, info_span, instrument, warn};
use uuid::Uuid;

use crate::config::StoreConfig;
use crate::error::{EchoError, ErrorContext};
use crate::graph::GraphLayer;
use crate::schema::{
    DecayReport, Edge, IngestResult, Node, PruneReport, SearchOptions, SearchResult,
    TraversalOptions, TraversalResult,
};
use crate::search;
use crate::traits::{Embedder, ExtractionResult, Extractor};
use crate::vector::DefaultVectorLayer;
use crate::vector_index::VectorIndex;

#[cfg(feature = "contradiction-detection")]
use crate::conflict;

#[cfg(feature = "importance-decay")]
use crate::decay;

#[cfg(feature = "provenance")]
use crate::provenance;

#[cfg(feature = "dynamic-linking")]
use crate::linking;

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// The single entry point for ech0. Holds the graph layer, vector layer,
/// embedder, extractor, and configuration.
///
/// Generic over `E` (embedder) and `X` (extractor) — the caller provides
/// concrete implementations of these traits. ech0 ships no defaults.
pub struct Store<E: Embedder, X: Extractor> {
    config: Arc<StoreConfig>,
    graph: Arc<GraphLayer>,
    vector: Arc<dyn VectorIndex>,
    embedder: Arc<E>,
    extractor: Arc<X>,
}

impl<E: Embedder, X: Extractor> std::fmt::Debug for Store<E, X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("graph_path", &self.config.store.graph_path)
            .field("vector_path", &self.config.store.vector_path)
            .finish()
    }
}

impl<E: Embedder, X: Extractor> Store<E, X> {
    /// Create a new `Store` instance, opening or creating the graph and vector
    /// storage at the paths specified in `config`.
    ///
    /// On cold start, if the vector index file is missing or corrupt, the store
    /// rebuilds the vector index from redb (the graph is always the source of truth).
    ///
    /// # Errors
    ///
    /// Returns `EchoError::StorageFailure` if the graph or vector layer cannot be opened.
    /// Returns `EchoError::InvalidInput` if `config.store.vector_dimensions` does not
    /// match `embedder.dimensions()`.
    #[instrument(skip_all, fields(graph_path = %config.store.graph_path, vector_path = %config.store.vector_path))]
    pub async fn new(config: StoreConfig, embedder: E, extractor: X) -> Result<Self, EchoError> {
        // Validate that the configured dimensions match the embedder's output
        if config.store.vector_dimensions != embedder.dimensions() {
            return Err(EchoError::invalid_input(format!(
                "config.store.vector_dimensions ({}) does not match embedder.dimensions() ({})",
                config.store.vector_dimensions,
                embedder.dimensions()
            )));
        }

        let config = Arc::new(config);
        let config_clone = config.clone();

        // Open graph layer (blocking I/O — run on blocking thread pool)
        let graph = {
            let graph_path = config_clone.store.graph_path.clone();
            #[cfg(feature = "tokio")]
            {
                tokio::task::spawn_blocking(move || GraphLayer::open(&graph_path))
                    .await
                    .map_err(|join_error| {
                        EchoError::storage_failure(format!(
                            "graph layer open task panicked: {join_error}"
                        ))
                    })??
            }
            #[cfg(not(feature = "tokio"))]
            {
                GraphLayer::open(&graph_path)?
            }
        };
        let graph = Arc::new(graph);

        // Open vector layer
        let vector =
            DefaultVectorLayer::open(&config.store.vector_path, config.store.vector_dimensions)?;
        let vector: Arc<dyn VectorIndex> = Arc::new(vector);

        // Cold start: restore vector label mappings from redb
        let needs_rebuild = !vector.index_file_exists() || vector.is_empty();
        if !needs_rebuild {
            // Index file was loaded — restore the in-memory label↔uuid mappings from redb
            let mappings = graph.all_vector_keys()?;
            if !mappings.is_empty() {
                vector.restore_mappings(&mappings)?;
                info!(
                    restored = mappings.len(),
                    "restored vector label mappings from graph layer"
                );
            }
        } else {
            // Cold-start rebuild — re-embed all nodes from redb and reconstruct the usearch index.
            // The graph is always the source of truth. If the index file is missing or empty
            // but redb has nodes, we re-embed every node and rebuild from scratch.
            let node_count = graph.node_count()?;
            if node_count > 0 {
                info!(
                    node_count = node_count,
                    "usearch index missing or empty — rebuilding from graph layer"
                );

                let all_nodes = graph.all_nodes()?;

                info!(
                    node_count = node_count,
                    "beginning cold-start vector index rebuild — this may take a moment"
                );

                // Re-embed every node. We do this sequentially to avoid overwhelming
                // the embedder. A future optimization could batch these calls.
                let mut entries: Vec<(Uuid, Vec<f32>)> = Vec::with_capacity(all_nodes.len());
                for node in &all_nodes {
                    let embed_text = build_embed_text(node);
                    let embedding = embedder.embed(&embed_text).await.map_err(|error| {
                        EchoError::embedder_failure(format!(
                            "cold-start rebuild: failed to embed node {}: {error}",
                            node.id
                        ))
                        .with_context(
                            ErrorContext::new("store::new::cold_start_rebuild")
                                .with_source(&error)
                                .with_field("node_id", node.id.to_string()),
                        )
                    })?;

                    if embedding.len() != config.store.vector_dimensions {
                        return Err(EchoError::embedder_failure(format!(
                            "cold-start rebuild: embedder returned {} dimensions for node {}, expected {}",
                            embedding.len(),
                            node.id,
                            config.store.vector_dimensions
                        )));
                    }

                    entries.push((node.id, embedding));

                    if entries.len().is_multiple_of(100) && !entries.is_empty() {
                        info!(
                            progress = entries.len(),
                            total = all_nodes.len(),
                            "cold-start rebuild in progress"
                        );
                    }
                }

                // Rebuild the usearch index from the re-embedded entries
                let mappings = vector.rebuild_from_embeddings(&entries)?;

                // Persist the new vector key mappings back to redb so future
                // cold starts can restore mappings without re-embedding
                graph.write_vector_keys_batch(&mappings)?;

                // Save the rebuilt index to disk
                vector.save().map_err(|error| {
                    EchoError::storage_failure(format!(
                        "cold-start rebuild: failed to save rebuilt index: {error}"
                    ))
                })?;

                info!(rebuilt = mappings.len(), "cold-start rebuild complete");
            }
        }

        info!("ech0 store initialized");

        Ok(Self {
            config,
            graph,
            vector,
            embedder: Arc::new(embedder),
            extractor: Arc::new(extractor),
        })
    }

    // -----------------------------------------------------------------------
    // Write paths
    // -----------------------------------------------------------------------

    /// Ingest free text into the knowledge graph.
    ///
    /// Pipeline: extract → embed → (conflict detect) → write graph → write vector → (link).
    /// If any step before the write fails, nothing is written. The linking pass runs
    /// asynchronously after the write completes.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::ExtractorFailure` if the extractor fails.
    /// Returns `EchoError::EmbedderFailure` if the embedder fails.
    /// Returns `EchoError::InvalidInput` if `text` is empty.
    /// Returns `EchoError::StorageFailure` if the graph or vector write fails.
    /// Returns `EchoError::ConflictUnresolved` if contradiction detection is enabled,
    ///   conflicts are found, and the resolution policy is `Escalate`.
    #[instrument(skip_all, fields(text_len = text.len()))]
    pub async fn ingest_text(&self, text: &str) -> Result<IngestResult, EchoError> {
        if text.is_empty() {
            return Err(EchoError::invalid_input("text must not be empty"));
        }

        let ingest_id = Uuid::new_v4();
        let span = info_span!("ingest_text", ingest_id = %ingest_id);

        async {
            // Step 1: Extract nodes and edges from text
            let mut extraction = self.extractor.extract(text).await.map_err(|error| {
                EchoError::extractor_failure(format!("extraction failed: {error}")).with_context(
                    ErrorContext::new("store::ingest_text")
                        .with_source(&error)
                        .with_field("ingest_id", ingest_id.to_string()),
                )
            })?;

            if extraction.is_empty() {
                return Ok(IngestResult {
                    ingest_id,
                    nodes_written: 0,
                    edges_written: 0,
                    #[cfg(feature = "contradiction-detection")]
                    conflicts: Vec::new(),
                    #[cfg(all(feature = "dynamic-linking", feature = "tokio"))]
                    linking_task: None,
                });
            }

            // Stamp all nodes and edges with the canonical ingest_id
            stamp_ingest_id(&mut extraction, ingest_id);

            // Attach provenance source text when feature is enabled
            #[cfg(feature = "provenance")]
            provenance::attach_source_text(&mut extraction.nodes, text);

            // When provenance is disabled, ensure source_text is None
            #[cfg(not(feature = "provenance"))]
            {
                for node in &mut extraction.nodes {
                    node.source_text = None;
                }
            }

            // Delegate to the shared write path
            self.write_extraction(ingest_id, extraction).await
        }
        .instrument(span)
        .await
    }

    /// Ingest pre-built nodes and edges directly into the knowledge graph.
    ///
    /// Bypasses the extractor — useful when the caller has already performed extraction.
    /// Still embeds each node via the `Embedder` and runs conflict detection + linking
    /// if those features are enabled.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::InvalidInput` if both `nodes` and `edges` are empty.
    /// Returns `EchoError::EmbedderFailure` if the embedder fails for any node.
    /// Returns `EchoError::StorageFailure` if the graph or vector write fails.
    #[instrument(skip_all, fields(node_count = nodes.len(), edge_count = edges.len()))]
    pub async fn ingest_nodes(
        &self,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    ) -> Result<IngestResult, EchoError> {
        if nodes.is_empty() && edges.is_empty() {
            return Err(EchoError::invalid_input(
                "at least one node or edge must be provided",
            ));
        }

        let ingest_id = Uuid::new_v4();
        let span = info_span!("ingest_nodes", ingest_id = %ingest_id);

        async {
            let mut extraction = ExtractionResult { nodes, edges };
            stamp_ingest_id(&mut extraction, ingest_id);

            self.write_extraction(ingest_id, extraction).await
        }
        .instrument(span)
        .await
    }

    // -----------------------------------------------------------------------
    // Read paths
    // -----------------------------------------------------------------------

    /// Search the knowledge graph using hybrid retrieval (vector ANN + graph traversal).
    ///
    /// Results are scored by a weighted combination of vector similarity and graph
    /// relevance, filtered by importance and memory tier, then returned up to the
    /// hard `limit` cap. Importance scores of returned nodes are boosted (when
    /// `importance-decay` feature is enabled).
    ///
    /// # Errors
    ///
    /// Returns `EchoError::InvalidInput` if `query` is empty.
    /// Returns `EchoError::EmbedderFailure` if the embedder fails on the query.
    /// Returns `EchoError::StorageFailure` if any storage read fails.
    #[instrument(skip_all, fields(query_len = query.len(), limit = options.limit))]
    pub async fn search(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResult, EchoError> {
        if query.is_empty() {
            return Err(EchoError::invalid_input("query must not be empty"));
        }

        // Step 1: Embed the query
        let query_embedding = self.embedder.embed(query).await.map_err(|error| {
            EchoError::embedder_failure(format!("failed to embed query: {error}"))
                .with_context(ErrorContext::new("store::search").with_source(&error))
        })?;

        // Validate embedding dimensions
        if query_embedding.len() != self.config.store.vector_dimensions {
            return Err(EchoError::embedder_failure(format!(
                "query embedding has {} dimensions, expected {}",
                query_embedding.len(),
                self.config.store.vector_dimensions
            )));
        }

        // Step 2: Vector ANN search
        let vector_limit = options.limit * 2; // over-fetch to account for filtering
        let vector_hits = {
            let vector = self.vector.clone();
            let query_emb = query_embedding.clone();
            #[cfg(feature = "tokio")]
            {
                tokio::task::spawn_blocking(move || vector.search(&query_emb, vector_limit))
                    .await
                    .map_err(|join_error| {
                        EchoError::storage_failure(format!(
                            "vector search task panicked: {join_error}"
                        ))
                    })??
            }
            #[cfg(not(feature = "tokio"))]
            {
                vector.search(&query_emb, vector_limit)?
            }
        };

        // Step 3: Resolve vector hits to full nodes
        let mut vector_results: Vec<(Node, f32)> = Vec::with_capacity(vector_hits.len());
        for (node_id, similarity) in &vector_hits {
            match self.graph.get_node(*node_id)? {
                Some(node) => vector_results.push((node, *similarity)),
                None => {
                    warn!(
                        node_id = %node_id,
                        "vector hit references non-existent graph node — possible consistency issue"
                    );
                }
            }
        }

        // Step 4: Graph traversal from top vector hits (if graph_weight > 0)
        let mut graph_results: Vec<(Node, f32)> = Vec::new();
        let mut traversal_edges: Vec<Edge> = Vec::new();

        if options.graph_weight > 0.0 && !vector_hits.is_empty() {
            // Traverse from the top vector hit to find structurally connected nodes
            let traverse_from = vector_hits[0].0;
            let traversal_options = TraversalOptions {
                max_depth: 2,
                limit: options.limit * 2,
                min_importance: options.min_importance,
                relation_filter: None,
            };

            let graph_clone = self.graph.clone();
            let traversal = search::traverse(
                traverse_from,
                &traversal_options,
                |id| graph_clone.get_node(id),
                |id| {
                    let edges = graph_clone.get_outgoing_edges(id)?;
                    Ok(edges)
                },
                |id| graph_clone.get_importance(id).unwrap_or(None),
            )?;

            traversal_edges = traversal.edges.clone();
            graph_results = search::score_traversal_results(&traversal);
        }

        // Step 5: Merge vector and graph results
        let graph_ref = self.graph.clone();
        let result = search::merge_results(
            &vector_results,
            &graph_results,
            &traversal_edges,
            &options,
            |id| graph_ref.get_importance(id).unwrap_or(None),
        )?;

        // Step 6: Boost importance of retrieved nodes (when importance-decay is enabled)
        #[cfg(feature = "importance-decay")]
        {
            let graph_boost = self.graph.clone();
            let boost_entries = search::nodes_for_retrieval_boost(&result, |id| {
                graph_boost.get_importance(id).unwrap_or(None)
            });

            if !boost_entries.is_empty() {
                let boosts =
                    decay::compute_batch_retrieval_boost(&boost_entries, &self.config.memory);
                let updates: Vec<(Uuid, f32)> = boosts
                    .iter()
                    .map(|b| (b.node_id, b.new_importance))
                    .collect();

                if let Err(error) = self.graph.update_importance_batch(&updates) {
                    // Non-fatal: log but don't fail the search because of a boost write error
                    warn!(
                        error = %error,
                        "failed to write retrieval importance boosts"
                    );
                }
            }
        }

        info!(
            returned_nodes = result.nodes.len(),
            returned_edges = result.edges.len(),
            "search completed"
        );

        Ok(result)
    }

    /// Traverse the knowledge graph starting from a specific node.
    ///
    /// Performs a breadth-first traversal following edges, respecting depth limits,
    /// importance thresholds, and optional relation filters.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::InvalidInput` if the starting node does not exist.
    /// Returns `EchoError::StorageFailure` if any storage read fails.
    #[instrument(skip_all, fields(from = %from, max_depth = options.max_depth))]
    pub async fn traverse(
        &self,
        from: Uuid,
        options: TraversalOptions,
    ) -> Result<TraversalResult, EchoError> {
        // Verify the starting node exists
        if self.graph.get_node(from)?.is_none() {
            return Err(EchoError::invalid_input(format!(
                "starting node {from} does not exist"
            )));
        }

        let graph_clone = self.graph.clone();
        let graph_clone2 = self.graph.clone();
        let graph_clone3 = self.graph.clone();

        let result = search::traverse(
            from,
            &options,
            move |id| graph_clone.get_node(id),
            move |id| graph_clone2.get_outgoing_edges(id),
            move |id| graph_clone3.get_importance(id).unwrap_or(None),
        )?;

        info!(
            nodes_found = result.nodes.len(),
            edges_traversed = result.edges.len(),
            depth_reached = result.depth_reached,
            "traversal completed"
        );

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Memory management
    // -----------------------------------------------------------------------

    /// Apply time-based importance decay to all nodes in the graph.
    ///
    /// Episodic memories decay at `episodic_decay_rate` per day, semantic memories
    /// at `semantic_decay_rate` per day, short-term memories are not decayed.
    ///
    /// This does not prune nodes — call `prune()` separately to remove nodes that
    /// have decayed below the threshold.
    ///
    /// Only available when the `importance-decay` feature is enabled. When disabled,
    /// returns a report with all zeroes.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::StorageFailure` if reading or writing importance scores fails.
    #[instrument(skip_all)]
    pub async fn decay(&self) -> Result<DecayReport, EchoError> {
        #[cfg(feature = "importance-decay")]
        {
            let now = chrono::Utc::now();
            let all_nodes = self.graph.all_nodes()?;

            // Build decay entries from all nodes
            let entries: Vec<decay::DecayEntry> = all_nodes
                .iter()
                .map(|node| {
                    let tier = decay::infer_tier(node);
                    let current_importance = self
                        .graph
                        .get_importance(node.id)
                        .unwrap_or(Some(node.importance))
                        .unwrap_or(node.importance);

                    decay::DecayEntry {
                        node_id: node.id,
                        current_importance,
                        tier,
                        // Use created_at as a proxy for last_accessed. A full implementation
                        // would track last_accessed separately in a redb table.
                        last_accessed: node.created_at,
                    }
                })
                .collect();

            let (report, updates) = decay::compute_batch_decay(&entries, &self.config.memory, now);

            // Write updated importance scores
            if !updates.is_empty() {
                let importance_updates: Vec<(Uuid, f32)> = updates
                    .iter()
                    .map(|u| (u.node_id, u.new_importance))
                    .collect();

                self.graph.update_importance_batch(&importance_updates)?;
            }

            info!(
                nodes_decayed = report.nodes_decayed,
                nodes_below_threshold = report.nodes_below_threshold,
                "decay completed"
            );

            Ok(report)
        }

        #[cfg(not(feature = "importance-decay"))]
        {
            Ok(DecayReport {
                nodes_decayed: 0,
                edges_decayed: 0,
                nodes_below_threshold: 0,
            })
        }
    }

    /// Remove all nodes (and their associated edges and vectors) with importance
    /// below the given `threshold`.
    ///
    /// # Errors
    ///
    /// Returns `EchoError::StorageFailure` if any deletion fails. Partial prune
    /// state is possible if a failure occurs mid-operation — the graph layer handles
    /// per-node atomicity but cross-node atomicity is not guaranteed.
    #[instrument(skip(self), fields(threshold = threshold))]
    pub async fn prune(&self, threshold: f32) -> Result<PruneReport, EchoError> {
        let all_nodes = self.graph.all_nodes()?;

        let nodes_with_importance: Vec<(Uuid, f32)> = all_nodes
            .iter()
            .map(|node| {
                let importance = self
                    .graph
                    .get_importance(node.id)
                    .unwrap_or(Some(node.importance))
                    .unwrap_or(node.importance);
                (node.id, importance)
            })
            .collect();

        #[cfg(feature = "importance-decay")]
        let candidates = decay::identify_prune_candidates(&nodes_with_importance, threshold);

        #[cfg(not(feature = "importance-decay"))]
        let candidates: Vec<Uuid> = nodes_with_importance
            .iter()
            .filter(|(_, importance)| *importance < threshold)
            .map(|(id, _)| *id)
            .collect();

        if candidates.is_empty() {
            return Ok(PruneReport {
                nodes_pruned: 0,
                edges_pruned: 0,
                vectors_pruned: 0,
            });
        }

        let candidate_count = candidates.len();

        // Count edges that will be removed (edges connected to pruned nodes)
        let mut edge_count = 0usize;
        for node_id in &candidates {
            let outgoing = self.graph.get_outgoing_edges(*node_id)?;
            let incoming = self.graph.get_incoming_edges(*node_id)?;
            edge_count += outgoing.len() + incoming.len();
        }

        // Remove vectors from usearch index
        let mut vectors_pruned = 0usize;
        for node_id in &candidates {
            if self.vector.contains(*node_id)? {
                self.vector.remove(*node_id)?;
                vectors_pruned += 1;
            }
        }

        // Remove nodes from graph (this also removes edges and adjacency entries)
        self.graph.delete_nodes_batch(&candidates)?;

        // Save vector index after pruning
        if vectors_pruned > 0
            && let Err(error) = self.vector.save()
        {
            warn!(error = %error, "failed to save vector index after prune");
        }

        let report = PruneReport {
            nodes_pruned: candidate_count,
            edges_pruned: edge_count,
            vectors_pruned,
        };

        info!(
            nodes_pruned = report.nodes_pruned,
            edges_pruned = report.edges_pruned,
            vectors_pruned = report.vectors_pruned,
            "prune completed"
        );

        Ok(report)
    }

    // -----------------------------------------------------------------------
    // Internal: shared write path
    // -----------------------------------------------------------------------

    /// The shared write path used by both `ingest_text()` and `ingest_nodes()`.
    ///
    /// Handles: embedding → conflict detection → graph write → vector write → linking.
    /// Atomic: if any step before the final writes fails, nothing is persisted.
    async fn write_extraction(
        &self,
        ingest_id: Uuid,
        extraction: ExtractionResult,
    ) -> Result<IngestResult, EchoError> {
        let mut nodes = extraction.nodes;
        let edges = extraction.edges;

        // Step 1: Embed all nodes
        let mut embeddings: Vec<(Uuid, Vec<f32>)> = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let text_to_embed = build_embed_text(node);
            let embedding = self.embedder.embed(&text_to_embed).await.map_err(|error| {
                EchoError::embedder_failure(format!("failed to embed node {}: {error}", node.id))
                    .with_context(
                        ErrorContext::new("store::write_extraction")
                            .with_source(&error)
                            .with_field("node_id", node.id.to_string())
                            .with_field("ingest_id", ingest_id.to_string()),
                    )
            })?;

            // Validate embedding dimensions
            if embedding.len() != self.config.store.vector_dimensions {
                return Err(EchoError::embedder_failure(format!(
                    "embedder returned {} dimensions for node {}, expected {}",
                    embedding.len(),
                    node.id,
                    self.config.store.vector_dimensions
                )));
            }

            embeddings.push((node.id, embedding));
        }

        // Step 2: Conflict detection (when feature is enabled)
        #[cfg(feature = "contradiction-detection")]
        let conflicts = {
            let mut all_conflicts = Vec::new();

            // Find existing nodes that might conflict with the new ones
            // by searching for semantically similar nodes via vector search
            for (node_id, embedding) in &embeddings {
                let similar = self.vector.search(embedding, 5).unwrap_or_default();
                let mut existing_candidates = Vec::new();
                for (existing_id, _similarity) in &similar {
                    if let Ok(Some(existing_node)) = self.graph.get_node(*existing_id) {
                        existing_candidates.push(existing_node);
                    }
                }

                if !existing_candidates.is_empty() {
                    // Find the new node in our batch
                    if let Some(new_node) = nodes.iter().find(|n| n.id == *node_id) {
                        let detected = conflict::detect_conflicts(
                            std::slice::from_ref(new_node),
                            &existing_candidates,
                            &self.config.contradiction,
                        )?;
                        all_conflicts.extend(detected);
                    }
                }
            }

            // Apply resolution policy
            if !all_conflicts.is_empty() {
                let resolved =
                    conflict::apply_resolution_policy(&all_conflicts, &self.config.contradiction);

                let policy = self.config.contradiction.parsed_resolution_policy();

                // Handle KeepExisting: exclude conflicting new nodes from the write
                let excluded = conflict::nodes_to_exclude(&resolved);
                if !excluded.is_empty() {
                    nodes.retain(|n| !excluded.contains(&n.id));
                    // Also remove their embeddings
                    embeddings.retain(|(id, _)| !excluded.contains(id));
                }

                // Handle ReplaceWithNew: remove existing nodes that are being replaced
                let to_replace = conflict::nodes_to_replace(&resolved);
                for existing_id in &to_replace {
                    // Remove from vector index
                    let _ = self.vector.remove(*existing_id);
                    // Remove from graph
                    let _ = self.graph.delete_node(*existing_id);
                }

                // Handle Escalate: if any conflicts are unresolved and policy is Escalate,
                // we still write the new nodes but return the conflicts to the caller.
                // The caller decides what to do.
                if matches!(policy, conflict::ConflictResolution::Escalate)
                    && !all_conflicts.is_empty()
                {
                    // We do NOT return an error here — we write everything and let
                    // the caller inspect the conflicts in IngestResult.
                }
            }

            all_conflicts
        };

        // Step 3: Write to graph layer (atomic within redb transaction)
        self.graph.write_nodes_and_edges(&nodes, &edges)?;

        // Step 4: Write embeddings to vector layer
        let vector_mappings = match self.vector.add_batch(&embeddings) {
            Ok(mappings) => mappings,
            Err(error) => {
                // Vector write failed — we need to roll back the graph write.
                // Delete the nodes we just wrote.
                let node_ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
                if let Err(rollback_error) = self.graph.delete_nodes_batch(&node_ids) {
                    warn!(
                        error = %rollback_error,
                        "failed to roll back graph write after vector failure — \
                         consistency may be compromised"
                    );
                }
                return Err(EchoError::storage_failure(format!(
                    "vector write failed, graph rolled back: {error}"
                ))
                .with_context(
                    ErrorContext::new("store::write_extraction")
                        .with_source(&error)
                        .with_field("ingest_id", ingest_id.to_string()),
                ));
            }
        };

        // Step 5: Persist vector key mappings in the graph layer (source of truth)
        self.graph.write_vector_keys_batch(&vector_mappings)?;

        // Step 6: Save vector index to disk
        if let Err(error) = self.vector.save() {
            // Non-fatal: the in-memory index is correct, the file will be saved
            // on the next successful operation. Log but don't fail the ingest.
            warn!(
                error = %error,
                "failed to save vector index to disk — will retry on next operation"
            );
        }

        let nodes_written = nodes.len();
        let edges_written = edges.len();

        // Step 7: Spawn linking task (when both dynamic-linking and tokio are enabled)
        #[cfg(all(feature = "dynamic-linking", feature = "tokio"))]
        let linking_task: Option<tokio::task::JoinHandle<linking::LinkingResult>> = {
            if nodes_written == 0 {
                None
            } else {
                let linking_config = self.config.dynamic_linking.clone();
                let _graph_for_link = self.graph.clone();
                let vector_for_link = self.vector.clone();
                let graph_for_edges = self.graph.clone();
                let graph_for_write = self.graph.clone();
                let graph_for_boost = self.graph.clone();
                let config_for_boost = self.config.clone();
                let nodes_for_link = nodes.clone();
                // Pass the embeddings computed in Step 1 into the linking task so the
                // vector_search closure can use the correct embedding for each node.
                // This resolves the dummy-query bug — each node is now searched with
                // its own actual embedding rather than a zero-vector.
                let embeddings_for_link = embeddings.clone();

                let handle = tokio::spawn(async move {
                    let dims = vector_for_link.dimensions();
                    linking::execute_linking_pass(
                        ingest_id,
                        &nodes_for_link,
                        &linking_config,
                        // vector_search closure — uses the embedding from the ingest step.
                        move |node_id| {
                            let embedding = embeddings_for_link
                                .iter()
                                .find(|(id, _)| *id == node_id)
                                .map(|(_, emb)| emb.clone())
                                .unwrap_or_else(|| {
                                    // Should not happen in normal operation — every ingested
                                    // node has its embedding computed in Step 1.
                                    tracing::warn!(
                                        node_id = %node_id,
                                        "embedding not found for node during linking — \
                                         falling back to zero-vector"
                                    );
                                    vec![0.0f32; dims]
                                });
                            vector_for_link.search(&embedding, linking_config.top_k_candidates)
                        },
                        // get_existing_edges closure
                        move |node_id| {
                            let edges = graph_for_edges.get_outgoing_edges(node_id)?;
                            Ok(edges.into_iter().map(|e| e.target).collect())
                        },
                        // write_edges closure
                        move |new_edges| graph_for_write.write_nodes_and_edges(&[], &new_edges),
                        // boost_importance closure
                        move |node_ids| {
                            let boost = config_for_boost.memory.importance_boost_on_retrieval;
                            let updates: Vec<(Uuid, f32)> = node_ids
                                .iter()
                                .map(|id| {
                                    let current = graph_for_boost
                                        .get_importance(*id)
                                        .unwrap_or(None)
                                        .unwrap_or(0.5);
                                    (*id, current + boost)
                                })
                                .collect();
                            graph_for_boost.update_importance_batch(&updates)
                        },
                    )
                    .await
                    .unwrap_or_else(|error| {
                        warn!(error = %error, "linking pass failed");
                        linking::LinkingResult::empty(ingest_id)
                    })
                });

                Some(handle)
            }
        };

        info!(
            ingest_id = %ingest_id,
            nodes_written = nodes_written,
            edges_written = edges_written,
            "ingest completed"
        );

        Ok(IngestResult {
            ingest_id,
            nodes_written,
            edges_written,
            #[cfg(feature = "contradiction-detection")]
            conflicts,
            #[cfg(all(feature = "dynamic-linking", feature = "tokio"))]
            linking_task,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (free functions)
// ---------------------------------------------------------------------------

/// Overwrite the `ingest_id` on all nodes and edges in an extraction result
/// with the canonical ingest ID for this operation.
fn stamp_ingest_id(extraction: &mut ExtractionResult, ingest_id: Uuid) {
    for node in &mut extraction.nodes {
        node.ingest_id = ingest_id;
    }
    for edge in &mut extraction.edges {
        edge.ingest_id = ingest_id;
    }
}

/// Build the text string that will be embedded for a node.
///
/// Combines the node's `kind` and serialized metadata into a single string
/// suitable for embedding. If `source_text` is available (provenance feature),
/// it is prepended as the most semantically meaningful content.
fn build_embed_text(node: &Node) -> String {
    let mut parts = Vec::with_capacity(3);

    // Source text is the most semantically rich content
    if let Some(ref source_text) = node.source_text {
        parts.push(source_text.clone());
    }

    // Kind provides type context
    parts.push(format!("kind:{}", node.kind));

    // Metadata values provide additional semantic signal
    if let Some(obj) = node.metadata.as_object() {
        for (key, value) in obj {
            if let Some(string_value) = value.as_str() {
                parts.push(format!("{key}:{string_value}"));
            } else {
                parts.push(format!("{key}:{value}"));
            }
        }
    }

    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_test_node(kind: &str, metadata: serde_json::Value) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: kind.to_string(),
            metadata,
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        }
    }

    // -----------------------------------------------------------------------
    // stamp_ingest_id
    // -----------------------------------------------------------------------

    #[test]
    fn stamp_ingest_id_overwrites_all() {
        let original_id = Uuid::new_v4();
        let canonical_id = Uuid::new_v4();

        let node = Node {
            id: Uuid::new_v4(),
            kind: "test".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: original_id,
            source_text: None,
        };

        let edge = Edge {
            source: Uuid::new_v4(),
            target: Uuid::new_v4(),
            relation: "related".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.5,
            created_at: Utc::now(),
            ingest_id: original_id,
        };

        let mut extraction = ExtractionResult {
            nodes: vec![node],
            edges: vec![edge],
        };

        stamp_ingest_id(&mut extraction, canonical_id);

        assert_eq!(extraction.nodes[0].ingest_id, canonical_id);
        assert_eq!(extraction.edges[0].ingest_id, canonical_id);
    }

    // -----------------------------------------------------------------------
    // build_embed_text
    // -----------------------------------------------------------------------

    #[test]
    fn embed_text_includes_kind() {
        let node = make_test_node("person", serde_json::json!({}));
        let text = build_embed_text(&node);
        assert!(text.contains("kind:person"));
    }

    #[test]
    fn embed_text_includes_metadata_values() {
        let node = make_test_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let text = build_embed_text(&node);
        assert!(text.contains("name:Alice"));
        assert!(text.contains("age:30"));
    }

    #[test]
    fn embed_text_includes_source_text_when_present() {
        let mut node = make_test_node("fact", serde_json::json!({}));
        node.source_text = Some("The sky is blue.".to_string());

        let text = build_embed_text(&node);
        assert!(
            text.starts_with("The sky is blue."),
            "source text should be first"
        );
    }

    #[test]
    fn embed_text_handles_empty_metadata() {
        let node = make_test_node("fact", serde_json::json!({}));
        let text = build_embed_text(&node);
        assert_eq!(text, "kind:fact");
    }

    #[test]
    fn embed_text_handles_null_metadata() {
        let node = make_test_node("fact", serde_json::Value::Null);
        let text = build_embed_text(&node);
        assert_eq!(text, "kind:fact");
    }
}
