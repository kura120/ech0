//! A-MEM dynamic linking background pass for ech0.
//!
//! When the `dynamic-linking` feature is enabled, every successful ingest triggers a
//! background linking pass that:
//!
//! 1. For each newly ingested node, runs vector search to find top-k semantically similar
//!    existing nodes.
//! 2. For each similar node above the similarity threshold, writes a new dynamic edge
//!    between the new node and the existing node.
//! 3. Increases importance scores of all linked nodes (memory reinforcement).
//!
//! The linking pass runs as a `tokio::spawn` background task — it does not block
//! `ingest_text()` return. The caller receives `IngestResult` immediately and can
//! optionally await `linking_task` or drop it.
//!
//! This module contains the pure logic for linking. The `Store` module is responsible
//! for spawning the task and wiring in the graph/vector layers.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use crate::config::DynamicLinkingConfig;
use crate::error::EchoError;
use crate::schema::{Edge, Node};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of a completed linking pass. Returned via the `JoinHandle` in `IngestResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingResult {
    /// The ingest ID that triggered this linking pass.
    pub ingest_id: Uuid,

    /// Number of new dynamic edges created by the linking pass.
    pub edges_created: usize,

    /// Number of existing nodes whose importance was boosted due to linking.
    pub nodes_boosted: usize,

    /// IDs of all newly created dynamic edges (source, target pairs).
    pub new_edges: Vec<(Uuid, Uuid)>,

    /// Any errors encountered during linking that were non-fatal.
    /// Fatal errors cause the entire linking pass to fail with `EchoError`.
    pub warnings: Vec<String>,
}

impl LinkingResult {
    /// Create an empty linking result for a given ingest.
    pub fn empty(ingest_id: Uuid) -> Self {
        Self {
            ingest_id,
            edges_created: 0,
            nodes_boosted: 0,
            new_edges: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

/// A candidate link discovered by vector similarity search.
#[derive(Debug, Clone)]
pub struct LinkCandidate {
    /// The newly ingested node.
    pub new_node_id: Uuid,

    /// The existing node that is semantically similar.
    pub existing_node_id: Uuid,

    /// Cosine similarity score between the two nodes' embeddings.
    pub similarity: f32,
}

/// A confirmed link that should be written to the graph as a dynamic edge.
#[derive(Debug, Clone)]
pub struct ConfirmedLink {
    /// The newly ingested node (edge source).
    pub new_node_id: Uuid,

    /// The existing node (edge target).
    pub existing_node_id: Uuid,

    /// Cosine similarity that triggered this link.
    pub similarity: f32,

    /// The relation label for the dynamic edge.
    pub relation: String,
}

// ---------------------------------------------------------------------------
// Candidate discovery
// ---------------------------------------------------------------------------

/// Filter vector search results into link candidates based on the configured
/// similarity threshold.
///
/// # Arguments
///
/// * `new_node_id` — the newly ingested node to find links for.
/// * `search_results` — vector ANN search results: `(existing_node_id, similarity)` pairs,
///   sorted by descending similarity.
/// * `config` — dynamic linking configuration (similarity threshold, top-k, max links).
/// * `already_linked` — set of existing node IDs that already have an edge to `new_node_id`.
///   These are excluded to avoid duplicate edges.
///
/// # Returns
///
/// Filtered and capped list of `LinkCandidate`s above the similarity threshold.
#[instrument(skip_all, fields(new_node_id = %new_node_id, candidates_in = search_results.len()))]
pub fn discover_candidates(
    new_node_id: Uuid,
    search_results: &[(Uuid, f32)],
    config: &DynamicLinkingConfig,
    already_linked: &[Uuid],
) -> Vec<LinkCandidate> {
    let candidates: Vec<LinkCandidate> = search_results
        .iter()
        .filter(|(existing_id, similarity)| {
            // Exclude self-links
            if *existing_id == new_node_id {
                return false;
            }
            // Exclude nodes that already have an edge from the new node
            if already_linked.contains(existing_id) {
                return false;
            }
            // Apply similarity threshold
            *similarity >= config.similarity_threshold
        })
        .take(config.top_k_candidates)
        .map(|(existing_id, similarity)| LinkCandidate {
            new_node_id,
            existing_node_id: *existing_id,
            similarity: *similarity,
        })
        .collect();

    debug!(
        new_node_id = %new_node_id,
        candidates_found = candidates.len(),
        "link candidates discovered"
    );

    candidates
}

/// Discover candidates for a batch of newly ingested nodes.
///
/// Aggregates candidates across all new nodes, respecting `max_links_per_ingest` as
/// a global cap on the total number of dynamic edges created in one ingest operation.
///
/// # Arguments
///
/// * `new_node_ids` — all node IDs from the current ingest batch.
/// * `search_results_per_node` — for each new node, the vector search results.
///   Keyed by new node ID, values are `(existing_node_id, similarity)` pairs.
/// * `config` — dynamic linking configuration.
/// * `existing_edges_per_node` — for each new node, the list of existing node IDs
///   that already have edges. Used to avoid duplicate links.
///
/// # Returns
///
/// Aggregated list of `LinkCandidate`s, capped at `max_links_per_ingest`.
pub fn discover_candidates_batch(
    new_node_ids: &[Uuid],
    search_results_per_node: &[(Uuid, Vec<(Uuid, f32)>)],
    config: &DynamicLinkingConfig,
    existing_edges_per_node: &[(Uuid, Vec<Uuid>)],
) -> Vec<LinkCandidate> {
    let mut all_candidates = Vec::new();

    for node_id in new_node_ids {
        // Find search results for this node
        let search_results = search_results_per_node
            .iter()
            .find(|(id, _)| id == node_id)
            .map(|(_, results)| results.as_slice())
            .unwrap_or(&[]);

        // Find existing edges for this node
        let already_linked = existing_edges_per_node
            .iter()
            .find(|(id, _)| id == node_id)
            .map(|(_, edges)| edges.as_slice())
            .unwrap_or(&[]);

        let node_candidates = discover_candidates(*node_id, search_results, config, already_linked);
        all_candidates.extend(node_candidates);

        // Check global cap
        if all_candidates.len() >= config.max_links_per_ingest {
            all_candidates.truncate(config.max_links_per_ingest);
            break;
        }
    }

    debug!(
        total_candidates = all_candidates.len(),
        max_links = config.max_links_per_ingest,
        "batch link candidate discovery complete"
    );

    all_candidates
}

// ---------------------------------------------------------------------------
// Link confirmation
// ---------------------------------------------------------------------------

/// Convert link candidates into confirmed links with relation labels.
///
/// In a full implementation, this would call the `Extractor` to determine the
/// specific relationship between each pair of nodes. For V1, we use a default
/// "dynamically_linked" relation label with the similarity score stored in
/// edge metadata.
///
/// # Arguments
///
/// * `candidates` — link candidates from `discover_candidates`.
///
/// # Returns
///
/// Confirmed links ready to be written as edges to the graph layer.
pub fn confirm_links(candidates: &[LinkCandidate]) -> Vec<ConfirmedLink> {
    // TODO: In a future version, call the Extractor to determine the semantic
    // relationship between each pair. For V1, all dynamic links use a generic
    // relation label. The similarity score is preserved in edge metadata so
    // callers can inspect link quality.

    candidates
        .iter()
        .map(|candidate| ConfirmedLink {
            new_node_id: candidate.new_node_id,
            existing_node_id: candidate.existing_node_id,
            similarity: candidate.similarity,
            relation: "dynamically_linked".to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Edge construction
// ---------------------------------------------------------------------------

/// Build `Edge` objects from confirmed links, ready to be written to the graph layer.
///
/// # Arguments
///
/// * `confirmed_links` — links that should become edges.
/// * `ingest_id` — the ingest operation that triggered this linking pass.
///
/// # Returns
///
/// A list of `Edge` objects with metadata containing the similarity score and
/// a `"dynamic": true` marker so callers can distinguish dynamic links from
/// extraction-time edges.
pub fn build_dynamic_edges(confirmed_links: &[ConfirmedLink], ingest_id: Uuid) -> Vec<Edge> {
    let now = Utc::now();

    confirmed_links
        .iter()
        .map(|link| Edge {
            source: link.new_node_id,
            target: link.existing_node_id,
            relation: link.relation.clone(),
            metadata: serde_json::json!({
                "dynamic": true,
                "similarity": link.similarity,
                "linking_ingest_id": ingest_id.to_string(),
            }),
            // Dynamic edges start with importance proportional to the similarity
            // that triggered them — stronger similarity means more important link.
            importance: link.similarity,
            created_at: now,
            ingest_id,
        })
        .collect()
}

/// Identify node IDs whose importance should be boosted because they were linked to.
///
/// Both the new node and the existing node in each link receive a boost —
/// being linked reinforces both sides of the relationship.
///
/// # Returns
///
/// Deduplicated list of node IDs to boost.
pub fn nodes_to_boost(confirmed_links: &[ConfirmedLink]) -> Vec<Uuid> {
    let mut node_ids = Vec::with_capacity(confirmed_links.len() * 2);

    for link in confirmed_links {
        node_ids.push(link.new_node_id);
        node_ids.push(link.existing_node_id);
    }

    // Deduplicate — a node that appears in multiple links should only be boosted once
    node_ids.sort();
    node_ids.dedup();

    node_ids
}

// ---------------------------------------------------------------------------
// Full linking pass (orchestration)
// ---------------------------------------------------------------------------

/// Execute the full linking pass for a completed ingest.
///
/// This is the function that `Store` calls (potentially via `tokio::spawn`) to
/// perform the background linking. It takes closures for graph and vector operations
/// so it remains decoupled from the storage layer implementations.
///
/// # Arguments
///
/// * `ingest_id` — the ingest operation to link.
/// * `new_nodes` — the newly ingested nodes.
/// * `config` — dynamic linking configuration.
/// * `vector_search` — closure that performs vector ANN search for a given node ID.
///   Returns `(existing_node_id, similarity)` pairs.
/// * `get_existing_edges` — closure that returns existing edge target IDs for a node.
/// * `write_edges` — closure that writes new dynamic edges to the graph layer.
/// * `boost_importance` — closure that boosts importance for a list of node IDs.
///
/// # Returns
///
/// `LinkingResult` summarizing what happened.
pub async fn execute_linking_pass<VSearch, GetEdges, WriteEdges, BoostFn>(
    ingest_id: Uuid,
    new_nodes: &[Node],
    config: &DynamicLinkingConfig,
    vector_search: VSearch,
    get_existing_edges: GetEdges,
    write_edges: WriteEdges,
    boost_importance: BoostFn,
) -> Result<LinkingResult, EchoError>
where
    VSearch: Fn(Uuid) -> Result<Vec<(Uuid, f32)>, EchoError>,
    GetEdges: Fn(Uuid) -> Result<Vec<Uuid>, EchoError>,
    WriteEdges: Fn(Vec<Edge>) -> Result<(), EchoError>,
    BoostFn: Fn(Vec<Uuid>) -> Result<(), EchoError>,
{
    if new_nodes.is_empty() {
        return Ok(LinkingResult::empty(ingest_id));
    }

    let mut result = LinkingResult::empty(ingest_id);

    // Step 1: For each new node, run vector search and collect candidates
    let mut search_results_per_node: Vec<(Uuid, Vec<(Uuid, f32)>)> =
        Vec::with_capacity(new_nodes.len());
    let mut existing_edges_per_node: Vec<(Uuid, Vec<Uuid>)> =
        Vec::with_capacity(new_nodes.len());

    for node in new_nodes {
        match vector_search(node.id) {
            Ok(results) => {
                search_results_per_node.push((node.id, results));
            }
            Err(error) => {
                // Non-fatal: log and skip this node. The linking pass should not
                // fail entirely because one node's vector search failed.
                warn!(
                    node_id = %node.id,
                    error = %error,
                    "vector search failed for node during linking — skipping"
                );
                result.warnings.push(format!(
                    "vector search failed for node {}: {}",
                    node.id, error
                ));
                continue;
            }
        }

        match get_existing_edges(node.id) {
            Ok(edges) => {
                existing_edges_per_node.push((node.id, edges));
            }
            Err(error) => {
                warn!(
                    node_id = %node.id,
                    error = %error,
                    "failed to read existing edges during linking — assuming none"
                );
                existing_edges_per_node.push((node.id, Vec::new()));
                result.warnings.push(format!(
                    "failed to read existing edges for node {}: {}",
                    node.id, error
                ));
            }
        }
    }

    let new_node_ids: Vec<Uuid> = new_nodes.iter().map(|n| n.id).collect();

    // Step 2: Discover candidates across all new nodes
    let candidates = discover_candidates_batch(
        &new_node_ids,
        &search_results_per_node,
        config,
        &existing_edges_per_node,
    );

    if candidates.is_empty() {
        debug!(ingest_id = %ingest_id, "no link candidates found");
        return Ok(result);
    }

    // Step 3: Confirm links (in V1 this is a simple passthrough with default relation)
    let confirmed = confirm_links(&candidates);

    // Step 4: Build and write dynamic edges
    let dynamic_edges = build_dynamic_edges(&confirmed, ingest_id);
    let edge_pairs: Vec<(Uuid, Uuid)> = dynamic_edges
        .iter()
        .map(|edge| (edge.source, edge.target))
        .collect();

    write_edges(dynamic_edges).map_err(|error| {
        EchoError::storage_failure(format!(
            "failed to write dynamic edges during linking: {error}"
        ))
    })?;

    result.edges_created = confirmed.len();
    result.new_edges = edge_pairs;

    // Step 5: Boost importance of all linked nodes
    let boost_node_ids = nodes_to_boost(&confirmed);
    result.nodes_boosted = boost_node_ids.len();

    boost_importance(boost_node_ids).map_err(|error| {
        EchoError::storage_failure(format!(
            "failed to boost importance during linking: {error}"
        ))
    })?;

    debug!(
        ingest_id = %ingest_id,
        edges_created = result.edges_created,
        nodes_boosted = result.nodes_boosted,
        "linking pass completed"
    );

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DynamicLinkingConfig {
        DynamicLinkingConfig {
            top_k_candidates: 5,
            similarity_threshold: 0.75,
            max_links_per_ingest: 10,
        }
    }

    fn make_test_node(node_id: Uuid) -> Node {
        Node {
            id: node_id,
            kind: "test".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        }
    }

    // -----------------------------------------------------------------------
    // discover_candidates
    // -----------------------------------------------------------------------

    #[test]
    fn candidates_above_threshold_are_included() {
        let config = make_config(); // threshold = 0.75
        let new_node = Uuid::new_v4();
        let existing_a = Uuid::new_v4();
        let existing_b = Uuid::new_v4();

        let search_results = vec![
            (existing_a, 0.90), // above threshold
            (existing_b, 0.80), // above threshold
        ];

        let candidates = discover_candidates(new_node, &search_results, &config, &[]);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].existing_node_id, existing_a);
        assert_eq!(candidates[1].existing_node_id, existing_b);
    }

    #[test]
    fn candidates_below_threshold_are_excluded() {
        let config = make_config(); // threshold = 0.75
        let new_node = Uuid::new_v4();
        let existing_a = Uuid::new_v4();
        let existing_b = Uuid::new_v4();

        let search_results = vec![
            (existing_a, 0.90), // above
            (existing_b, 0.50), // below
        ];

        let candidates = discover_candidates(new_node, &search_results, &config, &[]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].existing_node_id, existing_a);
    }

    #[test]
    fn self_links_are_excluded() {
        let config = make_config();
        let new_node = Uuid::new_v4();

        let search_results = vec![
            (new_node, 1.0),           // self — should be excluded
            (Uuid::new_v4(), 0.85),    // valid
        ];

        let candidates = discover_candidates(new_node, &search_results, &config, &[]);
        assert_eq!(candidates.len(), 1);
        assert_ne!(candidates[0].existing_node_id, new_node);
    }

    #[test]
    fn already_linked_nodes_are_excluded() {
        let config = make_config();
        let new_node = Uuid::new_v4();
        let existing_a = Uuid::new_v4();
        let existing_b = Uuid::new_v4();

        let search_results = vec![
            (existing_a, 0.90),
            (existing_b, 0.85),
        ];

        // existing_a is already linked
        let candidates = discover_candidates(new_node, &search_results, &config, &[existing_a]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].existing_node_id, existing_b);
    }

    #[test]
    fn candidates_capped_at_top_k() {
        let mut config = make_config();
        config.top_k_candidates = 2;
        config.similarity_threshold = 0.0; // accept everything

        let new_node = Uuid::new_v4();
        let search_results: Vec<(Uuid, f32)> = (0..10)
            .map(|i| (Uuid::new_v4(), 0.9 - i as f32 * 0.01))
            .collect();

        let candidates = discover_candidates(new_node, &search_results, &config, &[]);
        assert_eq!(candidates.len(), 2, "should be capped at top_k");
    }

    #[test]
    fn empty_search_results_produce_no_candidates() {
        let config = make_config();
        let new_node = Uuid::new_v4();

        let candidates = discover_candidates(new_node, &[], &config, &[]);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // discover_candidates_batch
    // -----------------------------------------------------------------------

    #[test]
    fn batch_discovery_respects_max_links_per_ingest() {
        let mut config = make_config();
        config.max_links_per_ingest = 3;
        config.similarity_threshold = 0.0;
        config.top_k_candidates = 10;

        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        let search_a: Vec<(Uuid, f32)> = (0..5)
            .map(|_| (Uuid::new_v4(), 0.85))
            .collect();
        let search_b: Vec<(Uuid, f32)> = (0..5)
            .map(|_| (Uuid::new_v4(), 0.80))
            .collect();

        let search_results = vec![
            (node_a, search_a),
            (node_b, search_b),
        ];

        let candidates = discover_candidates_batch(
            &[node_a, node_b],
            &search_results,
            &config,
            &[],
        );

        assert_eq!(candidates.len(), 3, "should be capped at max_links_per_ingest");
    }

    #[test]
    fn batch_discovery_with_no_nodes() {
        let config = make_config();
        let candidates = discover_candidates_batch(&[], &[], &config, &[]);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // confirm_links
    // -----------------------------------------------------------------------

    #[test]
    fn confirm_links_produces_default_relation() {
        let candidate = LinkCandidate {
            new_node_id: Uuid::new_v4(),
            existing_node_id: Uuid::new_v4(),
            similarity: 0.88,
        };

        let confirmed = confirm_links(&[candidate.clone()]);
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].relation, "dynamically_linked");
        assert_eq!(confirmed[0].new_node_id, candidate.new_node_id);
        assert_eq!(confirmed[0].existing_node_id, candidate.existing_node_id);
        assert!((confirmed[0].similarity - 0.88).abs() < f32::EPSILON);
    }

    #[test]
    fn confirm_links_empty_input() {
        let confirmed = confirm_links(&[]);
        assert!(confirmed.is_empty());
    }

    // -----------------------------------------------------------------------
    // build_dynamic_edges
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_edges_have_correct_metadata() {
        let ingest_id = Uuid::new_v4();
        let link = ConfirmedLink {
            new_node_id: Uuid::new_v4(),
            existing_node_id: Uuid::new_v4(),
            similarity: 0.91,
            relation: "dynamically_linked".to_string(),
        };

        let edges = build_dynamic_edges(&[link.clone()], ingest_id);
        assert_eq!(edges.len(), 1);

        let edge = &edges[0];
        assert_eq!(edge.source, link.new_node_id);
        assert_eq!(edge.target, link.existing_node_id);
        assert_eq!(edge.relation, "dynamically_linked");
        assert_eq!(edge.ingest_id, ingest_id);

        // Check metadata markers
        assert_eq!(edge.metadata["dynamic"], true);
        assert!((edge.metadata["similarity"].as_f64().unwrap() - 0.91).abs() < 0.001);
        assert!(edge.metadata["linking_ingest_id"].is_string());

        // Importance should be proportional to similarity
        assert!((edge.importance - 0.91).abs() < f32::EPSILON);
    }

    #[test]
    fn dynamic_edges_empty_input() {
        let edges = build_dynamic_edges(&[], Uuid::new_v4());
        assert!(edges.is_empty());
    }

    // -----------------------------------------------------------------------
    // nodes_to_boost
    // -----------------------------------------------------------------------

    #[test]
    fn nodes_to_boost_includes_both_sides() {
        let new_node = Uuid::new_v4();
        let existing_node = Uuid::new_v4();

        let link = ConfirmedLink {
            new_node_id: new_node,
            existing_node_id: existing_node,
            similarity: 0.85,
            relation: "dynamically_linked".to_string(),
        };

        let boost_ids = nodes_to_boost(&[link]);
        assert_eq!(boost_ids.len(), 2);
        assert!(boost_ids.contains(&new_node));
        assert!(boost_ids.contains(&existing_node));
    }

    #[test]
    fn nodes_to_boost_deduplicates() {
        let shared_node = Uuid::new_v4();
        let other_a = Uuid::new_v4();
        let other_b = Uuid::new_v4();

        let links = vec![
            ConfirmedLink {
                new_node_id: shared_node,
                existing_node_id: other_a,
                similarity: 0.85,
                relation: "dynamically_linked".to_string(),
            },
            ConfirmedLink {
                new_node_id: shared_node,
                existing_node_id: other_b,
                similarity: 0.80,
                relation: "dynamically_linked".to_string(),
            },
        ];

        let boost_ids = nodes_to_boost(&links);
        // shared_node, other_a, other_b — shared_node should appear only once
        assert_eq!(boost_ids.len(), 3);

        let shared_count = boost_ids.iter().filter(|id| **id == shared_node).count();
        assert_eq!(shared_count, 1, "shared_node should be deduplicated");
    }

    #[test]
    fn nodes_to_boost_empty_input() {
        let boost_ids = nodes_to_boost(&[]);
        assert!(boost_ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // LinkingResult
    // -----------------------------------------------------------------------

    #[test]
    fn empty_linking_result_has_zeroes() {
        let ingest_id = Uuid::new_v4();
        let result = LinkingResult::empty(ingest_id);

        assert_eq!(result.ingest_id, ingest_id);
        assert_eq!(result.edges_created, 0);
        assert_eq!(result.nodes_boosted, 0);
        assert!(result.new_edges.is_empty());
        assert!(result.warnings.is_empty());
    }

    // -----------------------------------------------------------------------
    // execute_linking_pass
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn linking_pass_with_no_nodes_returns_empty_result() {
        let config = make_config();
        let ingest_id = Uuid::new_v4();

        let result = execute_linking_pass(
            ingest_id,
            &[],
            &config,
            |_| Ok(Vec::new()),
            |_| Ok(Vec::new()),
            |_| Ok(()),
            |_| Ok(()),
        )
        .await
        .expect("linking pass should succeed");

        assert_eq!(result.ingest_id, ingest_id);
        assert_eq!(result.edges_created, 0);
        assert_eq!(result.nodes_boosted, 0);
    }

    #[tokio::test]
    async fn linking_pass_creates_edges_for_similar_nodes() {
        let config = make_config(); // threshold = 0.75
        let ingest_id = Uuid::new_v4();
        let new_node_id = Uuid::new_v4();
        let existing_node_id = Uuid::new_v4();

        let new_nodes = vec![make_test_node(new_node_id)];

        let written_edges = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let written_edges_clone = written_edges.clone();

        let boosted_nodes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let boosted_nodes_clone = boosted_nodes.clone();

        let result = execute_linking_pass(
            ingest_id,
            &new_nodes,
            &config,
            move |_node_id| {
                // Return a similar existing node above threshold
                Ok(vec![(existing_node_id, 0.88)])
            },
            |_node_id| {
                // No existing edges
                Ok(Vec::new())
            },
            move |edges| {
                written_edges_clone.lock().unwrap().extend(edges);
                Ok(())
            },
            move |node_ids| {
                boosted_nodes_clone.lock().unwrap().extend(node_ids);
                Ok(())
            },
        )
        .await
        .expect("linking pass should succeed");

        assert_eq!(result.edges_created, 1);
        assert_eq!(result.nodes_boosted, 2); // new + existing
        assert_eq!(result.new_edges.len(), 1);
        assert_eq!(result.new_edges[0], (new_node_id, existing_node_id));

        // Verify the edge was actually written
        let edges = written_edges.lock().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, new_node_id);
        assert_eq!(edges[0].target, existing_node_id);
        assert_eq!(edges[0].relation, "dynamically_linked");

        // Verify boost was called for both nodes
        let boosted = boosted_nodes.lock().unwrap();
        assert!(boosted.contains(&new_node_id));
        assert!(boosted.contains(&existing_node_id));
    }

    #[tokio::test]
    async fn linking_pass_skips_below_threshold() {
        let config = make_config(); // threshold = 0.75
        let ingest_id = Uuid::new_v4();
        let new_node_id = Uuid::new_v4();
        let existing_node_id = Uuid::new_v4();

        let new_nodes = vec![make_test_node(new_node_id)];

        let result = execute_linking_pass(
            ingest_id,
            &new_nodes,
            &config,
            move |_| Ok(vec![(existing_node_id, 0.50)]), // below threshold
            |_| Ok(Vec::new()),
            |_| Ok(()),
            |_| Ok(()),
        )
        .await
        .expect("linking pass should succeed");

        assert_eq!(result.edges_created, 0);
        assert_eq!(result.nodes_boosted, 0);
    }

    #[tokio::test]
    async fn linking_pass_handles_vector_search_failure_gracefully() {
        let config = make_config();
        let ingest_id = Uuid::new_v4();
        let new_nodes = vec![make_test_node(Uuid::new_v4())];

        let result = execute_linking_pass(
            ingest_id,
            &new_nodes,
            &config,
            |_| Err(EchoError::storage_failure("vector search exploded")),
            |_| Ok(Vec::new()),
            |_| Ok(()),
            |_| Ok(()),
        )
        .await
        .expect("linking pass should succeed even with search failure");

        assert_eq!(result.edges_created, 0);
        assert!(!result.warnings.is_empty(), "should contain a warning");
    }

    #[tokio::test]
    async fn linking_pass_propagates_write_failure() {
        let config = make_config();
        let ingest_id = Uuid::new_v4();
        let new_node_id = Uuid::new_v4();
        let existing_node_id = Uuid::new_v4();
        let new_nodes = vec![make_test_node(new_node_id)];

        let result = execute_linking_pass(
            ingest_id,
            &new_nodes,
            &config,
            move |_| Ok(vec![(existing_node_id, 0.90)]),
            |_| Ok(Vec::new()),
            |_| Err(EchoError::storage_failure("disk full")),
            |_| Ok(()),
        )
        .await;

        assert!(result.is_err(), "write failure should propagate");
    }
}
