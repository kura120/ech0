//! Hybrid retrieval for ech0.
//!
//! Merges results from two retrieval paths:
//! - **Vector path**: ANN similarity search via usearch, finds semantically related memories.
//! - **Graph path**: relationship traversal via redb, finds structurally connected memories.
//!
//! Results are scored by a weighted combination of vector similarity and graph relevance,
//! filtered by importance threshold and memory tier, then returned up to the hard `limit` cap.
//!
//! This module contains the pure merging and scoring logic. The `Store` module is responsible
//! for calling into the graph and vector layers and passing results here for combination.

use std::collections::HashMap;

use tracing::{debug, instrument};
use uuid::Uuid;

use crate::EchoError;

use crate::schema::{
    Edge, MemoryTier, Node, RetrievalSource, RetrievalStep, ScoredEdge, ScoredNode, SearchOptions,
    SearchResult, TraversalOptions, TraversalResult,
};

// ---------------------------------------------------------------------------
// Intermediate types for merging
// ---------------------------------------------------------------------------

/// An intermediate scored entry before final ranking. Holds scores from each
/// retrieval path separately so they can be combined with configurable weights.
#[derive(Debug, Clone)]
struct MergeEntry {
    node: Node,
    /// Score from vector ANN search (0.0 if not found via vector path).
    vector_score: f32,
    /// Score from graph traversal (0.0 if not found via graph path).
    graph_score: f32,
}

// ---------------------------------------------------------------------------
// Hybrid merge
// ---------------------------------------------------------------------------

/// Merge vector search results and graph traversal results into a single ranked list.
///
/// This is the core hybrid retrieval algorithm. It:
/// 1. Combines results from both paths, deduplicating by node ID.
/// 2. Computes a weighted combined score for each node.
/// 3. Filters by `min_importance` threshold.
/// 4. Filters by memory tier (if specified).
/// 5. Sorts by combined score descending.
/// 6. Truncates to the hard `limit` cap.
///
/// # Arguments
///
/// * `vector_results` — `(Node, similarity_score)` pairs from vector ANN search.
/// * `graph_results` — `(Node, graph_relevance_score)` pairs from graph traversal.
/// * `edges` — edges discovered during graph traversal, included in the result for context.
/// * `options` — search options controlling weights, limits, filters.
/// * `importance_lookup` — closure that returns the current importance score for a node ID.
///   The importance table may have been updated by decay since the node was last written,
///   so we always read the latest value.
///
/// # Returns
///
/// A `SearchResult` with scored nodes, context edges, and retrieval path explanation.
#[instrument(skip_all, fields(
    vector_count = vector_results.len(),
    graph_count = graph_results.len(),
    limit = options.limit
))]
pub fn merge_results<F>(
    vector_results: &[(Node, f32)],
    graph_results: &[(Node, f32)],
    edges: &[Edge],
    options: &SearchOptions,
    importance_lookup: F,
) -> Result<SearchResult, EchoError>
where
    F: Fn(Uuid) -> Option<f32>,
{
    // Step 1: Build a merged map keyed by node ID
    let mut entries: HashMap<Uuid, MergeEntry> = HashMap::new();

    for (node, similarity) in vector_results {
        entries
            .entry(node.id)
            .and_modify(|entry| {
                entry.vector_score = *similarity;
            })
            .or_insert_with(|| MergeEntry {
                node: node.clone(),
                vector_score: *similarity,
                graph_score: 0.0,
            });
    }

    for (node, relevance) in graph_results {
        entries
            .entry(node.id)
            .and_modify(|entry| {
                entry.graph_score = *relevance;
            })
            .or_insert_with(|| MergeEntry {
                node: node.clone(),
                vector_score: 0.0,
                graph_score: *relevance,
            });
    }

    // Step 2: Compute combined scores and build scored nodes
    let mut scored_nodes: Vec<(ScoredNode, RetrievalStep)> = Vec::with_capacity(entries.len());

    for (node_id, entry) in &entries {
        // Read latest importance from the importance table
        let current_importance = importance_lookup(*node_id).unwrap_or(entry.node.importance);

        // Filter by min_importance threshold — nodes below this are excluded
        if current_importance < options.min_importance {
            continue;
        }

        // Filter by memory tier if specified
        if !options.tiers.is_empty() {
            let node_tier = infer_tier_for_search(&entry.node);
            if !options.tiers.contains(&node_tier) {
                continue;
            }
        }

        let combined_score = compute_combined_score(
            entry.vector_score,
            entry.graph_score,
            options.vector_weight,
            options.graph_weight,
        );

        let source = match (entry.vector_score > 0.0, entry.graph_score > 0.0) {
            (true, true) => RetrievalSource::Both,
            (true, false) => RetrievalSource::Vector,
            (false, true) => RetrievalSource::Graph,
            (false, false) => RetrievalSource::Vector, // should not happen, but safe default
        };

        let scored_node = ScoredNode {
            node: entry.node.clone(),
            score: combined_score,
            source,
        };

        let step = RetrievalStep {
            node_id: *node_id,
            source,
            vector_score: entry.vector_score,
            graph_score: entry.graph_score,
            combined_score,
        };

        scored_nodes.push((scored_node, step));
    }

    // Step 3: Sort by combined score descending
    scored_nodes.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 4: Apply hard limit cap — limit is a maximum, not a target
    scored_nodes.truncate(options.limit);

    // Step 5: Build result
    let result_node_ids: std::collections::HashSet<Uuid> =
        scored_nodes.iter().map(|(sn, _)| sn.node.id).collect();

    // Filter edges to only those connecting returned nodes
    let scored_edges: Vec<ScoredEdge> = edges
        .iter()
        .filter(|edge| {
            result_node_ids.contains(&edge.source) || result_node_ids.contains(&edge.target)
        })
        .map(|edge| {
            // Edge score is derived from the scores of its connected nodes
            let source_score = scored_nodes
                .iter()
                .find(|(sn, _)| sn.node.id == edge.source)
                .map(|(sn, _)| sn.score)
                .unwrap_or(0.0);
            let target_score = scored_nodes
                .iter()
                .find(|(sn, _)| sn.node.id == edge.target)
                .map(|(sn, _)| sn.score)
                .unwrap_or(0.0);

            ScoredEdge {
                edge: edge.clone(),
                score: (source_score + target_score) / 2.0,
            }
        })
        .collect();

    let (nodes, retrieval_path): (Vec<ScoredNode>, Vec<RetrievalStep>) =
        scored_nodes.into_iter().unzip();

    debug!(
        returned_nodes = nodes.len(),
        returned_edges = scored_edges.len(),
        "hybrid search merge completed"
    );

    Ok(SearchResult {
        nodes,
        edges: scored_edges,
        retrieval_path,
    })
}

// ---------------------------------------------------------------------------
// Graph traversal
// ---------------------------------------------------------------------------

/// Execute a breadth-first graph traversal from a starting node.
///
/// This is the pure traversal logic. The caller provides closures for reading
/// nodes and edges from the graph layer so this function stays decoupled from
/// storage details.
///
/// # Arguments
///
/// * `start_id` — the node to start traversal from.
/// * `options` — traversal configuration (depth, limit, importance filter, relation filter).
/// * `get_node` — closure to read a node by ID.
/// * `get_outgoing_edges` — closure to read outgoing edges for a node.
/// * `importance_lookup` — closure to read the latest importance score for a node.
///
/// # Returns
///
/// A `TraversalResult` with discovered nodes, traversed edges, and the depth reached.
#[instrument(skip_all, fields(start_id = %start_id, max_depth = options.max_depth, limit = options.limit))]
pub fn traverse<GetNode, GetEdges, ImportanceFn>(
    start_id: Uuid,
    options: &TraversalOptions,
    get_node: GetNode,
    get_outgoing_edges: GetEdges,
    importance_lookup: ImportanceFn,
) -> Result<TraversalResult, EchoError>
where
    GetNode: Fn(Uuid) -> Result<Option<Node>, EchoError>,
    GetEdges: Fn(Uuid) -> Result<Vec<Edge>, EchoError>,
    ImportanceFn: Fn(Uuid) -> Option<f32>,
{
    let mut visited: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let mut result_nodes: Vec<Node> = Vec::new();
    let mut result_edges: Vec<Edge> = Vec::new();
    let mut depth_reached: usize = 0;

    // BFS queue: (node_id, current_depth)
    let mut queue: std::collections::VecDeque<(Uuid, usize)> = std::collections::VecDeque::new();

    queue.push_back((start_id, 0));
    visited.insert(start_id);

    while let Some((current_id, current_depth)) = queue.pop_front() {
        // Stop if we've reached the limit
        if result_nodes.len() >= options.limit {
            break;
        }

        // Read the node
        let node = match get_node(current_id)? {
            Some(node) => node,
            None => continue, // node deleted between traversal steps — skip
        };

        // Check importance threshold
        let importance = importance_lookup(current_id).unwrap_or(node.importance);
        if importance < options.min_importance && current_id != start_id {
            // Always include the start node regardless of importance
            continue;
        }

        depth_reached = depth_reached.max(current_depth);
        result_nodes.push(node);

        // Don't explore deeper if we've hit max depth
        if current_depth >= options.max_depth {
            continue;
        }

        // Get outgoing edges and explore neighbors
        let outgoing_edges = get_outgoing_edges(current_id)?;

        for edge in outgoing_edges {
            // Apply relation filter if configured
            if let Some(ref filter) = options.relation_filter {
                if !filter.contains(&edge.relation) {
                    continue;
                }
            }

            let target_id = edge.target;

            if !visited.contains(&target_id) {
                visited.insert(target_id);
                result_edges.push(edge);
                queue.push_back((target_id, current_depth + 1));
            }
        }
    }

    debug!(
        nodes_found = result_nodes.len(),
        edges_traversed = result_edges.len(),
        depth_reached = depth_reached,
        "graph traversal completed"
    );

    Ok(TraversalResult {
        nodes: result_nodes,
        edges: result_edges,
        depth_reached,
    })
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

/// Compute the weighted combined score from vector and graph scores.
///
/// The formula is:
///   `combined = (vector_score * vector_weight + graph_score * graph_weight) / total_weight`
///
/// If both weights are zero (degenerate case), returns the average of both scores.
fn compute_combined_score(
    vector_score: f32,
    graph_score: f32,
    vector_weight: f32,
    graph_weight: f32,
) -> f32 {
    let total_weight = vector_weight + graph_weight;

    if total_weight <= f32::EPSILON {
        // Degenerate case: both weights are zero. Return simple average.
        return (vector_score + graph_score) / 2.0;
    }

    (vector_score * vector_weight + graph_score * graph_weight) / total_weight
}

/// Infer memory tier for search filtering purposes.
///
/// Uses the same heuristic as `decay::infer_tier` but is duplicated here to avoid
/// a hard dependency on the `importance-decay` feature. If that feature is enabled,
/// callers should prefer `decay::infer_tier` for consistency.
fn infer_tier_for_search(node: &Node) -> MemoryTier {
    // Check explicit tier in metadata first
    if let Some(tier_value) = node.metadata.get("tier") {
        if let Some(tier_str) = tier_value.as_str() {
            match tier_str {
                "short_term" => return MemoryTier::ShortTerm,
                "episodic" => return MemoryTier::Episodic,
                "semantic" => return MemoryTier::Semantic,
                _ => {} // unrecognized — fall through
            }
        }
    }

    // Fall back to kind-based heuristic
    match node.kind.as_str() {
        "event" | "episode" => MemoryTier::Episodic,
        "working" | "short_term" | "session" => MemoryTier::ShortTerm,
        _ => MemoryTier::Semantic,
    }
}

/// Convert graph traversal results into `(Node, f32)` pairs with graph relevance scores.
///
/// Graph relevance is computed as:
///   `score = 1.0 / (1.0 + depth)`
///
/// Nodes closer to the query origin are scored higher. The starting node gets score 1.0,
/// direct neighbors get 0.5, two hops away get 0.33, etc.
///
/// # Arguments
///
/// * `traversal` — the raw traversal result from `traverse()`.
///
/// # Returns
///
/// `(Node, graph_relevance_score)` pairs suitable for merging with vector results.
pub fn score_traversal_results(traversal: &TraversalResult) -> Vec<(Node, f32)> {
    if traversal.nodes.is_empty() {
        return Vec::new();
    }

    // Build a depth map from edges. The first node in the traversal result
    // is the start node at depth 0. Use BFS reconstruction.
    let mut depth_map: HashMap<Uuid, usize> = HashMap::new();

    if let Some(start_node) = traversal.nodes.first() {
        depth_map.insert(start_node.id, 0);
    }

    // Reconstruct depths from edges. Edges are in BFS order, so we can compute
    // target depth as source depth + 1.
    for edge in &traversal.edges {
        if let Some(source_depth) = depth_map.get(&edge.source) {
            let target_depth = source_depth + 1;
            depth_map.entry(edge.target).or_insert(target_depth);
        }
    }

    traversal
        .nodes
        .iter()
        .map(|node| {
            let depth = depth_map.get(&node.id).copied().unwrap_or(0);
            let score = 1.0 / (1.0 + depth as f32);
            (node.clone(), score)
        })
        .collect()
}

/// Boost importance scores for all nodes in a search result.
///
/// Returns `(node_id, current_importance)` pairs for all nodes in the result,
/// suitable for passing to `decay::compute_batch_retrieval_boost`.
///
/// # Arguments
///
/// * `result` — the search result whose nodes should receive a retrieval boost.
/// * `importance_lookup` — closure to read the latest importance score for a node.
pub fn nodes_for_retrieval_boost<F>(result: &SearchResult, importance_lookup: F) -> Vec<(Uuid, f32)>
where
    F: Fn(Uuid) -> Option<f32>,
{
    result
        .nodes
        .iter()
        .filter_map(|scored_node| {
            let node_id = scored_node.node.id;
            let importance = importance_lookup(node_id).unwrap_or(scored_node.node.importance);
            Some((node_id, importance))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_node(kind: &str) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: kind.to_string(),
            metadata: serde_json::json!({}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        }
    }

    fn make_node_with_importance(importance: f32) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: "fact".to_string(),
            metadata: serde_json::json!({}),
            importance,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        }
    }

    fn make_node_with_tier_metadata(tier: &str) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: "fact".to_string(),
            metadata: serde_json::json!({"tier": tier}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        }
    }

    fn make_edge(source: Uuid, target: Uuid, relation: &str) -> Edge {
        Edge {
            source,
            target,
            relation: relation.to_string(),
            metadata: serde_json::json!({}),
            importance: 0.5,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
        }
    }

    fn default_options() -> SearchOptions {
        SearchOptions::default()
    }

    fn no_importance_lookup(_id: Uuid) -> Option<f32> {
        None
    }

    // -----------------------------------------------------------------------
    // compute_combined_score
    // -----------------------------------------------------------------------

    #[test]
    fn combined_score_balanced_weights() {
        let score = compute_combined_score(0.8, 0.6, 0.5, 0.5);
        // (0.8 * 0.5 + 0.6 * 0.5) / 1.0 = 0.4 + 0.3 = 0.7
        assert!((score - 0.7).abs() < 0.001);
    }

    #[test]
    fn combined_score_vector_only() {
        let score = compute_combined_score(0.8, 0.0, 1.0, 0.0);
        // (0.8 * 1.0 + 0.0) / 1.0 = 0.8
        assert!((score - 0.8).abs() < 0.001);
    }

    #[test]
    fn combined_score_graph_only() {
        let score = compute_combined_score(0.0, 0.6, 0.0, 1.0);
        // (0.0 + 0.6 * 1.0) / 1.0 = 0.6
        assert!((score - 0.6).abs() < 0.001);
    }

    #[test]
    fn combined_score_zero_weights_returns_average() {
        let score = compute_combined_score(0.8, 0.4, 0.0, 0.0);
        assert!((score - 0.6).abs() < 0.001);
    }

    #[test]
    fn combined_score_asymmetric_weights() {
        let score = compute_combined_score(0.8, 0.4, 0.7, 0.3);
        // (0.8 * 0.7 + 0.4 * 0.3) / 1.0 = 0.56 + 0.12 = 0.68
        assert!((score - 0.68).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // merge_results
    // -----------------------------------------------------------------------

    #[test]
    fn merge_deduplicates_by_node_id() {
        let node = make_node("fact");
        let vector_results = vec![(node.clone(), 0.9)];
        let graph_results = vec![(node.clone(), 0.7)];
        let options = default_options();

        let result = merge_results(
            &vector_results,
            &graph_results,
            &[],
            &options,
            no_importance_lookup,
        )
        .expect("merge should succeed");

        assert_eq!(result.nodes.len(), 1, "duplicate nodes should be merged");
        assert_eq!(
            result.nodes[0].source,
            RetrievalSource::Both,
            "node found by both paths"
        );
    }

    #[test]
    fn merge_respects_limit() {
        let mut options = default_options();
        options.limit = 2;

        let nodes: Vec<(Node, f32)> = (0..10)
            .map(|i| (make_node("fact"), 0.9 - i as f32 * 0.05))
            .collect();

        let result = merge_results(&nodes, &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert_eq!(result.nodes.len(), 2, "should respect hard limit cap");
    }

    #[test]
    fn merge_filters_below_min_importance() {
        let mut options = default_options();
        options.min_importance = 0.5;

        let high_importance = make_node_with_importance(0.8);
        let low_importance = make_node_with_importance(0.3);

        let vector_results = vec![(high_importance, 0.9), (low_importance, 0.85)];

        let result = merge_results(&vector_results, &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert_eq!(
            result.nodes.len(),
            1,
            "low importance node should be filtered"
        );
        assert!(
            result.nodes[0].node.importance >= 0.5,
            "returned node should be above threshold"
        );
    }

    #[test]
    fn merge_uses_importance_lookup_over_node_field() {
        let mut options = default_options();
        options.min_importance = 0.5;

        // Node has high importance in its struct field
        let node = make_node_with_importance(0.8);
        let node_id = node.id;

        let vector_results = vec![(node, 0.9)];

        // But the importance lookup returns a low value (decay happened)
        let result = merge_results(&vector_results, &[], &[], &options, |id| {
            if id == node_id {
                Some(0.2) // below threshold
            } else {
                None
            }
        })
        .expect("merge should succeed");

        assert_eq!(
            result.nodes.len(),
            0,
            "node should be filtered by importance lookup"
        );
    }

    #[test]
    fn merge_filters_by_memory_tier() {
        let mut options = default_options();
        options.tiers = vec![MemoryTier::Episodic];

        let episodic_node = make_node("event");
        let semantic_node = make_node("fact");

        let vector_results = vec![(episodic_node, 0.9), (semantic_node, 0.85)];

        let result = merge_results(&vector_results, &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert_eq!(result.nodes.len(), 1, "only episodic nodes should pass");
    }

    #[test]
    fn merge_empty_tiers_filter_means_search_all() {
        let options = default_options(); // tiers is empty

        let episodic_node = make_node("event");
        let semantic_node = make_node("fact");

        let vector_results = vec![(episodic_node, 0.9), (semantic_node, 0.85)];

        let result = merge_results(&vector_results, &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert_eq!(result.nodes.len(), 2, "empty tiers should include all");
    }

    #[test]
    fn merge_sorted_by_combined_score_descending() {
        let options = default_options();

        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");

        let vector_results = vec![
            (node_a.clone(), 0.5), // lowest
            (node_b.clone(), 0.9), // highest
            (node_c.clone(), 0.7), // middle
        ];

        let result = merge_results(&vector_results, &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert_eq!(result.nodes.len(), 3);
        assert_eq!(result.nodes[0].node.id, node_b.id, "highest score first");
        assert_eq!(result.nodes[1].node.id, node_c.id, "middle score second");
        assert_eq!(result.nodes[2].node.id, node_a.id, "lowest score last");
    }

    #[test]
    fn merge_includes_relevant_edges() {
        let options = default_options();

        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let _node_c = make_node("fact");

        let edge_ab = make_edge(node_a.id, node_b.id, "related_to");
        let edge_unrelated = make_edge(Uuid::new_v4(), Uuid::new_v4(), "unrelated");

        let vector_results = vec![(node_a.clone(), 0.9), (node_b.clone(), 0.8)];

        let result = merge_results(
            &vector_results,
            &[],
            &[edge_ab, edge_unrelated],
            &options,
            no_importance_lookup,
        )
        .expect("merge should succeed");

        assert_eq!(
            result.edges.len(),
            1,
            "only edges connecting returned nodes should be included"
        );
        assert_eq!(result.edges[0].edge.source, node_a.id);
        assert_eq!(result.edges[0].edge.target, node_b.id);
    }

    #[test]
    fn merge_retrieval_path_records_source() {
        let options = default_options();

        let vector_only = make_node("fact");
        let graph_only = make_node("fact");
        let both = make_node("fact");

        let vector_results = vec![(vector_only.clone(), 0.9), (both.clone(), 0.7)];
        let graph_results = vec![(graph_only.clone(), 0.8), (both.clone(), 0.6)];

        let result = merge_results(
            &vector_results,
            &graph_results,
            &[],
            &options,
            no_importance_lookup,
        )
        .expect("merge should succeed");

        // Find each node's retrieval step
        let vector_only_step = result
            .retrieval_path
            .iter()
            .find(|s| s.node_id == vector_only.id)
            .expect("vector_only should be in retrieval path");
        assert_eq!(vector_only_step.source, RetrievalSource::Vector);

        let graph_only_step = result
            .retrieval_path
            .iter()
            .find(|s| s.node_id == graph_only.id)
            .expect("graph_only should be in retrieval path");
        assert_eq!(graph_only_step.source, RetrievalSource::Graph);

        let both_step = result
            .retrieval_path
            .iter()
            .find(|s| s.node_id == both.id)
            .expect("both should be in retrieval path");
        assert_eq!(both_step.source, RetrievalSource::Both);
    }

    #[test]
    fn merge_empty_inputs() {
        let options = default_options();

        let result = merge_results(&[], &[], &[], &options, no_importance_lookup)
            .expect("merge should succeed");

        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
        assert!(result.retrieval_path.is_empty());
    }

    // -----------------------------------------------------------------------
    // traverse
    // -----------------------------------------------------------------------

    #[test]
    fn traverse_single_node_no_edges() {
        let start = make_node("fact");
        let start_id = start.id;

        let options = TraversalOptions::default();

        let result = traverse(
            start_id,
            &options,
            |id| {
                if id == start_id {
                    Ok(Some(start.clone()))
                } else {
                    Ok(None)
                }
            },
            |_| Ok(Vec::new()),
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].id, start_id);
        assert!(result.edges.is_empty());
        assert_eq!(result.depth_reached, 0);
    }

    #[test]
    fn traverse_follows_edges() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");
        let edge_bc = make_edge(node_b.id, node_c.id, "knows");

        let a_id = node_a.id;
        let b_id = node_b.id;
        let c_id = node_c.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();
        let node_c_clone = node_c.clone();

        let edge_ab_clone = edge_ab.clone();
        let edge_bc_clone = edge_bc.clone();

        let options = TraversalOptions {
            max_depth: 3,
            limit: 50,
            min_importance: 0.0,
            relation_filter: None,
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else if id == c_id {
                    Ok(Some(node_c_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone()])
                } else if id == b_id {
                    Ok(vec![edge_bc_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(result.nodes.len(), 3, "should find all three nodes");
        assert_eq!(result.edges.len(), 2, "should traverse both edges");
        assert_eq!(result.depth_reached, 2, "depth should be 2");
    }

    #[test]
    fn traverse_respects_max_depth() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");
        let edge_bc = make_edge(node_b.id, node_c.id, "knows");

        let a_id = node_a.id;
        let b_id = node_b.id;
        let c_id = node_c.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();
        let node_c_clone = node_c.clone();

        let edge_ab_clone = edge_ab.clone();
        let edge_bc_clone = edge_bc.clone();

        let options = TraversalOptions {
            max_depth: 1, // only go 1 hop deep
            limit: 50,
            min_importance: 0.0,
            relation_filter: None,
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else if id == c_id {
                    Ok(Some(node_c_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone()])
                } else if id == b_id {
                    Ok(vec![edge_bc_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(result.nodes.len(), 2, "should only find A and B at depth 1");
        assert_eq!(result.depth_reached, 1);
    }

    #[test]
    fn traverse_respects_limit() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");
        let edge_ac = make_edge(node_a.id, node_c.id, "knows");

        let a_id = node_a.id;
        let b_id = node_b.id;
        let c_id = node_c.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();
        let node_c_clone = node_c.clone();

        let edge_ab_clone = edge_ab.clone();
        let edge_ac_clone = edge_ac.clone();

        let options = TraversalOptions {
            max_depth: 3,
            limit: 2, // only return 2 nodes
            min_importance: 0.0,
            relation_filter: None,
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else if id == c_id {
                    Ok(Some(node_c_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone(), edge_ac_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(result.nodes.len(), 2, "should respect limit cap");
    }

    #[test]
    fn traverse_filters_by_relation() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");
        let edge_ab_knows = make_edge(node_a.id, node_b.id, "knows");
        let edge_ac_hates = make_edge(node_a.id, node_c.id, "hates");

        let a_id = node_a.id;
        let b_id = node_b.id;
        let c_id = node_c.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();
        let node_c_clone = node_c.clone();

        let edge_ab_clone = edge_ab_knows.clone();
        let edge_ac_clone = edge_ac_hates.clone();

        let options = TraversalOptions {
            max_depth: 3,
            limit: 50,
            min_importance: 0.0,
            relation_filter: Some(vec!["knows".to_string()]),
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else if id == c_id {
                    Ok(Some(node_c_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone(), edge_ac_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(
            result.nodes.len(),
            2,
            "should find A and B (only 'knows' edges)"
        );
        assert_eq!(result.edges.len(), 1, "should only traverse 'knows' edge");
        assert_eq!(result.edges[0].relation, "knows");
    }

    #[test]
    fn traverse_filters_by_importance() {
        let node_a = make_node_with_importance(0.8);
        let node_b = make_node_with_importance(0.02); // below threshold
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");

        let a_id = node_a.id;
        let b_id = node_b.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();

        let edge_ab_clone = edge_ab.clone();

        let options = TraversalOptions {
            max_depth: 3,
            limit: 50,
            min_importance: 0.1,
            relation_filter: None,
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(
            result.nodes.len(),
            1,
            "low importance node should be filtered"
        );
        assert_eq!(result.nodes[0].id, a_id);
    }

    #[test]
    fn traverse_does_not_revisit_nodes() {
        // Create a cycle: A -> B -> A
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");
        let edge_ba = make_edge(node_b.id, node_a.id, "knows");

        let a_id = node_a.id;
        let b_id = node_b.id;

        let node_a_clone = node_a.clone();
        let node_b_clone = node_b.clone();

        let edge_ab_clone = edge_ab.clone();
        let edge_ba_clone = edge_ba.clone();

        let options = TraversalOptions {
            max_depth: 10,
            limit: 50,
            min_importance: 0.0,
            relation_filter: None,
        };

        let result = traverse(
            a_id,
            &options,
            move |id| {
                if id == a_id {
                    Ok(Some(node_a_clone.clone()))
                } else if id == b_id {
                    Ok(Some(node_b_clone.clone()))
                } else {
                    Ok(None)
                }
            },
            move |id| {
                if id == a_id {
                    Ok(vec![edge_ab_clone.clone()])
                } else if id == b_id {
                    Ok(vec![edge_ba_clone.clone()])
                } else {
                    Ok(Vec::new())
                }
            },
            |_| None,
        )
        .expect("traverse should succeed");

        assert_eq!(
            result.nodes.len(),
            2,
            "cycle should not cause infinite loop"
        );
    }

    // -----------------------------------------------------------------------
    // score_traversal_results
    // -----------------------------------------------------------------------

    #[test]
    fn traversal_scoring_start_node_gets_max_score() {
        let start = make_node("fact");

        let traversal = TraversalResult {
            nodes: vec![start.clone()],
            edges: Vec::new(),
            depth_reached: 0,
        };

        let scored = score_traversal_results(&traversal);
        assert_eq!(scored.len(), 1);
        assert!(
            (scored[0].1 - 1.0).abs() < f32::EPSILON,
            "start node should get score 1.0"
        );
    }

    #[test]
    fn traversal_scoring_decreases_with_depth() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");
        let node_c = make_node("fact");
        let edge_ab = make_edge(node_a.id, node_b.id, "knows");
        let edge_bc = make_edge(node_b.id, node_c.id, "knows");

        let traversal = TraversalResult {
            nodes: vec![node_a.clone(), node_b.clone(), node_c.clone()],
            edges: vec![edge_ab, edge_bc],
            depth_reached: 2,
        };

        let scored = score_traversal_results(&traversal);
        assert_eq!(scored.len(), 3);

        // Find scores by node ID
        let score_a = scored.iter().find(|(n, _)| n.id == node_a.id).unwrap().1;
        let score_b = scored.iter().find(|(n, _)| n.id == node_b.id).unwrap().1;
        let score_c = scored.iter().find(|(n, _)| n.id == node_c.id).unwrap().1;

        assert!((score_a - 1.0).abs() < f32::EPSILON, "depth 0 → score 1.0");
        assert!((score_b - 0.5).abs() < f32::EPSILON, "depth 1 → score 0.5");
        assert!((score_c - 1.0 / 3.0).abs() < 0.001, "depth 2 → score 0.333");
    }

    #[test]
    fn traversal_scoring_empty() {
        let traversal = TraversalResult {
            nodes: Vec::new(),
            edges: Vec::new(),
            depth_reached: 0,
        };

        let scored = score_traversal_results(&traversal);
        assert!(scored.is_empty());
    }

    // -----------------------------------------------------------------------
    // infer_tier_for_search
    // -----------------------------------------------------------------------

    #[test]
    fn tier_inferred_from_metadata() {
        let node = make_node_with_tier_metadata("episodic");
        assert_eq!(infer_tier_for_search(&node), MemoryTier::Episodic);
    }

    #[test]
    fn tier_inferred_from_kind() {
        let node = make_node("event");
        assert_eq!(infer_tier_for_search(&node), MemoryTier::Episodic);

        let node = make_node("working");
        assert_eq!(infer_tier_for_search(&node), MemoryTier::ShortTerm);

        let node = make_node("concept");
        assert_eq!(infer_tier_for_search(&node), MemoryTier::Semantic);
    }

    #[test]
    fn tier_defaults_to_semantic() {
        let node = make_node("unknown_kind");
        assert_eq!(infer_tier_for_search(&node), MemoryTier::Semantic);
    }

    // -----------------------------------------------------------------------
    // nodes_for_retrieval_boost
    // -----------------------------------------------------------------------

    #[test]
    fn retrieval_boost_collects_all_result_nodes() {
        let node_a = make_node("fact");
        let node_b = make_node("fact");

        let result = SearchResult {
            nodes: vec![
                ScoredNode {
                    node: node_a.clone(),
                    score: 0.9,
                    source: RetrievalSource::Vector,
                },
                ScoredNode {
                    node: node_b.clone(),
                    score: 0.8,
                    source: RetrievalSource::Graph,
                },
            ],
            edges: Vec::new(),
            retrieval_path: Vec::new(),
        };

        let boost_entries = nodes_for_retrieval_boost(&result, no_importance_lookup);
        assert_eq!(boost_entries.len(), 2);
    }

    #[test]
    fn retrieval_boost_uses_lookup_when_available() {
        let node = make_node_with_importance(0.5);
        let node_id = node.id;

        let result = SearchResult {
            nodes: vec![ScoredNode {
                node,
                score: 0.9,
                source: RetrievalSource::Vector,
            }],
            edges: Vec::new(),
            retrieval_path: Vec::new(),
        };

        let boost_entries = nodes_for_retrieval_boost(&result, |id| {
            if id == node_id {
                Some(0.3) // importance from lookup differs from node field
            } else {
                None
            }
        });

        assert_eq!(boost_entries.len(), 1);
        assert!(
            (boost_entries[0].1 - 0.3).abs() < f32::EPSILON,
            "should use lookup value, not node field"
        );
    }

    #[test]
    fn retrieval_boost_empty_result() {
        let result = SearchResult {
            nodes: Vec::new(),
            edges: Vec::new(),
            retrieval_path: Vec::new(),
        };

        let boost_entries = nodes_for_retrieval_boost(&result, no_importance_lookup);
        assert!(boost_entries.is_empty());
    }
}
