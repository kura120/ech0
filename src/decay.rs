//! Importance scoring, time-based decay, and threshold pruning for ech0.
//!
//! When the `importance-decay` feature is enabled, ech0 supports:
//! - **Time-based decay**: importance scores decrease over time when memories are not accessed.
//!   Episodic memories decay at `episodic_decay_rate` per day, semantic memories at
//!   `semantic_decay_rate` per day.
//! - **Retrieval boost**: importance scores increase when a memory is accessed via search
//!   or traversal, by `importance_boost_on_retrieval`.
//! - **Threshold pruning**: nodes with importance below `prune_threshold` are removed from
//!   both the graph and vector layers.
//!
//! Decay does not run on a timer — it is applied when `Store::decay()` is called explicitly
//! or (optionally) as part of the ingest pipeline.

use chrono::{DateTime, Utc};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::config::MemoryConfig;

use crate::schema::{DecayReport, MemoryTier, Node, PruneReport};

// ---------------------------------------------------------------------------
// Decay computation
// ---------------------------------------------------------------------------

/// Compute the new importance score for a single node after time-based decay.
///
/// The decay formula is:
///   `new_importance = current_importance - (decay_rate * days_since_last_access)`
///
/// The result is clamped to `[0.0, f32::MAX]` — importance never goes negative.
///
/// # Arguments
///
/// * `current_importance` — the node's current importance score.
/// * `decay_rate` — per-day decay rate (from config, depends on memory tier).
/// * `last_accessed` — when the node was last accessed (retrieved or written).
///   For nodes that have never been accessed after creation, this is `created_at`.
/// * `now` — the current timestamp to compute elapsed time against.
pub fn compute_decayed_importance(
    current_importance: f32,
    decay_rate: f32,
    last_accessed: DateTime<Utc>,
    now: DateTime<Utc>,
) -> f32 {
    let elapsed = now.signed_duration_since(last_accessed);
    let days_elapsed = elapsed.num_seconds().max(0) as f32 / 86400.0;

    let decay_amount = decay_rate * days_elapsed;
    let new_importance = current_importance - decay_amount;

    // Clamp to non-negative — importance never goes below zero
    new_importance.max(0.0)
}

/// Compute the importance boost for a node that was just retrieved.
///
/// The boost is additive: `new_importance = current_importance + boost_amount`.
/// There is no upper clamp — callers decide if they want to cap importance.
///
/// # Arguments
///
/// * `current_importance` — the node's current importance score.
/// * `boost_amount` — the configured `importance_boost_on_retrieval` value.
pub fn compute_retrieval_boost(current_importance: f32, boost_amount: f32) -> f32 {
    current_importance + boost_amount
}

/// Select the appropriate decay rate for a node based on its memory tier.
///
/// Episodic memories decay faster than semantic memories. Short-term memories
/// are not subject to time-based decay (they are managed by capacity eviction).
///
/// # Arguments
///
/// * `tier` — the memory tier classification of the node.
/// * `config` — memory configuration containing per-tier decay rates.
pub fn decay_rate_for_tier(tier: MemoryTier, config: &MemoryConfig) -> f32 {
    match tier {
        MemoryTier::ShortTerm => 0.0, // short-term memories do not decay by time
        MemoryTier::Episodic => config.episodic_decay_rate,
        MemoryTier::Semantic => config.semantic_decay_rate,
    }
}

// ---------------------------------------------------------------------------
// Batch decay
// ---------------------------------------------------------------------------

/// Information needed to decay a single node.
#[derive(Debug, Clone)]
pub struct DecayEntry {
    /// The node's unique ID.
    pub node_id: Uuid,

    /// The node's current importance score (from the importance table).
    pub current_importance: f32,

    /// The memory tier of this node, used to select the correct decay rate.
    pub tier: MemoryTier,

    /// When this node was last accessed (retrieved or modified).
    /// For nodes never accessed after creation, this is `created_at`.
    pub last_accessed: DateTime<Utc>,
}

/// Result of computing decay for a single node.
#[derive(Debug, Clone)]
pub struct DecayUpdate {
    /// The node's unique ID.
    pub node_id: Uuid,

    /// The new importance score after decay.
    pub new_importance: f32,

    /// The previous importance score before decay.
    pub previous_importance: f32,

    /// Whether this node is now below the prune threshold.
    pub below_threshold: bool,
}

/// Compute decay updates for a batch of nodes.
///
/// This is a pure computation — it does not touch storage. The caller (Store)
/// is responsible for writing the updated importance scores to the graph layer.
///
/// # Arguments
///
/// * `entries` — nodes to decay, with their current state.
/// * `config` — memory configuration containing decay rates and prune threshold.
/// * `now` — the current timestamp to compute elapsed time against.
///
/// # Returns
///
/// A `DecayReport` summary and the list of individual `DecayUpdate`s to apply.
#[instrument(skip_all, fields(entry_count = entries.len()))]
pub fn compute_batch_decay(
    entries: &[DecayEntry],
    config: &MemoryConfig,
    now: DateTime<Utc>,
) -> (DecayReport, Vec<DecayUpdate>) {
    let mut updates = Vec::with_capacity(entries.len());
    let mut nodes_decayed: usize = 0;
    let mut nodes_below_threshold: usize = 0;

    for entry in entries {
        let rate = decay_rate_for_tier(entry.tier, config);

        // Short-term nodes have a rate of 0.0 — they won't change, so skip them
        // to avoid unnecessary writes.
        if rate == 0.0 {
            continue;
        }

        let new_importance =
            compute_decayed_importance(entry.current_importance, rate, entry.last_accessed, now);

        // Only emit an update if the score actually changed (avoid spurious writes
        // for nodes that were just accessed moments ago).
        let delta = (entry.current_importance - new_importance).abs();
        if delta < f32::EPSILON {
            continue;
        }

        let below_threshold = new_importance < config.prune_threshold;
        if below_threshold {
            nodes_below_threshold += 1;
        }

        nodes_decayed += 1;

        updates.push(DecayUpdate {
            node_id: entry.node_id,
            new_importance,
            previous_importance: entry.current_importance,
            below_threshold,
        });
    }

    let report = DecayReport {
        nodes_decayed,
        // Edge decay is handled separately — edges inherit importance from their
        // connected nodes. For now, report zero edge decay; the Store can update
        // this after propagating node decay to edges.
        edges_decayed: 0,
        nodes_below_threshold,
    };

    debug!(
        nodes_decayed = nodes_decayed,
        nodes_below_threshold = nodes_below_threshold,
        "batch decay computed"
    );

    (report, updates)
}

// ---------------------------------------------------------------------------
// Batch retrieval boost
// ---------------------------------------------------------------------------

/// Result of boosting importance for retrieved nodes.
#[derive(Debug, Clone)]
pub struct BoostUpdate {
    /// The node's unique ID.
    pub node_id: Uuid,

    /// The new importance score after boost.
    pub new_importance: f32,

    /// The previous importance score before boost.
    pub previous_importance: f32,
}

/// Compute retrieval boost updates for a batch of nodes that were just accessed.
///
/// Pure computation — does not touch storage.
///
/// # Arguments
///
/// * `accessed_nodes` — list of `(node_id, current_importance)` pairs for all nodes
///   that were returned in a search or traversal result.
/// * `config` — memory configuration containing the boost amount.
#[instrument(skip_all, fields(node_count = accessed_nodes.len()))]
pub fn compute_batch_retrieval_boost(
    accessed_nodes: &[(Uuid, f32)],
    config: &MemoryConfig,
) -> Vec<BoostUpdate> {
    let boost = config.importance_boost_on_retrieval;
    if boost <= 0.0 {
        return Vec::new();
    }

    accessed_nodes
        .iter()
        .map(|(node_id, current_importance)| {
            let new_importance = compute_retrieval_boost(*current_importance, boost);
            BoostUpdate {
                node_id: *node_id,
                new_importance,
                previous_importance: *current_importance,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

/// Identify nodes that should be pruned based on a given importance threshold.
///
/// Returns the UUIDs of all nodes with importance strictly below `threshold`.
/// Does not perform the actual deletion — the caller (Store) handles that to
/// ensure atomic deletion from both graph and vector layers.
///
/// # Arguments
///
/// * `nodes_with_importance` — list of `(node_id, current_importance)` pairs,
///   typically from the importance table in the graph layer.
/// * `threshold` — nodes below this importance are candidates for pruning.
pub fn identify_prune_candidates(
    nodes_with_importance: &[(Uuid, f32)],
    threshold: f32,
) -> Vec<Uuid> {
    nodes_with_importance
        .iter()
        .filter(|(_, importance)| *importance < threshold)
        .map(|(node_id, _)| *node_id)
        .collect()
}

/// Estimate the prune report for a set of candidates before actually deleting.
///
/// This is used to build the `PruneReport` that is returned to the caller.
/// The actual deletion is performed by the graph and vector layers.
///
/// # Arguments
///
/// * `candidates` — node IDs that will be pruned.
/// * `edge_count` — number of edges that will be removed as a result of pruning
///   (caller computes this by checking edges connected to pruned nodes).
pub fn build_prune_report(
    candidates: &[Uuid],
    edge_count: usize,
    vector_count: usize,
) -> PruneReport {
    PruneReport {
        nodes_pruned: candidates.len(),
        edges_pruned: edge_count,
        vectors_pruned: vector_count,
    }
}

/// Infer a memory tier for a node based on its metadata.
///
/// In V1 ech0 does not store an explicit tier field on nodes — the tier is inferred
/// from the node's `kind` or metadata. This is a heuristic and callers can override
/// it by storing a `"tier"` key in the node's metadata.
///
/// The default mapping is:
/// - `kind == "event"` or metadata contains `"tier": "episodic"` → `Episodic`
/// - `kind == "working"` or metadata contains `"tier": "short_term"` → `ShortTerm`
/// - Everything else → `Semantic`
///
/// This function is intentionally simple. A future version may store the tier
/// explicitly as a redb column.
pub fn infer_tier(node: &Node) -> MemoryTier {
    // Check explicit tier in metadata first
    if let Some(tier_value) = node.metadata.get("tier")
        && let Some(tier_str) = tier_value.as_str()
    {
        match tier_str {
            "short_term" => return MemoryTier::ShortTerm,
            "episodic" => return MemoryTier::Episodic,
            "semantic" => return MemoryTier::Semantic,
            _ => {} // unrecognized tier value — fall through to kind-based heuristic
        }
    }

    // Fall back to kind-based heuristic
    match node.kind.as_str() {
        "event" | "episode" => MemoryTier::Episodic,
        "working" | "short_term" | "session" => MemoryTier::ShortTerm,
        _ => MemoryTier::Semantic,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_config() -> MemoryConfig {
        MemoryConfig {
            short_term_capacity: 50,
            episodic_decay_rate: 0.01,
            semantic_decay_rate: 0.005,
            prune_threshold: 0.1,
            importance_boost_on_retrieval: 0.1,
        }
    }

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
    // compute_decayed_importance
    // -----------------------------------------------------------------------

    #[test]
    fn no_decay_when_zero_time_elapsed() {
        let now = Utc::now();
        let result = compute_decayed_importance(0.8, 0.01, now, now);
        assert!((result - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_proportional_to_elapsed_days() {
        let now = Utc::now();
        let ten_days_ago = now - Duration::days(10);

        // 0.8 - (0.01 * 10) = 0.8 - 0.1 = 0.7
        let result = compute_decayed_importance(0.8, 0.01, ten_days_ago, now);
        assert!((result - 0.7).abs() < 0.001, "expected ~0.7, got {result}");
    }

    #[test]
    fn decay_clamps_to_zero() {
        let now = Utc::now();
        let long_ago = now - Duration::days(1000);

        // 0.5 - (0.01 * 1000) = 0.5 - 10 = -9.5 → clamped to 0.0
        let result = compute_decayed_importance(0.5, 0.01, long_ago, now);
        assert!(
            (result - 0.0).abs() < f32::EPSILON,
            "should clamp to 0.0, got {result}"
        );
    }

    #[test]
    fn decay_with_zero_rate_is_unchanged() {
        let now = Utc::now();
        let week_ago = now - Duration::days(7);

        let result = compute_decayed_importance(0.8, 0.0, week_ago, now);
        assert!(
            (result - 0.8).abs() < f32::EPSILON,
            "zero rate should not change importance"
        );
    }

    #[test]
    fn decay_handles_future_timestamp_gracefully() {
        // If last_accessed is in the future (clock skew), elapsed is negative → clamped to 0
        let now = Utc::now();
        let future = now + Duration::days(5);

        let result = compute_decayed_importance(0.8, 0.01, future, now);
        // elapsed is negative → days_elapsed = 0 → no decay
        assert!(
            (result - 0.8).abs() < f32::EPSILON,
            "future timestamp should not cause decay, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // compute_retrieval_boost
    // -----------------------------------------------------------------------

    #[test]
    fn retrieval_boost_increases_importance() {
        let result = compute_retrieval_boost(0.5, 0.1);
        assert!((result - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn retrieval_boost_with_zero_is_unchanged() {
        let result = compute_retrieval_boost(0.5, 0.0);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn retrieval_boost_can_exceed_one() {
        // No upper clamp — callers decide
        let result = compute_retrieval_boost(0.95, 0.1);
        assert!(result > 1.0, "boost should allow exceeding 1.0");
    }

    // -----------------------------------------------------------------------
    // decay_rate_for_tier
    // -----------------------------------------------------------------------

    #[test]
    fn short_term_has_zero_decay_rate() {
        let config = make_config();
        assert!((decay_rate_for_tier(MemoryTier::ShortTerm, &config) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn episodic_uses_episodic_rate() {
        let config = make_config();
        assert!(
            (decay_rate_for_tier(MemoryTier::Episodic, &config) - config.episodic_decay_rate).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn semantic_uses_semantic_rate() {
        let config = make_config();
        assert!(
            (decay_rate_for_tier(MemoryTier::Semantic, &config) - config.semantic_decay_rate).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn episodic_decays_faster_than_semantic() {
        let config = make_config();
        let episodic_rate = decay_rate_for_tier(MemoryTier::Episodic, &config);
        let semantic_rate = decay_rate_for_tier(MemoryTier::Semantic, &config);
        assert!(
            episodic_rate > semantic_rate,
            "episodic should decay faster than semantic"
        );
    }

    // -----------------------------------------------------------------------
    // compute_batch_decay
    // -----------------------------------------------------------------------

    #[test]
    fn batch_decay_computes_updates_for_decayable_nodes() {
        let config = make_config();
        let now = Utc::now();
        let ten_days_ago = now - Duration::days(10);

        let entries = vec![
            DecayEntry {
                node_id: Uuid::new_v4(),
                current_importance: 0.8,
                tier: MemoryTier::Episodic,
                last_accessed: ten_days_ago,
            },
            DecayEntry {
                node_id: Uuid::new_v4(),
                current_importance: 0.5,
                tier: MemoryTier::Semantic,
                last_accessed: ten_days_ago,
            },
        ];

        let (report, updates) = compute_batch_decay(&entries, &config, now);

        assert_eq!(report.nodes_decayed, 2);
        assert_eq!(updates.len(), 2);

        // Episodic: 0.8 - (0.01 * 10) = 0.7
        assert!(
            (updates[0].new_importance - 0.7).abs() < 0.001,
            "episodic decay: expected ~0.7, got {}",
            updates[0].new_importance
        );

        // Semantic: 0.5 - (0.005 * 10) = 0.45
        assert!(
            (updates[1].new_importance - 0.45).abs() < 0.001,
            "semantic decay: expected ~0.45, got {}",
            updates[1].new_importance
        );
    }

    #[test]
    fn batch_decay_skips_short_term_nodes() {
        let config = make_config();
        let now = Utc::now();
        let week_ago = now - Duration::days(7);

        let entries = vec![DecayEntry {
            node_id: Uuid::new_v4(),
            current_importance: 0.8,
            tier: MemoryTier::ShortTerm,
            last_accessed: week_ago,
        }];

        let (report, updates) = compute_batch_decay(&entries, &config, now);

        assert_eq!(report.nodes_decayed, 0, "short-term nodes should not decay");
        assert!(updates.is_empty());
    }

    #[test]
    fn batch_decay_flags_below_threshold() {
        let config = make_config(); // prune_threshold = 0.1
        let now = Utc::now();
        let hundred_days_ago = now - Duration::days(100);

        let entries = vec![DecayEntry {
            node_id: Uuid::new_v4(),
            current_importance: 0.5,
            tier: MemoryTier::Episodic,
            last_accessed: hundred_days_ago,
        }];

        let (report, updates) = compute_batch_decay(&entries, &config, now);

        assert_eq!(report.nodes_below_threshold, 1);
        assert!(updates[0].below_threshold);
        // 0.5 - (0.01 * 100) = 0.5 - 1.0 = -0.5 → clamped to 0.0
        assert!(
            updates[0].new_importance < config.prune_threshold,
            "should be below prune threshold"
        );
    }

    #[test]
    fn batch_decay_skips_recently_accessed_nodes() {
        let config = make_config();
        let now = Utc::now();

        // Node was just accessed — delta is ~0 so no update should be emitted
        let entries = vec![DecayEntry {
            node_id: Uuid::new_v4(),
            current_importance: 0.8,
            tier: MemoryTier::Episodic,
            last_accessed: now,
        }];

        let (report, updates) = compute_batch_decay(&entries, &config, now);
        assert_eq!(
            report.nodes_decayed, 0,
            "recently accessed nodes should not produce updates"
        );
        assert!(updates.is_empty());
    }

    #[test]
    fn batch_decay_with_empty_input() {
        let config = make_config();
        let now = Utc::now();

        let (report, updates) = compute_batch_decay(&[], &config, now);

        assert_eq!(report.nodes_decayed, 0);
        assert_eq!(report.nodes_below_threshold, 0);
        assert!(updates.is_empty());
    }

    // -----------------------------------------------------------------------
    // compute_batch_retrieval_boost
    // -----------------------------------------------------------------------

    #[test]
    fn batch_boost_applies_to_all_accessed_nodes() {
        let config = make_config();

        let accessed = vec![(Uuid::new_v4(), 0.5f32), (Uuid::new_v4(), 0.3f32)];

        let boosts = compute_batch_retrieval_boost(&accessed, &config);
        assert_eq!(boosts.len(), 2);

        assert!((boosts[0].new_importance - 0.6).abs() < f32::EPSILON);
        assert!((boosts[0].previous_importance - 0.5).abs() < f32::EPSILON);

        assert!((boosts[1].new_importance - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn batch_boost_with_zero_boost_returns_empty() {
        let mut config = make_config();
        config.importance_boost_on_retrieval = 0.0;

        let accessed = vec![(Uuid::new_v4(), 0.5f32)];
        let boosts = compute_batch_retrieval_boost(&accessed, &config);
        assert!(boosts.is_empty());
    }

    #[test]
    fn batch_boost_with_empty_input() {
        let config = make_config();
        let boosts = compute_batch_retrieval_boost(&[], &config);
        assert!(boosts.is_empty());
    }

    // -----------------------------------------------------------------------
    // identify_prune_candidates
    // -----------------------------------------------------------------------

    #[test]
    fn prune_identifies_nodes_below_threshold() {
        let nodes = vec![
            (Uuid::new_v4(), 0.05f32), // below 0.1
            (Uuid::new_v4(), 0.5f32),  // above
            (Uuid::new_v4(), 0.09f32), // below 0.1
            (Uuid::new_v4(), 0.1f32),  // exactly at threshold — NOT below
        ];

        let candidates = identify_prune_candidates(&nodes, 0.1);
        assert_eq!(candidates.len(), 2, "two nodes below threshold");
    }

    #[test]
    fn prune_with_zero_threshold_prunes_nothing() {
        let nodes = vec![(Uuid::new_v4(), 0.0f32), (Uuid::new_v4(), 0.001f32)];

        // 0.0 is not strictly less than 0.0
        let candidates = identify_prune_candidates(&nodes, 0.0);
        assert!(candidates.is_empty());
    }

    #[test]
    fn prune_with_threshold_one_prunes_all_below() {
        let nodes = vec![(Uuid::new_v4(), 0.5f32), (Uuid::new_v4(), 0.99f32)];

        let candidates = identify_prune_candidates(&nodes, 1.0);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn prune_empty_input() {
        let candidates = identify_prune_candidates(&[], 0.1);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // build_prune_report
    // -----------------------------------------------------------------------

    #[test]
    fn prune_report_counts_correctly() {
        let candidates = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let report = build_prune_report(&candidates, 5, 3);

        assert_eq!(report.nodes_pruned, 3);
        assert_eq!(report.edges_pruned, 5);
        assert_eq!(report.vectors_pruned, 3);
    }

    #[test]
    fn prune_report_empty() {
        let report = build_prune_report(&[], 0, 0);
        assert_eq!(report.nodes_pruned, 0);
        assert_eq!(report.edges_pruned, 0);
        assert_eq!(report.vectors_pruned, 0);
    }

    // -----------------------------------------------------------------------
    // infer_tier
    // -----------------------------------------------------------------------

    #[test]
    fn infer_tier_from_explicit_metadata() {
        let node = make_test_node("fact", serde_json::json!({"tier": "episodic"}));
        assert_eq!(infer_tier(&node), MemoryTier::Episodic);

        let node = make_test_node("fact", serde_json::json!({"tier": "short_term"}));
        assert_eq!(infer_tier(&node), MemoryTier::ShortTerm);

        let node = make_test_node("fact", serde_json::json!({"tier": "semantic"}));
        assert_eq!(infer_tier(&node), MemoryTier::Semantic);
    }

    #[test]
    fn infer_tier_from_kind_when_no_metadata_tier() {
        let node = make_test_node("event", serde_json::json!({}));
        assert_eq!(infer_tier(&node), MemoryTier::Episodic);

        let node = make_test_node("episode", serde_json::json!({}));
        assert_eq!(infer_tier(&node), MemoryTier::Episodic);

        let node = make_test_node("working", serde_json::json!({}));
        assert_eq!(infer_tier(&node), MemoryTier::ShortTerm);

        let node = make_test_node("session", serde_json::json!({}));
        assert_eq!(infer_tier(&node), MemoryTier::ShortTerm);
    }

    #[test]
    fn infer_tier_defaults_to_semantic() {
        let node = make_test_node("person", serde_json::json!({}));
        assert_eq!(infer_tier(&node), MemoryTier::Semantic);

        let node = make_test_node("concept", serde_json::json!({"foo": "bar"}));
        assert_eq!(infer_tier(&node), MemoryTier::Semantic);
    }

    #[test]
    fn infer_tier_metadata_overrides_kind() {
        // Kind says "event" (episodic), but metadata explicitly says "semantic"
        let node = make_test_node("event", serde_json::json!({"tier": "semantic"}));
        assert_eq!(infer_tier(&node), MemoryTier::Semantic);
    }

    #[test]
    fn infer_tier_ignores_invalid_metadata_tier() {
        // Invalid tier string in metadata falls through to kind-based heuristic
        let node = make_test_node("event", serde_json::json!({"tier": "nonsense"}));
        assert_eq!(infer_tier(&node), MemoryTier::Episodic);
    }

    #[test]
    fn infer_tier_ignores_non_string_metadata_tier() {
        // tier is a number, not a string — should be ignored
        let node = make_test_node("fact", serde_json::json!({"tier": 42}));
        assert_eq!(infer_tier(&node), MemoryTier::Semantic);
    }
}
