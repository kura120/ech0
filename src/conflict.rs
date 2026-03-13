//! Contradiction detection for ech0.
//!
//! When the `contradiction-detection` feature is enabled, every ingest checks for conflicts
//! between newly extracted nodes and existing nodes in the graph. A conflict exists when a
//! newly extracted node asserts something that directly contradicts an existing node of the
//! same kind and subject.
//!
//! ech0 never resolves conflicts silently. The default resolution policy is `Escalate` —
//! the caller receives a `ConflictReport` and decides what to do.

use serde::{Deserialize, Serialize};

use crate::config::ContradictionConfig;
use crate::error::EchoError;
use crate::schema::Node;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Report describing a detected contradiction between a new and existing node.
#[derive(Debug, Clone)]
pub struct ConflictReport {
    /// The newly ingested node that triggered the conflict.
    pub new_node: Node,

    /// The existing node in the graph that contradicts the new node.
    pub existing_node: Node,

    /// Classification of the contradiction.
    pub conflict_type: ConflictType,

    /// Confidence that this is a real contradiction, not noise. Range: 0.0–1.0.
    /// Only conflicts above `ContradictionConfig::confidence_threshold` are reported.
    pub confidence: f32,
}

/// Classification of contradiction between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictType {
    /// Direct logical contradiction: "X is Y" vs "X is not Y".
    DirectContradiction,

    /// Value conflict: "X is 30" vs "X is 25".
    ValueConflict,

    /// Temporal conflict: an older fact may have been superseded by newer information.
    TemporalConflict,
}

/// Policy for resolving a detected contradiction.
///
/// Configured per `Store` instance via `ContradictionConfig::resolution_policy`.
/// Default is `Escalate` — ech0 never silently resolves conflicts unless explicitly
/// configured otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Keep the existing memory, discard the new one.
    KeepExisting,

    /// Replace the existing memory with the new one.
    ReplaceWithNew,

    /// Store both memories, potentially with adjusted confidence/importance scores.
    KeepBoth,

    /// Return the conflict to the caller for manual resolution. This is the default.
    Escalate,
}

impl Default for ConflictResolution {
    fn default() -> Self {
        Self::Escalate
    }
}

// ---------------------------------------------------------------------------
// Detection engine
// ---------------------------------------------------------------------------

/// Runs contradiction detection for a batch of newly ingested nodes against
/// the set of existing nodes retrieved from the graph.
///
/// # Arguments
///
/// * `new_nodes` — nodes that are about to be written in the current ingest.
/// * `existing_candidates` — existing nodes from the graph that are semantically
///   similar to the new nodes (pre-filtered by vector search). Only these candidates
///   are checked — we do not scan the entire graph.
/// * `config` — contradiction detection configuration (confidence threshold, etc.).
///
/// # Returns
///
/// A list of `ConflictReport`s for all detected contradictions above the confidence
/// threshold. An empty list means no contradictions were found.
pub fn detect_conflicts(
    new_nodes: &[Node],
    existing_candidates: &[Node],
    config: &ContradictionConfig,
) -> Result<Vec<ConflictReport>, EchoError> {
    let mut reports = Vec::new();

    for new_node in new_nodes {
        for existing_node in existing_candidates {
            // Only compare nodes of the same kind — a "person" node cannot contradict
            // an "event" node in a meaningful way.
            if new_node.kind != existing_node.kind {
                continue;
            }

            if let Some(report) =
                check_pair_for_conflict(new_node, existing_node, config.confidence_threshold)?
            {
                reports.push(report);
            }
        }
    }

    Ok(reports)
}

/// Check a single (new, existing) node pair for contradiction.
///
/// Returns `Some(ConflictReport)` if a contradiction is detected above the confidence
/// threshold, `None` otherwise.
fn check_pair_for_conflict(
    new_node: &Node,
    existing_node: &Node,
    confidence_threshold: f32,
) -> Result<Option<ConflictReport>, EchoError> {
    // TODO: Implement real contradiction detection logic. This requires semantic
    // comparison of node metadata to determine if two nodes of the same kind
    // assert conflicting information about the same subject.
    //
    // The production implementation should:
    // 1. Extract the "subject" from each node's metadata (e.g. entity name or ID)
    // 2. Determine if both nodes are making assertions about the same subject
    // 3. Compare the assertions to detect direct contradictions, value conflicts,
    //    or temporal conflicts
    // 4. Assign a confidence score based on how clearly the assertions conflict
    //
    // For now, use a simple heuristic: if two nodes of the same kind share a common
    // metadata key with different values, flag it as a potential ValueConflict.

    let confidence = estimate_conflict_confidence(new_node, existing_node);

    if confidence < confidence_threshold {
        return Ok(None);
    }

    let conflict_type = classify_conflict(new_node, existing_node);

    Ok(Some(ConflictReport {
        new_node: new_node.clone(),
        existing_node: existing_node.clone(),
        conflict_type,
        confidence,
    }))
}

/// Estimate how confident we are that two nodes contradict each other.
///
/// Returns a value in the range 0.0–1.0. Higher means more likely a real contradiction.
///
/// TODO: Replace this stub with real semantic comparison. The current implementation
/// uses a simple metadata key overlap heuristic.
fn estimate_conflict_confidence(new_node: &Node, existing_node: &Node) -> f32 {
    let new_obj = match new_node.metadata.as_object() {
        Some(object) => object,
        None => return 0.0,
    };
    let existing_obj = match existing_node.metadata.as_object() {
        Some(object) => object,
        None => return 0.0,
    };

    if new_obj.is_empty() || existing_obj.is_empty() {
        return 0.0;
    }

    let mut shared_keys = 0u32;
    let mut conflicting_values = 0u32;

    for (key, new_value) in new_obj {
        if let Some(existing_value) = existing_obj.get(key) {
            shared_keys += 1;
            if new_value != existing_value {
                conflicting_values += 1;
            }
        }
    }

    if shared_keys == 0 {
        return 0.0;
    }

    // Rough heuristic: ratio of conflicting values to shared keys, scaled down
    // because metadata key overlap alone is weak signal.
    let ratio = conflicting_values as f32 / shared_keys as f32;
    // Scale to 0.0–0.9 range — never return 1.0 from a heuristic, leave room
    // for the real implementation to express higher confidence.
    ratio * 0.9
}

/// Classify the type of conflict between two nodes.
///
/// TODO: Replace this stub with real classification logic that examines
/// the semantic content of the assertions.
fn classify_conflict(new_node: &Node, existing_node: &Node) -> ConflictType {
    // Simple heuristic: if the existing node is significantly older, treat it
    // as a temporal conflict (the old fact may have been superseded).
    let age_difference = new_node
        .created_at
        .signed_duration_since(existing_node.created_at);

    if age_difference.num_days() > 30 {
        return ConflictType::TemporalConflict;
    }

    // Default to ValueConflict since our heuristic is based on metadata value differences.
    ConflictType::ValueConflict
}

/// Apply the configured resolution policy to a set of conflicts.
///
/// This is called by `Store` after detection to determine which nodes to actually write.
///
/// # Returns
///
/// A list of `(ConflictReport, ConflictResolution)` pairs. For `Escalate` policy, all
/// conflicts are returned with `Escalate` resolution — the caller must handle them.
///
/// For other policies, the resolution is applied automatically and the report is still
/// returned so the caller knows what happened.
pub fn apply_resolution_policy(
    conflicts: &[ConflictReport],
    config: &ContradictionConfig,
) -> Vec<(ConflictReport, ConflictResolution)> {
    let policy = config.parsed_resolution_policy();

    conflicts
        .iter()
        .map(|report| (report.clone(), policy))
        .collect()
}

/// Determine which node IDs should be excluded from the write based on resolved conflicts.
///
/// Returns a list of new node IDs that should NOT be written because the resolution
/// policy determined they should be discarded (e.g. `KeepExisting` policy).
pub fn nodes_to_exclude(resolved: &[(ConflictReport, ConflictResolution)]) -> Vec<uuid::Uuid> {
    resolved
        .iter()
        .filter_map(|(report, resolution)| match resolution {
            ConflictResolution::KeepExisting => Some(report.new_node.id),
            ConflictResolution::Escalate => {
                // Escalate means we still write the node but return the conflict
                // to the caller. The node is written so the caller has both versions
                // available for manual resolution.
                None
            }
            ConflictResolution::ReplaceWithNew | ConflictResolution::KeepBoth => None,
        })
        .collect()
}

/// Determine which existing node IDs should be removed because the resolution policy
/// chose to replace them with the new version.
///
/// Returns a list of existing node IDs that should be deleted from the graph.
pub fn nodes_to_replace(resolved: &[(ConflictReport, ConflictResolution)]) -> Vec<uuid::Uuid> {
    resolved
        .iter()
        .filter_map(|(report, resolution)| match resolution {
            ConflictResolution::ReplaceWithNew => Some(report.existing_node.id),
            _ => None,
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
    use uuid::Uuid;

    fn make_node(kind: &str, metadata: serde_json::Value) -> Node {
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

    fn default_config() -> ContradictionConfig {
        ContradictionConfig::default()
    }

    #[test]
    fn no_conflicts_when_nodes_have_different_kinds() {
        let new_node = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let existing_node = make_node("event", serde_json::json!({"name": "Alice", "age": 25}));

        let reports = detect_conflicts(&[new_node], &[existing_node], &default_config())
            .expect("detection should not fail");

        assert!(
            reports.is_empty(),
            "different kinds should not trigger conflict"
        );
    }

    #[test]
    fn no_conflicts_when_metadata_values_match() {
        let new_node = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let existing_node = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));

        let reports = detect_conflicts(&[new_node], &[existing_node], &default_config())
            .expect("detection should not fail");

        assert!(
            reports.is_empty(),
            "identical metadata should not trigger conflict"
        );
    }

    #[test]
    fn conflict_detected_when_metadata_values_differ() {
        let new_node = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let existing_node = make_node("person", serde_json::json!({"name": "Alice", "age": 25}));

        // Use a low threshold so the heuristic triggers
        let mut config = default_config();
        config.confidence_threshold = 0.1;

        let reports = detect_conflicts(&[new_node], &[existing_node], &config)
            .expect("detection should not fail");

        assert!(
            !reports.is_empty(),
            "differing metadata values should trigger conflict with low threshold"
        );
        assert_eq!(reports[0].conflict_type, ConflictType::ValueConflict);
    }

    #[test]
    fn no_conflicts_when_metadata_is_empty() {
        let new_node = make_node("person", serde_json::json!({}));
        let existing_node = make_node("person", serde_json::json!({}));

        let reports = detect_conflicts(&[new_node], &[existing_node], &default_config())
            .expect("detection should not fail");

        assert!(
            reports.is_empty(),
            "empty metadata should not trigger conflict"
        );
    }

    #[test]
    fn no_conflicts_when_no_shared_keys() {
        let new_node = make_node("person", serde_json::json!({"name": "Alice"}));
        let existing_node = make_node("person", serde_json::json!({"age": 30}));

        let mut config = default_config();
        config.confidence_threshold = 0.01;

        let reports = detect_conflicts(&[new_node], &[existing_node], &config)
            .expect("detection should not fail");

        assert!(
            reports.is_empty(),
            "no shared keys means no conflict signal"
        );
    }

    #[test]
    fn conflict_below_threshold_is_ignored() {
        let new_node = make_node(
            "person",
            serde_json::json!({"name": "Alice", "city": "NYC", "age": 30}),
        );
        let existing_node = make_node(
            "person",
            serde_json::json!({"name": "Alice", "city": "NYC", "age": 25}),
        );

        // High threshold — only 1 of 3 keys conflicts, so confidence is ~0.3 * 0.9 = 0.27
        let mut config = default_config();
        config.confidence_threshold = 0.95;

        let reports = detect_conflicts(&[new_node], &[existing_node], &config)
            .expect("detection should not fail");

        assert!(
            reports.is_empty(),
            "conflict below threshold should be ignored"
        );
    }

    #[test]
    fn default_resolution_is_escalate() {
        assert_eq!(ConflictResolution::default(), ConflictResolution::Escalate);
    }

    #[test]
    fn apply_resolution_policy_returns_configured_policy() {
        let new_node = make_node("person", serde_json::json!({"age": 30}));
        let existing_node = make_node("person", serde_json::json!({"age": 25}));

        let report = ConflictReport {
            new_node,
            existing_node,
            conflict_type: ConflictType::ValueConflict,
            confidence: 0.9,
        };

        let config = default_config(); // resolution_policy = "escalate"
        let resolved = apply_resolution_policy(&[report], &config);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].1, ConflictResolution::Escalate);
    }

    #[test]
    fn nodes_to_exclude_with_keep_existing_policy() {
        let new_node = make_node("person", serde_json::json!({"age": 30}));
        let new_node_id = new_node.id;
        let existing_node = make_node("person", serde_json::json!({"age": 25}));

        let report = ConflictReport {
            new_node,
            existing_node,
            conflict_type: ConflictType::ValueConflict,
            confidence: 0.9,
        };

        let resolved = vec![(report, ConflictResolution::KeepExisting)];
        let excluded = nodes_to_exclude(&resolved);

        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0], new_node_id);
    }

    #[test]
    fn nodes_to_exclude_with_escalate_policy_excludes_nothing() {
        let new_node = make_node("person", serde_json::json!({"age": 30}));
        let existing_node = make_node("person", serde_json::json!({"age": 25}));

        let report = ConflictReport {
            new_node,
            existing_node,
            conflict_type: ConflictType::ValueConflict,
            confidence: 0.9,
        };

        let resolved = vec![(report, ConflictResolution::Escalate)];
        let excluded = nodes_to_exclude(&resolved);

        assert!(excluded.is_empty(), "escalate should not exclude any nodes");
    }

    #[test]
    fn nodes_to_replace_with_replace_policy() {
        let new_node = make_node("person", serde_json::json!({"age": 30}));
        let existing_node = make_node("person", serde_json::json!({"age": 25}));
        let existing_id = existing_node.id;

        let report = ConflictReport {
            new_node,
            existing_node,
            conflict_type: ConflictType::ValueConflict,
            confidence: 0.9,
        };

        let resolved = vec![(report, ConflictResolution::ReplaceWithNew)];
        let to_replace = nodes_to_replace(&resolved);

        assert_eq!(to_replace.len(), 1);
        assert_eq!(to_replace[0], existing_id);
    }

    #[test]
    fn nodes_to_replace_with_keep_both_replaces_nothing() {
        let new_node = make_node("person", serde_json::json!({"age": 30}));
        let existing_node = make_node("person", serde_json::json!({"age": 25}));

        let report = ConflictReport {
            new_node,
            existing_node,
            conflict_type: ConflictType::ValueConflict,
            confidence: 0.9,
        };

        let resolved = vec![(report, ConflictResolution::KeepBoth)];
        let to_replace = nodes_to_replace(&resolved);

        assert!(
            to_replace.is_empty(),
            "keep_both should not replace anything"
        );
    }

    #[test]
    fn conflict_type_equality() {
        assert_eq!(
            ConflictType::DirectContradiction,
            ConflictType::DirectContradiction
        );
        assert_ne!(
            ConflictType::DirectContradiction,
            ConflictType::ValueConflict
        );
        assert_ne!(ConflictType::ValueConflict, ConflictType::TemporalConflict);
    }

    #[test]
    fn conflict_resolution_equality() {
        assert_eq!(ConflictResolution::Escalate, ConflictResolution::Escalate);
        assert_ne!(
            ConflictResolution::Escalate,
            ConflictResolution::KeepExisting
        );
        assert_ne!(
            ConflictResolution::KeepExisting,
            ConflictResolution::ReplaceWithNew
        );
        assert_ne!(
            ConflictResolution::ReplaceWithNew,
            ConflictResolution::KeepBoth
        );
    }
}
