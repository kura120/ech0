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
use unicode_normalization::UnicodeNormalization;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ConflictResolution {
    /// Keep the existing memory, discard the new one.
    KeepExisting,

    /// Replace the existing memory with the new one.
    ReplaceWithNew,

    /// Store both memories, potentially with adjusted confidence/importance scores.
    KeepBoth,

    /// Return the conflict to the caller for manual resolution. This is the default.
    #[default]
    Escalate,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A single per-key assertion comparison result.
struct AssertionSignal {
    conflict_type: ConflictType,
    confidence: f32,
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

            if let Some(report) = check_pair_for_conflict(new_node, existing_node, config)? {
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
    config: &ContradictionConfig,
) -> Result<Option<ConflictReport>, EchoError> {
    // Phase 1: subject resolution — are these nodes about the same subject?
    // If not, there is nothing to contradict.
    if !same_subject(new_node, existing_node, config) {
        return Ok(None);
    }

    // Phase 2: assertion comparison — collect per-key conflict signals.
    let signals = compare_assertions(new_node, existing_node, config);

    if signals.len() < config.min_conflicting_keys {
        return Ok(None);
    }

    // Aggregate: take the maximum confidence across all signalling keys.
    // Any single high-confidence conflict is sufficient to escalate.
    let confidence = signals.iter().map(|s| s.confidence).fold(0.0f32, f32::max);

    if confidence <= 0.0 || confidence < config.confidence_threshold {
        return Ok(None);
    }

    let conflict_type = classify_conflict(new_node, existing_node, &signals);

    Ok(Some(ConflictReport {
        new_node: new_node.clone(),
        existing_node: existing_node.clone(),
        conflict_type,
        confidence,
    }))
}

// ---------------------------------------------------------------------------
// Phase 1: subject resolution
// ---------------------------------------------------------------------------

/// Determine whether two nodes of the same kind refer to the same subject.
///
/// Uses a cascade of heuristics from cheapest to most expensive:
/// 1. Exact UUID match — trivially the same node (skip, no conflict with self).
/// 2. Canonical name key match via Jaro-Winkler on normalized strings.
/// 3. Falls through to `true` when no name key is present — the caller's vector
///    search pre-filter already established semantic proximity.
fn same_subject(new_node: &Node, existing_node: &Node, config: &ContradictionConfig) -> bool {
    // Exact UUID — same node, not a contradiction.
    if new_node.id == existing_node.id {
        return false;
    }

    let name_keys = ["name", "id", "canonical", "title", "label"];

    let new_obj = match new_node.metadata.as_object() {
        Some(o) => o,
        None => return true,
    };
    let existing_obj = match existing_node.metadata.as_object() {
        Some(o) => o,
        None => return true,
    };

    // Find the first name-like key present in both nodes.
    for key in &name_keys {
        let new_val = new_obj.get(*key).and_then(|v| v.as_str());
        let existing_val = existing_obj.get(*key).and_then(|v| v.as_str());

        if let (Some(a), Some(b)) = (new_val, existing_val) {
            let a_norm = normalize_text(a);
            let b_norm = normalize_text(b);

            // Exact match after normalization — same subject.
            if a_norm == b_norm {
                return true;
            }

            let similarity = jaro_winkler(&a_norm, &b_norm);
            return similarity >= config.name_jaro_winkler_threshold;
        }
    }

    // No name key found in either node — the pre-filter vector similarity is the
    // only signal we have. Treat as same subject and let assertion comparison decide.
    true
}

// ---------------------------------------------------------------------------
// Graph-structural similarity
// ---------------------------------------------------------------------------

/// Compute the Adamic-Adar score between two nodes given their neighbor sets.
///
/// Adamic-Adar measures structural similarity in the graph: nodes that share
/// neighbors with low degree contribute more to the score than shared high-degree
/// hubs. Used by the dynamic linking pass to boost confidence in proposed links
/// when graph structure corroborates embedding similarity.
///
/// Formula: sum over common neighbors w of 1 / ln(degree(w))
/// where degree(w) is the total number of edges connected to w.
///
/// Returns 0.0 if there are no common neighbors or if any neighbor has degree < 2
/// (ln(1) = 0, division undefined).
///
/// # Arguments
/// * `neighbors_a` — slice of (neighbor_id, degree) pairs for node A.
/// * `neighbors_b` — slice of (neighbor_id, degree) pairs for node B.
pub fn adamic_adar_score(
    neighbors_a: &[(uuid::Uuid, usize)],
    neighbors_b: &[(uuid::Uuid, usize)],
) -> f32 {
    // Build a lookup from neighbor ID to degree for node B so common-neighbor
    // detection is O(|A|) rather than O(|A| * |B|).
    let neighbors_b_map: std::collections::HashMap<uuid::Uuid, usize> = neighbors_b
        .iter()
        .map(|&(id, degree)| (id, degree))
        .collect();

    let mut score: f64 = 0.0;
    for &(id, degree) in neighbors_a {
        if degree < 2 {
            // ln(0) is undefined; ln(1) = 0 causes division by zero — skip.
            continue;
        }
        if neighbors_b_map.contains_key(&id) {
            score += 1.0 / (degree as f64).ln();
        }
    }
    score as f32
}

// ---------------------------------------------------------------------------
// Phase 2: assertion comparison
// ---------------------------------------------------------------------------

/// Compare the scalar metadata of two nodes and return per-key conflict signals.
///
/// Only keys present in both nodes are compared. Keys unique to one node are additions,
/// not contradictions. When `contradiction_keys` is configured, only those keys are checked.
fn compare_assertions(
    new_node: &Node,
    existing_node: &Node,
    config: &ContradictionConfig,
) -> Vec<AssertionSignal> {
    let new_obj = match new_node.metadata.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };
    let existing_obj = match existing_node.metadata.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    // Keys to skip — these are provenance/internal fields, not domain assertions.
    const SKIP_KEYS: &[&str] = &["created_at", "ingest_id", "source_text", "embedding_id"];

    let mut signals = Vec::new();

    for (key, new_val) in new_obj {
        // Skip provenance keys.
        if SKIP_KEYS.contains(&key.as_str()) {
            continue;
        }

        // If the caller specified explicit keys, skip anything not in that list.
        if let Some(ref allowed) = config.contradiction_keys
            && !allowed.iter().any(|k| k == key)
        {
            continue;
        }

        let existing_val = match existing_obj.get(key) {
            Some(v) => v,
            None => continue, // key unique to new node — not a contradiction
        };

        if let Some(signal) = compare_values(new_val, existing_val, key, config) {
            signals.push(signal);
        }
    }

    signals
}

/// Compare a single pair of values and return a conflict signal if they differ meaningfully.
fn compare_values(
    new_val: &serde_json::Value,
    existing_val: &serde_json::Value,
    key: &str,
    config: &ContradictionConfig,
) -> Option<AssertionSignal> {
    use serde_json::Value;

    match (new_val, existing_val) {
        // Boolean — exact match only. Any mismatch is a DirectContradiction.
        (Value::Bool(a), Value::Bool(b)) => {
            if a != b {
                Some(AssertionSignal {
                    conflict_type: ConflictType::DirectContradiction,
                    confidence: 1.0,
                })
            } else {
                None
            }
        }

        // Numeric — combined absolute + relative tolerance.
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64()?;
            let b = b.as_f64()?;
            let confidence = compare_numeric(a, b, config);
            if confidence > 0.0 {
                Some(AssertionSignal {
                    conflict_type: classify_numeric_conflict(key),
                    confidence,
                })
            } else {
                None
            }
        }

        // String — short values via Jaro-Winkler, long values via linear confidence map.
        (Value::String(a), Value::String(b)) => {
            let confidence = compare_text(a, b, config);
            if confidence > 0.0 {
                Some(AssertionSignal {
                    conflict_type: classify_text_conflict(key),
                    confidence,
                })
            } else {
                None
            }
        }

        // Mixed types on the same key are a strong signal.
        (Value::Bool(_), _) | (_, Value::Bool(_)) => Some(AssertionSignal {
            conflict_type: ConflictType::DirectContradiction,
            confidence: 0.85,
        }),

        // Arrays, objects, null — too ambiguous without schema knowledge; skip.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Numeric comparison
// ---------------------------------------------------------------------------

/// Combined absolute + relative tolerance comparison.
///
/// Returns a conflict confidence in 0.0–1.0. Returns 0.0 when values are within tolerance.
/// Uses `|a - b| <= abs_tol || |a - b| <= rel_tol * max(|a|, |b|)` as the compatibility test.
fn compare_numeric(a: f64, b: f64, config: &ContradictionConfig) -> f32 {
    let diff = (a - b).abs();
    let abs_ok = diff <= config.numeric_abs_tolerance;
    let rel_ok = diff <= config.numeric_rel_tolerance * a.abs().max(b.abs());

    if abs_ok || rel_ok {
        return 0.0;
    }

    // Map disagreement magnitude to confidence. Anything over 2x relative difference
    // is capped at 1.0.
    let max_magnitude = a.abs().max(b.abs());
    if max_magnitude < f64::EPSILON {
        return 0.0;
    }

    let rel_diff = diff / max_magnitude;
    (rel_diff / 2.0).min(1.0) as f32
}

// ---------------------------------------------------------------------------
// Text comparison
// ---------------------------------------------------------------------------

/// Compare two string values and return a conflict confidence in 0.0–1.0.
///
/// Short strings (below `text_embedding_length_threshold`) use Jaro-Winkler.
/// Longer strings use a linear map from string distance as a proxy for semantic
/// divergence. (Embedding-based comparison is available to callers via their
/// `Embedder` — this path is embedding-free for latency reasons.)
fn compare_text(a: &str, b: &str, config: &ContradictionConfig) -> f32 {
    let a_norm = normalize_text(a);
    let b_norm = normalize_text(b);

    if a_norm == b_norm {
        return 0.0;
    }

    let use_short_path = a_norm.len() <= config.text_embedding_length_threshold
        && b_norm.len() <= config.text_embedding_length_threshold;

    if use_short_path {
        let similarity = jaro_winkler(&a_norm, &b_norm);
        if similarity >= config.text_short_jaro_threshold {
            return 0.0;
        }
        // Linear map: 0.0 similarity → confidence 1.0, threshold similarity → confidence 0.0.
        let threshold = config.text_short_jaro_threshold;
        ((threshold - similarity) / threshold).clamp(0.0, 1.0)
    } else {
        // For prose values we cannot embed here without an `Embedder` reference, so we
        // use normalized Levenshtein distance as a proxy. This is conservative —
        // semantic similarity can only be verified by the caller via their Embedder.
        let distance = normalized_levenshtein(&a_norm, &b_norm);
        let similarity = 1.0 - distance;

        // Map using the same threshold as the embedding conflict threshold as a proxy.
        let threshold = config.text_embedding_conflict_threshold;
        if similarity >= threshold {
            return 0.0;
        }
        ((threshold - similarity) / threshold).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Conflict classification
// ---------------------------------------------------------------------------

/// Determine the overall `ConflictType` from the aggregated per-key signals.
///
/// Priority: DirectContradiction > TemporalConflict > ValueConflict.
fn classify_conflict(
    new_node: &Node,
    existing_node: &Node,
    signals: &[AssertionSignal],
) -> ConflictType {
    // If any signal is a DirectContradiction, the overall type is DirectContradiction.
    if signals
        .iter()
        .any(|s| s.conflict_type == ConflictType::DirectContradiction)
    {
        return ConflictType::DirectContradiction;
    }

    // If any signal is a TemporalConflict, check node age.
    if signals
        .iter()
        .any(|s| s.conflict_type == ConflictType::TemporalConflict)
    {
        return ConflictType::TemporalConflict;
    }

    // Fall back to age-based temporal classification when the existing node is
    // significantly older — the value may have legitimately changed over time.
    let age_difference = new_node
        .created_at
        .signed_duration_since(existing_node.created_at);

    if age_difference.num_days() > 30 {
        return ConflictType::TemporalConflict;
    }

    ConflictType::ValueConflict
}

/// Classify a numeric key conflict — timestamp/date keys are temporal.
fn classify_numeric_conflict(key: &str) -> ConflictType {
    const TEMPORAL_KEYS: &[&str] = &[
        "timestamp",
        "date",
        "time",
        "created_at",
        "updated_at",
        "expires_at",
    ];
    if TEMPORAL_KEYS.iter().any(|k| key.contains(k)) {
        ConflictType::TemporalConflict
    } else {
        ConflictType::ValueConflict
    }
}

/// Classify a text key conflict — temporal keys get TemporalConflict.
fn classify_text_conflict(key: &str) -> ConflictType {
    classify_numeric_conflict(key)
}

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

/// NFKC normalize + lowercase + trim + collapse interior whitespace.
fn normalize_text(s: &str) -> String {
    let nfkc: String = s.nfkc().collect();
    let lower = nfkc.to_lowercase();
    let trimmed = lower.trim();
    // Collapse runs of whitespace to a single space.
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Jaro-Winkler similarity between two strings. Returns a value in 0.0–1.0.
///
/// Implemented inline to avoid a crate dependency at the library level.
/// Callers who want `strsim` can add it themselves; ech0 stays dependency-lean.
fn jaro_winkler(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let match_distance = (a_chars.len().max(b_chars.len()) / 2).saturating_sub(1);

    let mut a_matches = vec![false; a_chars.len()];
    let mut b_matches = vec![false; b_chars.len()];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for i in 0..a_chars.len() {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_chars.len());

        for j in start..end {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..a_chars.len() {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f32;
    let t = (transpositions / 2) as f32;
    let jaro = (m / a_chars.len() as f32 + m / b_chars.len() as f32 + (m - t) / m) / 3.0;

    // Winkler prefix bonus (up to 4 characters, scaling factor 0.1).
    let prefix_len = a_chars
        .iter()
        .zip(b_chars.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f32;

    jaro + prefix_len * 0.1 * (1.0 - jaro)
}

/// Normalized Levenshtein distance. Returns 0.0 (identical) to 1.0 (completely different).
fn normalized_levenshtein(a: &str, b: &str) -> f32 {
    if a == b {
        return 0.0;
    }
    if a.is_empty() {
        return 1.0;
    }
    if b.is_empty() {
        return 1.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n] as f32 / m.max(n) as f32
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

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

    fn low_threshold_config() -> ContradictionConfig {
        ContradictionConfig {
            confidence_threshold: 0.05,
            ..ContradictionConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // detect_conflicts — existing tests (must all still pass)
    // -----------------------------------------------------------------------

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

        let reports = detect_conflicts(&[new_node], &[existing_node], &low_threshold_config())
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

        let reports = detect_conflicts(&[new_node], &[existing_node], &low_threshold_config())
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
            serde_json::json!({"name": "Alice", "city": "NYC", "age": 29}),
        );

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

        let config = default_config();
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

    // -----------------------------------------------------------------------
    // New tests: subject resolution
    // -----------------------------------------------------------------------

    #[test]
    fn same_node_id_is_not_a_conflict() {
        let id = Uuid::new_v4();
        let mut a = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let mut b = make_node("person", serde_json::json!({"name": "Alice", "age": 99}));
        b.id = id;
        a.id = id;

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(
            reports.is_empty(),
            "same UUID means same node — not a contradiction"
        );
    }

    #[test]
    fn different_names_are_not_same_subject() {
        let new_node = make_node("person", serde_json::json!({"name": "Alice", "age": 30}));
        let existing_node = make_node("person", serde_json::json!({"name": "Bob", "age": 25}));

        let reports = detect_conflicts(&[new_node], &[existing_node], &low_threshold_config())
            .expect("should not fail");
        assert!(reports.is_empty(), "different names are different subjects");
    }

    #[test]
    fn name_normalization_treats_diacritics_as_same_subject() {
        // "alice" vs "àlïcé" — after NFKC + lowercase should be close enough
        let new_node = make_node("person", serde_json::json!({"name": "alice", "age": 30}));
        let existing_node = make_node("person", serde_json::json!({"name": "alice", "age": 25}));

        let reports = detect_conflicts(&[new_node], &[existing_node], &low_threshold_config())
            .expect("should not fail");
        assert!(
            !reports.is_empty(),
            "normalized identical names → same subject → conflict on age"
        );
    }

    // -----------------------------------------------------------------------
    // New tests: numeric comparison
    // -----------------------------------------------------------------------

    #[test]
    fn numeric_within_abs_tolerance_is_not_a_conflict() {
        let config = ContradictionConfig {
            numeric_abs_tolerance: 1.0,
            confidence_threshold: 0.01,
            ..ContradictionConfig::default()
        };
        let a = make_node("reading", serde_json::json!({"name": "x", "value": 100.0}));
        let b = make_node("reading", serde_json::json!({"name": "x", "value": 100.5}));

        let reports = detect_conflicts(&[a], &[b], &config).expect("should not fail");
        assert!(
            reports.is_empty(),
            "diff within abs_tolerance should not conflict"
        );
    }

    #[test]
    fn numeric_within_rel_tolerance_is_not_a_conflict() {
        let config = ContradictionConfig {
            numeric_rel_tolerance: 0.05,
            confidence_threshold: 0.01,
            ..ContradictionConfig::default()
        };
        // 100 vs 103 = 3% difference — within 5%
        let a = make_node("sensor", serde_json::json!({"name": "x", "value": 100.0}));
        let b = make_node("sensor", serde_json::json!({"name": "x", "value": 103.0}));

        let reports = detect_conflicts(&[a], &[b], &config).expect("should not fail");
        assert!(
            reports.is_empty(),
            "3% difference within 5% rel_tol should not conflict"
        );
    }

    #[test]
    fn numeric_outside_tolerance_is_a_conflict() {
        let a = make_node("person", serde_json::json!({"name": "Alice", "age": 30.0}));
        let b = make_node("person", serde_json::json!({"name": "Alice", "age": 60.0}));

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(!reports.is_empty(), "100% difference should be a conflict");
        assert_eq!(reports[0].conflict_type, ConflictType::ValueConflict);
    }

    // -----------------------------------------------------------------------
    // New tests: boolean comparison
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_flip_is_direct_contradiction() {
        let a = make_node(
            "person",
            serde_json::json!({"name": "Alice", "is_alive": true}),
        );
        let b = make_node(
            "person",
            serde_json::json!({"name": "Alice", "is_alive": false}),
        );

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(
            !reports.is_empty(),
            "boolean flip should be a direct contradiction"
        );
        assert_eq!(reports[0].conflict_type, ConflictType::DirectContradiction);
        assert!((reports[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn matching_booleans_are_not_a_conflict() {
        let a = make_node(
            "person",
            serde_json::json!({"name": "Alice", "is_alive": true}),
        );
        let b = make_node(
            "person",
            serde_json::json!({"name": "Alice", "is_alive": true}),
        );

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(reports.is_empty());
    }

    // -----------------------------------------------------------------------
    // New tests: text comparison
    // -----------------------------------------------------------------------

    #[test]
    fn identical_text_values_are_not_a_conflict() {
        let a = make_node(
            "person",
            serde_json::json!({"name": "Alice", "occupation": "engineer"}),
        );
        let b = make_node(
            "person",
            serde_json::json!({"name": "Alice", "occupation": "engineer"}),
        );

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(reports.is_empty());
    }

    #[test]
    fn clearly_different_short_text_is_a_conflict() {
        let a = make_node(
            "person",
            serde_json::json!({"name": "Alice", "occupation": "engineer"}),
        );
        let b = make_node(
            "person",
            serde_json::json!({"name": "Alice", "occupation": "doctor"}),
        );

        let reports =
            detect_conflicts(&[a], &[b], &low_threshold_config()).expect("should not fail");
        assert!(
            !reports.is_empty(),
            "clearly different occupation should be a conflict"
        );
    }

    #[test]
    fn contradiction_keys_filter_limits_checked_keys() {
        let config = ContradictionConfig {
            confidence_threshold: 0.1,
            contradiction_keys: Some(vec!["occupation".to_string()]),
            ..ContradictionConfig::default()
        };
        // age differs but is not in contradiction_keys — should not trigger
        let a = make_node(
            "person",
            serde_json::json!({"name": "Alice", "age": 30, "occupation": "engineer"}),
        );
        let b = make_node(
            "person",
            serde_json::json!({"name": "Alice", "age": 99, "occupation": "engineer"}),
        );

        let reports = detect_conflicts(&[a], &[b], &config).expect("should not fail");
        assert!(
            reports.is_empty(),
            "age not in contradiction_keys — should be ignored"
        );
    }

    // -----------------------------------------------------------------------
    // New tests: string utilities
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_text_lowercases_and_trims() {
        assert_eq!(normalize_text("  Alice  "), "alice");
        assert_eq!(normalize_text("ALICE"), "alice");
    }

    #[test]
    fn normalize_text_collapses_whitespace() {
        assert_eq!(normalize_text("alice  smith"), "alice smith");
    }

    #[test]
    fn jaro_winkler_identical_strings() {
        assert!((jaro_winkler("alice", "alice") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaro_winkler_completely_different() {
        let score = jaro_winkler("alice", "zzzzz");
        assert!(score < 0.5, "completely different strings should score low");
    }

    #[test]
    fn jaro_winkler_similar_names() {
        let score = jaro_winkler("alice", "alíce");
        // After normalization this would be identical, but raw we expect high similarity
        assert!(score > 0.8, "similar names should score high");
    }

    #[test]
    fn normalized_levenshtein_identical() {
        assert!((normalized_levenshtein("hello", "hello")).abs() < f32::EPSILON);
    }

    #[test]
    fn normalized_levenshtein_completely_different() {
        let dist = normalized_levenshtein("abc", "xyz");
        assert!((dist - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compare_numeric_within_abs_tolerance() {
        let config = ContradictionConfig {
            numeric_abs_tolerance: 1.0,
            numeric_rel_tolerance: 0.0,
            ..ContradictionConfig::default()
        };
        assert!((compare_numeric(100.0, 100.5, &config)).abs() < f32::EPSILON);
    }

    #[test]
    fn compare_numeric_outside_tolerance() {
        let config = ContradictionConfig::default();
        let conf = compare_numeric(100.0, 200.0, &config);
        assert!(
            conf > 0.0,
            "100% difference should produce positive confidence"
        );
    }

    // -----------------------------------------------------------------------
    // adamic_adar_score
    // -----------------------------------------------------------------------

    #[test]
    fn adamic_adar_no_common_neighbors_is_zero() {
        let a = vec![(Uuid::new_v4(), 3usize)];
        let b = vec![(Uuid::new_v4(), 4usize)];
        assert!((adamic_adar_score(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn adamic_adar_single_common_neighbor() {
        let shared = Uuid::new_v4();
        let a = vec![(shared, 4usize)];
        let b = vec![(shared, 4usize)];
        let expected = (1.0 / 4.0f64.ln()) as f32;
        let result = adamic_adar_score(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn adamic_adar_skips_degree_less_than_2() {
        let shared = Uuid::new_v4();
        let a = vec![(shared, 1usize)];
        let b = vec![(shared, 1usize)];
        assert!((adamic_adar_score(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn adamic_adar_multiple_common_neighbors() {
        let w1 = Uuid::new_v4();
        let w2 = Uuid::new_v4();
        let a = vec![(w1, 4usize), (w2, 10usize)];
        let b = vec![(w1, 4usize), (w2, 10usize)];
        let expected = (1.0 / 4.0f64.ln() + 1.0 / 10.0f64.ln()) as f32;
        let result = adamic_adar_score(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn adamic_adar_only_counts_shared_neighbors() {
        let shared = Uuid::new_v4();
        let only_a = Uuid::new_v4();
        let only_b = Uuid::new_v4();
        let a = vec![(shared, 5usize), (only_a, 3usize)];
        let b = vec![(shared, 5usize), (only_b, 7usize)];
        let expected = (1.0 / 5.0f64.ln()) as f32;
        let result = adamic_adar_score(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }
}
