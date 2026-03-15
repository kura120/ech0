//! Provenance metadata helpers for ech0.
//!
//! When the `provenance` feature is enabled, every node stores its original source text
//! alongside the standard metadata (timestamp, ingest ID). This module provides helpers
//! for attaching and querying provenance information.
//!
//! Provenance is never secret data — it is queryable metadata that helps callers understand
//! why a memory exists and how confident to be in it.

use uuid::Uuid;

use crate::error::EchoError;
use crate::schema::{Edge, Node};

// ---------------------------------------------------------------------------
// Provenance attachment
// ---------------------------------------------------------------------------

/// Attach provenance source text to a batch of nodes.
///
/// Called during `ingest_text()` before writing to the graph layer. When the `provenance`
/// feature is enabled, each node's `source_text` field is set to the original input text
/// that produced it. When the feature is disabled, this function is not compiled.
///
/// # Arguments
///
/// * `nodes` — mutable slice of nodes to annotate.
/// * `source_text` — the original text that was passed to `ingest_text()`.
pub fn attach_source_text(nodes: &mut [Node], source_text: &str) {
    for node in nodes.iter_mut() {
        node.source_text = Some(source_text.to_string());
    }
}

/// Strip provenance source text from a batch of nodes.
///
/// Useful when callers want to export or transmit nodes without including
/// potentially sensitive source text.
pub fn strip_source_text(nodes: &mut [Node]) {
    for node in nodes.iter_mut() {
        node.source_text = None;
    }
}

// ---------------------------------------------------------------------------
// Provenance queries
// ---------------------------------------------------------------------------

/// Filter nodes to only those that have provenance source text attached.
pub fn nodes_with_provenance(nodes: &[Node]) -> Vec<&Node> {
    nodes
        .iter()
        .filter(|node| node.source_text.is_some())
        .collect()
}

/// Filter nodes to only those missing provenance source text.
///
/// Useful for identifying nodes that were ingested via `ingest_nodes()` (which does
/// not automatically attach source text) or nodes ingested before the `provenance`
/// feature was enabled.
pub fn nodes_without_provenance(nodes: &[Node]) -> Vec<&Node> {
    nodes
        .iter()
        .filter(|node| node.source_text.is_none())
        .collect()
}

/// Group nodes by their ingest ID.
///
/// Returns a map from ingest ID to the list of nodes created in that ingest operation.
/// This is useful for understanding which nodes were created together and tracing back
/// to the original ingest call.
pub fn group_by_ingest(nodes: &[Node]) -> std::collections::HashMap<Uuid, Vec<&Node>> {
    let mut groups: std::collections::HashMap<Uuid, Vec<&Node>> = std::collections::HashMap::new();
    for node in nodes {
        groups.entry(node.ingest_id).or_default().push(node);
    }
    groups
}

/// Group edges by their ingest ID.
pub fn group_edges_by_ingest(edges: &[Edge]) -> std::collections::HashMap<Uuid, Vec<&Edge>> {
    let mut groups: std::collections::HashMap<Uuid, Vec<&Edge>> = std::collections::HashMap::new();
    for edge in edges {
        groups.entry(edge.ingest_id).or_default().push(edge);
    }
    groups
}

/// Validate that all nodes in the batch have provenance source text.
///
/// Returns an error listing the IDs of nodes that are missing source text.
/// Useful as a pre-write check when provenance completeness is required.
pub fn validate_provenance_complete(nodes: &[Node]) -> Result<(), EchoError> {
    let missing: Vec<Uuid> = nodes
        .iter()
        .filter(|node| node.source_text.is_none())
        .map(|node| node.id)
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(EchoError::invalid_input(format!(
            "{} node(s) missing provenance source text: {}",
            missing.len(),
            missing
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

/// Extract unique source texts from a set of nodes.
///
/// Returns deduplicated source texts in no guaranteed order. Useful for understanding
/// the distinct inputs that contributed to a set of search results.
pub fn unique_source_texts(nodes: &[Node]) -> Vec<&str> {
    let mut seen = std::collections::HashSet::new();
    let mut texts = Vec::new();

    for node in nodes {
        if let Some(ref text) = node.source_text
            && seen.insert(text.as_str())
        {
            texts.push(text.as_str());
        }
    }

    texts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_node_with_source(source: Option<&str>) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: "test".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: source.map(|s| s.to_string()),
        }
    }

    fn make_node_with_ingest_id(ingest_id: Uuid) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: "test".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id,
            source_text: None,
        }
    }

    fn make_edge_with_ingest_id(ingest_id: Uuid) -> Edge {
        Edge {
            source: Uuid::new_v4(),
            target: Uuid::new_v4(),
            relation: "related_to".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.5,
            created_at: Utc::now(),
            ingest_id,
        }
    }

    #[test]
    fn attach_source_text_sets_all_nodes() {
        let mut nodes = vec![
            make_node_with_source(None),
            make_node_with_source(None),
            make_node_with_source(None),
        ];

        assert!(nodes.iter().all(|n| n.source_text.is_none()));

        attach_source_text(&mut nodes, "Alice is 30 years old.");

        for node in &nodes {
            assert_eq!(node.source_text.as_deref(), Some("Alice is 30 years old."));
        }
    }

    #[test]
    fn attach_source_text_overwrites_existing() {
        let mut nodes = vec![make_node_with_source(Some("old text"))];

        attach_source_text(&mut nodes, "new text");

        assert_eq!(nodes[0].source_text.as_deref(), Some("new text"));
    }

    #[test]
    fn attach_source_text_handles_empty_slice() {
        let mut nodes: Vec<Node> = Vec::new();
        // Should not panic on empty input
        attach_source_text(&mut nodes, "some text");
        assert!(nodes.is_empty());
    }

    #[test]
    fn strip_source_text_clears_all() {
        let mut nodes = vec![
            make_node_with_source(Some("text one")),
            make_node_with_source(Some("text two")),
            make_node_with_source(None),
        ];

        strip_source_text(&mut nodes);

        for node in &nodes {
            assert!(node.source_text.is_none());
        }
    }

    #[test]
    fn nodes_with_provenance_filters_correctly() {
        let nodes = vec![
            make_node_with_source(Some("has source")),
            make_node_with_source(None),
            make_node_with_source(Some("also has source")),
        ];

        let with = nodes_with_provenance(&nodes);
        assert_eq!(with.len(), 2);
        assert!(with.iter().all(|n| n.source_text.is_some()));
    }

    #[test]
    fn nodes_without_provenance_filters_correctly() {
        let nodes = vec![
            make_node_with_source(Some("has source")),
            make_node_with_source(None),
            make_node_with_source(None),
        ];

        let without = nodes_without_provenance(&nodes);
        assert_eq!(without.len(), 2);
        assert!(without.iter().all(|n| n.source_text.is_none()));
    }

    #[test]
    fn group_by_ingest_groups_correctly() {
        let ingest_a = Uuid::new_v4();
        let ingest_b = Uuid::new_v4();

        let nodes = vec![
            make_node_with_ingest_id(ingest_a),
            make_node_with_ingest_id(ingest_a),
            make_node_with_ingest_id(ingest_b),
        ];

        let groups = group_by_ingest(&nodes);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&ingest_a].len(), 2);
        assert_eq!(groups[&ingest_b].len(), 1);
    }

    #[test]
    fn group_edges_by_ingest_groups_correctly() {
        let ingest_a = Uuid::new_v4();
        let ingest_b = Uuid::new_v4();

        let edges = vec![
            make_edge_with_ingest_id(ingest_a),
            make_edge_with_ingest_id(ingest_b),
            make_edge_with_ingest_id(ingest_b),
        ];

        let groups = group_edges_by_ingest(&edges);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&ingest_a].len(), 1);
        assert_eq!(groups[&ingest_b].len(), 2);
    }

    #[test]
    fn validate_provenance_complete_succeeds_when_all_present() {
        let nodes = vec![
            make_node_with_source(Some("text a")),
            make_node_with_source(Some("text b")),
        ];

        let result = validate_provenance_complete(&nodes);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_provenance_complete_fails_when_some_missing() {
        let nodes = vec![
            make_node_with_source(Some("text a")),
            make_node_with_source(None),
        ];

        let result = validate_provenance_complete(&nodes);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error.code, crate::error::ErrorCode::InvalidInput));
        assert!(
            error.message.contains("1 node(s) missing"),
            "error message should indicate count: {}",
            error.message
        );
    }

    #[test]
    fn validate_provenance_complete_succeeds_on_empty() {
        let nodes: Vec<Node> = Vec::new();
        let result = validate_provenance_complete(&nodes);
        assert!(result.is_ok());
    }

    #[test]
    fn unique_source_texts_deduplicates() {
        let nodes = vec![
            make_node_with_source(Some("same text")),
            make_node_with_source(Some("same text")),
            make_node_with_source(Some("different text")),
            make_node_with_source(None),
        ];

        let texts = unique_source_texts(&nodes);
        assert_eq!(texts.len(), 2);
        assert!(texts.contains(&"same text"));
        assert!(texts.contains(&"different text"));
    }

    #[test]
    fn unique_source_texts_empty_when_no_provenance() {
        let nodes = vec![make_node_with_source(None), make_node_with_source(None)];

        let texts = unique_source_texts(&nodes);
        assert!(texts.is_empty());
    }

    #[test]
    fn unique_source_texts_handles_empty_input() {
        let nodes: Vec<Node> = Vec::new();
        let texts = unique_source_texts(&nodes);
        assert!(texts.is_empty());
    }
}
