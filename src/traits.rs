//! Caller-provided trait abstractions for embedding and extraction.
//!
//! ech0 provides no default implementations of these traits. The caller wires in
//! their own LLM backend — llama.cpp, OpenAI, Anthropic, or anything else.
//! Test stubs live in `tests/` only.

use async_trait::async_trait;

use crate::error::EchoError;
use crate::schema::{Edge, Node};

/// Produces vector embeddings from text.
///
/// The caller must ensure that `dimensions()` returns the correct dimensionality
/// for all vectors produced by `embed()`. ech0 validates this at ingest time and
/// returns `EchoError::InvalidInput` on mismatch.
///
/// # Contract
///
/// - `embed()` must return a vector of exactly `dimensions()` elements.
/// - `embed()` must not silently fail — all errors must be propagated via `EchoError`.
/// - Implementations must be `Send + Sync` for use across async tasks.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Produce a vector embedding for the given text.
    ///
    /// Returns a `Vec<f32>` of length `self.dimensions()`. Returns
    /// `EchoError` with code `EmbedderFailure` on any failure.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EchoError>;

    /// The fixed dimensionality of vectors produced by this embedder.
    /// Must match `StoreConfig.store.vector_dimensions`.
    fn dimensions(&self) -> usize;
}

/// Extracts structured knowledge (nodes and edges) from raw text.
///
/// The caller decides how extraction works — LLM-based, rule-based, or hybrid.
/// ech0 calls this trait during `ingest_text()` to convert free text into graph
/// structure before storage.
///
/// # Contract
///
/// - `extract()` must not silently fail — all errors must be propagated via `EchoError`.
/// - Returned `Node`s should have `id` and `ingest_id` set to placeholder values;
///   ech0 overwrites them with the canonical ingest ID during the write path.
/// - Returned `Edge`s must reference `source` and `target` IDs that exist in the
///   returned `ExtractionResult.nodes` (or in the existing graph, for linking edges).
/// - Implementations must be `Send + Sync` for use across async tasks.
#[async_trait]
pub trait Extractor: Send + Sync {
    /// Extract structured nodes and edges from the given text.
    ///
    /// Returns an `ExtractionResult` containing the extracted graph fragment.
    /// Returns `EchoError` with code `ExtractorFailure` on any failure.
    async fn extract(&self, text: &str) -> Result<ExtractionResult, EchoError>;
}

/// The output of an `Extractor::extract()` call — a graph fragment ready for ingest.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Nodes extracted from the input text.
    pub nodes: Vec<Node>,

    /// Edges extracted from the input text, connecting the returned nodes.
    pub edges: Vec<Edge>,
}

impl ExtractionResult {
    /// Returns `true` if this extraction produced no nodes or edges.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// Total number of graph elements (nodes + edges) in this extraction.
    pub fn len(&self) -> usize {
        self.nodes.len() + self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn extraction_result_is_empty_when_no_elements() {
        let result = ExtractionResult {
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn extraction_result_not_empty_with_nodes() {
        let node = Node {
            id: Uuid::new_v4(),
            kind: "test".to_string(),
            metadata: serde_json::json!({}),
            importance: 1.0,
            created_at: Utc::now(),
            ingest_id: Uuid::new_v4(),
            source_text: None,
        };
        let result = ExtractionResult {
            nodes: vec![node],
            edges: Vec::new(),
        };
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn extraction_result_len_counts_both() {
        let ingest_id = Uuid::new_v4();
        let now = Utc::now();
        let node_a = Node {
            id: Uuid::new_v4(),
            kind: "person".to_string(),
            metadata: serde_json::json!({"name": "Alice"}),
            importance: 0.9,
            created_at: now,
            ingest_id,
            source_text: None,
        };
        let node_b = Node {
            id: Uuid::new_v4(),
            kind: "person".to_string(),
            metadata: serde_json::json!({"name": "Bob"}),
            importance: 0.8,
            created_at: now,
            ingest_id,
            source_text: None,
        };
        let edge = Edge {
            source: node_a.id,
            target: node_b.id,
            relation: "knows".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.7,
            created_at: now,
            ingest_id,
        };
        let result = ExtractionResult {
            nodes: vec![node_a, node_b],
            edges: vec![edge],
        };
        assert!(!result.is_empty());
        assert_eq!(result.len(), 3);
    }
}
