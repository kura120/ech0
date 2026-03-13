//! Test stub implementations of `Embedder` and `Extractor`.
//!
//! **Not part of the public API.** These stubs exist for integration testing only.
//! They are gated behind the `_test-helpers` feature which is enabled automatically
//! for dev builds via `[dev-dependencies]`.
//!
//! - `StubEmbedder` — returns a fixed-dimension zero vector for any input.
//! - `StubExtractor` — returns a hardcoded single "fact" node with no edges.
//! - `MultiNodeStubExtractor` — returns two "person" nodes with a "knows" edge.
//! - `FailingExtractor` — always returns `EchoError::ExtractorFailure`.
//! - `FailingEmbedder` — always returns `EchoError::EmbedderFailure`.
//! - `WrongDimsEmbedder` — claims one dimensionality but produces another.

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::error::EchoError;
use crate::schema::{Edge, Node};
use crate::traits::{Embedder, ExtractionResult, Extractor};

// ---------------------------------------------------------------------------
// StubEmbedder
// ---------------------------------------------------------------------------

/// Stub embedder that returns a fixed-dimension zero vector for any input.
///
/// For testing only — never shipped as a library default.
pub struct StubEmbedder {
    pub dimensions: usize,
}

impl StubEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait]
impl Embedder for StubEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EchoError> {
        Ok(vec![0.0f32; self.dimensions])
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ---------------------------------------------------------------------------
// StubExtractor
// ---------------------------------------------------------------------------

/// Stub extractor that returns a hardcoded single "fact" node with no edges.
///
/// The node's metadata contains the input text under the `"text"` key, so
/// different inputs produce nodes with different metadata (useful for conflict
/// detection testing).
///
/// For testing only — never shipped as a library default.
pub struct StubExtractor;

#[async_trait]
impl Extractor for StubExtractor {
    async fn extract(&self, text: &str) -> Result<ExtractionResult, EchoError> {
        let node = Node {
            id: Uuid::new_v4(),
            kind: "fact".to_string(),
            metadata: serde_json::json!({"text": text}),
            importance: 1.0,
            created_at: Utc::now(),
            ingest_id: Uuid::nil(), // will be overwritten by Store
            source_text: None,      // will be set by provenance feature if enabled
        };

        Ok(ExtractionResult {
            nodes: vec![node],
            edges: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// MultiNodeStubExtractor
// ---------------------------------------------------------------------------

/// Stub extractor that returns two "person" nodes (Alice, Bob) with a "knows" edge.
///
/// Useful for testing graph traversal, edge storage, and linking scenarios
/// that require richer graph structure than a single node.
///
/// For testing only — never shipped as a library default.
pub struct MultiNodeStubExtractor;

#[async_trait]
impl Extractor for MultiNodeStubExtractor {
    async fn extract(&self, _text: &str) -> Result<ExtractionResult, EchoError> {
        let ingest_id = Uuid::nil();
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

        Ok(ExtractionResult {
            nodes: vec![node_a, node_b],
            edges: vec![edge],
        })
    }
}

// ---------------------------------------------------------------------------
// FailingExtractor
// ---------------------------------------------------------------------------

/// Stub extractor that always fails with `EchoError::ExtractorFailure`.
///
/// For testing error propagation through the ingest pipeline.
pub struct FailingExtractor;

#[async_trait]
impl Extractor for FailingExtractor {
    async fn extract(&self, _text: &str) -> Result<ExtractionResult, EchoError> {
        Err(EchoError::extractor_failure(
            "stub extractor intentionally failed",
        ))
    }
}

// ---------------------------------------------------------------------------
// FailingEmbedder
// ---------------------------------------------------------------------------

/// Stub embedder that always fails with `EchoError::EmbedderFailure`.
///
/// Claims the correct dimensionality so it passes `Store::new()` validation,
/// but always fails on `embed()`. For testing error propagation.
pub struct FailingEmbedder {
    pub dimensions: usize,
}

#[async_trait]
impl Embedder for FailingEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EchoError> {
        Err(EchoError::embedder_failure(
            "stub embedder intentionally failed",
        ))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ---------------------------------------------------------------------------
// WrongDimsEmbedder
// ---------------------------------------------------------------------------

/// Stub embedder that claims one dimensionality but produces vectors of another.
///
/// `claimed_dimensions` is what `dimensions()` returns (used for `Store::new()` validation).
/// `actual_dimensions` is what `embed()` actually produces.
///
/// For testing dimension-mismatch validation in the ingest pipeline.
pub struct WrongDimsEmbedder {
    pub claimed_dimensions: usize,
    pub actual_dimensions: usize,
}

#[async_trait]
impl Embedder for WrongDimsEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EchoError> {
        Ok(vec![0.1f32; self.actual_dimensions])
    }

    fn dimensions(&self) -> usize {
        self.claimed_dimensions
    }
}
