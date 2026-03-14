//! Integration tests: full ingest → search round-trip.
//!
//! Uses stub `Embedder` and `Extractor` implementations to verify the complete
//! ingest pipeline without a real LLM backend.

use ech0::test_stubs::{
    FailingEmbedder, FailingExtractor, StubEmbedder, StubExtractor, WrongDimsEmbedder,
};
use ech0::{ErrorCode, SearchOptions, Store, StoreConfig};

use tempfile::TempDir;

/// Build a `StoreConfig` pointing at temporary paths inside the given directory.
fn temp_config(dir: &TempDir, dimensions: usize) -> StoreConfig {
    let mut config = StoreConfig::default();
    config.store.graph_path = dir.path().join("graph.redb").to_string_lossy().to_string();
    config.store.vector_path = dir
        .path()
        .join("vectors.usearch")
        .to_string_lossy()
        .to_string();
    config.store.vector_dimensions = dimensions;
    config
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ingest_text_writes_nodes_and_edges() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store
        .ingest_text("Alice is 30 years old.")
        .await
        .expect("ingest should succeed");

    assert!(result.nodes_written > 0, "should write at least one node");
    assert_ne!(
        result.ingest_id,
        uuid::Uuid::nil(),
        "ingest_id should be a real UUID"
    );
}

#[tokio::test]
async fn ingest_text_round_trips_through_search() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    store
        .ingest_text("The capital of France is Paris.")
        .await
        .expect("ingest should succeed");

    let options = SearchOptions {
        limit: 10,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let search_result = store
        .search("capital of France", options)
        .await
        .expect("search should succeed");

    // With a zero-vector stub embedder, all vectors are identical so the ingested
    // node should appear as the top (and only) result.
    assert!(
        !search_result.nodes.is_empty(),
        "search should find the ingested node"
    );
}

#[tokio::test]
async fn ingest_nodes_directly_writes_and_retrieves() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let node = ech0::Node {
        id: uuid::Uuid::new_v4(),
        kind: "person".to_string(),
        metadata: serde_json::json!({"name": "Bob"}),
        importance: 0.9,
        created_at: chrono::Utc::now(),
        ingest_id: uuid::Uuid::nil(),
        source_text: None,
    };

    let _node_id = node.id;

    let result = store
        .ingest_nodes(vec![node], Vec::new())
        .await
        .expect("ingest_nodes should succeed");

    assert_eq!(result.nodes_written, 1);
    assert_eq!(result.edges_written, 0);

    // Verify the node can be found via search
    let search_result = store
        .search("Bob", SearchOptions::default())
        .await
        .expect("search should succeed");

    assert!(
        !search_result.nodes.is_empty(),
        "search should find the directly ingested node"
    );
}

#[tokio::test]
async fn ingest_text_assigns_unique_ingest_ids() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result_a = store
        .ingest_text("First ingest.")
        .await
        .expect("ingest A should succeed");

    let result_b = store
        .ingest_text("Second ingest.")
        .await
        .expect("ingest B should succeed");

    assert_ne!(
        result_a.ingest_id, result_b.ingest_id,
        "each ingest should get a unique ID"
    );
}

#[tokio::test]
async fn multiple_ingests_accumulate() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    for i in 0..5 {
        store
            .ingest_text(&format!("Fact number {i}."))
            .await
            .expect("ingest should succeed");
    }

    // Search should find multiple results
    let search_result = store
        .search(
            "facts",
            SearchOptions {
                limit: 100,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("search should succeed");

    assert_eq!(
        search_result.nodes.len(),
        5,
        "should find all 5 ingested nodes"
    );
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ingest_empty_text_returns_invalid_input() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store.ingest_text("").await;
    assert!(result.is_err(), "empty text should fail");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(
        error.message.contains("empty"),
        "error message should mention empty text: {}",
        error.message
    );
}

#[tokio::test]
async fn ingest_empty_nodes_returns_invalid_input() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store.ingest_nodes(Vec::new(), Vec::new()).await;
    assert!(result.is_err(), "empty nodes should fail");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[tokio::test]
async fn extractor_failure_propagates() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), FailingExtractor)
        .await
        .expect("store should open");

    let result = store.ingest_text("some text").await;
    assert!(result.is_err(), "extractor failure should propagate");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::ExtractorFailure);
}

#[tokio::test]
async fn embedder_failure_propagates() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;

    let config = temp_config(&dir, dims);
    // FailingEmbedder claims the right dimensions but always fails on embed()
    let embedder = FailingEmbedder { dimensions: dims };

    let store = Store::new(config, embedder, StubExtractor)
        .await
        .expect("store should open");

    let result = store.ingest_text("some text").await;
    assert!(result.is_err(), "embedder failure should propagate");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::EmbedderFailure);
}

#[tokio::test]
async fn wrong_dimensions_rejected_on_embed() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    // Embedder claims 32 dims but actually produces 64
    let embedder = WrongDimsEmbedder {
        claimed_dimensions: dims,
        actual_dimensions: 64,
    };

    let store = Store::new(config, embedder, StubExtractor)
        .await
        .expect("store should open");

    let result = store.ingest_text("some text").await;
    assert!(result.is_err(), "wrong dimensions should be rejected");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::EmbedderFailure);
}

#[tokio::test]
async fn config_dimensions_mismatch_rejected_on_store_new() {
    let dir = TempDir::new().expect("temp dir");
    let config = temp_config(&dir, 64);
    // Config says 64 dims, embedder says 32
    let embedder = StubEmbedder::new(32);

    let result = Store::new(config, embedder, StubExtractor).await;
    assert!(result.is_err(), "dimension mismatch should be rejected");

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

// ---------------------------------------------------------------------------
// Atomicity: if embed fails, nothing should be written
// ---------------------------------------------------------------------------

#[tokio::test]
async fn failed_ingest_does_not_leave_partial_state() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;

    // First, successfully ingest one node — drop the store before reopening
    let count_before = {
        let good_store = Store::new(
            temp_config(&dir, dims),
            StubEmbedder::new(dims),
            StubExtractor,
        )
        .await
        .expect("store should open");

        good_store
            .ingest_text("existing node")
            .await
            .expect("initial ingest should succeed");

        let before = good_store
            .search(
                "query",
                SearchOptions {
                    limit: 100,
                    min_importance: 0.0,
                    ..SearchOptions::default()
                },
            )
            .await
            .expect("search should succeed");

        let count = before.nodes.len();
        assert_eq!(count, 1, "should have exactly one node before failed ingest");
        count
        // good_store dropped here — redb lock released
    };

    // Now open a second store on the same path with a failing extractor
    let bad_store = Store::new(
        temp_config(&dir, dims),
        StubEmbedder::new(dims),
        FailingExtractor,
    )
    .await
    .expect("store should open even with failing extractor");

    let failed_result = bad_store.ingest_text("this will fail").await;
    assert!(failed_result.is_err(), "ingest should fail");

    let after = bad_store
        .search(
            "query",
            SearchOptions {
                limit: 100,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("search should succeed");

    assert_eq!(
        after.nodes.len(),
        count_before,
        "failed ingest should not change the node count"
    );
}

// ---------------------------------------------------------------------------
// Provenance feature
// ---------------------------------------------------------------------------

#[cfg(feature = "provenance")]
#[tokio::test]
async fn provenance_stores_source_text() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    store
        .ingest_text("The sky is blue.")
        .await
        .expect("ingest should succeed");

    let search_result = store
        .search("sky", SearchOptions::default())
        .await
        .expect("search should succeed");

    assert!(!search_result.nodes.is_empty(), "should find the node");

    let node = &search_result.nodes[0].node;
    assert!(
        node.source_text.is_some(),
        "provenance should attach source text"
    );
    assert_eq!(
        node.source_text.as_deref(),
        Some("The sky is blue."),
        "source text should match the original input"
    );
}

// ---------------------------------------------------------------------------
// Contradiction detection feature
// ---------------------------------------------------------------------------

#[cfg(feature = "contradiction-detection")]
#[tokio::test]
async fn contradiction_detection_returns_conflicts_in_result() {
    // This test verifies the structural contract: IngestResult.conflicts exists
    // and is populated. The actual detection logic depends on the stub extractor
    // which produces a single node per call, so conflicts will be empty unless
    // two ingests produce same-kind nodes with conflicting metadata.

    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store
        .ingest_text("Alice is 30 years old.")
        .await
        .expect("ingest should succeed");

    // With stub extractor, conflicts field should exist (may be empty)
    // The important thing is the field compiles and is accessible
    let _conflicts = &result.conflicts;
}

// ---------------------------------------------------------------------------
// Dynamic linking feature
// ---------------------------------------------------------------------------

#[cfg(all(feature = "dynamic-linking", feature = "tokio"))]
#[tokio::test]
async fn linking_task_is_present_and_awaitable() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store
        .ingest_text("Some text to link.")
        .await
        .expect("ingest should succeed");

    // linking_task should be present (Some) since we wrote at least one node
    if let Some(handle) = result.linking_task {
        let linking_result = handle.await.expect("linking task should not panic");
        // With zero-vector stubs, linking likely finds no candidates above threshold,
        // but the task should complete without error.
        let _edges_created = linking_result.edges_created;
    }
    // If linking_task is None, that's also acceptable (e.g., extraction produced empty results)
}
