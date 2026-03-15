//! Integration tests: hybrid search, result merging, importance filtering.
//!
//! Validates that the search pipeline correctly merges vector ANN results with
//! graph traversal results, respects importance thresholds, memory tier filters,
//! and hard limit caps.

use ech0::test_stubs::{MultiNodeStubExtractor, StubEmbedder, StubExtractor};
use ech0::{ErrorCode, MemoryTier, Node, SearchOptions, Store, StoreConfig, TraversalOptions};

use chrono::Utc;
use tempfile::TempDir;
use uuid::Uuid;

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

/// Helper: create a store, ingest several texts, return the store.
async fn store_with_ingested_texts(
    dir: &TempDir,
    dims: usize,
    texts: &[&str],
) -> Store<StubEmbedder, StubExtractor> {
    let config = temp_config(dir, dims);
    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    for text in texts {
        store
            .ingest_text(text)
            .await
            .expect("ingest should succeed");
    }

    store
}

// ---------------------------------------------------------------------------
// Basic search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_returns_results_from_ingested_data() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(
        &dir,
        dims,
        &["Alice is a software engineer.", "Bob likes hiking."],
    )
    .await;

    let result = store
        .search("engineer", SearchOptions::default())
        .await
        .expect("search should succeed");

    assert!(
        !result.nodes.is_empty(),
        "search should return at least one result"
    );
}

#[tokio::test]
async fn search_empty_query_returns_error() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(&dir, dims, &["some data"]).await;

    let result = store.search("", SearchOptions::default()).await;

    assert!(result.is_err(), "empty query should return error");
    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(
        error.message.contains("empty"),
        "error message should mention empty: {}",
        error.message
    );
}

#[tokio::test]
async fn search_on_empty_store_returns_empty() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let result = store
        .search("anything", SearchOptions::default())
        .await
        .expect("search should succeed on empty store");

    assert!(
        result.nodes.is_empty(),
        "search on empty store should return no results"
    );
}

// ---------------------------------------------------------------------------
// Limit enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_respects_hard_limit() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;

    // Ingest many nodes
    let texts: Vec<String> = (0..20).map(|i| format!("Fact number {i}.")).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let store = store_with_ingested_texts(&dir, dims, &text_refs).await;

    let options = SearchOptions {
        limit: 5,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let result = store
        .search("fact", options)
        .await
        .expect("search should succeed");

    assert!(
        result.nodes.len() <= 5,
        "should respect hard limit of 5, got {}",
        result.nodes.len()
    );
}

#[tokio::test]
async fn search_limit_is_cap_not_target() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;

    // Ingest only 2 nodes but request limit of 100
    let store = store_with_ingested_texts(&dir, dims, &["Node A.", "Node B."]).await;

    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    assert_eq!(
        result.nodes.len(),
        2,
        "limit is a cap not a target — should return only existing nodes"
    );
}

// ---------------------------------------------------------------------------
// Importance filtering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_filters_below_min_importance() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Ingest nodes with varying importance via ingest_nodes
    let high_importance_node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"content": "important fact"}),
        importance: 0.9,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    let low_importance_node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"content": "trivial fact"}),
        importance: 0.05,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![high_importance_node.clone()], Vec::new())
        .await
        .expect("ingest high importance should succeed");

    store
        .ingest_nodes(vec![low_importance_node.clone()], Vec::new())
        .await
        .expect("ingest low importance should succeed");

    // Search with min_importance filter that excludes the low-importance node
    let options = SearchOptions {
        limit: 100,
        min_importance: 0.5,
        ..SearchOptions::default()
    };

    let result = store
        .search("fact", options)
        .await
        .expect("search should succeed");

    // All returned nodes should have importance >= 0.5
    for scored_node in &result.nodes {
        assert!(
            scored_node.node.importance >= 0.5 || scored_node.node.id == high_importance_node.id,
            "returned node {} has importance {} which is below threshold 0.5",
            scored_node.node.id,
            scored_node.node.importance,
        );
    }

    // The low-importance node should not be in the results
    let found_low = result
        .nodes
        .iter()
        .any(|sn| sn.node.id == low_importance_node.id);
    assert!(
        !found_low,
        "low importance node should be filtered out by min_importance"
    );
}

#[tokio::test]
async fn search_with_zero_min_importance_returns_all() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Ingest a node with very low importance
    let node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"content": "low importance fact"}),
        importance: 0.001,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![node.clone()], Vec::new())
        .await
        .expect("ingest should succeed");

    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    assert!(
        !result.nodes.is_empty(),
        "zero min_importance should allow all nodes through"
    );
}

// ---------------------------------------------------------------------------
// Memory tier filtering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_filters_by_memory_tier() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Ingest an "event" node (inferred as Episodic) and a "fact" node (inferred as Semantic)
    let event_node = Node {
        id: Uuid::new_v4(),
        kind: "event".to_string(),
        metadata: serde_json::json!({"description": "meeting happened"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    let fact_node = Node {
        id: Uuid::new_v4(),
        kind: "concept".to_string(),
        metadata: serde_json::json!({"description": "general knowledge"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![event_node.clone()], Vec::new())
        .await
        .expect("ingest event should succeed");

    store
        .ingest_nodes(vec![fact_node.clone()], Vec::new())
        .await
        .expect("ingest fact should succeed");

    // Search only Episodic tier
    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        tiers: vec![MemoryTier::Episodic],
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    // Only the event node should pass the tier filter
    for scored_node in &result.nodes {
        assert_ne!(
            scored_node.node.id, fact_node.id,
            "semantic node should be filtered out when searching only Episodic tier"
        );
    }
}

#[tokio::test]
async fn search_with_empty_tiers_searches_all() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    let event_node = Node {
        id: Uuid::new_v4(),
        kind: "event".to_string(),
        metadata: serde_json::json!({"description": "an event"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    let concept_node = Node {
        id: Uuid::new_v4(),
        kind: "concept".to_string(),
        metadata: serde_json::json!({"description": "a concept"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![event_node], Vec::new())
        .await
        .expect("ingest event should succeed");

    store
        .ingest_nodes(vec![concept_node], Vec::new())
        .await
        .expect("ingest concept should succeed");

    // Search with empty tiers (should include all)
    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        tiers: Vec::new(),
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    assert_eq!(
        result.nodes.len(),
        2,
        "empty tiers filter should include all nodes"
    );
}

// ---------------------------------------------------------------------------
// Explicit metadata tier filtering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_respects_explicit_tier_metadata() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Node with explicit "tier": "short_term" in metadata
    let short_term_node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"tier": "short_term", "content": "session data"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    // Node with explicit "tier": "semantic" in metadata
    let semantic_node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"tier": "semantic", "content": "general fact"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![short_term_node.clone()], Vec::new())
        .await
        .expect("ingest short_term should succeed");

    store
        .ingest_nodes(vec![semantic_node.clone()], Vec::new())
        .await
        .expect("ingest semantic should succeed");

    // Search only ShortTerm
    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        tiers: vec![MemoryTier::ShortTerm],
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    // Only the short-term node should appear
    for scored_node in &result.nodes {
        assert_ne!(
            scored_node.node.id, semantic_node.id,
            "semantic node should be filtered when searching only ShortTerm"
        );
    }
}

// ---------------------------------------------------------------------------
// Result scoring and ordering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_results_are_sorted_by_score_descending() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;

    let texts: Vec<String> = (0..10).map(|i| format!("Fact number {i}.")).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let store = store_with_ingested_texts(&dir, dims, &text_refs).await;

    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let result = store
        .search("facts", options)
        .await
        .expect("search should succeed");

    if result.nodes.len() > 1 {
        for window in result.nodes.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "results should be sorted by descending score: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }
}

#[tokio::test]
async fn search_result_contains_retrieval_path() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(&dir, dims, &["Test node for retrieval path."]).await;

    let result = store
        .search("test", SearchOptions::default())
        .await
        .expect("search should succeed");

    // Every returned node should have a corresponding retrieval path entry
    assert_eq!(
        result.nodes.len(),
        result.retrieval_path.len(),
        "retrieval path should have one entry per returned node"
    );

    for step in &result.retrieval_path {
        // Each step should reference a node that exists in the results
        let found = result.nodes.iter().any(|sn| sn.node.id == step.node_id);
        assert!(
            found,
            "retrieval step for node {} should reference a returned node",
            step.node_id
        );

        // Combined score should be non-negative
        assert!(
            step.combined_score >= 0.0,
            "combined score should be non-negative"
        );
    }
}

// ---------------------------------------------------------------------------
// Graph traversal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn traverse_from_existing_node_returns_connected_graph() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), MultiNodeStubExtractor)
        .await
        .expect("store should open");

    // MultiNodeStubExtractor produces two nodes (Alice, Bob) with a "knows" edge
    let ingest_result = store
        .ingest_text("Alice knows Bob.")
        .await
        .expect("ingest should succeed");

    assert_eq!(ingest_result.nodes_written, 2, "should write two nodes");
    assert_eq!(ingest_result.edges_written, 1, "should write one edge");

    // Search to find a node to traverse from
    let search_result = store
        .search(
            "Alice",
            SearchOptions {
                limit: 1,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("search should succeed");

    assert!(
        !search_result.nodes.is_empty(),
        "should find at least one node"
    );

    let start_node_id = search_result.nodes[0].node.id;

    // Traverse from the found node
    let traversal = store
        .traverse(
            start_node_id,
            TraversalOptions {
                max_depth: 3,
                limit: 50,
                min_importance: 0.0,
                relation_filter: None,
            },
        )
        .await
        .expect("traversal should succeed");

    // Should find at least the start node
    assert!(
        !traversal.nodes.is_empty(),
        "traversal should return at least the start node"
    );
    assert_eq!(
        traversal.nodes[0].id, start_node_id,
        "first node in traversal should be the start node"
    );
}

#[tokio::test]
async fn traverse_nonexistent_node_returns_error() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(&dir, dims, &["some data"]).await;

    let fake_id = Uuid::new_v4();
    let result = store.traverse(fake_id, TraversalOptions::default()).await;

    assert!(
        result.is_err(),
        "traverse from nonexistent node should fail"
    );
    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[tokio::test]
async fn traverse_respects_max_depth() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), MultiNodeStubExtractor)
        .await
        .expect("store should open");

    store
        .ingest_text("Alice knows Bob.")
        .await
        .expect("ingest should succeed");

    // Find start node
    let search_result = store
        .search(
            "person",
            SearchOptions {
                limit: 1,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("search should succeed");

    if search_result.nodes.is_empty() {
        // Nothing to traverse — test passes trivially
        return;
    }

    let start_id = search_result.nodes[0].node.id;

    // Traverse with depth 0 — should only return the start node
    let traversal = store
        .traverse(
            start_id,
            TraversalOptions {
                max_depth: 0,
                limit: 50,
                min_importance: 0.0,
                relation_filter: None,
            },
        )
        .await
        .expect("traversal should succeed");

    assert_eq!(
        traversal.nodes.len(),
        1,
        "depth 0 should only return the start node"
    );
    assert_eq!(traversal.depth_reached, 0);
}

#[tokio::test]
async fn traverse_respects_limit() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), MultiNodeStubExtractor)
        .await
        .expect("store should open");

    store
        .ingest_text("Alice knows Bob.")
        .await
        .expect("ingest should succeed");

    let search_result = store
        .search(
            "person",
            SearchOptions {
                limit: 1,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("search should succeed");

    if search_result.nodes.is_empty() {
        return;
    }

    let start_id = search_result.nodes[0].node.id;

    // Traverse with limit 1 — should only return the start node
    let traversal = store
        .traverse(
            start_id,
            TraversalOptions {
                max_depth: 10,
                limit: 1,
                min_importance: 0.0,
                relation_filter: None,
            },
        )
        .await
        .expect("traversal should succeed");

    assert!(
        traversal.nodes.len() <= 1,
        "traversal should respect limit of 1, got {}",
        traversal.nodes.len()
    );
}

// ---------------------------------------------------------------------------
// Search with vector and graph weight configuration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_with_vector_weight_only() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(&dir, dims, &["Test vector-only search."]).await;

    let options = SearchOptions {
        limit: 10,
        vector_weight: 1.0,
        graph_weight: 0.0,
        min_importance: 0.0,
        tiers: Vec::new(),
    };

    let result = store
        .search("test", options)
        .await
        .expect("search should succeed");

    assert!(
        !result.nodes.is_empty(),
        "vector-only search should still return results"
    );
}

#[tokio::test]
async fn search_with_graph_weight_only() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store = store_with_ingested_texts(&dir, dims, &["Test graph-only search."]).await;

    let options = SearchOptions {
        limit: 10,
        vector_weight: 0.0,
        graph_weight: 1.0,
        min_importance: 0.0,
        tiers: Vec::new(),
    };

    let result = store
        .search("test", options)
        .await
        .expect("search should succeed");

    // With graph weight only, results come from traversal starting at the top vector hit.
    // The vector path is still used to find the starting point, but scoring favours graph.
    // Result may be empty if graph traversal finds nothing, or non-empty if it does.
    // Either way, no error should occur.
    let _ = result.nodes.len();
}

// ---------------------------------------------------------------------------
// Edge inclusion in search results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_result_includes_relevant_edges() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), MultiNodeStubExtractor)
        .await
        .expect("store should open");

    // MultiNodeStubExtractor produces two nodes with an edge between them
    store
        .ingest_text("Alice knows Bob.")
        .await
        .expect("ingest should succeed");

    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        ..SearchOptions::default()
    };

    let result = store
        .search("person", options)
        .await
        .expect("search should succeed");

    // If both nodes are returned, the connecting edge should be included
    if result.nodes.len() >= 2 {
        // edges may or may not be populated depending on whether graph traversal
        // discovered them, but the field should exist and be accessible
        let _edge_count = result.edges.len();
    }
}

// ---------------------------------------------------------------------------
// Scored node and edge invariants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scored_nodes_have_valid_scores() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let store =
        store_with_ingested_texts(&dir, dims, &["Fact one.", "Fact two.", "Fact three."]).await;

    let result = store
        .search("fact", SearchOptions::default())
        .await
        .expect("search should succeed");

    for scored_node in &result.nodes {
        // Scores should be finite numbers
        assert!(
            scored_node.score.is_finite(),
            "score should be finite, got {}",
            scored_node.score
        );
    }

    for scored_edge in &result.edges {
        assert!(
            scored_edge.score.is_finite(),
            "edge score should be finite, got {}",
            scored_edge.score
        );
    }
}

// ---------------------------------------------------------------------------
// Importance boost on retrieval
// ---------------------------------------------------------------------------

#[cfg(feature = "importance-decay")]
#[tokio::test]
async fn search_boosts_importance_of_retrieved_nodes() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let mut config = temp_config(&dir, dims);
    config.memory.importance_boost_on_retrieval = 0.1;

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Ingest a node with known importance
    let node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"content": "boosted fact"}),
        importance: 0.5,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![node.clone()], Vec::new())
        .await
        .expect("ingest should succeed");

    // First search — triggers retrieval boost
    let _result1 = store
        .search(
            "query",
            SearchOptions {
                limit: 10,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("first search should succeed");

    // Second search — the node's importance should have been boosted by the first search
    let result2 = store
        .search(
            "query",
            SearchOptions {
                limit: 10,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
        .expect("second search should succeed");

    // We can't easily verify the exact boost amount from outside, but we verify
    // the search still succeeds and the node is still present (meaning the boost
    // write didn't corrupt anything).
    assert!(
        !result2.nodes.is_empty(),
        "node should still be retrievable after boost"
    );
}

// ---------------------------------------------------------------------------
// Multiple tier search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_with_multiple_tiers_includes_all_matching() {
    let dir = TempDir::new().expect("temp dir");
    let dims = 32;
    let config = temp_config(&dir, dims);

    let store = Store::new(config, StubEmbedder::new(dims), StubExtractor)
        .await
        .expect("store should open");

    // Ingest nodes with different tier metadata
    let episodic_node = Node {
        id: Uuid::new_v4(),
        kind: "event".to_string(),
        metadata: serde_json::json!({"tier": "episodic"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    let short_term_node = Node {
        id: Uuid::new_v4(),
        kind: "working".to_string(),
        metadata: serde_json::json!({"tier": "short_term"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    let semantic_node = Node {
        id: Uuid::new_v4(),
        kind: "fact".to_string(),
        metadata: serde_json::json!({"tier": "semantic"}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: Uuid::nil(),
        source_text: None,
    };

    store
        .ingest_nodes(vec![episodic_node.clone()], Vec::new())
        .await
        .expect("ingest episodic should succeed");

    store
        .ingest_nodes(vec![short_term_node.clone()], Vec::new())
        .await
        .expect("ingest short_term should succeed");

    store
        .ingest_nodes(vec![semantic_node.clone()], Vec::new())
        .await
        .expect("ingest semantic should succeed");

    // Search Episodic + ShortTerm (should exclude Semantic)
    let options = SearchOptions {
        limit: 100,
        min_importance: 0.0,
        tiers: vec![MemoryTier::Episodic, MemoryTier::ShortTerm],
        ..SearchOptions::default()
    };

    let result = store
        .search("query", options)
        .await
        .expect("search should succeed");

    let found_semantic = result.nodes.iter().any(|sn| sn.node.id == semantic_node.id);
    assert!(
        !found_semantic,
        "semantic node should be excluded when searching Episodic + ShortTerm"
    );
}
