//! ech0 V1 end-to-end smoke test binary.
//!
//! Run with: `cargo run --bin smoke`
//!
//! Exercises the full V1 pipeline against a real redb + usearch store:
//!   1. Basic ingest and retrieval
//!   2. Contradiction detection
//!   3. Dynamic linking
//!   4. Decay and prune
//!   5. Graph traversal
//!
//! Not a `#[test]`. Not a default Embedder implementation. Not a framework.
//! FixedEmbedder and NoopExtractor are declared here and exist only here.

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use ech0::{
    ConflictType, ContradictionConfig, DecayReport, DynamicLinkingConfig, EchoError, Edge,
    Embedder, ExtractionResult, Extractor, IngestResult, LinkingResult, Node, PruneReport,
    SearchOptions, SearchResult, StoreConfig, StorePathConfig, TraversalOptions, TraversalResult,
};

// ---------------------------------------------------------------------------
// FixedEmbedder
//
// Smoke-test-only. NOT a default implementation. Lives only in this file.
//
// Returns a unit vector seeded by (text.len() + i) % 7 for each dimension i,
// then L2-normalised. Different-length texts get different vectors.
// Same text always returns the same vector.
// ---------------------------------------------------------------------------

struct FixedEmbedder {
    dimensions: usize,
}

#[async_trait]
impl Embedder for FixedEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EchoError> {
        let seed = text.len();
        let mut vector: Vec<f32> = (0..self.dimensions)
            .map(|index| ((seed + index) % 7) as f32)
            .collect();

        // L2-normalise so the vector lies on the unit sphere.
        // usearch cosine search requires unit vectors for meaningful similarity.
        let magnitude: f32 = vector
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();

        // Guard: if every component is zero, magnitude is zero. Substitute a
        // uniform unit vector so usearch always receives a valid non-zero input.
        if magnitude < f32::EPSILON {
            let uniform = (1.0_f32 / self.dimensions as f32).sqrt();
            vector.fill(uniform);
        } else {
            for component in &mut vector {
                *component /= magnitude;
            }
        }

        Ok(vector)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ---------------------------------------------------------------------------
// NoopExtractor
//
// ingest_text() is never called in this binary, but Store::new requires an
// Extractor in its type signature. This satisfies the bound without doing work.
// ---------------------------------------------------------------------------

struct NoopExtractor;

#[async_trait]
impl Extractor for NoopExtractor {
    async fn extract(&self, _text: &str) -> Result<ExtractionResult, EchoError> {
        Ok(ExtractionResult {
            nodes: Vec::new(),
            edges: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Node construction helper
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Assertion helper
//
// Prints "FAIL: <reason>" and exits with code 1. Using a macro so the message
// has access to format args without requiring anyhow or std::error::Error boxes.
// ---------------------------------------------------------------------------

macro_rules! smoke_assert {
    ($cond:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        if !$cond {
            eprintln!("FAIL: {}", format!($fmt, $($arg),*));
            std::process::exit(1);
        }
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Clean slate on each run. redb and usearch store data as plain files, not
    // directories, so remove_file is correct here. Errors are intentionally
    // ignored — the files simply may not exist on a first run.
    let _ = std::fs::remove_file("./smoke_graph");
    let _ = std::fs::remove_file("./smoke_vectors");
    // usearch may also write a companion index file with a .usearch extension
    let _ = std::fs::remove_file("./smoke_vectors.usearch");

    const DIMS: usize = 64;

    let config = StoreConfig {
        store: StorePathConfig {
            graph_path: "./smoke_graph".to_string(),
            vector_path: "./smoke_vectors".to_string(),
            vector_dimensions: DIMS,
        },
        contradiction: ContradictionConfig {
            // Low threshold so metadata value differences trigger detection even
            // when the fixed embedder produces non-semantic vectors.
            confidence_threshold: 0.05,
            ..Default::default()
        },
        dynamic_linking: DynamicLinkingConfig {
            // Low threshold because the fixed embedder is not semantically meaningful —
            // we are exercising pipeline machinery, not semantic quality.
            similarity_threshold: 0.1,
            top_k_candidates: 5,
            max_links_per_ingest: 10,
        },
        ..Default::default()
    };

    let store =
        match ech0::Store::new(config, FixedEmbedder { dimensions: DIMS }, NoopExtractor).await {
            Ok(store) => store,
            Err(error) => {
                eprintln!("FAIL: store::new failed: {error}");
                std::process::exit(1);
            }
        };

    // -----------------------------------------------------------------------
    // Scenario 1: Basic ingest and retrieval
    // -----------------------------------------------------------------------

    println!("=== Scenario 1: Basic ingest and retrieval ===");

    let alice = make_node(
        "person",
        serde_json::json!({"name": "Alice", "occupation": "engineer", "age": 30}),
    );
    let london = make_node(
        "city",
        serde_json::json!({"name": "London", "country": "UK"}),
    );

    let alice_id = alice.id;
    let london_id = london.id;

    let placeholder_ingest_id = Uuid::new_v4();
    let lives_in_edge = Edge {
        source: alice_id,
        target: london_id,
        relation: "lives_in".to_string(),
        metadata: serde_json::json!({}),
        importance: 0.8,
        created_at: Utc::now(),
        ingest_id: placeholder_ingest_id,
    };

    let ingest_result: IngestResult = match store
        .ingest_nodes(vec![alice, london], vec![lives_in_edge])
        .await
    {
        Ok(result) => result,
        Err(error) => {
            eprintln!("FAIL: scenario 1 ingest failed: {error}");
            std::process::exit(1);
        }
    };

    println!(
        "Ingest: nodes={} edges={} conflicts={}",
        ingest_result.nodes_written,
        ingest_result.edges_written,
        ingest_result.conflicts.len(),
    );

    smoke_assert!(
        ingest_result.nodes_written == 2,
        "expected nodes_written=2, got {}",
        ingest_result.nodes_written,
    );
    smoke_assert!(
        ingest_result.edges_written == 1,
        "expected edges_written=1, got {}",
        ingest_result.edges_written,
    );
    smoke_assert!(
        ingest_result.conflicts.is_empty(),
        "expected no conflicts on first ingest, got {}",
        ingest_result.conflicts.len(),
    );

    // Drop the linking task — scenario 1 does not need to wait for linking.
    // Scenario 3 explicitly awaits its linking task.
    drop(ingest_result.linking_task);

    let search_result: SearchResult = match store
        .search(
            "engineer",
            SearchOptions {
                limit: 3,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
    {
        Ok(result) => result,
        Err(error) => {
            eprintln!("FAIL: scenario 1 search failed: {error}");
            std::process::exit(1);
        }
    };

    println!(
        "Search \"engineer\": {} result(s)",
        search_result.nodes.len()
    );

    for (index, scored) in search_result.nodes.iter().enumerate() {
        let name = scored
            .node
            .metadata
            .get("name")
            .and_then(|value: &serde_json::Value| value.as_str())
            .unwrap_or("<unnamed>");
        println!(
            "  [{}] {} / {} (importance: {:.3})",
            index, scored.node.kind, name, scored.node.importance,
        );
    }

    smoke_assert!(
        !search_result.nodes.is_empty(),
        "expected at least 1 search result for \"engineer\"",
    );

    println!("PASS\n");

    // -----------------------------------------------------------------------
    // Scenario 2: Contradiction detection
    // -----------------------------------------------------------------------

    println!("=== Scenario 2: Contradiction detection ===");

    let alice_conflict = make_node(
        "person",
        serde_json::json!({"name": "Alice", "occupation": "engineer", "age": 45}),
    );

    let conflict_ingest: IngestResult = match store.ingest_nodes(vec![alice_conflict], vec![]).await
    {
        Ok(result) => result,
        Err(error) => {
            eprintln!("FAIL: scenario 2 ingest failed: {error}");
            std::process::exit(1);
        }
    };

    drop(conflict_ingest.linking_task);

    println!("Conflicts: {}", conflict_ingest.conflicts.len());

    for (index, report) in conflict_ingest.conflicts.iter().enumerate() {
        println!(
            "  [{}] type={:?} confidence={:.2}",
            index, report.conflict_type, report.confidence,
        );
    }

    smoke_assert!(
        !conflict_ingest.conflicts.is_empty(),
        "expected at least 1 conflict when age changes 30 → 45",
    );

    let conflict_type_ok = conflict_ingest.conflicts.iter().any(|report| {
        matches!(
            report.conflict_type,
            ConflictType::ValueConflict | ConflictType::DirectContradiction
        )
    });
    smoke_assert!(
        conflict_type_ok,
        "expected conflict type ValueConflict or DirectContradiction",
    );

    println!("PASS\n");

    // -----------------------------------------------------------------------
    // Scenario 3: Dynamic linking
    // -----------------------------------------------------------------------

    println!("=== Scenario 3: Dynamic linking ===");

    let carol = make_node(
        "person",
        serde_json::json!({"name": "Carol", "occupation": "engineer", "age": 28}),
    );

    let linking_ingest: IngestResult = match store.ingest_nodes(vec![carol], vec![]).await {
        Ok(result) => result,
        Err(error) => {
            eprintln!("FAIL: scenario 3 ingest failed: {error}");
            std::process::exit(1);
        }
    };

    // Await the linking task — do not drop it silently.
    let linking_result: LinkingResult = if let Some(handle) = linking_ingest.linking_task {
        // JoinHandle<LinkingResult>: .await gives LinkingResult directly (not Result<LinkingResult>)
        // because the linking pass is spawned with tokio::spawn and the task does not return
        // a Result — it returns LinkingResult. JoinHandle::await returns Result<T, JoinError>
        // where the Err case indicates a panic in the task.
        match handle.await {
            Ok(result) => result,
            Err(join_error) => {
                eprintln!("FAIL: linking task panicked: {join_error}");
                std::process::exit(1);
            }
        }
    } else {
        LinkingResult::empty(Uuid::new_v4())
    };

    println!(
        "Linking: edges_created={} nodes_boosted={} warnings={}",
        linking_result.edges_created,
        linking_result.nodes_boosted,
        linking_result.warnings.len(),
    );

    for warning in &linking_result.warnings {
        println!("  warning: {warning}");
    }

    // Assert only that the task ran without panicking and the result is readable.
    // We do not assert a specific edge count — the fixed embedder may or may not
    // produce cosine similarity above 0.1 for these nodes, and that is acceptable.
    let _ = linking_result.ingest_id;
    let _ = linking_result.new_edges.len();

    println!("PASS\n");

    // -----------------------------------------------------------------------
    // Scenario 4: Decay and prune
    // -----------------------------------------------------------------------

    println!("=== Scenario 4: Decay and prune ===");

    let decay_report: DecayReport = match store.decay().await {
        Ok(report) => report,
        Err(error) => {
            eprintln!("FAIL: decay failed: {error}");
            std::process::exit(1);
        }
    };

    println!("Decay: {:?}", decay_report);

    // Prune at a low threshold — nodes with importance=0.8 and near-zero elapsed
    // time since ingest should all survive. We exercise the code path, not the count.
    let prune_report: PruneReport = match store.prune(0.05).await {
        Ok(report) => report,
        Err(error) => {
            eprintln!("FAIL: prune failed: {error}");
            std::process::exit(1);
        }
    };

    println!(
        "Prune: nodes_removed={} edges_removed={}",
        prune_report.nodes_pruned, prune_report.edges_pruned,
    );

    let post_prune_search: SearchResult = match store
        .search(
            "engineer",
            SearchOptions {
                limit: 10,
                min_importance: 0.0,
                ..SearchOptions::default()
            },
        )
        .await
    {
        Ok(result) => result,
        Err(error) => {
            eprintln!("FAIL: post-prune search failed: {error}");
            std::process::exit(1);
        }
    };

    println!(
        "Search \"engineer\" post-prune: {} result(s)",
        post_prune_search.nodes.len(),
    );

    // The store must still be queryable after decay + prune. With threshold=0.05
    // and importance=0.8, no nodes should be pruned, so at least Alice must appear.
    smoke_assert!(
        !post_prune_search.nodes.is_empty(),
        "store returned no results after decay+prune — at least Alice should survive",
    );

    println!("PASS\n");

    // -----------------------------------------------------------------------
    // Scenario 5: Graph traversal
    // -----------------------------------------------------------------------

    println!("=== Scenario 5: Graph traversal ===");

    let traversal: TraversalResult =
        match store.traverse(alice_id, TraversalOptions::default()).await {
            Ok(result) => result,
            Err(error) => {
                eprintln!("FAIL: traversal from alice_id failed: {error}");
                std::process::exit(1);
            }
        };

    println!(
        "Traversal from Alice: {} node(s) reached",
        traversal.nodes.len(),
    );

    for node in &traversal.nodes {
        let name = node
            .metadata
            .get("name")
            .and_then(|value: &serde_json::Value| value.as_str())
            .unwrap_or("<unnamed>");
        println!("  {} / {}", node.kind, name);
    }

    smoke_assert!(
        !traversal.nodes.is_empty(),
        "traversal from alice_id must reach at least 1 node",
    );

    // Confirm London is reachable via the lives_in edge within the default depth=3.
    let reached_london = traversal.nodes.iter().any(|node| node.id == london_id);
    if reached_london {
        println!("  (London reached via lives_in edge — graph connectivity confirmed)");
    }

    println!("PASS\n");

    // -----------------------------------------------------------------------
    // All done
    // -----------------------------------------------------------------------

    println!("=== All scenarios passed ===");
}
