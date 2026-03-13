# ech0 — Product Requirements Document
**Version:** 0.1.0  
**Status:** Draft  
**Last Updated:** 2026-03-10

---

## 1. Vision

ech0 is a local-first, LLM-agnostic knowledge graph memory crate for Rust. It gives any LLM-powered application persistent, structured memory that grows smarter over time — without cloud dependencies, without API keys, and without data leaving the machine.

Memory is not a prompt dump. ech0 treats memory as a living network — new knowledge links to existing knowledge, existing knowledge evolves as new knowledge refines it, and irrelevant knowledge decays naturally. The result is a memory system that behaves less like a database and more like human associative memory.

ech0 is the memory layer. The caller provides the model.

---

## 2. Goals

- Provide hybrid memory (knowledge graph + vector search) in a single embedded crate with no external process dependencies
- Implement 2026 state-of-the-art memory practices: A-MEM dynamic linking, memory evolution, contradiction detection, importance decay, provenance tracking
- Expose a clean trait-based API that works with any LLM backend — llama.cpp, OpenAI, Anthropic, or anything else the caller wires in
- Be genuinely useful for any Rust project that needs local-first LLM memory
- Remain lightweight and composable — feature flags let callers pay only for what they use

---

## 3. Non-Goals

- ech0 does not provide an LLM — the caller brings their own via the `Embedder` and `Extractor` traits
- ech0 does not provide cloud storage backends in V1
- ech0 does not provide a query language — retrieval is via Rust API only
- ech0 does not manage conversation history or prompt assembly — that is the caller's responsibility

---

## 4. Core Concepts

### 4.1 Memory as a Living Network

Traditional memory systems are passive stores — write a fact, retrieve a fact. ech0 treats memory as an active network. Every new memory triggers a linking pass that asks how it connects to existing memories. Existing memories update their attributes when new memories refine them. The graph is never static.

This is based on A-MEM (Agentic Memory) and Zettelkasten principles — knowledge is not hierarchical, it is associative. ech0 mirrors this.

### 4.2 Hybrid Retrieval

ech0 stores every memory in two forms simultaneously:
- **Graph form** — nodes and edges in redb, queryable by relationship traversal
- **Vector form** — embedding in usearch, queryable by semantic similarity

Retrieval uses both paths and merges results. Graph traversal finds relational knowledge ("who did I meet through John"). Vector search finds semantic knowledge ("memories about exhaustion"). Neither alone is sufficient.

### 4.3 Memory Tiers

| Tier | What it holds | Lifespan | Retrieval |
|---|---|---|---|
| Short-term | Current session working context | Session duration | Direct, always injected |
| Episodic | Time-stamped events that happened | Long, decays slowly | Recency + relevance |
| Semantic | General facts, relationships, attributes | Long, decays by importance | Semantic similarity + graph traversal |

### 4.4 Importance and Decay

Every stored memory has an importance score. Score increases when the memory is retrieved, linked to, or confirmed by new information. Score decreases over time when the memory is never accessed. Memories below the configured decay threshold are pruned. The graph stays lean and accurate — it does not grow unbounded.

### 4.5 Provenance

Every node and edge knows:
- Where it came from (source text or caller-provided)
- When it was created (timestamp)
- Which ingest operation created it (ingest ID)
- Which memories it was linked to at creation time

Provenance is never secret data — it is queryable metadata that helps callers understand why a memory exists and how confident to be in it.

### 4.6 Contradiction Detection

When a new memory contradicts an existing one, ech0 flags the conflict. It does not silently overwrite. The caller receives a `ConflictReport` and decides resolution policy — keep old, replace with new, keep both with different confidence scores, or escalate to the application layer.

---

## 5. API Design

### 5.1 Entry Point

```rust
pub struct Store<E: Embedder, X: Extractor> {
    // single entry point — holds graph + vector layers
}

impl<E: Embedder, X: Extractor> Store<E, X> {
    pub async fn new(config: StoreConfig, embedder: E, extractor: X) -> Result<Self, EchoError>;
    
    // write paths
    pub async fn ingest_text(&self, text: &str) -> Result<IngestResult, EchoError>;
    pub async fn ingest_nodes(&self, nodes: Vec<Node>, edges: Vec<Edge>) -> Result<IngestResult, EchoError>;
    
    // read paths
    pub async fn search(&self, query: &str, options: SearchOptions) -> Result<SearchResult, EchoError>;
    pub async fn traverse(&self, from: Uuid, options: TraversalOptions) -> Result<TraversalResult, EchoError>;
    
    // memory management
    pub async fn decay(&self) -> Result<DecayReport, EchoError>;
    pub async fn prune(&self, threshold: f32) -> Result<PruneReport, EchoError>;
}
```

### 5.2 Traits

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EchoError>;
    fn dimensions(&self) -> usize;
}

#[async_trait]
pub trait Extractor: Send + Sync {
    async fn extract(&self, text: &str) -> Result<ExtractionResult, EchoError>;
}

pub struct ExtractionResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
```

### 5.3 Schema

```rust
pub struct Node {
    pub id: Uuid,
    pub kind: String,                    // caller-defined type label
    pub metadata: serde_json::Value,     // open bag — caller puts whatever they need
    
    // populated by ech0 internals — read-only from caller perspective
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    pub ingest_id: Uuid,
    pub source_text: Option<String>,     // only when provenance feature is on
}

pub struct Edge {
    pub source: Uuid,
    pub target: Uuid,
    pub relation: String,
    pub metadata: serde_json::Value,
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    pub ingest_id: Uuid,
}
```

### 5.4 Search

```rust
pub struct SearchOptions {
    pub limit: usize,
    pub vector_weight: f32,    // 0.0 - 1.0, balance between vector and graph results
    pub graph_weight: f32,
    pub min_importance: f32,   // filter out decayed memories below threshold
    pub tiers: Vec<MemoryTier>, // which tiers to search
}

pub struct SearchResult {
    pub nodes: Vec<ScoredNode>,
    pub edges: Vec<ScoredEdge>,
    pub retrieval_path: Vec<RetrievalStep>, // how each result was found
}
```

---

## 6. Feature Flags

All features are additive. No two features are mutually exclusive. Enabling any combination is safe.

| Feature | Default | What it enables |
|---|---|---|
| `dynamic-linking` | on | A-MEM background linking pass on every ingest — new memories link to related existing memories, existing memories evolve their attributes |
| `importance-decay` | on | Importance scoring, time-based decay, threshold-based pruning |
| `provenance` | on | Source text, timestamp, and ingest ID stored per node and edge |
| `contradiction-detection` | on | Conflict flagging when new memory contradicts existing memory, `ConflictReport` returned to caller |
| `tokio` | on | Async runtime support via tokio |
| `full` | off | Enables all of the above |

```toml
# minimal — graph + vector, no extras
ech0 = { version = "0.1", default-features = false }

# default — everything on
ech0 = { version = "0.1" }

# customized
ech0 = { version = "0.1", features = ["dynamic-linking", "importance-decay", "provenance", "contradiction-detection"] }
```

---

## 7. Storage Architecture

### 7.1 Graph Layer — redb

redb is a pure Rust embedded key-value store with a stable file format. ech0 uses it to store nodes, edges, and the adjacency index.

Table layout:
```
nodes           — Uuid → Node (msgpack serialized)
edges           — Uuid → Edge (msgpack serialized)  
adjacency_out   — Uuid → Vec<Uuid> (outgoing edge IDs per node)
adjacency_in    — Uuid → Vec<Uuid> (incoming edge IDs per node)
importance      — Uuid → f32 (importance scores, updated independently)
```

All graph writes are transactional. A failed ingest never leaves partial state.

### 7.2 Vector Layer — usearch

usearch is a pure Rust approximate nearest neighbor library. ech0 uses it to store embeddings alongside references to their corresponding graph nodes.

```
index           — embedding vectors, labeled with node Uuid
```

Vector writes are paired with graph writes in the same ingest operation. A node always has both a graph entry and a vector entry — they are never out of sync.

### 7.3 Write Atomicity

`ingest_text()` and `ingest_nodes()` write to both layers atomically from the caller's perspective:

1. Run extraction (if ingest_text)
2. Begin redb write transaction
3. Write nodes and edges to graph tables
4. Write embeddings to usearch index
5. Commit redb transaction
6. On any failure — rollback graph transaction, remove any written embeddings, return error

If the process crashes between steps 4 and 5, the next startup runs a consistency check and removes orphaned embeddings.

---

## 8. A-MEM Dynamic Linking

When `dynamic-linking` feature is enabled, every successful ingest triggers a background linking pass:

1. For each newly ingested node, run vector search to find top-k semantically similar existing nodes
2. For each similar node above the similarity threshold, ask the `Extractor` to determine the relationship between the new node and the existing node
3. If a relationship is found, write a new dynamic edge between them
4. Update the existing node's metadata to reflect the new connection — this is memory evolution
5. Increase importance scores of all linked nodes

The linking pass runs as a `tokio::spawn` background task — it does not block the `ingest_text()` return. The caller receives `IngestResult` immediately. Links are formed asynchronously.

```rust
pub struct IngestResult {
    pub ingest_id: Uuid,
    pub nodes_written: usize,
    pub edges_written: usize,
    pub conflicts: Vec<ConflictReport>,  // populated if contradiction-detection is on
    pub linking_task: Option<JoinHandle<LinkingResult>>, // caller can await or drop
}
```

Caller can await the linking task if they need it to complete before querying, or drop it and let it run in the background.

---

## 9. Contradiction Detection

When `contradiction-detection` feature is enabled, every ingest checks for conflicts before writing:

A conflict exists when a newly extracted node asserts something that directly contradicts an existing node of the same kind and subject.

```rust
pub struct ConflictReport {
    pub new_node: Node,
    pub existing_node: Node,
    pub conflict_type: ConflictType,
    pub confidence: f32,
}

pub enum ConflictType {
    DirectContradiction,   // "X is Y" vs "X is not Y"
    ValueConflict,         // "X is 30" vs "X is 25"
    TemporalConflict,      // older fact may have been superseded
}

pub enum ConflictResolution {
    KeepExisting,
    ReplaceWithNew,
    KeepBoth,              // both stored, different confidence scores
    Escalate,              // caller handles
}
```

Conflict resolution policy is configured per `Store` instance. Default is `Escalate` — ech0 never silently resolves conflicts unless explicitly configured to.

---

## 10. Error Handling

```rust
pub struct EchoError {
    pub code: ErrorCode,
    pub message: String,
    pub context: Option<ErrorContext>,  // internal debug only, never exposed across crate boundary
}

pub enum ErrorCode {
    StorageFailure,
    EmbedderFailure,
    ExtractorFailure,
    ConsistencyError,
    ConflictUnresolved,
    InvalidInput,
    CapacityExceeded,
}
```

No silent failures. Every error returns a typed `EchoError`. Callers handle all error cases explicitly — ech0 never swallows errors or returns partial results without indicating they are partial.

---

## 11. Configuration

```toml
# ech0.toml — all tunable values, nothing hardcoded in library source

[store]
graph_path = "./ech0_graph"          # redb database file path
vector_path = "./ech0_vectors"       # usearch index file path
vector_dimensions = 768              # must match embedder output

[memory]
short_term_capacity = 50             # max entries in short-term tier
episodic_decay_rate = 0.01           # importance decrease per day without access
semantic_decay_rate = 0.005          # semantic tier decays slower
prune_threshold = 0.1                # nodes below this importance are pruned
importance_boost_on_retrieval = 0.1  # importance increase when memory is accessed

[dynamic_linking]
top_k_candidates = 5                 # how many similar nodes to consider for linking
similarity_threshold = 0.75          # minimum similarity to trigger linking attempt
max_links_per_ingest = 10            # cap on new dynamic edges per ingest operation

[contradiction]
resolution_policy = "escalate"       # escalate / keep_existing / replace_with_new / keep_both
confidence_threshold = 0.8           # minimum confidence to flag as contradiction
```

---

## 12. Testing Strategy

| Test | Scope | Purpose |
|---|---|---|
| Unit: graph write/read | store/graph | Assert nodes and edges round-trip correctly through redb |
| Unit: vector write/search | store/vector | Assert embeddings store and retrieve by cosine similarity correctly |
| Unit: atomic write failure | store/ | Assert partial failure rolls back both layers cleanly |
| Unit: importance decay | store/memory | Assert importance scores decrease correctly over simulated time |
| Unit: pruning | store/memory | Assert nodes below threshold are removed, above threshold are kept |
| Unit: contradiction detection | store/conflict | Assert known contradictions are flagged correctly |
| Unit: conflict resolution policies | store/conflict | Assert each resolution policy produces correct state |
| Integration: full ingest cycle | store/ | ingest_text → extract → embed → graph write → vector write → search → retrieve |
| Integration: A-MEM linking | store/linking | Assert dynamic edges are created between related memories after linking pass |
| Integration: memory evolution | store/linking | Assert existing node metadata updates when new memory refines it |
| Fuzz: extractor output | store/ | Assert malformed ExtractionResult never corrupts graph state |
| Fuzz: embedder output | store/ | Assert wrong-dimension embeddings are rejected cleanly |

---

## 13. Version Roadmap

### V1 — Foundation
- Hybrid graph + vector storage (redb + usearch)
- `Embedder` + `Extractor` traits
- `ingest_text()` + `ingest_nodes()`
- Hybrid search + graph traversal
- Three memory tiers
- All default feature flags: dynamic-linking, importance-decay, provenance, contradiction-detection

### V2 — Extended Memory
- Procedural memory component — how to do things, workflow patterns
- Resource memory component — file references, external tool references
- Additional storage backends as feature flags (pluggable)
- Performance profiling + optimization pass

### V3 — Full MIRIX
- Knowledge Vault component — high-confidence long-term facts, never decays
- Full MIRIX six-component architecture
- Cloud storage backends (if funded)
- Multi-model embedder support — different embedders per memory tier

---

## 14. Project Structure

```
ech0/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── ech0.toml.example
├── src/
│   ├── lib.rs              — public API surface, re-exports
│   ├── store.rs            — Store entry point
│   ├── config.rs           — StoreConfig, all config structs
│   ├── error.rs            — EchoError, ErrorCode
│   ├── schema.rs           — Node, Edge, MemoryTier
│   ├── traits.rs           — Embedder, Extractor
│   ├── graph.rs            — redb graph layer
│   ├── vector.rs           — usearch vector layer
│   ├── search.rs           — hybrid retrieval, result merging
│   ├── linking.rs          — A-MEM dynamic linking pass
│   ├── decay.rs            — importance scoring, decay, pruning
│   ├── conflict.rs         — contradiction detection, ConflictReport
│   └── provenance.rs       — provenance tracking
└── tests/
    ├── integration/
    │   ├── ingest.rs
    │   ├── search.rs
    │   ├── linking.rs
    │   └── conflict.rs
    └── fuzz/
        ├── extractor_output.rs
        └── embedder_output.rs
```

---

## 15. Open Source Identity

ech0 is published as a standalone crate on crates.io. It has no dependency on any other system. Any Rust project that needs local-first LLM memory can use it.

**Tagline:** Local-first knowledge graph memory for LLMs. No cloud. No API keys. No data leaving your machine.

**Target audience:** Rust developers building LLM-powered applications who need persistent, structured memory without cloud dependencies.

**Funding path:** Launch with llama.cpp-focused community, expand to cloud model support and additional storage backends as funding allows.