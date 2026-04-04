# Changelog

## [0.1.2] — 2026-06-01

### Changed

- **Default vector backend switched from usearch to hora** — `backend-hora` is now in `default` features; `backend-usearch` is an opt-in feature
- `src/vector.rs` moved to `src/vector/` folder (modular multi-backend layout)
  - `src/vector/usearch.rs` — usearch C++-backed backend (unchanged behaviour)
  - `src/vector/hora.rs` — new hora pure-Rust backend (no C++ toolchain required)
  - `src/vector/mod.rs` — `VectorIndex` trait definition and `DefaultVectorLayer` alias
- `src/vector_index.rs` is now a backwards-compat re-export shim for `VectorIndex`
- `Store` now constructs `DefaultVectorLayer` (feature-gated) instead of hardcoding usearch
- `StubEmbedder` updated to return `vec![1.0f32; dims]` instead of zeros — zero-magnitude embeddings are not valid cosine-similarity inputs

### Added

- `backend-hora` feature: hora brute-force ANN index (pure Rust, default)
- `backend-usearch` feature: usearch HNSW index (C++ via cxx, optional)
- `full` feature now includes both `backend-hora` and `backend-usearch`
- `HoraVectorLayer` public re-export from crate root when `backend-hora` is active
- `UsearchVectorLayer` public re-export from crate root when `backend-usearch` is active
- Feature flags table in `lib.rs` doc updated to include `backend-hora` and `backend-usearch`

### Fixed

- hora's `Metric::CosineSimilarity` produces NaN (`dot_product` returns `-(actual dot)` in hora 0.1.1). Worked around by normalising embeddings to unit length and using `Metric::Euclidean` (squared Euclidean), then converting back via `cos(θ) = 1 − dist/2`.

### Subsystems affected

- `src/vector/` (new module layout)
- `src/store.rs` (backend selection)
- `src/lib.rs` (public re-exports)
- `src/test_stubs.rs` (StubEmbedder fix)
- `Cargo.toml` (feature flags, version bump)

---

## [0.1.0] — 2026-03-14

### Added

- Hybrid graph + vector storage via `redb` (graph layer) and `usearch` (vector layer)
- `Embedder` and `Extractor` caller-provided traits — no default LLM implementation shipped
- `Store::ingest_text()` — full pipeline: extraction, embedding, graph write, vector write
- `Store::ingest_nodes()` — direct graph + vector write bypassing extraction
- `Store::search()` — hybrid ANN + graph expansion search
- `Store::traverse()` — graph traversal from a starting node
- `Store::decay()` — time-based importance decay across all nodes and edges
- `Store::prune()` — removal of nodes and edges below importance threshold
- Three memory tiers: short-term, episodic, semantic
- `dynamic-linking` feature — A-MEM background linking pass on every ingest
- `importance-decay` feature — importance scoring, decay, and pruning
- `provenance` feature — source text, timestamp, and ingest ID per node and edge
- `contradiction-detection` feature — typed conflict detection with `Escalate` default policy
- `ConflictReport` — typed conflict reports returned to caller, never silently resolved
- `ConflictResolution` — `Escalate`, `KeepExisting`, `ReplaceWithNew`, `KeepBoth`
- `StoreConfig` — fully programmatic configuration, TOML-deserializable
- `_test-helpers` dev feature — `StubEmbedder` and `StubExtractor` for integration tests
- 224 unit and integration tests
- `smoke` binary — end-to-end pipeline validation against live redb + usearch store