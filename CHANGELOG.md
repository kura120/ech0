# Changelog

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