# ech0

Local-first knowledge graph memory for LLMs. No cloud. No API keys. No data leaving your machine.

ech0 is a standalone Rust crate, which means... no external process, no setup. Drop it into any Rust project and give your LLM a memory that actually works.

> [!NOTE]
> Active development. API is not yet stable.

## What it does

Most LLM memory solutions are either stateless (context window only) or cloud-dependent. ech0 is neither.

It maintains a persistent hybrid store: a knowledge graph for structure and a vector index for semantics. ech0 also applies best memory practices that keep it accurate over time without growing unbounded.

## How memory stays accurate

**A-MEM dynamic linking** — every new memory triggers a background linking pass. Memories form a living network, not an append-only list.

**Memory evolution** — new experiences retroactively refine existing memories. Old nodes update their attributes when new context changes their meaning.

**Contradiction detection** — when a new memory contradicts an existing one, ech0 surfaces a `ConflictReport` to the caller. Silent overwrites never happen.

**Importance decay** — every node has an importance score. Score rises on retrieval and linking, falls over time. Nodes below threshold are pruned. The graph stays lean.

**Sparse retrieval** — ech0 always returns the minimum relevant context, never the maximum. Context poisoning is a correctness property, not a performance tradeoff.

**Provenance tracking** — every node knows its source, timestamp, and ingest ID.

## Storage

| Layer | Library | Purpose |
|---|---|---|
| Graph | [redb](https://github.com/cberner/redb) | Node, edge, and adjacency storage. Pure Rust, embedded, transactional. |
| Vector | [usearch](https://github.com/unum-cloud/usearch) | Embedding storage and approximate nearest neighbor search. Pure Rust, no external process. |

No Postgres. No Redis. No network calls. Two embedded files on disk.

## Roadmap

| Version | What's included |
|---|---|
| V1 | Episodic + semantic memory, hybrid retrieval, A-MEM dynamic linking, contradiction detection, memory evolution, importance decay, provenance |
| V2 | Procedural memory (workflow patterns), resource memory (file/document references), pluggable backends |
| V3 | Full MIRIX six-component architecture, Knowledge Vault |

## Motivation

Originally developed for [Sena](https://github.com/kura120/sena): local-first AI companion. ech0 is Sena's memory layer, consumed via `memory-engine`.

## License

MIT