#![cfg_attr(not(test), deny(unused_crate_dependencies))]

//! # ech0 — Local-first knowledge graph memory for LLMs
//!
//! ech0 is a hybrid knowledge graph + vector memory store backed by redb (graph) and
//! usearch (vector). The caller provides an LLM via the [`Embedder`] and [`Extractor`]
//! traits. ech0 handles storage, retrieval, decay, linking, and conflict detection.
//!
//! No cloud. No API keys. No data leaving your machine.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use ech0::{Store, StoreConfig, Embedder, Extractor};
//!
//! // Implement Embedder and Extractor for your LLM backend
//! let config = StoreConfig::default();
//! let store = Store::new(config, my_embedder, my_extractor).await?;
//!
//! let result = store.ingest_text("Alice is 30 years old.").await?;
//! let search = store.search("how old is Alice?", Default::default()).await?;
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Default | What it enables |
//! |---|---|---|
//! | `dynamic-linking` | on | A-MEM background linking pass on every ingest |
//! | `importance-decay` | on | Importance scoring, time-based decay, threshold pruning |
//! | `provenance` | on | Source text stored per node for traceability |
//! | `contradiction-detection` | on | Conflict flagging when new memory contradicts existing |
//! | `tokio` | on | Async runtime support via tokio |
//! | `full` | off | Enables all of the above |

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod config;
pub mod error;
pub mod graph;
pub mod schema;
pub mod search;
pub mod store;
pub mod traits;
pub mod vector;

#[cfg(feature = "dynamic-linking")]
pub mod linking;

#[cfg(feature = "importance-decay")]
pub mod decay;

#[cfg(feature = "contradiction-detection")]
pub mod conflict;

#[cfg(feature = "provenance")]
pub mod provenance;

/// Test stub implementations of `Embedder` and `Extractor`.
///
/// **Not part of the public API.** These stubs exist for integration testing only.
/// They are gated behind the `_test-helpers` feature which is enabled automatically
/// for dev builds via `[dev-dependencies]`.
#[cfg(feature = "_test-helpers")]
#[doc(hidden)]
pub mod test_stubs;

// ---------------------------------------------------------------------------
// Public re-exports — the caller-facing API surface
// ---------------------------------------------------------------------------

// Entry point
pub use store::Store;

// Configuration
pub use config::{
    ContradictionConfig, DynamicLinkingConfig, MemoryConfig, StoreConfig, StorePathConfig,
};

// Error types
pub use error::{EchoError, ErrorCode, ErrorContext};

// Core schema types
pub use schema::{
    DecayReport, Edge, IngestResult, MemoryTier, Node, PruneReport, RetrievalSource, RetrievalStep,
    ScoredEdge, ScoredNode, SearchOptions, SearchResult, TraversalOptions, TraversalResult,
};

// Caller-implemented traits
pub use traits::{Embedder, ExtractionResult, Extractor};

// Feature-gated re-exports

#[cfg(feature = "dynamic-linking")]
pub use linking::LinkingResult;

#[cfg(feature = "contradiction-detection")]
pub use conflict::{ConflictReport, ConflictResolution, ConflictType};
