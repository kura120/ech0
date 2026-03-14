//! Configuration types for ech0.
//!
//! All tunable values — thresholds, decay rates, capacity limits, paths — live here.
//! Nothing is hardcoded in library source. `StoreConfig` can be constructed programmatically
//! or deserialized from a TOML file by the caller.

use serde::{Deserialize, Serialize};

#[cfg(feature = "contradiction-detection")]
use crate::conflict::ConflictResolution;

/// Top-level configuration for an ech0 `Store` instance.
///
/// Every field has a sensible default via `Default`. Callers can construct this
/// programmatically, deserialize from TOML, or mix both approaches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Storage paths and vector dimensions.
    #[serde(default)]
    pub store: StorePathConfig,

    /// Memory tier capacities and decay parameters.
    #[serde(default)]
    pub memory: MemoryConfig,

    /// A-MEM dynamic linking parameters (used when `dynamic-linking` feature is enabled).
    #[serde(default)]
    pub dynamic_linking: DynamicLinkingConfig,

    /// Contradiction detection parameters (used when `contradiction-detection` feature is enabled).
    #[serde(default)]
    pub contradiction: ContradictionConfig,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            store: StorePathConfig::default(),
            memory: MemoryConfig::default(),
            dynamic_linking: DynamicLinkingConfig::default(),
            contradiction: ContradictionConfig::default(),
        }
    }
}

/// Storage paths and vector dimensionality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorePathConfig {
    #[serde(default = "default_graph_path")]
    pub graph_path: String,

    #[serde(default = "default_vector_path")]
    pub vector_path: String,

    #[serde(default = "default_vector_dimensions")]
    pub vector_dimensions: usize,
}

fn default_graph_path() -> String { "./ech0_graph".to_string() }
fn default_vector_path() -> String { "./ech0_vectors".to_string() }
fn default_vector_dimensions() -> usize { 768 }

impl Default for StorePathConfig {
    fn default() -> Self {
        Self {
            graph_path: "./ech0_graph".to_string(),
            vector_path: "./ech0_vectors".to_string(),
            vector_dimensions: 768,
        }
    }
}

/// Memory tier capacities and importance decay parameters.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum number of entries in the short-term memory tier.
    /// Once capacity is reached, oldest short-term memories are promoted to episodic or pruned.
    #[serde(default = "default_short_term_capacity")]
    pub short_term_capacity: usize,

    /// Importance score decrease per day without access for episodic memories.
    /// Applied when `decay()` is called. Higher values mean faster forgetting.
    #[serde(default = "default_episodic_decay_rate")]
    pub episodic_decay_rate: f32,

    /// Importance score decrease per day without access for semantic memories.
    /// Semantic memories decay slower than episodic — they represent general facts, not events.
    #[serde(default = "default_semantic_decay_rate")]
    pub semantic_decay_rate: f32,

    /// Nodes with importance score below this threshold are removed when `prune()` is called.
    /// Range: 0.0–1.0.
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f32,

    /// Importance score increase when a memory is accessed via search or traversal.
    /// Frequently retrieved memories stay alive; unused memories decay naturally.
    #[serde(default = "default_importance_boost_on_retrieval")]
    pub importance_boost_on_retrieval: f32,
}

fn default_short_term_capacity() -> usize { 50 }
fn default_episodic_decay_rate() -> f32 { 0.01 }
fn default_semantic_decay_rate() -> f32 { 0.005 }
fn default_prune_threshold() -> f32 { 0.1 }
fn default_importance_boost_on_retrieval() -> f32 { 0.1 }


impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 50,
            episodic_decay_rate: 0.01,
            semantic_decay_rate: 0.005,
            prune_threshold: 0.1,
            importance_boost_on_retrieval: 0.1,
        }
    }
}

/// Parameters for A-MEM dynamic linking pass.
///
/// Only used when the `dynamic-linking` feature is enabled. The struct always exists
/// so that config deserialization works regardless of feature flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicLinkingConfig {
    /// Number of semantically similar existing nodes to consider when linking a newly ingested node.
    /// Higher values find more connections but cost more embedder calls.
    #[serde(default = "default_top_k_candidates")]
    pub top_k_candidates: usize,

    /// Minimum cosine similarity between a new node and an existing node to attempt linking.
    /// Range: 0.0–1.0. Higher values produce fewer but higher-quality links.
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Maximum number of new dynamic edges created per ingest operation.
    /// Prevents a single large ingest from flooding the graph with links.
    #[serde(default = "default_max_links_per_ingest")]
    pub max_links_per_ingest: usize,
}

fn default_top_k_candidates() -> usize { 50 }
fn default_similarity_threshold() -> f32 { 0.75 }
fn default_max_links_per_ingest() -> usize { 10 }

impl Default for DynamicLinkingConfig {
    fn default() -> Self {
        Self {
            top_k_candidates: 5,
            similarity_threshold: 0.75,
            max_links_per_ingest: 10,
        }
    }
}

/// Parameters for contradiction detection.
///
/// Only used when the `contradiction-detection` feature is enabled. The struct always exists
/// so that config deserialization works regardless of feature flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionConfig {
    /// Policy for handling detected contradictions between new and existing memories.
    ///
    /// Accepts: "escalate", "keep_existing", "replace_with_new", "keep_both".
    /// Default is "escalate" — ech0 never silently resolves conflicts.
    #[serde(default = "default_resolution_policy")]
    pub resolution_policy: String,

    /// Minimum confidence score to flag a potential contradiction.
    /// Below this threshold, conflicts are ignored as noise.
    /// Range: 0.0–1.0.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
}

fn default_resolution_policy() -> String { "escalate".to_string() }
fn default_confidence_threshold() -> f32 { 0.8 }

impl Default for ContradictionConfig {
    fn default() -> Self {
        Self {
            resolution_policy: "escalate".to_string(),
            confidence_threshold: 0.8,
        }
    }
}

impl ContradictionConfig {
    /// Parse the string-based `resolution_policy` into the typed enum.
    ///
    /// Only available when the `contradiction-detection` feature is enabled, since
    /// `ConflictResolution` lives in the feature-gated `conflict` module.
    #[cfg(feature = "contradiction-detection")]
    pub fn parsed_resolution_policy(&self) -> ConflictResolution {
        match self.resolution_policy.as_str() {
            "keep_existing" => ConflictResolution::KeepExisting,
            "replace_with_new" => ConflictResolution::ReplaceWithNew,
            "keep_both" => ConflictResolution::KeepBoth,
            // Default to Escalate for any unrecognized value — never silently resolve
            _ => ConflictResolution::Escalate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = StoreConfig::default();

        assert_eq!(config.store.graph_path, "./ech0_graph");
        assert_eq!(config.store.vector_path, "./ech0_vectors");
        assert_eq!(config.store.vector_dimensions, 768);
        assert_eq!(config.memory.short_term_capacity, 50);
        assert!((config.memory.episodic_decay_rate - 0.01).abs() < f32::EPSILON);
        assert!((config.memory.semantic_decay_rate - 0.005).abs() < f32::EPSILON);
        assert!((config.memory.prune_threshold - 0.1).abs() < f32::EPSILON);
        assert!((config.memory.importance_boost_on_retrieval - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.dynamic_linking.top_k_candidates, 5);
        assert!((config.dynamic_linking.similarity_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.dynamic_linking.max_links_per_ingest, 10);
        assert_eq!(config.contradiction.resolution_policy, "escalate");
        assert!((config.contradiction.confidence_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = StoreConfig::default();
        let toml_string =
            toml::to_string(&config).expect("default config should serialize to TOML");
        let deserialized: StoreConfig =
            toml::from_str(&toml_string).expect("serialized TOML should deserialize back");

        assert_eq!(deserialized.store.graph_path, config.store.graph_path);
        assert_eq!(
            deserialized.store.vector_dimensions,
            config.store.vector_dimensions
        );
        assert_eq!(
            deserialized.memory.short_term_capacity,
            config.memory.short_term_capacity
        );
        assert_eq!(
            deserialized.dynamic_linking.top_k_candidates,
            config.dynamic_linking.top_k_candidates
        );
        assert_eq!(
            deserialized.contradiction.resolution_policy,
            config.contradiction.resolution_policy
        );
    }

    #[test]
    fn partial_toml_fills_defaults() {
        let partial = r#"
[store]
graph_path = "/custom/path"
"#;
        let config: StoreConfig =
            toml::from_str(partial).expect("partial TOML should deserialize with defaults");

        assert_eq!(config.store.graph_path, "/custom/path");
        // Other fields should be default
        assert_eq!(config.store.vector_path, "./ech0_vectors");
        assert_eq!(config.store.vector_dimensions, 768);
        assert_eq!(config.memory.short_term_capacity, 50);
    }

    #[cfg(feature = "contradiction-detection")]
    #[test]
    fn parsed_resolution_policy_maps_correctly() {
        let mut config = ContradictionConfig::default();

        config.resolution_policy = "escalate".to_string();
        assert!(matches!(
            config.parsed_resolution_policy(),
            ConflictResolution::Escalate
        ));

        config.resolution_policy = "keep_existing".to_string();
        assert!(matches!(
            config.parsed_resolution_policy(),
            ConflictResolution::KeepExisting
        ));

        config.resolution_policy = "replace_with_new".to_string();
        assert!(matches!(
            config.parsed_resolution_policy(),
            ConflictResolution::ReplaceWithNew
        ));

        config.resolution_policy = "keep_both".to_string();
        assert!(matches!(
            config.parsed_resolution_policy(),
            ConflictResolution::KeepBoth
        ));
    }

    #[cfg(feature = "contradiction-detection")]
    #[test]
    fn unknown_resolution_policy_defaults_to_escalate() {
        let mut config = ContradictionConfig::default();
        config.resolution_policy = "nonsense_value".to_string();
        assert!(matches!(
            config.parsed_resolution_policy(),
            ConflictResolution::Escalate
        ));
    }
}
