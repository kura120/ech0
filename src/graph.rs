//! redb graph storage layer.
//!
//! Owns all node, edge, and adjacency CRUD operations against the embedded redb database.
//! All writes go through redb transactions. No module other than this one touches redb
//! tables directly — all cross-module access goes through `Store`.
//!
//! Table layout:
//! - `nodes`         — `[u8; 16]` (Uuid) → msgpack-serialized `Node`
//! - `edges`         — `[u8; 16]` (composite key: source+target hash) → msgpack-serialized `Edge`
//! - `adjacency_out` — `[u8; 16]` (node Uuid) → msgpack-serialized `Vec<[u8; 16]>` (outgoing edge keys)
//! - `adjacency_in`  — `[u8; 16]` (node Uuid) → msgpack-serialized `Vec<[u8; 16]>` (incoming edge keys)
//! - `importance`    — `[u8; 16]` (node Uuid) → `f32` bytes (updated independently of full node)
//! - `vector_keys`   — `[u8; 16]` (node Uuid) → `u64` bytes (usearch label mapping)

use std::path::Path;
use std::sync::Arc;

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use crate::error::{EchoError, ErrorContext};
use crate::schema::{Edge, Node};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Nodes table: Uuid bytes → msgpack-serialized Node.
const NODES_TABLE: TableDefinition<&[u8; 16], &[u8]> = TableDefinition::new("nodes");

/// Edges table: deterministic edge key bytes → msgpack-serialized Edge.
const EDGES_TABLE: TableDefinition<&[u8; 16], &[u8]> = TableDefinition::new("edges");

/// Outgoing adjacency: node Uuid → list of edge keys (msgpack `Vec<[u8; 16]>`).
const ADJACENCY_OUT_TABLE: TableDefinition<&[u8; 16], &[u8]> =
    TableDefinition::new("adjacency_out");

/// Incoming adjacency: node Uuid → list of edge keys (msgpack `Vec<[u8; 16]>`).
const ADJACENCY_IN_TABLE: TableDefinition<&[u8; 16], &[u8]> = TableDefinition::new("adjacency_in");

/// Importance scores stored separately so decay/boost writes do not require
/// deserializing the full node.
const IMPORTANCE_TABLE: TableDefinition<&[u8; 16], &[u8; 4]> = TableDefinition::new("importance");

/// Mapping from node Uuid to the `u64` label used in the usearch vector index.
/// Maintained here so the graph layer is the single source of truth for this mapping.
const VECTOR_KEYS_TABLE: TableDefinition<&[u8; 16], &[u8; 8]> = TableDefinition::new("vector_keys");

// ---------------------------------------------------------------------------
// GraphLayer
// ---------------------------------------------------------------------------

/// The redb-backed graph storage layer. Handles all node, edge, and adjacency I/O.
///
/// All public methods on `GraphLayer` operate within redb transactions. A failed
/// write never leaves partial state — the transaction is rolled back automatically
/// by redb if not committed.
pub struct GraphLayer {
    database: Arc<Database>,
}

impl GraphLayer {
    /// Open or create the redb database at the given path.
    ///
    /// Creates all required tables if they do not already exist.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self, EchoError> {
        let database = Database::create(path.as_ref()).map_err(|error| {
            EchoError::storage_failure(format!("failed to open redb database: {error}"))
                .with_context(
                    ErrorContext::new("graph::open")
                        .with_source(&error)
                        .with_field("path", path.as_ref().display().to_string()),
                )
        })?;

        // Ensure all tables exist by opening a write transaction on startup.
        let write_txn = database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin init transaction: {error}"))
                .with_context(ErrorContext::new("graph::open").with_source(&error))
        })?;

        {
            // Opening a table in a write transaction creates it if it does not exist.
            let _nodes = write_txn.open_table(NODES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init nodes table: {error}"))
            })?;
            let _edges = write_txn.open_table(EDGES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init edges table: {error}"))
            })?;
            let _adj_out = write_txn.open_table(ADJACENCY_OUT_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init adjacency_out table: {error}"))
            })?;
            let _adj_in = write_txn.open_table(ADJACENCY_IN_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init adjacency_in table: {error}"))
            })?;
            let _importance = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init importance table: {error}"))
            })?;
            let _vector_keys = write_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to init vector_keys table: {error}"))
            })?;
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit init transaction: {error}"))
                .with_context(ErrorContext::new("graph::open").with_source(&error))
        })?;

        debug!("redb database opened successfully");
        Ok(Self {
            database: Arc::new(database),
        })
    }

    /// Returns a reference to the underlying redb `Database`.
    ///
    /// Exposed so that `Store` can coordinate cross-layer transactions when needed,
    /// but no other module should use this directly.
    pub fn database(&self) -> &Arc<Database> {
        &self.database
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Write a batch of nodes and edges atomically in a single redb transaction.
    ///
    /// Also writes importance scores and adjacency entries. If any step fails,
    /// the entire transaction is rolled back — no partial state is written.
    #[instrument(skip_all, fields(node_count = nodes.len(), edge_count = edges.len()))]
    pub fn write_nodes_and_edges(&self, nodes: &[Node], edges: &[Edge]) -> Result<(), EchoError> {
        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin write transaction: {error}"))
                .with_context(ErrorContext::new("graph::write_nodes_and_edges").with_source(&error))
        })?;

        {
            let mut nodes_table = write_txn.open_table(NODES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open nodes table: {error}"))
            })?;
            let mut edges_table = write_txn.open_table(EDGES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open edges table: {error}"))
            })?;
            let mut adj_out_table = write_txn.open_table(ADJACENCY_OUT_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_out table: {error}"))
            })?;
            let mut adj_in_table = write_txn.open_table(ADJACENCY_IN_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_in table: {error}"))
            })?;
            let mut importance_table = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open importance table: {error}"))
            })?;

            // Write nodes
            for node in nodes {
                let key = node.id.as_bytes();
                let value = rmp_serde::to_vec(node).map_err(|error| {
                    EchoError::storage_failure(format!("failed to serialize node: {error}"))
                        .with_context(
                            ErrorContext::new("graph::write_nodes_and_edges")
                                .with_source(&error)
                                .with_field("node_id", node.id.to_string()),
                        )
                })?;

                nodes_table.insert(key, value.as_slice()).map_err(|error| {
                    EchoError::storage_failure(format!("failed to insert node: {error}"))
                        .with_context(
                            ErrorContext::new("graph::write_nodes_and_edges")
                                .with_field("node_id", node.id.to_string()),
                        )
                })?;

                // Write importance separately for efficient decay/boost updates
                let importance_bytes = node.importance.to_le_bytes();
                importance_table
                    .insert(key, &importance_bytes)
                    .map_err(|error| {
                        EchoError::storage_failure(format!(
                            "failed to insert importance score: {error}"
                        ))
                    })?;
            }

            // Write edges and update adjacency lists
            for edge in edges {
                let edge_key = edge_key(edge.source, edge.target);
                let value = rmp_serde::to_vec(edge).map_err(|error| {
                    EchoError::storage_failure(format!("failed to serialize edge: {error}"))
                        .with_context(
                            ErrorContext::new("graph::write_nodes_and_edges")
                                .with_source(&error)
                                .with_field("source", edge.source.to_string())
                                .with_field("target", edge.target.to_string()),
                        )
                })?;

                edges_table
                    .insert(&edge_key, value.as_slice())
                    .map_err(|error| {
                        EchoError::storage_failure(format!("failed to insert edge: {error}"))
                    })?;

                // Update outgoing adjacency for source node
                append_to_adjacency_list(&mut adj_out_table, edge.source, &edge_key)?;

                // Update incoming adjacency for target node
                append_to_adjacency_list(&mut adj_in_table, edge.target, &edge_key)?;
            }
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit write transaction: {error}"))
                .with_context(ErrorContext::new("graph::write_nodes_and_edges").with_source(&error))
        })?;

        debug!(
            node_count = nodes.len(),
            edge_count = edges.len(),
            "graph write committed"
        );
        Ok(())
    }

    /// Register a mapping from a node Uuid to its usearch vector index label.
    ///
    /// This is written as part of the ingest transaction so the mapping is always
    /// consistent with the graph state. Called by `Store` after vector insertion.
    pub fn write_vector_key(&self, node_id: Uuid, vector_label: u64) -> Result<(), EchoError> {
        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to begin vector_key write transaction: {error}"
            ))
        })?;

        {
            let mut table = write_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
            })?;

            let label_bytes = vector_label.to_le_bytes();
            table
                .insert(node_id.as_bytes(), &label_bytes)
                .map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to insert vector key mapping: {error}"
                    ))
                })?;
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to commit vector_key write transaction: {error}"
            ))
        })?;

        Ok(())
    }

    /// Write multiple vector key mappings in a single transaction.
    pub fn write_vector_keys_batch(&self, mappings: &[(Uuid, u64)]) -> Result<(), EchoError> {
        if mappings.is_empty() {
            return Ok(());
        }

        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to begin vector_keys batch write transaction: {error}"
            ))
        })?;

        {
            let mut table = write_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
            })?;

            for (node_id, vector_label) in mappings {
                let label_bytes = vector_label.to_le_bytes();
                table
                    .insert(node_id.as_bytes(), &label_bytes)
                    .map_err(|error| {
                        EchoError::storage_failure(format!(
                            "failed to insert vector key mapping: {error}"
                        ))
                    })?;
            }
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit vector_keys batch write: {error}"))
        })?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Read a single node by its Uuid.
    ///
    /// Returns `None` if the node does not exist. Returns `EchoError` on I/O
    /// or deserialization failure.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn get_node(&self, node_id: Uuid) -> Result<Option<Node>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
                .with_context(ErrorContext::new("graph::get_node").with_source(&error))
        })?;

        let table = read_txn.open_table(NODES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open nodes table: {error}"))
        })?;

        let key = node_id.as_bytes();
        match table.get(key) {
            Ok(Some(value)) => {
                let node: Node = rmp_serde::from_slice(value.value()).map_err(|error| {
                    EchoError::storage_failure(format!("failed to deserialize node: {error}"))
                        .with_context(
                            ErrorContext::new("graph::get_node")
                                .with_source(&error)
                                .with_field("node_id", node_id.to_string()),
                        )
                })?;
                Ok(Some(node))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(EchoError::storage_failure(format!(
                "failed to read node: {error}"
            ))),
        }
    }

    /// Read a single edge by source and target Uuids.
    ///
    /// Returns `None` if the edge does not exist.
    #[instrument(skip(self), fields(source = %source, target = %target))]
    pub fn get_edge(&self, source: Uuid, target: Uuid) -> Result<Option<Edge>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(EDGES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open edges table: {error}"))
        })?;

        let key = edge_key(source, target);
        match table.get(&key) {
            Ok(Some(value)) => {
                let edge: Edge = rmp_serde::from_slice(value.value()).map_err(|error| {
                    EchoError::storage_failure(format!("failed to deserialize edge: {error}"))
                })?;
                Ok(Some(edge))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(EchoError::storage_failure(format!(
                "failed to read edge: {error}"
            ))),
        }
    }

    /// Get all outgoing edges from a node.
    ///
    /// Returns the deserialized `Edge` objects. Returns an empty vec if the node
    /// has no outgoing edges or does not exist.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn get_outgoing_edges(&self, node_id: Uuid) -> Result<Vec<Edge>, EchoError> {
        self.get_adjacent_edges(node_id, AdjacencyDirection::Outgoing)
    }

    /// Get all incoming edges to a node.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn get_incoming_edges(&self, node_id: Uuid) -> Result<Vec<Edge>, EchoError> {
        self.get_adjacent_edges(node_id, AdjacencyDirection::Incoming)
    }

    /// Read the current importance score for a node.
    ///
    /// Returns the score from the dedicated importance table, which may differ from
    /// the score stored in the full node record if decay/boost has been applied since
    /// the last full node write.
    pub fn get_importance(&self, node_id: Uuid) -> Result<Option<f32>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open importance table: {error}"))
        })?;

        let key = node_id.as_bytes();
        match table.get(key) {
            Ok(Some(value)) => {
                let bytes = value.value();
                let score = f32::from_le_bytes(*bytes);
                Ok(Some(score))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(EchoError::storage_failure(format!(
                "failed to read importance score: {error}"
            ))),
        }
    }

    /// Update the importance score for a node without rewriting the full node record.
    pub fn update_importance(&self, node_id: Uuid, new_score: f32) -> Result<(), EchoError> {
        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin write transaction: {error}"))
        })?;

        {
            let mut table = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open importance table: {error}"))
            })?;

            let key = node_id.as_bytes();
            let bytes = new_score.to_le_bytes();
            table.insert(key, &bytes).map_err(|error| {
                EchoError::storage_failure(format!("failed to update importance score: {error}"))
            })?;
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit importance update: {error}"))
        })?;

        Ok(())
    }

    /// Batch update importance scores in a single transaction.
    pub fn update_importance_batch(&self, updates: &[(Uuid, f32)]) -> Result<(), EchoError> {
        if updates.is_empty() {
            return Ok(());
        }

        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin write transaction: {error}"))
        })?;

        {
            let mut table = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open importance table: {error}"))
            })?;

            for (node_id, new_score) in updates {
                let key = node_id.as_bytes();
                let bytes = new_score.to_le_bytes();
                table.insert(key, &bytes).map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to update importance score: {error}"
                    ))
                })?;
            }
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit batch importance update: {error}"))
        })?;

        Ok(())
    }

    /// Read the usearch vector label for a node.
    pub fn get_vector_label(&self, node_id: Uuid) -> Result<Option<u64>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
        })?;

        let key = node_id.as_bytes();
        match table.get(key) {
            Ok(Some(value)) => {
                let bytes = value.value();
                let label = u64::from_le_bytes(*bytes);
                Ok(Some(label))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(EchoError::storage_failure(format!(
                "failed to read vector label: {error}"
            ))),
        }
    }

    /// Iterate over all nodes in the graph. Used for cold-start vector index rebuild
    /// and for decay/prune operations.
    ///
    /// Returns all nodes. For large graphs this may be expensive — callers should
    /// prefer targeted reads when possible.
    pub fn all_nodes(&self) -> Result<Vec<Node>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(NODES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open nodes table: {error}"))
        })?;

        let mut nodes = Vec::new();
        let iter = table.iter().map_err(|error| {
            EchoError::storage_failure(format!("failed to iterate nodes table: {error}"))
        })?;

        for entry in iter {
            let entry = entry.map_err(|error| {
                EchoError::storage_failure(format!("failed to read node entry: {error}"))
            })?;
            let node: Node = rmp_serde::from_slice(entry.1.value()).map_err(|error| {
                EchoError::storage_failure(format!("failed to deserialize node: {error}"))
            })?;
            nodes.push(node);
        }

        Ok(nodes)
    }

    /// Iterate over all edges in the graph.
    pub fn all_edges(&self) -> Result<Vec<Edge>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(EDGES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open edges table: {error}"))
        })?;

        let mut edges = Vec::new();
        let iter = table.iter().map_err(|error| {
            EchoError::storage_failure(format!("failed to iterate edges table: {error}"))
        })?;

        for entry in iter {
            let entry = entry.map_err(|error| {
                EchoError::storage_failure(format!("failed to read edge entry: {error}"))
            })?;
            let edge: Edge = rmp_serde::from_slice(entry.1.value()).map_err(|error| {
                EchoError::storage_failure(format!("failed to deserialize edge: {error}"))
            })?;
            edges.push(edge);
        }

        Ok(edges)
    }

    /// Iterate over all vector key mappings. Used for cold-start rebuild of the
    /// usearch index from redb.
    pub fn all_vector_keys(&self) -> Result<Vec<(Uuid, u64)>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
        })?;

        let mut mappings = Vec::new();
        let iter = table.iter().map_err(|error| {
            EchoError::storage_failure(format!("failed to iterate vector_keys table: {error}"))
        })?;

        for entry in iter {
            let entry = entry.map_err(|error| {
                EchoError::storage_failure(format!("failed to read vector_keys entry: {error}"))
            })?;
            let uuid = Uuid::from_bytes(*entry.0.value());
            let label = u64::from_le_bytes(*entry.1.value());
            mappings.push((uuid, label));
        }

        Ok(mappings)
    }

    /// Return the total number of nodes in the graph.
    pub fn node_count(&self) -> Result<usize, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let table = read_txn.open_table(NODES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open nodes table: {error}"))
        })?;

        let count = table.len().map_err(|error| {
            EchoError::storage_failure(format!("failed to count nodes: {error}"))
        })?;

        Ok(count as usize)
    }

    // -----------------------------------------------------------------------
    // Delete operations
    // -----------------------------------------------------------------------

    /// Delete a node and all its associated edges, adjacency entries, importance
    /// score, and vector key mapping.
    ///
    /// Used by prune operations. Atomic within a single redb transaction.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub fn delete_node(&self, node_id: Uuid) -> Result<(), EchoError> {
        // First read outgoing and incoming edges so we know what adjacency entries to clean up
        let outgoing_edges = self.get_outgoing_edges(node_id)?;
        let incoming_edges = self.get_incoming_edges(node_id)?;

        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin delete transaction: {error}"))
        })?;

        {
            let mut nodes_table = write_txn.open_table(NODES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open nodes table: {error}"))
            })?;
            let mut edges_table = write_txn.open_table(EDGES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open edges table: {error}"))
            })?;
            let mut adj_out_table = write_txn.open_table(ADJACENCY_OUT_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_out table: {error}"))
            })?;
            let mut adj_in_table = write_txn.open_table(ADJACENCY_IN_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_in table: {error}"))
            })?;
            let mut importance_table = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open importance table: {error}"))
            })?;
            let mut vector_keys_table =
                write_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
                    EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
                })?;

            // Remove the node itself
            let key = node_id.as_bytes();
            nodes_table.remove(key).map_err(|error| {
                EchoError::storage_failure(format!("failed to delete node: {error}"))
            })?;

            // Remove importance score
            importance_table.remove(key).map_err(|error| {
                EchoError::storage_failure(format!("failed to delete importance score: {error}"))
            })?;

            // Remove vector key mapping
            vector_keys_table.remove(key).map_err(|error| {
                EchoError::storage_failure(format!("failed to delete vector key mapping: {error}"))
            })?;

            // Remove outgoing adjacency list
            adj_out_table.remove(key).map_err(|error| {
                EchoError::storage_failure(format!("failed to delete outgoing adjacency: {error}"))
            })?;

            // Remove incoming adjacency list
            adj_in_table.remove(key).map_err(|error| {
                EchoError::storage_failure(format!("failed to delete incoming adjacency: {error}"))
            })?;

            // Remove all outgoing edges and clean up their targets' incoming adjacency
            for edge in &outgoing_edges {
                let ek = edge_key(edge.source, edge.target);
                edges_table.remove(&ek).map_err(|error| {
                    EchoError::storage_failure(format!("failed to delete edge: {error}"))
                })?;
                // Remove this edge from the target's incoming adjacency list
                remove_from_adjacency_list(&mut adj_in_table, edge.target, &ek)?;
            }

            // Remove all incoming edges and clean up their sources' outgoing adjacency
            for edge in &incoming_edges {
                let ek = edge_key(edge.source, edge.target);
                edges_table.remove(&ek).map_err(|error| {
                    EchoError::storage_failure(format!("failed to delete edge: {error}"))
                })?;
                // Remove this edge from the source's outgoing adjacency list
                remove_from_adjacency_list(&mut adj_out_table, edge.source, &ek)?;
            }
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!("failed to commit delete transaction: {error}"))
        })?;

        debug!(node_id = %node_id, "node deleted from graph");
        Ok(())
    }

    /// Delete multiple nodes atomically within a single redb transaction.
    ///
    /// All deletions — nodes, edges, adjacency entries, importance scores, and vector key
    /// mappings — are committed in a single transaction. If the commit fails, no deletions
    /// are persisted and the database is left in its original state.
    ///
    /// Non-existent node IDs are silently skipped; they do not cause an error or corrupt
    /// any other node in the batch.
    #[instrument(skip_all, fields(count = node_ids.len()))]
    pub fn delete_nodes_batch(&self, node_ids: &[Uuid]) -> Result<(), EchoError> {
        if node_ids.is_empty() {
            return Ok(());
        }

        // Build a set of IDs being deleted so we can skip adjacency updates for
        // nodes that are themselves in the batch (they will be fully removed anyway).
        let deleting: std::collections::HashSet<Uuid> = node_ids.iter().cloned().collect();

        // Read phase: collect outgoing and incoming edges for every node before
        // opening the write transaction. Separate read transactions per node are
        // acceptable here — this is pre-collection only.
        let mut node_outgoing: Vec<Vec<Edge>> = Vec::with_capacity(node_ids.len());
        let mut node_incoming: Vec<Vec<Edge>> = Vec::with_capacity(node_ids.len());
        for &node_id in node_ids {
            node_outgoing.push(self.get_outgoing_edges(node_id)?);
            node_incoming.push(self.get_incoming_edges(node_id)?);
        }

        // Write phase: single transaction covering every deletion.
        let write_txn = self.database.begin_write().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin batch delete transaction: {error}"))
        })?;

        {
            let mut nodes_table = write_txn.open_table(NODES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open nodes table: {error}"))
            })?;
            let mut edges_table = write_txn.open_table(EDGES_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open edges table: {error}"))
            })?;
            let mut adj_out_table = write_txn.open_table(ADJACENCY_OUT_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_out table: {error}"))
            })?;
            let mut adj_in_table = write_txn.open_table(ADJACENCY_IN_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open adjacency_in table: {error}"))
            })?;
            let mut importance_table = write_txn.open_table(IMPORTANCE_TABLE).map_err(|error| {
                EchoError::storage_failure(format!("failed to open importance table: {error}"))
            })?;
            let mut vector_keys_table =
                write_txn.open_table(VECTOR_KEYS_TABLE).map_err(|error| {
                    EchoError::storage_failure(format!("failed to open vector_keys table: {error}"))
                })?;

            for (index, &node_id) in node_ids.iter().enumerate() {
                let key = node_id.as_bytes();

                // Remove the node itself
                nodes_table.remove(key).map_err(|error| {
                    EchoError::storage_failure(format!("failed to delete node: {error}"))
                })?;

                // Remove importance score
                importance_table.remove(key).map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to delete importance score: {error}"
                    ))
                })?;

                // Remove vector key mapping
                vector_keys_table.remove(key).map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to delete vector key mapping: {error}"
                    ))
                })?;

                // Remove outgoing adjacency list
                adj_out_table.remove(key).map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to delete outgoing adjacency: {error}"
                    ))
                })?;

                // Remove incoming adjacency list
                adj_in_table.remove(key).map_err(|error| {
                    EchoError::storage_failure(format!(
                        "failed to delete incoming adjacency: {error}"
                    ))
                })?;

                // Remove all outgoing edges and clean up target nodes' incoming adjacency,
                // but only when the target is not also being deleted in this batch.
                for edge in &node_outgoing[index] {
                    let edge_key_bytes = edge_key(edge.source, edge.target);
                    edges_table.remove(&edge_key_bytes).map_err(|error| {
                        EchoError::storage_failure(format!("failed to delete edge: {error}"))
                    })?;
                    if !deleting.contains(&edge.target) {
                        remove_from_adjacency_list(
                            &mut adj_in_table,
                            edge.target,
                            &edge_key_bytes,
                        )?;
                    }
                }

                // Remove all incoming edges and clean up source nodes' outgoing adjacency,
                // but only when the source is not also being deleted in this batch.
                for edge in &node_incoming[index] {
                    let edge_key_bytes = edge_key(edge.source, edge.target);
                    edges_table.remove(&edge_key_bytes).map_err(|error| {
                        EchoError::storage_failure(format!("failed to delete edge: {error}"))
                    })?;
                    if !deleting.contains(&edge.source) {
                        remove_from_adjacency_list(
                            &mut adj_out_table,
                            edge.source,
                            &edge_key_bytes,
                        )?;
                    }
                }
            }
        }

        write_txn.commit().map_err(|error| {
            EchoError::storage_failure(format!(
                "failed to commit batch delete transaction: {error}"
            ))
        })?;

        debug!(count = node_ids.len(), "batch node delete committed");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Read adjacent edges from the specified direction's adjacency table.
    fn get_adjacent_edges(
        &self,
        node_id: Uuid,
        direction: AdjacencyDirection,
    ) -> Result<Vec<Edge>, EchoError> {
        let read_txn = self.database.begin_read().map_err(|error| {
            EchoError::storage_failure(format!("failed to begin read transaction: {error}"))
        })?;

        let adj_table_def = match direction {
            AdjacencyDirection::Outgoing => ADJACENCY_OUT_TABLE,
            AdjacencyDirection::Incoming => ADJACENCY_IN_TABLE,
        };

        let adj_table = read_txn.open_table(adj_table_def).map_err(|error| {
            EchoError::storage_failure(format!("failed to open adjacency table: {error}"))
        })?;

        let edges_table = read_txn.open_table(EDGES_TABLE).map_err(|error| {
            EchoError::storage_failure(format!("failed to open edges table: {error}"))
        })?;

        let key = node_id.as_bytes();
        let edge_keys: Vec<[u8; 16]> = match adj_table.get(key) {
            Ok(Some(value)) => rmp_serde::from_slice(value.value()).map_err(|error| {
                EchoError::storage_failure(format!("failed to deserialize adjacency list: {error}"))
            })?,
            Ok(None) => return Ok(Vec::new()),
            Err(error) => {
                return Err(EchoError::storage_failure(format!(
                    "failed to read adjacency list: {error}"
                )));
            }
        };

        let mut edges = Vec::with_capacity(edge_keys.len());
        for edge_key_bytes in &edge_keys {
            match edges_table.get(edge_key_bytes) {
                Ok(Some(value)) => {
                    let edge: Edge = rmp_serde::from_slice(value.value()).map_err(|error| {
                        EchoError::storage_failure(format!("failed to deserialize edge: {error}"))
                    })?;
                    edges.push(edge);
                }
                Ok(None) => {
                    // Adjacency list references a non-existent edge — consistency issue,
                    // but we log and skip rather than failing the entire read.
                    warn!(
                        node_id = %node_id,
                        "adjacency list references non-existent edge — possible consistency issue"
                    );
                }
                Err(error) => {
                    return Err(EchoError::storage_failure(format!(
                        "failed to read edge from adjacency: {error}"
                    )));
                }
            }
        }

        Ok(edges)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Direction for adjacency lookups.
enum AdjacencyDirection {
    Outgoing,
    Incoming,
}

/// Produce a deterministic 16-byte key for an edge from its source and target Uuids.
///
/// XORs the two Uuids together. This means the key for (A→B) differs from (B→A),
/// which is correct for a directed graph — but requires that we store edges in both
/// adjacency tables (out and in) for efficient bidirectional lookup.
fn edge_key(source: Uuid, target: Uuid) -> [u8; 16] {
    let source_bytes = source.as_bytes();
    let target_bytes = target.as_bytes();
    let mut key = [0u8; 16];
    // Use a simple combination: first 8 bytes from source, last 8 from target.
    // This avoids XOR symmetry issues (where A^B == B^A) while keeping the key fixed-size.
    key[..8].copy_from_slice(&source_bytes[..8]);
    key[8..].copy_from_slice(&target_bytes[8..]);
    key
}

/// Append an edge key to a node's adjacency list within an open write transaction table.
fn append_to_adjacency_list(
    table: &mut redb::Table<&[u8; 16], &[u8]>,
    node_id: Uuid,
    new_edge_key: &[u8; 16],
) -> Result<(), EchoError> {
    let key = node_id.as_bytes();

    let mut edge_keys: Vec<[u8; 16]> = match table.get(key) {
        Ok(Some(value)) => rmp_serde::from_slice(value.value()).map_err(|error| {
            EchoError::storage_failure(format!("failed to deserialize adjacency list: {error}"))
        })?,
        Ok(None) => Vec::new(),
        Err(error) => {
            return Err(EchoError::storage_failure(format!(
                "failed to read adjacency list for append: {error}"
            )));
        }
    };

    edge_keys.push(*new_edge_key);

    let serialized = rmp_serde::to_vec(&edge_keys).map_err(|error| {
        EchoError::storage_failure(format!("failed to serialize adjacency list: {error}"))
    })?;

    table.insert(key, serialized.as_slice()).map_err(|error| {
        EchoError::storage_failure(format!("failed to write adjacency list: {error}"))
    })?;

    Ok(())
}

/// Remove an edge key from a node's adjacency list within an open write transaction table.
fn remove_from_adjacency_list(
    table: &mut redb::Table<&[u8; 16], &[u8]>,
    node_id: Uuid,
    edge_key_to_remove: &[u8; 16],
) -> Result<(), EchoError> {
    let key = node_id.as_bytes();

    let mut edge_keys: Vec<[u8; 16]> = match table.get(key) {
        Ok(Some(value)) => rmp_serde::from_slice(value.value()).map_err(|error| {
            EchoError::storage_failure(format!("failed to deserialize adjacency list: {error}"))
        })?,
        Ok(None) => return Ok(()),
        Err(error) => {
            return Err(EchoError::storage_failure(format!(
                "failed to read adjacency list for removal: {error}"
            )));
        }
    };

    edge_keys.retain(|existing_key| existing_key != edge_key_to_remove);

    let serialized = rmp_serde::to_vec(&edge_keys).map_err(|error| {
        EchoError::storage_failure(format!("failed to serialize adjacency list: {error}"))
    })?;

    table.insert(key, serialized.as_slice()).map_err(|error| {
        EchoError::storage_failure(format!(
            "failed to write adjacency list after removal: {error}"
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::NamedTempFile;

    /// Create a temporary graph layer for testing.
    fn temp_graph() -> (GraphLayer, NamedTempFile) {
        let file = NamedTempFile::new().expect("failed to create temp file");
        let _path = file.path().to_path_buf();
        // redb needs exclusive access — close the temp file handle first then use the path
        drop(file);
        let file = NamedTempFile::new().expect("failed to create temp file");
        let graph = GraphLayer::open(file.path()).expect("failed to open graph layer");
        (graph, file)
    }

    fn make_test_node(ingest_id: Uuid) -> Node {
        Node {
            id: Uuid::new_v4(),
            kind: "test".to_string(),
            metadata: serde_json::json!({"key": "value"}),
            importance: 0.8,
            created_at: Utc::now(),
            ingest_id,
            source_text: None,
        }
    }

    fn make_test_edge(source: Uuid, target: Uuid, ingest_id: Uuid) -> Edge {
        Edge {
            source,
            target,
            relation: "related_to".to_string(),
            metadata: serde_json::json!({}),
            importance: 0.6,
            created_at: Utc::now(),
            ingest_id,
        }
    }

    #[test]
    fn open_creates_database() {
        let (_graph, _file) = temp_graph();
        // If we get here, the database was created successfully
    }

    #[test]
    fn write_and_read_node() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();
        let node = make_test_node(ingest_id);
        let node_id = node.id;

        graph
            .write_nodes_and_edges(&[node.clone()], &[])
            .expect("write should succeed");

        let retrieved = graph
            .get_node(node_id)
            .expect("read should succeed")
            .expect("node should exist");

        assert_eq!(retrieved.id, node_id);
        assert_eq!(retrieved.kind, "test");
        assert_eq!(retrieved.ingest_id, ingest_id);
    }

    #[test]
    fn read_nonexistent_node_returns_none() {
        let (graph, _file) = temp_graph();
        let result = graph
            .get_node(Uuid::new_v4())
            .expect("read should not fail");
        assert!(result.is_none());
    }

    #[test]
    fn write_and_read_edge_with_adjacency() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();
        let node_a = make_test_node(ingest_id);
        let node_b = make_test_node(ingest_id);
        let edge = make_test_edge(node_a.id, node_b.id, ingest_id);

        graph
            .write_nodes_and_edges(&[node_a.clone(), node_b.clone()], &[edge.clone()])
            .expect("write should succeed");

        // Read edge directly
        let retrieved_edge = graph
            .get_edge(node_a.id, node_b.id)
            .expect("read should succeed")
            .expect("edge should exist");
        assert_eq!(retrieved_edge.relation, "related_to");

        // Check outgoing adjacency
        let outgoing = graph
            .get_outgoing_edges(node_a.id)
            .expect("adjacency read should succeed");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].source, node_a.id);
        assert_eq!(outgoing[0].target, node_b.id);

        // Check incoming adjacency
        let incoming = graph
            .get_incoming_edges(node_b.id)
            .expect("adjacency read should succeed");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source, node_a.id);
    }

    #[test]
    fn importance_read_write_independent_of_node() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();
        let node = make_test_node(ingest_id);
        let node_id = node.id;

        graph
            .write_nodes_and_edges(&[node], &[])
            .expect("write should succeed");

        // Read initial importance
        let initial = graph
            .get_importance(node_id)
            .expect("read should succeed")
            .expect("importance should exist");
        assert!((initial - 0.8).abs() < f32::EPSILON);

        // Update importance independently
        graph
            .update_importance(node_id, 0.3)
            .expect("update should succeed");

        let updated = graph
            .get_importance(node_id)
            .expect("read should succeed")
            .expect("importance should still exist");
        assert!((updated - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn vector_key_mapping_round_trip() {
        let (graph, _file) = temp_graph();
        let node_id = Uuid::new_v4();
        let label: u64 = 42;

        graph
            .write_vector_key(node_id, label)
            .expect("write should succeed");

        let retrieved = graph
            .get_vector_label(node_id)
            .expect("read should succeed")
            .expect("label should exist");
        assert_eq!(retrieved, label);
    }

    #[test]
    fn all_nodes_returns_all_written() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();
        let nodes: Vec<Node> = (0..5).map(|_| make_test_node(ingest_id)).collect();

        graph
            .write_nodes_and_edges(&nodes, &[])
            .expect("write should succeed");

        let all = graph.all_nodes().expect("all_nodes should succeed");
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn delete_node_removes_everything() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();
        let node_a = make_test_node(ingest_id);
        let node_b = make_test_node(ingest_id);
        let edge = make_test_edge(node_a.id, node_b.id, ingest_id);

        graph
            .write_nodes_and_edges(&[node_a.clone(), node_b.clone()], &[edge])
            .expect("write should succeed");

        graph
            .write_vector_key(node_a.id, 100)
            .expect("vector key write should succeed");

        // Delete node_a
        graph.delete_node(node_a.id).expect("delete should succeed");

        // Node should be gone
        assert!(
            graph
                .get_node(node_a.id)
                .expect("read should succeed")
                .is_none()
        );

        // Importance should be gone
        assert!(
            graph
                .get_importance(node_a.id)
                .expect("read should succeed")
                .is_none()
        );

        // Vector key should be gone
        assert!(
            graph
                .get_vector_label(node_a.id)
                .expect("read should succeed")
                .is_none()
        );

        // Edge should be gone
        assert!(
            graph
                .get_edge(node_a.id, node_b.id)
                .expect("read should succeed")
                .is_none()
        );

        // node_b's incoming adjacency should be empty
        let incoming = graph
            .get_incoming_edges(node_b.id)
            .expect("adjacency read should succeed");
        assert!(incoming.is_empty());

        // node_b should still exist
        assert!(
            graph
                .get_node(node_b.id)
                .expect("read should succeed")
                .is_some()
        );
    }

    #[test]
    fn node_count_is_accurate() {
        let (graph, _file) = temp_graph();
        assert_eq!(
            graph.node_count().expect("count should succeed"),
            0,
            "empty graph should have 0 nodes"
        );

        let ingest_id = Uuid::new_v4();
        let nodes: Vec<Node> = (0..3).map(|_| make_test_node(ingest_id)).collect();
        graph
            .write_nodes_and_edges(&nodes, &[])
            .expect("write should succeed");

        assert_eq!(graph.node_count().expect("count should succeed"), 3);
    }

    #[test]
    fn edge_key_is_directional() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let key_ab = edge_key(a, b);
        let key_ba = edge_key(b, a);
        assert_ne!(
            key_ab, key_ba,
            "edge key should differ for opposite directions"
        );
    }

    /// Verify that delete_nodes_batch handles non-existent node IDs without corrupting
    /// valid nodes. Because all deletions occur in a single redb transaction, any failure
    /// rolls back the entire batch — no partial state is written.
    ///
    /// This test exercises the common case: a batch that includes both valid and
    /// non-existent IDs. Neither the valid nodes nor any adjacency entries should be
    /// left in an inconsistent state.
    #[test]
    fn delete_nodes_batch_is_atomic_on_failure() {
        let (graph, _file) = temp_graph();
        let ingest_id = Uuid::new_v4();

        // Set up: two real nodes connected by an edge.
        let node_a = make_test_node(ingest_id);
        let node_b = make_test_node(ingest_id);
        let edge = make_test_edge(node_a.id, node_b.id, ingest_id);

        graph
            .write_nodes_and_edges(&[node_a.clone(), node_b.clone()], &[edge])
            .expect("write should succeed");

        // Include a never-written ID in the batch alongside the two real nodes.
        let phantom_id = Uuid::new_v4();
        let batch = [node_a.id, phantom_id, node_b.id];

        // The batch should succeed — missing nodes are silently skipped.
        graph
            .delete_nodes_batch(&batch)
            .expect("batch delete with non-existent ID should not error");

        // Both real nodes must be gone.
        assert!(
            graph
                .get_node(node_a.id)
                .expect("read should succeed")
                .is_none(),
            "node_a should be deleted"
        );
        assert!(
            graph
                .get_node(node_b.id)
                .expect("read should succeed")
                .is_none(),
            "node_b should be deleted"
        );

        // The edge between them must be gone.
        assert!(
            graph
                .get_edge(node_a.id, node_b.id)
                .expect("read should succeed")
                .is_none(),
            "edge should be deleted"
        );

        // The phantom node never existed — verify nothing was inserted for it.
        assert!(
            graph
                .get_node(phantom_id)
                .expect("read should succeed")
                .is_none(),
            "phantom node should not exist"
        );
        assert!(
            graph
                .get_importance(phantom_id)
                .expect("read should succeed")
                .is_none(),
            "phantom importance should not exist"
        );
    }
}
