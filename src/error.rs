//! Error types for ech0.
//!
//! Every fallible operation in ech0 returns `EchoError`. Errors are never swallowed —
//! they are either handled with a recovery path or propagated with full context.

use std::fmt;

/// Typed error returned by all ech0 operations.
///
/// `message` is always safe to surface to callers.
/// `context` is internal debug information — never exposed across the crate boundary.
#[derive(Debug, Clone)]
pub struct EchoError {
    pub code: ErrorCode,
    pub message: String,
    /// Internal debug context. Never expose this to callers across the crate boundary.
    pub context: Option<ErrorContext>,
}

impl EchoError {
    /// Create a new error with the given code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: None,
        }
    }

    /// Attach internal debug context to this error.
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Some(context);
        self
    }

    // -- Convenience constructors per error code --

    pub fn storage_failure(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::StorageFailure, message)
    }

    pub fn embedder_failure(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::EmbedderFailure, message)
    }

    pub fn extractor_failure(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ExtractorFailure, message)
    }

    pub fn consistency_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ConsistencyError, message)
    }

    pub fn conflict_unresolved(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ConflictUnresolved, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidInput, message)
    }

    pub fn capacity_exceeded(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::CapacityExceeded, message)
    }
}

impl fmt::Display for EchoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ech0 error [{:?}]: {}", self.code, self.message)
    }
}

impl std::error::Error for EchoError {}

/// Classification of error origin. Callers can match on this to decide recovery strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// redb or usearch I/O failure, file corruption, transaction failure.
    StorageFailure,
    /// The caller-provided `Embedder` returned an error or produced invalid output.
    EmbedderFailure,
    /// The caller-provided `Extractor` returned an error or produced invalid output.
    ExtractorFailure,
    /// Graph and vector layers are inconsistent — should not happen during normal operation.
    ConsistencyError,
    /// A contradiction was detected and the resolution policy is `Escalate`.
    ConflictUnresolved,
    /// Caller provided invalid arguments (empty text, wrong dimensions, etc.).
    InvalidInput,
    /// A configured capacity limit was reached (e.g. short-term tier full).
    CapacityExceeded,
}

/// Internal debug context attached to errors. Contains information useful for
/// debugging but that should never be surfaced to callers across the crate boundary.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The module or function where the error originated.
    pub location: String,
    /// Upstream error message if this error wraps another.
    pub source_error: Option<String>,
    /// Arbitrary key-value pairs for structured debugging.
    pub fields: Vec<(String, String)>,
}

impl ErrorContext {
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            source_error: None,
            fields: Vec::new(),
        }
    }

    pub fn with_source(mut self, source: impl fmt::Display) -> Self {
        self.source_error = Some(source.to_string());
        self
    }

    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.push((key.into(), value.into()));
        self
    }
}

// -- Conversions from upstream error types --

impl From<redb::DatabaseError> for EchoError {
    fn from(error: redb::DatabaseError) -> Self {
        EchoError::storage_failure(format!("redb database error: {error}"))
            .with_context(ErrorContext::new("graph").with_source(&error))
    }
}

impl From<redb::TableError> for EchoError {
    fn from(error: redb::TableError) -> Self {
        EchoError::storage_failure(format!("redb table error: {error}"))
            .with_context(ErrorContext::new("graph").with_source(&error))
    }
}

impl From<redb::TransactionError> for EchoError {
    fn from(error: redb::TransactionError) -> Self {
        EchoError::storage_failure(format!("redb transaction error: {error}"))
            .with_context(ErrorContext::new("graph").with_source(&error))
    }
}

impl From<redb::StorageError> for EchoError {
    fn from(error: redb::StorageError) -> Self {
        EchoError::storage_failure(format!("redb storage error: {error}"))
            .with_context(ErrorContext::new("graph").with_source(&error))
    }
}

impl From<redb::CommitError> for EchoError {
    fn from(error: redb::CommitError) -> Self {
        EchoError::storage_failure(format!("redb commit error: {error}"))
            .with_context(ErrorContext::new("graph").with_source(&error))
    }
}

impl From<rmp_serde::encode::Error> for EchoError {
    fn from(error: rmp_serde::encode::Error) -> Self {
        EchoError::storage_failure(format!("msgpack encode error: {error}"))
            .with_context(ErrorContext::new("serialization").with_source(&error))
    }
}

impl From<rmp_serde::decode::Error> for EchoError {
    fn from(error: rmp_serde::decode::Error) -> Self {
        EchoError::storage_failure(format!("msgpack decode error: {error}"))
            .with_context(ErrorContext::new("serialization").with_source(&error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_code_and_message() {
        let error = EchoError::new(ErrorCode::InvalidInput, "text must not be empty");
        let display = format!("{error}");
        assert!(
            display.contains("InvalidInput"),
            "display should contain error code"
        );
        assert!(
            display.contains("text must not be empty"),
            "display should contain message"
        );
    }

    #[test]
    fn error_with_context_preserves_all_fields() {
        let error = EchoError::storage_failure("write failed").with_context(
            ErrorContext::new("graph::write_nodes")
                .with_source("disk full")
                .with_field("node_count", "5"),
        );

        let context = error.context.expect("context should be present");
        assert_eq!(context.location, "graph::write_nodes");
        assert_eq!(context.source_error.as_deref(), Some("disk full"));
        assert_eq!(context.fields.len(), 1);
        assert_eq!(
            context.fields[0],
            ("node_count".to_string(), "5".to_string())
        );
    }

    #[test]
    fn convenience_constructors_set_correct_code() {
        assert_eq!(
            EchoError::storage_failure("x").code,
            ErrorCode::StorageFailure
        );
        assert_eq!(
            EchoError::embedder_failure("x").code,
            ErrorCode::EmbedderFailure
        );
        assert_eq!(
            EchoError::extractor_failure("x").code,
            ErrorCode::ExtractorFailure
        );
        assert_eq!(
            EchoError::consistency_error("x").code,
            ErrorCode::ConsistencyError
        );
        assert_eq!(
            EchoError::conflict_unresolved("x").code,
            ErrorCode::ConflictUnresolved
        );
        assert_eq!(EchoError::invalid_input("x").code, ErrorCode::InvalidInput);
        assert_eq!(
            EchoError::capacity_exceeded("x").code,
            ErrorCode::CapacityExceeded
        );
    }

    #[test]
    fn error_implements_std_error() {
        let error = EchoError::new(ErrorCode::StorageFailure, "test");
        // Verify the Error trait is implemented by using it as a trait object
        let _dyn_error: &dyn std::error::Error = &error;
    }
}
