use thiserror::Error;
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err.to_string())
    }
}

impl From<prost::DecodeError> for Error {
    fn from(err: prost::DecodeError) -> Self {
        Error::ProtobufError(err.to_string())
    }
}

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("I/O error: {0}")]
    IoError(String),
    
    #[error("Protobuf parsing error: {0}")]
    ProtobufError(String),
    
    #[error("Invalid ONNX model: {0}")]
    InvalidModel(String),
    
    #[error("Invalid operator: {0}")]
    InvalidOperator(String),
    
    #[error("Version incompatibility: {0}")]
    VersionIncompatible(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),
    
    #[error("Failed to load model from {0}: {1}")]
    ModelLoadError(PathBuf, String),
    
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Execution was cancelled: {0}")]
    ExecutionCancelled(String),
    
    #[error("Operation timed out: {0}")]
    OperationTimeout(String),
    
    #[error("Lock acquisition failed: {0}")]
    LockAcquisitionError(String),
    
    #[error("Concurrent execution error: {0}")]
    ConcurrencyError(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}