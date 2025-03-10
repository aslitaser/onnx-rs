use thiserror::Error;
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Protobuf parsing error: {0}")]
    ProtobufError(#[from] prost::DecodeError),
    
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
}