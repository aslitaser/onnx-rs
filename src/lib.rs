pub mod parser;
pub mod error;
pub mod model;
pub mod proto;
pub mod ops;
pub mod execution;
pub mod optimization;

// Re-export commonly used types
pub use model::{OnnxModel, Node, Graph, NodeId, Tensor, TensorInfo, ModelMetadata, ExecutionGraph, Subgraph};
pub use error::{Error, Result};
pub use ops::tensor::Tensor as ComputeTensor;
pub use ops::registry::{Operator, OperatorRegistry};
pub use execution::engine::ExecutionEngine;
pub use execution::context::{ExecutionContext, ExecutionOptions, OptimizationLevel, WorkspaceGuard};
pub use optimization::graph_optimizer::{GraphOptimizer, OptimizationPass, PassResult, OptimizationStats};