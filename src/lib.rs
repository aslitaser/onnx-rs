pub mod parser;
pub mod error;
pub mod model;
pub mod proto;
pub mod ops;
pub mod execution;
pub mod optimization;
pub mod memory;
pub mod layout;

// Re-export commonly used types
pub use model::{OnnxModel, Node, Graph, NodeId, Tensor, TensorInfo, ModelMetadata, ExecutionGraph, Subgraph};
pub use error::{Error, Result};
pub use ops::tensor::Tensor as ComputeTensor;
pub use ops::registry::{Operator, OperatorRegistry};
pub use execution::engine::{ExecutionEngine, ThreadSafeTensorCache, ExecutionPriority};
pub use execution::context::{ExecutionContext, ExecutionOptions, OptimizationLevel, WorkspaceGuard};
pub use optimization::graph_optimizer::{GraphOptimizer, OptimizationPass, PassResult, OptimizationStats};
pub use memory::{MemoryAllocator, MemoryBlock, SystemAllocator, ArenaAllocator, PoolAllocator, create_default_allocator};
pub use layout::TensorLayout;