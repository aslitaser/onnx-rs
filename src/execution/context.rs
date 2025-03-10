use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::model::{NodeId, TensorId};
use crate::ops::tensor::Tensor;

/// Optimization level for graph execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations like constant folding
    Basic,
    /// Standard set of optimizations (default)
    Standard,
    /// Aggressive optimizations that may change numerical precision
    Aggressive,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Standard
    }
}

/// Options for execution engine
#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    /// Number of threads to use (0 = use system default)
    pub thread_count: usize,
    /// Memory limit in bytes (0 = no limit)
    pub memory_limit_bytes: usize,
    /// Enable profiling of operator execution times
    pub enable_profiling: bool,
    /// Level of optimization to apply
    pub optimization_level: OptimizationLevel,
    /// Maximum temporary memory buffer size
    pub workspace_size_bytes: usize,
    /// Enable operator fusion
    pub enable_operator_fusion: bool,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            thread_count: 0, // Use system default
            memory_limit_bytes: 0, // No limit
            enable_profiling: false,
            optimization_level: OptimizationLevel::Standard,
            workspace_size_bytes: 64 * 1024 * 1024, // 64MB
            enable_operator_fusion: true,
            enable_memory_optimization: true,
        }
    }
}

impl ExecutionOptions {
    /// Create a new execution options object
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the number of threads to use
    pub fn set_thread_count(mut self, thread_count: usize) -> Self {
        self.thread_count = thread_count;
        self
    }
    
    /// Set the memory limit
    pub fn set_memory_limit(mut self, memory_limit_bytes: usize) -> Self {
        self.memory_limit_bytes = memory_limit_bytes;
        self
    }
    
    /// Enable or disable profiling
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }
    
    /// Set the optimization level
    pub fn set_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    /// Set the workspace size
    pub fn set_workspace_size(mut self, workspace_size_bytes: usize) -> Self {
        self.workspace_size_bytes = workspace_size_bytes;
        self
    }
    
    /// Enable or disable operator fusion
    pub fn enable_operator_fusion(mut self, enable: bool) -> Self {
        self.enable_operator_fusion = enable;
        self
    }
    
    /// Enable or disable memory optimization
    pub fn enable_memory_optimization(mut self, enable: bool) -> Self {
        self.enable_memory_optimization = enable;
        self
    }
}

/// Workspace memory guard
pub struct WorkspaceGuard<'a> {
    data: &'a mut [u8],
}

impl<'a> WorkspaceGuard<'a> {
    /// Create a new workspace guard
    pub(crate) fn new(data: &'a mut [u8]) -> Self {
        Self { data }
    }
    
    /// Get a mutable slice to the workspace
    pub fn data(&mut self) -> &mut [u8] {
        self.data
    }
    
    /// Get size of the workspace
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Execution context for the model
pub struct ExecutionContext {
    /// Tensors available during execution
    tensors: HashMap<TensorId, Tensor>,
    /// Workspace memory for temporary allocations
    workspace: Vec<u8>,
    /// Operator caches for stateful operators
    operator_caches: HashMap<NodeId, HashMap<String, Box<dyn Any + Send + Sync>>>,
    /// Options for execution
    options: ExecutionOptions,
    /// Thread pool for parallel execution
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new(options: ExecutionOptions) -> Self {
        // Create thread pool if thread count is specified
        let thread_pool = if options.thread_count > 0 {
            let builder = rayon::ThreadPoolBuilder::new()
                .num_threads(options.thread_count);
            Some(Arc::new(builder.build().unwrap()))
        } else {
            None
        };
        
        Self {
            tensors: HashMap::new(),
            workspace: Vec::with_capacity(options.workspace_size_bytes),
            operator_caches: HashMap::new(),
            options,
            thread_pool,
        }
    }
    
    /// Get a tensor by ID
    pub fn get_tensor(&self, tensor_id: &TensorId) -> Option<&Tensor> {
        self.tensors.get(tensor_id)
    }
    
    /// Get a mutable tensor by ID
    pub fn get_tensor_mut(&mut self, tensor_id: &TensorId) -> Option<&mut Tensor> {
        self.tensors.get_mut(tensor_id)
    }
    
    /// Set a tensor by ID
    pub fn set_tensor(&mut self, tensor_id: TensorId, tensor: Tensor) {
        self.tensors.insert(tensor_id, tensor);
    }
    
    /// Remove a tensor by ID
    pub fn remove_tensor(&mut self, tensor_id: &TensorId) -> Option<Tensor> {
        self.tensors.remove(tensor_id)
    }
    
    /// Check if a tensor exists
    pub fn has_tensor(&self, tensor_id: &TensorId) -> bool {
        self.tensors.contains_key(tensor_id)
    }
    
    /// Get a workspace of the specified size
    pub fn get_workspace(&mut self, size_bytes: usize) -> Result<WorkspaceGuard> {
        if size_bytes > self.options.workspace_size_bytes {
            return Err(Error::InvalidModel(format!(
                "Requested workspace size ({} bytes) exceeds limit ({} bytes)",
                size_bytes, self.options.workspace_size_bytes
            )));
        }
        
        // Ensure the workspace is large enough
        if self.workspace.len() < size_bytes {
            self.workspace.resize(size_bytes, 0);
        }
        
        Ok(WorkspaceGuard::new(&mut self.workspace[..size_bytes]))
    }
    
    /// Get the operator cache for a node
    pub fn get_operator_cache(&mut self, node_id: NodeId) -> &mut HashMap<String, Box<dyn Any + Send + Sync>> {
        self.operator_caches.entry(node_id).or_insert_with(HashMap::new)
    }
    
    /// Get execution options
    pub fn options(&self) -> &ExecutionOptions {
        &self.options
    }
    
    /// Get thread pool
    pub fn thread_pool(&self) -> Option<&rayon::ThreadPool> {
        self.thread_pool.as_ref().map(|tp| tp.as_ref())
    }
    
    /// Clear all tensors
    pub fn clear_tensors(&mut self) {
        self.tensors.clear();
    }
    
    /// Clear caches for all operators
    pub fn clear_operator_caches(&mut self) {
        self.operator_caches.clear();
    }
    
    /// Get all tensor IDs
    pub fn tensor_ids(&self) -> impl Iterator<Item = &TensorId> {
        self.tensors.keys()
    }
    
    /// Get number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}