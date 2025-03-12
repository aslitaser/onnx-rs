use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    /// Enable fine-grained tensor locking (more concurrent operations but higher overhead)
    pub enable_fine_grained_tensor_locking: bool,
    /// Cancel all operations if any operation fails
    pub cancel_on_error: bool,
    /// Timeout for operations in milliseconds (0 = no timeout)
    pub operation_timeout_ms: u64,
    /// Dynamically rebalance work based on execution times
    pub enable_dynamic_rebalancing: bool,
    /// Maximum number of operations to run concurrently (0 = no limit)
    pub max_concurrent_operations: usize,
    /// Lock acquisition timeout in milliseconds (0 = no timeout)
    pub lock_timeout_ms: u64,
    /// Enable advanced scheduling of operations based on critical path
    pub enable_critical_path_scheduling: bool,
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
            enable_fine_grained_tensor_locking: false,
            cancel_on_error: false,
            operation_timeout_ms: 0, // No timeout
            enable_dynamic_rebalancing: true,
            max_concurrent_operations: 0, // No limit
            lock_timeout_ms: 5000, // 5 second timeout
            enable_critical_path_scheduling: true,
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
    
    /// Enable or disable fine-grained tensor locking
    pub fn enable_fine_grained_tensor_locking(mut self, enable: bool) -> Self {
        self.enable_fine_grained_tensor_locking = enable;
        self
    }
    
    /// Set behavior for cancellation on error
    pub fn set_cancel_on_error(mut self, cancel_on_error: bool) -> Self {
        self.cancel_on_error = cancel_on_error;
        self
    }
    
    /// Set operation timeout in milliseconds (0 = no timeout)
    pub fn set_operation_timeout(mut self, timeout_ms: u64) -> Self {
        self.operation_timeout_ms = timeout_ms;
        self
    }
    
    /// Enable or disable dynamic rebalancing of work
    pub fn enable_dynamic_rebalancing(mut self, enable: bool) -> Self {
        self.enable_dynamic_rebalancing = enable;
        self
    }
    
    /// Set the maximum number of concurrent operations (0 = no limit)
    pub fn set_max_concurrent_operations(mut self, max_ops: usize) -> Self {
        self.max_concurrent_operations = max_ops;
        self
    }
    
    /// Set lock acquisition timeout in milliseconds (0 = no timeout)
    pub fn set_lock_timeout(mut self, timeout_ms: u64) -> Self {
        self.lock_timeout_ms = timeout_ms;
        self
    }
    
    /// Enable or disable critical path scheduling
    pub fn enable_critical_path_scheduling(mut self, enable: bool) -> Self {
        self.enable_critical_path_scheduling = enable;
        self
    }
}

/// Thread-safe workspace memory storage
struct ThreadSafeWorkspace {
    /// The raw workspace memory
    data: UnsafeCell<Vec<u8>>,
    /// Lock for exclusive access to the workspace
    in_use: AtomicBool,
    /// Current size in bytes
    size: AtomicUsize,
}

// Manually implement Send and Sync since we're using UnsafeCell
unsafe impl Send for ThreadSafeWorkspace {}
unsafe impl Sync for ThreadSafeWorkspace {}

impl ThreadSafeWorkspace {
    /// Create a new thread-safe workspace
    fn new(initial_size: usize) -> Self {
        Self {
            data: UnsafeCell::new(Vec::with_capacity(initial_size)),
            in_use: AtomicBool::new(false),
            size: AtomicUsize::new(0),
        }
    }
    
    /// Ensure the workspace is at least the specified size
    fn ensure_size(&self, size_bytes: usize) -> Result<()> {
        let current_size = self.size.load(Ordering::Acquire);
        
        if size_bytes > current_size {
            // We need to resize
            let data = unsafe { &mut *self.data.get() };
            
            if data.capacity() < size_bytes {
                data.reserve(size_bytes - data.capacity());
            }
            
            // Update the size
            data.resize(size_bytes, 0);
            self.size.store(size_bytes, Ordering::Release);
        }
        
        Ok(())
    }
    
    /// Try to acquire the workspace
    fn try_acquire(&self) -> bool {
        !self.in_use.swap(true, Ordering::AcqRel)
    }
    
    /// Release the workspace
    fn release(&self) {
        self.in_use.store(false, Ordering::Release);
    }
    
    /// Get a raw pointer to the workspace data
    fn get_data_ptr(&self) -> *mut u8 {
        let data = unsafe { &mut *self.data.get() };
        data.as_mut_ptr()
    }
}

/// Workspace memory guard
pub struct WorkspaceGuard {
    /// Pointer to the workspace memory
    ptr: *mut u8,
    /// Size of the workspace allocation
    size: usize,
    /// Reference to the workspace for release
    workspace: Arc<ThreadSafeWorkspace>,
}

// Manually implement Send and Sync since we're using raw pointers
unsafe impl Send for WorkspaceGuard {}
unsafe impl Sync for WorkspaceGuard {}

impl WorkspaceGuard {
    /// Create a new workspace guard
    fn new(ptr: *mut u8, size: usize, workspace: Arc<ThreadSafeWorkspace>) -> Self {
        Self { ptr, size, workspace }
    }
    
    /// Get a mutable slice to the workspace
    pub fn data(&mut self) -> &mut [u8] {
        if self.ptr.is_null() {
            // Return an empty slice if the pointer is null
            return &mut [];
        }
        
        // SAFETY: The WorkspaceGuard ensures that:
        // 1. The pointer is valid for the lifetime of the guard
        // 2. The memory pointed to is properly initialized
        // 3. The memory range [ptr, ptr+size) is owned exclusively by this guard
        // 4. Alignment requirements are satisfied by the allocator
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
    
    /// Get size of the workspace
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        // Release the workspace when the guard is dropped
        self.workspace.release();
    }
}

/// Execution context for the model
pub struct ExecutionContext {
    /// Tensors available during execution (protected by a RwLock for concurrent read access)
    tensors: RwLock<HashMap<TensorId, Tensor>>,
    /// Thread-safe workspace memory for temporary allocations
    workspace: Arc<ThreadSafeWorkspace>,
    /// Operator caches for stateful operators (protected by a mutex)
    operator_caches: Mutex<HashMap<NodeId, HashMap<String, Box<dyn Any + Send + Sync>>>>,
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
            tensors: RwLock::new(HashMap::new()),
            workspace: Arc::new(ThreadSafeWorkspace::new(options.workspace_size_bytes)),
            operator_caches: Mutex::new(HashMap::new()),
            options,
            thread_pool,
        }
    }
    
    /// Get a tensor by ID (thread-safe read access)
    pub fn get_tensor(&self, tensor_id: &TensorId) -> Result<Option<Tensor>> {
        match self.tensors.read() {
            Ok(tensors) => Ok(tensors.get(tensor_id).cloned()),
            Err(e) => Err(Error::InvalidModel(format!(
                "Failed to acquire read lock for tensors: {}", e
            )))
        }
    }
    
    /// Try to get a tensor by ID, with a simplified interface that returns None on lock failure
    /// This is useful for non-critical operations where lock failure is acceptable
    pub fn try_get_tensor(&self, tensor_id: &TensorId) -> Option<Tensor> {
        self.tensors.read().ok()
            .and_then(|tensors| tensors.get(tensor_id).cloned())
    }
    
    /// Get a tensor by ID for updating its value (semantically the same as get_tensor but with a name
    /// that indicates the intent is to modify the tensor after retrieval)
    pub fn get_tensor_for_update(&self, tensor_id: &TensorId) -> Result<Option<Tensor>> {
        match self.tensors.read() {
            Ok(tensors) => Ok(tensors.get(tensor_id).cloned()),
            Err(e) => Err(Error::InvalidModel(format!(
                "Failed to acquire read lock for tensors: {}", e
            )))
        }
    }
    
    /// Set a tensor by ID (thread-safe write access)
    pub fn set_tensor(&self, tensor_id: TensorId, tensor: Tensor) -> Result<()> {
        if let Ok(mut tensors) = self.tensors.write() {
            tensors.insert(tensor_id, tensor);
            Ok(())
        } else {
            Err(Error::InvalidModel("Failed to acquire write lock for tensors".to_string()))
        }
    }
    
    /// Remove a tensor by ID (thread-safe write access)
    pub fn remove_tensor(&self, tensor_id: &TensorId) -> Result<Option<Tensor>> {
        if let Ok(mut tensors) = self.tensors.write() {
            Ok(tensors.remove(tensor_id))
        } else {
            Err(Error::InvalidModel("Failed to acquire write lock for tensors".to_string()))
        }
    }
    
    /// Check if a tensor exists (thread-safe read access)
    pub fn has_tensor(&self, tensor_id: &TensorId) -> bool {
        if let Ok(tensors) = self.tensors.read() {
            tensors.contains_key(tensor_id)
        } else {
            false
        }
    }
    
    /// Get a workspace of the specified size (thread-safe)
    pub fn get_workspace(&self, size_bytes: usize) -> Result<WorkspaceGuard> {
        // Check workspace size limit
        if size_bytes > self.options.workspace_size_bytes {
            return Err(Error::InvalidModel(format!(
                "Requested workspace size ({} bytes) exceeds limit ({} bytes)",
                size_bytes, self.options.workspace_size_bytes
            )));
        }
        
        // Try to acquire the workspace
        if !self.workspace.try_acquire() {
            return Err(Error::InvalidModel(
                "Workspace is already in use by another thread".to_string()
            ));
        }
        
        // Ensure the workspace is large enough
        self.workspace.ensure_size(size_bytes)?;
        
        // Get a pointer to the workspace memory
        let ptr = self.workspace.get_data_ptr();
        
        // Return a guard with the workspace
        Ok(WorkspaceGuard::new(ptr, size_bytes, self.workspace.clone()))
    }
    
    /// Get the operator cache for a node (thread-safe)
    pub fn get_operator_cache(&self, node_id: NodeId) -> Result<HashMap<String, Box<dyn Any + Send + Sync>>> {
        if let Ok(mut caches) = self.operator_caches.lock() {
            if !caches.contains_key(&node_id) {
                caches.insert(node_id, HashMap::new());
            }
            
            // Clone the cache for the node
            Ok(caches.get(&node_id)
                .expect("Node cache should exist")
                .clone())
        } else {
            Err(Error::InvalidModel("Failed to acquire lock for operator caches".to_string()))
        }
    }
    
    /// Update the operator cache for a node (thread-safe)
    pub fn update_operator_cache(&self, node_id: NodeId, cache: HashMap<String, Box<dyn Any + Send + Sync>>) -> Result<()> {
        if let Ok(mut caches) = self.operator_caches.lock() {
            caches.insert(node_id, cache);
            Ok(())
        } else {
            Err(Error::InvalidModel("Failed to acquire lock for operator caches".to_string()))
        }
    }
    
    /// Get execution options
    pub fn options(&self) -> &ExecutionOptions {
        &self.options
    }
    
    /// Get thread pool
    pub fn thread_pool(&self) -> Option<&rayon::ThreadPool> {
        self.thread_pool.as_ref().map(|tp| tp.as_ref())
    }
    
    /// Clear all tensors (thread-safe)
    pub fn clear_tensors(&self) -> Result<()> {
        if let Ok(mut tensors) = self.tensors.write() {
            tensors.clear();
            Ok(())
        } else {
            Err(Error::InvalidModel("Failed to acquire write lock for tensors".to_string()))
        }
    }
    
    /// Clear caches for all operators (thread-safe)
    pub fn clear_operator_caches(&self) -> Result<()> {
        if let Ok(mut caches) = self.operator_caches.lock() {
            caches.clear();
            Ok(())
        } else {
            Err(Error::InvalidModel("Failed to acquire lock for operator caches".to_string()))
        }
    }
    
    /// Get all tensor IDs (thread-safe read access)
    pub fn tensor_ids(&self) -> Result<Vec<TensorId>> {
        if let Ok(tensors) = self.tensors.read() {
            Ok(tensors.keys().cloned().collect())
        } else {
            Err(Error::InvalidModel("Failed to acquire read lock for tensors".to_string()))
        }
    }
    
    /// Get number of tensors (thread-safe read access)
    pub fn tensor_count(&self) -> Result<usize> {
        if let Ok(tensors) = self.tensors.read() {
            Ok(tensors.len())
        } else {
            Err(Error::InvalidModel("Failed to acquire read lock for tensors".to_string()))
        }
    }
}