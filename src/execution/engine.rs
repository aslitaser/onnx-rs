use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;
use std::sync::{Arc, RwLock, Mutex, Condvar, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime};

use crate::error::{Error, Result};
use crate::model::{ExecutionGraph, NodeId, OnnxModel, TensorId};
use crate::ops::{Operator, OperatorRegistry, Tensor};
use crate::optimization::graph_optimizer::GraphOptimizer;

use super::context::{ExecutionContext, ExecutionOptions, WorkspaceGuard};

/// Unique identifier for a profiling event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProfileEventId(usize);

/// Profiling event data
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Event identifier
    pub id: ProfileEventId,
    /// Event name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// End time (if completed)
    pub end_time: Option<Instant>,
    /// Duration (if completed)
    pub duration: Option<Duration>,
    /// Associated node ID (if applicable)
    pub node_id: Option<NodeId>,
}

/// A thread-safe execution engine for ONNX models
///
/// This structure implements thread-safety for all internal state, allowing
/// multiple threads to safely interact with the engine simultaneously.
pub struct ExecutionEngine {
    /// The ONNX model (immutable once created)
    model: Arc<OnnxModel>,
    /// Execution options (immutable configuration)
    options: Arc<ExecutionOptions>,
    /// Shared execution context with internal thread-safety
    context: Arc<ExecutionContext>,
    /// Operator registry (thread-safe due to immutability after initialization)
    operator_registry: Arc<OperatorRegistry>,
    /// Execution graph (protected with RwLock for concurrent access)
    execution_graph: Arc<RwLock<Option<ExecutionGraph>>>,
    /// Mapping from tensor names to tensor IDs (protected for thread safety)
    tensor_name_map: Arc<RwLock<HashMap<String, TensorId>>>,
    /// Mapping from node IDs to operators (protected for thread safety)
    node_operators: Arc<RwLock<HashMap<NodeId, Box<dyn Operator>>>>,
    /// Input tensor names (immutable after initialization)
    input_names: Arc<Vec<String>>,
    /// Output tensor names (immutable after initialization)
    output_names: Arc<Vec<String>>,
    /// Profiling events (protected for thread safety)
    profile_events: Arc<Mutex<Vec<ProfileEvent>>>,
    /// Current profile event ID counter (protected for thread safety)
    profile_event_counter: Arc<Mutex<usize>>,
    /// Flag indicating if the engine is prepared (protected for thread safety)
    is_prepared: Arc<RwLock<bool>>,
    /// Callback for memory allocation events
    memory_allocation_callback: Arc<Mutex<Option<Box<dyn Fn(usize, Option<TensorId>, Option<NodeId>, usize, &str, HashMap<String, String>) + Send + Sync>>>>,
    /// Callback for memory deallocation events
    memory_deallocation_callback: Arc<Mutex<Option<Box<dyn Fn(usize, Option<TensorId>, Option<NodeId>, usize, &str, HashMap<String, String>) + Send + Sync>>>>,
    /// Callback for memory reuse events
    memory_reuse_callback: Arc<Mutex<Option<Box<dyn Fn(usize, Option<TensorId>, Option<TensorId>, Option<NodeId>, usize, &str) + Send + Sync>>>>,
    /// Callback for workspace allocation events
    workspace_allocation_callback: Arc<Mutex<Option<Box<dyn Fn(usize, Option<NodeId>, String) + Send + Sync>>>>,
}

impl ExecutionEngine {
    /// Create a new thread-safe execution engine
    pub fn new(model: OnnxModel, options: ExecutionOptions) -> Result<Self> {
        let input_names = model.graph.inputs.iter()
            .map(|input| input.name.clone())
            .collect();
            
        let output_names = model.graph.outputs.iter()
            .map(|output| output.name.clone())
            .collect();
        
        // Initialize shared context with the specified options
        let options_arc = Arc::new(options);
        let context = Arc::new(ExecutionContext::new(options_arc.as_ref().clone()));
        
        Ok(Self {
            model: Arc::new(model),
            options: options_arc,
            context,
            operator_registry: Arc::new(OperatorRegistry::initialize_standard_operators()),
            execution_graph: Arc::new(RwLock::new(None)),
            tensor_name_map: Arc::new(RwLock::new(HashMap::new())),
            node_operators: Arc::new(RwLock::new(HashMap::new())),
            input_names: Arc::new(input_names),
            output_names: Arc::new(output_names),
            profile_events: Arc::new(Mutex::new(Vec::new())),
            profile_event_counter: Arc::new(Mutex::new(0)),
            is_prepared: Arc::new(RwLock::new(false)),
            // Initialize callback fields as None
            memory_allocation_callback: Arc::new(Mutex::new(None)),
            memory_deallocation_callback: Arc::new(Mutex::new(None)),
            memory_reuse_callback: Arc::new(Mutex::new(None)),
            workspace_allocation_callback: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Prepare the engine for execution in a thread-safe manner
    pub fn prepare(&self) -> Result<()> {
        // Try to acquire a read lock to check if already prepared
        let is_prepared = match self.is_prepared.read() {
            Ok(guard) => *guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for is_prepared".to_string())),
        };
        
        if is_prepared {
            return Ok(());
        }
        
        // Acquire a write lock for preparation
        let mut is_prepared_guard = match self.is_prepared.write() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire write lock for is_prepared".to_string())),
        };
        
        // Double-check in case another thread prepared while we were waiting
        if *is_prepared_guard {
            return Ok(());
        }
        
        // Build execution graph from the model
        let mut graph = build_execution_graph(&self.model)?;
        
        // Apply optimizations
        if self.options.optimization_level != crate::execution::context::OptimizationLevel::None {
            let optimizer = GraphOptimizer::new();
            optimizer.optimize(&mut graph, self.options.optimization_level)?;
        }
        
        // Assign tensor IDs
        self.assign_tensor_ids(&graph)?;
        
        // Create operators for each node
        self.create_node_operators(&graph)?;
        
        // Allocate tensors for intermediate outputs
        self.allocate_intermediate_tensors()?;
        
        // Store the execution graph
        {
            // Acquire a write lock for execution_graph
            let mut graph_guard = match self.execution_graph.write() {
                Ok(guard) => guard,
                Err(_) => return Err(Error::InvalidModel("Failed to acquire write lock for execution_graph".to_string())),
            };
            *graph_guard = Some(graph);
        }
        
        // Mark as prepared
        *is_prepared_guard = true;
        Ok(())
    }
    
    /// Run the model with the given inputs in a thread-safe manner
    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Ensure the model is prepared
        let is_prepared = match self.is_prepared.read() {
            Ok(guard) => *guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for is_prepared".to_string())),
        };
        
        if !is_prepared {
            self.prepare()?;
        }
        
        // Set input tensors (already thread-safe)
        for (name, tensor) in inputs {
            self.set_input_tensor(&name, tensor)?;
        }
        
        // Create execution order
        let execution_order = self.create_execution_order()?;
        
        // Start profiling if enabled
        let profile_id = if self.options.enable_profiling {
            Some(self.start_profile_event("model_execution", None))
        } else {
            None
        };
        
        // Execute nodes in order
        for &node_id in &execution_order {
            self.run_node(node_id)?;
        }
        
        // End profiling if enabled
        if let Some(id) = profile_id {
            self.end_profile_event(id);
        }
        
        // Collect outputs
        let mut outputs = HashMap::new();
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        for name in self.output_names.iter() {
            if let Some(tensor_id) = tensor_name_map.get(name) {
                if let Ok(Some(tensor)) = self.context.get_tensor(tensor_id) {
                    // Create a named copy of the tensor
                    let mut output_tensor = tensor.clone();
                    output_tensor.name = Some(name.clone());
                    outputs.insert(name.clone(), output_tensor);
                } else {
                    return Err(Error::InvalidGraph(format!(
                        "Output tensor '{}' not found in context", name
                    )));
                }
            } else {
                return Err(Error::InvalidGraph(format!(
                    "Output tensor '{}' not found in tensor map", name
                )));
            }
        }
        
        Ok(outputs)
    }
    
    /// Run a single node in a thread-safe manner
    pub fn run_node(&self, node_id: NodeId) -> Result<()> {
        // Get a read lock on the execution graph
        let graph_guard = match self.execution_graph.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for execution_graph".to_string())),
        };
        
        let graph = graph_guard.as_ref().ok_or_else(|| {
            Error::InvalidGraph("Execution graph not prepared".to_string())
        })?;
        
        // Find the node by ID
        let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Node with ID {} not found", node_id))
        })?;
        
        // Get a read lock on the node operators map
        let node_operators = match self.node_operators.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for node_operators".to_string())),
        };
        
        // Get the operator for this node
        let operator = node_operators.get(&node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Operator for node {} not found", node_id))
        })?;
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        // Collect input tensors
        let mut input_tensors = Vec::new();
        for input_name in &node.inputs {
            // Skip optional inputs (indicated by empty name)
            if input_name.is_empty() {
                continue;
            }
            
            let tensor_id = tensor_name_map.get(input_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor ID for input '{}' not found", input_name))
            })?;
            
            let tensor = self.context.get_tensor(tensor_id)?.ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor '{}' not found in context", input_name))
            })?;
            
            input_tensors.push(tensor);
        }
        
        // Prepare output tensors
        let mut output_tensors = Vec::new();
        for output_name in &node.outputs {
            let tensor_id = tensor_name_map.get(output_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor ID for output '{}' not found", output_name))
            })?;
            
            if let Ok(Some(tensor)) = self.context.get_tensor(tensor_id) {
                output_tensors.push(tensor.clone());
            } else {
                // Create a placeholder tensor - the operator will fill in the details
                let tensor = Tensor::new(&[], crate::ops::tensor::DataType::Float32);
                output_tensors.push(tensor);
            }
        }
        
        // Set up execution context for the operator
        let op_context = crate::ops::registry::ExecutionContext::default();
        
        // Start profiling if enabled
        let profile_id = if self.options.enable_profiling {
            Some(self.start_profile_event(&format!("op_{}", node.op_type), Some(node_id)))
        } else {
            None
        };
        
        // Execute the operator
        let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
        operator.compute(&input_refs, &mut output_tensors, &op_context)?;
        
        // End profiling if enabled
        if let Some(id) = profile_id {
            self.end_profile_event(id);
        }
        
        // Store output tensors in context
        for (i, output_name) in node.outputs.iter().enumerate() {
            if i < output_tensors.len() {
                let tensor_id = tensor_name_map.get(output_name).ok_or_else(|| {
                    Error::InvalidGraph(format!("Tensor ID for output '{}' not found", output_name))
                })?;
                
                // Set the name of the tensor
                let mut output_tensor = output_tensors[i].clone();
                output_tensor.name = Some(output_name.clone());
                
                self.context.set_tensor(*tensor_id, output_tensor)?;
            }
        }
        
        Ok(())
    }
    
    /// Set an input tensor in a thread-safe manner
    pub fn set_input_tensor(&self, name: &str, tensor: Tensor) -> Result<()> {
        if !self.input_names.iter().any(|n| n == name) {
            return Err(Error::InvalidGraph(format!(
                "Input tensor '{}' is not defined in the model", name
            )));
        }
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        let tensor_id = tensor_name_map.get(name).ok_or_else(|| {
            Error::InvalidGraph(format!("Tensor ID for input '{}' not found", name))
        })?;
        
        // Set the name of the tensor
        let mut input_tensor = tensor;
        input_tensor.name = Some(name.to_string());
        
        // Set the tensor in the context (already thread-safe)
        self.context.set_tensor(*tensor_id, input_tensor)?;
        Ok(())
    }
    
    /// Get an output tensor in a thread-safe manner
    pub fn get_output_tensor(&self, name: &str) -> Result<Tensor> {
        if !self.output_names.iter().any(|n| n == name) {
            return Err(Error::InvalidGraph(format!(
                "Output tensor '{}' is not defined in the model", name
            )));
        }
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        let tensor_id = tensor_name_map.get(name).ok_or_else(|| {
            Error::InvalidGraph(format!("Tensor ID for output '{}' not found", name))
        })?;
        
        // Get the tensor from the context (already thread-safe)
        let tensor = self.context.get_tensor(tensor_id)?.ok_or_else(|| {
            Error::InvalidGraph(format!("Output tensor '{}' not found in context", name))
        })?;
        
        // Return a clone to avoid lifetime issues with the lock
        Ok(tensor)
    }
    
    /// Allocate workspace memory in a thread-safe manner
    pub fn allocate_workspace(&self, size_bytes: usize) -> Result<WorkspaceGuard> {
        // The context's get_workspace method is already thread-safe
        let workspace = self.context.get_workspace(size_bytes)?;
        
        // Trigger workspace allocation callback if set
        if let Ok(callback) = self.workspace_allocation_callback.lock() {
            if let Some(cb) = callback.as_ref() {
                // Get the current node ID if available from profiling events
                let node_id = if self.options.enable_profiling {
                    let events = self.profile_events.lock().ok()?;
                    events.last().and_then(|e| e.node_id)
                } else {
                    None
                };
                
                // Get the current operation type
                let op_type = if let Some(node_id) = node_id {
                    if let Ok(operators) = self.node_operators.read() {
                        if let Some(graph) = self.execution_graph.read().ok().and_then(|g| g.clone()) {
                            graph.nodes.iter()
                                .find(|n| n.id == node_id)
                                .map(|n| n.op_type.clone())
                                .unwrap_or_else(|| "unknown".to_string())
                        } else {
                            "unknown".to_string()
                        }
                    } else {
                        "unknown".to_string()
                    }
                } else {
                    "unknown".to_string()
                };
                
                // Call the callback
                cb(size_bytes, node_id, op_type);
            }
        }
        
        Ok(workspace)
    }
    
    /// Set callback for memory allocation events
    pub fn set_memory_allocation_callback(
        &self, 
        callback: Box<dyn Fn(usize, Option<TensorId>, Option<NodeId>, usize, &str, HashMap<String, String>) + Send + Sync>
    ) {
        if let Ok(mut cb) = self.memory_allocation_callback.lock() {
            *cb = Some(callback);
        }
    }
    
    /// Set callback for memory deallocation events
    pub fn set_memory_deallocation_callback(
        &self, 
        callback: Box<dyn Fn(usize, Option<TensorId>, Option<NodeId>, usize, &str, HashMap<String, String>) + Send + Sync>
    ) {
        if let Ok(mut cb) = self.memory_deallocation_callback.lock() {
            *cb = Some(callback);
        }
    }
    
    /// Set callback for memory reuse events
    pub fn set_memory_reuse_callback(
        &self, 
        callback: Box<dyn Fn(usize, Option<TensorId>, Option<TensorId>, Option<NodeId>, usize, &str) + Send + Sync>
    ) -> Option<Box<dyn Fn(usize, Option<TensorId>, Option<TensorId>, Option<NodeId>, usize, &str) + Send + Sync>> {
        if let Ok(mut cb) = self.memory_reuse_callback.lock() {
            let old_callback = cb.take();
            *cb = Some(callback);
            old_callback
        } else {
            None
        }
    }
    
    /// Set callback for workspace allocation events
    pub fn set_workspace_allocation_callback(
        &self, 
        callback: Box<dyn Fn(usize, Option<NodeId>, String) + Send + Sync>
    ) {
        if let Ok(mut cb) = self.workspace_allocation_callback.lock() {
            *cb = Some(callback);
        }
    }
    
    /// Allocate tensors for intermediate outputs in a thread-safe manner
    pub fn allocate_intermediate_tensors(&self) -> Result<()> {
        // This is a simplified implementation
        // In a real system, you would analyze the graph and allocate memory
        // efficiently, reusing buffers when possible
        
        // The context's tensor allocation methods are already thread-safe
        Ok(())
    }
    
    /// Create execution order for the graph in a thread-safe manner
    pub fn create_execution_order(&self) -> Result<Vec<NodeId>> {
        // Get a read lock on the execution graph
        let graph_guard = match self.execution_graph.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for execution_graph".to_string())),
        };
        
        let graph = graph_guard.as_ref().ok_or_else(|| {
            Error::InvalidGraph("Execution graph not prepared".to_string())
        })?;
        
        // Topological sort (thread-safe since we have a read lock on the graph)
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        
        for &node_id in &graph.input_nodes {
            self.topological_sort(node_id, graph, &mut visited, &mut visiting, &mut result)?;
        }
        
        Ok(result)
    }
    
    /// Helper for topological sort
    fn topological_sort(
        &self,
        node_id: NodeId,
        graph: &ExecutionGraph,
        visited: &mut HashSet<NodeId>,
        visiting: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        
        if visiting.contains(&node_id) {
            return Err(Error::InvalidGraph(format!(
                "Cycle detected in graph at node {}", node_id
            )));
        }
        
        visiting.insert(node_id);
        
        // Visit dependencies
        if let Some(deps) = graph.dependencies.get(&node_id) {
            for &dep_id in deps {
                self.topological_sort(dep_id, graph, visited, visiting, result)?;
            }
        }
        
        visiting.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);
        
        Ok(())
    }
    
    /// Profile execution time for a model run in a thread-safe manner
    pub fn profile_execution_time(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Duration>> {
        // We can't modify the options directly since they're now immutable
        // Instead, we'll create a temporary copy of the engine with profiling enabled
        
        // Create a new engine with profiling enabled
        let mut temp_options = (*self.options).clone();
        temp_options.enable_profiling = true;
        
        // Clear existing profiling data
        {
            let mut profile_events = match self.profile_events.lock() {
                Ok(guard) => guard,
                Err(_) => return Err(Error::InvalidModel("Failed to acquire lock for profile_events".to_string())),
            };
            profile_events.clear();
            
            // Reset the counter
            let mut counter = match self.profile_event_counter.lock() {
                Ok(guard) => guard,
                Err(_) => return Err(Error::InvalidModel("Failed to acquire lock for profile_event_counter".to_string())),
            };
            *counter = 0;
        }
        
        // Run the model
        self.run(inputs)?;
        
        // Collect profiling results
        let mut results = HashMap::new();
        
        // Get profiling data
        let profile_events = match self.profile_events.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire lock for profile_events".to_string())),
        };
        
        for event in profile_events.iter() {
            if let Some(duration) = event.duration {
                results.insert(event.name.clone(), duration);
            }
        }
        
        Ok(results)
    }
    
    /// Start a profiling event in a thread-safe manner
    fn start_profile_event(&self, name: &str, node_id: Option<NodeId>) -> ProfileEventId {
        // Get a lock on the profile event counter
        let mut counter_guard = match self.profile_event_counter.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If we fail to acquire the lock, create a fallback ID
                // This is not ideal but allows execution to continue
                return ProfileEventId(std::usize::MAX);
            }
        };
        
        let id = ProfileEventId(*counter_guard);
        *counter_guard += 1;
        
        // Create the event
        let event = ProfileEvent {
            id,
            name: name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
            node_id,
        };
        
        // Get a lock on the profile events
        let mut events_guard = match self.profile_events.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If we fail to acquire the lock, return the ID
                // The event won't be recorded, but execution can continue
                return id;
            }
        };
        
        events_guard.push(event);
        id
    }
    
    /// End a profiling event in a thread-safe manner
    fn end_profile_event(&self, id: ProfileEventId) {
        // Get a lock on the profile events
        let mut events_guard = match self.profile_events.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If we fail to acquire the lock, just return
                return;
            }
        };
        
        if let Some(event) = events_guard.iter_mut().find(|e| e.id == id) {
            let end_time = Instant::now();
            event.end_time = Some(end_time);
            event.duration = Some(end_time.duration_since(event.start_time));
        }
    }
    
    /// Assign tensor IDs to all tensors in the graph in a thread-safe manner
    fn assign_tensor_ids(&self, graph: &ExecutionGraph) -> Result<()> {
        let mut next_id = 0;
        let mut tensor_name_map = HashMap::new();
        
        // Assign IDs to input tensors
        for node in &graph.nodes {
            for input_name in &node.inputs {
                // Skip empty inputs (optional)
                if input_name.is_empty() {
                    continue;
                }
                
                if !tensor_name_map.contains_key(input_name) {
                    tensor_name_map.insert(input_name.clone(), next_id);
                    next_id += 1;
                }
            }
            
            // Assign IDs to output tensors
            for output_name in &node.outputs {
                if !tensor_name_map.contains_key(output_name) {
                    tensor_name_map.insert(output_name.clone(), next_id);
                    next_id += 1;
                }
            }
        }
        
        // Update the tensor name map with a write lock
        let mut tensor_map_guard = match self.tensor_name_map.write() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire write lock for tensor_name_map".to_string())),
        };
        
        *tensor_map_guard = tensor_name_map;
        Ok(())
    }
    
    /// Create operators for each node in the graph in a thread-safe manner
    fn create_node_operators(&self, graph: &ExecutionGraph) -> Result<()> {
        let mut node_operators = HashMap::new();
        
        for node in &graph.nodes {
            let operator = self.operator_registry.create_operator_for_node(node)?;
            node_operators.insert(node.id, operator);
        }
        
        // Update the node operators map with a write lock
        let mut operators_guard = match self.node_operators.write() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire write lock for node_operators".to_string())),
        };
        
        *operators_guard = node_operators;
        Ok(())
    }
    
    /// Get the model in a thread-safe manner
    pub fn model(&self) -> Arc<OnnxModel> {
        self.model.clone()
    }
    
    /// Get the execution graph in a thread-safe manner
    pub fn execution_graph(&self) -> Result<Option<ExecutionGraph>> {
        match self.execution_graph.read() {
            Ok(guard) => Ok(guard.clone()),
            Err(_) => Err(Error::InvalidModel("Failed to acquire read lock for execution_graph".to_string())),
        }
    }
    
    /// Get the input tensor names in a thread-safe manner
    pub fn input_names(&self) -> Arc<Vec<String>> {
        self.input_names.clone()
    }
    
    /// Get the output tensor names in a thread-safe manner
    pub fn output_names(&self) -> Arc<Vec<String>> {
        self.output_names.clone()
    }
    
    /// Get profile events in a thread-safe manner
    pub fn profile_events(&self) -> Result<Vec<ProfileEvent>> {
        match self.profile_events.lock() {
            Ok(guard) => Ok(guard.clone()),
            Err(_) => Err(Error::InvalidModel("Failed to acquire lock for profile_events".to_string())),
        }
    }
    
    /// Run the model with concurrent node execution
    /// 
    /// This method executes independent nodes in parallel using the thread pool
    /// from the ExecutionContext, providing better performance on multi-core systems.
    pub fn run_concurrent(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Ensure the model is prepared
        let is_prepared = match self.is_prepared.read() {
            Ok(guard) => *guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for is_prepared".to_string())),
        };
        
        if !is_prepared {
            self.prepare()?;
        }
        
        // Set input tensors
        for (name, tensor) in inputs {
            self.set_input_tensor(&name, tensor)?;
        }
        
        // Create execution order and dependency graph
        let execution_order = self.create_execution_order()?;
        
        // Start profiling if enabled
        let profile_id = if self.options.enable_profiling {
            Some(self.start_profile_event("model_execution_concurrent", None))
        } else {
            None
        };
        
        // Get the execution graph for dependency analysis
        let graph_guard = match self.execution_graph.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for execution_graph".to_string())),
        };
        
        let graph = graph_guard.as_ref().ok_or_else(|| {
            Error::InvalidGraph("Execution graph not prepared".to_string())
        })?;
        
        // Build dependency tracking
        let mut remaining_deps = HashMap::new();
        let mut completed_nodes = HashSet::new();
        let mut ready_nodes = Vec::new();
        
        // Initialize dependency counts
        for &node_id in &execution_order {
            let deps = graph.dependencies.get(&node_id).map_or(Vec::new(), |d| d.clone());
            remaining_deps.insert(node_id, deps.len());
            
            // Add nodes with no dependencies to the ready queue
            if deps.is_empty() {
                ready_nodes.push(node_id);
            }
        }
        
        // Execute until all nodes are processed
        while !ready_nodes.is_empty() {
            let current_ready = std::mem::take(&mut ready_nodes);
            
            // Execute ready nodes in parallel if thread pool is available
            if let Some(thread_pool) = self.context.thread_pool() {
                // Use thread pool for parallel execution
                thread_pool.scope(|s| {
                    for &node_id in &current_ready {
                        s.spawn(move |_| {
                            // Ignore errors in individual nodes - they will be checked later
                            let _ = self.run_node(node_id);
                        });
                    }
                });
            } else {
                // Sequential fallback
                for &node_id in &current_ready {
                    self.run_node(node_id)?;
                }
            }
            
            // Mark these nodes as completed
            for &node_id in &current_ready {
                completed_nodes.insert(node_id);
                
                // Update dependencies and find newly ready nodes
                for &next_node in &execution_order {
                    if completed_nodes.contains(&next_node) {
                        continue;
                    }
                    
                    let deps = match graph.dependencies.get(&next_node) {
                        Some(deps) => deps,
                        None => continue,
                    };
                    
                    if deps.contains(&node_id) {
                        let remaining = remaining_deps.entry(next_node).or_insert(deps.len());
                        *remaining -= 1;
                        
                        if *remaining == 0 {
                            ready_nodes.push(next_node);
                        }
                    }
                }
            }
        }
        
        // End profiling if enabled
        if let Some(id) = profile_id {
            self.end_profile_event(id);
        }
        
        // Check if all nodes were executed
        if completed_nodes.len() != execution_order.len() {
            return Err(Error::InvalidGraph(
                "Not all nodes were executed. This may indicate a cycle in the graph.".to_string()
            ));
        }
        
        // Collect outputs
        let mut outputs = HashMap::new();
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        for name in self.output_names.iter() {
            if let Some(tensor_id) = tensor_name_map.get(name) {
                if let Ok(Some(tensor)) = self.context.get_tensor(tensor_id) {
                    // Create a named copy of the tensor
                    let mut output_tensor = tensor.clone();
                    output_tensor.name = Some(name.clone());
                    outputs.insert(name.clone(), output_tensor);
                } else {
                    return Err(Error::InvalidGraph(format!(
                        "Output tensor '{}' not found in context", name
                    )));
                }
            } else {
                return Err(Error::InvalidGraph(format!(
                    "Output tensor '{}' not found in tensor map", name
                )));
            }
        }
        
        Ok(outputs)
    }
}

/// Build an execution graph from an ONNX model
fn build_execution_graph(model: &OnnxModel) -> Result<ExecutionGraph> {
    let mut nodes = model.graph.nodes.clone();
    let mut dependencies = HashMap::new();
    let mut input_nodes = Vec::new();
    let mut output_nodes = Vec::new();
    
    // Map from tensor name to producing node ID
    let mut tensor_producers = HashMap::new();
    
    // Record which node produces each tensor
    for node in &nodes {
        for output in &node.outputs {
            tensor_producers.insert(output.clone(), node.id);
        }
    }
    
    // Find dependencies between nodes
    for node in &nodes {
        let mut node_deps = Vec::new();
        
        for input in &node.inputs {
            // Skip empty inputs (optional)
            if input.is_empty() {
                continue;
            }
            
            if let Some(&producer_id) = tensor_producers.get(input) {
                node_deps.push(producer_id);
            } else {
                // This is an input to the graph
                let is_initializer = model.graph.initializers.iter()
                    .any(|init| &init.name == input);
                
                if !is_initializer {
                    // This node takes a graph input
                    input_nodes.push(node.id);
                }
            }
        }
        
        dependencies.insert(node.id, node_deps);
    }
    
    // Find output nodes
    for node in &nodes {
        for output in &node.outputs {
            if model.graph.outputs.iter().any(|o| &o.name == output) {
                output_nodes.push(node.id);
                break;
            }
        }
    }
    
    Ok(ExecutionGraph {
        nodes,
        input_nodes,
        output_nodes,
        dependencies,
    })
}

/// Execution priorities for nodes in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecutionPriority {
    /// Critical path operations (highest priority)
    Critical = 0,
    /// High priority operations
    High = 1,
    /// Normal priority operations
    Normal = 2,
    /// Low priority operations
    Low = 3,
}

impl Default for ExecutionPriority {
    fn default() -> Self {
        ExecutionPriority::Normal
    }
}

/// Information about an operation to be executed
#[derive(Debug)]
struct OperationTask {
    /// Node ID in the execution graph
    node_id: NodeId,
    /// Priority of this operation
    priority: ExecutionPriority,
    /// Estimated cost (in arbitrary units, higher means more expensive)
    estimated_cost: u32,
    /// Dependencies that must complete before this operation
    dependencies: Vec<NodeId>,
    /// Number of remaining dependencies to be satisfied
    remaining_deps: AtomicUsize,
    /// Nodes that depend on this operation
    dependents: Vec<NodeId>,
    /// Whether execution has started
    started: AtomicBool,
    /// Whether execution has completed
    completed: AtomicBool,
    /// Whether execution failed
    failed: AtomicBool,
    /// Start time of execution
    start_time: Mutex<Option<Instant>>,
    /// End time of execution
    end_time: Mutex<Option<Instant>>,
    /// Error that occurred during execution, if any
    error: Mutex<Option<Error>>,
}

impl OperationTask {
    fn new(node_id: NodeId, priority: ExecutionPriority, cost: u32) -> Self {
        Self {
            node_id,
            priority,
            estimated_cost: cost,
            dependencies: Vec::new(),
            remaining_deps: AtomicUsize::new(0),
            dependents: Vec::new(),
            started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            start_time: Mutex::new(None),
            end_time: Mutex::new(None),
            error: Mutex::new(None),
        }
    }
    
    fn mark_started(&self) {
        self.started.store(true, Ordering::Release);
        let mut start_time = self.start_time.lock().unwrap();
        *start_time = Some(Instant::now());
    }
    
    fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
        let mut end_time = self.end_time.lock().unwrap();
        *end_time = Some(Instant::now());
    }
    
    fn mark_failed(&self, error: Error) {
        self.failed.store(true, Ordering::Release);
        let mut end_time = self.end_time.lock().unwrap();
        *end_time = Some(Instant::now());
        let mut err = self.error.lock().unwrap();
        *err = Some(error);
    }
    
    fn is_ready(&self) -> bool {
        self.remaining_deps.load(Ordering::Acquire) == 0 &&
        !self.started.load(Ordering::Acquire) &&
        !self.completed.load(Ordering::Acquire) &&
        !self.failed.load(Ordering::Acquire)
    }
}

/// A task in the execution queue
#[derive(Debug)]
struct QueuedTask {
    /// Task reference
    task: Arc<OperationTask>,
    /// Scheduling priority (combines operation priority and other factors)
    scheduling_priority: i32,
}

impl PartialEq for QueuedTask {
    fn eq(&self, other: &Self) -> bool {
        self.scheduling_priority == other.scheduling_priority
    }
}

impl Eq for QueuedTask {}

impl PartialOrd for QueuedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse ordering for min-heap based on priority
        other.scheduling_priority.partial_cmp(&self.scheduling_priority)
    }
}

impl Ord for QueuedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap based on priority
        other.scheduling_priority.cmp(&self.scheduling_priority)
    }
}

/// Manages the concurrent execution of operations in the graph
#[derive(Debug)]
struct TaskScheduler {
    /// Tasks by node ID
    tasks: HashMap<NodeId, Arc<OperationTask>>,
    /// Ready queue for tasks that can be executed
    ready_queue: Mutex<BinaryHeap<QueuedTask>>,
    /// Lock for modifying the task graph
    graph_lock: Mutex<()>,
    /// Condition variable for waiting on task completion
    completion_cv: Condvar,
    /// Count of tasks in progress
    tasks_in_progress: AtomicUsize,
    /// Count of completed tasks
    completed_tasks: AtomicUsize,
    /// Count of failed tasks
    failed_tasks: AtomicUsize,
    /// Whether execution should be cancelled
    should_cancel: AtomicBool,
    /// Whether execution is completed
    is_completed: AtomicBool,
    /// Execution options
    options: Arc<ExecutionOptions>,
}

impl TaskScheduler {
    fn new(options: Arc<ExecutionOptions>) -> Self {
        Self {
            tasks: HashMap::new(),
            ready_queue: Mutex::new(BinaryHeap::new()),
            graph_lock: Mutex::new(()),
            completion_cv: Condvar::new(),
            tasks_in_progress: AtomicUsize::new(0),
            completed_tasks: AtomicUsize::new(0),
            failed_tasks: AtomicUsize::new(0),
            should_cancel: AtomicBool::new(false),
            is_completed: AtomicBool::new(false),
            options,
        }
    }
    
    /// Build the task graph from the execution graph
    fn build_from_graph(&mut self, graph: &ExecutionGraph) -> Result<()> {
        // Lock to modify the graph
        let _guard = self.graph_lock.lock().unwrap();
        
        // Clear any existing tasks
        self.tasks.clear();
        
        // Create tasks for each node
        for node in &graph.nodes {
            // Basic cost estimate (can be refined based on op type)
            let cost = 10;
            
            // Default to normal priority
            let priority = ExecutionPriority::Normal;
            
            // Create the task
            let task = Arc::new(OperationTask::new(node.id, priority, cost));
            self.tasks.insert(node.id, task);
        }
        
        // Set up dependencies
        for (&node_id, deps) in &graph.dependencies {
            if let Some(task) = self.tasks.get(&node_id) {
                for &dep_id in deps {
                    if let Some(dep_task) = self.tasks.get(&dep_id) {
                        // Add this dependency
                        task.dependencies.push(dep_id);
                        // Update the remaining deps counter
                        task.remaining_deps.fetch_add(1, Ordering::AcqRel);
                        // Add this task as a dependent of the dependency
                        dep_task.dependents.push(node_id);
                    }
                }
            }
        }
        
        // Identify critical paths
        self.identify_critical_paths(graph);
        
        // Add all ready tasks to the queue
        let mut ready_queue = self.ready_queue.lock().unwrap();
        for task in self.tasks.values() {
            if task.is_ready() {
                // Calculate scheduling priority
                let scheduling_priority = self.calculate_priority(task);
                
                ready_queue.push(QueuedTask {
                    task: task.clone(),
                    scheduling_priority,
                });
            }
        }
        
        Ok(())
    }
    
    /// Identify critical paths in the graph to prioritize operations
    fn identify_critical_paths(&mut self, graph: &ExecutionGraph) {
        // Calculate the earliest time each node can execute
        let mut earliest_times: HashMap<NodeId, u32> = HashMap::new();
        
        // Start with input nodes
        for &node_id in &graph.input_nodes {
            earliest_times.insert(node_id, 0);
        }
        
        // Calculate earliest times by traversing forward
        let mut nodes_in_progress = graph.input_nodes.clone();
        while !nodes_in_progress.is_empty() {
            let node_id = nodes_in_progress.remove(0);
            let task = match self.tasks.get(&node_id) {
                Some(t) => t,
                None => continue,
            };
            
            let current_time = *earliest_times.get(&node_id).unwrap_or(&0);
            let next_time = current_time + task.estimated_cost;
            
            // Update dependent nodes
            for &dep_id in &task.dependents {
                let existing_time = earliest_times.get(&dep_id).unwrap_or(&0);
                if next_time > *existing_time {
                    earliest_times.insert(dep_id, next_time);
                }
                
                nodes_in_progress.push(dep_id);
            }
        }
        
        // Calculate the latest time each node can execute without delaying the output
        let mut latest_times: HashMap<NodeId, u32> = HashMap::new();
        
        // Find the maximum earliest time to set as the deadline
        let deadline = earliest_times.values().max().unwrap_or(&0).clone();
        
        // Start with output nodes
        for &node_id in &graph.output_nodes {
            latest_times.insert(node_id, deadline);
        }
        
        // Calculate latest times by traversing backward
        let mut nodes_in_progress = graph.output_nodes.clone();
        while !nodes_in_progress.is_empty() {
            let node_id = nodes_in_progress.remove(0);
            let task = match self.tasks.get(&node_id) {
                Some(t) => t,
                None => continue,
            };
            
            let current_time = *latest_times.get(&node_id).unwrap_or(&deadline);
            let prev_time = current_time.saturating_sub(task.estimated_cost);
            
            // Update dependent nodes
            for &dep_id in &task.dependencies {
                let existing_time = latest_times.get(&dep_id).unwrap_or(&deadline);
                if prev_time < *existing_time {
                    latest_times.insert(dep_id, prev_time);
                }
                
                nodes_in_progress.push(dep_id);
            }
        }
        
        // Calculate slack for each node
        for (node_id, task) in &mut self.tasks {
            let earliest = earliest_times.get(node_id).unwrap_or(&0);
            let latest = latest_times.get(node_id).unwrap_or(&deadline);
            
            // Slack is the difference between latest and earliest start times
            let slack = latest.saturating_sub(*earliest);
            
            // Nodes on the critical path have zero slack
            if slack == 0 {
                // This is an immutable borrow, so we're updating the priority in a temporary clone
                let priority = ExecutionPriority::Critical;
                
                // Create a new task with the updated priority
                let mut deps = Vec::new();
                let mut deps_count = 0;
                
                {
                    let task_ref = task;
                    deps = task_ref.dependencies.clone();
                    deps_count = task_ref.remaining_deps.load(Ordering::Acquire);
                }
                
                let new_task = Arc::new(OperationTask {
                    node_id: *node_id,
                    priority,
                    estimated_cost: task.estimated_cost,
                    dependencies: deps,
                    remaining_deps: AtomicUsize::new(deps_count),
                    dependents: task.dependents.clone(),
                    started: AtomicBool::new(false),
                    completed: AtomicBool::new(false),
                    failed: AtomicBool::new(false),
                    start_time: Mutex::new(None),
                    end_time: Mutex::new(None),
                    error: Mutex::new(None),
                });
                
                // Replace the task in the map
                self.tasks.insert(*node_id, new_task);
            }
        }
    }
    
    /// Calculate scheduling priority for a task
    fn calculate_priority(&self, task: &Arc<OperationTask>) -> i32 {
        // Base priority from operation priority
        let mut priority = match task.priority {
            ExecutionPriority::Critical => 0,
            ExecutionPriority::High => 100,
            ExecutionPriority::Normal => 200,
            ExecutionPriority::Low => 300,
        };
        
        // Add priority based on number of dependents
        // Operations with more dependents should run first
        priority -= task.dependents.len() as i32;
        
        // Add priority based on estimated cost
        // More expensive operations should start earlier
        priority -= (task.estimated_cost as i32) / 10;
        
        priority
    }
    
    /// Get the next task to execute
    fn next_task(&self) -> Option<Arc<OperationTask>> {
        let mut queue = self.ready_queue.lock().unwrap();
        if let Some(task) = queue.pop() {
            Some(task.task)
        } else {
            None
        }
    }
    
    /// Mark a task as started
    fn mark_task_started(&self, node_id: NodeId) {
        if let Some(task) = self.tasks.get(&node_id) {
            task.mark_started();
            self.tasks_in_progress.fetch_add(1, Ordering::Release);
        }
    }
    
    /// Mark a task as completed
    fn mark_task_completed(&self, node_id: NodeId) {
        if let Some(task) = self.tasks.get(&node_id) {
            task.mark_completed();
            self.tasks_in_progress.fetch_sub(1, Ordering::Release);
            self.completed_tasks.fetch_add(1, Ordering::Release);
            
            // Mark dependents as ready if this was their last dependency
            for &dep_id in &task.dependents {
                if let Some(dep_task) = self.tasks.get(&dep_id) {
                    let remaining = dep_task.remaining_deps.fetch_sub(1, Ordering::AcqRel);
                    
                    // If this was the last dependency, add to ready queue
                    if remaining == 1 && !dep_task.started.load(Ordering::Acquire) {
                        let scheduling_priority = self.calculate_priority(dep_task);
                        let mut queue = self.ready_queue.lock().unwrap();
                        
                        queue.push(QueuedTask {
                            task: dep_task.clone(),
                            scheduling_priority,
                        });
                    }
                }
            }
            
            // Notify waiters
            self.completion_cv.notify_all();
        }
    }
    
    /// Mark a task as failed
    fn mark_task_failed(&self, node_id: NodeId, error: Error) {
        if let Some(task) = self.tasks.get(&node_id) {
            task.mark_failed(error);
            self.tasks_in_progress.fetch_sub(1, Ordering::Release);
            self.failed_tasks.fetch_add(1, Ordering::Release);
            
            // Set cancellation flag if configured to cancel on error
            if self.options.cancel_on_error {
                self.should_cancel.store(true, Ordering::Release);
            }
            
            // Notify waiters
            self.completion_cv.notify_all();
        }
    }
    
    /// Wait for all tasks to complete
    fn wait_for_completion(&self) -> Result<()> {
        let mut guard = self.graph_lock.lock().unwrap();
        
        while self.tasks_in_progress.load(Ordering::Acquire) > 0 
               && !self.should_cancel.load(Ordering::Acquire) {
            // Wait on the condition variable
            guard = self.completion_cv.wait(guard).unwrap();
        }
        
        // Check if we should cancel
        if self.should_cancel.load(Ordering::Acquire) {
            return Err(Error::ExecutionCancelled("Execution was cancelled".to_string()));
        }
        
        // Check if any tasks failed
        if self.failed_tasks.load(Ordering::Acquire) > 0 {
            // Find the first error
            for task in self.tasks.values() {
                if task.failed.load(Ordering::Acquire) {
                    let err = task.error.lock().unwrap();
                    if let Some(error) = &*err {
                        return Err(error.clone());
                    }
                }
            }
            
            // If we don't find a specific error, return a generic one
            return Err(Error::ExecutionError("One or more operations failed".to_string()));
        }
        
        Ok(())
    }
    
    /// Cancel execution
    fn cancel(&self) {
        self.should_cancel.store(true, Ordering::Release);
        self.completion_cv.notify_all();
    }
    
    /// Get the overall execution progress as a percentage
    fn get_progress(&self) -> f32 {
        let total = self.tasks.len();
        let completed = self.completed_tasks.load(Ordering::Acquire);
        
        if total == 0 {
            return 1.0;
        }
        
        completed as f32 / total as f32
    }
    
    /// Check if execution should be cancelled
    fn should_cancel(&self) -> bool {
        self.should_cancel.load(Ordering::Acquire)
    }
    
    /// Reset the scheduler
    fn reset(&self) {
        let _guard = self.graph_lock.lock().unwrap();
        
        // Reset atomic counters
        self.tasks_in_progress.store(0, Ordering::Release);
        self.completed_tasks.store(0, Ordering::Release);
        self.failed_tasks.store(0, Ordering::Release);
        self.should_cancel.store(false, Ordering::Release);
        self.is_completed.store(false, Ordering::Release);
        
        // Clear the ready queue
        let mut queue = self.ready_queue.lock().unwrap();
        while queue.pop().is_some() {}
        
        // Reset all tasks
        for task in self.tasks.values() {
            task.started.store(false, Ordering::Release);
            task.completed.store(false, Ordering::Release);
            task.failed.store(false, Ordering::Release);
            
            // Reset times
            {
                let mut start_time = task.start_time.lock().unwrap();
                *start_time = None;
            }
            {
                let mut end_time = task.end_time.lock().unwrap();
                *end_time = None;
            }
            {
                let mut err = task.error.lock().unwrap();
                *err = None;
            }
        }
    }
}

/// A thread-safe cache for tensors used during execution
pub struct ThreadSafeTensorCache {
    /// Mapping from tensor IDs to tensors, protected with a RwLock
    tensors: RwLock<HashMap<TensorId, Tensor>>,
    /// Lock acquisition timeout in milliseconds (0 = no timeout)
    lock_timeout_ms: u64,
}

impl ThreadSafeTensorCache {
    /// Create a new thread-safe tensor cache
    pub fn new() -> Self {
        Self {
            tensors: RwLock::new(HashMap::new()),
            lock_timeout_ms: 1000, // Default 1 second timeout
        }
    }
    
    /// Create a new thread-safe tensor cache with a custom lock timeout
    pub fn with_timeout(lock_timeout_ms: u64) -> Self {
        Self {
            tensors: RwLock::new(HashMap::new()),
            lock_timeout_ms,
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
    
    /// Clear all tensors (thread-safe write access)
    pub fn clear(&self) -> Result<()> {
        if let Ok(mut tensors) = self.tensors.write() {
            tensors.clear();
            Ok(())
        } else {
            Err(Error::InvalidModel("Failed to acquire write lock for tensors".to_string()))
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
    
    /// Get the number of tensors in the cache (thread-safe read access)
    pub fn len(&self) -> Result<usize> {
        if let Ok(tensors) = self.tensors.read() {
            Ok(tensors.len())
        } else {
            Err(Error::InvalidModel("Failed to acquire read lock for tensors".to_string()))
        }
    }
    
    /// Check if the cache is empty (thread-safe read access)
    pub fn is_empty(&self) -> Result<bool> {
        if let Ok(tensors) = self.tensors.read() {
            Ok(tensors.is_empty())
        } else {
            Err(Error::InvalidModel("Failed to acquire read lock for tensors".to_string()))
        }
    }
}

impl ExecutionEngine {
    /// Run the model with advanced concurrent execution
    /// 
    /// This method provides sophisticated parallel execution using the TaskScheduler:
    /// - Builds a detailed DAG of operations with dependencies
    /// - Identifies critical paths to prioritize important operations
    /// - Uses dynamic priority-based scheduling
    /// - Provides error handling and cancellation capabilities
    /// - Supports progress tracking
    pub fn run_advanced_concurrent(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Ensure the model is prepared
        let is_prepared = match self.is_prepared.read() {
            Ok(guard) => *guard,
            Err(_) => return Err(Error::LockAcquisitionError("Failed to acquire read lock for is_prepared".to_string())),
        };
        
        if !is_prepared {
            self.prepare()?;
        }
        
        // Set input tensors
        for (name, tensor) in inputs {
            self.set_input_tensor(&name, tensor)?;
        }
        
        // Get the execution graph
        let graph_guard = match self.execution_graph.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::LockAcquisitionError("Failed to acquire read lock for execution_graph".to_string())),
        };
        
        let graph = match graph_guard.as_ref() {
            Some(g) => g,
            None => return Err(Error::InvalidGraph("Execution graph not prepared".to_string())),
        };
        
        // Create task scheduler
        let mut scheduler = TaskScheduler::new(self.options.clone());
        
        // Build task graph
        scheduler.build_from_graph(graph)?;
        
        // Start profiling if enabled
        let profile_id = if self.options.enable_profiling {
            Some(self.start_profile_event("model_execution_advanced", None))
        } else {
            None
        };
        
        // Execute the graph using the thread pool
        let pool = match self.context.thread_pool() {
            Some(p) => p,
            None => return Err(Error::ConcurrencyError("Thread pool is required for advanced concurrent execution".to_string())),
        };
        
        // Get the maximum number of concurrent operations
        let max_concurrent = if self.options.max_concurrent_operations > 0 {
            self.options.max_concurrent_operations
        } else {
            // Default to number of threads + 2 to allow for some over-subscription
            pool.current_num_threads() + 2
        };
        
        // Create a thread-safe shared cancellation flag
        let should_cancel = Arc::new(AtomicBool::new(false));
        
        // Create thread-safe counters
        let tasks_running = Arc::new(AtomicUsize::new(0));
        let total_tasks = scheduler.tasks.len();
        
        // Execute tasks
        pool.scope(|s| {
            // Launch worker threads up to max_concurrent
            let worker_count = std::cmp::min(max_concurrent, total_tasks);
            
            for _ in 0..worker_count {
                let scheduler_ref = &scheduler;
                let engine_ref = self;
                let should_cancel_clone = should_cancel.clone();
                let tasks_running_clone = tasks_running.clone();
                
                s.spawn(move |_| {
                    // Worker loop - keep getting tasks until none left or cancellation
                    while !should_cancel_clone.load(Ordering::Acquire) {
                        // Try to get the next task
                        let task = match scheduler_ref.next_task() {
                            Some(t) => t,
                            None => break, // No more tasks
                        };
                        
                        // Skip if task is already completed or failed
                        if task.completed.load(Ordering::Acquire) || task.failed.load(Ordering::Acquire) {
                            continue;
                        }
                        
                        // Mark task as started
                        scheduler_ref.mark_task_started(task.node_id);
                        tasks_running_clone.fetch_add(1, Ordering::Release);
                        
                        // Execute the task
                        let result = engine_ref.run_node(task.node_id);
                        
                        // Update task status
                        tasks_running_clone.fetch_sub(1, Ordering::Release);
                        
                        match result {
                            Ok(_) => {
                                scheduler_ref.mark_task_completed(task.node_id);
                            }
                            Err(e) => {
                                scheduler_ref.mark_task_failed(task.node_id, e);
                                
                                // Check if we should cancel execution
                                if engine_ref.options.cancel_on_error {
                                    should_cancel_clone.store(true, Ordering::Release);
                                    scheduler_ref.cancel();
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });
        
        // Wait for all tasks to complete
        scheduler.wait_for_completion()?;
        
        // End profiling if enabled
        if let Some(id) = profile_id {
            self.end_profile_event(id);
        }
        
        // Collect outputs
        let mut outputs = HashMap::new();
        
        // Get a read lock on the tensor name map
        let tensor_name_map = match self.tensor_name_map.read() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::LockAcquisitionError("Failed to acquire read lock for tensor_name_map".to_string())),
        };
        
        for name in self.output_names.iter() {
            if let Some(tensor_id) = tensor_name_map.get(name) {
                if let Ok(Some(tensor)) = self.context.get_tensor(tensor_id) {
                    // Create a named copy of the tensor
                    let mut output_tensor = tensor.clone();
                    output_tensor.name = Some(name.clone());
                    outputs.insert(name.clone(), output_tensor);
                } else {
                    return Err(Error::ExecutionError(format!(
                        "Output tensor '{}' not found in context", name
                    )));
                }
            } else {
                return Err(Error::ExecutionError(format!(
                    "Output tensor '{}' not found in tensor map", name
                )));
            }
        }
        
        Ok(outputs)
    }

    /// Create a new instance that shares the same model and execution graph
    /// but has its own independent execution context.
    ///
    /// This is useful for running multiple models in parallel with different inputs.
    pub fn clone_for_parallel_execution(&self) -> Result<Self> {
        // Wait for the model to be prepared
        if !self.is_prepared.read().map_or(false, |guard| *guard) {
            self.prepare()?;
        }
        
        // Create a new execution context
        let context = Arc::new(ExecutionContext::new((*self.options).clone()));
        
        // Clone the engine with shared components but a new context
        let cloned = Self {
            model: self.model.clone(),
            options: self.options.clone(),
            context,
            operator_registry: self.operator_registry.clone(),
            execution_graph: self.execution_graph.clone(),
            tensor_name_map: self.tensor_name_map.clone(),
            node_operators: self.node_operators.clone(),
            input_names: self.input_names.clone(),
            output_names: self.output_names.clone(),
            profile_events: Arc::new(Mutex::new(Vec::new())),
            profile_event_counter: Arc::new(Mutex::new(0)),
            is_prepared: Arc::new(RwLock::new(true)), // Mark as prepared since we verified above
            // Clone shared callbacks
            memory_allocation_callback: self.memory_allocation_callback.clone(),
            memory_deallocation_callback: self.memory_deallocation_callback.clone(),
            memory_reuse_callback: self.memory_reuse_callback.clone(),
            workspace_allocation_callback: self.workspace_allocation_callback.clone(),
        };
        
        Ok(cloned)
    }
}

impl std::fmt::Debug for ExecutionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Get read lock on is_prepared, handling error gracefully
        let is_prepared = self.is_prepared.read().map_or(false, |guard| *guard);
        
        let mut debug_struct = f.debug_struct("ThreadSafeExecutionEngine");
        debug_struct
            .field("model", &self.model)
            .field("options", &self.options)
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .field("is_prepared", &is_prepared);
            
        // Try to get execution graph info if available
        if let Ok(guard) = self.execution_graph.read() {
            if let Some(graph) = guard.as_ref() {
                debug_struct.field("num_nodes", &graph.nodes.len());
            }
        }
        
        debug_struct.finish()
    }
}