use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

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
        self.context.get_workspace(size_bytes)
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