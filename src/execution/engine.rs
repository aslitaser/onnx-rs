use std::collections::{HashMap, HashSet, VecDeque};
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

/// The execution engine for ONNX models
pub struct ExecutionEngine {
    /// The ONNX model
    model: OnnxModel,
    /// Execution options
    options: ExecutionOptions,
    /// Execution context
    context: ExecutionContext,
    /// Operator registry
    operator_registry: OperatorRegistry,
    /// Execution graph (optimized from model)
    execution_graph: Option<ExecutionGraph>,
    /// Mapping from tensor names to tensor IDs
    tensor_name_map: HashMap<String, TensorId>,
    /// Mapping from node IDs to operators
    node_operators: HashMap<NodeId, Box<dyn Operator>>,
    /// Input tensor names
    input_names: Vec<String>,
    /// Output tensor names
    output_names: Vec<String>,
    /// Profiling events (if profiling enabled)
    profile_events: Vec<ProfileEvent>,
    /// Current profile event ID counter
    profile_event_counter: usize,
    /// Flag indicating if the engine is prepared
    is_prepared: bool,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new(model: OnnxModel, options: ExecutionOptions) -> Result<Self> {
        let input_names = model.graph.inputs.iter()
            .map(|input| input.name.clone())
            .collect();
            
        let output_names = model.graph.outputs.iter()
            .map(|output| output.name.clone())
            .collect();
        
        Ok(Self {
            model,
            options,
            context: ExecutionContext::new(options.clone()),
            operator_registry: OperatorRegistry::initialize_standard_operators(),
            execution_graph: None,
            tensor_name_map: HashMap::new(),
            node_operators: HashMap::new(),
            input_names,
            output_names,
            profile_events: Vec::new(),
            profile_event_counter: 0,
            is_prepared: false,
        })
    }
    
    /// Prepare the engine for execution
    pub fn prepare(&mut self) -> Result<()> {
        if self.is_prepared {
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
        self.execution_graph = Some(graph);
        
        self.is_prepared = true;
        Ok(())
    }
    
    /// Run the model with the given inputs
    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        if !self.is_prepared {
            self.prepare()?;
        }
        
        // Set input tensors
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
        for name in &self.output_names {
            if let Some(tensor_id) = self.tensor_name_map.get(name) {
                if let Some(tensor) = self.context.get_tensor(tensor_id) {
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
    
    /// Run a single node
    pub fn run_node(&mut self, node_id: NodeId) -> Result<()> {
        let graph = self.execution_graph.as_ref().ok_or_else(|| {
            Error::InvalidGraph("Execution graph not prepared".to_string())
        })?;
        
        // Find the node by ID
        let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Node with ID {} not found", node_id))
        })?;
        
        // Get the operator for this node
        let operator = self.node_operators.get(&node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Operator for node {} not found", node_id))
        })?;
        
        // Collect input tensors
        let mut input_tensors = Vec::new();
        for input_name in &node.inputs {
            // Skip optional inputs (indicated by empty name)
            if input_name.is_empty() {
                continue;
            }
            
            let tensor_id = self.tensor_name_map.get(input_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor ID for input '{}' not found", input_name))
            })?;
            
            let tensor = self.context.get_tensor(tensor_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor '{}' not found in context", input_name))
            })?;
            
            input_tensors.push(tensor);
        }
        
        // Prepare output tensors
        let mut output_tensors = Vec::new();
        for output_name in &node.outputs {
            let tensor_id = self.tensor_name_map.get(output_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Tensor ID for output '{}' not found", output_name))
            })?;
            
            if let Some(tensor) = self.context.get_tensor(tensor_id) {
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
                let tensor_id = self.tensor_name_map.get(output_name).ok_or_else(|| {
                    Error::InvalidGraph(format!("Tensor ID for output '{}' not found", output_name))
                })?;
                
                // Set the name of the tensor
                let mut output_tensor = output_tensors[i].clone();
                output_tensor.name = Some(output_name.clone());
                
                self.context.set_tensor(*tensor_id, output_tensor);
            }
        }
        
        Ok(())
    }
    
    /// Set an input tensor
    pub fn set_input_tensor(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        if !self.input_names.contains(&name.to_string()) {
            return Err(Error::InvalidGraph(format!(
                "Input tensor '{}' is not defined in the model", name
            )));
        }
        
        let tensor_id = self.tensor_name_map.get(name).ok_or_else(|| {
            Error::InvalidGraph(format!("Tensor ID for input '{}' not found", name))
        })?;
        
        // Set the name of the tensor
        let mut input_tensor = tensor;
        input_tensor.name = Some(name.to_string());
        
        self.context.set_tensor(*tensor_id, input_tensor);
        Ok(())
    }
    
    /// Get an output tensor
    pub fn get_output_tensor(&self, name: &str) -> Result<&Tensor> {
        if !self.output_names.contains(&name.to_string()) {
            return Err(Error::InvalidGraph(format!(
                "Output tensor '{}' is not defined in the model", name
            )));
        }
        
        let tensor_id = self.tensor_name_map.get(name).ok_or_else(|| {
            Error::InvalidGraph(format!("Tensor ID for output '{}' not found", name))
        })?;
        
        self.context.get_tensor(tensor_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Output tensor '{}' not found in context", name))
        })
    }
    
    /// Allocate workspace memory
    pub fn allocate_workspace(&mut self, size_bytes: usize) -> Result<WorkspaceGuard> {
        self.context.get_workspace(size_bytes)
    }
    
    /// Allocate tensors for intermediate outputs
    pub fn allocate_intermediate_tensors(&mut self) -> Result<()> {
        // This is a simplified implementation
        // In a real system, you would analyze the graph and allocate memory
        // efficiently, reusing buffers when possible
        
        Ok(())
    }
    
    /// Create execution order for the graph
    pub fn create_execution_order(&self) -> Result<Vec<NodeId>> {
        let graph = self.execution_graph.as_ref().ok_or_else(|| {
            Error::InvalidGraph("Execution graph not prepared".to_string())
        })?;
        
        // Topological sort
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
    
    /// Profile execution time for a model run
    pub fn profile_execution_time(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Duration>> {
        // Enable profiling
        let original_profiling = self.options.enable_profiling;
        self.options.enable_profiling = true;
        
        // Clear existing profiling data
        self.profile_events.clear();
        self.profile_event_counter = 0;
        
        // Run the model
        self.run(inputs)?;
        
        // Collect profiling results
        let mut results = HashMap::new();
        for event in &self.profile_events {
            if let Some(duration) = event.duration {
                results.insert(event.name.clone(), duration);
            }
        }
        
        // Restore original profiling setting
        self.options.enable_profiling = original_profiling;
        
        Ok(results)
    }
    
    /// Start a profiling event
    fn start_profile_event(&mut self, name: &str, node_id: Option<NodeId>) -> ProfileEventId {
        let id = ProfileEventId(self.profile_event_counter);
        self.profile_event_counter += 1;
        
        let event = ProfileEvent {
            id,
            name: name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
            node_id,
        };
        
        self.profile_events.push(event);
        id
    }
    
    /// End a profiling event
    fn end_profile_event(&mut self, id: ProfileEventId) {
        if let Some(event) = self.profile_events.iter_mut().find(|e| e.id == id) {
            let end_time = Instant::now();
            event.end_time = Some(end_time);
            event.duration = Some(end_time.duration_since(event.start_time));
        }
    }
    
    /// Assign tensor IDs to all tensors in the graph
    fn assign_tensor_ids(&mut self, graph: &ExecutionGraph) -> Result<()> {
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
        
        self.tensor_name_map = tensor_name_map;
        Ok(())
    }
    
    /// Create operators for each node in the graph
    fn create_node_operators(&mut self, graph: &ExecutionGraph) -> Result<()> {
        let mut node_operators = HashMap::new();
        
        for node in &graph.nodes {
            let operator = self.operator_registry.create_operator_for_node(node)?;
            node_operators.insert(node.id, operator);
        }
        
        self.node_operators = node_operators;
        Ok(())
    }
    
    /// Get the model
    pub fn model(&self) -> &OnnxModel {
        &self.model
    }
    
    /// Get the execution graph
    pub fn execution_graph(&self) -> Option<&ExecutionGraph> {
        self.execution_graph.as_ref()
    }
    
    /// Get the input tensor names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }
    
    /// Get the output tensor names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
    
    /// Get profile events
    pub fn profile_events(&self) -> &[ProfileEvent] {
        &self.profile_events
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

impl std::fmt::Debug for ExecutionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionEngine")
            .field("model", &self.model)
            .field("options", &self.options)
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .field("is_prepared", &self.is_prepared)
            .finish()
    }
}