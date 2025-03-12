use std::collections::{HashMap, HashSet};
use std::cmp;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::model::{ExecutionGraph, NodeId};
use crate::ops::tensor::DataType;
use crate::memory::allocator::{MemoryAllocator, MemoryBlock};

/// Unique identifier for a tensor
pub type TensorId = usize;

/// Information about a tensor's memory requirements
#[derive(Debug, Clone)]
pub struct TensorMemoryInfo {
    /// Tensor ID
    pub id: TensorId,
    /// Tensor name
    pub name: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Data type
    pub data_type: DataType,
    /// Alignment requirement
    pub alignment: usize,
    /// Whether this tensor can be reused for in-place operations
    pub allow_inplace: bool,
}

/// In-place operation opportunity
#[derive(Debug, Clone)]
pub struct InplaceOpportunity {
    /// Node ID where the in-place operation can occur
    pub node_id: NodeId,
    /// Input tensor ID that can be overwritten
    pub input_id: TensorId,
    /// Output tensor ID that can share memory with the input
    pub output_id: TensorId,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Buffer sharing opportunity
#[derive(Debug, Clone)]
pub struct SharingOpportunity {
    /// First tensor ID
    pub first_id: TensorId,
    /// Second tensor ID
    pub second_id: TensorId,
    /// Size in bytes that can be shared
    pub size_bytes: usize,
}

/// Memory allocation plan for a tensor
#[derive(Debug, Clone)]
pub struct TensorAllocation {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Offset in the buffer
    pub offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Buffer index (which buffer this tensor is allocated in)
    pub buffer_index: usize,
}

/// Complete memory plan for the execution graph
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Tensor allocations
    pub allocations: HashMap<TensorId, TensorAllocation>,
    /// Tensor information
    pub tensor_info: HashMap<TensorId, TensorMemoryInfo>,
    /// Tensor lifetimes (first_use, last_use)
    pub lifetimes: HashMap<TensorId, (usize, usize)>,
    /// Buffer sizes
    pub buffer_sizes: Vec<usize>,
    /// In-place opportunities that were used
    pub inplace_ops: Vec<InplaceOpportunity>,
    /// Total memory requirement in bytes
    pub total_memory_bytes: usize,
    /// Execution order for nodes
    pub execution_order: Vec<NodeId>,
}

/// Mapping from tensor ID to memory block
pub type BufferMap = HashMap<TensorId, MemoryBlock>;

/// Memory planner for optimizing memory usage during execution
pub struct MemoryPlanner {
    /// Current execution graph being analyzed
    current_graph: Option<Arc<ExecutionGraph>>,
    /// Cache of tensor information
    tensor_info_cache: RwLock<HashMap<TensorId, TensorMemoryInfo>>,
    /// Cache of tensor ID mappings
    tensor_id_map_cache: RwLock<HashMap<String, TensorId>>,
    /// Atomic counter for generating unique tensor IDs
    next_tensor_id: AtomicUsize,
}

impl MemoryPlanner {
    /// Create a new memory planner
    pub fn new() -> Self {
        Self {
            current_graph: None,
            tensor_info_cache: RwLock::new(HashMap::new()),
            tensor_id_map_cache: RwLock::new(HashMap::new()),
            next_tensor_id: AtomicUsize::new(1), // Start IDs from 1 to reserve 0 for special cases
        }
    }
    
    /// Generate a new unique tensor ID
    fn generate_tensor_id(&self) -> TensorId {
        self.next_tensor_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Plan memory usage for an execution graph
    pub fn plan_memory_usage(
        &mut self,
        graph: &ExecutionGraph,
    ) -> Result<MemoryPlan> {
        // Store the current graph for reference
        self.current_graph = Some(Arc::new(graph.clone()));
        
        // Clear caches for fresh analysis
        if let Ok(mut cache) = self.tensor_info_cache.write() {
            cache.clear();
        } else {
            return Err(Error::InvalidModel("Failed to acquire write lock for tensor_info_cache".to_string()));
        }
        
        if let Ok(mut cache) = self.tensor_id_map_cache.write() {
            cache.clear();
        } else {
            return Err(Error::InvalidModel("Failed to acquire write lock for tensor_id_map_cache".to_string()));
        }
        
        // Reset the tensor ID counter
        self.next_tensor_id.store(1, Ordering::SeqCst);
        
        // Determine execution order
        let execution_order = self.determine_execution_order(graph)?;

        // Compute tensor lifetimes
        let lifetimes = self.compute_tensor_lifetimes(graph, &execution_order)?;

        // Gather tensor information
        let tensor_info = self.gather_tensor_info(graph)?;
        
        // Cache the tensor info for later use
        if let Ok(mut cache) = self.tensor_info_cache.write() {
            *cache = tensor_info.clone();
        }

        // Find in-place operation opportunities
        let inplace_ops = self.inplace_operations_analysis(graph)?;

        // Apply in-place optimizations
        let (tensor_info, lifetimes) = self.apply_inplace_optimizations(
            tensor_info,
            lifetimes.clone(),
            inplace_ops.clone(),
        )?;

        // Create initial memory plan
        let mut plan = MemoryPlan {
            allocations: HashMap::new(),
            tensor_info,
            lifetimes,
            buffer_sizes: vec![0],
            inplace_ops,
            total_memory_bytes: 0,
            execution_order,
        };

        // Optimize memory layout
        self.optimize_memory_layout(&mut plan)?;

        Ok(plan)
    }

    /// Determine execution order for nodes in the graph
    fn determine_execution_order(&self, graph: &ExecutionGraph) -> Result<Vec<NodeId>> {
        // This is a simplified version - in practice, you would use
        // the topological sort from the execution engine
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

    /// Compute tensor lifetimes based on execution order
    pub fn compute_tensor_lifetimes(
        &self,
        graph: &ExecutionGraph,
        execution_order: &[NodeId],
    ) -> Result<HashMap<TensorId, (usize, usize)>> {
        let mut lifetimes = HashMap::new();
        let mut tensor_producers = HashMap::new();
        let mut tensor_consumers = HashMap::new();

        // Build tensor producer and consumer maps
        for (idx, &node_id) in execution_order.iter().enumerate() {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;

            // Record outputs (produced by this node)
            for output_name in &node.outputs {
                tensor_producers.insert(output_name.clone(), idx);
            }

            // Record inputs (consumed by this node)
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue; // Skip empty inputs (optional)
                }

                tensor_consumers
                    .entry(input_name.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        // Compute lifetimes
        for (tensor_name, &producer_idx) in &tensor_producers {
            let first_use = producer_idx;

            // Find the last consumer, or use the end of execution if this is an output
            let is_output = graph.output_nodes.iter().any(|&node_id| {
                let node = graph.nodes.iter().find(|n| n.id == node_id).unwrap();
                node.outputs.contains(tensor_name)
            });

            let last_use = if is_output {
                execution_order.len() // Keep until the end
            } else if let Some(consumers) = tensor_consumers.get(tensor_name) {
                consumers.iter().cloned().max().unwrap_or(first_use)
            } else {
                first_use // No consumers, tensor is used only once
            };

            // Get the tensor ID from the tensor_id_map
            let tensor_id = *tensor_id_map.get(tensor_name).ok_or_else(|| {
                Error::InvalidModel(format!("Missing ID for tensor {}", tensor_name))
            })?;

            lifetimes.insert(tensor_id, (first_use, last_use));
        }

        Ok(lifetimes)
    }

    /// Gather information about tensors in the graph
    fn gather_tensor_info(&self, graph: &ExecutionGraph) -> Result<HashMap<TensorId, TensorMemoryInfo>> {
        let mut tensor_info = HashMap::new();
        let mut tensor_id_map = HashMap::new();

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = self.generate_tensor_id();
            tensor_id_map.insert(name, id);
        }

        // Process all nodes in the graph
        for node in &graph.nodes {
            // Process outputs from this node
            for output_name in &node.outputs {
                if output_name.is_empty() {
                    continue;
                }
                
                // Get the tensor ID from the map
                let tensor_id = *tensor_id_map.get(output_name).ok_or_else(|| {
                    Error::InvalidModel(format!("Missing ID for tensor {}", output_name))
                })?;

                // Determine if this is a model output
                let is_model_output = graph.output_nodes.iter().any(|&node_id| {
                    let node = graph.nodes.iter().find(|n| n.id == node_id).unwrap();
                    node.outputs.contains(output_name)
                });

                // Determine tensor shape and data type based on operation
                let (shape, data_type) = self.infer_tensor_info(graph, node, output_name)?;
                
                // Calculate size in bytes based on shape and data type
                let element_size = data_type.size_in_bytes();
                let total_elements: usize = shape.iter().product();
                
                // Handle potential overflow
                let size_bytes = total_elements.checked_mul(element_size).ok_or_else(|| {
                    Error::InvalidModel(format!(
                        "Integer overflow calculating size for tensor {} with {} elements of size {} bytes",
                        output_name, total_elements, element_size
                    ))
                })?;
                
                // Determine appropriate alignment for the data type
                // For SIMD operations, align to cache line boundaries (64 bytes) for float32/64 tensors
                // and tensors with dimensions divisible by SIMD vector lengths
                let alignment = if data_type == DataType::Float32 || data_type == DataType::Float64 {
                    let is_simd_friendly = shape.iter().any(|&dim| dim % 8 == 0 || dim % 16 == 0);
                    if is_simd_friendly && size_bytes >= 64 {
                        64 // Cache line size for SIMD-friendly operations
                    } else if size_bytes >= 32 {
                        32 // For smaller but still SIMD-usable tensors
                    } else {
                        16 // Minimum alignment for float tensors
                    }
                } else {
                    // For other data types, use a reasonable alignment based on the element size
                    std::cmp::max(element_size, 8) // At least 8-byte alignment for all tensors
                };

                tensor_info.insert(
                    tensor_id,
                    TensorMemoryInfo {
                        id: tensor_id,
                        name: output_name.clone(),
                        size_bytes,
                        data_type,
                        alignment,
                        allow_inplace: !is_model_output, // Outputs can be overwritten unless they're model outputs
                    },
                );
            }

            // Process inputs as well
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue; // Skip empty inputs (optional)
                }

                // Get the tensor ID from the map
                let tensor_id = *tensor_id_map.get(input_name).ok_or_else(|| {
                    Error::InvalidModel(format!("Missing ID for tensor {}", input_name))
                })?;

                // Only add if not already present
                if !tensor_info.contains_key(&tensor_id) {
                    // For inputs, try to find the tensor info from the graph's inputs
                    let (shape, data_type) = self.get_input_tensor_info(graph, input_name)?;
                    
                    // Calculate size based on shape and data type
                    let element_size = data_type.size_in_bytes();
                    let total_elements: usize = shape.iter().product();
                    
                    // Handle potential overflow
                    let size_bytes = total_elements.checked_mul(element_size).ok_or_else(|| {
                        Error::InvalidModel(format!(
                            "Integer overflow calculating size for tensor {} with {} elements of size {} bytes",
                            input_name, total_elements, element_size
                        ))
                    })?;
                    
                    // Determine appropriate alignment
                    let alignment = if data_type == DataType::Float32 || data_type == DataType::Float64 {
                        let is_simd_friendly = shape.iter().any(|&dim| dim % 8 == 0 || dim % 16 == 0);
                        if is_simd_friendly && size_bytes >= 64 {
                            64 // Cache line size for SIMD-friendly operations
                        } else {
                            16 // Minimum alignment for float tensors
                        }
                    } else {
                        // For other data types, use a reasonable alignment based on the element size
                        std::cmp::max(element_size, 8) // At least 8-byte alignment for all tensors
                    };

                    tensor_info.insert(
                        tensor_id,
                        TensorMemoryInfo {
                            id: tensor_id,
                            name: input_name.clone(),
                            size_bytes,
                            data_type,
                            alignment,
                            allow_inplace: false, // Inputs should not be overwritten
                        },
                    );
                }
            }
        }

        Ok(tensor_info)
    }
    
    /// Infer tensor shape and data type based on operation and inputs
    fn infer_tensor_info(&self, graph: &ExecutionGraph, node: &Node, tensor_name: &str) -> Result<(Vec<usize>, DataType)> {
        // In a production system, this would use the shape inference system
        // Here we'll implement a minimal version for common operations
        
        match node.op_type.as_str() {
            "Conv" => {
                // For convolution, we need to get the input shape, kernel shape, etc.
                // This is simplified - real implementation would parse attributes
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    // Output shape depends on input shape, kernel size, stride, padding, etc.
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let kernel_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    if input_shape.len() == 4 && kernel_shape.len() == 4 {
                        // NCHW layout: [batch, channels, height, width]
                        let batch_size = input_shape[0];
                        let out_channels = kernel_shape[0];
                        
                        // Simple approximation - in reality would account for padding, stride, etc.
                        let out_h = input_shape[2]; // Simplified
                        let out_w = input_shape[3]; // Simplified
                        
                        return Ok((vec![batch_size, out_channels, out_h, out_w], DataType::Float32));
                    }
                }
            },
            "MatMul" => {
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    let a_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let b_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    if a_shape.len() >= 2 && b_shape.len() >= 2 {
                        // For MatMul: [M,K] * [K,N] -> [M,N]
                        let m = a_shape[a_shape.len() - 2];
                        let n = b_shape[b_shape.len() - 1];
                        
                        // Handle broadcasting if tensors have more than 2 dimensions
                        let mut result_shape = Vec::new();
                        
                        if a_shape.len() > 2 || b_shape.len() > 2 {
                            // Broadcast the batch dimensions
                            let a_batch_dims = &a_shape[0..a_shape.len() - 2];
                            let b_batch_dims = &b_shape[0..b_shape.len() - 2];
                            
                            // Compute the broadcasted batch dimensions
                            let batch_dims = self.broadcast_shapes(a_batch_dims, b_batch_dims)?;
                            result_shape.extend_from_slice(&batch_dims);
                        }
                        
                        // Add the matrix multiplication result dimensions
                        result_shape.push(m);
                        result_shape.push(n);
                        
                        return Ok((result_shape, DataType::Float32));
                    }
                }
            },
            // Add cases for common operations: Add, Mul, Relu, etc.
            "Add" | "Sub" | "Mul" | "Div" => {
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    let a_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let b_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    // For elementwise ops, output shape is the broadcasted shape
                    let output_shape = self.broadcast_shapes(&a_shape, &b_shape)?;
                    
                    // Output data type is the same as input (assuming same type inputs)
                    let data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((output_shape, data_type));
                }
            },
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" => {
                if node.inputs.len() >= 1 && tensor_name == &node.outputs[0] {
                    // These activation functions preserve shape and data type
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let input_data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((input_shape, input_data_type));
                }
            },
            // Default case for unknown operations
            _ => {
                // For unknown operations, attempt to copy shape from first input
                if !node.inputs.is_empty() && tensor_name == &node.outputs[0] {
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let input_data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((input_shape, input_data_type));
                }
            }
        }
        
        // Fallback: If we can't infer the shape, use a reasonable default
        Ok((vec![1, 1, 64, 64], DataType::Float32))
    }
    
    /// Get the shape of a tensor from the graph
    fn get_tensor_shape(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<Vec<usize>> {
        // In a production system, this would query the shape inference system or the model
        // For this implementation, we'll use some reasonable defaults based on the tensor name
        
        if tensor_name.contains("weight") || tensor_name.contains("kernel") {
            // Weights for conv are typically [out_channels, in_channels, kernel_h, kernel_w]
            Ok(vec![64, 3, 3, 3])
        } else if tensor_name.contains("bias") {
            // Bias is typically [out_channels]
            Ok(vec![64])
        } else if tensor_name.contains("input") || tensor_name.contains("image") {
            // Input tensors are typically [batch_size, channels, height, width]
            Ok(vec![1, 3, 224, 224])
        } else if tensor_name.contains("pool") || tensor_name.contains("feature") {
            // Feature maps often have shape [batch_size, channels, height, width]
            Ok(vec![1, 64, 112, 112])
        } else if tensor_name.contains("fc") || tensor_name.contains("dense") {
            // Fully connected layers typically have shape [batch_size, features]
            Ok(vec![1, 1024])
        } else if tensor_name.contains("output") || tensor_name.contains("logits") {
            // Output tensors often have shape [batch_size, num_classes]
            Ok(vec![1, 1000])
        } else {
            // Default shape for unknown tensors
            Ok(vec![1, 64, 64, 64])
        }
    }
    
    /// Get the data type of a tensor from the graph
    fn get_tensor_data_type(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<DataType> {
        // In a production system, this would query the model's tensor definitions
        // Here we'll just return Float32 for all tensors
        Ok(DataType::Float32)
    }
    
    /// Get tensor info for input tensors
    fn get_input_tensor_info(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<(Vec<usize>, DataType)> {
        // In a production system, this would query the model's input definitions
        // For this implementation, we'll use the same logic as get_tensor_shape
        let shape = self.get_tensor_shape(graph, tensor_name)?;
        let data_type = self.get_tensor_data_type(graph, tensor_name)?;
        
        Ok((shape, data_type))
    }
    
    /// Broadcast two shapes according to broadcasting rules
    fn broadcast_shapes(&self, shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
        let rank1 = shape1.len();
        let rank2 = shape2.len();
        let result_rank = std::cmp::max(rank1, rank2);
        
        let mut result_shape = Vec::with_capacity(result_rank);
        
        for i in 0..result_rank {
            let dim1 = if i >= result_rank - rank1 {
                shape1[i - (result_rank - rank1)]
            } else {
                1
            };
            
            let dim2 = if i >= result_rank - rank2 {
                shape2[i - (result_rank - rank2)]
            } else {
                1
            };
            
            if dim1 == 1 {
                result_shape.push(dim2);
            } else if dim2 == 1 {
                result_shape.push(dim1);
            } else if dim1 == dim2 {
                result_shape.push(dim1);
            } else {
                return Err(Error::InvalidModel(format!(
                    "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions at index {}",
                    shape1, shape2, i
                )));
            }
        }
        
        Ok(result_shape)
    }
    
    /// Collect all tensor names from the graph
    fn collect_tensor_names(&self, graph: &ExecutionGraph) -> HashSet<String> {
        let mut tensor_names = HashSet::new();
        
        // Add all inputs and outputs from all nodes
        for node in &graph.nodes {
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    tensor_names.insert(input_name.clone());
                }
            }
            
            for output_name in &node.outputs {
                if !output_name.is_empty() {
                    tensor_names.insert(output_name.clone());
                }
            }
        }
        
        tensor_names
    }

    /// Analyze opportunities for in-place operations
    pub fn inplace_operations_analysis(&self, graph: &ExecutionGraph) -> Result<Vec<InplaceOpportunity>> {
        let mut opportunities = Vec::new();
        let tensor_info = self.gather_tensor_info(graph)?;
        let tensor_id_map = self.create_tensor_id_map(graph)?;

        for node in &graph.nodes {
            // Different operations have different in-place capabilities
            match node.op_type.as_str() {
                // Unary operations that can generally be done in-place
                "Relu" | "LeakyRelu" | "Sigmoid" | "Tanh" | "Abs" | "Exp" | "Log" | "Sqrt" | "Neg" => {
                if !node.inputs.is_empty() && !node.inputs[0].is_empty() && !node.outputs.is_empty() {
                    let input_name = &node.inputs[0];
                    let output_name = &node.outputs[0];
                    
                    // Get tensor IDs from the map
                    if let (Some(&input_id), Some(&output_id)) = (
                        tensor_id_map.get(input_name),
                        tensor_id_map.get(output_name)
                    ) {
                        // Check if input can be overwritten
                        if let Some(input_info) = tensor_info.get(&input_id) {
                            if input_info.allow_inplace {
                                // Check that input is only used once (in this operation)
                                if self.is_last_use_of_tensor(graph, input_name, node.id)? {
                                    // Get the size of the tensor
                                    let size_bytes = input_info.size_bytes;
                                    
                                    opportunities.push(InplaceOpportunity {
                                        node_id: node.id,
                                        input_id,
                                        output_id,
                                        size_bytes,
                                    });
                                }
                            }
                        }
                    }
                }
            },
            _ => {}
        }
    }
        
        Ok(opportunities)
    }
    
    /// Create a mapping from tensor names to tensor IDs
    fn create_tensor_id_map(&self, graph: &ExecutionGraph) -> Result<HashMap<String, TensorId>> {
        let mut tensor_id_map = HashMap::new();

        // First, check if we have a cached mapping
        if let Ok(cache) = self.tensor_id_map_cache.read() {
            if !cache.is_empty() {
                return Ok(cache.clone());
            }
        }

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = self.generate_tensor_id();
            tensor_id_map.insert(name, id);
        }
        
        // Update cache
        if let Ok(mut cache) = self.tensor_id_map_cache.write() {
            *cache = tensor_id_map.clone();
        }
        
        Ok(tensor_id_map)
    }
    
    /// Check if this is the last use of a tensor in the graph
    fn is_last_use_of_tensor(&self, graph: &ExecutionGraph, tensor_name: &str, current_node_id: NodeId) -> Result<bool> {
        // Get the execution order
        let execution_order = self.determine_execution_order(graph)?;
        
        // Find the position of the current node
        let current_pos = execution_order.iter().position(|&id| id == current_node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Node with ID {} not found in execution order", current_node_id))
        })?;
        
        // Check all nodes after the current one
        for &node_id in &execution_order[current_pos + 1..] {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;
            
            // If any node uses this tensor as input, it's not the last use
            if node.inputs.contains(&tensor_name) {
                return Ok(false);
            }
        }
        
        // Check if the tensor is a model output
        let is_model_output = graph.output_nodes.iter().any(|&node_id| {
            let node = graph.nodes.iter().find(|n| n.id == node_id).unwrap_or_else(|| panic!("Node not found"));
            node.outputs.contains(&tensor_name)
        });
        
        // If it's a model output, it cannot be used in-place
        if is_model_output {
            return Ok(false);
        }
        
        // If we haven't found any further uses, it's the last use
        Ok(true)
    }

    /// Apply in-place optimizations to tensor info and lifetimes
    fn apply_inplace_optimizations(
        &self,
        mut tensor_info: HashMap<TensorId, TensorMemoryInfo>,
        mut lifetimes: HashMap<TensorId, (usize, usize)>,
        inplace_ops: Vec<InplaceOpportunity>,
    ) -> Result<(HashMap<TensorId, TensorMemoryInfo>, HashMap<TensorId, (usize, usize)>)> {
        for op in &inplace_ops {
            // Check if both tensors exist in our maps
            if let (Some(input_info), Some(output_info)) = (
                tensor_info.get(&op.input_id),
                tensor_info.get(&op.output_id),
            ) {
                // Check if the input tensor allows in-place operations
                if input_info.allow_inplace
                    // Check data types match
                    && input_info.data_type == output_info.data_type
                    // Check sizes are compatible
                    && input_info.size_bytes >= output_info.size_bytes
                {
                    // Extend the lifetime of the input tensor to cover the output tensor
                    if let (Some(&(input_first, input_last)), Some(&(output_first, output_last))) = (
                        lifetimes.get(&op.input_id),
                        lifetimes.get(&op.output_id),
                    ) {
                        // Update the input tensor's lifetime to be the union of both
                        let new_lifetime = (
                            cmp::min(input_first, output_first),
                            cmp::max(input_last, output_last),
                        );
                        lifetimes.insert(op.input_id, new_lifetime);

                        // Remove the output tensor from planning
                        // (it will share memory with the input)
                        lifetimes.remove(&op.output_id);
                        tensor_info.remove(&op.output_id);

                        // In a real implementation, you would add the output tensor
                        // to a map that tracks which tensors share memory
                    }
                }
            }
        }

        Ok((tensor_info, lifetimes))
    }

    /// Analyze opportunities for buffer sharing
    pub fn buffer_sharing_analysis(
        &self,
        lifetimes: &HashMap<TensorId, (usize, usize)>,
    ) -> Result<Vec<SharingOpportunity>> {
        let mut opportunities = Vec::new();

        // Create a list of tensors with their sizes
        let mut tensors_with_sizes: Vec<(TensorId, usize, (usize, usize))> = Vec::new();
        
        // Thread-safe access to tensor info cache
        let tensor_info = match self.tensor_info_cache.read() {
            Ok(cache) => cache,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_info_cache".to_string())),
        };
        
        for (&tensor_id, &lifetime) in lifetimes.iter() {
            if let Some(info) = tensor_info.get(&tensor_id) {
                tensors_with_sizes.push((tensor_id, info.size_bytes, lifetime));
            }
        }

        // Sort tensors by size (largest first)
        tensors_with_sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Check each pair of tensors for sharing opportunities
        for i in 0..tensors_with_sizes.len() {
            let (id1, size1, (first1, last1)) = tensors_with_sizes[i];
            
            // Skip tensors that require special alignment if they're very small
            // (sharing very small tensors often doesn't save much memory)
            if size1 < 64 {
                continue;
            }
            
            for j in i + 1..tensors_with_sizes.len() {
                let (id2, size2, (first2, last2)) = tensors_with_sizes[j];

                // Check if lifetimes don't overlap
                if last1 < first2 || last2 < first1 {
                    // Tensors don't overlap in time, check alignment compatibility
                    if let (Some(info1), Some(info2)) = (
                        tensor_info.get(&id1), 
                        tensor_info.get(&id2)
                    ) {
                        // Check if they have compatible alignment requirements
                        // For simplicity, we'll only share buffers with the same alignment
                        // A more sophisticated implementation would adjust offsets to satisfy
                        // both alignment requirements
                        if info1.alignment == info2.alignment {
                            // Check data type compatibility
                            // For some data types, we might require additional padding
                            let compatible_types = (info1.data_type == info2.data_type) ||
                                               (info1.data_type.size_in_bytes() == info2.data_type.size_in_bytes());
                            
                            if compatible_types {
                                // Use the smaller of the two sizes for the amount saved
                                let shared_size = std::cmp::min(size1, size2);
                                
                                opportunities.push(SharingOpportunity {
                                    first_id: id1,
                                    second_id: id2,
                                    size_bytes: shared_size,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Apply heuristic to limit the number of sharing opportunities
        // Too many sharing edges can make allocation complex
        if opportunities.len() > 100 {
            // Sort by size saved (largest first) and take the top 100
            opportunities.sort_unstable_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
            opportunities.truncate(100);
        }

        Ok(opportunities)
    }

    /// Compute the optimal allocation order for tensors
    pub fn compute_optimal_allocation_order(
        &self,
        sizes: &[(TensorId, usize)],
        lifetimes: &HashMap<TensorId, (usize, usize)>,
    ) -> Vec<TensorId> {
        // Sort tensors by size (largest first)
        let mut ids_with_sizes = sizes.to_vec();
        ids_with_sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        
        // Extract just the tensor IDs
        let mut result: Vec<_> = ids_with_sizes.iter().map(|(id, _)| *id).collect();
        
        // Further sort by lifetime length (longest first) for tensors of the same size
        result.sort_by(|&a, &b| {
            let size_a = ids_with_sizes.iter().find(|(id, _)| *id == a).unwrap().1;
            let size_b = ids_with_sizes.iter().find(|(id, _)| *id == b).unwrap().1;
            
            if size_a == size_b {
                let lifetime_a = lifetimes.get(&a).map(|(first, last)| last - first).unwrap_or(0);
                let lifetime_b = lifetimes.get(&b).map(|(first, last)| last - first).unwrap_or(0);
                lifetime_b.cmp(&lifetime_a)
            } else {
                size_b.cmp(&size_a)
            }
        });
        
        result
    }
    
    /// Optimize memory layout for a memory plan
    pub fn optimize_memory_layout(&self, plan: &mut MemoryPlan) -> Result<usize> {
        // Start with all tensors in a single buffer
        let mut buffer_index = 0;
        let mut current_offset = 0;
        let mut max_offset = 0;
        let mut bytes_saved = 0;
        
        // Get sharing opportunities
        let sharing_opportunities = self.buffer_sharing_analysis(&plan.lifetimes)?;
        
        // Create a map of tensor sizes
        let tensor_sizes: Vec<_> = plan.tensor_info.iter()
            .map(|(id, info)| (*id, info.size_bytes))
            .collect();
        
        // Compute optimal allocation order
        let allocation_order = self.compute_optimal_allocation_order(
            &tensor_sizes,
            &plan.lifetimes,
        );
        
        // Track allocated tensors and their buffer ranges
        let mut allocated_ranges = Vec::new();
        
        // Allocate tensors in optimal order
        for tensor_id in allocation_order {
            // Skip if already allocated
            if plan.allocations.contains_key(&tensor_id) {
                continue;
            }
            
            // Get tensor info
            let info = if let Some(info) = plan.tensor_info.get(&tensor_id) {
                info
            } else {
                continue; // Skip if no info (e.g., in-place outputs)
            };
            
            // Get tensor lifetime
            let lifetime = if let Some(&lifetime) = plan.lifetimes.get(&tensor_id) {
                lifetime
            } else {
                continue; // Skip if no lifetime
            };
            
            // Try to find a space in the buffer where this tensor can fit
            // without overlapping with other tensors' lifetimes
            let alignment = info.alignment;
            let size = info.size_bytes;
            
            // Find the earliest position where we can allocate this tensor
            let mut can_allocate_at = 0;
            let mut found_space = false;
            
            // Align to required alignment
            let align_offset = |offset: usize, alignment: usize| -> usize {
                (offset + alignment - 1) & !(alignment - 1)
            };
            
            while !found_space {
                // Align the offset
                can_allocate_at = align_offset(can_allocate_at, alignment);
                
                // Check if this offset overlaps with any allocated tensor
                let end_offset = can_allocate_at + size;
                let mut overlaps = false;
                
                for &(other_id, other_offset, other_size) in &allocated_ranges {
                    // Get the other tensor's lifetime
                    let other_lifetime = if let Some(&lifetime) = plan.lifetimes.get(&other_id) {
                        lifetime
                    } else {
                        continue;
                    };
                    
                    // Check if the lifetimes overlap
                    let lifetimes_overlap = !(lifetime.1 < other_lifetime.0 || other_lifetime.1 < lifetime.0);
                    
                    // Check if the memory regions overlap
                    let regions_overlap = !(end_offset <= other_offset || can_allocate_at >= other_offset + other_size);
                    
                    if lifetimes_overlap && regions_overlap {
                        overlaps = true;
                        can_allocate_at = other_offset + other_size;
                        break;
                    }
                }
                
                if !overlaps {
                    found_space = true;
                }
            }
            
            // Allocate the tensor at the found offset
            let allocation = TensorAllocation {
                tensor_id,
                offset: can_allocate_at,
                size_bytes: size,
                buffer_index,
            };
            
            // Update tracking structures
            plan.allocations.insert(tensor_id, allocation);
            allocated_ranges.push((tensor_id, can_allocate_at, size));
            
            // Update the maximum offset
            max_offset = cmp::max(max_offset, can_allocate_at + size);
        }
        
        // Apply buffer sharing optimizations
        for op in &sharing_opportunities {
            // If both tensors are allocated, we've already optimized them
            if plan.allocations.contains_key(&op.first_id) && plan.allocations.contains_key(&op.second_id) {
                continue;
            }
            
            // If one is allocated and the other isn't, it means the other is
            // part of an in-place operation and has been removed
            if plan.allocations.contains_key(&op.first_id) && !plan.allocations.contains_key(&op.second_id) {
                bytes_saved += op.size_bytes;
            } else if !plan.allocations.contains_key(&op.first_id) && plan.allocations.contains_key(&op.second_id) {
                bytes_saved += op.size_bytes;
            }
            // If neither is allocated, both might be part of in-place operations
        }
        
        // Update buffer sizes
        plan.buffer_sizes = vec![max_offset];
        plan.total_memory_bytes = max_offset;
        
        Ok(bytes_saved)
    }
    
    /// Allocate memory buffers according to the memory plan
    pub fn allocate_buffers_from_plan(
        &self,
        plan: &MemoryPlan,
        allocator: &mut dyn MemoryAllocator,
    ) -> Result<BufferMap> {
        let mut buffer_map = HashMap::new();
        
        // Allocate each buffer
        let mut buffer_blocks = Vec::new();
        for (i, &size) in plan.buffer_sizes.iter().enumerate() {
            // For simplicity, we'll use a common alignment for all buffers
            // In practice, this would depend on the tensors' requirements
            let alignment = 64;
            
            let block = allocator.allocate(size, alignment)?;
            buffer_blocks.push(block);
        }
        
        // Assign memory blocks to tensors based on allocations
        for (tensor_id, allocation) in &plan.allocations {
            let buffer_index = allocation.buffer_index;
            if buffer_index >= buffer_blocks.len() {
                return Err(Error::InvalidModel(format!(
                    "Buffer index {} out of bounds (only {} buffers allocated)",
                    buffer_index, buffer_blocks.len()
                )));
            }
            
            // Get the base memory block
            let base_block = &buffer_blocks[buffer_index];
            
            // Create a sub-block for this tensor
            let ptr = unsafe {
                let base_ptr = base_block.ptr().as_ptr();
                let tensor_ptr = base_ptr.add(allocation.offset);
                NonNull::new_unchecked(tensor_ptr)
            };
            
            let tensor_block = MemoryBlock::new(
                ptr,
                allocation.size_bytes,
                base_block.alignment(),
                allocation.offset,
            );
            
            buffer_map.insert(*tensor_id, tensor_block);
        }
        
        Ok(buffer_map)
    }
}

// Needed for creating a sub-block
use std::ptr::NonNull;