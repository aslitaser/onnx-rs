use std::collections::{HashMap, HashSet};
use std::cmp;
use std::sync::Arc;

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
pub struct MemoryPlanner;

impl MemoryPlanner {
    /// Create a new memory planner
    pub fn new() -> Self {
        Self
    }

    /// Plan memory usage for an execution graph
    pub fn plan_memory_usage(
        &self,
        graph: &ExecutionGraph,
    ) -> Result<MemoryPlan> {
        // Determine execution order
        let execution_order = self.determine_execution_order(graph)?;

        // Compute tensor lifetimes
        let lifetimes = self.compute_tensor_lifetimes(graph, &execution_order)?;

        // Gather tensor information
        let tensor_info = self.gather_tensor_info(graph)?;

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
        let mut next_id: TensorId = 1; // Start IDs from 1 to reserve 0 for special cases

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = next_id;
            tensor_id_map.insert(name, id);
            next_id += 1;
        }

        // In a real implementation, you would analyze the graph to determine
        // tensor shapes, data types, and memory requirements
        // This is simplified and uses placeholder values

        for node in &graph.nodes {
            // Process outputs from this node
            for output_name in &node.outputs {
                // Create a tensor ID for the name
                let tensor_id = output_name.as_bytes().iter().sum::<u8>() as usize;

                // In a real system, you would compute actual sizes based on shape and data type
                // For this implementation, we'll use a simple placeholder
                let size_bytes = 1024; // Placeholder
                let data_type = DataType::Float32; // Placeholder
                let alignment = 64; // Common alignment for SIMD operations

                tensor_info.insert(
                    tensor_id,
                    TensorMemoryInfo {
                        id: tensor_id,
                        name: output_name.clone(),
                        size_bytes,
                        data_type,
                        alignment,
                        allow_inplace: true, // Default to allowed
                    },
                );
            }

            // Process inputs as well
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue; // Skip empty inputs (optional)
                }

                // Create a tensor ID for the name
                let tensor_id = input_name.as_bytes().iter().sum::<u8>() as usize;

                // Only add if not already present
                if !tensor_info.contains_key(&tensor_id) {
                    // For inputs, we would typically get this information from the graph's inputs
                    let size_bytes = 1024; // Placeholder
                    let data_type = DataType::Float32; // Placeholder
                    let alignment = 64;

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

                    // Create tensor IDs
                    let input_id = input_name.as_bytes().iter().sum::<u8>() as usize;
                    let output_id = output_name.as_bytes().iter().sum::<u8>() as usize;

                    // In a real system, you would check shapes and data types
                    // to ensure they're compatible
                    let size_bytes = 1024; // Placeholder

                    opportunities.push(InplaceOpportunity {
                        node_id: node.id,
                        input_id,
                        output_id,
                        size_bytes,
                    });
                }
            }
        }

        Ok(opportunities)
    }
    
    /// Create a mapping from tensor names to tensor IDs
    fn create_tensor_id_map(&self, graph: &ExecutionGraph) -> Result<HashMap<String, TensorId>> {
        let mut tensor_id_map = HashMap::new();
        let mut next_id: TensorId = 1; // Start IDs from 1 to reserve 0 for special cases

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = next_id;
            tensor_id_map.insert(name, id);
            next_id += 1;
        }
        
        Ok(tensor_id_map)
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
    ) -> Vec<SharingOpportunity> {
        let mut opportunities = Vec::new();

        // Create a list of tensors sorted by size (largest first)
        let mut tensor_ids: Vec<_> = lifetimes.keys().cloned().collect();
        tensor_ids.sort_unstable_by(|a, b| {
            let size_a = 1024; // Placeholder - would use actual sizes
            let size_b = 1024; // Placeholder
            size_b.cmp(&size_a) // Sort largest first
        });

        // Check each pair of tensors for sharing opportunities
        for i in 0..tensor_ids.len() {
            for j in i + 1..tensor_ids.len() {
                let id1 = tensor_ids[i];
                let id2 = tensor_ids[j];

                // Check if lifetimes don't overlap
                if let (Some(&(first1, last1)), Some(&(first2, last2))) =
                    (lifetimes.get(&id1), lifetimes.get(&id2))
                {
                    if last1 < first2 || last2 < first1 {
                        // Tensors don't overlap in time, they can share memory
                        let size_bytes = 1024; // Placeholder - would use minimum of both sizes

                        opportunities.push(SharingOpportunity {
                            first_id: id1,
                            second_id: id2,
                            size_bytes,
                        });
                    }
                }
            }
        }

        opportunities
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
        let sharing_opportunities = self.buffer_sharing_analysis(&plan.lifetimes);
        
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