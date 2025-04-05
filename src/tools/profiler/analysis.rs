// Analysis module for profiler
// Contains functions for analyzing profile data, finding bottlenecks, and memory efficiency

use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{Duration, Instant};
use std::cmp::Ordering;
use anyhow::{Result, anyhow};

use crate::{
    model::{NodeId, TensorId, Graph},
    ops::tensor::{DataType, Tensor},
    memory::{allocator::MemoryAllocator, workspace::MemoryWorkspace},
};

use super::types::{
    ProfileEventType, ProfileEvent, ProfileResults,
    PerformanceStats, MemoryProfileResults, MemoryEvent, MemoryEventType,
    TensorLifetime, ParallelismStats, TensorStats, ModelInfo, GraphInfo,
    EdgeType, TimedNode, ExecutionEdge, OptimizationImpact, MemoryEfficiencyMetrics,
    TensorReuseOpportunity, MemoryOptimizationRecommendation, MemoryOptimizationType,
};

impl PerformanceStats {
    /// Create performance statistics from profile events
    pub fn from_profile_events(events: &[ProfileEvent]) -> Self {
        let mut total_execution_time_ns = 0;
        let mut per_op_type_time_ns = HashMap::new();
        let mut per_op_instance_time_ns = HashMap::new();
        let mut memory_by_tensor_type = HashMap::new();
        let mut peak_memory_bytes = 0;

        // Process execution events
        for event in events {
            if let Some(duration) = event.duration {
                let duration_ns = duration.as_nanos() as u64;
                
                if event.event_type == ProfileEventType::OpExecution {
                    // Add to total execution time
                    total_execution_time_ns += duration_ns;
                    
                    // Add to per-operator-type time
                    let op_type = event.name.clone();
                    *per_op_type_time_ns.entry(op_type).or_insert(0) += duration_ns;
                    
                    // Add to per-operator-instance time if node_id is available
                    if let Some(node_id) = event.node_id {
                        per_op_instance_time_ns.insert(node_id, duration_ns);
                    }
                }
            }
            
            // Track peak memory usage
            if let Some(memory_usage) = event.memory_usage {
                peak_memory_bytes = peak_memory_bytes.max(memory_usage);
            }
        }
        
        // Build execution graph and find critical path
        let execution_graph = Self::build_execution_dag(events);
        let critical_path = Self::compute_critical_path(&execution_graph, total_execution_time_ns);
        
        Self {
            total_execution_time_ns,
            per_op_type_time_ns,
            per_op_instance_time_ns,
            critical_path,
            peak_memory_bytes,
            memory_by_tensor_type,
        }
    }
    
    /// Build a directed acyclic graph representing the execution flow
    fn build_execution_dag(events: &[ProfileEvent]) -> HashMap<NodeId, TimedNode> {
        let mut graph = HashMap::new();
        let mut op_end_times = HashMap::new();
        let mut tensor_producers = HashMap::new();
        let mut tensor_consumers = HashMap::new();
        
        // First pass: create nodes and record tensor producers/consumers
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && 
               event.node_id.is_some() && 
               event.duration.is_some() {
                
                let node_id = event.node_id.unwrap();
                let duration = event.duration.unwrap();
                
                // Create node if not exists
                let node = graph.entry(node_id).or_insert_with(|| TimedNode {
                    node_id,
                    execution_time: duration,
                    incoming_edges: Vec::new(),
                    outgoing_edges: Vec::new(),
                    earliest_start_time: None,
                    earliest_completion_time: None,
                    latest_start_time: None,
                    latest_completion_time: None,
                    slack: None,
                });
                
                // Record when this operator finished
                op_end_times.insert(node_id, event.start_time + duration);
                
                // Record tensor producer relationships if available
                if let Some(tensor_id) = event.tensor_id {
                    tensor_producers.insert(tensor_id, node_id);
                    tensor_consumers.entry(tensor_id).or_insert_with(Vec::new).push(node_id);
                }
            }
        }
        
        // Second pass: add edges based on tensor producer/consumer relationships
        for (tensor_id, producer) in tensor_producers.iter() {
            if let Some(consumers) = tensor_consumers.get(tensor_id) {
                for consumer in consumers {
                    if producer != consumer {
                        // Add edge from producer to consumer
                        if let Some(producer_node) = graph.get_mut(producer) {
                            producer_node.outgoing_edges.push((*consumer, EdgeType::DataDependency));
                        }
                        
                        if let Some(consumer_node) = graph.get_mut(consumer) {
                            consumer_node.incoming_edges.push((*producer, EdgeType::DataDependency));
                        }
                    }
                }
            }
        }
        
        // Third pass: add implicit control dependencies based on execution order
        // (This would connect nodes that don't have explicit data dependencies but executed sequentially)
        
        graph
    }
    
    /// Compute the critical path through the execution graph
    fn compute_critical_path(graph: &HashMap<NodeId, TimedNode>, _total_time_ns: u64) -> Vec<NodeId> {
        let mut critical_path = Vec::new();
        let mut completion_times = HashMap::new();
        
        // Find nodes with no incoming edges (start nodes)
        let mut start_nodes: Vec<NodeId> = graph.values()
            .filter(|node| node.incoming_edges.is_empty())
            .map(|node| node.node_id)
            .collect();
        
        // If no start nodes found (circular dependencies?), use arbitrary node
        if start_nodes.is_empty() && !graph.is_empty() {
            start_nodes.push(*graph.keys().next().unwrap());
        }
        
        // Compute earliest completion times for all nodes
        Self::compute_earliest_completion_times(graph, &start_nodes, &mut completion_times);
        
        // Find the node with the latest completion time (end of critical path)
        if let Some((&node_id, &time)) = completion_times.iter().max_by_key(|&(_, time)| time) {
            // Backtrack to find the critical path
            let mut current = node_id;
            critical_path.push(current);
            
            while let Some(node) = graph.get(&current) {
                // Find the predecessor with the latest completion time
                if let Some((&pred_id, _)) = node.incoming_edges.iter()
                    .filter_map(|(pred_id, _)| {
                        completion_times.get(pred_id).map(|time| (pred_id, time))
                    })
                    .max_by_key(|&(_, time)| time) {
                    current = pred_id;
                    critical_path.push(current);
                } else {
                    break;
                }
            }
            
            // Reverse to get the path from start to end
            critical_path.reverse();
        }
        
        critical_path
    }
    
    /// Compute earliest completion times for nodes in the execution graph
    fn compute_earliest_completion_times(
        graph: &HashMap<NodeId, TimedNode>, 
        start_nodes: &[NodeId],
        completion_times: &mut HashMap<NodeId, Duration>
    ) {
        let mut visited = HashSet::new();
        let mut queue = Vec::new();
        
        // Initialize with start nodes
        for &node_id in start_nodes {
            if let Some(node) = graph.get(&node_id) {
                let completion_time = node.execution_time;
                completion_times.insert(node_id, completion_time);
                queue.push(node_id);
            }
        }
        
        // Process nodes in topological order
        while let Some(node_id) = queue.pop() {
            if !visited.insert(node_id) {
                continue; // Already processed
            }
            
            if let Some(node) = graph.get(&node_id) {
                let node_completion_time = *completion_times.get(&node_id).unwrap_or(&Duration::from_secs(0));
                
                // Update successors
                for &(succ_id, _) in &node.outgoing_edges {
                    if let Some(succ_node) = graph.get(&succ_id) {
                        let new_completion_time = node_completion_time + succ_node.execution_time;
                        
                        // Update if this path results in a later completion time
                        let current_time = completion_times.get(&succ_id).unwrap_or(&Duration::from_secs(0));
                        if new_completion_time > *current_time {
                            completion_times.insert(succ_id, new_completion_time);
                        }
                        
                        queue.push(succ_id);
                    }
                }
            }
        }
    }
    
    /// Calculate impact scores for optimizing critical path nodes
    pub fn calculate_optimization_impact(&self, graph: &HashMap<NodeId, TimedNode>) -> HashMap<NodeId, OptimizationImpact> {
        let mut impact_scores = HashMap::new();
        let total_time = Duration::from_nanos(self.total_execution_time_ns);
        
        for &node_id in &self.critical_path {
            if let Some(node) = graph.get(&node_id) {
                let execution_time = node.execution_time;
                let percentage = execution_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;
                
                // Count dependent nodes
                let dependent_count = self.count_dependent_nodes(graph, node_id);
                
                // Calculate potential speedup (if this node were 50% faster)
                let time_saved = execution_time / 2;
                let new_total = total_time.checked_sub(time_saved).unwrap_or(Duration::from_secs(0));
                let speedup = total_time.as_secs_f64() / new_total.as_secs_f64();
                
                // Calculate impact score (percentage * dependent nodes)
                let impact = percentage * (1.0 + (dependent_count as f64 / 10.0));
                
                impact_scores.insert(node_id, OptimizationImpact {
                    node_id,
                    impact_score: impact,
                    percentage_of_total_time: percentage,
                    potential_speedup: speedup,
                    dependent_node_count: dependent_count,
                });
            }
        }
        
        impact_scores
    }
    
    /// Count the number of nodes that depend (directly or indirectly) on a given node
    fn count_dependent_nodes(&self, graph: &HashMap<NodeId, TimedNode>, node_id: NodeId) -> usize {
        let mut visited = HashSet::new();
        let mut queue = Vec::new();
        queue.push(node_id);
        
        // Don't count the node itself
        visited.insert(node_id);
        
        let mut dependent_count = 0;
        
        while let Some(current) = queue.pop() {
            if let Some(node) = graph.get(&current) {
                for &(succ_id, _) in &node.outgoing_edges {
                    if visited.insert(succ_id) {
                        dependent_count += 1;
                        queue.push(succ_id);
                    }
                }
            }
        }
        
        dependent_count
    }
    
    /// Get the total execution time in milliseconds
    pub fn total_execution_time_ms(&self) -> f64 {
        self.total_execution_time_ns as f64 / 1_000_000.0
    }
    
    /// Get operations sorted by execution time (descending)
    pub fn operations_by_time(&self) -> Vec<(String, u64)> {
        let mut ops: Vec<(String, u64)> = self.per_op_type_time_ns.iter()
            .map(|(name, time)| (name.clone(), *time))
            .collect();
        
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        ops
    }
    
    /// Get the percentage of total execution time for each operation type
    pub fn operation_time_percentage(&self) -> HashMap<String, f64> {
        let mut percentages = HashMap::new();
        let total = self.total_execution_time_ns as f64;
        
        for (op_type, time) in &self.per_op_type_time_ns {
            let percentage = (*time as f64 / total) * 100.0;
            percentages.insert(op_type.clone(), percentage);
        }
        
        percentages
    }
}

/// Find the bottleneck operations in a profile result
pub fn find_bottleneck_operations(profile_results: &ProfileResults) -> Vec<(NodeId, f64)> {
    let mut bottlenecks = Vec::new();
    let total_time = profile_results.performance.total_execution_time_ns as f64;
    
    // Convert node times to percentage of total execution time
    for (node_id, duration) in &profile_results.node_execution_times {
        let percentage = duration.as_nanos() as f64 / total_time * 100.0;
        bottlenecks.push((*node_id, percentage));
    }
    
    // Sort by percentage (descending)
    bottlenecks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    
    bottlenecks
}

/// Analyze memory efficiency from memory profiling results
pub fn analyze_memory_efficiency(mem_results: &MemoryProfileResults) -> MemoryEfficiencyMetrics {
    // Calculate memory utilization (active memory / peak memory)
    let active_memory = calculate_active_memory_usage(mem_results);
    let peak_memory = mem_results.peak_memory_bytes;
    let utilization = if peak_memory > 0 {
        (active_memory as f64 / peak_memory as f64) * 100.0
    } else {
        0.0
    };
    
    // Calculate optimization potential
    let optimization_potential = 100.0 - utilization;
    
    // Find tensor reuse opportunities
    let reuse_opportunities = find_tensor_reuse_opportunities(mem_results);
    
    // Calculate memory fragmentation
    let fragmentation = calculate_memory_fragmentation(mem_results);
    
    // Generate recommendations
    let recommendations = generate_memory_recommendations(mem_results, &reuse_opportunities);
    
    // Calculate overall efficiency score (0-100)
    let efficiency_score = calculate_efficiency_score(utilization, fragmentation);
    
    MemoryEfficiencyMetrics {
        utilization_percent: utilization,
        optimization_potential_percent: optimization_potential,
        reuse_opportunities,
        fragmentation_percent: fragmentation,
        efficiency_score,
        recommendations,
    }
}

/// Calculate active memory usage (memory currently in use by tensors)
fn calculate_active_memory_usage(mem_results: &MemoryProfileResults) -> usize {
    // Implementation stub
    // Sum the size of tensors that are actively used at any point
    mem_results.memory_by_tensor.values().sum()
}

/// Calculate memory fragmentation percentage
fn calculate_memory_fragmentation(_mem_results: &MemoryProfileResults) -> f64 {
    // Implementation stub for refactoring exercise
    0.0
}

/// Calculate overall memory efficiency score
fn calculate_efficiency_score(utilization: f64, fragmentation: f64) -> u32 {
    // Implementation stub
    // Higher utilization and lower fragmentation is better
    let score = utilization * (100.0 - fragmentation) / 100.0;
    score.min(100.0).max(0.0) as u32
}

/// Generate memory optimization recommendations
fn generate_memory_recommendations(
    _mem_results: &MemoryProfileResults,
    _reuse_opportunities: &[TensorReuseOpportunity],
) -> Vec<MemoryOptimizationRecommendation> {
    // Implementation stub for refactoring exercise
    Vec::new()
}

/// Find opportunities for tensor memory reuse
fn find_tensor_reuse_opportunities(_mem_results: &MemoryProfileResults) -> Vec<TensorReuseOpportunity> {
    // Implementation stub for refactoring exercise
    Vec::new()
}

/// Compute parallelism statistics from profile events
pub fn compute_parallelism_stats(events: &[ProfileEvent]) -> ParallelismStats {
    // Map of timestamp to number of active operations
    let mut active_ops = BTreeMap::new();
    
    // Record start and end times of all operation executions
    for event in events {
        if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
            let start_time = event.start_time;
            let end_time = start_time + event.duration.unwrap();
            
            // Increment active operation count at start time
            *active_ops.entry(start_time).or_insert(0) += 1;
            
            // Decrement active operation count at end time
            *active_ops.entry(end_time).or_insert(0) -= 1;
        }
    }
    
    // Calculate running count of active operations at each timestamp
    let mut current_count = 0;
    let mut max_parallel = 0;
    let mut histogram = HashMap::new();
    let mut time_at_parallelism = HashMap::new();
    let mut prev_time = None;
    let mut total_time = Duration::from_secs(0);
    
    for (time, delta) in active_ops {
        // Update time spent at current parallelism level
        if let Some(prev) = prev_time {
            let duration = time.duration_since(prev);
            *time_at_parallelism.entry(current_count).or_insert(Duration::from_secs(0)) += duration;
            total_time += duration;
        }
        
        // Update current parallelism count
        current_count += delta;
        *histogram.entry(current_count).or_insert(0) += 1;
        
        // Update maximum parallelism seen
        max_parallel = max_parallel.max(current_count);
        
        prev_time = Some(time);
    }
    
    // Calculate average parallelism
    let mut total_weighted_parallelism = 0.0;
    let mut parallelism_percentages = HashMap::new();
    
    for (count, time) in &time_at_parallelism {
        let percentage = if !total_time.is_zero() {
            time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        } else {
            0.0
        };
        
        parallelism_percentages.insert(*count, percentage);
        total_weighted_parallelism += *count as f64 * percentage;
    }
    
    let avg_parallel = total_weighted_parallelism / 100.0;
    
    ParallelismStats {
        max_parallel_ops: max_parallel,
        avg_parallel_ops: avg_parallel,
        parallelism_histogram: histogram,
        parallelism_percentages,
    }
}

/// Calculate tensor memory usage based on dimensions and data type
pub fn calculate_tensor_memory_usage(shape: &[usize], data_type: DataType) -> usize {
    let element_count = shape.iter().product::<usize>();
    let bytes_per_element = match data_type {
        DataType::Float32 => 4,
        DataType::Float64 => 8,
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 4,
        DataType::Int64 => 8,
        DataType::Uint8 => 1,
        DataType::Uint16 => 2,
        DataType::Uint32 => 4,
        DataType::Uint64 => 8,
        DataType::Bool => 1,
        DataType::String => 8, // Pointer size, actual string content is elsewhere
        DataType::Float16 => 2,
        DataType::BFloat16 => 2,
        DataType::Complex64 => 8,
        DataType::Complex128 => 16,
    };
    
    element_count * bytes_per_element
}

/// Estimate the memory requirements for common ONNX operations
pub fn estimate_operation_memory(
    op_type: &str, 
    input_shapes: &[Vec<usize>], 
    input_types: &[DataType],
    output_shapes: &[Vec<usize>],
    output_types: &[DataType],
) -> usize {
    let mut total_memory = 0;
    
    // Add input memory requirements
    for (shape, data_type) in input_shapes.iter().zip(input_types) {
        total_memory += calculate_tensor_memory_usage(shape, *data_type);
    }
    
    // Add output memory requirements
    for (shape, data_type) in output_shapes.iter().zip(output_types) {
        total_memory += calculate_tensor_memory_usage(shape, *data_type);
    }
    
    // Add operation-specific workspace requirements
    match op_type {
        "Conv" => {
            // Convolution often needs workspace memory for certain algorithms
            if input_shapes.len() > 0 && input_shapes[0].len() >= 4 {
                let batch_size = input_shapes[0][0];
                let channels = input_shapes[0][1];
                
                // Rough estimation for im2col workspace
                if channels > 4 {
                    let extra_workspace = batch_size * channels * 256 * 4; // Rough estimate
                    total_memory += extra_workspace;
                }
            }
        },
        "BatchNormalization" => {
            // Add space for intermediate stats
            if input_shapes.len() > 0 && input_shapes[0].len() >= 4 {
                let channels = input_shapes[0][1];
                total_memory += channels * 16; // Mean, variance, etc.
            }
        },
        "MatMul" | "Gemm" => {
            // Some BLAS implementations use workspace memory
            if input_shapes.len() >= 2 {
                // Rough estimate for potential workspace
                let m = input_shapes[0][0];
                let n = input_shapes[1].len() > 1 ? input_shapes[1][1] : 1;
                total_memory += m * n * 4; // Extra workspace for blocked algorithms
            }
        },
        // Handle other operations as needed
        _ => {}
    }
    
    total_memory
}