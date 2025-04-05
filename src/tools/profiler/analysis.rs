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
fn calculate_memory_fragmentation(mem_results: &MemoryProfileResults) -> f64 {
    // Analyze memory allocation and deallocation events to calculate fragmentation
    let memory_events = &mem_results.memory_events;
    
    // Track active allocations and free blocks
    let mut active_allocations = HashMap::new();
    let mut memory_blocks = Vec::new();
    
    // Process memory events chronologically
    let mut sorted_events = memory_events.clone();
    sorted_events.sort_by_key(|e| e.timestamp);
    
    for event in sorted_events {
        match event.event_type {
            MemoryEventType::Allocation => {
                // Record this allocation
                active_allocations.insert(event.address, (event.size_bytes, event.timestamp));
                
                // Add to memory blocks
                memory_blocks.push((event.address, event.address + event.size_bytes, true));
            },
            MemoryEventType::Deallocation => {
                // Mark block as free
                if let Some((size, _)) = active_allocations.remove(&event.address) {
                    // Find and update the block status
                    if let Some(index) = memory_blocks.iter().position(|&(addr, _, _)| addr == event.address) {
                        memory_blocks[index] = (event.address, event.address + size, false);
                    }
                }
            },
            // For other event types, continue
            _ => continue,
        }
    }
    
    // Calculate fragmentation metrics
    let (fragmentation, _) = calculate_memory_fragmentation_metrics(&memory_blocks);
    
    fragmentation
}

/// Calculate memory fragmentation metrics from a list of memory blocks
fn calculate_memory_fragmentation_metrics(blocks: &[(usize, usize, bool)]) -> (f64, usize) {
    if blocks.is_empty() {
        return (0.0, 0);
    }
    
    // Sort blocks by address
    let mut sorted_blocks = blocks.to_vec();
    sorted_blocks.sort_by_key(|&(addr, _, _)| addr);
    
    // Calculate total memory range and used memory
    let start_addr = sorted_blocks.first().unwrap().0;
    let end_addr = sorted_blocks.last().unwrap().1;
    let memory_range = end_addr - start_addr;
    
    // Calculate used memory and free blocks
    let mut used_memory = 0;
    let mut free_blocks = 0;
    let mut prev_end = start_addr;
    
    for &(start, end, is_used) in &sorted_blocks {
        // Count gaps (external fragmentation)
        if start > prev_end {
            free_blocks += 1;
        }
        
        // Add used memory
        if is_used {
            used_memory += end - start;
        }
        
        prev_end = end;
    }
    
    // Calculate fragmentation percentage
    // Fragmentation is higher when:
    // 1. The ratio of used memory to total range is lower
    // 2. The number of free blocks is higher
    
    let memory_utilization = used_memory as f64 / memory_range as f64;
    let normalized_free_blocks = (free_blocks as f64 / blocks.len() as f64) * 100.0;
    
    // Weigh both factors to calculate fragmentation
    let fragmentation = (1.0 - memory_utilization) * 70.0 + normalized_free_blocks * 30.0;
    
    // Clamp to 0-100 range
    let fragmentation = fragmentation.min(100.0).max(0.0);
    
    (fragmentation, free_blocks)
}

/// Calculate overall memory efficiency score
fn calculate_efficiency_score(utilization: f64, fragmentation: f64) -> u32 {
    // Implementation stub
    // Higher utilization and lower fragmentation is better
    let score = utilization * (100.0 - fragmentation) / 100.0;
    score.min(100.0).max(0.0) as u32
}

/// Generate memory optimization recommendations based on profiling results
fn generate_memory_recommendations(
    mem_results: &MemoryProfileResults,
    reuse_opportunities: &[TensorReuseOpportunity],
) -> Vec<MemoryOptimizationRecommendation> {
    let mut recommendations = Vec::new();
    
    // 1. Process workspace memory usage recommendations
    analyze_workspace_memory_usage(mem_results, &mut recommendations);
    
    // 2. Process tensor reuse opportunities
    analyze_tensor_reuse_opportunities(reuse_opportunities, &mut recommendations);
    
    // 3. Identify in-place operation opportunities
    identify_inplace_operation_opportunities(mem_results, &mut recommendations);
    
    // 4. Identify operations with excessive memory usage
    identify_excessive_memory_users(mem_results, &mut recommendations);
    
    // 5. Check for memory pool optimization opportunities
    analyze_memory_pool_opportunities(mem_results, &mut recommendations);
    
    // Sort recommendations by potential savings
    recommendations.sort_by(|a, b| b.potential_savings_bytes.cmp(&a.potential_savings_bytes));
    
    recommendations
}

/// Analyzes workspace memory usage and generates recommendations
fn analyze_workspace_memory_usage(
    mem_results: &MemoryProfileResults,
    recommendations: &mut Vec<MemoryOptimizationRecommendation>
) {
    let workspace_usage = &mem_results.workspace_usage;
    let peak_bytes = workspace_usage.peak_bytes;
    
    // Find operators with excessive workspace usage
    // Consider an operator excessive if it uses more than 20% of peak workspace memory
    let excessive_threshold = peak_bytes / 5;
    
    // Collect operators exceeding the threshold
    let mut excessive_operators = Vec::new();
    for (op_type, usage) in &workspace_usage.usage_per_operator {
        if *usage >= excessive_threshold {
            excessive_operators.push((op_type.clone(), *usage));
        }
    }
    
    // Sort by usage (descending)
    excessive_operators.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Generate recommendations for operators with high workspace usage
    for (op_type, usage) in excessive_operators {
        let affected_nodes = find_nodes_by_op_type(mem_results, &op_type);
        
        // Find the specific allocation events for this operator
        let affected_tensors = workspace_usage.allocation_events
            .iter()
            .filter(|event| event.op_type == op_type)
            .filter_map(|event| event.node_id)
            .collect::<Vec<_>>();
        
        // Calculate potential savings (estimate 30-50% reduction with algorithm optimization)
        let potential_savings = (usage as f64 * 0.3) as usize;
        
        if usage >= excessive_threshold {
            // For convolution operators, suggest workspace algorithm optimizations
            if op_type.contains("Conv") {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::AlgorithmicOptimization,
                    description: format!(
                        "Operator '{}' uses excessive workspace memory ({}MB). Consider using a more memory-efficient convolution algorithm or reducing workspace size.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: potential_savings,
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: affected_tensors.clone(),
                });
            } 
            // For matrix multiplication operators, suggest tiling or blocking
            else if op_type.contains("Gemm") || op_type.contains("MatMul") {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::AlgorithmicOptimization,
                    description: format!(
                        "Operator '{}' uses excessive workspace memory ({}MB). Consider using tiling or blocking techniques to reduce workspace requirements.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: potential_savings,
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: affected_tensors.clone(),
                });
            }
            // For pooling operators, suggest reducing buffer sizes
            else if op_type.contains("Pool") {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::MemoryManagement,
                    description: format!(
                        "Operator '{}' uses excessive workspace memory ({}MB). Consider implementing a line buffer approach to reduce memory requirements.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: potential_savings,
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: affected_tensors.clone(),
                });
            }
            // Generic recommendation for other operators
            else {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::CustomMemoryPool,
                    description: format!(
                        "Operator '{}' uses significant workspace memory ({}MB). Consider customizing workspace allocation strategy for this operator.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: potential_savings,
                    affected_nodes, 
                    affected_tensors,
                });
            }
        }
    }
    
    // Check for overall workspace memory patterns
    let allocation_events = &workspace_usage.allocation_events;
    
    // If many short-lived workspace allocations, suggest a custom memory pool
    if allocation_events.len() > 10 {
        let short_lived_allocations = allocation_events.iter()
            .filter(|event| {
                event.deallocation_time.map_or(false, |dealloc_time| {
                    dealloc_time - event.allocation_time < 1_000_000_000 // Less than 1 second
                })
            })
            .count();
        
        if short_lived_allocations > allocation_events.len() / 2 {
            recommendations.push(MemoryOptimizationRecommendation {
                recommendation_type: MemoryOptimizationType::CustomMemoryPool,
                description: format!(
                    "Found {} short-lived workspace allocations. Consider implementing a specialized memory pool to reduce allocation overhead.",
                    short_lived_allocations
                ),
                potential_savings_bytes: peak_bytes / 5, // Estimate 20% savings
                affected_nodes: Vec::new(),
                affected_tensors: Vec::new(),
            });
        }
    }
}

/// Analyzes tensor reuse opportunities and generates recommendations
fn analyze_tensor_reuse_opportunities(
    reuse_opportunities: &[TensorReuseOpportunity],
    recommendations: &mut Vec<MemoryOptimizationRecommendation>
) {
    // Group similar opportunities to avoid too many recommendations
    let mut grouped_opportunities = Vec::new();
    let mut current_group = Vec::new();
    let mut current_savings = 0;
    
    // Only process the top opportunities (limit to first 10)
    for opportunity in reuse_opportunities.iter().take(10) {
        if opportunity.confidence > 70 {
            current_group.push(opportunity);
            current_savings += opportunity.potential_savings_bytes;
            
            // If this group is large enough, add a recommendation
            if current_group.len() >= 3 || current_savings > 1024 * 1024 {
                grouped_opportunities.push((current_group.clone(), current_savings));
                current_group = Vec::new();
                current_savings = 0;
            }
        }
    }
    
    // Add any remaining group
    if !current_group.is_empty() {
        grouped_opportunities.push((current_group, current_savings));
    }
    
    // Generate recommendations for each group
    for (group, savings) in grouped_opportunities {
        let affected_tensors = group.iter()
            .flat_map(|opp| vec![opp.tensor1_id, opp.tensor2_id])
            .collect::<Vec<_>>();
        
        recommendations.push(MemoryOptimizationRecommendation {
            recommendation_type: MemoryOptimizationType::TensorReuse,
            description: format!(
                "Found {} tensor reuse opportunities with potential savings of {}MB. Consider implementing memory reuse for these tensors.",
                group.len(), savings / (1024 * 1024)
            ),
            potential_savings_bytes: savings,
            affected_nodes: Vec::new(),
            affected_tensors,
        });
    }
}

/// Identifies potential in-place operation opportunities
fn identify_inplace_operation_opportunities(
    mem_results: &MemoryProfileResults,
    recommendations: &mut Vec<MemoryOptimizationRecommendation>
) {
    // Look for patterns that can be optimized with in-place operations
    // For example, operations like ReLU, Sigmoid, or element-wise operations
    
    // In-place candidates are operations with matching input and output shapes
    let inplace_candidates = mem_results.memory_events.iter()
        .filter(|event| {
            // In-place candidates typically have specific metadata
            event.metadata.get("op_type").map_or(false, |op_type| {
                op_type == "Relu" || op_type == "Sigmoid" || op_type == "Tanh" || 
                op_type.contains("Add") || op_type.contains("Mul") || op_type.contains("Div")
            })
        })
        .filter_map(|event| event.node_id)
        .collect::<Vec<_>>();
    
    // Group by node ID to avoid duplicates
    let unique_candidates: Vec<_> = inplace_candidates.into_iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    if !unique_candidates.is_empty() {
        // Estimate memory savings (pessimistically: 5% of total memory)
        let potential_savings = mem_results.peak_memory_bytes / 20;
        
        recommendations.push(MemoryOptimizationRecommendation {
            recommendation_type: MemoryOptimizationType::InPlaceOperation,
            description: format!(
                "Found {} unary or element-wise operations that could be performed in-place to reduce memory usage.",
                unique_candidates.len()
            ),
            potential_savings_bytes: potential_savings,
            affected_nodes: unique_candidates,
            affected_tensors: Vec::new(),
        });
    }
}

/// Identifies operators with excessive memory usage
fn identify_excessive_memory_users(
    mem_results: &MemoryProfileResults,
    recommendations: &mut Vec<MemoryOptimizationRecommendation>
) {
    // Find the operators consuming the most memory
    let mut memory_by_op_type = HashMap::new();
    
    // Group memory usage by operator type
    for (node_id, memory_usage) in &mem_results.memory_by_operator {
        // Find the operator type for this node ID
        // This is simplified - in a real implementation you would look up the node type
        let op_type = mem_results.memory_events.iter()
            .find(|event| event.node_id == Some(*node_id))
            .and_then(|event| event.metadata.get("op_type").cloned())
            .unwrap_or_else(|| "Unknown".to_string());
        
        *memory_by_op_type.entry(op_type).or_insert(0) += memory_usage;
    }
    
    // Sort operators by memory usage
    let mut operators_by_usage: Vec<_> = memory_by_op_type.into_iter().collect();
    operators_by_usage.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Consider the top 3 memory users
    for (op_type, usage) in operators_by_usage.iter().take(3) {
        // Only consider if usage is significant (> 10% of peak)
        if *usage > mem_results.peak_memory_bytes / 10 {
            let affected_nodes = find_nodes_by_op_type(mem_results, op_type);
            
            // Generate operator-specific recommendations
            if op_type.contains("Conv") {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::PrecisionReduction,
                    description: format!(
                        "Operator '{}' consumes significant memory ({}MB). Consider using mixed precision or operator-specific optimizations.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: usage / 2, // Estimate 50% savings
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: Vec::new(),
                });
            } else if op_type.contains("MatMul") || op_type.contains("Gemm") {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::OperationFusion,
                    description: format!(
                        "Operator '{}' consumes significant memory ({}MB). Consider fusing with adjacent operations.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: usage / 3, // Estimate 33% savings
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: Vec::new(),
                });
            } else {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::CustomMemoryPool,
                    description: format!(
                        "Operator '{}' consumes significant memory ({}MB). Review allocation patterns and consider optimization.",
                        op_type, usage / (1024 * 1024)
                    ),
                    potential_savings_bytes: usage / 4, // Estimate 25% savings
                    affected_nodes: affected_nodes.clone(),
                    affected_tensors: Vec::new(),
                });
            }
        }
    }
}

/// Analyzes memory pool opportunities
fn analyze_memory_pool_opportunities(
    mem_results: &MemoryProfileResults,
    recommendations: &mut Vec<MemoryOptimizationRecommendation>
) {
    // Count allocations and deallocations to detect inefficient patterns
    let allocation_count = mem_results.memory_events.iter()
        .filter(|event| event.event_type == MemoryEventType::Allocation)
        .count();
    
    let deallocation_count = mem_results.memory_events.iter()
        .filter(|event| event.event_type == MemoryEventType::Deallocation)
        .count();
    
    // If we have many small allocations, suggest a memory pool
    if allocation_count > 100 && allocation_count == deallocation_count {
        // Count small allocations (< 4KB)
        let small_allocation_count = mem_results.memory_events.iter()
            .filter(|event| {
                event.event_type == MemoryEventType::Allocation && event.size_bytes < 4 * 1024
            })
            .count();
        
        if small_allocation_count > allocation_count / 2 {
            recommendations.push(MemoryOptimizationRecommendation {
                recommendation_type: MemoryOptimizationType::CustomMemoryPool,
                description: format!(
                    "Found {} small memory allocations (< 4KB). Consider implementing a small allocation pool to reduce fragmentation and allocation overhead.",
                    small_allocation_count
                ),
                potential_savings_bytes: mem_results.peak_memory_bytes / 10, // Estimate 10% savings
                affected_nodes: Vec::new(),
                affected_tensors: Vec::new(),
            });
        }
    }
}

/// Helper function to find nodes by operator type
fn find_nodes_by_op_type(mem_results: &MemoryProfileResults, op_type: &str) -> Vec<NodeId> {
    mem_results.memory_events.iter()
        .filter(|event| event.metadata.get("op_type").map_or(false, |t| t == op_type))
        .filter_map(|event| event.node_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}

/// Find opportunities for tensor memory reuse by analyzing tensor lifetimes
fn find_tensor_reuse_opportunities(mem_results: &MemoryProfileResults) -> Vec<TensorReuseOpportunity> {
    let mut opportunities = Vec::new();
    let tensor_lifetimes = &mem_results.tensor_lifetimes;
    
    // Create a list of tensor lifetimes for analysis
    let lifetimes: Vec<_> = tensor_lifetimes.values().collect();
    
    // Compare each pair of tensors to check for potential reuse
    for i in 0..lifetimes.len() {
        for j in (i+1)..lifetimes.len() {
            let tensor1 = lifetimes[i];
            let tensor2 = lifetimes[j];
            
            // If tensors have the same data type and similar size, they are candidates for reuse
            if tensor1.data_type == tensor2.data_type {
                // Check if lifetimes don't overlap (indicating potential for reuse)
                let non_overlapping = match (tensor1.deallocation_time, tensor2.allocation_time) {
                    (Some(t1_dealloc), _) if t1_dealloc <= tensor2.allocation_time => true,
                    (_, Some(t2_dealloc)) if t2_dealloc <= tensor1.allocation_time => true,
                    _ => false,
                };
                
                if non_overlapping {
                    // Calculate similarity in sizes (higher confidence for similar sizes)
                    let size_ratio = if tensor1.size_bytes >= tensor2.size_bytes {
                        tensor2.size_bytes as f64 / tensor1.size_bytes as f64
                    } else {
                        tensor1.size_bytes as f64 / tensor2.size_bytes as f64
                    };
                    
                    // Only consider reasonably similar sizes (at least 50% similar)
                    if size_ratio >= 0.5 {
                        // Calculate potential savings
                        let savings = tensor1.size_bytes.min(tensor2.size_bytes);
                        
                        // Calculate confidence based on size similarity
                        let confidence = (size_ratio * 100.0) as u32;
                        
                        opportunities.push(TensorReuseOpportunity {
                            tensor1_id: tensor1.tensor_id,
                            tensor2_id: tensor2.tensor_id,
                            potential_savings_bytes: savings,
                            confidence,
                        });
                    }
                }
            }
        }
    }
    
    // Sort by potential savings (descending)
    opportunities.sort_by(|a, b| b.potential_savings_bytes.cmp(&a.potential_savings_bytes));
    
    opportunities
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