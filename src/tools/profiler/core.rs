// Core profiler implementation module

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::{Result, anyhow};

use crate::{
    execution::{ExecutionEngine, context::ExecutionContext},
    model::{NodeId, TensorId},
    ops::tensor::{DataType, Tensor},
    memory::{allocator::MemoryAllocator, workspace::MemoryWorkspace},
    error,
};

use super::types::{
    ProfileEventType, ProfileEvent, ProfileResults, 
    PerformanceStats, MemoryProfileResults, MemoryEvent, MemoryEventType,
    TensorLifetime, ParallelismStats, TensorStats, ModelInfo, GraphInfo,
};

/// Thread-safe basic profiling event recorder
#[derive(Clone)]
pub struct EventCollector {
    /// Events recorded by the profiler
    events: Arc<Mutex<Vec<ProfileEvent>>>,
    /// Unique event ID counter
    next_id: Arc<AtomicUsize>,
    /// Whether profiling is enabled
    enabled: bool,
}

impl EventCollector {
    /// Create a new profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(AtomicUsize::new(0)),
            enabled,
        }
    }

    /// Start a new profiling event
    pub fn start_event(
        &self,
        name: &str,
        event_type: ProfileEventType,
        node_id: Option<NodeId>,
        tensor_id: Option<TensorId>,
        parent_id: Option<usize>,
    ) -> Option<usize> {
        if !self.enabled {
            return None;
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let event = ProfileEvent {
            id,
            name: name.to_string(),
            start_time: Instant::now(),
            duration: None,
            event_type,
            node_id,
            tensor_id,
            metadata: HashMap::new(),
            parent_id,
            memory_usage: None,
        };

        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }

        Some(id)
    }

    /// End a profiling event by ID
    pub fn end_event(&self, id: usize, memory_usage: Option<usize>) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();
        if let Ok(mut events) = self.events.lock() {
            for event in events.iter_mut() {
                if event.id == id {
                    event.duration = Some(now.duration_since(event.start_time));
                    event.memory_usage = memory_usage;
                    break;
                }
            }
        }
    }

    /// Add metadata to an event
    pub fn add_metadata(&self, id: usize, key: &str, value: &str) {
        if !self.enabled {
            return;
        }

        if let Ok(mut events) = self.events.lock() {
            for event in events.iter_mut() {
                if event.id == id {
                    event.metadata.insert(key.to_string(), value.to_string());
                    break;
                }
            }
        }
    }

    /// Get all recorded events
    pub fn events(&self) -> Vec<ProfileEvent> {
        if let Ok(events) = self.events.lock() {
            events.clone()
        } else {
            Vec::new()
        }
    }

    /// Clear all recorded events
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }

    /// Record a complete operation with timing
    pub fn record_operation<F, T>(
        &self,
        name: &str,
        event_type: ProfileEventType,
        node_id: Option<NodeId>,
        tensor_id: Option<TensorId>,
        parent_id: Option<usize>,
        operation: F,
    ) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return operation();
        }

        let event_id = self.start_event(name, event_type, node_id, tensor_id, parent_id);
        let result = operation();
        if let Some(id) = event_id {
            self.end_event(id, None);
        }
        result
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Comprehensive profiler for detailed performance analysis
pub struct ComprehensiveProfiler {
    /// Internal event collector for capturing basic profile events
    event_collector: Arc<EventCollector>,
    /// Memory events collected during profiling
    memory_events: Vec<MemoryEvent>,
    /// Container for shared memory events (for callbacks)
    memory_events_container: Option<Arc<Mutex<Vec<MemoryEvent>>>>,
    /// Start time of the profiling session
    start_time: Instant,
    /// Whether to track memory usage
    track_memory: bool,
    /// Whether to track parallelism
    track_parallelism: bool,
    /// Whether to generate a flamegraph
    generate_flamegraph: bool,
    /// Maximum depth for call stack tracking
    max_stack_depth: usize,
    /// Peak memory usage in bytes
    peak_memory_bytes: AtomicUsize,
    /// Current memory usage in bytes
    current_memory_bytes: AtomicUsize,
    /// Memory tracking by data type
    memory_by_type: Arc<Mutex<HashMap<DataType, usize>>>,
    /// Memory tracking by tensor
    memory_by_tensor: Arc<Mutex<HashMap<TensorId, usize>>>,
    /// Memory tracking by operator
    memory_by_operator: Arc<Mutex<HashMap<NodeId, usize>>>,
}

impl ComprehensiveProfiler {
    /// Create a new comprehensive profiler
    pub fn new(track_memory: bool, track_parallelism: bool) -> Self {
        Self {
            event_collector: Arc::new(EventCollector::new(true)),
            memory_events: Vec::new(),
            memory_events_container: None,
            start_time: Instant::now(),
            track_memory,
            track_parallelism,
            generate_flamegraph: false,
            max_stack_depth: 32,
            peak_memory_bytes: AtomicUsize::new(0),
            current_memory_bytes: AtomicUsize::new(0),
            memory_by_type: Arc::new(Mutex::new(HashMap::new())),
            memory_by_tensor: Arc::new(Mutex::new(HashMap::new())),
            memory_by_operator: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Setup memory tracking hooks to intercept memory allocations and deallocations
    fn setup_memory_tracking(&mut self, engine: &mut ExecutionEngine) -> Result<()> {
        // Create a shared container for memory events
        let memory_events = Arc::new(Mutex::new(Vec::new()));
        self.memory_events_container = Some(memory_events.clone());
        
        // Create weak references to the tracking containers
        let memory_by_type = Arc::downgrade(&self.memory_by_type);
        let memory_by_tensor = Arc::downgrade(&self.memory_by_tensor);
        let memory_by_operator = Arc::downgrade(&self.memory_by_operator);
        let peak_memory = &self.peak_memory_bytes;
        let current_memory = &self.current_memory_bytes;
        let start_time = self.start_time;
        
        // Set up allocation tracking
        engine.set_memory_allocation_callback(Box::new(move |size, tensor_id, node_id, addr, allocator_id, metadata| {
            // Update current memory usage
            let current = current_memory.fetch_add(size, Ordering::Relaxed) + size;
            
            // Update peak memory if needed
            let mut peak = peak_memory.load(Ordering::Relaxed);
            while current > peak {
                match peak_memory.compare_exchange_weak(
                    peak, 
                    current,
                    Ordering::SeqCst,
                    Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
            
            // Update memory by data type if available
            if let Some(data_type) = metadata.get("data_type").and_then(|s| s.parse().ok()) {
                if let Some(memory_by_type) = memory_by_type.upgrade() {
                    if let Ok(mut map) = memory_by_type.lock() {
                        *map.entry(data_type).or_insert(0) += size;
                    }
                }
            }
            
            // Update memory by tensor if applicable
            if let Some(tensor_id) = tensor_id {
                if let Some(memory_by_tensor) = memory_by_tensor.upgrade() {
                    if let Ok(mut map) = memory_by_tensor.lock() {
                        *map.entry(tensor_id).or_insert(0) += size;
                    }
                }
            }
            
            // Update memory by operator if applicable
            if let Some(node_id) = node_id {
                if let Some(memory_by_operator) = memory_by_operator.upgrade() {
                    if let Ok(mut map) = memory_by_operator.lock() {
                        *map.entry(node_id).or_insert(0) += size;
                    }
                }
            }
            
            // Record the event
            let event = MemoryEvent {
                event_type: if metadata.get("memory_type").map_or(false, |t| t == "workspace") {
                    MemoryEventType::WorkspaceAllocation
                } else {
                    MemoryEventType::Allocation
                },
                timestamp: start_time.elapsed().as_nanos() as u64,
                size_bytes: size,
                tensor_id,
                node_id,
                address: addr,
                allocator_id: allocator_id.to_string(),
                metadata: metadata.clone(),
            };
            
            // Add event to the memory events container
            if let Ok(mut events) = memory_events.lock() {
                events.push(event);
            }
        }));
        
        // Set up deallocation tracking
        engine.set_memory_deallocation_callback(Box::new(move |size, tensor_id, node_id, addr, allocator_id, metadata| {
            // Update current memory usage
            let current = current_memory.fetch_sub(size, Ordering::Relaxed) - size;
            
            // Record the event
            let event = MemoryEvent {
                event_type: if metadata.get("memory_type").map_or(false, |t| t == "workspace") {
                    MemoryEventType::WorkspaceAllocation // Reuse same type but with metadata indicating deallocation
                } else {
                    MemoryEventType::Deallocation
                },
                timestamp: start_time.elapsed().as_nanos() as u64,
                size_bytes: size,
                tensor_id,
                node_id,
                address: addr,
                allocator_id: allocator_id.to_string(),
                metadata: {
                    let mut meta = metadata.clone();
                    meta.insert("operation".to_string(), "deallocate".to_string());
                    meta
                },
            };
            
            // Add event to the memory events container
            if let Ok(mut events) = memory_events.lock() {
                events.push(event);
            }
        }));
        
        // Set up reuse tracking, if supported by the engine
        if let Some(callback) = engine.set_memory_reuse_callback(Box::new(move |size, old_tensor_id, new_tensor_id, node_id, addr, allocator_id| {
            // Record the event
            let event = MemoryEvent {
                event_type: MemoryEventType::Reuse,
                timestamp: start_time.elapsed().as_nanos() as u64,
                size_bytes: size,
                tensor_id: new_tensor_id,
                node_id,
                address: addr,
                allocator_id: allocator_id.to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("old_tensor_id".to_string(), format!("{:?}", old_tensor_id));
                    meta.insert("new_tensor_id".to_string(), format!("{:?}", new_tensor_id));
                    meta
                },
            };
            
            // Add event to the memory events container
            if let Ok(mut events) = memory_events.lock() {
                events.push(event);
            }
        })) {
            // Store callback if needed
            drop(callback);
        }
        
        // Hook into workspace allocator events specifically
        engine.set_workspace_allocation_callback(Box::new(move |size, node_id, op_type| {
            // Record the event with special workspace metadata
            let event = MemoryEvent {
                event_type: MemoryEventType::WorkspaceAllocation,
                timestamp: start_time.elapsed().as_nanos() as u64,
                size_bytes: size,
                tensor_id: None,
                node_id,
                address: 0, // We don't track the actual address for workspace allocations
                allocator_id: "workspace".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("memory_type".to_string(), "workspace".to_string());
                    meta.insert("operation".to_string(), "allocate".to_string());
                    meta.insert("size_bytes".to_string(), size.to_string());
                    meta.insert("op_type".to_string(), op_type.clone());
                    meta
                },
            };
            
            // Add event to the memory events container
            if let Ok(mut events) = memory_events.lock() {
                events.push(event);
            }
        }));
        
        Ok(())
    }

    /// Extract execution times for each node from profile events
    fn extract_node_execution_times(&self, events: &[ProfileEvent]) -> HashMap<NodeId, Duration> {
        let mut node_times = HashMap::new();
        
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && 
                event.node_id.is_some() && event.duration.is_some() {
                node_times.insert(event.node_id.unwrap(), event.duration.unwrap());
            }
        }
        
        node_times
    }
    
    /// Extract execution times by operator type
    fn extract_op_type_execution_times(&self, events: &[ProfileEvent]) -> HashMap<String, Duration> {
        let mut op_times = HashMap::new();
        
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && 
                event.duration.is_some() {
                let op_type = event.name.clone();
                let duration = event.duration.unwrap();
                
                op_times.entry(op_type)
                    .and_modify(|total: &mut Duration| *total += duration)
                    .or_insert(duration);
            }
        }
        
        op_times
    }
    
    /// Collect statistics about tensors
    fn collect_tensor_stats(
        &self, 
        engine: &ExecutionEngine,
        outputs: &HashMap<String, Tensor>
    ) -> HashMap<TensorId, TensorStats> {
        // Stub implementation for refactoring exercise
        HashMap::new()
    }
    
    /// Extract model information
    fn extract_model_info(&self, engine: &ExecutionEngine) -> ModelInfo {
        // Stub implementation for refactoring exercise
        ModelInfo {
            name: "unknown".to_string(),
            op_count: 0,
            input_count: 0,
            output_count: 0,
            graph_info: GraphInfo {
                max_depth: 0,
                avg_depth: 0.0,
                max_width: 0,
                avg_width: 0.0,
                branch_count: 0,
            },
            op_type_counts: HashMap::new(),
        }
    }
    
    /// Calculate memory usage by data type
    fn calculate_memory_by_type(&self) -> HashMap<DataType, usize> {
        if let Ok(memory_by_type) = self.memory_by_type.lock() {
            memory_by_type.clone()
        } else {
            HashMap::new()
        }
    }
    
    /// Calculate memory usage by tensor
    fn calculate_memory_by_tensor(&self) -> HashMap<TensorId, usize> {
        if let Ok(memory_by_tensor) = self.memory_by_tensor.lock() {
            memory_by_tensor.clone()
        } else {
            HashMap::new()
        }
    }
    
    /// Calculate memory usage by operator
    fn calculate_memory_by_operator(&self) -> HashMap<NodeId, usize> {
        if let Ok(memory_by_operator) = self.memory_by_operator.lock() {
            memory_by_operator.clone()
        } else {
            HashMap::new()
        }
    }
    
    /// Calculate tensor lifetimes from memory events
    fn calculate_tensor_lifetimes(&self) -> HashMap<TensorId, TensorLifetime> {
        let mut tensor_lifetimes = HashMap::new();
        let memory_events = &self.memory_events;
        
        // Temporary storage for tracking allocation times and tensor metadata
        let mut allocation_times = HashMap::new();
        let mut tensor_metadata = HashMap::new();
        
        // First pass: collect all allocation events
        for event in memory_events {
            // Skip workspace allocations - they're tracked separately
            if event.event_type == MemoryEventType::WorkspaceAllocation {
                continue;
            }
            
            if let Some(tensor_id) = event.tensor_id {
                match event.event_type {
                    MemoryEventType::Allocation => {
                        // Record allocation time
                        allocation_times.insert(tensor_id, event.timestamp);
                        
                        // Store tensor metadata
                        let data_type = event.metadata.get("data_type")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(DataType::Float32); // Default if not specified
                        
                        let producer_node = event.node_id;
                        
                        tensor_metadata.insert(tensor_id, (event.size_bytes, data_type, producer_node));
                    },
                    MemoryEventType::Deallocation => {
                        // If we have an allocation time for this tensor
                        if let Some(allocation_time) = allocation_times.get(&tensor_id) {
                            // Get tensor metadata
                            if let Some((size_bytes, data_type, producer_node)) = tensor_metadata.get(&tensor_id).cloned() {
                                // Create tensor lifetime entry
                                let lifetime = TensorLifetime {
                                    tensor_id,
                                    allocation_time: *allocation_time,
                                    deallocation_time: Some(event.timestamp),
                                    size_bytes,
                                    data_type,
                                    producer_node,
                                    consumer_nodes: Vec::new(), // Will be populated in second pass
                                };
                                
                                tensor_lifetimes.insert(tensor_id, lifetime);
                            }
                        }
                    },
                    _ => {} // Skip other event types
                }
            }
        }
        
        // Second pass: identify tensor consumers
        for event in memory_events {
            if event.event_type == ProfileEventType::OpExecution && event.node_id.is_some() {
                let node_id = event.node_id.unwrap();
                
                // Check if this operation used any tensors as inputs
                for metadata_key in event.metadata.keys() {
                    if metadata_key.starts_with("input_tensor_") {
                        if let Some(tensor_id_str) = event.metadata.get(metadata_key) {
                            if let Ok(tensor_id) = tensor_id_str.parse::<TensorId>() {
                                // Add this node as a consumer of the tensor
                                if let Some(lifetime) = tensor_lifetimes.get_mut(&tensor_id) {
                                    if !lifetime.consumer_nodes.contains(&node_id) {
                                        lifetime.consumer_nodes.push(node_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add missing deallocations for tensors that were never explicitly deallocated
        for (tensor_id, allocation_time) in allocation_times {
            if !tensor_lifetimes.contains_key(&tensor_id) {
                if let Some((size_bytes, data_type, producer_node)) = tensor_metadata.get(&tensor_id).cloned() {
                    // Create tensor lifetime with no deallocation time
                    let lifetime = TensorLifetime {
                        tensor_id,
                        allocation_time,
                        deallocation_time: None, // No explicit deallocation
                        size_bytes,
                        data_type,
                        producer_node,
                        consumer_nodes: Vec::new(),
                    };
                    
                    tensor_lifetimes.insert(tensor_id, lifetime);
                }
            }
        }
        
        tensor_lifetimes
    }
    
    /// Calculate workspace memory usage by analyzing profile events
    fn calculate_workspace_usage(&self, events: &[ProfileEvent]) -> super::types::WorkspaceUsage {
        // Track workspace allocations and deallocations
        let mut allocation_events = Vec::new();
        let mut active_workspaces = HashMap::new();
        let mut usage_per_operator = HashMap::new();
        let mut current_bytes = 0;
        let mut peak_bytes = 0;
        
        // Find workspace allocation events in the profile timeline
        for event in events.iter() {
            // Look for workspace memory events (they have specific event types or metadata)
            if event.event_type == ProfileEventType::MemoryAllocation {
                // Check if this is a workspace allocation by looking at metadata
                if let Some(mem_type) = event.metadata.get("memory_type") {
                    if mem_type == "workspace" {
                        // Extract relevant information
                        let size_bytes = event.metadata.get("size_bytes")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        
                        let op_type = event.name.clone();
                        let node_id = event.node_id;
                        
                        // Track allocation event
                        let allocation_time = event.start_time
                            .duration_since(self.start_time)
                            .as_nanos() as u64;
                        
                        // Add to allocation events
                        let workspace_event = super::types::WorkspaceAllocationEvent {
                            allocation_time,
                            deallocation_time: None,
                            size_bytes,
                            node_id,
                            op_type: op_type.clone(),
                        };
                        
                        // Store the workspace allocation with its ID for later deallocation matching
                        active_workspaces.insert(event.id, (workspace_event, size_bytes));
                        
                        // Track usage per operator type
                        *usage_per_operator.entry(op_type).or_insert(0) += size_bytes;
                        
                        // Update current and peak memory usage
                        current_bytes += size_bytes;
                        peak_bytes = peak_bytes.max(current_bytes);
                    }
                }
            }
            // Look for workspace deallocation events
            else if event.event_type == ProfileEventType::MemoryAllocation && 
                    event.metadata.get("operation").map_or(false, |op| op == "deallocate") {
                // Try to find matching allocation event using parent_id
                if let Some(parent_id) = event.parent_id {
                    if let Some((mut allocation_event, size)) = active_workspaces.remove(&parent_id) {
                        // Record deallocation time
                        let deallocation_time = event.start_time
                            .duration_since(self.start_time)
                            .as_nanos() as u64;
                        
                        allocation_event.deallocation_time = Some(deallocation_time);
                        allocation_events.push(allocation_event);
                        
                        // Update current bytes
                        current_bytes = current_bytes.saturating_sub(size);
                    }
                }
            }
            // Also scan for explicit workspace events if the execution engine emits them
            else if event.event_type == ProfileEventType::Other && 
                    event.name.contains("workspace") {
                if let Some(op_str) = event.metadata.get("operation") {
                    let size_bytes = event.metadata.get("size_bytes")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    
                    let op_type = event.metadata.get("op_type")
                        .unwrap_or(&event.name)
                        .to_string();
                    
                    let allocation_time = event.start_time
                        .duration_since(self.start_time)
                        .as_nanos() as u64;
                    
                    if op_str == "allocate" {
                        // Create workspace allocation event
                        let workspace_event = super::types::WorkspaceAllocationEvent {
                            allocation_time,
                            deallocation_time: None,
                            size_bytes,
                            node_id: event.node_id,
                            op_type: op_type.clone(),
                        };
                        
                        // Store for tracking
                        active_workspaces.insert(event.id, (workspace_event, size_bytes));
                        
                        // Track usage per operator type
                        *usage_per_operator.entry(op_type).or_insert(0) += size_bytes;
                        
                        // Update current and peak memory usage
                        current_bytes += size_bytes;
                        peak_bytes = peak_bytes.max(current_bytes);
                    } else if op_str == "deallocate" {
                        // Try to find the matching allocation
                        if let Some(parent_id) = event.parent_id {
                            if let Some((mut allocation_event, size)) = active_workspaces.remove(&parent_id) {
                                // Record deallocation time
                                let deallocation_time = event.start_time
                                    .duration_since(self.start_time)
                                    .as_nanos() as u64;
                                
                                allocation_event.deallocation_time = Some(deallocation_time);
                                allocation_events.push(allocation_event);
                                
                                // Update current bytes
                                current_bytes = current_bytes.saturating_sub(size);
                            }
                        }
                    }
                }
            }
        }
        
        // Any remaining active workspaces should be considered leaked or not yet deallocated
        // Add them to the allocation events list
        for (_, (allocation_event, _)) in active_workspaces {
            allocation_events.push(allocation_event);
        }
        
        // If we didn't find any workspace events but observed peak memory usage,
        // use the peak memory usage from the workspace manager
        if peak_bytes == 0 {
            peak_bytes = self.peak_memory_bytes.load(std::sync::atomic::Ordering::Relaxed);
        }
        
        super::types::WorkspaceUsage {
            peak_bytes,
            allocation_events,
            usage_per_operator,
        }
    }
    
    /// Calculate peak memory usage
    fn calculate_peak_memory(&self) -> usize {
        self.peak_memory_bytes.load(Ordering::Relaxed)
    }
    
    /// Profile the execution of an ONNX model
    pub fn profile_model_execution(
        &mut self, 
        engine: &mut ExecutionEngine, 
        inputs: &HashMap<String, Tensor>
    ) -> Result<ProfileResults> {
        // Set up profiling
        self.start_time = Instant::now();
        self.memory_events.clear();
        
        // Enable profiling in the engine
        engine.enable_profiling(true);
        engine.set_profiler(Arc::clone(&self.event_collector));
        
        // Hook memory tracking if enabled
        if self.track_memory {
            self.setup_memory_tracking(engine)?;
        }
        
        // Run the model
        let outputs = engine.run(inputs.clone())?;
        
        // Collect profiling data
        let events = self.event_collector.events();
        let perf_stats = PerformanceStats::from_profile_events(&events);
        
        // Process node execution times
        let node_execution_times = self.extract_node_execution_times(&events);
        let op_type_execution_times = self.extract_op_type_execution_times(&events);
        
        // Process parallelism stats if enabled
        let parallelism_stats = if self.track_parallelism {
            self.compute_parallelism_stats(&events)
        } else {
            ParallelismStats {
                max_parallel_ops: 0,
                avg_parallel_ops: 0.0,
                parallelism_histogram: HashMap::new(),
                parallelism_percentages: HashMap::new(),
            }
        };
        
        // Collect tensor statistics
        let tensor_stats = self.collect_tensor_stats(engine, &outputs);
        
        // Extract model information
        let model_info = self.extract_model_info(engine);
        
        // Disable profiling
        engine.enable_profiling(false);
        
        // Create results
        let profile_results = ProfileResults {
            performance: perf_stats,
            node_execution_times,
            op_type_execution_times,
            memory_events: self.memory_events.clone(),
            timeline: events,
            parallelism_stats,
            tensor_stats,
            model_info,
            critical_path_report: None,
            optimization_impact_scores: HashMap::new(),
        };
        
        Ok(profile_results)
    }
    
    /// Profile memory usage during model execution
    pub fn profile_memory_usage(
        &mut self,
        engine: &mut ExecutionEngine, 
        inputs: &HashMap<String, Tensor>
    ) -> Result<MemoryProfileResults> {
        // Set up memory profiling
        self.track_memory = true;
        self.memory_events.clear();
        self.start_time = Instant::now();
        
        // Enable profiling in the engine
        engine.enable_profiling(true);
        engine.set_profiler(Arc::clone(&self.event_collector));
        
        // Hook memory tracking
        self.setup_memory_tracking(engine)?;
        
        // Run the model
        let _ = engine.run(inputs.clone())?;
        
        // Get profiling events
        let events = self.event_collector.events();
        
        // Collect memory events from container if available
        if let Some(ref container) = self.memory_events_container {
            if let Ok(events) = container.lock() {
                self.memory_events = events.clone();
            }
        }
        
        // Process memory events
        let memory_by_type = self.calculate_memory_by_type();
        let memory_by_tensor = self.calculate_memory_by_tensor();
        let memory_by_operator = self.calculate_memory_by_operator();
        let tensor_lifetimes = self.calculate_tensor_lifetimes();
        let workspace_usage = self.calculate_workspace_usage(&events);
        
        // Calculate peak memory
        let peak_memory_bytes = self.calculate_peak_memory();
        
        // Disable profiling
        engine.enable_profiling(false);
        
        // Create memory profile results
        let memory_results = MemoryProfileResults {
            peak_memory_bytes,
            memory_events: self.memory_events.clone(),
            memory_by_type,
            memory_by_tensor,
            memory_by_operator,
            tensor_lifetimes,
            workspace_usage,
        };
        
        Ok(memory_results)
    }
    
    /// Record operation execution times
    pub fn record_op_execution_times(
        &mut self, 
        engine: &mut ExecutionEngine, 
        inputs: &HashMap<String, Tensor>
    ) -> Result<HashMap<NodeId, Duration>> {
        // Enable profiling
        engine.enable_profiling(true);
        engine.set_profiler(Arc::clone(&self.event_collector));
        
        // Run the model
        let _ = engine.run(inputs.clone())?;
        
        // Retrieve the profiling events
        let events = self.event_collector.events();
        
        // Extract execution times per operation
        let node_times = self.extract_node_execution_times(&events);
        
        // Disable profiling
        engine.enable_profiling(false);
        
        Ok(node_times)
    }
    
    /// Compute parallelism statistics from profile events
    fn compute_parallelism_stats(&self, events: &[ProfileEvent]) -> ParallelismStats {
        // Stub implementation for refactoring exercise
        ParallelismStats {
            max_parallel_ops: 0,
            avg_parallel_ops: 0.0,
            parallelism_histogram: HashMap::new(),
            parallelism_percentages: HashMap::new(),
        }
    }
}

// Public API functions
pub fn profile_model_execution(
    engine: &mut ExecutionEngine,
    inputs: &HashMap<String, Tensor>,
    track_memory: bool,
    track_parallelism: bool,
) -> Result<ProfileResults> {
    let mut profiler = ComprehensiveProfiler::new(track_memory, track_parallelism);
    profiler.profile_model_execution(engine, inputs)
}

pub fn profile_memory_usage(
    engine: &mut ExecutionEngine,
    inputs: &HashMap<String, Tensor>,
) -> Result<MemoryProfileResults> {
    let mut profiler = ComprehensiveProfiler::new(true, false);
    profiler.profile_memory_usage(engine, inputs)
}

pub fn record_op_execution_times(
    engine: &mut ExecutionEngine,
    inputs: &HashMap<String, Tensor>,
) -> Result<HashMap<NodeId, Duration>> {
    let mut profiler = ComprehensiveProfiler::new(false, false);
    profiler.record_op_execution_times(engine, inputs)
}