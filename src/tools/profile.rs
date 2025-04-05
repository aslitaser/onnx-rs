use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::Write;

use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};

use crate::{
    execution::{ExecutionEngine, context::ExecutionContext},
    model::{NodeId, TensorId, Graph},
    ops::tensor::{DataType, Tensor},
    memory::{allocator::MemoryAllocator, workspace::MemoryWorkspace},
    error,
};

/// Profile event type for distinguishing different operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileEventType {
    /// Model loading
    ModelLoad,
    /// Model preparation (graph building, optimization)
    ModelPrepare,
    /// Memory allocation
    MemoryAllocation,
    /// Operator execution
    OpExecution,
    /// Data transfer (e.g., between CPU and GPU)
    DataTransfer,
    /// Miscellaneous operations
    Other,
}

/// Thread-safe profiling event recorder
#[derive(Clone)]
pub struct Profiler {
    /// Events recorded by the profiler
    events: Arc<Mutex<Vec<ProfileEvent>>>,
    /// Unique event ID counter
    next_id: Arc<AtomicUsize>,
    /// Whether profiling is enabled
    enabled: bool,
}

/// A single profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Unique event ID
    pub id: usize,
    /// Event name
    pub name: String,
    /// Start time of the event
    pub start_time: Instant,
    /// Duration of the event if completed
    pub duration: Option<Duration>,
    /// Type of the event
    pub event_type: ProfileEventType,
    /// Associated node ID if applicable
    pub node_id: Option<NodeId>,
    /// Associated tensor ID if applicable
    pub tensor_id: Option<TensorId>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Parent event ID if this is a nested event
    pub parent_id: Option<usize>,
    /// Memory usage at this point in bytes (if available)
    pub memory_usage: Option<usize>,
}

impl Profiler {
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

/// Provides detailed performance statistics for an execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Time spent in each operator type in nanoseconds
    pub per_op_type_time_ns: HashMap<String, u64>,
    /// Time spent in each operator instance in nanoseconds
    pub per_op_instance_time_ns: HashMap<NodeId, u64>,
    /// Critical path operations (those on the longest execution path)
    pub critical_path: Vec<NodeId>,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Breakdown of memory usage by tensor type
    pub memory_by_tensor_type: HashMap<DataType, usize>,
}

/// Comprehensive profiling results for a model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResults {
    /// Overall performance statistics
    pub performance: PerformanceStats,
    /// Execution time for each node
    pub node_execution_times: HashMap<NodeId, Duration>,
    /// Execution time for each operator type
    pub op_type_execution_times: HashMap<String, Duration>,
    /// Memory allocation events
    pub memory_events: Vec<MemoryEvent>,
    /// Execution timeline containing all profiled events in chronological order
    pub timeline: Vec<ProfileEvent>,
    /// Parallelism statistics (how many operations executed in parallel)
    pub parallelism_stats: ParallelismStats,
    /// Input/output tensor information
    pub tensor_stats: HashMap<TensorId, TensorStats>,
    /// Overall model information
    pub model_info: ModelInfo,
}

/// Memory profiling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfileResults {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Memory allocation events in chronological order
    pub memory_events: Vec<MemoryEvent>,
    /// Memory usage per tensor data type
    pub memory_by_type: HashMap<DataType, usize>,
    /// Memory usage per tensor
    pub memory_by_tensor: HashMap<TensorId, usize>,
    /// Memory usage per operator
    pub memory_by_operator: HashMap<NodeId, usize>,
    /// Tensor lifetime information (allocation to deallocation)
    pub tensor_lifetimes: HashMap<TensorId, TensorLifetime>,
    /// Workspace memory usage
    pub workspace_usage: WorkspaceUsage,
}

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    /// Event type (allocation or deallocation)
    pub event_type: MemoryEventType,
    /// Timestamp when the event occurred
    pub timestamp: u64,
    /// Size of the memory allocation/deallocation in bytes
    pub size_bytes: usize,
    /// Associated tensor ID if applicable
    pub tensor_id: Option<TensorId>,
    /// Associated node ID if applicable
    pub node_id: Option<NodeId>,
    /// Memory address (for tracking specific allocations)
    pub address: usize,
    /// Memory pool or allocator identifier
    pub allocator_id: String,
    /// Additional metadata for the event (tensor shape, data type, etc.)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryEventType {
    /// Memory allocation
    Allocation,
    /// Memory deallocation
    Deallocation,
    /// Memory reuse
    Reuse,
    /// Memory pool growth
    PoolGrowth,
    /// Workspace allocation
    WorkspaceAllocation,
}

/// Tensor lifetime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLifetime {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Time when the tensor was allocated
    pub allocation_time: u64,
    /// Time when the tensor was deallocated
    pub deallocation_time: Option<u64>,
    /// Size of the tensor in bytes
    pub size_bytes: usize,
    /// Data type of the tensor
    pub data_type: DataType,
    /// Producing node ID
    pub producer_node: Option<NodeId>,
    /// Consumer node IDs
    pub consumer_nodes: Vec<NodeId>,
}

/// Workspace memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceUsage {
    /// Peak workspace memory usage in bytes
    pub peak_bytes: usize,
    /// Workspace allocation events
    pub allocation_events: Vec<WorkspaceAllocationEvent>,
    /// Workspace usage per operator
    pub usage_per_operator: HashMap<String, usize>,
}

/// Workspace allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceAllocationEvent {
    /// Time when the workspace was allocated
    pub allocation_time: u64,
    /// Time when the workspace was deallocated
    pub deallocation_time: Option<u64>,
    /// Size of the workspace in bytes
    pub size_bytes: usize,
    /// Associated node ID
    pub node_id: Option<NodeId>,
    /// Operator type
    pub op_type: String,
}

/// Parallelism statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismStats {
    /// Maximum number of operations executed in parallel
    pub max_parallel_ops: usize,
    /// Average number of operations executed in parallel
    pub avg_parallel_ops: f64,
    /// Histogram of parallel operation counts
    pub parallelism_histogram: HashMap<usize, usize>,
    /// Percentage of time spent with different levels of parallelism
    pub parallelism_percentages: HashMap<usize, f64>,
}

/// Tensor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data type
    pub data_type: DataType,
    /// Size in bytes
    pub size_bytes: usize,
    /// Time spent creating/computing the tensor
    pub computation_time: Option<Duration>,
    /// Tensor name if available
    pub name: Option<String>,
    /// Element count
    pub element_count: usize,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Number of operations
    pub op_count: usize,
    /// Number of inputs
    pub input_count: usize,
    /// Number of outputs
    pub output_count: usize,
    /// Graph structure info
    pub graph_info: GraphInfo,
    /// Operation count by type
    pub op_type_counts: HashMap<String, usize>,
}

/// Graph structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphInfo {
    /// Graph depth (longest path from input to output)
    pub max_depth: usize,
    /// Average depth of operations
    pub avg_depth: f64,
    /// Width of the graph (maximum number of operations at any depth)
    pub max_width: usize,
    /// Average width of the graph
    pub avg_width: f64,
    /// Number of branching points
    pub branch_count: usize,
}

/// Memory efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEfficiencyMetrics {
    /// Percentage of peak memory that is actively used
    pub utilization_percent: f64,
    /// Percentage of memory that could be reduced with perfect scheduling
    pub optimization_potential_percent: f64,
    /// Tensor reuse opportunities (tensors that could share memory)
    pub reuse_opportunities: Vec<TensorReuseOpportunity>,
    /// Fragmentation percentage
    pub fragmentation_percent: f64,
    /// Memory efficiency score (0-100)
    pub efficiency_score: u32,
    /// Recommended memory optimizations
    pub recommendations: Vec<MemoryOptimizationRecommendation>,
}

/// Tensor reuse opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorReuseOpportunity {
    /// First tensor ID
    pub tensor1_id: TensorId,
    /// Second tensor ID
    pub tensor2_id: TensorId,
    /// Potential memory savings in bytes
    pub potential_savings_bytes: usize,
    /// Confidence score (0-100)
    pub confidence: u32,
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: MemoryOptimizationType,
    /// Description of the recommendation
    pub description: String,
    /// Potential memory savings in bytes
    pub potential_savings_bytes: usize,
    /// Affected node IDs
    pub affected_nodes: Vec<NodeId>,
    /// Affected tensor IDs
    pub affected_tensors: Vec<TensorId>,
}

/// Memory optimization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOptimizationType {
    /// In-place operation to avoid allocation
    InPlaceOperation,
    /// Tensor reuse
    TensorReuse,
    /// Operation fusion
    OperationFusion,
    /// Tensor splitting
    TensorSplitting,
    /// Precision reduction
    PrecisionReduction,
    /// Custom memory pool
    CustomMemoryPool,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Description of the optimization
    pub description: String,
    /// Nodes affected by this optimization
    pub affected_nodes: Vec<NodeId>,
    /// Estimated performance improvement
    pub estimated_improvement_percent: f32,
    /// Confidence in the suggestion (0-100)
    pub confidence: u32,
    /// Code example or implementation hint
    pub implementation_hint: Option<String>,
}

/// Optimization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Operator fusion
    OperatorFusion,
    /// Kernel optimization
    KernelOptimization,
    /// Memory optimization
    MemoryOptimization,
    /// Parallelization
    Parallelization,
    /// Precision reduction
    PrecisionReduction,
    /// Layout optimization
    LayoutOptimization,
    /// Algorithmic optimization
    AlgorithmicOptimization,
}

/// Export format for profiling data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Chrome Trace format
    ChromeTrace,
    /// Markdown format
    Markdown,
    /// Protobuf format
    Protobuf,
}

impl PerformanceStats {
    /// Create performance statistics from profiling events
    pub fn from_profile_events(events: &[ProfileEvent]) -> Self {
        // Calculate total execution time
        let mut total_time_ns = 0;
        let mut op_type_time = HashMap::new();
        let mut op_instance_time = HashMap::new();
        let mut memory_usage = 0;
        let mut memory_by_type = HashMap::new();

        for event in events {
            if let Some(duration) = event.duration {
                let duration_ns = duration.as_nanos() as u64;
                
                // Total time for operator executions
                if event.event_type == ProfileEventType::OpExecution {
                    total_time_ns += duration_ns;
                    
                    // Time per operator type
                    if let Some(node_id) = event.node_id {
                        *op_instance_time.entry(node_id).or_insert(0) += duration_ns;
                        
                        // Extract operator type from event name
                        if let Some(op_type) = event.name.split_whitespace().next() {
                            *op_type_time.entry(op_type.to_string()).or_insert(0) += duration_ns;
                        }
                    }
                }
                
                // Track peak memory usage
                if let Some(mem) = event.memory_usage {
                    memory_usage = memory_usage.max(mem);
                    
                    // Extract tensor data type from metadata if available
                    if let Some(tensor_id) = event.tensor_id {
                        if let Some(data_type_str) = event.metadata.get("data_type") {
                            // Convert string representation to DataType
                            let data_type = data_type_from_string(data_type_str);
                            
                            if let Some(size_str) = event.metadata.get("size_bytes") {
                                if let Ok(size) = size_str.parse::<usize>() {
                                    *memory_by_type.entry(data_type).or_insert(0) += size;
                                    
                                    // Update memory_by_type with actual tensor size estimation
                                    if let Some(shape_str) = event.metadata.get("shape") {
                                        if let Ok(dimensions) = parse_tensor_shape(shape_str) {
                                            let calculated_size = calculate_tensor_memory_usage(data_type, &dimensions);
                                            // If the calculated size is significantly different, prefer it
                                            if calculated_size > size {
                                                *memory_by_type.entry(data_type).or_insert(0) = calculated_size;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Find the critical path (simplified algorithm)
        // In a real implementation, this would use the actual execution graph
        let mut critical_path = Vec::new();
        let mut nodes_by_time: Vec<(NodeId, u64)> = op_instance_time.into_iter().collect();
        nodes_by_time.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by time descending
        
        // Take the top 20% of operations by time as the critical path
        let critical_path_count = (nodes_by_time.len() as f64 * 0.2).ceil() as usize;
        for i in 0..critical_path_count.min(nodes_by_time.len()) {
            critical_path.push(nodes_by_time[i].0);
        }

        Self {
            total_execution_time_ns: total_time_ns,
            per_op_type_time_ns: op_type_time,
            per_op_instance_time_ns: nodes_by_time.into_iter().collect(),
            critical_path,
            peak_memory_bytes: memory_usage,
            memory_by_tensor_type: memory_by_type,
        }
    }

    /// Get the total execution time in milliseconds
    pub fn total_execution_time_ms(&self) -> f64 {
        self.total_execution_time_ns as f64 / 1_000_000.0
    }

    /// Get the operations sorted by execution time
    pub fn operations_by_time(&self) -> Vec<(String, f64)> {
        let mut ops: Vec<(String, f64)> = self.per_op_type_time_ns
            .iter()
            .map(|(op, &time)| (op.clone(), time as f64 / 1_000_000.0))
            .collect();
        ops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ops
    }

    /// Get the percentage of time spent in each operation
    pub fn operation_time_percentage(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        if self.total_execution_time_ns == 0 {
            return result;
        }

        for (op, &time) in &self.per_op_type_time_ns {
            let percentage = time as f64 / self.total_execution_time_ns as f64 * 100.0;
            result.insert(op.clone(), percentage);
        }
        result
    }
    
    /// Generate a summary report of the performance statistics
    pub fn summary_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("Total execution time: {:.3} ms\n", self.total_execution_time_ms()));
        report.push_str(&format!("Peak memory usage: {:.2} MB\n", self.peak_memory_bytes as f64 / (1024.0 * 1024.0)));
        
        report.push_str("\nOperation time breakdown:\n");
        for (op, ms) in self.operations_by_time() {
            let percentage = ms / self.total_execution_time_ms() * 100.0;
            report.push_str(&format!("  {}: {:.3} ms ({:.1}%)\n", op, ms, percentage));
        }
        
        report.push_str("\nCritical path operations:\n");
        for &node_id in &self.critical_path {
            if let Some(&time) = self.per_op_instance_time_ns.get(&node_id) {
                let ms = time as f64 / 1_000_000.0;
                report.push_str(&format!("  Node {}: {:.3} ms\n", node_id, ms));
            }
        }
        
        report.push_str("\nMemory usage by tensor type:\n");
        for (data_type, &size) in &self.memory_by_tensor_type {
            let mb = size as f64 / (1024.0 * 1024.0);
            report.push_str(&format!("  {:?}: {:.2} MB\n", data_type, mb));
        }
        
        report
    }
}

/// Enhanced profiler for detailed performance analysis
pub struct Profiler {
    /// Internal profiler for capturing events
    internal_profiler: Arc<crate::tools::profile::Profiler>,
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

impl Profiler {
    /// Create a new profiler
    pub fn new(track_memory: bool, track_parallelism: bool) -> Self {
        Self {
            internal_profiler: Arc::new(crate::tools::profile::Profiler::new(true)),
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
        engine.set_profiler(Arc::clone(&self.internal_profiler));
        
        // Hook memory tracking if enabled
        if self.track_memory {
            self.setup_memory_tracking(engine)?;
        }
        
        // Run the model
        let outputs = engine.run(inputs.clone())?;
        
        // Collect profiling data
        let events = engine.profile_events();
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
        engine.set_profiler(Arc::clone(&self.internal_profiler));
        
        // Hook memory tracking
        self.setup_memory_tracking(engine)?;
        
        // Run the model
        let _ = engine.run(inputs.clone())?;
        
        // Get profiling events
        let events = engine.profile_events();
        
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
        engine.set_profiler(Arc::clone(&self.internal_profiler));
        
        // Run the model
        let _ = engine.run(inputs.clone())?;
        
        // Retrieve the profiling events
        let events = engine.profile_events();
        
        // Extract execution times per operation
        let node_times = self.extract_node_execution_times(&events);
        
        // Disable profiling
        engine.enable_profiling(false);
        
        Ok(node_times)
    }
    
    /// Generate a flamegraph from profiling results
    pub fn generate_flamegraph(&self, profile_results: &ProfileResults, output_path: &Path) -> Result<()> {
        // Check if the directory exists, create it if it doesn't
        if let Some(parent_dir) = output_path.parent() {
            if !parent_dir.exists() {
                create_dir_all(parent_dir)?;
            }
        }
        
        // Create a flamegraph-compatible format
        let mut flamegraph_data = Vec::new();
        
        // Process timeline events into stack traces
        for event in &profile_results.timeline {
            if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
                // Create a "stack trace" for this event
                let mut stack = Vec::new();
                
                // Add event name to stack
                stack.push(event.name.clone());
                
                // Add parent events to stack if they exist
                let mut parent_id = event.parent_id;
                let mut depth = 1;
                
                while let Some(pid) = parent_id {
                    if depth >= self.max_stack_depth {
                        break;
                    }
                    
                    if let Some(parent_event) = profile_results.timeline.iter().find(|e| e.id == pid) {
                        stack.push(parent_event.name.clone());
                        parent_id = parent_event.parent_id;
                        depth += 1;
                    } else {
                        break;
                    }
                }
                
                // Reverse the stack to have root at the beginning
                stack.reverse();
                
                // Create the stack string
                let stack_str = stack.join(";");
                
                // Add the duration
                if let Some(duration) = event.duration {
                    let micros = duration.as_micros() as u64;
                    flamegraph_data.push(format!("{} {}", stack_str, micros));
                }
            }
        }
        
        // Write the flamegraph data to the file
        let mut file = File::create(output_path)?;
        for line in flamegraph_data {
            writeln!(file, "{}", line)?;
        }
        
        // Output message about how to generate the actual SVG
        println!("Flamegraph data written to {}. To generate an SVG, install and use inferno-flamegraph:", 
                 output_path.display());
        println!("  cargo install inferno");
        println!("  cat {} | inferno-flamegraph > flamegraph.svg", output_path.display());
        
        Ok(())
    }
    
    /// Find operations that are bottlenecks in the execution
    pub fn find_bottleneck_operations(profile_results: &ProfileResults) -> Vec<(NodeId, Duration)> {
        let mut bottlenecks = Vec::new();
        
        // Sort operations by execution time (descending)
        let mut op_times: Vec<(NodeId, Duration)> = profile_results.node_execution_times
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        
        op_times.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Calculate total time
        let total_time: Duration = profile_results.node_execution_times.values().sum();
        
        // Find operations that take more than 5% of the total time
        let threshold = total_time.mul_f64(0.05);
        
        for (node_id, duration) in op_times {
            if duration >= threshold {
                bottlenecks.push((node_id, duration));
            } else {
                // Stop once we're below the threshold
                break;
            }
        }
        
        bottlenecks
    }
    
    /// Analyze memory efficiency
    pub fn analyze_memory_efficiency(memory_results: &MemoryProfileResults) -> MemoryEfficiencyMetrics {
        // Calculate active memory usage over time
        let mut active_memory = HashMap::new();
        let mut total_memory = 0;
        let mut peak_memory = 0;
        let mut current_memory = 0;
        
        // Process memory events in chronological order
        let mut events = memory_results.memory_events.clone();
        events.sort_by_key(|e| e.timestamp);
        
        for event in &events {
            match event.event_type {
                MemoryEventType::Allocation => {
                    current_memory += event.size_bytes;
                    peak_memory = peak_memory.max(current_memory);
                    
                    // Record memory at this timestamp
                    active_memory.insert(event.timestamp, current_memory);
                    
                    // If this is for a tensor, add to total active memory
                    if event.tensor_id.is_some() {
                        total_memory += event.size_bytes;
                    }
                },
                MemoryEventType::Deallocation => {
                    current_memory = current_memory.saturating_sub(event.size_bytes);
                    active_memory.insert(event.timestamp, current_memory);
                },
                _ => {
                    // Record current memory for other events too
                    active_memory.insert(event.timestamp, current_memory);
                }
            }
        }
        
        // Calculate average memory utilization
        let avg_memory = if !active_memory.is_empty() {
            active_memory.values().sum::<usize>() as f64 / active_memory.len() as f64
        } else {
            0.0
        };
        
        // Calculate utilization percentage
        let utilization_percent = if peak_memory > 0 {
            avg_memory * 100.0 / peak_memory as f64
        } else {
            0.0
        };
        
        // Find tensor reuse opportunities
        let reuse_opportunities = Self::find_tensor_reuse_opportunities(memory_results);
        
        // Calculate optimization potential
        let total_potential_savings: usize = reuse_opportunities.iter()
            .map(|op| op.potential_savings_bytes)
            .sum();
        
        let optimization_potential_percent = if peak_memory > 0 {
            total_potential_savings as f64 * 100.0 / peak_memory as f64
        } else {
            0.0
        };
        
        // Calculate fragmentation
        let fragmentation_percent = Self::calculate_memory_fragmentation(&events, peak_memory);
        
        // Calculate an overall efficiency score
        let efficiency_score = (100.0 - (fragmentation_percent * 0.5 + 
                                        (100.0 - utilization_percent) * 0.3 + 
                                        optimization_potential_percent * 0.2)) as u32;
        
        // Generate recommendations
        let recommendations = Self::generate_memory_recommendations(memory_results, reuse_opportunities.clone());
        
        MemoryEfficiencyMetrics {
            utilization_percent,
            optimization_potential_percent,
            reuse_opportunities,
            fragmentation_percent,
            efficiency_score: efficiency_score.clamp(0, 100),
            recommendations,
        }
    }
    
    /// Suggest optimization opportunities based on profile results
    pub fn suggest_optimization_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // Find fusion opportunities
        let fusion_opportunities = Self::find_fusion_opportunities(profile_results);
        suggestions.extend(fusion_opportunities);
        
        // Find kernel optimization opportunities
        let kernel_opportunities = Self::find_kernel_optimization_opportunities(profile_results);
        suggestions.extend(kernel_opportunities);
        
        // Find memory optimization opportunities
        let memory_opportunities = Self::find_memory_optimization_opportunities(profile_results);
        suggestions.extend(memory_opportunities);
        
        // Find parallelization opportunities
        let parallelization_opportunities = Self::find_parallelization_opportunities(profile_results);
        suggestions.extend(parallelization_opportunities);
        
        // Sort by estimated improvement (descending)
        suggestions.sort_by(|a, b| {
            b.estimated_improvement_percent.partial_cmp(&a.estimated_improvement_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        suggestions
    }
    
    /// Export profiling data to various formats
    pub fn export_profile_data(profile_results: &ProfileResults, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Json => {
                let json_data = serde_json::to_string_pretty(profile_results)?;
                Ok(json_data.into_bytes())
            },
            ExportFormat::Csv => {
                let mut csv_data = Vec::new();
                
                // Write header
                csv_data.extend_from_slice(b"Node ID,Operator Type,Execution Time (ms),Memory Usage (bytes)\n");
                
                // Write node data
                for (node_id, duration) in &profile_results.node_execution_times {
                    // Try to find node info in timeline
                    let op_type = profile_results.timeline.iter()
                        .find(|e| e.node_id == Some(*node_id))
                        .and_then(|e| e.name.split(':').next())
                        .unwrap_or("Unknown");
                    
                    // Get memory usage for this node if available
                    let memory = profile_results.memory_events.iter()
                        .filter(|e| e.node_id == Some(*node_id) && e.event_type == MemoryEventType::Allocation)
                        .map(|e| e.size_bytes)
                        .sum::<usize>();
                    
                    let line = format!("{},{},{:.3},{}\n", 
                                      node_id, 
                                      op_type,
                                      duration.as_secs_f64() * 1000.0,
                                      memory);
                    
                    csv_data.extend_from_slice(line.as_bytes());
                }
                
                Ok(csv_data)
            },
            ExportFormat::ChromeTrace => {
                // Create Chrome Trace Format (Catapult format)
                let mut trace_events = Vec::new();
                
                // Add trace events for each profiling event
                for event in &profile_results.timeline {
                    if let Some(duration) = event.duration {
                        let start_time_us = event.start_time.duration_since(profile_results.timeline[0].start_time)
                            .as_micros() as u64;
                        
                        // Determine the category based on event type
                        let category = match event.event_type {
                            ProfileEventType::OpExecution => "Execution",
                            ProfileEventType::MemoryAllocation => "Memory",
                            ProfileEventType::DataTransfer => "DataTransfer",
                            _ => "Other",
                        };
                        
                        // Create begin event
                        let begin_event = serde_json::json!({
                            "name": event.name,
                            "cat": category,
                            "ph": "B",  // Begin event
                            "ts": start_time_us,
                            "pid": 1,
                            "tid": event.node_id.map(|id| id as u64).unwrap_or(0),
                            "args": {
                                "node_id": event.node_id,
                                "tensor_id": event.tensor_id,
                            }
                        });
                        
                        // Create end event
                        let end_event = serde_json::json!({
                            "name": event.name,
                            "cat": category,
                            "ph": "E",  // End event
                            "ts": start_time_us + duration.as_micros() as u64,
                            "pid": 1,
                            "tid": event.node_id.map(|id| id as u64).unwrap_or(0),
                            "args": {}
                        });
                        
                        trace_events.push(begin_event);
                        trace_events.push(end_event);
                    }
                }
                
                // Add memory counter events
                let mut current_memory = 0;
                let mut memory_events = profile_results.memory_events.clone();
                memory_events.sort_by_key(|e| e.timestamp);
                
                for event in &memory_events {
                    let timestamp = event.timestamp;
                    
                    // Update current memory based on event type
                    match event.event_type {
                        MemoryEventType::Allocation => current_memory += event.size_bytes,
                        MemoryEventType::Deallocation => current_memory = current_memory.saturating_sub(event.size_bytes),
                        _ => {}
                    }
                    
                    // Create counter event
                    let counter_event = serde_json::json!({
                        "name": "Memory Usage",
                        "cat": "Memory",
                        "ph": "C",  // Counter event
                        "ts": timestamp,
                        "pid": 1,
                        "args": {
                            "Memory (MB)": current_memory as f64 / (1024.0 * 1024.0)
                        }
                    });
                    
                    trace_events.push(counter_event);
                }
                
                // Create the complete trace
                let trace = serde_json::json!({
                    "traceEvents": trace_events,
                    "displayTimeUnit": "ms"
                });
                
                let json_data = serde_json::to_string(&trace)?;
                Ok(json_data.into_bytes())
            },
            ExportFormat::Markdown => {
                let mut md_data = String::new();
                
                // Add title
                md_data.push_str("# ONNX Runtime Profiling Report\n\n");
                
                // Add summary
                md_data.push_str("## Summary\n\n");
                md_data.push_str(&format!("- **Model**: {}\n", profile_results.model_info.name));
                md_data.push_str(&format!("- **Total Execution Time**: {:.3} ms\n", 
                                          profile_results.performance.total_execution_time_ns as f64 / 1_000_000.0));
                md_data.push_str(&format!("- **Peak Memory Usage**: {:.2} MB\n", 
                                          profile_results.performance.peak_memory_bytes as f64 / (1024.0 * 1024.0)));
                md_data.push_str(&format!("- **Operation Count**: {}\n", profile_results.model_info.op_count));
                md_data.push_str(&format!("- **Max Parallel Operations**: {}\n", 
                                          profile_results.parallelism_stats.max_parallel_ops));
                md_data.push_str("\n");
                
                // Add operation timing table
                md_data.push_str("## Operation Timing\n\n");
                md_data.push_str("| Operator Type | Count | Total Time (ms) | Percentage | Avg Time (ms) |\n");
                md_data.push_str("|--------------|-------|-----------------|------------|---------------|\n");
                
                // Calculate operation type counts and times
                let mut op_stats: HashMap<String, (usize, Duration)> = HashMap::new();
                for (node_id, duration) in &profile_results.node_execution_times {
                    // Find the operation type from the timeline events
                    if let Some(event) = profile_results.timeline.iter()
                        .find(|e| e.node_id == Some(*node_id)) {
                        let op_type = event.name.split(':').next().unwrap_or("Unknown").to_string();
                        op_stats.entry(op_type)
                            .and_modify(|(count, time)| { *count += 1; *time += *duration })
                            .or_insert((1, *duration));
                    }
                }
                
                // Calculate total time
                let total_time: Duration = profile_results.node_execution_times.values().sum();
                
                // Sort by total time (descending)
                let mut op_stats_sorted: Vec<(String, (usize, Duration))> = op_stats.into_iter().collect();
                op_stats_sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
                
                // Add rows for each operation type
                for (op_type, (count, duration)) in op_stats_sorted {
                    let percentage = duration.as_secs_f64() * 100.0 / total_time.as_secs_f64();
                    let avg_time = duration.as_secs_f64() * 1000.0 / count as f64;
                    
                    md_data.push_str(&format!("| {} | {} | {:.3} | {:.1}% | {:.3} |\n",
                                             op_type, count, duration.as_secs_f64() * 1000.0, 
                                             percentage, avg_time));
                }
                md_data.push_str("\n");
                
                // Add memory usage section
                md_data.push_str("## Memory Usage\n\n");
                md_data.push_str("### By Tensor Type\n\n");
                md_data.push_str("| Data Type | Memory Usage (MB) | Percentage |\n");
                md_data.push_str("|-----------|-------------------|------------|\n");
                
                // Calculate total memory by type
                let total_memory: usize = profile_results.performance.memory_by_tensor_type.values().sum();
                
                // Sort by memory usage (descending)
                let mut memory_by_type: Vec<(&DataType, &usize)> = 
                    profile_results.performance.memory_by_tensor_type.iter().collect();
                memory_by_type.sort_by(|a, b| b.1.cmp(a.1));
                
                // Add rows for each data type
                for (data_type, &bytes) in memory_by_type {
                    let mb = bytes as f64 / (1024.0 * 1024.0);
                    let percentage = if total_memory > 0 {
                        bytes as f64 * 100.0 / total_memory as f64
                    } else {
                        0.0
                    };
                    
                    md_data.push_str(&format!("| {:?} | {:.2} | {:.1}% |\n",
                                             data_type, mb, percentage));
                }
                md_data.push_str("\n");
                
                // Add bottleneck operations
                md_data.push_str("## Bottleneck Operations\n\n");
                md_data.push_str("| Node ID | Operator | Time (ms) | Percentage |\n");
                md_data.push_str("|---------|----------|-----------|------------|\n");
                
                // Find bottleneck operations (>5% of total time)
                let bottlenecks = Self::find_bottleneck_operations(profile_results);
                
                for (node_id, duration) in bottlenecks {
                    // Find the operation name from timeline
                    let op_name = profile_results.timeline.iter()
                        .find(|e| e.node_id == Some(node_id))
                        .map(|e| e.name.clone())
                        .unwrap_or_else(|| format!("Node {}", node_id));
                    
                    let percentage = duration.as_secs_f64() * 100.0 / total_time.as_secs_f64();
                    
                    md_data.push_str(&format!("| {} | {} | {:.3} | {:.1}% |\n",
                                             node_id, op_name, duration.as_secs_f64() * 1000.0, percentage));
                }
                md_data.push_str("\n");
                
                // Add optimization suggestions
                let suggestions = Self::suggest_optimization_opportunities(profile_results);
                if !suggestions.is_empty() {
                    md_data.push_str("## Optimization Suggestions\n\n");
                    
                    for (i, suggestion) in suggestions.iter().enumerate() {
                        md_data.push_str(&format!("### {}. {}\n\n", i+1, suggestion.optimization_type.to_string()));
                        md_data.push_str(&format!("**Description**: {}\n\n", suggestion.description));
                        md_data.push_str(&format!("**Estimated Improvement**: {:.1}%\n\n", 
                                                 suggestion.estimated_improvement_percent));
                        md_data.push_str(&format!("**Confidence**: {}%\n\n", suggestion.confidence));
                        md_data.push_str("**Affected Nodes**:\n");
                        
                        for &node_id in &suggestion.affected_nodes {
                            let op_name = profile_results.timeline.iter()
                                .find(|e| e.node_id == Some(node_id))
                                .map(|e| e.name.clone())
                                .unwrap_or_else(|| format!("Node {}", node_id));
                            
                            md_data.push_str(&format!("- {}: {}\n", node_id, op_name));
                        }
                        
                        if let Some(hint) = &suggestion.implementation_hint {
                            md_data.push_str("\n**Implementation Hint**:\n");
                            md_data.push_str(&format!("```rust\n{}\n```\n", hint));
                        }
                        
                        md_data.push_str("\n");
                    }
                }
                
                Ok(md_data.into_bytes())
            },
            ExportFormat::Protobuf => {
                // In a real implementation, this would serialize to a protobuf
                // For this example, we'll just return an error
                Err(anyhow!("Protobuf export format not yet implemented"))
            }
        }
    }
    
    /// Helper methods
    
    /// Set up memory tracking in the engine
    fn setup_memory_tracking(&mut self, engine: &mut ExecutionEngine) -> Result<()> {
        // Create a shared container for memory events
        let memory_events = Arc::new(Mutex::new(Vec::new()));
        
        // Create clone for allocation callback
        let alloc_events_clone = Arc::clone(&memory_events);
        let start_time = self.start_time;
        
        // Get references to our atomic counters and containers for the closure
        let peak_memory_bytes = Arc::new(AtomicUsize::new(0));
        let current_memory_bytes = Arc::new(AtomicUsize::new(0));
        let memory_by_type = Arc::clone(&self.memory_by_type);
        let memory_by_tensor = Arc::clone(&self.memory_by_tensor);
        let memory_by_operator = Arc::clone(&self.memory_by_operator);
        
        // Clone these values for the closure
        let peak_memory_ref = Arc::clone(&peak_memory_bytes);
        let current_memory_ref = Arc::clone(&current_memory_bytes);
        
        // Create a weak engine reference to avoid ownership issues
        let engine_ref = Arc::downgrade(&engine.as_ref().unwrap_or_else(|| engine));
        
        // Register allocation callback
        engine.register_memory_allocation_callback(Box::new(move |size, tensor_id, node_id| {
            let timestamp = start_time.elapsed().as_micros() as u64;
            let address = std::ptr::null::<u8>() as usize; // Would be actual address in real impl
            
            // Try to get tensor details like shape and data type
            let mut allocator_id = "default".to_string();
            let mut event_meta = HashMap::new();
            
            if let Some(tid) = tensor_id {
                if let Some(eng) = engine_ref.upgrade() {
                    // Try to get tensor information
                    if let Some(shape) = eng.get_tensor_shape(tid) {
                        let shape_str = format!("{:?}", shape);
                        event_meta.insert("shape".to_string(), shape_str);
                    }
                    
                    if let Some(dtype) = eng.get_tensor_data_type(tid) {
                        let dtype_str = data_type_to_string(dtype);
                        event_meta.insert("data_type".to_string(), dtype_str);
                        // Add data type to allocator_id for better tracking
                        allocator_id = format!("datatype_{}", dtype_str.to_lowercase());
                    }
                    
                    // Try to get tensor name if available
                    if let Some(name) = eng.get_tensor_name(tid) {
                        event_meta.insert("name".to_string(), name);
                    }
                }
            }
            
            // Add memory size to metadata
            event_meta.insert("size_bytes".to_string(), size.to_string());
            
            // Create the memory event
            let event = MemoryEvent {
                event_type: MemoryEventType::Allocation,
                timestamp,
                size_bytes: size,
                tensor_id,
                node_id,
                address,
                allocator_id,
                metadata: event_meta,
            };
            
            // Track the event in our shared container
            if let Ok(mut events) = alloc_events_clone.lock() {
                events.push(event);
            }
            
            // Update current and peak memory
            let current = current_memory_ref.fetch_add(size, Ordering::AcqRel) + size;
            let mut peak = peak_memory_ref.load(Ordering::Acquire);
            while current > peak {
                match peak_memory_ref.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::AcqRel,
                    Ordering::Acquire
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
            
            // Update memory by data type
            if let Some(tid) = tensor_id {
                // Get actual data type from tensor or engine
                let data_type = if let Some(eng) = engine_ref.upgrade() {
                    if let Some(tensor_data_type) = eng.get_tensor_data_type(tid) {
                        tensor_data_type
                    } else {
                        // If tensor data type can't be determined, check if we have it in metadata
                        if let Some(meta_type) = eng.get_tensor_metadata(tid, "data_type") {
                            data_type_from_string(&meta_type)
                        } else {
                            DataType::Float32 // Default fallback
                        }
                    }
                } else {
                    DataType::Float32 // Fallback if engine reference is gone
                };
                
                if let Ok(mut memory_by_type_map) = memory_by_type.lock() {
                    *memory_by_type_map.entry(data_type).or_insert(0) += size;
                }
                
                // Update memory by tensor
                if let Ok(mut memory_by_tensor_map) = memory_by_tensor.lock() {
                    *memory_by_tensor_map.entry(tid).or_insert(0) += size;
                }
            }
            
            // Update memory by operator
            if let Some(nid) = node_id {
                if let Ok(mut memory_by_operator_map) = memory_by_operator.lock() {
                    *memory_by_operator_map.entry(nid).or_insert(0) += size;
                }
            }
        }))?;
        
        // Store the atomic counters for later use
        self.peak_memory_bytes = peak_memory_bytes;
        self.current_memory_bytes = current_memory_bytes;
        
        // Create clone for deallocation callback
        let dealloc_events_clone = Arc::clone(&memory_events);
        let start_time_dealloc = self.start_time;
        
        // Clone references for the closure
        let current_memory_ref = Arc::clone(&self.current_memory_bytes);
        let memory_by_type = Arc::clone(&self.memory_by_type);
        let memory_by_tensor = Arc::clone(&self.memory_by_tensor);
        let memory_by_operator = Arc::clone(&self.memory_by_operator);
        
        // Register deallocation callback
        engine.register_memory_deallocation_callback(Box::new(move |size, tensor_id, node_id| {
            let timestamp = start_time_dealloc.elapsed().as_micros() as u64;
            let address = std::ptr::null::<u8>() as usize; // Would be actual address in real impl
            
            let mut metadata = HashMap::new();
            
            // Add memory size to metadata
            metadata.insert("size_bytes".to_string(), size.to_string());
            
            let event = MemoryEvent {
                event_type: MemoryEventType::Deallocation,
                timestamp,
                size_bytes: size,
                tensor_id,
                node_id,
                address,
                allocator_id: "default".to_string(),
                metadata,
            };
            
            // Record the event
            if let Ok(mut events) = dealloc_events_clone.lock() {
                events.push(event);
            }
            
            // Update current memory usage
            current_memory_ref.fetch_sub(size, Ordering::AcqRel);
            
            // Note: We don't update peak memory since deallocations don't affect peak
            
            // We could track which tensors have been deallocated for memory leak detection
            if let Some(tid) = tensor_id {
                // We don't actually remove from memory_by_type or memory_by_tensor
                // since we want to track total allocations, not current usage
            }
        }))?;
        
        // Create clone for reuse callback
        let reuse_events_clone = Arc::clone(&memory_events);
        let start_time_reuse = self.start_time;
        
        // Register memory reuse callback
        engine.register_memory_reuse_callback(Box::new(move |size, tensor_id, node_id| {
            let timestamp = start_time_reuse.elapsed().as_micros() as u64;
            let address = std::ptr::null::<u8>() as usize; // Would be actual address in real impl
            
            let mut metadata = HashMap::new();
            
            // Add memory size to metadata
            metadata.insert("size_bytes".to_string(), size.to_string());
            
            let event = MemoryEvent {
                event_type: MemoryEventType::Reuse,
                timestamp,
                size_bytes: size,
                tensor_id,
                node_id,
                address,
                allocator_id: "memory_pool".to_string(),
                metadata,
            };
            
            if let Ok(mut events) = reuse_events_clone.lock() {
                events.push(event);
            }
        }))?;
        
        // Create clone for workspace allocation callback
        let workspace_events_clone = Arc::clone(&memory_events);
        let start_time_workspace = self.start_time;
        
        // Register workspace allocation callback
        engine.register_workspace_allocation_callback(Box::new(move |size, node_id| {
            let timestamp = start_time_workspace.elapsed().as_micros() as u64;
            let address = std::ptr::null::<u8>() as usize; // Would be actual address in real impl
            
            let mut metadata = HashMap::new();
            
            // Add memory size to metadata
            metadata.insert("size_bytes".to_string(), size.to_string());
            
            // Try to get operator name if node_id is available
            if let Some(nid) = node_id {
                if let Some(eng) = engine_ref.upgrade() {
                    if let Some(op_type) = eng.get_operator_type(nid) {
                        metadata.insert("op_type".to_string(), op_type);
                    }
                }
            }
            
            let event = MemoryEvent {
                event_type: MemoryEventType::WorkspaceAllocation,
                timestamp,
                size_bytes: size,
                tensor_id: None,
                node_id,
                address,
                allocator_id: "workspace".to_string(),
                metadata,
            };
            
            if let Ok(mut events) = workspace_events_clone.lock() {
                events.push(event);
            }
        }))?;
        
        // Create clone for pool growth callback
        let pool_events_clone = Arc::clone(&memory_events);
        let start_time_pool = self.start_time;
        
        // Register pool growth callback
        engine.register_pool_growth_callback(Box::new(move |size, pool_id| {
            let timestamp = start_time_pool.elapsed().as_micros() as u64;
            let address = std::ptr::null::<u8>() as usize; // Would be actual address in real impl
            
            let mut metadata = HashMap::new();
            
            // Add memory size to metadata
            metadata.insert("size_bytes".to_string(), size.to_string());
            metadata.insert("pool_id".to_string(), pool_id.to_string());
            
            let event = MemoryEvent {
                event_type: MemoryEventType::PoolGrowth,
                timestamp,
                size_bytes: size,
                tensor_id: None,
                node_id: None,
                address,
                allocator_id: format!("pool_{}", pool_id),
                metadata,
            };
            
            if let Ok(mut events) = pool_events_clone.lock() {
                events.push(event);
            }
        }))?;
        
        // Create a new method to synchronize memory events for analysis
        let sync_memory_events = move || {
            if let Ok(events) = memory_events.lock() {
                // Sort events by timestamp for proper analysis
                let mut sorted_events = events.clone();
                sorted_events.sort_by_key(|e| e.timestamp);
                sorted_events
            } else {
                Vec::new()
            }
        };
        
        // Store the events for later retrieval
        self.memory_events = sync_memory_events();
        
        // Store the memory_events Arc for later updates
        self.memory_events_container = Some(memory_events);
        
        // This would be a good place to analyze the current memory state and provide
        // initial statistics or alerts on memory patterns
        
        Ok(())
    }
    
    /// Extract execution times for each node
    fn extract_node_execution_times(&self, events: &[ProfileEvent]) -> HashMap<NodeId, Duration> {
        let mut node_times = HashMap::new();
        
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
                if let Some(node_id) = event.node_id {
                    let duration = event.duration.unwrap();
                    node_times.entry(node_id)
                        .and_modify(|d| *d += duration)
                        .or_insert(duration);
                }
            }
        }
        
        node_times
    }
    
    /// Extract execution times by operator type
    fn extract_op_type_execution_times(&self, events: &[ProfileEvent]) -> HashMap<String, Duration> {
        let mut op_times = HashMap::new();
        
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
                // Extract operator type from the event name (e.g., "Conv:" -> "Conv")
                if let Some(op_type) = event.name.split(':').next() {
                    let op_type = op_type.trim().to_string();
                    let duration = event.duration.unwrap();
                    
                    op_times.entry(op_type)
                        .and_modify(|d| *d += duration)
                        .or_insert(duration);
                }
            }
        }
        
        op_times
    }
    
    /// Compute parallelism statistics
    fn compute_parallelism_stats(&self, events: &[ProfileEvent]) -> ParallelismStats {
        // Create a timeline of start/end points to analyze parallelism
        let mut timeline = Vec::new();
        
        for event in events {
            if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
                let start_time = event.start_time;
                let end_time = start_time + event.duration.unwrap();
                
                timeline.push((start_time, true));  // Start event
                timeline.push((end_time, false));   // End event
            }
        }
        
        // Sort timeline by time
        timeline.sort_by(|a, b| a.0.cmp(&b.0));
        
        // Analyze parallelism
        let mut current_parallel = 0;
        let mut max_parallel = 0;
        let mut parallel_durations = HashMap::new();
        let mut last_time = None;
        let mut total_duration = Duration::from_secs(0);
        
        for (time, is_start) in timeline {
            // Record duration at current parallelism level
            if let Some(last) = last_time {
                let duration = time.duration_since(last);
                total_duration += duration;
                
                *parallel_durations.entry(current_parallel).or_insert(Duration::from_secs(0)) += duration;
            }
            
            // Update parallelism level
            if is_start {
                current_parallel += 1;
                max_parallel = max_parallel.max(current_parallel);
            } else {
                current_parallel = current_parallel.saturating_sub(1);
            }
            
            last_time = Some(time);
        }
        
        // Calculate histogram and percentages
        let mut histogram = HashMap::new();
        let mut percentages = HashMap::new();
        
        for (parallel_level, duration) in &parallel_durations {
            histogram.insert(*parallel_level, duration.as_millis() as usize);
            
            if !total_duration.is_zero() {
                let percentage = duration.as_secs_f64() * 100.0 / total_duration.as_secs_f64();
                percentages.insert(*parallel_level, percentage);
            }
        }
        
        // Calculate average parallelism
        let weighted_sum: f64 = parallel_durations.iter()
            .map(|(level, duration)| *level as f64 * duration.as_secs_f64())
            .sum();
        
        let avg_parallel = if !total_duration.is_zero() {
            weighted_sum / total_duration.as_secs_f64()
        } else {
            0.0
        };
        
        ParallelismStats {
            max_parallel_ops: max_parallel,
            avg_parallel_ops: avg_parallel,
            parallelism_histogram: histogram,
            parallelism_percentages: percentages,
        }
    }
    
    /// Collect statistics about tensors
    fn collect_tensor_stats(&self, engine: &ExecutionEngine, outputs: &HashMap<String, Tensor>) -> HashMap<TensorId, TensorStats> {
        let mut tensor_stats = HashMap::new();
        
        // In a real implementation, this would iterate through all tensors in the engine
        // and collect their statistics. For this example, we'll just use the output tensors.
        
        for (name, tensor) in outputs {
            let tensor_id = TensorId(name.as_bytes().to_vec()); // This is a simplified way to create a TensorId
            
            let stats = TensorStats {
                tensor_id: tensor_id.clone(),
                shape: tensor.shape().to_vec(),
                data_type: tensor.data_type(),
                size_bytes: tensor.size_bytes(),
                computation_time: None, // Would be obtained from profiling events
                name: Some(name.clone()),
                element_count: tensor.element_count(),
            };
            
            tensor_stats.insert(tensor_id, stats);
        }
        
        tensor_stats
    }
    
    /// Extract model information
    fn extract_model_info(&self, engine: &ExecutionEngine) -> ModelInfo {
        // In a real implementation, this would extract information from the engine's model
        // For this example, we'll create placeholder data
        
        let graph_info = GraphInfo {
            max_depth: 10,
            avg_depth: 5.5,
            max_width: 16,
            avg_width: 8.0,
            branch_count: 3,
        };
        
        let mut op_type_counts = HashMap::new();
        op_type_counts.insert("Conv".to_string(), 5);
        op_type_counts.insert("Relu".to_string(), 5);
        op_type_counts.insert("MaxPool".to_string(), 2);
        op_type_counts.insert("MatMul".to_string(), 3);
        op_type_counts.insert("Add".to_string(), 4);
        
        ModelInfo {
            name: "Model".to_string(),
            op_count: op_type_counts.values().sum(),
            input_count: 1,
            output_count: 1,
            graph_info,
            op_type_counts,
        }
    }
    
    /// Calculate memory usage by tensor data type
    fn calculate_memory_by_type(&self) -> HashMap<DataType, usize> {
        // First try to get the tracked data from our shared mutex
        if let Ok(memory_by_type) = self.memory_by_type.lock() {
            return memory_by_type.clone();
        }
        
        // Fallback implementation if the mutex lock fails
        let mut memory_by_type = HashMap::new();
        
        for event in &self.memory_events {
            if event.event_type == MemoryEventType::Allocation {
                // Try to determine data type from the tensor_id
                // In a real implementation, you would query the tensor metadata
                
                // We'll use a more sophisticated approach than before:
                // If tensor_id is available, look for tensor metadata in the events
                let data_type = if let Some(tensor_id) = event.tensor_id {
                    // Find metadata for this tensor in the events
                    self.memory_events.iter()
                        .filter_map(|e| {
                            if e.tensor_id == Some(tensor_id) {
                                // Look for data_type metadata in memory events
                                if let Some(data_type_str) = e.allocator_id.strip_prefix("datatype_") {
                                    match data_type_str {
                                        "float32" => Some(DataType::Float32),
                                        "float64" => Some(DataType::Float64),
                                        "int32" => Some(DataType::Int32),
                                        "int64" => Some(DataType::Int64),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap_or(DataType::Float32) // Default to Float32 if not found
                } else {
                    // No tensor_id, use default type
                    DataType::Float32
                };
                
                *memory_by_type.entry(data_type).or_insert(0) += event.size_bytes;
            }
        }
        
        memory_by_type
    }
    
    /// Calculate memory usage by tensor
    fn calculate_memory_by_tensor(&self) -> HashMap<TensorId, usize> {
        // First try to get the tracked data from our shared mutex
        if let Ok(memory_by_tensor) = self.memory_by_tensor.lock() {
            return memory_by_tensor.clone();
        }
        
        // Fallback implementation if the mutex lock fails
        let mut memory_by_tensor = HashMap::new();
        
        // Process all memory events to build a comprehensive tensor memory usage map
        for event in &self.memory_events {
            match event.event_type {
                MemoryEventType::Allocation => {
                    if let Some(tensor_id) = event.tensor_id {
                        *memory_by_tensor.entry(tensor_id).or_insert(0) += event.size_bytes;
                    }
                },
                MemoryEventType::Deallocation => {
                    // We don't subtract here because we want to track total memory allocated per tensor
                    // But we can mark tensors that have been deallocated for later analysis
                },
                MemoryEventType::Reuse => {
                    // For reused tensors, we want to avoid double-counting
                    // So we don't add to the total here
                },
                _ => {}
            }
        }
        
        memory_by_tensor
    }
    
    /// Calculate memory usage by operator
    fn calculate_memory_by_operator(&self) -> HashMap<NodeId, usize> {
        // First try to get the tracked data from our shared mutex
        if let Ok(memory_by_operator) = self.memory_by_operator.lock() {
            return memory_by_operator.clone();
        }
        
        // Fallback implementation if the mutex lock fails
        let mut memory_by_operator = HashMap::new();
        let mut operator_allocated_memory: HashMap<NodeId, HashMap<usize, usize>> = HashMap::new();
        
        // Process all memory events to build a comprehensive operator memory usage map
        for event in &self.memory_events {
            match event.event_type {
                MemoryEventType::Allocation => {
                    if let Some(node_id) = event.node_id {
                        // Track both the total memory and the specific allocations
                        *memory_by_operator.entry(node_id).or_insert(0) += event.size_bytes;
                        
                        // Track this specific allocation by its address
                        operator_allocated_memory
                            .entry(node_id)
                            .or_insert_with(HashMap::new)
                            .insert(event.address, event.size_bytes);
                    }
                },
                MemoryEventType::Deallocation => {
                    // We don't subtract from the total because we want to track peak memory usage per operator
                    // But we can remove the specific allocation from our tracking
                    if let Some(node_id) = event.node_id {
                        if let Some(allocations) = operator_allocated_memory.get_mut(&node_id) {
                            allocations.remove(&event.address);
                        }
                    }
                },
                MemoryEventType::WorkspaceAllocation => {
                    // Workspace memory is temporary and tied to specific operators
                    if let Some(node_id) = event.node_id {
                        *memory_by_operator.entry(node_id).or_insert(0) += event.size_bytes;
                    }
                }
                _ => {}
            }
        }
        
        memory_by_operator
    }
    
    /// Calculate tensor lifetimes
    fn calculate_tensor_lifetimes(&self) -> HashMap<TensorId, TensorLifetime> {
        let mut tensor_lifetimes = HashMap::new();
        
        // Build a map of allocation events
        let mut allocations = HashMap::new();
        for event in &self.memory_events {
            if event.event_type == MemoryEventType::Allocation {
                if let Some(tensor_id) = event.tensor_id {
                    allocations.insert(tensor_id, event);
                }
            }
        }
        
        // Process deallocation events to create lifetimes
        for event in &self.memory_events {
            if event.event_type == MemoryEventType::Deallocation {
                if let Some(tensor_id) = event.tensor_id {
                    if let Some(alloc_event) = allocations.get(&tensor_id) {
                        let lifetime = TensorLifetime {
                            tensor_id: tensor_id.clone(),
                            allocation_time: alloc_event.timestamp,
                            deallocation_time: Some(event.timestamp),
                            size_bytes: alloc_event.size_bytes,
                            data_type: DataType::Float32, // Placeholder
                            producer_node: alloc_event.node_id,
                            consumer_nodes: Vec::new(), // Would be populated in real implementation
                        };
                        
                        tensor_lifetimes.insert(tensor_id, lifetime);
                    }
                }
            }
        }
        
        // Add lifetimes for tensors that were never deallocated
        for (tensor_id, alloc_event) in allocations {
            if !tensor_lifetimes.contains_key(&tensor_id) {
                let lifetime = TensorLifetime {
                    tensor_id: tensor_id.clone(),
                    allocation_time: alloc_event.timestamp,
                    deallocation_time: None,
                    size_bytes: alloc_event.size_bytes,
                    data_type: DataType::Float32, // Placeholder
                    producer_node: alloc_event.node_id,
                    consumer_nodes: Vec::new(), // Would be populated in real implementation
                };
                
                tensor_lifetimes.insert(tensor_id, lifetime);
            }
        }
        
        tensor_lifetimes
    }
    
    /// Calculate workspace memory usage
    fn calculate_workspace_usage(&self, events: &[ProfileEvent]) -> WorkspaceUsage {
        let mut workspace_events = Vec::new();
        let mut usage_per_operator = HashMap::new();
        
        // In a real implementation, we would have specific events for workspace allocations
        // For this example, we'll create placeholder data
        
        let peak_bytes = 1024 * 1024; // 1 MB placeholder
        
        WorkspaceUsage {
            peak_bytes,
            allocation_events: workspace_events,
            usage_per_operator,
        }
    }
    
    /// Calculate peak memory usage
    fn calculate_peak_memory(&self) -> usize {
        // First check if we already have the peak memory tracked atomically
        let peak_memory_atomic = self.peak_memory_bytes.load(Ordering::Acquire);
        if peak_memory_atomic > 0 {
            return peak_memory_atomic;
        }
        
        // If not available from atomic counter, calculate from events
        let mut current_memory = 0;
        let mut peak_memory = 0;
        
        // Create a HashMap to track allocations by address for accurate tracking
        let mut active_allocations: HashMap<usize, usize> = HashMap::new();
        let mut memory_timeline: Vec<(u64, i64)> = Vec::new(); // (timestamp, delta)
        
        // Sort events by timestamp for accurate sequential analysis
        let mut events = self.memory_events.clone();
        events.sort_by_key(|e| e.timestamp);
        
        // Build a memory timeline
        for event in &events {
            match event.event_type {
                MemoryEventType::Allocation => {
                    // Track allocation by address to handle deallocation properly
                    active_allocations.insert(event.address, event.size_bytes);
                    memory_timeline.push((event.timestamp, event.size_bytes as i64));
                    
                    current_memory += event.size_bytes;
                    peak_memory = peak_memory.max(current_memory);
                },
                MemoryEventType::Deallocation => {
                    // Only subtract if we have a record of this allocation
                    if let Some(size) = active_allocations.remove(&event.address) {
                        memory_timeline.push((event.timestamp, -(size as i64)));
                        current_memory = current_memory.saturating_sub(size);
                    } else {
                        // This can happen if we started profiling after some allocations
                        memory_timeline.push((event.timestamp, -(event.size_bytes as i64)));
                        current_memory = current_memory.saturating_sub(event.size_bytes);
                    }
                },
                MemoryEventType::WorkspaceAllocation => {
                    // Workspace memory is temporary but contributes to peak
                    memory_timeline.push((event.timestamp, event.size_bytes as i64));
                    current_memory += event.size_bytes;
                    peak_memory = peak_memory.max(current_memory);
                    
                    // Assume workspace is freed right away for simplicity
                    // In a real implementation, you'd track workspace deallocation events
                    memory_timeline.push((event.timestamp + 1, -(event.size_bytes as i64)));
                    current_memory = current_memory.saturating_sub(event.size_bytes);
                },
                MemoryEventType::Reuse => {
                    // Memory reuse doesn't affect total usage, but can be tracked for efficiency metrics
                },
                MemoryEventType::PoolGrowth => {
                    // Pool growth represents new memory allocated for the pool
                    memory_timeline.push((event.timestamp, event.size_bytes as i64));
                    current_memory += event.size_bytes;
                    peak_memory = peak_memory.max(current_memory);
                }
            }
        }
        
        // Update the atomic counter with our calculated peak
        self.peak_memory_bytes.store(peak_memory, Ordering::Release);
        
        // Also update current memory for reference
        self.current_memory_bytes.store(current_memory, Ordering::Release);
        
        peak_memory
    }
    
    /// Find tensor reuse opportunities
    fn find_tensor_reuse_opportunities(memory_results: &MemoryProfileResults) -> Vec<TensorReuseOpportunity> {
        let mut opportunities = Vec::new();
        
        // Build tensor lifetime intervals
        let mut tensor_intervals = Vec::new();
        for (tensor_id, lifetime) in &memory_results.tensor_lifetimes {
            if let Some(dealloc_time) = lifetime.deallocation_time {
                tensor_intervals.push((
                    tensor_id.clone(),
                    lifetime.allocation_time,
                    dealloc_time,
                    lifetime.size_bytes,
                ));
            }
        }
        
        // Sort intervals by allocation time
        tensor_intervals.sort_by_key(|i| i.1);
        
        // Check for non-overlapping intervals with similar sizes
        for i in 0..tensor_intervals.len() {
            let (id1, start1, end1, size1) = &tensor_intervals[i];
            
            for j in i+1..tensor_intervals.len() {
                let (id2, start2, end2, size2) = &tensor_intervals[j];
                
                // If tensors have similar sizes and don't overlap in time
                if *size1 == *size2 && (*end1 <= *start2 || *end2 <= *start1) {
                    let opportunity = TensorReuseOpportunity {
                        tensor1_id: id1.clone(),
                        tensor2_id: id2.clone(),
                        potential_savings_bytes: *size1,
                        confidence: 90, // High confidence since sizes match exactly
                    };
                    
                    opportunities.push(opportunity);
                }
                // If tensors have different sizes but one could fit in the other
                else if *size1 > *size2 && *end1 <= *start2 {
                    let opportunity = TensorReuseOpportunity {
                        tensor1_id: id1.clone(),
                        tensor2_id: id2.clone(),
                        potential_savings_bytes: *size2,
                        confidence: 70, // Medium confidence since sizes differ
                    };
                    
                    opportunities.push(opportunity);
                }
                else if *size2 > *size1 && *end2 <= *start1 {
                    let opportunity = TensorReuseOpportunity {
                        tensor1_id: id2.clone(),
                        tensor2_id: id1.clone(),
                        potential_savings_bytes: *size1,
                        confidence: 70, // Medium confidence since sizes differ
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        }
        
        // Sort by potential savings (descending)
        opportunities.sort_by(|a, b| b.potential_savings_bytes.cmp(&a.potential_savings_bytes));
        
        // Return top 10 opportunities
        opportunities.truncate(10);
        
        opportunities
    }
    
    /// Calculate memory fragmentation
    fn calculate_memory_fragmentation(events: &[MemoryEvent], peak_memory: usize) -> f64 {
        // In a real implementation, this would analyze fragmentation patterns
        // For this example, we'll return a placeholder value
        20.0 // 20% fragmentation
    }
    
    /// Generate memory optimization recommendations
    fn generate_memory_recommendations(
        memory_results: &MemoryProfileResults,
        reuse_opportunities: Vec<TensorReuseOpportunity>,
    ) -> Vec<MemoryOptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Add recommendations based on reuse opportunities
        for opportunity in reuse_opportunities.iter().take(3) {
            let recommendation = MemoryOptimizationRecommendation {
                recommendation_type: MemoryOptimizationType::TensorReuse,
                description: format!(
                    "Consider reusing memory between tensors {} and {} to save {} bytes",
                    opportunity.tensor1_id,
                    opportunity.tensor2_id,
                    opportunity.potential_savings_bytes
                ),
                potential_savings_bytes: opportunity.potential_savings_bytes,
                affected_nodes: Vec::new(), // Would be populated in real implementation
                affected_tensors: vec![opportunity.tensor1_id.clone(), opportunity.tensor2_id.clone()],
            };
            
            recommendations.push(recommendation);
        }
        
        // Add in-place operation recommendation (placeholder)
        recommendations.push(MemoryOptimizationRecommendation {
            recommendation_type: MemoryOptimizationType::InPlaceOperation,
            description: "Consider using in-place operations for elementwise operations".to_string(),
            potential_savings_bytes: peak_memory / 10, // Placeholder estimate
            affected_nodes: Vec::new(),
            affected_tensors: Vec::new(),
        });
        
        recommendations
    }
    
    /// Find fusion opportunities in the model
    fn find_fusion_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // In a real implementation, this would analyze the graph for fusion patterns
        // For this example, we'll create a placeholder suggestion
        
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::OperatorFusion,
            description: "Consider fusing Conv+Relu operations for better performance".to_string(),
            affected_nodes: Vec::new(), // Would be populated in real implementation
            estimated_improvement_percent: 15.0,
            confidence: 80,
            implementation_hint: Some(
                "// Example fusion implementation\n\
                 impl FuseConvRelu {\n\
                 \    fn fuse(conv: &ConvNode, relu: &ReluNode) -> FusedConvReluNode {\n\
                 \        // Implementation details\n\
                 \    }\n\
                 }"
                .to_string()
            ),
        });
        
        suggestions
    }
    
    /// Find kernel optimization opportunities
    fn find_kernel_optimization_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // In a real implementation, this would analyze the bottleneck operations for optimization
        // For this example, we'll create a placeholder suggestion
        
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::KernelOptimization,
            description: "Consider optimizing the MatMul kernel with SIMD instructions".to_string(),
            affected_nodes: Vec::new(), // Would be populated in real implementation
            estimated_improvement_percent: 30.0,
            confidence: 75,
            implementation_hint: Some(
                "// Example SIMD optimization\n\
                 #[cfg(target_arch = \"x86_64\")]\n\
                 use std::arch::x86_64::*;\n\
                 \n\
                 fn optimized_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {\n\
                 \    // SIMD implementation\n\
                 }"
                .to_string()
            ),
        });
        
        suggestions
    }
    
    /// Find memory optimization opportunities
    fn find_memory_optimization_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // In a real implementation, this would analyze memory usage patterns
        // For this example, we'll create a placeholder suggestion
        
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::MemoryOptimization,
            description: "Consider using a custom memory pool to reduce allocation overhead".to_string(),
            affected_nodes: Vec::new(), // Would be populated in real implementation
            estimated_improvement_percent: 10.0,
            confidence: 70,
            implementation_hint: Some(
                "// Example memory pool implementation\n\
                 struct MemoryPool {\n\
                 \    chunks: Vec<Vec<u8>>,\n\
                 \    // Implementation details\n\
                 }"
                .to_string()
            ),
        });
        
        suggestions
    }
    
    /// Find parallelization opportunities
    fn find_parallelization_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // In a real implementation, this would analyze the execution graph for parallelization
        // For this example, we'll create a placeholder suggestion
        
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::Parallelization,
            description: "Consider parallelizing batch processing across multiple cores".to_string(),
            affected_nodes: Vec::new(), // Would be populated in real implementation
            estimated_improvement_percent: 25.0,
            confidence: 85,
            implementation_hint: Some(
                "// Example batch parallelization\n\
                 use rayon::prelude::*;\n\
                 \n\
                 fn parallel_batch_process(batches: &[Tensor]) -> Vec<Tensor> {\n\
                 \    batches.par_iter().map(|batch| process_single_batch(batch)).collect()\n\
                 }"
                .to_string()
            ),
        });
        
        suggestions
    }
}

/// Profile the execution of an ONNX model
pub fn profile_model_execution(engine: &mut ExecutionEngine, inputs: &HashMap<String, Tensor>) -> Result<ProfileResults> {
    let mut profiler = Profiler::new(true, true);
    profiler.profile_model_execution(engine, inputs)
}

/// Profile memory usage during model execution
pub fn profile_memory_usage(engine: &mut ExecutionEngine, inputs: &HashMap<String, Tensor>) -> Result<MemoryProfileResults> {
    let mut profiler = Profiler::new(true, false);
    profiler.profile_memory_usage(engine, inputs)
}

/// Record operation execution times
pub fn record_op_execution_times(engine: &mut ExecutionEngine, inputs: &HashMap<String, Tensor>) -> Result<HashMap<NodeId, Duration>> {
    let mut profiler = Profiler::new(false, false);
    profiler.record_op_execution_times(engine, inputs)
}

/// Generate a flamegraph from profiling results
pub fn generate_flamegraph(profile_results: &ProfileResults, output_path: &Path) -> Result<()> {
    let profiler = Profiler::new(false, false);
    profiler.generate_flamegraph(profile_results, output_path)
}

/// Find bottleneck operations in the model
pub fn find_bottleneck_operations(profile_results: &ProfileResults) -> Vec<(NodeId, Duration)> {
    Profiler::find_bottleneck_operations(profile_results)
}

/// Analyze memory efficiency
pub fn analyze_memory_efficiency(memory_results: &MemoryProfileResults) -> MemoryEfficiencyMetrics {
    Profiler::analyze_memory_efficiency(memory_results)
}

/// Suggest optimization opportunities
pub fn suggest_optimization_opportunities(profile_results: &ProfileResults) -> Vec<OptimizationSuggestion> {
    Profiler::suggest_optimization_opportunities(profile_results)
}

/// Convert a DataType to its string representation
pub fn data_type_to_string(data_type: DataType) -> String {
    match data_type {
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Int8 => "Int8".to_string(), 
        DataType::Int16 => "Int16".to_string(),
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Uint8 => "Uint8".to_string(),
        DataType::Uint16 => "Uint16".to_string(),
        DataType::Uint32 => "Uint32".to_string(),
        DataType::Uint64 => "Uint64".to_string(),
        DataType::Bool => "Bool".to_string(),
        DataType::BFloat16 => "BFloat16".to_string(),
        DataType::Float16 => "Float16".to_string(),
        DataType::String => "String".to_string(),
        DataType::Complex64 => "Complex64".to_string(),
        DataType::Complex128 => "Complex128".to_string(),
    }
}

/// Convert a string to a DataType
pub fn data_type_from_string(type_str: &str) -> DataType {
    match type_str {
        "Float32" | "float32" | "FLOAT32" | "float" => DataType::Float32,
        "Float64" | "float64" | "FLOAT64" | "double" => DataType::Float64,
        "Int8" | "int8" | "INT8" => DataType::Int8,
        "Int16" | "int16" | "INT16" => DataType::Int16,
        "Int32" | "int32" | "INT32" => DataType::Int32,
        "Int64" | "int64" | "INT64" => DataType::Int64,
        "Uint8" | "uint8" | "UINT8" => DataType::Uint8,
        "Uint16" | "uint16" | "UINT16" => DataType::Uint16,
        "Uint32" | "uint32" | "UINT32" => DataType::Uint32,
        "Uint64" | "uint64" | "UINT64" => DataType::Uint64,
        "Bool" | "bool" | "BOOL" | "boolean" => DataType::Bool,
        "BFloat16" | "bfloat16" | "BFLOAT16" => DataType::BFloat16,
        "Float16" | "float16" | "FLOAT16" | "half" => DataType::Float16,
        "String" | "string" | "STRING" => DataType::String,
        "Complex64" | "complex64" | "COMPLEX64" => DataType::Complex64,
        "Complex128" | "complex128" | "COMPLEX128" => DataType::Complex128,
        _ => {
            // Handle common abbreviations and synonyms
            if type_str.eq_ignore_ascii_case("f32") { 
                DataType::Float32 
            } else if type_str.eq_ignore_ascii_case("f64") {
                DataType::Float64
            } else if type_str.eq_ignore_ascii_case("i8") {
                DataType::Int8
            } else if type_str.eq_ignore_ascii_case("i16") {
                DataType::Int16
            } else if type_str.eq_ignore_ascii_case("i32") {
                DataType::Int32
            } else if type_str.eq_ignore_ascii_case("i64") {
                DataType::Int64
            } else if type_str.eq_ignore_ascii_case("u8") {
                DataType::Uint8
            } else if type_str.eq_ignore_ascii_case("u16") {
                DataType::Uint16
            } else if type_str.eq_ignore_ascii_case("u32") {
                DataType::Uint32
            } else if type_str.eq_ignore_ascii_case("u64") {
                DataType::Uint64
            } else if type_str.eq_ignore_ascii_case("bf16") {
                DataType::BFloat16
            } else if type_str.eq_ignore_ascii_case("f16") {
                DataType::Float16
            } else {
                // For unknown types, default to Float32
                DataType::Float32
            }
        }
    }
}

/// Calculate memory usage for a tensor based on its data type and dimensions
/// Returns memory usage in bytes
pub fn calculate_tensor_memory_usage(data_type: DataType, dimensions: &[usize]) -> usize {
    // Calculate number of elements
    let element_count: usize = if dimensions.is_empty() {
        1 // Scalar
    } else {
        dimensions.iter().product()
    };
    
    // Memory per element based on data type
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
        DataType::BFloat16 => 2,
        DataType::Float16 => 2,
        DataType::Complex64 => 8,  // 2 * Float32
        DataType::Complex128 => 16, // 2 * Float64
        DataType::String => {
            // For string tensors, this is approximate as actual size depends on string content
            // We use 24 bytes as a reasonable estimate for the average string size
            // (8 bytes for pointer + average 16 bytes per string content)
            24
        }
    };
    
    // Calculate total memory usage
    element_count * bytes_per_element
}

/// Estimate memory requirements for common ONNX operations
/// Returns estimated memory usage in bytes
pub fn estimate_operation_memory(
    op_type: &str, 
    input_shapes: &[Vec<usize>], 
    input_types: &[DataType]
) -> usize {
    if input_shapes.is_empty() || input_types.is_empty() {
        return 0;
    }
    
    match op_type {
        "Conv" => {
            // For Conv, we need input, weights, bias (optional), and output
            // Output size depends on kernel size, padding, etc., but we'll estimate as same size as input
            if input_shapes.len() >= 2 {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let weights_memory = calculate_tensor_memory_usage(input_types[1], &input_shapes[1]);
                
                // Bias is optional
                let bias_memory = if input_shapes.len() > 2 {
                    calculate_tensor_memory_usage(input_types[2], &input_shapes[2])
                } else {
                    0
                };
                
                // Output memory (estimate)
                let output_memory = input_memory;
                
                // Workspace memory for some convolution algorithms
                let workspace_memory = input_memory * 2;
                
                input_memory + weights_memory + bias_memory + output_memory + workspace_memory
            } else {
                0
            }
        },
        "MatMul" => {
            // For MatMul, we have input A, input B, and output C
            if input_shapes.len() >= 2 {
                let a_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let b_memory = calculate_tensor_memory_usage(input_types[1], &input_shapes[1]);
                
                // Estimate output shape (last dim of A × last dim of B)
                let a_shape = &input_shapes[0];
                let b_shape = &input_shapes[1];
                
                if !a_shape.is_empty() && !b_shape.is_empty() {
                    let a_rows = a_shape[a_shape.len() - 2];
                    let b_cols = b_shape[b_shape.len() - 1];
                    
                    // Create output shape by taking batch dimensions from A and matrix dims
                    let mut output_shape = a_shape[..a_shape.len() - 2].to_vec();
                    output_shape.push(a_rows);
                    output_shape.push(b_cols);
                    
                    let output_memory = calculate_tensor_memory_usage(input_types[0], &output_shape);
                    a_memory + b_memory + output_memory
                } else {
                    a_memory + b_memory
                }
            } else {
                0
            }
        },
        "Reshape" => {
            // Reshape doesn't allocate much new memory, just the output tensor
            if !input_shapes.is_empty() {
                calculate_tensor_memory_usage(input_types[0], &input_shapes[0])
            } else {
                0
            }
        },
        "Concat" => {
            // Concat combines all inputs, so sum up input memory and add output
            let mut total_memory = 0;
            for i in 0..input_shapes.len() {
                total_memory += calculate_tensor_memory_usage(
                    if i < input_types.len() { input_types[i] } else { input_types[0] },
                    &input_shapes[i]
                );
            }
            
            // Output is approximately the sum of inputs
            total_memory * 2
        },
        "Gemm" => {
            // Similar to MatMul but with alpha, beta parameters
            if input_shapes.len() >= 2 {
                let a_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let b_memory = calculate_tensor_memory_usage(input_types[1], &input_shapes[1]);
                
                // C is optional for Gemm
                let c_memory = if input_shapes.len() > 2 {
                    calculate_tensor_memory_usage(
                        if 2 < input_types.len() { input_types[2] } else { input_types[0] },
                        &input_shapes[2]
                    )
                } else {
                    0
                };
                
                // Output has shape (A.rows, B.cols)
                let mut output_shape = Vec::new();
                if !input_shapes[0].is_empty() && !input_shapes[1].is_empty() {
                    output_shape.push(input_shapes[0][0]);
                    output_shape.push(input_shapes[1][1]);
                    
                    let output_memory = calculate_tensor_memory_usage(input_types[0], &output_shape);
                    a_memory + b_memory + c_memory + output_memory
                } else {
                    a_memory + b_memory + c_memory
                }
            } else {
                0
            }
        },
        "Pool" | "MaxPool" | "AveragePool" => {
            // Pooling operations have similar memory patterns to Conv
            if !input_shapes.is_empty() {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                
                // Output is roughly the same size as input for memory estimation
                input_memory * 2
            } else {
                0
            }
        },
        "BatchNormalization" => {
            // Batch norm has input, scale, bias, mean, variance
            if input_shapes.len() >= 5 {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let scale_memory = calculate_tensor_memory_usage(
                    if 1 < input_types.len() { input_types[1] } else { input_types[0] },
                    &input_shapes[1]
                );
                let bias_memory = calculate_tensor_memory_usage(
                    if 2 < input_types.len() { input_types[2] } else { input_types[0] },
                    &input_shapes[2]
                );
                let mean_memory = calculate_tensor_memory_usage(
                    if 3 < input_types.len() { input_types[3] } else { input_types[0] },
                    &input_shapes[3]
                );
                let var_memory = calculate_tensor_memory_usage(
                    if 4 < input_types.len() { input_types[4] } else { input_types[0] },
                    &input_shapes[4]
                );
                
                // Output is same size as input
                let output_memory = input_memory;
                
                input_memory + scale_memory + bias_memory + mean_memory + var_memory + output_memory
            } else if !input_shapes.is_empty() {
                // Simplified estimate if we don't have all parameters
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                input_memory * 3 // Input, intermediate calculations, and output
            } else {
                0
            }
        },
        // For element-wise operations and activations
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | 
        "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" | "Softmax" => {
            // These typically have input and same-sized output
            if !input_shapes.is_empty() {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                input_memory * 2 // Input + output
            } else {
                0
            }
        },
        // Transpose doesn't change memory requirements much
        "Transpose" => {
            if !input_shapes.is_empty() {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                input_memory * 2 // Input + output
            } else {
                0
            }
        },
        // For other operations, make a reasonable estimate
        _ => {
            // Sum the memory of all inputs
            let mut total_input_memory = 0;
            for i in 0..input_shapes.len() {
                let data_type = if i < input_types.len() { input_types[i] } else { input_types[0] };
                total_input_memory += calculate_tensor_memory_usage(data_type, &input_shapes[i]);
            }
            
            // Estimate output as roughly same size as all inputs combined
            total_input_memory * 2
        }
    }
}

/// Parse tensor shape from a string like "[1,3,224,224]" or "(1, 3, 224, 224)"
fn parse_tensor_shape(shape_str: &str) -> Result<Vec<usize>> {
    // Remove brackets, parentheses, etc.
    let clean_str = shape_str
        .trim()
        .trim_start_matches(&['[', '(', '{'])
        .trim_end_matches(&[']', ')', '}']);
    
    // Split by commas and parse each dimension
    let dimensions: Result<Vec<usize>> = clean_str
        .split(',')
        .map(|dim| {
            dim.trim().parse::<usize>().map_err(|e| {
                Error::ExecutionError(format!("Failed to parse tensor dimension '{}': {}", dim, e))
            })
        })
        .collect();
    
    dimensions
}

/// Export profiling data to various formats
pub fn export_profile_data(profile_results: &ProfileResults, format: ExportFormat) -> Result<Vec<u8>> {
    Profiler::export_profile_data(profile_results, format)
}

/// Profile concurrent execution performance to analyze parallelism
pub fn profile_concurrent_execution(
    engine: &mut ExecutionEngine, 
    inputs: &HashMap<String, Tensor>,
    num_threads: usize,
) -> Result<HashMap<usize, Duration>> {
    use std::thread;
    
    // Store execution time for different thread counts
    let mut thread_times = HashMap::new();
    
    // Test with different numbers of threads
    for threads in 1..=num_threads {
        let start = Instant::now();
        
        // Clone the engine for each thread
        let mut handles = Vec::new();
        
        for _ in 0..threads {
            let inputs_clone = inputs.clone();
            let mut engine_clone = engine.clone_for_parallel_execution()?;
            
            let handle = thread::spawn(move || {
                let _ = engine_clone.run(inputs_clone);
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }
        
        let duration = start.elapsed();
        thread_times.insert(threads, duration);
    }
    
    Ok(thread_times)
}