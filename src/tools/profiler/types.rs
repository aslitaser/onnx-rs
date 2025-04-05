// Types module for profiler
// Contains data structure definitions for profile events and results

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};

use serde::{Serialize, Deserialize};

use crate::{
    model::{NodeId, TensorId},
    ops::tensor::DataType,
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
    /// Detailed critical path analysis report
    #[serde(skip_serializing_if = "Option::is_none")]
    pub critical_path_report: Option<String>,
    /// Optimization impact scores for critical nodes
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub optimization_impact_scores: HashMap<NodeId, OptimizationImpact>,
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

/// Type of edge in the execution graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Data dependency (one node produces data consumed by another)
    DataDependency,
    /// Control dependency (one node must execute before another)
    ControlDependency,
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

/// Optimization impact score for a node on the critical path
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Node ID
    pub node_id: NodeId,
    /// Impact score (higher means optimizing this node has more impact)
    pub impact_score: f64,
    /// Percentage of total execution time
    pub percentage_of_total_time: f64,
    /// Potential speedup if this node were optimized by 50%
    pub potential_speedup: f64,
    /// Number of dependent nodes affected by this node
    pub dependent_node_count: usize,
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
    /// Algorithmic optimization
    AlgorithmicOptimization,
    /// Memory management improvements
    MemoryManagement,
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

// Helper struct for execution graph analysis
#[derive(Debug, Clone)]
pub(crate) struct TimedNode {
    pub node_id: NodeId,
    pub execution_time: Duration,
    pub incoming_edges: Vec<(NodeId, EdgeType)>,
    pub outgoing_edges: Vec<(NodeId, EdgeType)>,
    pub earliest_start_time: Option<Duration>,
    pub earliest_completion_time: Option<Duration>,
    pub latest_start_time: Option<Duration>,
    pub latest_completion_time: Option<Duration>,
    pub slack: Option<Duration>,
}

// Edge in the execution graph
#[derive(Debug, Clone)]
pub(crate) struct ExecutionEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub data: Option<TensorId>,
}