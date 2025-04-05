// Profiler module
//
// This module provides comprehensive profiling capabilities for ONNX models,
// including performance measurement, memory usage tracking, and optimization
// recommendations.

mod types;
mod core;
mod analysis;
mod recommendations;
mod export;
mod utils;

// Re-export public API
pub use types::{
    ProfileEventType, ProfileEvent, ProfileResults, 
    PerformanceStats, MemoryProfileResults, MemoryEvent, MemoryEventType,
    TensorLifetime, ParallelismStats, TensorStats, ModelInfo, GraphInfo,
    EdgeType, OptimizationType, ExportFormat,
};
pub use core::{EventCollector, ComprehensiveProfiler};
pub use analysis::{
    find_bottleneck_operations,
    analyze_memory_efficiency,
    compute_parallelism_stats,
};
pub use recommendations::{
    OptimizationSuggestion,
    MemoryOptimizationRecommendation,
    suggest_optimization_opportunities,
    find_tensor_reuse_opportunities,
};
pub use export::{
    export_profile_data,
    generate_flamegraph,
};

// Public API functions
pub use core::{
    profile_model_execution,
    profile_memory_usage,
    record_op_execution_times,
};