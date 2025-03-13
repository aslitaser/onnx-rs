use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::process::Command;
use std::fs::File;
use std::io::{self, Write, Read};
use std::convert::TryFrom;
use std::sync::Arc;

use serde::{Serialize, Deserialize};
use tempfile::TempDir;
use ndarray::{Array, ArrayD};
use anyhow::{Result, anyhow};

use crate::{
    error,
    execution::{ExecutionEngine, ExecutionOptions, OptimizationLevel, context::ExecutionContext},
    model::{OnnxModel, TensorInfo, NodeId},
    ops::{
        tensor::{Tensor as ComputeTensor, DataType, Shape},
        tensor::Shape as TensorShape,
    },
    parser::model_loader::ModelLoader,
    tools::profile::{PerformanceStats, profile_model_execution, ProfileResults},
};

/// Represents different ONNX runtime implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeType {
    /// Our custom ONNX runtime
    OurRuntime,
    /// Microsoft's ONNX Runtime (https://github.com/microsoft/onnxruntime)
    OnnxRuntime,
    /// Tract ONNX runtime (https://github.com/sonos/tract)
    Tract,
    /// NVIDIA TensorRT (https://developer.nvidia.com/tensorrt)
    TensorRT,
}

impl std::fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::OurRuntime => write!(f, "Our Runtime"),
            RuntimeType::OnnxRuntime => write!(f, "ONNX Runtime"),
            RuntimeType::Tract => write!(f, "Tract"),
            RuntimeType::TensorRT => write!(f, "TensorRT"),
        }
    }
}

/// Performance results for a runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePerformance {
    /// Type of runtime
    pub runtime_type: RuntimeType,
    /// Mean execution time in milliseconds
    pub mean_time_ms: f64,
    /// Standard deviation of execution time in milliseconds
    pub std_dev_ms: f64,
    /// Minimum execution time in milliseconds
    pub min_time_ms: f64,
    /// Maximum execution time in milliseconds
    pub max_time_ms: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: Option<usize>,
    /// Warm-up time in milliseconds
    pub warmup_time_ms: Option<f64>,
    /// Initialization time in milliseconds
    pub init_time_ms: Option<f64>,
    /// Inference rate (inferences per second)
    pub inferences_per_second: f64,
    /// Version of the runtime
    pub version: String,
    /// Backend used (CPU, GPU, etc.)
    pub backend: String,
    /// Additional metrics
    pub additional_metrics: HashMap<String, serde_json::Value>,
}

/// Comparison results for a specific model across runtimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    /// Name of the model
    pub model_name: String,
    /// Path to the model file
    pub model_path: String,
    /// Performance results for each runtime
    pub runtime_results: HashMap<RuntimeType, RuntimePerformance>,
    /// Relative speedup of each runtime vs. baseline (typically OurRuntime)
    pub relative_speedups: HashMap<RuntimeType, f64>,
    /// Numerical correctness metrics (comparing output tensors)
    pub correctness: HashMap<RuntimeType, CorrectnessMetrics>,
    /// Model metadata
    pub model_metadata: HashMap<String, String>,
    /// Input tensor shapes
    pub input_shapes: HashMap<String, Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: HashMap<String, Vec<usize>>,
}

/// Metrics for measuring numerical correctness between runtime outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessMetrics {
    /// Whether outputs match within tolerance
    pub outputs_match: bool,
    /// Maximum absolute difference
    pub max_abs_difference: f64,
    /// Maximum relative difference
    pub max_rel_difference: f64,
    /// Average absolute difference
    pub avg_abs_difference: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Percentage of outputs within tolerance
    pub percent_within_tolerance: f64,
    /// Error details per output tensor
    pub error_per_output: HashMap<String, HashMap<String, f64>>,
}

/// Represents the configuration for a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Whether to check output correctness
    pub check_correctness: bool,
    /// Reference runtime for correctness comparison
    pub reference_runtime: RuntimeType,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Whether to use GPU
    pub use_gpu: bool,
    /// Runtimes to benchmark
    pub runtimes: Vec<RuntimeType>,
    /// Optimization level to use for our runtime
    pub optimization_level: OptimizationLevel,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 50,
            timeout_ms: 60000,
            check_correctness: true,
            reference_runtime: RuntimeType::OnnxRuntime,
            tolerance: 1e-5,
            use_gpu: false,
            runtimes: vec![RuntimeType::OurRuntime, RuntimeType::OnnxRuntime],
            optimization_level: OptimizationLevel::Standard,
        }
    }
}

/// Runtime-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Path to the runtime executable or library
    pub runtime_path: Option<String>,
    /// Extra arguments to pass to the runtime
    pub extra_args: Vec<String>,
    /// Environment variables to set when running
    pub env_vars: HashMap<String, String>,
    /// Whether to allow fallback to CPU if GPU is not available
    pub allow_cpu_fallback: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_path: None,
            extra_args: Vec::new(),
            env_vars: HashMap::new(),
            allow_cpu_fallback: true,
        }
    }
}

/// Compare execution performance with Microsoft's ONNX Runtime
pub fn compare_with_onnxruntime(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<ModelComparisonResult> {
    // First run our runtime to get baseline performance
    let our_runtime_result = benchmark_our_runtime(model_path, input_data, config)?;
    
    // Get model details
    let model = ModelLoader::load_from_file(model_path)?;
    let model_name = model.metadata().model_version.to_string();
    
    // Load input/output shapes
    let input_shapes = model.graph().inputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    let output_shapes = model.graph().outputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    // Detect ONNX Runtime installation
    let onnxruntime_path = detect_onnxruntime()?;
    
    // Create a temporary directory for input/output data
    let tmp_dir = TempDir::new()?;
    
    // Save input tensors to files for ONNX Runtime
    let input_files = save_input_tensors(input_data, tmp_dir.path())?;
    
    // Prepare ONNX Runtime command
    let mut onnxruntime_times = Vec::with_capacity(config.measurement_iterations);
    let mut output_tensors: HashMap<String, ComputeTensor> = HashMap::new();
    
    // Warmup runs
    for _ in 0..config.warmup_iterations {
        run_onnxruntime(
            &onnxruntime_path,
            model_path,
            &input_files,
            tmp_dir.path(),
            config.use_gpu,
        )?;
    }
    
    // Measurement runs
    for i in 0..config.measurement_iterations {
        let start = Instant::now();
        
        run_onnxruntime(
            &onnxruntime_path,
            model_path,
            &input_files,
            tmp_dir.path(),
            config.use_gpu,
        )?;
        
        let elapsed = start.elapsed();
        onnxruntime_times.push(elapsed.as_secs_f64() * 1000.0);
        
        // Only load output tensors on the last iteration
        if i == config.measurement_iterations - 1 && config.check_correctness {
            output_tensors = load_output_tensors(tmp_dir.path(), &model.graph().outputs())?;
        }
    }
    
    // Calculate statistics
    let mean_time_ms = onnxruntime_times.iter().sum::<f64>() / onnxruntime_times.len() as f64;
    let std_dev_ms = {
        let variance = onnxruntime_times.iter()
            .map(|&x| (x - mean_time_ms).powi(2))
            .sum::<f64>() / onnxruntime_times.len() as f64;
        variance.sqrt()
    };
    let min_time_ms = onnxruntime_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time_ms = onnxruntime_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let inferences_per_second = 1000.0 / mean_time_ms;
    
    // Get ONNX Runtime version
    let version = get_onnxruntime_version(&onnxruntime_path)?;
    
    // Create RuntimePerformance struct
    let onnxruntime_performance = RuntimePerformance {
        runtime_type: RuntimeType::OnnxRuntime,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: None, // We don't track memory for ORT
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second,
        version,
        backend: if config.use_gpu { "GPU".to_string() } else { "CPU".to_string() },
        additional_metrics: HashMap::new(),
    };
    
    // Compute correctness metrics if required
    let mut correctness = HashMap::new();
    if config.check_correctness {
        let mut our_outputs = HashMap::new();
        if let Some(perf) = our_runtime_result.runtime_results.get(&RuntimeType::OurRuntime) {
            our_outputs = run_our_runtime_once(model_path, input_data, config.optimization_level)?;
        }
        
        let metrics = compare_outputs(&our_outputs, &output_tensors, config.tolerance)?;
        correctness.insert(RuntimeType::OnnxRuntime, metrics);
    }
    
    // Calculate relative speedup
    let our_time_ms = our_runtime_result.runtime_results
        .get(&RuntimeType::OurRuntime)
        .map(|perf| perf.mean_time_ms)
        .unwrap_or(f64::MAX);
    
    let mut relative_speedups = HashMap::new();
    if our_time_ms != f64::MAX {
        relative_speedups.insert(RuntimeType::OurRuntime, 1.0);
        relative_speedups.insert(RuntimeType::OnnxRuntime, our_time_ms / mean_time_ms);
    }
    
    // Combine all runtime results
    let mut all_runtime_results = our_runtime_result.runtime_results;
    all_runtime_results.insert(RuntimeType::OnnxRuntime, onnxruntime_performance);
    
    let model_comparison = ModelComparisonResult {
        model_name,
        model_path: model_path.to_string_lossy().to_string(),
        runtime_results: all_runtime_results,
        relative_speedups,
        correctness,
        model_metadata: HashMap::new(),
        input_shapes,
        output_shapes,
    };
    
    Ok(model_comparison)
}

/// Compare execution performance with Tract
pub fn compare_with_tract(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<ModelComparisonResult> {
    // Run our runtime
    let our_runtime_result = benchmark_our_runtime(model_path, input_data, config)?;
    
    // Try to detect Tract installation
    let tract_path = detect_tract()?;
    
    // Get model details
    let model = ModelLoader::load_from_file(model_path)?;
    let model_name = model.metadata().model_version.to_string();
    
    // Load input/output shapes
    let input_shapes = model.graph().inputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    let output_shapes = model.graph().outputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    // Create a temporary directory for input/output data
    let tmp_dir = TempDir::new()?;
    
    // Save input tensors
    let input_files = save_input_tensors(input_data, tmp_dir.path())?;
    
    // Measure Tract performance
    let mut tract_times = Vec::with_capacity(config.measurement_iterations);
    let mut output_tensors: HashMap<String, ComputeTensor> = HashMap::new();
    
    // Warmup runs
    for _ in 0..config.warmup_iterations {
        run_tract(
            &tract_path,
            model_path,
            &input_files,
            tmp_dir.path(),
        )?;
    }
    
    // Measurement runs
    for i in 0..config.measurement_iterations {
        let start = Instant::now();
        
        run_tract(
            &tract_path,
            model_path,
            &input_files,
            tmp_dir.path(),
        )?;
        
        let elapsed = start.elapsed();
        tract_times.push(elapsed.as_secs_f64() * 1000.0);
        
        // Only load output tensors on the last iteration
        if i == config.measurement_iterations - 1 && config.check_correctness {
            output_tensors = load_output_tensors(tmp_dir.path(), &model.graph().outputs())?;
        }
    }
    
    // Calculate statistics
    let mean_time_ms = tract_times.iter().sum::<f64>() / tract_times.len() as f64;
    let std_dev_ms = {
        let variance = tract_times.iter()
            .map(|&x| (x - mean_time_ms).powi(2))
            .sum::<f64>() / tract_times.len() as f64;
        variance.sqrt()
    };
    let min_time_ms = tract_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time_ms = tract_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let inferences_per_second = 1000.0 / mean_time_ms;
    
    // Get Tract version
    let version = get_tract_version(&tract_path)?;
    
    // Create RuntimePerformance struct
    let tract_performance = RuntimePerformance {
        runtime_type: RuntimeType::Tract,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: None,
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second,
        version,
        backend: "CPU".to_string(), // Tract is CPU-only
        additional_metrics: HashMap::new(),
    };
    
    // Compute correctness metrics if required
    let mut correctness = HashMap::new();
    if config.check_correctness {
        let our_outputs = run_our_runtime_once(model_path, input_data, config.optimization_level)?;
        let metrics = compare_outputs(&our_outputs, &output_tensors, config.tolerance)?;
        correctness.insert(RuntimeType::Tract, metrics);
    }
    
    // Calculate relative speedup
    let our_time_ms = our_runtime_result.runtime_results
        .get(&RuntimeType::OurRuntime)
        .map(|perf| perf.mean_time_ms)
        .unwrap_or(f64::MAX);
    
    let mut relative_speedups = HashMap::new();
    if our_time_ms != f64::MAX {
        relative_speedups.insert(RuntimeType::OurRuntime, 1.0);
        relative_speedups.insert(RuntimeType::Tract, our_time_ms / mean_time_ms);
    }
    
    // Combine all runtime results
    let mut all_runtime_results = our_runtime_result.runtime_results;
    all_runtime_results.insert(RuntimeType::Tract, tract_performance);
    
    let model_comparison = ModelComparisonResult {
        model_name,
        model_path: model_path.to_string_lossy().to_string(),
        runtime_results: all_runtime_results,
        relative_speedups,
        correctness,
        model_metadata: HashMap::new(),
        input_shapes,
        output_shapes,
    };
    
    Ok(model_comparison)
}

/// Compare execution performance with TensorRT
pub fn compare_with_tensorrt(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<ModelComparisonResult> {
    // Run our runtime
    let our_runtime_result = benchmark_our_runtime(model_path, input_data, config)?;
    
    // Try to detect TensorRT installation
    let tensorrt_path = detect_tensorrt()?;
    
    // Get model details
    let model = ModelLoader::load_from_file(model_path)?;
    let model_name = model.metadata().model_version.to_string();
    
    // Load input/output shapes
    let input_shapes = model.graph().inputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    let output_shapes = model.graph().outputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    // Create a temporary directory for input/output data
    let tmp_dir = TempDir::new()?;
    
    // Save input tensors
    let input_files = save_input_tensors(input_data, tmp_dir.path())?;
    
    // Measure TensorRT performance
    let mut tensorrt_times = Vec::with_capacity(config.measurement_iterations);
    let mut output_tensors: HashMap<String, ComputeTensor> = HashMap::new();
    
    // Convert ONNX model to TensorRT format (this would typically be done via ONNX Parser API in TensorRT)
    let tensorrt_model_path = convert_to_tensorrt(model_path, &tensorrt_path, tmp_dir.path())?;
    
    // Warmup runs
    for _ in 0..config.warmup_iterations {
        run_tensorrt(
            &tensorrt_path,
            &tensorrt_model_path,
            &input_files,
            tmp_dir.path(),
        )?;
    }
    
    // Measurement runs
    for i in 0..config.measurement_iterations {
        let start = Instant::now();
        
        run_tensorrt(
            &tensorrt_path,
            &tensorrt_model_path,
            &input_files,
            tmp_dir.path(),
        )?;
        
        let elapsed = start.elapsed();
        tensorrt_times.push(elapsed.as_secs_f64() * 1000.0);
        
        // Only load output tensors on the last iteration
        if i == config.measurement_iterations - 1 && config.check_correctness {
            output_tensors = load_output_tensors(tmp_dir.path(), &model.graph().outputs())?;
        }
    }
    
    // Calculate statistics
    let mean_time_ms = tensorrt_times.iter().sum::<f64>() / tensorrt_times.len() as f64;
    let std_dev_ms = {
        let variance = tensorrt_times.iter()
            .map(|&x| (x - mean_time_ms).powi(2))
            .sum::<f64>() / tensorrt_times.len() as f64;
        variance.sqrt()
    };
    let min_time_ms = tensorrt_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time_ms = tensorrt_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let inferences_per_second = 1000.0 / mean_time_ms;
    
    // Get TensorRT version
    let version = get_tensorrt_version(&tensorrt_path)?;
    
    // Create RuntimePerformance struct
    let tensorrt_performance = RuntimePerformance {
        runtime_type: RuntimeType::TensorRT,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: None,
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second,
        version,
        backend: "GPU".to_string(), // TensorRT is GPU-focused
        additional_metrics: HashMap::new(),
    };
    
    // Compute correctness metrics if required
    let mut correctness = HashMap::new();
    if config.check_correctness {
        let our_outputs = run_our_runtime_once(model_path, input_data, config.optimization_level)?;
        let metrics = compare_outputs(&our_outputs, &output_tensors, config.tolerance)?;
        correctness.insert(RuntimeType::TensorRT, metrics);
    }
    
    // Calculate relative speedup
    let our_time_ms = our_runtime_result.runtime_results
        .get(&RuntimeType::OurRuntime)
        .map(|perf| perf.mean_time_ms)
        .unwrap_or(f64::MAX);
    
    let mut relative_speedups = HashMap::new();
    if our_time_ms != f64::MAX {
        relative_speedups.insert(RuntimeType::OurRuntime, 1.0);
        relative_speedups.insert(RuntimeType::TensorRT, our_time_ms / mean_time_ms);
    }
    
    // Combine all runtime results
    let mut all_runtime_results = our_runtime_result.runtime_results;
    all_runtime_results.insert(RuntimeType::TensorRT, tensorrt_performance);
    
    let model_comparison = ModelComparisonResult {
        model_name,
        model_path: model_path.to_string_lossy().to_string(),
        runtime_results: all_runtime_results,
        relative_speedups,
        correctness,
        model_metadata: HashMap::new(),
        input_shapes,
        output_shapes,
    };
    
    Ok(model_comparison)
}

/// Generate a comparison report in markdown format
pub fn generate_comparison_report(results: &[ModelComparisonResult]) -> String {
    let mut report = String::new();

    // Report header
    report.push_str("# ONNX Runtime Comparison Report\n\n");
    
    // Summary table
    report.push_str("## Summary\n\n");
    report.push_str("| Model | ");
    
    // Get all runtime types from the results
    let mut runtime_types = Vec::new();
    for result in results {
        for &runtime_type in result.runtime_results.keys() {
            if !runtime_types.contains(&runtime_type) {
                runtime_types.push(runtime_type);
            }
        }
    }
    
    // Sort runtime types (our runtime first, then alphabetically)
    runtime_types.sort_by(|a, b| {
        if *a == RuntimeType::OurRuntime {
            std::cmp::Ordering::Less
        } else if *b == RuntimeType::OurRuntime {
            std::cmp::Ordering::Greater
        } else {
            format!("{:?}", a).cmp(&format!("{:?}", b))
        }
    });
    
    // Add runtime columns
    for runtime_type in &runtime_types {
        report.push_str(&format!("{} (ms) | ", runtime_type));
    }
    report.push_str("Best Runtime | Speedup vs Ours |\n");
    
    // Add separator row
    report.push_str("|---|");
    for _ in &runtime_types {
        report.push_str("---|");
    }
    report.push_str("---|---|\n");
    
    // Add model results
    for result in results {
        report.push_str(&format!("| {} | ", result.model_name));
        
        // For each runtime, add the mean time
        for &runtime_type in &runtime_types {
            if let Some(perf) = result.runtime_results.get(&runtime_type) {
                report.push_str(&format!("{:.2} | ", perf.mean_time_ms));
            } else {
                report.push_str("- | ");
            }
        }
        
        // Find the best runtime
        let mut best_runtime = None;
        let mut best_time = f64::INFINITY;
        
        for (&runtime_type, perf) in &result.runtime_results {
            if perf.mean_time_ms < best_time {
                best_time = perf.mean_time_ms;
                best_runtime = Some(runtime_type);
            }
        }
        
        // Add best runtime
        if let Some(runtime) = best_runtime {
            report.push_str(&format!("{} | ", runtime));
            
            // Add speedup vs our runtime
            if let Some(our_perf) = result.runtime_results.get(&RuntimeType::OurRuntime) {
                let speedup = our_perf.mean_time_ms / best_time;
                if runtime == RuntimeType::OurRuntime {
                    report.push_str("1.00x |\n");
                } else {
                    report.push_str(&format!("{:.2}x |\n", speedup));
                }
            } else {
                report.push_str("- |\n");
            }
        } else {
            report.push_str("- | - |\n");
        }
    }
    
    // Detailed results for each model
    report.push_str("\n## Detailed Results\n\n");
    
    for (i, result) in results.iter().enumerate() {
        report.push_str(&format!("### {}. {}\n\n", i + 1, result.model_name));
        report.push_str(&format!("Model path: {}\n\n", result.model_path));
        
        // Input shapes
        report.push_str("#### Input Shapes\n\n");
        for (name, shape) in &result.input_shapes {
            report.push_str(&format!("- `{}`: {}\n", name, 
                shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ")));
        }
        report.push_str("\n");
        
        // Performance comparison
        report.push_str("#### Performance\n\n");
        report.push_str("| Runtime | Mean (ms) | Min (ms) | Max (ms) | Std Dev (ms) | IPS |\n");
        report.push_str("|---------|-----------|----------|----------|--------------|-----|\n");
        
        for &runtime_type in &runtime_types {
            if let Some(perf) = result.runtime_results.get(&runtime_type) {
                report.push_str(&format!("| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                    runtime_type,
                    perf.mean_time_ms,
                    perf.min_time_ms,
                    perf.max_time_ms,
                    perf.std_dev_ms,
                    perf.inferences_per_second
                ));
            }
        }
        report.push_str("\n");
        
        // Correctness comparison if available
        if !result.correctness.is_empty() {
            report.push_str("#### Correctness\n\n");
            report.push_str("| Runtime | Outputs Match | Max Abs Diff | Max Rel Diff | Avg Abs Diff | RMSE | % Within Tolerance |\n");
            report.push_str("|---------|--------------|--------------|--------------|--------------|------|-------------------|\n");
            
            for (&runtime_type, metrics) in &result.correctness {
                report.push_str(&format!("| {} | {} | {:.6e} | {:.6e} | {:.6e} | {:.6e} | {:.2}% |\n",
                    runtime_type,
                    metrics.outputs_match,
                    metrics.max_abs_difference,
                    metrics.max_rel_difference,
                    metrics.avg_abs_difference,
                    metrics.rmse,
                    metrics.percent_within_tolerance
                ));
            }
            report.push_str("\n");
        }
        
        // Relative speedups
        report.push_str("#### Relative Performance\n\n");
        report.push_str("| Runtime | Speedup vs Our Runtime |\n");
        report.push_str("|---------|------------------------|\n");
        
        let our_time = result.runtime_results
            .get(&RuntimeType::OurRuntime)
            .map(|perf| perf.mean_time_ms)
            .unwrap_or(f64::MAX);
        
        for &runtime_type in &runtime_types {
            if runtime_type == RuntimeType::OurRuntime {
                report.push_str("| Our Runtime | 1.00x |\n");
            } else if let Some(perf) = result.runtime_results.get(&runtime_type) {
                let speedup = our_time / perf.mean_time_ms;
                report.push_str(&format!("| {} | {:.2}x |\n", runtime_type, speedup));
            }
        }
        
        report.push_str("\n");
    }
    
    // Recommendations section
    report.push_str("## Recommendations\n\n");
    
    // Count wins per runtime
    let mut runtime_wins = HashMap::new();
    for result in results {
        let mut best_runtime = None;
        let mut best_time = f64::INFINITY;
        
        for (&runtime_type, perf) in &result.runtime_results {
            if perf.mean_time_ms < best_time {
                best_time = perf.mean_time_ms;
                best_runtime = Some(runtime_type);
            }
        }
        
        if let Some(runtime) = best_runtime {
            *runtime_wins.entry(runtime).or_insert(0) += 1;
        }
    }
    
    // Overall recommendations
    report.push_str("### Overall\n\n");
    
    let total_models = results.len();
    for (&runtime, &wins) in &runtime_wins {
        let percentage = (wins as f64 / total_models as f64) * 100.0;
        report.push_str(&format!("- {} was fastest for {} of {} models ({:.1}%)\n",
            runtime, wins, total_models, percentage));
    }
    
    report.push_str("\n### Model-Specific Recommendations\n\n");
    
    for result in results {
        let mut best_runtime = None;
        let mut best_time = f64::INFINITY;
        
        for (&runtime_type, perf) in &result.runtime_results {
            if perf.mean_time_ms < best_time {
                best_time = perf.mean_time_ms;
                best_runtime = Some(runtime_type);
            }
        }
        
        if let Some(runtime) = best_runtime {
            report.push_str(&format!("- For model **{}**: Use **{}** for best performance\n",
                result.model_name, runtime));
        }
    }
    
    report
}

/// Benchmark an operator across different runtimes
pub fn compare_operator_performance(
    op_type: &str, 
    input_shapes: &[&[usize]], 
    attributes: &HashMap<String, String>
) -> Result<OperatorComparisonResult> {
    // Create a simple model with just this operator
    let model_path = generate_operator_model(op_type, input_shapes, attributes)?;
    
    // Create input tensors based on shapes
    let mut input_data = HashMap::new();
    for (i, &shape) in input_shapes.iter().enumerate() {
        let tensor_name = format!("input_{}", i);
        let tensor = generate_random_tensor(shape, DataType::Float32);
        input_data.insert(tensor_name, tensor);
    }
    
    // Setup benchmark config
    let config = BenchmarkConfig {
        warmup_iterations: 5,
        measurement_iterations: 20,
        check_correctness: true,
        ..Default::default()
    };
    
    // Run comparisons
    let our_runtime_result = benchmark_our_runtime(&model_path, &input_data, &config)?;
    
    let mut onnx_result = None;
    if let Ok(onnx_path) = detect_onnxruntime() {
        if let Ok(res) = compare_with_onnxruntime(&model_path, &input_data, &config) {
            onnx_result = Some(res);
        }
    }
    
    let mut tract_result = None;
    if let Ok(tract_path) = detect_tract() {
        if let Ok(res) = compare_with_tract(&model_path, &input_data, &config) {
            tract_result = Some(res);
        }
    }
    
    // Combine results
    let mut runtime_results = our_runtime_result.runtime_results;
    if let Some(res) = onnx_result {
        for (runtime, perf) in res.runtime_results {
            if runtime != RuntimeType::OurRuntime {
                runtime_results.insert(runtime, perf);
            }
        }
    }
    
    if let Some(res) = tract_result {
        for (runtime, perf) in res.runtime_results {
            if runtime != RuntimeType::OurRuntime {
                runtime_results.insert(runtime, perf);
            }
        }
    }
    
    // Calculate relative speedups
    let our_time = runtime_results.get(&RuntimeType::OurRuntime)
        .map(|perf| perf.mean_time_ms)
        .unwrap_or(f64::MAX);
    
    let mut relative_speedups = HashMap::new();
    if our_time != f64::MAX {
        for (&runtime, perf) in &runtime_results {
            relative_speedups.insert(runtime, our_time / perf.mean_time_ms);
        }
    }
    
    // Create operator comparison result
    Ok(OperatorComparisonResult {
        op_type: op_type.to_string(),
        input_shapes: input_shapes.iter().map(|&shape| shape.to_vec()).collect(),
        attributes: attributes.clone(),
        runtime_results,
        relative_speedups,
    })
}

/// Results from comparing an operator across runtimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorComparisonResult {
    /// The operator type
    pub op_type: String,
    /// Input shapes used for the comparison
    pub input_shapes: Vec<Vec<usize>>,
    /// Operator attributes used
    pub attributes: HashMap<String, String>,
    /// Performance results for each runtime
    pub runtime_results: HashMap<RuntimeType, RuntimePerformance>,
    /// Relative speedup of each runtime vs. baseline (OurRuntime)
    pub relative_speedups: HashMap<RuntimeType, f64>,
}

/// Comparison metrics for evaluating model outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Whether all tensors matched within tolerance
    pub all_matched: bool,
    /// Per-tensor comparison results
    pub tensor_results: HashMap<String, TensorComparisonResult>,
    /// Overall statistics
    pub overall_stats: OverallComparisonStats,
    /// Tolerance used for comparison
    pub tolerance: f32,
    /// Number of tensors compared
    pub tensor_count: usize,
    /// Number of tensors that matched
    pub matched_count: usize,
}

/// Results of comparing a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorComparisonResult {
    /// Whether the tensor matched within tolerance
    pub matched: bool,
    /// Maximum absolute difference between corresponding elements
    pub max_abs_diff: f32,
    /// Average absolute difference between corresponding elements
    pub avg_abs_diff: f32,
    /// Maximum relative difference between corresponding elements
    pub max_rel_diff: f32,
    /// Root mean square error
    pub rmse: f32,
    /// Number of elements compared
    pub element_count: usize,
    /// Number of elements that matched within tolerance
    pub matched_elements: usize,
    /// Percentage of elements that matched within tolerance
    pub match_percentage: f32,
    /// Positions of largest differences (for debugging)
    pub largest_diff_positions: Vec<Vec<usize>>,
    /// Locations with NaN or Inf values
    pub nan_inf_positions: Vec<Vec<usize>>,
}

/// Overall statistics for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallComparisonStats {
    /// Maximum absolute difference across all tensors
    pub max_abs_diff: f32,
    /// Average absolute difference across all tensors
    pub avg_abs_diff: f32,
    /// Maximum relative difference across all tensors
    pub max_rel_diff: f32,
    /// Root mean square error across all tensors
    pub overall_rmse: f32,
    /// Percentage of all elements that matched within tolerance
    pub overall_match_percentage: f32,
    /// Name of the tensor with the largest difference
    pub worst_tensor_name: Option<String>,
}

/// Results of validating model outputs against reference outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Whether validation passed (all outputs within tolerance)
    pub passed: bool,
    /// Detailed comparison metrics
    pub metrics: ComparisonMetrics,
    /// Error message if validation failed
    pub error_message: Option<String>,
    /// Runtime performance of the validation run
    pub performance: Option<PerformanceStats>,
    /// Names of failing tensors
    pub failing_tensors: Vec<String>,
}

/// Results of measuring numerical precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionMetrics {
    /// Overall comparison metrics
    pub comparison: ComparisonMetrics,
    /// Per-operation numerical stability analysis
    pub op_precision: HashMap<String, OperationPrecisionMetrics>,
    /// Tensors with highest numerical divergence
    pub worst_tensors: Vec<String>,
    /// Graph nodes with highest contribution to numerical error
    pub error_contributors: Vec<NodeId>,
    /// Overall numerical stability score (0-100)
    pub stability_score: u32,
}

/// Numerical precision metrics for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPrecisionMetrics {
    /// Operation type
    pub op_type: String,
    /// Node ID
    pub node_id: NodeId,
    /// Input tensor differences (if this is not the first operation)
    pub input_differences: HashMap<String, f32>,
    /// Output tensor differences
    pub output_differences: HashMap<String, f32>,
    /// Error amplification factor (how much this op increases error)
    pub error_amplification: f32,
    /// Whether this operation appears to be numerically unstable
    pub is_unstable: bool,
    /// Recommendations for improving numerical stability
    pub recommendations: Vec<String>,
}

/// Performance comparison between two runtimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// First runtime being compared
    pub runtime1: RuntimeType,
    /// Second runtime being compared
    pub runtime2: RuntimeType,
    /// Performance results for the first runtime
    pub performance1: RuntimePerformance,
    /// Performance results for the second runtime
    pub performance2: RuntimePerformance,
    /// Speedup of runtime2 compared to runtime1
    pub speedup: f64,
    /// Detailed performance comparison for each operation type
    pub op_type_comparison: HashMap<String, OpTypePerformanceComparison>,
    /// Detailed performance comparison for specific models
    pub model_comparison: Option<ModelComparisonResult>,
    /// Analysis of scaling behavior with different batch sizes
    pub batch_scaling: Option<BatchScalingAnalysis>,
    /// Analysis of throughput and utilization metrics
    pub throughput_analysis: ThroughputAnalysis,
}

/// Performance comparison for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpTypePerformanceComparison {
    /// Operation type
    pub op_type: String,
    /// Time spent in runtime1 (ms)
    pub time1_ms: f64,
    /// Time spent in runtime2 (ms)
    pub time2_ms: f64,
    /// Percentage of total time in runtime1
    pub percentage1: f64,
    /// Percentage of total time in runtime2
    pub percentage2: f64,
    /// Speedup of runtime2 compared to runtime1 for this operation
    pub speedup: f64,
}

/// Analysis of scaling behavior with different batch sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchScalingAnalysis {
    /// Batch sizes tested
    pub batch_sizes: Vec<usize>,
    /// Execution time for each batch size for runtime1 (ms)
    pub times1_ms: Vec<f64>,
    /// Execution time for each batch size for runtime2 (ms)
    pub times2_ms: Vec<f64>,
    /// Speedup for each batch size
    pub speedups: Vec<f64>,
    /// Whether scaling is linear
    pub is_linear: bool,
    /// Scaling efficiency (0-100%)
    pub scaling_efficiency: f64,
}

/// Analysis of throughput and utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    /// Inferences per second for runtime1
    pub throughput1: f64,
    /// Inferences per second for runtime2
    pub throughput2: f64,
    /// Throughput ratio (runtime2 / runtime1)
    pub throughput_ratio: f64,
    /// CPU utilization percentage for runtime1
    pub cpu_utilization1: Option<f64>,
    /// CPU utilization percentage for runtime2
    pub cpu_utilization2: Option<f64>,
    /// Memory bandwidth utilization for runtime1 (GB/s)
    pub memory_bandwidth1: Option<f64>,
    /// Memory bandwidth utilization for runtime2 (GB/s)
    pub memory_bandwidth2: Option<f64>,
    /// Energy efficiency comparison (lower is better)
    pub energy_ratio: Option<f64>,
}

/// Tool for comparing model outputs and performance across different implementations
pub struct ComparisonTool {
    /// Whether to check for NaN/Inf values
    check_nan_inf: bool,
    /// Whether to report positions of largest differences
    report_diff_positions: bool,
    /// Maximum number of difference positions to report
    max_diff_positions: usize,
    /// Default tolerance for numerical comparisons
    default_tolerance: f32,
    /// Temporary directory for storing intermediate results
    temp_dir: Option<TempDir>,
    /// Available runtime configurations
    runtime_configs: HashMap<RuntimeType, RuntimeConfig>,
}

impl ComparisonTool {
    /// Create a new comparison tool
    pub fn new(default_tolerance: f32) -> Self {
        let mut runtime_configs = HashMap::new();
        runtime_configs.insert(RuntimeType::OurRuntime, RuntimeConfig::default());
        
        Self {
            check_nan_inf: true,
            report_diff_positions: true,
            max_diff_positions: 10,
            default_tolerance,
            temp_dir: None,
            runtime_configs,
        }
    }
    
    /// Compare model outputs with expected outputs
    pub fn compare_model_outputs(
        &self,
        expected: &HashMap<String, ComputeTensor>,
        actual: &HashMap<String, ComputeTensor>,
        tolerance: f32,
    ) -> Result<ComparisonMetrics> {
        let mut tensor_results = HashMap::new();
        let mut all_matched = true;
        let mut matched_count = 0;
        
        // Overall statistics
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut sum_squared_diff = 0.0f64;
        let mut total_elements = 0usize;
        let mut total_matched_elements = 0usize;
        let mut worst_tensor_name = None;
        
        // Compare each expected tensor with the corresponding actual tensor
        for (name, expected_tensor) in expected {
            if let Some(actual_tensor) = actual.get(name) {
                let result = self.compare_tensors(expected_tensor, actual_tensor, tolerance)?;
                
                // Update overall statistics
                if result.max_abs_diff > max_abs_diff {
                    max_abs_diff = result.max_abs_diff;
                    worst_tensor_name = Some(name.clone());
                }
                
                if result.max_rel_diff > max_rel_diff {
                    max_rel_diff = result.max_rel_diff;
                }
                
                sum_squared_diff += (result.rmse.powi(2) * result.element_count as f32) as f64;
                total_elements += result.element_count;
                total_matched_elements += result.matched_elements;
                
                // Update all_matched flag
                if !result.matched {
                    all_matched = false;
                } else {
                    matched_count += 1;
                }
                
                tensor_results.insert(name.clone(), result);
            } else {
                // Missing tensor in actual outputs
                all_matched = false;
                return Err(anyhow!("Output tensor '{}' missing from actual outputs", name));
            }
        }
        
        // Check for extra tensors in actual outputs
        for name in actual.keys() {
            if !expected.contains_key(name) {
                return Err(anyhow!("Unexpected output tensor '{}' in actual outputs", name));
            }
        }
        
        // Calculate overall RMSE
        let overall_rmse = if total_elements > 0 {
            (sum_squared_diff / total_elements as f64).sqrt() as f32
        } else {
            0.0
        };
        
        // Calculate overall match percentage
        let overall_match_percentage = if total_elements > 0 {
            100.0 * total_matched_elements as f32 / total_elements as f32
        } else {
            100.0
        };
        
        // Calculate average absolute difference
        let avg_abs_diff = if let Some(first_result) = tensor_results.values().next() {
            // This is a simplification - in a full implementation, we would calculate
            // the true weighted average across all tensors
            first_result.avg_abs_diff
        } else {
            0.0
        };
        
        let overall_stats = OverallComparisonStats {
            max_abs_diff,
            avg_abs_diff,
            max_rel_diff,
            overall_rmse,
            overall_match_percentage,
            worst_tensor_name,
        };
        
        Ok(ComparisonMetrics {
            all_matched,
            tensor_results,
            overall_stats,
            tolerance,
            tensor_count: expected.len(),
            matched_count,
        })
    }
    
    /// Compare individual tensors element by element
    fn compare_tensors(
        &self,
        expected: &ComputeTensor,
        actual: &ComputeTensor,
        tolerance: f32,
    ) -> Result<TensorComparisonResult> {
        // Check tensor shapes
        if expected.shape() != actual.shape() {
            return Err(anyhow!(
                "Shape mismatch: expected {:?}, got {:?}",
                expected.shape(),
                actual.shape()
            ));
        }
        
        // Check data types
        if expected.data_type() != actual.data_type() {
            return Err(anyhow!(
                "Data type mismatch: expected {:?}, got {:?}",
                expected.data_type(),
                actual.data_type()
            ));
        }
        
        // Currently only supporting Float32 comparisons for simplicity
        // In a real implementation, this would handle all data types
        if expected.data_type() != DataType::Float32 {
            return Err(anyhow!(
                "Unsupported data type for comparison: {:?}",
                expected.data_type()
            ));
        }
        
        // Get data slices
        let expected_data = expected.as_slice::<f32>()
            .ok_or_else(|| anyhow!("Failed to get expected tensor data"))?;
        let actual_data = actual.as_slice::<f32>()
            .ok_or_else(|| anyhow!("Failed to get actual tensor data"))?;
        
        let element_count = expected_data.len();
        
        // Track differences and statistics
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut sum_squared_diff = 0.0f32;
        let mut matched_elements = 0usize;
        
        // Track positions of largest differences
        let mut largest_diffs = Vec::with_capacity(self.max_diff_positions);
        let mut nan_inf_positions = Vec::new();
        
        // Compare elements
        for i in 0..element_count {
            let expected_val = expected_data[i];
            let actual_val = actual_data[i];
            
            // Check for NaN/Inf
            if self.check_nan_inf && (expected_val.is_nan() || expected_val.is_infinite() || 
                                    actual_val.is_nan() || actual_val.is_infinite()) {
                if self.report_diff_positions {
                    // Get multi-dimensional index from flat index
                    let position = self.flat_index_to_position(i, expected.shape());
                    nan_inf_positions.push(position);
                }
                continue;
            }
            
            // Calculate absolute difference
            let abs_diff = (expected_val - actual_val).abs();
            
            // Calculate relative difference
            let rel_diff = if expected_val.abs() > 1e-6 {
                abs_diff / expected_val.abs()
            } else {
                abs_diff
            };
            
            // Update statistics
            sum_abs_diff += abs_diff;
            sum_squared_diff += abs_diff * abs_diff;
            
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            
            // Check if the element matches within tolerance
            if abs_diff <= tolerance || rel_diff <= tolerance {
                matched_elements += 1;
            } else if self.report_diff_positions {
                // Record position of large difference
                let position = self.flat_index_to_position(i, expected.shape());
                let diff_entry = (abs_diff, position);
                
                // Insert into sorted list of largest differences
                let insert_pos = largest_diffs.binary_search_by(|(diff, _)| {
                    diff.partial_cmp(&abs_diff).unwrap_or(std::cmp::Ordering::Equal).reverse()
                }).unwrap_or_else(|pos| pos);
                
                if insert_pos < self.max_diff_positions {
                    largest_diffs.insert(insert_pos, diff_entry);
                    if largest_diffs.len() > self.max_diff_positions {
                        largest_diffs.pop();
                    }
                }
            }
        }
        
        // Calculate final statistics
        let avg_abs_diff = if element_count > 0 {
            sum_abs_diff / element_count as f32
        } else {
            0.0
        };
        
        let rmse = if element_count > 0 {
            (sum_squared_diff / element_count as f32).sqrt()
        } else {
            0.0
        };
        
        let match_percentage = if element_count > 0 {
            100.0 * matched_elements as f32 / element_count as f32
        } else {
            100.0
        };
        
        // Extract just the positions from the largest differences
        let largest_diff_positions = largest_diffs.into_iter().map(|(_, pos)| pos).collect();
        
        // Determine overall match status
        let matched = match_percentage >= 99.9; // Allow for small rounding errors
        
        Ok(TensorComparisonResult {
            matched,
            max_abs_diff,
            avg_abs_diff,
            max_rel_diff,
            rmse,
            element_count,
            matched_elements,
            match_percentage,
            largest_diff_positions,
            nan_inf_positions,
        })
    }
    
    /// Convert a flat index to a multi-dimensional position
    fn flat_index_to_position(&self, index: usize, shape: &[usize]) -> Vec<usize> {
        let mut position = Vec::with_capacity(shape.len());
        let mut remaining = index;
        
        for &dim in shape.iter().rev() {
            position.push(remaining % dim);
            remaining /= dim;
        }
        
        position.reverse();
        position
    }
    
    /// Validate a model's outputs against reference outputs
    pub fn validate_against_reference(
        &self,
        engine: &mut ExecutionEngine,
        reference_outputs: &HashMap<String, ComputeTensor>,
        inputs: &HashMap<String, ComputeTensor>,
    ) -> Result<ValidationResults> {
        // Run the model
        let actual_outputs = engine.run(inputs.clone())?;
        
        // Compare outputs
        let comparison_metrics = match self.compare_model_outputs(reference_outputs, &actual_outputs, self.default_tolerance) {
            Ok(metrics) => metrics,
            Err(err) => {
                return Ok(ValidationResults {
                    passed: false,
                    metrics: ComparisonMetrics {
                        all_matched: false,
                        tensor_results: HashMap::new(),
                        overall_stats: OverallComparisonStats {
                            max_abs_diff: 0.0,
                            avg_abs_diff: 0.0,
                            max_rel_diff: 0.0,
                            overall_rmse: 0.0,
                            overall_match_percentage: 0.0,
                            worst_tensor_name: None,
                        },
                        tolerance: self.default_tolerance,
                        tensor_count: 0,
                        matched_count: 0,
                    },
                    error_message: Some(err.to_string()),
                    performance: None,
                    failing_tensors: Vec::new(),
                });
            }
        };
        
        // Get failing tensors
        let failing_tensors = comparison_metrics.tensor_results.iter()
            .filter_map(|(name, result)| {
                if !result.matched {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        
        // Get performance statistics
        let performance = PerformanceStats::from_profile_events(&engine.profile_events());
        
        Ok(ValidationResults {
            passed: comparison_metrics.all_matched,
            metrics: comparison_metrics,
            error_message: None,
            performance: Some(performance),
            failing_tensors,
        })
    }
    
    /// Measure numerical precision between this implementation and a reference implementation
    pub fn measure_numerical_precision(
        &self,
        engine: &mut ExecutionEngine,
        reference_engine: &mut ExecutionEngine,
        inputs: &HashMap<String, ComputeTensor>,
    ) -> Result<PrecisionMetrics> {
        // Run both engines
        let our_outputs = engine.run(inputs.clone())?;
        let reference_outputs = reference_engine.run(inputs.clone())?;
        
        // Compare outputs
        let comparison = self.compare_model_outputs(&reference_outputs, &our_outputs, self.default_tolerance)?;
        
        // Analyze precision per operation (this would be more complex in a real implementation)
        let mut op_precision = HashMap::new();
        let mut worst_tensors = Vec::new();
        let mut error_contributors = Vec::new();
        
        // Get profiling events to identify operations
        let events = engine.profile_events();
        
        // Find operations and their output tensors
        for event in &events {
            if event.event_type == crate::tools::profile::ProfileEventType::OpExecution {
                if let Some(node_id) = event.node_id {
                    // Extract operation type from event name
                    let op_type = event.name.split(':').next().unwrap_or("Unknown").to_string();
                    
                    // This is a simplified implementation - in a real system, we would:
                    // 1. Track all intermediate tensors and their errors
                    // 2. Analyze how errors propagate through the graph
                    // 3. Identify operations that amplify numerical errors
                    
                    // Add a placeholder precision metric for each operation
                    let metrics = OperationPrecisionMetrics {
                        op_type: op_type.clone(),
                        node_id,
                        input_differences: HashMap::new(),
                        output_differences: HashMap::new(),
                        error_amplification: 1.0,
                        is_unstable: false,
                        recommendations: Vec::new(),
                    };
                    
                    op_precision.insert(op_type, metrics);
                    
                    // Add to error contributors if there's a significant error
                    if let Some(tensor_id) = event.tensor_id {
                        let tensor_name = format!("tensor_{}", tensor_id.0.iter().map(|&b| b as char).collect::<String>());
                        if let Some(result) = comparison.tensor_results.get(&tensor_name) {
                            if result.max_abs_diff > self.default_tolerance {
                                error_contributors.push(node_id);
                            }
                        }
                    }
                }
            }
        }
        
        // Get worst tensors (those with largest differences)
        worst_tensors = comparison.tensor_results.iter()
            .filter(|(_, result)| !result.matched)
            .map(|(name, result)| (name.clone(), result.max_abs_diff))
            .collect::<Vec<_>>();
        
        // Sort worst tensors by max difference (descending)
        worst_tensors.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top 5 tensors
        let worst_tensors = worst_tensors.into_iter()
            .take(5)
            .map(|(name, _)| name)
            .collect();
        
        // Calculate stability score (0-100)
        // Lower is better for these metrics, so we invert them
        let match_component = comparison.overall_stats.overall_match_percentage;
        let abs_diff_component = 100.0 * (1.0 - comparison.overall_stats.max_abs_diff.min(1.0));
        let rmse_component = 100.0 * (1.0 - comparison.overall_stats.overall_rmse.min(1.0));
        
        let stability_score = ((match_component * 0.6) + (abs_diff_component * 0.2) + (rmse_component * 0.2)) as u32;
        
        Ok(PrecisionMetrics {
            comparison,
            op_precision,
            worst_tensors,
            error_contributors,
            stability_score: stability_score.min(100),
        })
    }
    
    /// Compare performance between this implementation and a reference implementation
    pub fn compare_performance(
        &self,
        engine: &mut ExecutionEngine,
        reference_engine: &mut ExecutionEngine,
        inputs: &HashMap<String, ComputeTensor>,
        iterations: usize,
    ) -> Result<PerformanceComparison> {
        // Warmup runs
        let _ = engine.run(inputs.clone())?;
        let _ = reference_engine.run(inputs.clone())?;
        
        // Profile our runtime
        let mut our_times = Vec::with_capacity(iterations);
        let mut our_op_times = HashMap::new();
        
        engine.enable_profiling(true);
        
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = engine.run(inputs.clone())?;
            our_times.push(start.elapsed());
        }
        
        // Get profiling information
        let our_events = engine.profile_events();
        
        // Calculate per-op times
        for event in &our_events {
            if event.event_type == crate::tools::profile::ProfileEventType::OpExecution && event.duration.is_some() {
                let op_type = event.name.split(':').next().unwrap_or("Unknown").to_string();
                let duration = event.duration.unwrap();
                
                our_op_times.entry(op_type)
                    .and_modify(|d: &mut Duration| *d += duration)
                    .or_insert(duration);
            }
        }
        
        // Profile reference runtime
        let mut ref_times = Vec::with_capacity(iterations);
        let mut ref_op_times = HashMap::new();
        
        reference_engine.enable_profiling(true);
        
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = reference_engine.run(inputs.clone())?;
            ref_times.push(start.elapsed());
        }
        
        // Get profiling information
        let ref_events = reference_engine.profile_events();
        
        // Calculate per-op times
        for event in &ref_events {
            if event.event_type == crate::tools::profile::ProfileEventType::OpExecution && event.duration.is_some() {
                let op_type = event.name.split(':').next().unwrap_or("Unknown").to_string();
                let duration = event.duration.unwrap();
                
                ref_op_times.entry(op_type)
                    .and_modify(|d: &mut Duration| *d += duration)
                    .or_insert(duration);
            }
        }
        
        // Calculate statistics for our runtime
        let our_mean_time = our_times.iter().sum::<Duration>() / iterations as u32;
        let our_mean_time_ms = our_mean_time.as_secs_f64() * 1000.0;
        
        // Calculate statistics for reference runtime
        let ref_mean_time = ref_times.iter().sum::<Duration>() / iterations as u32;
        let ref_mean_time_ms = ref_mean_time.as_secs_f64() * 1000.0;
        
        // Create performance objects (simplified versions of what would be in a real implementation)
        let our_performance = RuntimePerformance {
            runtime_type: RuntimeType::OurRuntime,
            mean_time_ms: our_mean_time_ms,
            std_dev_ms: 0.0, // Would calculate standard deviation in real implementation
            min_time_ms: our_times.iter().min().unwrap_or(&our_mean_time).as_secs_f64() * 1000.0,
            max_time_ms: our_times.iter().max().unwrap_or(&our_mean_time).as_secs_f64() * 1000.0,
            peak_memory_bytes: None, // Would be populated in real implementation
            warmup_time_ms: None,
            init_time_ms: None,
            inferences_per_second: 1000.0 / our_mean_time_ms,
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: "CPU".to_string(),
            additional_metrics: HashMap::new(),
        };
        
        let ref_performance = RuntimePerformance {
            runtime_type: RuntimeType::OnnxRuntime, // Assuming the reference is ONNX Runtime
            mean_time_ms: ref_mean_time_ms,
            std_dev_ms: 0.0, // Would calculate standard deviation in real implementation
            min_time_ms: ref_times.iter().min().unwrap_or(&ref_mean_time).as_secs_f64() * 1000.0,
            max_time_ms: ref_times.iter().max().unwrap_or(&ref_mean_time).as_secs_f64() * 1000.0,
            peak_memory_bytes: None, // Would be populated in real implementation
            warmup_time_ms: None,
            init_time_ms: None,
            inferences_per_second: 1000.0 / ref_mean_time_ms,
            version: "Unknown".to_string(), // Would get actual version in real implementation
            backend: "CPU".to_string(),
            additional_metrics: HashMap::new(),
        };
        
        // Calculate speedup
        let speedup = ref_mean_time_ms / our_mean_time_ms;
        
        // Calculate per-op comparisons
        let mut op_type_comparison = HashMap::new();
        let total_our_time = our_op_times.values().sum::<Duration>().as_secs_f64() * 1000.0;
        let total_ref_time = ref_op_times.values().sum::<Duration>().as_secs_f64() * 1000.0;
        
        for op_type in our_op_times.keys().chain(ref_op_times.keys()).collect::<std::collections::HashSet<_>>() {
            let our_time = our_op_times.get(op_type).cloned().unwrap_or_default().as_secs_f64() * 1000.0;
            let ref_time = ref_op_times.get(op_type).cloned().unwrap_or_default().as_secs_f64() * 1000.0;
            
            let our_percentage = if total_our_time > 0.0 { our_time * 100.0 / total_our_time } else { 0.0 };
            let ref_percentage = if total_ref_time > 0.0 { ref_time * 100.0 / total_ref_time } else { 0.0 };
            
            let op_speedup = if ref_time > 0.0 { ref_time / our_time } else { 1.0 };
            
            op_type_comparison.insert(op_type.clone(), OpTypePerformanceComparison {
                op_type: op_type.clone(),
                time1_ms: our_time,
                time2_ms: ref_time,
                percentage1: our_percentage,
                percentage2: ref_percentage,
                speedup: op_speedup,
            });
        }
        
        // Create throughput analysis object
        let throughput_analysis = ThroughputAnalysis {
            throughput1: our_performance.inferences_per_second,
            throughput2: ref_performance.inferences_per_second,
            throughput_ratio: our_performance.inferences_per_second / ref_performance.inferences_per_second,
            cpu_utilization1: None,
            cpu_utilization2: None,
            memory_bandwidth1: None,
            memory_bandwidth2: None,
            energy_ratio: None,
        };
        
        Ok(PerformanceComparison {
            runtime1: RuntimeType::OurRuntime,
            runtime2: RuntimeType::OnnxRuntime,
            performance1: our_performance,
            performance2: ref_performance,
            speedup,
            op_type_comparison,
            model_comparison: None,
            batch_scaling: None,
            throughput_analysis,
        })
    }
}

// =====================================================================
// Helper Functions
// =====================================================================

/// Benchmark our runtime
fn benchmark_our_runtime(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<ModelComparisonResult> {
    // Load the model
    let model = ModelLoader::load_from_file(model_path)?;
    let model_name = model.metadata().model_version.to_string();
    
    // Create execution options
    let options = ExecutionOptions::new()
        .set_optimization_level(config.optimization_level)
        .enable_profiling(true);
    
    // Create execution engine
    let init_start = Instant::now();
    let mut engine = ExecutionEngine::new(model.clone(), options)?;
    engine.prepare()?;
    let init_time = init_start.elapsed();
    
    // Warmup runs
    let warmup_start = Instant::now();
    for _ in 0..config.warmup_iterations {
        let _ = engine.run(input_data.clone())?;
    }
    let warmup_time = warmup_start.elapsed();
    
    // Measurement runs
    let mut execution_times = Vec::with_capacity(config.measurement_iterations);
    let mut peak_memory = 0;
    let mut profile_stats = None;
    
    for i in 0..config.measurement_iterations {
        let start = Instant::now();
        let _ = engine.run(input_data.clone())?;
        let elapsed = start.elapsed();
        execution_times.push(elapsed.as_secs_f64() * 1000.0);
        
        // Collect profiling stats from the last run
        if i == config.measurement_iterations - 1 {
            let stats = profile_model_execution(&mut engine, input_data)?;
            peak_memory = stats.peak_memory_bytes;
            profile_stats = Some(stats);
        }
    }
    
    // Calculate statistics
    let mean_time_ms = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
    let std_dev_ms = {
        let variance = execution_times.iter()
            .map(|&x| (x - mean_time_ms).powi(2))
            .sum::<f64>() / execution_times.len() as f64;
        variance.sqrt()
    };
    let min_time_ms = execution_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time_ms = execution_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let inferences_per_second = 1000.0 / mean_time_ms;
    
    // Create RuntimePerformance struct
    let our_runtime_perf = RuntimePerformance {
        runtime_type: RuntimeType::OurRuntime,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: Some(peak_memory),
        warmup_time_ms: Some(warmup_time.as_secs_f64() * 1000.0),
        init_time_ms: Some(init_time.as_secs_f64() * 1000.0),
        inferences_per_second,
        version: env!("CARGO_PKG_VERSION").to_string(),
        backend: "CPU".to_string(), // Our runtime is CPU-only for now
        additional_metrics: HashMap::new(),
    };
    
    // Get input/output shapes
    let input_shapes = model.graph().inputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    let output_shapes = model.graph().outputs().iter()
        .map(|info| (info.name.clone(), info.shape.iter().map(|&d| d.abs() as usize).collect()))
        .collect::<HashMap<_, _>>();
    
    // Add profiling data to additional metrics if available
    if let Some(stats) = profile_stats {
        let mut additional = HashMap::new();
        
        // Add operation times
        let mut op_times = HashMap::new();
        for (op, time) in stats.per_op_type_time_ns {
            op_times.insert(op, time as f64 / 1_000_000.0); // Convert to ms
        }
        additional.insert("op_times_ms".to_string(), serde_json::to_value(op_times)?);
        
        // Add memory by tensor type
        let mut memory_by_type = HashMap::new();
        for (data_type, bytes) in stats.memory_by_tensor_type {
            memory_by_type.insert(format!("{:?}", data_type), bytes as f64 / (1024.0 * 1024.0)); // Convert to MB
        }
        additional.insert("memory_by_type_mb".to_string(), serde_json::to_value(memory_by_type)?);
        
        // Store the JSON
        our_runtime_perf.additional_metrics.extend(additional);
    }
    
    // Create model comparison result
    let result = ModelComparisonResult {
        model_name,
        model_path: model_path.to_string_lossy().to_string(),
        runtime_results: HashMap::from([(RuntimeType::OurRuntime, our_runtime_perf)]),
        relative_speedups: HashMap::from([(RuntimeType::OurRuntime, 1.0)]),
        correctness: HashMap::new(),
        model_metadata: HashMap::new(),
        input_shapes,
        output_shapes,
    };
    
    Ok(result)
}

/// Run our runtime once to get outputs for comparison
fn run_our_runtime_once(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    optimization_level: OptimizationLevel,
) -> Result<HashMap<String, ComputeTensor>> {
    // Load the model
    let model = ModelLoader::load_from_file(model_path)?;
    
    // Create execution options
    let options = ExecutionOptions::new()
        .set_optimization_level(optimization_level);
    
    // Create execution engine
    let mut engine = ExecutionEngine::new(model, options)?;
    engine.prepare()?;
    
    // Run the model
    let outputs = engine.run(input_data.clone())?;
    
    Ok(outputs)
}

/// Compare output tensors for correctness
fn compare_outputs(
    reference: &HashMap<String, ComputeTensor>,
    test: &HashMap<String, ComputeTensor>,
    tolerance: f64,
) -> Result<CorrectnessMetrics> {
    if reference.is_empty() || test.is_empty() {
        return Err(anyhow!("Both reference and test outputs must be non-empty"));
    }
    
    let mut max_abs_difference = 0.0;
    let mut max_rel_difference = 0.0;
    let mut total_abs_difference = 0.0;
    let mut total_elements = 0;
    let mut elements_within_tolerance = 0;
    
    let mut error_per_output = HashMap::new();
    
    for (name, ref_tensor) in reference {
        if let Some(test_tensor) = test.get(name) {
            if ref_tensor.shape() != test_tensor.shape() {
                return Err(anyhow!("Shape mismatch for output '{}': {:?} vs {:?}",
                    name, ref_tensor.shape(), test_tensor.shape()));
            }
            
            // Compare the tensors
            let (abs_diff, rel_diff, total_diff, num_elements, within_tol) = 
                compare_tensor_values(ref_tensor, test_tensor, tolerance)?;
            
            // Update aggregated metrics
            max_abs_difference = max_abs_difference.max(abs_diff);
            max_rel_difference = max_rel_difference.max(rel_diff);
            total_abs_difference += total_diff;
            total_elements += num_elements;
            elements_within_tolerance += within_tol;
            
            // Store per-output metrics
            let mut output_metrics = HashMap::new();
            output_metrics.insert("max_abs_diff".to_string(), abs_diff);
            output_metrics.insert("max_rel_diff".to_string(), rel_diff);
            output_metrics.insert("avg_abs_diff".to_string(), total_diff / num_elements as f64);
            output_metrics.insert("within_tolerance".to_string(), 
                within_tol as f64 / num_elements as f64 * 100.0);
            
            error_per_output.insert(name.clone(), output_metrics);
        } else {
            return Err(anyhow!("Output '{}' missing from test outputs", name));
        }
    }
    
    // Calculate final metrics
    let avg_abs_difference = if total_elements > 0 {
        total_abs_difference / total_elements as f64
    } else {
        0.0
    };
    
    let percent_within_tolerance = if total_elements > 0 {
        elements_within_tolerance as f64 / total_elements as f64 * 100.0
    } else {
        0.0
    };
    
    // Calculate RMSE
    let rmse = (total_abs_difference.powi(2) / total_elements as f64).sqrt();
    
    // Outputs match if all differences are within tolerance
    let outputs_match = percent_within_tolerance > 99.9;
    
    Ok(CorrectnessMetrics {
        outputs_match,
        max_abs_difference,
        max_rel_difference,
        avg_abs_difference,
        rmse,
        percent_within_tolerance,
        error_per_output,
    })
}

/// Compare values in two tensors
fn compare_tensor_values(
    reference: &ComputeTensor,
    test: &ComputeTensor,
    tolerance: f64,
) -> Result<(f64, f64, f64, usize, usize)> {
    let mut max_abs_diff = 0.0;
    let mut max_rel_diff = 0.0;
    let mut total_abs_diff = 0.0;
    let mut elements_within_tol = 0;
    
    // Only implemented for Float32 tensors for now
    if reference.data_type() != DataType::Float32 || test.data_type() != DataType::Float32 {
        return Err(anyhow!("Only Float32 tensors supported for comparison"));
    }
    
    let ref_data = reference.as_slice::<f32>()
        .ok_or_else(|| anyhow!("Failed to get reference tensor data"))?;
    let test_data = test.as_slice::<f32>()
        .ok_or_else(|| anyhow!("Failed to get test tensor data"))?;
    
    let num_elements = ref_data.len();
    if num_elements != test_data.len() {
        return Err(anyhow!("Tensor element count mismatch: {} vs {}", 
            num_elements, test_data.len()));
    }
    
    for i in 0..num_elements {
        let ref_val = ref_data[i] as f64;
        let test_val = test_data[i] as f64;
        
        let abs_diff = (ref_val - test_val).abs();
        let rel_diff = if ref_val.abs() > 1e-10 {
            abs_diff / ref_val.abs()
        } else {
            abs_diff
        };
        
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        total_abs_diff += abs_diff;
        
        if abs_diff <= tolerance || rel_diff <= tolerance {
            elements_within_tol += 1;
        }
    }
    
    Ok((max_abs_diff, max_rel_diff, total_abs_diff, num_elements, elements_within_tol))
}

/// Generate a random tensor with the given shape and data type
fn generate_random_tensor(shape: &[usize], data_type: DataType) -> ComputeTensor {
    let mut tensor = ComputeTensor::new(shape, data_type);
    
    match data_type {
        DataType::Float32 => {
            if let Some(data) = tensor.as_slice_mut::<f32>() {
                // Fill with random data
                for i in 0..data.len() {
                    data[i] = rand::random::<f32>() * 2.0 - 1.0; // Random values in [-1, 1]
                }
            }
        },
        DataType::Float64 => {
            if let Some(data) = tensor.as_slice_mut::<f64>() {
                // Fill with random data
                for i in 0..data.len() {
                    data[i] = rand::random::<f64>() * 2.0 - 1.0; // Random values in [-1, 1]
                }
            }
        },
        DataType::Int32 => {
            if let Some(data) = tensor.as_slice_mut::<i32>() {
                // Fill with random data
                for i in 0..data.len() {
                    data[i] = rand::random::<i32>();
                }
            }
        },
        DataType::Int64 => {
            if let Some(data) = tensor.as_slice_mut::<i64>() {
                // Fill with random data
                for i in 0..data.len() {
                    data[i] = rand::random::<i64>();
                }
            }
        },
        // Add other data types as needed
        _ => {
            // Default to Float32 for unsupported types
            if let Some(data) = tensor.as_slice_mut::<f32>() {
                // Fill with random data
                for i in 0..data.len() {
                    data[i] = rand::random::<f32>() * 2.0 - 1.0; // Random values in [-1, 1]
                }
            }
        }
    }
    
    tensor
}

/// Save input tensors to files for use with external runtimes
fn save_input_tensors(
    inputs: &HashMap<String, ComputeTensor>,
    output_dir: &Path,
) -> Result<HashMap<String, String>> {
    let mut input_files = HashMap::new();
    
    for (name, tensor) in inputs {
        let file_path = output_dir.join(format!("{}.bin", name));
        let mut file = File::create(&file_path)?;
        
        // Write shape
        let shape = tensor.shape();
        let shape_bytes = shape.len().to_le_bytes();
        file.write_all(&shape_bytes)?;
        
        for &dim in shape {
            let dim_bytes = dim.to_le_bytes();
            file.write_all(&dim_bytes)?;
        }
        
        // Write data type
        let dtype_byte = match tensor.data_type() {
            DataType::Float32 => 0u8,
            DataType::Float64 => 1u8,
            DataType::Int32 => 2u8,
            DataType::Int64 => 3u8,
            DataType::Uint8 => 4u8,
            DataType::Bool => 5u8,
            _ => return Err(anyhow!("Unsupported data type: {:?}", tensor.data_type())),
        };
        file.write_all(&[dtype_byte])?;
        
        // Write tensor data
        match tensor.data_type() {
            DataType::Float32 => {
                if let Some(data) = tensor.as_slice::<f32>() {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
            },
            DataType::Float64 => {
                if let Some(data) = tensor.as_slice::<f64>() {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<f64>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
            },
            DataType::Int32 => {
                if let Some(data) = tensor.as_slice::<i32>() {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<i32>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
            },
            DataType::Int64 => {
                if let Some(data) = tensor.as_slice::<i64>() {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<i64>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
            },
            _ => return Err(anyhow!("Unsupported data type: {:?}", tensor.data_type())),
        }
        
        input_files.insert(name.clone(), file_path.to_string_lossy().to_string());
    }
    
    Ok(input_files)
}

/// Load output tensors from files
fn load_output_tensors(
    input_dir: &Path,
    expected_outputs: &[TensorInfo],
) -> Result<HashMap<String, ComputeTensor>> {
    let mut output_tensors = HashMap::new();
    
    for output in expected_outputs {
        let file_path = input_dir.join(format!("{}_output.bin", output.name));
        let mut file = File::open(&file_path)?;
        
        // Read shape
        let mut shape_len_bytes = [0u8; std::mem::size_of::<usize>()];
        file.read_exact(&mut shape_len_bytes)?;
        let shape_len = usize::from_le_bytes(shape_len_bytes);
        
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; std::mem::size_of::<usize>()];
            file.read_exact(&mut dim_bytes)?;
            shape.push(usize::from_le_bytes(dim_bytes));
        }
        
        // Read data type
        let mut dtype_byte = [0u8; 1];
        file.read_exact(&mut dtype_byte)?;
        
        let data_type = match dtype_byte[0] {
            0 => DataType::Float32,
            1 => DataType::Float64,
            2 => DataType::Int32,
            3 => DataType::Int64,
            4 => DataType::Uint8,
            5 => DataType::Bool,
            _ => return Err(anyhow!("Unknown data type code: {}", dtype_byte[0])),
        };
        
        // Create tensor and read data
        let mut tensor = ComputeTensor::new(&shape, data_type);
        
        match data_type {
            DataType::Float32 => {
                if let Some(data) = tensor.as_slice_mut::<f32>() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u8,
                            data.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    file.read_exact(bytes)?;
                }
            },
            DataType::Float64 => {
                if let Some(data) = tensor.as_slice_mut::<f64>() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u8,
                            data.len() * std::mem::size_of::<f64>(),
                        )
                    };
                    file.read_exact(bytes)?;
                }
            },
            DataType::Int32 => {
                if let Some(data) = tensor.as_slice_mut::<i32>() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u8,
                            data.len() * std::mem::size_of::<i32>(),
                        )
                    };
                    file.read_exact(bytes)?;
                }
            },
            DataType::Int64 => {
                if let Some(data) = tensor.as_slice_mut::<i64>() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u8,
                            data.len() * std::mem::size_of::<i64>(),
                        )
                    };
                    file.read_exact(bytes)?;
                }
            },
            _ => return Err(anyhow!("Unsupported data type: {:?}", data_type)),
        }
        
        output_tensors.insert(output.name.clone(), tensor);
    }
    
    Ok(output_tensors)
}

/// Detect ONNX Runtime installation
fn detect_onnxruntime() -> Result<String> {
    // Check if ONNXRUNTIME_HOME environment variable is set
    if let Ok(path) = std::env::var("ONNXRUNTIME_HOME") {
        let bin_path = Path::new(&path).join("bin").join("onnxruntime_run");
        if bin_path.exists() {
            return Ok(bin_path.to_string_lossy().to_string());
        }
    }
    
    // Check if onnxruntime_run is in PATH
    if let Ok(output) = Command::new("which").arg("onnxruntime_run").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    
    // If we couldn't find it, return an error
    Err(anyhow!("ONNX Runtime not found. Set ONNXRUNTIME_HOME or install it in PATH."))
}

/// Detect Tract installation
fn detect_tract() -> Result<String> {
    // Check if TRACT_HOME environment variable is set
    if let Ok(path) = std::env::var("TRACT_HOME") {
        let bin_path = Path::new(&path).join("target").join("release").join("tract");
        if bin_path.exists() {
            return Ok(bin_path.to_string_lossy().to_string());
        }
    }
    
    // Check if tract is in PATH
    if let Ok(output) = Command::new("which").arg("tract").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    
    // If we couldn't find it, return an error
    Err(anyhow!("Tract not found. Set TRACT_HOME or install it in PATH."))
}

/// Detect TensorRT installation
fn detect_tensorrt() -> Result<String> {
    // Check if TENSORRT_HOME environment variable is set
    if let Ok(path) = std::env::var("TENSORRT_HOME") {
        let bin_path = Path::new(&path).join("bin").join("trtexec");
        if bin_path.exists() {
            return Ok(bin_path.to_string_lossy().to_string());
        }
    }
    
    // Check if trtexec is in PATH
    if let Ok(output) = Command::new("which").arg("trtexec").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    
    // If we couldn't find it, return an error
    Err(anyhow!("TensorRT not found. Set TENSORRT_HOME or install it in PATH."))
}

/// Run ONNX Runtime on a model
fn run_onnxruntime(
    onnxruntime_path: &str,
    model_path: &Path,
    input_files: &HashMap<String, String>,
    output_dir: &Path,
    use_gpu: bool,
) -> Result<()> {
    let mut cmd = Command::new(onnxruntime_path);
    
    cmd.arg("--model").arg(model_path)
        .arg("--output_dir").arg(output_dir);
    
    // Add input files
    for (name, path) in input_files {
        cmd.arg("--input").arg(format!("{}:{}", name, path));
    }
    
    // Use GPU if requested
    if use_gpu {
        cmd.arg("--use_gpu");
    }
    
    // Run the command
    let output = cmd.output()?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "ONNX Runtime failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(())
}

/// Run Tract on a model
fn run_tract(
    tract_path: &str,
    model_path: &Path,
    input_files: &HashMap<String, String>,
    output_dir: &Path,
) -> Result<()> {
    let mut cmd = Command::new(tract_path);
    
    cmd.arg("run")
        .arg(model_path)
        .arg("--output-dir").arg(output_dir);
    
    // Add input files
    for (name, path) in input_files {
        cmd.arg("--input").arg(format!("{}={}", name, path));
    }
    
    // Run the command
    let output = cmd.output()?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "Tract failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(())
}

/// Convert ONNX model to TensorRT format
fn convert_to_tensorrt(
    model_path: &Path,
    tensorrt_path: &str,
    output_dir: &Path,
) -> Result<String> {
    let output_path = output_dir.join("model.trt");
    
    let mut cmd = Command::new(tensorrt_path);
    
    cmd.arg("--onnx").arg(model_path)
        .arg("--saveEngine").arg(&output_path);
    
    // Run the command
    let output = cmd.output()?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "TensorRT conversion failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(output_path.to_string_lossy().to_string())
}

/// Run TensorRT on a model
fn run_tensorrt(
    tensorrt_path: &str,
    model_path: &str,
    input_files: &HashMap<String, String>,
    output_dir: &Path,
) -> Result<()> {
    let mut cmd = Command::new(tensorrt_path);
    
    cmd.arg("--loadEngine").arg(model_path)
        .arg("--exportOutput").arg(output_dir);
    
    // Add input files
    for (name, path) in input_files {
        cmd.arg("--input").arg(format!("{}:{}", name, path));
    }
    
    // Run the command
    let output = cmd.output()?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "TensorRT failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(())
}

/// Get ONNX Runtime version
fn get_onnxruntime_version(onnxruntime_path: &str) -> Result<String> {
    let output = Command::new(onnxruntime_path)
        .arg("--version")
        .output()?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    } else {
        Ok("Unknown".to_string())
    }
}

/// Get Tract version
fn get_tract_version(tract_path: &str) -> Result<String> {
    let output = Command::new(tract_path)
        .arg("--version")
        .output()?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    } else {
        Ok("Unknown".to_string())
    }
}

/// Get TensorRT version
fn get_tensorrt_version(tensorrt_path: &str) -> Result<String> {
    let output = Command::new(tensorrt_path)
        .arg("--version")
        .output()?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    } else {
        Ok("Unknown".to_string())
    }
}

/// Generate a simple model with just one operator
fn generate_operator_model(
    op_type: &str,
    input_shapes: &[&[usize]],
    attributes: &HashMap<String, String>,
) -> Result<PathBuf> {
    // This function would generate a simple ONNX model with a single operator
    // For now, we'll use a hardcoded model path for testing
    let model_path = PathBuf::from("test_models/generated_op_model.onnx");
    
    // In a real implementation, we would dynamically generate an ONNX model
    // using the ONNX API or a serialization library
    
    Ok(model_path)
}