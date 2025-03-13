use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::process::Command;
use std::fs::{File, create_dir_all};
use std::io::{Write, Read};

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use serde::{Serialize, Deserialize};
use tempfile::TempDir;
use anyhow::{Result, anyhow, Context};

use onnx_parser::{
    ExecutionEngine, ExecutionOptions, OptimizationLevel, ComputeTensor,
    parser::model_loader::ModelLoader,
    ops::tensor::{DataType, Shape},
    error,
    tools::{
        profile::{profile_model_execution, ProfileEventType, PerformanceStats},
        comparison::{
            RuntimeType, ModelComparisonResult, BenchmarkConfig, CorrectnessMetrics,
            RuntimePerformance, compare_with_onnxruntime, compare_with_tract, compare_with_tensorrt,
            generate_comparison_report,
        },
    },
};

// Import the input generators from onnx_benchmarks
use crate::onnx_benchmarks::{
    RandomNormalGenerator, ZeroGenerator, InputGenerator, 
    benchmark_model, benchmark_operator, compare_optimization_levels,
};

// =====================================================================
// Comparison Benchmarks for Operators
// =====================================================================

/// Results from comparing an operator across runtimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpComparisonResults {
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

/// Compare convolution performance across runtimes
pub fn compare_convolution_performance() -> Result<OpComparisonResults, error::Error> {
    // Define convolution parameters
    let input_shapes = vec![
        vec![1, 3, 224, 224],  // Input (N, C, H, W)
        vec![64, 3, 3, 3],     // Weights (Cout, Cin, Kh, Kw)
    ];
    
    // Define attributes
    let attributes = HashMap::from([
        ("strides".to_string(), "1, 1".to_string()),
        ("padding".to_string(), "1, 1".to_string()),
        ("dilations".to_string(), "1, 1".to_string()),
        ("groups".to_string(), "1".to_string()),
    ]);
    
    // Run comparison for the convolution operator
    compare_operator_performance("Conv", &input_shapes, &attributes)
}

/// Compare matrix multiplication performance across runtimes
pub fn compare_matmul_performance() -> Result<OpComparisonResults, error::Error> {
    // Define matrix multiplication parameters
    let input_shapes = vec![
        vec![1, 1024, 1024],  // Input A (B, M, K)
        vec![1, 1024, 1024],  // Input B (B, K, N)
    ];
    
    // MatMul doesn't have attributes
    let attributes = HashMap::new();
    
    // Run comparison for the MatMul operator
    compare_operator_performance("MatMul", &input_shapes, &attributes)
}

/// Compare GEMM performance across runtimes
pub fn compare_gemm_performance() -> Result<OpComparisonResults, error::Error> {
    // Define GEMM parameters
    let input_shapes = vec![
        vec![1, 1024, 1024],  // Input A (B, M, K)
        vec![1, 1024, 1024],  // Input B (B, K, N)
        vec![1, 1024, 1024],  // Input C (B, M, N)
    ];
    
    // Define GEMM attributes
    let attributes = HashMap::from([
        ("alpha".to_string(), "1.0".to_string()),
        ("beta".to_string(), "1.0".to_string()),
        ("transA".to_string(), "0".to_string()),
        ("transB".to_string(), "0".to_string()),
    ]);
    
    // Run comparison for the Gemm operator
    compare_operator_performance("Gemm", &input_shapes, &attributes)
}

/// Compare pooling performance across runtimes
pub fn compare_pooling_performance() -> Result<OpComparisonResults, error::Error> {
    // Define pooling parameters
    let input_shapes = vec![
        vec![1, 64, 112, 112],  // Input (N, C, H, W)
    ];
    
    // Define pooling attributes
    let attributes = HashMap::from([
        ("kernel_shape".to_string(), "3, 3".to_string()),
        ("strides".to_string(), "2, 2".to_string()),
        ("padding".to_string(), "1, 1".to_string()),
        ("ceil_mode".to_string(), "0".to_string()),
    ]);
    
    // Run comparison for the MaxPool operator
    compare_operator_performance("MaxPool", &input_shapes, &attributes)
}

/// Compare implementation of a specific operator across different runtimes
fn compare_operator_performance(
    op_type: &str,
    input_shapes: &[Vec<usize>],
    attributes: &HashMap<String, String>,
) -> Result<OpComparisonResults, error::Error> {
    // First, create a simple ONNX model with just this operator
    let model_path = create_operator_model(op_type, input_shapes, attributes)?;
    
    // Create input tensors based on shapes
    let mut input_data = HashMap::new();
    for (i, shape) in input_shapes.iter().enumerate() {
        let input_name = format!("input_{}", i);
        let mut generator = RandomNormalGenerator::new(0.0, 1.0, DataType::Float32, Some(42));
        let tensor = generator.generate(&input_name, shape)?;
        input_data.insert(input_name, tensor);
    }
    
    // Setup benchmark config
    let config = BenchmarkConfig {
        warmup_iterations: 5,
        measurement_iterations: 20,
        check_correctness: true,
        ..Default::default()
    };
    
    // Run benchmarks for our runtime
    let our_model_result = benchmark_our_runtime_for_operator(&model_path, &input_data, &config)
        .map_err(|e| error::Error::External(e.to_string()))?;
    
    // Try ONNX Runtime if available
    let mut onnx_result = None;
    if let Ok(onnx_path) = detect_onnxruntime() {
        if let Ok(res) = run_onnxruntime_for_operator(&model_path, &input_data, &config)
            .map_err(|e| error::Error::External(e.to_string())) {
            onnx_result = Some(res);
        }
    }
    
    // Try Tract if available
    let mut tract_result = None;
    if let Ok(tract_path) = detect_tract() {
        if let Ok(res) = run_tract_for_operator(&model_path, &input_data, &config)
            .map_err(|e| error::Error::External(e.to_string())) {
            tract_result = Some(res);
        }
    }
    
    // Combine results
    let mut runtime_results = HashMap::new();
    runtime_results.insert(RuntimeType::OurRuntime, our_model_result);
    
    if let Some(res) = onnx_result {
        runtime_results.insert(RuntimeType::OnnxRuntime, res);
    }
    
    if let Some(res) = tract_result {
        runtime_results.insert(RuntimeType::Tract, res);
    }
    
    // Calculate relative speedups
    let our_time_ms = our_model_result.mean_time_ms;
    
    let mut relative_speedups = HashMap::new();
    relative_speedups.insert(RuntimeType::OurRuntime, 1.0);
    
    for (&runtime, perf) in &runtime_results {
        if runtime != RuntimeType::OurRuntime {
            relative_speedups.insert(runtime, our_time_ms / perf.mean_time_ms);
        }
    }
    
    // Create the final comparison results
    Ok(OpComparisonResults {
        op_type: op_type.to_string(),
        input_shapes: input_shapes.to_vec(),
        attributes: attributes.clone(),
        runtime_results,
        relative_speedups,
    })
}

/// Benchmark our runtime for the operator comparison
fn benchmark_our_runtime_for_operator(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<RuntimePerformance> {
    // Load the model
    let model = ModelLoader::load_from_file(model_path)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;
    
    // Create execution options
    let options = ExecutionOptions::new()
        .set_optimization_level(config.optimization_level)
        .enable_profiling(true);
    
    // Create and prepare the execution engine
    let mut engine = ExecutionEngine::new(model, options)
        .map_err(|e| anyhow!("Failed to create execution engine: {}", e))?;
    engine.prepare()
        .map_err(|e| anyhow!("Failed to prepare engine: {}", e))?;
    
    // Warmup iterations
    for _ in 0..config.warmup_iterations {
        let _ = engine.run(input_data.clone())
            .map_err(|e| anyhow!("Warmup failed: {}", e))?;
    }
    
    // Measurement iterations
    let mut execution_times = Vec::with_capacity(config.measurement_iterations);
    let mut peak_memory = 0;
    
    for i in 0..config.measurement_iterations {
        let start = Instant::now();
        let _ = engine.run(input_data.clone())
            .map_err(|e| anyhow!("Execution failed: {}", e))?;
        let elapsed = start.elapsed();
        execution_times.push(elapsed.as_secs_f64() * 1000.0);
        
        // Get profiling statistics for the last iteration
        if i == config.measurement_iterations - 1 {
            let stats = profile_model_execution(&mut engine, input_data)
                .map_err(|e| anyhow!("Profiling failed: {}", e))?;
            peak_memory = stats.peak_memory_bytes;
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
    
    Ok(RuntimePerformance {
        runtime_type: RuntimeType::OurRuntime,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: Some(peak_memory),
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second: 1000.0 / mean_time_ms,
        version: env!("CARGO_PKG_VERSION").to_string(),
        backend: "CPU".to_string(),
        additional_metrics: HashMap::new(),
    })
}

/// Run ONNX Runtime for operator comparison
fn run_onnxruntime_for_operator(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<RuntimePerformance> {
    // Detect ONNX Runtime installation
    let onnxruntime_path = detect_onnxruntime()?;
    
    // Create a temporary directory for I/O data
    let tmp_dir = TempDir::new()?;
    
    // Save input tensors
    let input_files = save_input_tensors(input_data, tmp_dir.path())?;
    
    // Warmup runs
    for _ in 0..config.warmup_iterations {
        run_onnxruntime_cmd(
            &onnxruntime_path,
            model_path,
            &input_files,
            tmp_dir.path(),
            config.use_gpu,
        )?;
    }
    
    // Measurement runs
    let mut execution_times = Vec::with_capacity(config.measurement_iterations);
    
    for _ in 0..config.measurement_iterations {
        let start = Instant::now();
        run_onnxruntime_cmd(
            &onnxruntime_path,
            model_path,
            &input_files,
            tmp_dir.path(),
            config.use_gpu,
        )?;
        let elapsed = start.elapsed();
        execution_times.push(elapsed.as_secs_f64() * 1000.0);
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
    
    // Get version
    let version = get_onnxruntime_version(&onnxruntime_path).unwrap_or_else(|_| "Unknown".to_string());
    
    Ok(RuntimePerformance {
        runtime_type: RuntimeType::OnnxRuntime,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: None,
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second: 1000.0 / mean_time_ms,
        version,
        backend: if config.use_gpu { "GPU".to_string() } else { "CPU".to_string() },
        additional_metrics: HashMap::new(),
    })
}

/// Run Tract for operator comparison
fn run_tract_for_operator(
    model_path: &Path,
    input_data: &HashMap<String, ComputeTensor>,
    config: &BenchmarkConfig,
) -> Result<RuntimePerformance> {
    // Detect Tract installation
    let tract_path = detect_tract()?;
    
    // Create a temporary directory for I/O data
    let tmp_dir = TempDir::new()?;
    
    // Save input tensors
    let input_files = save_input_tensors(input_data, tmp_dir.path())?;
    
    // Warmup runs
    for _ in 0..config.warmup_iterations {
        run_tract_cmd(
            &tract_path,
            model_path,
            &input_files,
            tmp_dir.path(),
        )?;
    }
    
    // Measurement runs
    let mut execution_times = Vec::with_capacity(config.measurement_iterations);
    
    for _ in 0..config.measurement_iterations {
        let start = Instant::now();
        run_tract_cmd(
            &tract_path,
            model_path,
            &input_files,
            tmp_dir.path(),
        )?;
        let elapsed = start.elapsed();
        execution_times.push(elapsed.as_secs_f64() * 1000.0);
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
    
    // Get version
    let version = get_tract_version(&tract_path).unwrap_or_else(|_| "Unknown".to_string());
    
    Ok(RuntimePerformance {
        runtime_type: RuntimeType::Tract,
        mean_time_ms,
        std_dev_ms,
        min_time_ms,
        max_time_ms,
        peak_memory_bytes: None,
        warmup_time_ms: None,
        init_time_ms: None,
        inferences_per_second: 1000.0 / mean_time_ms,
        version,
        backend: "CPU".to_string(),
        additional_metrics: HashMap::new(),
    })
}

// =====================================================================
// Helper Functions
// =====================================================================

/// Create a simple ONNX model with just one operator for testing
fn create_operator_model(
    op_type: &str,
    input_shapes: &[Vec<usize>],
    attributes: &HashMap<String, String>,
) -> Result<PathBuf, error::Error> {
    // Note: In a real implementation, this would dynamically generate an ONNX model
    // using the ONNX API. For now, we'll use pre-generated models for testing.
    
    let test_models_dir = PathBuf::from("test_models");
    if !test_models_dir.exists() {
        create_dir_all(&test_models_dir).map_err(|e| 
            error::Error::External(format!("Failed to create test_models directory: {}", e)))?;
    }
    
    let model_path = test_models_dir.join(format!("{}_test.onnx", op_type.to_lowercase()));
    
    // Since we can't generate real ONNX models here, we'll just return the path
    // where we expect the model to be. In a real implementation, we would create
    // the model if it doesn't exist.
    
    // Check if the model exists
    if !model_path.exists() {
        return Err(error::Error::External(format!(
            "Test model for {} not found at {}. Please create it manually or implement model generation.",
            op_type, model_path.display()
        )));
    }
    
    Ok(model_path)
}

/// Save input tensors to files
fn save_input_tensors(
    inputs: &HashMap<String, ComputeTensor>,
    output_dir: &Path,
) -> Result<HashMap<String, String>> {
    let mut input_files = HashMap::new();
    
    for (name, tensor) in inputs {
        let file_path = output_dir.join(format!("{}.bin", name));
        let mut file = File::create(&file_path)?;
        
        // Write tensor metadata (shape and data type)
        let shape = tensor.shape();
        let shape_len = shape.len() as u32;
        file.write_all(&shape_len.to_le_bytes())?;
        
        for &dim in shape {
            let dim_u32 = dim as u32;
            file.write_all(&dim_u32.to_le_bytes())?;
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
            _ => return Err(anyhow!("Unsupported data type for saving: {:?}", tensor.data_type())),
        }
        
        input_files.insert(name.clone(), file_path.to_string_lossy().to_string());
    }
    
    Ok(input_files)
}

/// Detect if ONNX Runtime is installed
fn detect_onnxruntime() -> Result<String> {
    // Check for environment variable
    if let Ok(path) = std::env::var("ONNXRUNTIME_HOME") {
        let bin_path = Path::new(&path).join("bin").join("onnxruntime_run");
        if bin_path.exists() {
            return Ok(bin_path.to_string_lossy().to_string());
        }
    }
    
    // Try to find it in PATH
    if let Ok(output) = Command::new("which").arg("onnxruntime_run").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    
    Err(anyhow!("ONNX Runtime not found. Please set ONNXRUNTIME_HOME environment variable."))
}

/// Detect if Tract is installed
fn detect_tract() -> Result<String> {
    // Check for environment variable
    if let Ok(path) = std::env::var("TRACT_HOME") {
        let bin_path = Path::new(&path).join("target").join("release").join("tract");
        if bin_path.exists() {
            return Ok(bin_path.to_string_lossy().to_string());
        }
    }
    
    // Try to find it in PATH
    if let Ok(output) = Command::new("which").arg("tract").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    
    Err(anyhow!("Tract not found. Please set TRACT_HOME environment variable."))
}

/// Run ONNX Runtime command
fn run_onnxruntime_cmd(
    onnxruntime_path: &str,
    model_path: &Path,
    input_files: &HashMap<String, String>,
    output_dir: &Path,
    use_gpu: bool,
) -> Result<()> {
    let mut cmd = Command::new(onnxruntime_path);
    
    cmd.arg("--model").arg(model_path)
        .arg("--output_dir").arg(output_dir);
    
    // Add inputs
    for (name, path) in input_files {
        cmd.arg("--input").arg(format!("{}:{}", name, path));
    }
    
    // Use GPU if requested
    if use_gpu {
        cmd.arg("--use_gpu");
    }
    
    // Run the command
    let output = cmd.output()
        .context("Failed to execute ONNX Runtime command")?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "ONNX Runtime failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(())
}

/// Run Tract command
fn run_tract_cmd(
    tract_path: &str,
    model_path: &Path,
    input_files: &HashMap<String, String>,
    output_dir: &Path,
) -> Result<()> {
    let mut cmd = Command::new(tract_path);
    
    cmd.arg("run")
        .arg(model_path)
        .arg("--output-dir").arg(output_dir);
    
    // Add inputs
    for (name, path) in input_files {
        cmd.arg("--input").arg(format!("{}={}", name, path));
    }
    
    // Run the command
    let output = cmd.output()
        .context("Failed to execute Tract command")?;
    
    if !output.status.success() {
        return Err(anyhow!(
            "Tract failed with exit code {}: {}",
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
        .output()
        .context("Failed to execute ONNX Runtime version command")?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    } else {
        Err(anyhow!("ONNX Runtime version command failed"))
    }
}

/// Get Tract version
fn get_tract_version(tract_path: &str) -> Result<String> {
    let output = Command::new(tract_path)
        .arg("--version")
        .output()
        .context("Failed to execute Tract version command")?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    } else {
        Err(anyhow!("Tract version command failed"))
    }
}

// =====================================================================
// Criterion Benchmark Functions
// =====================================================================

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_comparisons");
    
    // Benchmark operator comparisons if ONNX Runtime is available
    let has_onnxruntime = detect_onnxruntime().is_ok();
    let has_tract = detect_tract().is_ok();
    
    // Only run if we have at least one comparison runtime
    if has_onnxruntime || has_tract {
        // Convolution comparison
        group.bench_function(BenchmarkId::new("op_compare", "Conv"), |b| {
            b.iter(|| {
                let _ = compare_convolution_performance();
            });
        });
        
        // MatMul comparison
        group.bench_function(BenchmarkId::new("op_compare", "MatMul"), |b| {
            b.iter(|| {
                let _ = compare_matmul_performance();
            });
        });
        
        // GEMM comparison
        group.bench_function(BenchmarkId::new("op_compare", "Gemm"), |b| {
            b.iter(|| {
                let _ = compare_gemm_performance();
            });
        });
        
        // Pooling comparison
        group.bench_function(BenchmarkId::new("op_compare", "MaxPool"), |b| {
            b.iter(|| {
                let _ = compare_pooling_performance();
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);