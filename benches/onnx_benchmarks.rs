use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use anyhow::Result;

use onnx_parser::{
    ExecutionEngine, ExecutionOptions, OptimizationLevel, OnnxModel, ComputeTensor,
    parser::model_loader::ModelLoader,
    ops::tensor::{DataType, Shape},
    model::{Tensor, NodeId},
    error,
    tools::profile::{profile_model_execution, PerformanceStats},
};

// =====================================================================
// Benchmark Data Structures
// =====================================================================

/// Metrics related to execution latency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Mean latency in milliseconds
    pub mean_ms: f64,
    /// Standard deviation of latency in milliseconds
    pub std_dev_ms: f64,
    /// Minimum latency in milliseconds
    pub min_ms: f64,
    /// Maximum latency in milliseconds
    pub max_ms: f64,
    /// Median latency in milliseconds
    pub median_ms: f64,
    /// Latency at requested percentiles in milliseconds
    pub percentiles: HashMap<String, f64>,
}

/// Metrics related to throughput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Instances per second
    pub instances_per_second: f64,
    /// Total batches processed
    pub total_batches: usize,
    /// Total instances processed
    pub total_instances: usize,
    /// Execution time in seconds
    pub execution_time_seconds: f64,
    /// Batch size used
    pub batch_size: usize,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Average memory usage in bytes
    pub average_bytes: f64,
    /// Memory usage per operator in bytes
    pub per_operator_bytes: HashMap<String, usize>,
}

/// Target of the benchmark (model or operator)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkTarget {
    /// A complete ONNX model
    Model {
        /// Path to the model file
        path: PathBuf,
        /// Name of the model (e.g., "ResNet50")
        name: String,
        /// Input shapes
        input_shapes: HashMap<String, Vec<usize>>,
        /// Optimization level used
        optimization_level: String,
    },
    /// A single operator
    Operator {
        /// Type of the operator (e.g., "Conv")
        op_type: String,
        /// Input shapes
        input_shapes: Vec<Vec<usize>>,
        /// Operator attributes
        attributes: HashMap<String, String>,
    },
}

/// Results from a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Name of the benchmark
    pub name: String,
    /// Description of what was benchmarked
    pub description: String,
    /// Model or operator details
    pub target: BenchmarkTarget,
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: Option<ThroughputMetrics>,
    /// Memory usage metrics in bytes
    pub memory_usage: Option<MemoryMetrics>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when the benchmark was run
    pub timestamp: String,
}

/// Results from a comparison benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    /// Name of the comparison
    pub name: String,
    /// Description of what was compared
    pub description: String,
    /// Results from each configuration
    pub configurations: Vec<BenchmarkResults>,
    /// Speedup relative to baseline (first configuration)
    pub relative_speedups: HashMap<String, f64>,
    /// Metadata about the comparison
    pub metadata: HashMap<String, String>,
}

// =====================================================================
// Input Generators for Benchmarks
// =====================================================================

/// Interface for generating input tensors
pub trait InputGenerator {
    /// Generate an input tensor with the given name and shape
    fn generate(&mut self, name: &str, shape: &[usize]) -> Result<ComputeTensor, error::Error>;
    
    /// Get the expected data type of the generated tensor
    fn data_type(&self) -> DataType;
}

/// Generator for random float inputs following a normal distribution
pub struct RandomNormalGenerator {
    mean: f32,
    std_dev: f32,
    data_type: DataType,
    rng: StdRng,
}

impl RandomNormalGenerator {
    /// Create a new random normal generator
    pub fn new(
        mean: f32,
        std_dev: f32,
        data_type: DataType,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        Self {
            mean,
            std_dev,
            data_type,
            rng,
        }
    }
}

impl InputGenerator for RandomNormalGenerator {
    fn generate(&mut self, name: &str, shape: &[usize]) -> Result<ComputeTensor, error::Error> {
        let mut tensor = ComputeTensor::new(shape, self.data_type);
        
        // Fill with random data based on data type
        match self.data_type {
            DataType::Float32 => {
                let distribution = Normal::new(self.mean as f64, self.std_dev as f64).unwrap();
                let data = tensor.as_slice_mut::<f32>().unwrap();
                for value in data.iter_mut() {
                    *value = distribution.sample(&mut self.rng) as f32;
                }
            },
            DataType::Float64 => {
                let distribution = Normal::new(self.mean as f64, self.std_dev as f64).unwrap();
                let data = tensor.as_slice_mut::<f64>().unwrap();
                for value in data.iter_mut() {
                    *value = distribution.sample(&mut self.rng);
                }
            },
            DataType::Int32 => {
                let distribution = Normal::new(self.mean as f64, self.std_dev as f64).unwrap();
                let data = tensor.as_slice_mut::<i32>().unwrap();
                for value in data.iter_mut() {
                    *value = distribution.sample(&mut self.rng) as i32;
                }
            },
            DataType::Int64 => {
                let distribution = Normal::new(self.mean as f64, self.std_dev as f64).unwrap();
                let data = tensor.as_slice_mut::<i64>().unwrap();
                for value in data.iter_mut() {
                    *value = distribution.sample(&mut self.rng) as i64;
                }
            },
            _ => {
                return Err(error::Error::InvalidArgument(
                    format!("Unsupported data type for random generation: {:?}", self.data_type)
                ));
            }
        }
        
        Ok(tensor)
    }
    
    fn data_type(&self) -> DataType {
        self.data_type
    }
}

/// Generator for zero-filled inputs
pub struct ZeroGenerator {
    data_type: DataType,
}

impl ZeroGenerator {
    /// Create a new zero generator
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
        }
    }
}

impl InputGenerator for ZeroGenerator {
    fn generate(&mut self, _name: &str, shape: &[usize]) -> Result<ComputeTensor, error::Error> {
        // For zeros, we just create the tensor with default values
        Ok(ComputeTensor::new(shape, self.data_type))
    }
    
    fn data_type(&self) -> DataType {
        self.data_type
    }
}

// =====================================================================
// Benchmarking Functions
// =====================================================================

/// Benchmark a model with given inputs and iterations
pub fn benchmark_model(
    model_path: &Path,
    input_generator: &mut dyn InputGenerator,
    iterations: usize,
    optimization_level: OptimizationLevel,
) -> Result<BenchmarkResults, error::Error> {
    // Load the model
    let model = ModelLoader::load_from_file(model_path)?;
    
    // Get model metadata and input shapes
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    let input_shapes: HashMap<String, Vec<usize>> = model.graph().inputs().iter()
        .map(|input| {
            let shape = input.shape.iter()
                .map(|&dim| dim.abs() as usize)
                .collect::<Vec<_>>();
            (input.name.clone(), shape)
        })
        .collect();
    
    // Create execution options
    let options = ExecutionOptions::new()
        .set_optimization_level(optimization_level)
        .enable_profiling(true);
    
    // Create and prepare the execution engine
    let mut engine = ExecutionEngine::new(model, options)?;
    engine.prepare()?;
    
    // Generate inputs for the model
    let mut inputs = HashMap::new();
    for (name, shape) in &input_shapes {
        let tensor = input_generator.generate(name, shape)?;
        inputs.insert(name.clone(), tensor);
    }
    
    // Warmup iterations
    for _ in 0..5 {
        let _ = engine.run(inputs.clone())?;
    }
    
    // Measurement iterations
    let mut latencies = Vec::with_capacity(iterations);
    let mut peak_memory_bytes = 0;
    let mut perf_stats = None;
    
    for i in 0..iterations {
        let start = Instant::now();
        let _outputs = engine.run(inputs.clone())?;
        let elapsed = start.elapsed();
        
        latencies.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
        
        // Get profiling statistics for the last iteration
        if i == iterations - 1 {
            let stats = profile_model_execution(&mut engine, &inputs)?;
            peak_memory_bytes = stats.peak_memory_bytes;
            perf_stats = Some(stats);
        }
    }
    
    // Calculate latency statistics
    let mean_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let variance = latencies.iter().map(|&x| (x - mean_ms).powi(2)).sum::<f64>() / latencies.len() as f64;
    let std_dev_ms = variance.sqrt();
    let min_ms = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_ms = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate median
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if latencies.len() % 2 == 0 {
        (latencies[latencies.len() / 2 - 1] + latencies[latencies.len() / 2]) / 2.0
    } else {
        latencies[latencies.len() / 2]
    };
    
    // Calculate percentiles
    let mut percentiles = HashMap::new();
    let ps = [50.0, 90.0, 95.0, 99.0, 99.9];
    for p in &ps {
        let idx = ((latencies.len() as f64) * p / 100.0).ceil() as usize - 1;
        percentiles.insert(format!("p{}", p), latencies[idx.min(latencies.len() - 1)]);
    }
    
    // Calculate throughput
    let batch_size = 1; // Using batch size 1 for latency tests
    let throughput = ThroughputMetrics {
        instances_per_second: 1000.0 / mean_ms,
        total_batches: iterations,
        total_instances: iterations * batch_size,
        execution_time_seconds: mean_ms / 1000.0 * iterations as f64,
        batch_size,
    };
    
    // Build memory metrics if profiling data is available
    let memory_usage = if let Some(stats) = perf_stats {
        let mut per_operator_bytes = HashMap::new();
        
        // In a real implementation, we'd extract memory usage per operator
        // from profiling data. Here we're just creating a placeholder.
        per_operator_bytes.insert("all".to_string(), peak_memory_bytes);
        
        Some(MemoryMetrics {
            peak_bytes: peak_memory_bytes,
            average_bytes: peak_memory_bytes as f64 * 0.75, // Estimated average
            per_operator_bytes,
        })
    } else {
        None
    };
    
    // Create and return benchmark results
    Ok(BenchmarkResults {
        name: format!("{}_benchmark", model_name),
        description: format!("Benchmark of {} model", model_name),
        target: BenchmarkTarget::Model {
            path: model_path.to_path_buf(),
            name: model_name,
            input_shapes,
            optimization_level: format!("{:?}", optimization_level),
        },
        latency: LatencyMetrics {
            mean_ms,
            std_dev_ms,
            min_ms,
            max_ms,
            median_ms,
            percentiles,
        },
        throughput: Some(throughput),
        memory_usage,
        metadata: HashMap::new(),
        timestamp: chrono::Local::now().to_rfc3339(),
    })
}

/// Benchmark a specific operator type
pub fn benchmark_operator(
    op_type: &str,
    input_shapes: &[&[usize]],
    attributes: &HashMap<String, String>,
    iterations: usize,
) -> Result<BenchmarkResults, error::Error> {
    // In a real implementation, we would:
    // 1. Construct a simple model with just this operator
    // 2. Run it through our benchmarking infrastructure
    
    // For now, we'll return a placeholder result
    let mut latency_percentiles = HashMap::new();
    latency_percentiles.insert("p50".to_string(), 0.5);
    latency_percentiles.insert("p90".to_string(), 0.9);
    latency_percentiles.insert("p99".to_string(), 0.99);
    
    Ok(BenchmarkResults {
        name: format!("{}_operator_benchmark", op_type),
        description: format!("Benchmark of {} operator", op_type),
        target: BenchmarkTarget::Operator {
            op_type: op_type.to_string(),
            input_shapes: input_shapes.iter().map(|&s| s.to_vec()).collect(),
            attributes: attributes.clone(),
        },
        latency: LatencyMetrics {
            mean_ms: 1.0,
            std_dev_ms: 0.1,
            min_ms: 0.9,
            max_ms: 1.1,
            median_ms: 1.0,
            percentiles: latency_percentiles,
        },
        throughput: Some(ThroughputMetrics {
            instances_per_second: 1000.0,
            total_batches: iterations,
            total_instances: iterations,
            execution_time_seconds: 0.001 * iterations as f64,
            batch_size: 1,
        }),
        memory_usage: None,
        metadata: HashMap::from([
            ("implementation".to_string(), "placeholder".to_string()),
        ]),
        timestamp: chrono::Local::now().to_rfc3339(),
    })
}

/// Measure throughput of model execution
pub fn measure_throughput(
    engine: &mut ExecutionEngine,
    input_batch: &HashMap<String, ComputeTensor>,
    batch_size: usize,
) -> Result<ThroughputMetrics, error::Error> {
    // For real throughput tests, we'd create a proper batched input
    // For now, we'll just measure with the provided input
    
    // Warmup
    for _ in 0..3 {
        let _ = engine.run(input_batch.clone())?;
    }
    
    // Measure for specified time or iterations
    let duration = Duration::from_secs(5); // 5 second benchmark
    let start = Instant::now();
    let mut batches_processed = 0;
    
    while start.elapsed() < duration {
        let _ = engine.run(input_batch.clone())?;
        batches_processed += 1;
    }
    
    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();
    let instances = batches_processed * batch_size;
    let instances_per_second = instances as f64 / seconds;
    
    Ok(ThroughputMetrics {
        instances_per_second,
        total_batches: batches_processed,
        total_instances: instances,
        execution_time_seconds: seconds,
        batch_size,
    })
}

/// Measure latency of model execution with detailed percentiles
pub fn measure_latency(
    engine: &mut ExecutionEngine,
    inputs: &HashMap<String, ComputeTensor>,
    percentiles: &[f64],
) -> Result<LatencyMetrics, error::Error> {
    const ITERATIONS: usize = 100;
    
    // Warmup
    for _ in 0..10 {
        let _ = engine.run(inputs.clone())?;
    }
    
    // Measure execution times
    let mut latencies = Vec::with_capacity(ITERATIONS);
    
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = engine.run(inputs.clone())?;
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
    }
    
    // Calculate statistics
    let mean_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let variance = latencies.iter().map(|&x| (x - mean_ms).powi(2)).sum::<f64>() / latencies.len() as f64;
    let std_dev_ms = variance.sqrt();
    let min_ms = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_ms = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate median
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if latencies.len() % 2 == 0 {
        (latencies[latencies.len() / 2 - 1] + latencies[latencies.len() / 2]) / 2.0
    } else {
        latencies[latencies.len() / 2]
    };
    
    // Calculate requested percentiles
    let mut percentile_values = HashMap::new();
    for &p in percentiles {
        let idx = ((latencies.len() as f64) * p / 100.0).ceil() as usize - 1;
        let idx = idx.min(latencies.len() - 1);
        percentile_values.insert(format!("p{}", p), latencies[idx]);
    }
    
    Ok(LatencyMetrics {
        mean_ms,
        std_dev_ms,
        min_ms,
        max_ms,
        median_ms,
        percentiles: percentile_values,
    })
}

/// Compare a model at different optimization levels
pub fn compare_optimization_levels(
    model_path: &Path,
) -> Result<ComparisonResults, error::Error> {
    let optimization_levels = [
        OptimizationLevel::None,
        OptimizationLevel::Basic,
        OptimizationLevel::Standard,
        OptimizationLevel::Aggressive,
    ];
    
    let mut configurations = Vec::new();
    let mut input_generator = RandomNormalGenerator::new(
        0.0, 1.0, DataType::Float32, Some(42));
    
    // Benchmark with different optimization levels
    for level in &optimization_levels {
        let result = benchmark_model(model_path, &mut input_generator, 20, *level)?;
        configurations.push(result);
    }
    
    // Calculate relative speedups against baseline (no optimization)
    let mut relative_speedups = HashMap::new();
    if !configurations.is_empty() {
        let baseline_latency = configurations[0].latency.mean_ms;
        
        for (i, level) in optimization_levels.iter().enumerate().skip(1) {
            let current_latency = configurations[i].latency.mean_ms;
            let speedup = baseline_latency / current_latency;
            relative_speedups.insert(format!("{:?}", level), speedup);
        }
        
        // Add baseline for reference
        relative_speedups.insert(format!("{:?}", optimization_levels[0]), 1.0);
    }
    
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    Ok(ComparisonResults {
        name: format!("{}_optimization_comparison", model_name),
        description: format!("Comparison of optimization levels for {} model", model_name),
        configurations,
        relative_speedups,
        metadata: HashMap::new(),
    })
}

// =====================================================================
// Specific Model Benchmarks
// =====================================================================

/// Helper to create a default path for standard models
fn get_model_path(model_name: &str) -> PathBuf {
    PathBuf::from(format!("test_models/{}.onnx", model_name))
}

/// Benchmark ResNet50
pub fn benchmark_resnet50() -> Result<BenchmarkResults, error::Error> {
    let model_path = get_model_path("resnet50");
    
    // Create input generator with proper configuration for ResNet50
    let mut input_generator = RandomNormalGenerator::new(
        0.0, 1.0, DataType::Float32, Some(42));
    
    // Benchmark the model
    benchmark_model(
        &model_path,
        &mut input_generator,
        50, // Number of iterations
        OptimizationLevel::Standard,
    )
}

/// Benchmark MobileNet
pub fn benchmark_mobilenet() -> Result<BenchmarkResults, error::Error> {
    let model_path = get_model_path("mobilenet");
    
    // Create input generator with proper configuration for MobileNet
    let mut input_generator = RandomNormalGenerator::new(
        0.0, 1.0, DataType::Float32, Some(42));
    
    // Benchmark the model
    benchmark_model(
        &model_path,
        &mut input_generator,
        50, // Number of iterations
        OptimizationLevel::Standard,
    )
}

/// Benchmark BERT
pub fn benchmark_bert() -> Result<BenchmarkResults, error::Error> {
    let model_path = get_model_path("bert-base-uncased");
    
    // Create input generator with proper configuration for BERT
    // BERT uses int64 inputs for token IDs, attention masks, etc.
    let mut input_generator = ZeroGenerator::new(DataType::Int64);
    
    // Benchmark the model
    benchmark_model(
        &model_path,
        &mut input_generator,
        20, // Fewer iterations for a larger model
        OptimizationLevel::Standard,
    )
}

// =====================================================================
// Criterion Benchmark Functions
// =====================================================================

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_models");
    
    // Benchmark operators
    let conv_attributes = HashMap::new();
    let conv_shapes = [&[1, 3, 224, 224][..], &[32, 3, 3, 3][..]];
    
    group.bench_function(BenchmarkId::new("operator", "Conv"), |b| {
        b.iter(|| {
            let _ = benchmark_operator("Conv", &conv_shapes, &conv_attributes, 10);
        });
    });
    
    // Skip model benchmarks unless we detect they exist
    // This prevents errors when running on systems without the models
    
    // Benchmark models if available
    let resnet_path = get_model_path("resnet50");
    if resnet_path.exists() {
        group.bench_function(BenchmarkId::new("model", "ResNet50"), |b| {
            b.iter(|| {
                let _ = benchmark_resnet50();
            });
        });
    }
    
    let mobilenet_path = get_model_path("mobilenet");
    if mobilenet_path.exists() {
        group.bench_function(BenchmarkId::new("model", "MobileNet"), |b| {
            b.iter(|| {
                let _ = benchmark_mobilenet();
            });
        });
    }
    
    let bert_path = get_model_path("bert-base-uncased");
    if bert_path.exists() {
        group.bench_function(BenchmarkId::new("model", "BERT"), |b| {
            b.iter(|| {
                let _ = benchmark_bert();
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);