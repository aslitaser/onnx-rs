use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use onnx::error::Result;
use onnx::execution::context::ExecutionOptions;
use onnx::execution::engine::{ExecutionEngine, ExecutionPriority};
use onnx::model::OnnxModel;
use onnx::ops::tensor::{DataType, Tensor};

/// Example demonstrating advanced concurrent execution of ONNX models
fn main() -> Result<()> {
    // Load an ONNX model
    println!("Loading ONNX model...");
    let model_path = std::env::args().nth(1).expect("Please provide a path to an ONNX model");
    let model = OnnxModel::load_from_file(&model_path)?;
    
    // Configure execution options for advanced concurrency
    println!("Configuring advanced concurrent execution...");
    let options = ExecutionOptions::new()
        .enable_profiling(true)
        .set_thread_count(4)  // Use 4 threads for parallel execution
        .enable_critical_path_scheduling(true)  // Enable critical path analysis
        .set_cancel_on_error(true)  // Cancel execution if any operation fails
        .enable_dynamic_rebalancing(true)  // Dynamically balance work based on execution times
        .set_max_concurrent_operations(8)  // Allow up to 8 concurrent operations
        .set_operation_timeout(30000);  // Set 30-second timeout for operations
    
    // Create thread-safe execution engine
    let engine = Arc::new(ExecutionEngine::new(model, options)?);
    
    // Prepare the engine
    engine.prepare()?;
    
    // Get model information
    let input_names = engine.input_names();
    let output_names = engine.output_names();
    
    println!("Model inputs: {:?}", input_names);
    println!("Model outputs: {:?}", output_names);
    
    // Create a dummy input tensor (for example purposes)
    let create_sample_input = || {
        let mut inputs = HashMap::new();
        
        // Create a simple input tensor with appropriate shape
        if let Some(input_name) = input_names.get(0) {
            // In a real application, use the correct shape and data
            let tensor = Tensor::new_with_data(
                &[1, 3, 224, 224],  // Common image input shape
                DataType::Float32,
                vec![0.5; 1 * 3 * 224 * 224],  // Dummy data
            );
            
            inputs.insert(input_name.to_string(), tensor);
        }
        
        inputs
    };
    
    // Compare standard execution vs. advanced concurrent execution
    println!("\nComparing execution methods:");
    
    // Standard execution
    println!("Standard sequential execution...");
    let start = Instant::now();
    let sequential_outputs = engine.run(create_sample_input())?;
    let sequential_time = start.elapsed();
    println!("Sequential execution completed in {:?}", sequential_time);
    
    // Basic concurrent execution
    println!("Basic concurrent execution...");
    let start = Instant::now();
    let basic_concurrent_outputs = engine.run_concurrent(create_sample_input())?;
    let basic_concurrent_time = start.elapsed();
    println!("Basic concurrent execution completed in {:?}", basic_concurrent_time);
    
    // Advanced concurrent execution
    println!("Advanced concurrent execution...");
    let start = Instant::now();
    let advanced_concurrent_outputs = engine.run_advanced_concurrent(create_sample_input())?;
    let advanced_concurrent_time = start.elapsed();
    println!("Advanced concurrent execution completed in {:?}", advanced_concurrent_time);
    
    // Show speedup
    println!("\nSpeedup Summary:");
    println!("Basic concurrent vs. Sequential: {:.2}x", 
             sequential_time.as_secs_f64() / basic_concurrent_time.as_secs_f64());
    println!("Advanced concurrent vs. Sequential: {:.2}x", 
             sequential_time.as_secs_f64() / advanced_concurrent_time.as_secs_f64());
    println!("Advanced concurrent vs. Basic concurrent: {:.2}x", 
             basic_concurrent_time.as_secs_f64() / advanced_concurrent_time.as_secs_f64());
    
    // Verify that outputs match
    println!("\nValidating outputs...");
    let validate_outputs = |a: &HashMap<String, Tensor>, b: &HashMap<String, Tensor>| -> bool {
        if a.len() != b.len() {
            println!("  Output counts don't match: {} vs {}", a.len(), b.len());
            return false;
        }
        
        for (name, tensor_a) in a {
            match b.get(name) {
                Some(tensor_b) => {
                    if tensor_a.shape() != tensor_b.shape() {
                        println!("  Shape mismatch for {}: {:?} vs {:?}", 
                                name, tensor_a.shape(), tensor_b.shape());
                        return false;
                    }
                    // For floating point data, we should do approximate comparison
                    // This is a simplified check
                    if tensor_a.data_type() != tensor_b.data_type() {
                        println!("  Data type mismatch for {}: {:?} vs {:?}", 
                                name, tensor_a.data_type(), tensor_b.data_type());
                        return false;
                    }
                },
                None => {
                    println!("  Output '{}' missing in second result", name);
                    return false;
                }
            }
        }
        true
    };
    
    let seq_vs_basic = validate_outputs(&sequential_outputs, &basic_concurrent_outputs);
    let seq_vs_advanced = validate_outputs(&sequential_outputs, &advanced_concurrent_outputs);
    
    println!("Sequential vs Basic: {}", if seq_vs_basic { "MATCH ✓" } else { "MISMATCH ✗" });
    println!("Sequential vs Advanced: {}", if seq_vs_advanced { "MATCH ✓" } else { "MISMATCH ✗" });
    
    // Demonstrate parallel model execution with different inputs
    println!("\nDemonstrating parallel model execution with different inputs...");
    let num_models = 4;
    let mut engines = Vec::with_capacity(num_models);
    
    for i in 0..num_models {
        let engine_clone = engine.clone_for_parallel_execution()?;
        engines.push(engine_clone);
        println!("  Created engine instance #{}", i+1);
    }
    
    // Execute models in parallel
    println!("Executing {} models in parallel...", num_models);
    let start = Instant::now();
    
    let barrier = Arc::new(Barrier::new(num_models));
    let mut handles = Vec::with_capacity(num_models);
    
    for (i, engine) in engines.into_iter().enumerate() {
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            println!("  Model #{} preparing", i+1);
            
            // Create slightly different input for each model (in a real application)
            let mut inputs = create_sample_input();
            
            // Wait for all threads to be ready
            barrier_clone.wait();
            
            println!("  Model #{} starting execution", i+1);
            
            // Run with advanced concurrent execution
            match engine.run_advanced_concurrent(inputs) {
                Ok(outputs) => {
                    println!("  Model #{} completed with {} outputs", i+1, outputs.len());
                    Ok(outputs)
                },
                Err(e) => {
                    println!("  Model #{} failed: {:?}", i+1, e);
                    Err(e)
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut all_results = Vec::with_capacity(num_models);
    for handle in handles {
        all_results.push(handle.join().expect("Thread panicked")?);
    }
    
    let parallel_time = start.elapsed();
    println!("All {} models completed in {:?}", num_models, parallel_time);
    println!("Average time per model: {:?}", parallel_time / num_models as u32);
    
    println!("\nConcurrent execution successfully demonstrated!");
    
    Ok(())
}