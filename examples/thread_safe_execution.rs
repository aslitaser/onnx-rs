use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

use onnx::error::Result;
use onnx::execution::context::ExecutionOptions;
use onnx::execution::engine::ExecutionEngine;
use onnx::model::OnnxModel;
use onnx::ops::tensor::{DataType, Tensor};

/// Example demonstrating thread-safe execution of ONNX models
fn main() -> Result<()> {
    // Load an ONNX model
    println!("Loading ONNX model...");
    let model_path = std::env::args().nth(1).expect("Please provide a path to an ONNX model");
    let model = OnnxModel::load_from_file(&model_path)?;
    
    // Create thread-safe execution engine
    println!("Creating thread-safe execution engine...");
    let options = ExecutionOptions::default()
        .enable_profiling(true)
        .set_thread_count(4); // Use 4 threads for operator execution
    
    let engine = Arc::new(ExecutionEngine::new(model, options)?);
    
    // Prepare the engine
    engine.prepare()?;
    
    // Get model information
    let input_names = engine.input_names();
    let output_names = engine.output_names();
    
    println!("Model inputs: {:?}", input_names);
    println!("Model outputs: {:?}", output_names);
    
    // Create sample input (just random data for demonstration)
    // In a real application, this would be your actual input data
    let create_sample_input = || {
        let mut inputs = HashMap::new();
        
        // Create a simple 1x3x224x224 tensor (common image input shape)
        let tensor = Tensor::new_with_data(
            &[1, 3, 224, 224],
            DataType::Float32,
            vec![0.5; 1 * 3 * 224 * 224],
        );
        
        // Add to inputs map with the model's input name
        if let Some(input_name) = input_names.get(0) {
            inputs.insert(input_name.to_string(), tensor);
        }
        
        inputs
    };
    
    // Run the model in a single thread first
    println!("Running model in a single thread...");
    let outputs = engine.run(create_sample_input())?;
    
    println!("Single thread outputs: {} output tensors", outputs.len());
    
    // Now run concurrent inference on multiple threads
    println!("Running model concurrently on multiple threads...");
    
    const NUM_THREADS: usize = 8;
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::with_capacity(NUM_THREADS);
    
    for thread_id in 0..NUM_THREADS {
        let engine_clone = engine.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            println!("Thread {} preparing", thread_id);
            
            // Create input for this thread
            let inputs = create_sample_input();
            
            // Wait for all threads to reach this point
            barrier_clone.wait();
            
            println!("Thread {} starting inference", thread_id);
            
            // Run the model
            match engine_clone.run(inputs) {
                Ok(outputs) => {
                    println!("Thread {} finished with {} outputs", thread_id, outputs.len());
                    Ok(outputs)
                }
                Err(e) => {
                    println!("Thread {} error: {:?}", thread_id, e);
                    Err(e)
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut all_results = Vec::with_capacity(NUM_THREADS);
    for handle in handles {
        all_results.push(handle.join().expect("Thread panicked")?);
    }
    
    println!("All threads completed successfully");
    
    // You can also use the parallel execution method which runs nodes concurrently
    println!("Running with concurrent node execution...");
    let outputs = engine.run_concurrent(create_sample_input())?;
    
    println!("Concurrent execution outputs: {} output tensors", outputs.len());
    
    // If you need multiple completely independent model instances
    println!("Creating independent model instances...");
    let mut engines = Vec::with_capacity(4);
    
    for _ in 0..4 {
        let independent_engine = engine.clone_for_parallel_execution()?;
        engines.push(independent_engine);
    }
    
    println!("Created {} independent engines with their own contexts", engines.len());
    
    Ok(())
}