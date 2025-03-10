use std::collections::HashMap;
use std::path::Path;

use onnx_parser::{
    ExecutionEngine, ExecutionOptions, OptimizationLevel, ComputeTensor,
    parser::model_loader::ModelLoader,
    ops::tensor::DataType,
    Result,
};

fn main() -> Result<()> {
    // Load an ONNX model
    let model_path = Path::new("path/to/model.onnx");
    let model = ModelLoader::load_from_file(model_path)?;
    
    // Create execution options with optimization
    let options = ExecutionOptions::new()
        .set_optimization_level(OptimizationLevel::Standard)
        .enable_profiling(true);
    
    // Create execution engine
    let mut engine = ExecutionEngine::new(model, options)?;
    
    // Prepare the engine (builds execution graph, applies optimizations)
    engine.prepare()?;
    
    // Create input tensors
    let mut inputs = HashMap::new();
    
    // Example: Create a 1x3x224x224 float tensor (typical image input)
    let input_shape = vec![1, 3, 224, 224];
    let input_tensor = ComputeTensor::new(&input_shape, DataType::Float32);
    
    // Add the input tensor to the inputs map
    inputs.insert("input".to_string(), input_tensor);
    
    // Run the model
    let outputs = engine.run(inputs)?;
    
    // Process outputs
    for (name, tensor) in &outputs {
        println!("Output '{}': shape={:?}", name, tensor.shape);
    }
    
    // Print profiling information if enabled
    if options.enable_profiling {
        let profile_results = engine.profile_events();
        for event in profile_results {
            if let Some(duration) = event.duration {
                println!("{}: {:?}", event.name, duration);
            }
        }
    }
    
    Ok(())
}