use onnx_parser::{OnnxModelLoader, SchemaValidator, GraphBuilder, Result};
use std::path::Path;
use std::env;

fn main() -> Result<()> {
    // Get the model path from command-line arguments or use a default
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        panic!("Please provide a path to an ONNX model file");
    };
    
    println!("Loading model from: {}", model_path.display());
    
    // Load the model
    let model = OnnxModelLoader::load_model(model_path)?;
    
    // Print model metadata
    println!("\nModel Metadata:");
    println!("---------------");
    println!("Producer: {}", model.metadata.producer_name);
    println!("Producer Version: {}", model.metadata.producer_version);
    println!("Domain: {}", model.metadata.domain);
    println!("Model Version: {}", model.metadata.model_version);
    println!("IR Version: {}", model.metadata.ir_version);
    
    // Print inputs
    let inputs = OnnxModelLoader::get_input_info(&model);
    println!("\nInputs: {}", inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        println!("  Input #{}: {} - {:?} - {:?}", 
            i, input.name, input.shape, input.data_type);
    }
    
    // Print outputs
    let outputs = OnnxModelLoader::get_output_info(&model);
    println!("\nOutputs: {}", outputs.len());
    for (i, output) in outputs.iter().enumerate() {
        println!("  Output #{}: {} - {:?} - {:?}", 
            i, output.name, output.shape, output.data_type);
    }
    
    // Print graph overview
    println!("\nGraph Overview:");
    println!("--------------");
    println!("Graph Name: {}", model.graph.name);
    println!("Nodes: {}", model.graph.nodes.len());
    println!("Initializers: {}", model.graph.initializers.len());
    
    // Validate the model
    println!("\nValidating model...");
    let validator = SchemaValidator::new();
    
    match validator.validate_model(&model) {
        Ok(_) => println!("Model validation successful!"),
        Err(e) => println!("Model validation failed: {}", e),
    }
    
    // Build execution graph
    println!("\nBuilding execution graph...");
    let execution_graph = GraphBuilder::build_graph(&model)?;
    
    println!("Execution graph built successfully!");
    println!("Input nodes: {:?}", execution_graph.input_nodes);
    println!("Output nodes: {:?}", execution_graph.output_nodes);
    println!("Topologically sorted nodes: {}", execution_graph.nodes.len());
    
    // Print operators used in the model
    println!("\nOperators used:");
    let mut operators = std::collections::HashMap::new();
    
    for node in &model.graph.nodes {
        *operators.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    
    for (op, count) in &operators {
        println!("  {}: {} instance(s)", op, count);
    }
    
    Ok(())
}