# ONNX Parser & Validator

A Rust library for parsing, validating, and analyzing ONNX (Open Neural Network Exchange) models.

## Features

- Load ONNX models from files or byte arrays
- Validate models against operator schemas and ONNX specifications
- Extract metadata, inputs, and outputs from models
- Build computational graphs with topological sorting
- Analyze model structure and detect subgraphs

## Usage

### Loading a Model

```rust
use onnx_parser::{OnnxModelLoader, Result};
use std::path::Path;

fn main() -> Result<()> {
    // Load model from file
    let model_path = Path::new("path/to/model.onnx");
    let model = OnnxModelLoader::load_model(&model_path)?;
    
    // Get model metadata
    println!("Model producer: {}", model.metadata.producer_name);
    println!("Model version: {}", model.metadata.model_version);
    
    // Get input/output information
    let inputs = OnnxModelLoader::get_input_info(&model);
    let outputs = OnnxModelLoader::get_output_info(&model);
    
    println!("Inputs: {}", inputs.len());
    println!("Outputs: {}", outputs.len());
    
    Ok(())
}
```

### Validating a Model

```rust
use onnx_parser::{OnnxModelLoader, SchemaValidator, Result};
use std::path::Path;

fn main() -> Result<()> {
    // Load the model
    let model_path = Path::new("path/to/model.onnx");
    let model = OnnxModelLoader::load_model(&model_path)?;
    
    // Create a validator and validate the model
    let validator = SchemaValidator::new();
    validator.validate_model(&model)?;
    
    println!("Model is valid!");
    
    Ok(())
}
```

### Building an Execution Graph

```rust
use onnx_parser::{OnnxModelLoader, GraphBuilder, Result};
use std::path::Path;

fn main() -> Result<()> {
    // Load the model
    let model_path = Path::new("path/to/model.onnx");
    let model = OnnxModelLoader::load_model(&model_path)?;
    
    // Build execution graph
    let execution_graph = GraphBuilder::build_graph(&model)?;
    
    println!("Number of nodes: {}", execution_graph.nodes.len());
    println!("Input nodes: {:?}", execution_graph.input_nodes);
    println!("Output nodes: {:?}", execution_graph.output_nodes);
    
    Ok(())
}
```

## Architecture

The library consists of three main components:

1. **Model Loader**: Responsible for parsing ONNX files and converting them to an internal representation
2. **Schema Validator**: Validates models against operator schemas and the ONNX specification
3. **Graph Builder**: Builds a computational graph for execution, with topological sorting and dependency analysis

## Building

```
cargo build --release
```

## Testing

```
cargo test
```

## License

MIT License