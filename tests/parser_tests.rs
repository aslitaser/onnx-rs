use onnx_parser::{
    OnnxModelLoader, 
    SchemaValidator, 
    GraphBuilder, 
    model::{OnnxModel, ModelMetadata, Graph, Node, DataType},
    proto::{ModelProto, GraphProto, NodeProto},
    error::Result,
};
use std::path::Path;
use std::collections::HashMap;

// Helper to create a simple test model
fn create_test_model() -> OnnxModel {
    let metadata = ModelMetadata {
        producer_name: "Test".to_string(),
        producer_version: "1.0".to_string(),
        domain: "".to_string(),
        model_version: 1,
        doc_string: "Test model".to_string(),
        graph_name: "test_graph".to_string(),
        ir_version: 7,
    };
    
    // Create a simple graph with a single Relu node
    let node = Node {
        id: 0,
        name: "relu_node".to_string(),
        op_type: "Relu".to_string(),
        domain: "".to_string(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let graph = Graph {
        name: "test_graph".to_string(),
        nodes: vec![node],
        inputs: vec![
            onnx_parser::model::TensorInfo {
                name: "X".to_string(),
                shape: vec![1, 3, 224, 224],
                data_type: DataType::Float,
                doc_string: "".to_string(),
            }
        ],
        outputs: vec![
            onnx_parser::model::TensorInfo {
                name: "Y".to_string(),
                shape: vec![1, 3, 224, 224],
                data_type: DataType::Float,
                doc_string: "".to_string(),
            }
        ],
        initializers: vec![],
        doc_string: "".to_string(),
    };
    
    let mut opset_imports = HashMap::new();
    opset_imports.insert("".to_string(), 13);
    
    OnnxModel {
        metadata,
        graph,
        opset_imports,
    }
}

#[test]
fn test_model_metadata_extraction() {
    let model = create_test_model();
    
    assert_eq!(model.metadata.producer_name, "Test");
    assert_eq!(model.metadata.producer_version, "1.0");
    assert_eq!(model.metadata.model_version, 1);
    assert_eq!(model.metadata.ir_version, 7);
}

#[test]
fn test_get_input_output_info() {
    let model = create_test_model();
    
    let inputs = OnnxModelLoader::get_input_info(&model);
    let outputs = OnnxModelLoader::get_output_info(&model);
    
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
    
    assert_eq!(inputs[0].name, "X");
    assert_eq!(inputs[0].shape, vec![1, 3, 224, 224]);
    
    assert_eq!(outputs[0].name, "Y");
    assert_eq!(outputs[0].shape, vec![1, 3, 224, 224]);
}

#[test]
fn test_schema_validation() {
    let model = create_test_model();
    let validator = SchemaValidator::new();
    
    // Since we created a simple valid model with a Relu op, this should pass
    let result = validator.validate_model(&model);
    assert!(result.is_ok());
}

#[test]
fn test_graph_building() {
    let model = create_test_model();
    
    let result = GraphBuilder::build_graph(&model);
    assert!(result.is_ok());
    
    let graph = result.unwrap();
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.input_nodes.len(), 1);
    assert_eq!(graph.output_nodes.len(), 1);
}

#[test]
fn test_topological_sort() {
    // Create a more complex graph with multiple nodes
    let model = create_multi_node_test_model();
    
    let result = GraphBuilder::topological_sort(model.graph.nodes.clone());
    assert!(result.is_ok());
    
    let sorted_nodes = result.unwrap();
    
    // Verify the topological order:
    // Node 0 must come before Node 1 in the sorted result
    // Node 1 must come before Node 2 in the sorted result
    let node0_idx = sorted_nodes.iter().position(|n| n.id == 0).unwrap();
    let node1_idx = sorted_nodes.iter().position(|n| n.id == 1).unwrap();
    let node2_idx = sorted_nodes.iter().position(|n| n.id == 2).unwrap();
    
    assert!(node0_idx < node1_idx);
    assert!(node1_idx < node2_idx);
}

// Helper to create a more complex test model with multiple nodes that have dependencies
fn create_multi_node_test_model() -> OnnxModel {
    let metadata = ModelMetadata {
        producer_name: "Test".to_string(),
        producer_version: "1.0".to_string(),
        domain: "".to_string(),
        model_version: 1,
        doc_string: "Test model with multiple nodes".to_string(),
        graph_name: "multi_node_graph".to_string(),
        ir_version: 7,
    };
    
    // Create three nodes with sequential dependencies: node0 -> node1 -> node2
    let node0 = Node {
        id: 0,
        name: "conv_node".to_string(),
        op_type: "Conv".to_string(),
        domain: "".to_string(),
        inputs: vec!["input".to_string(), "weight".to_string()],
        outputs: vec!["conv_output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let node1 = Node {
        id: 1,
        name: "relu_node".to_string(),
        op_type: "Relu".to_string(),
        domain: "".to_string(),
        inputs: vec!["conv_output".to_string()], // Takes output from node0
        outputs: vec!["relu_output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let node2 = Node {
        id: 2,
        name: "pool_node".to_string(),
        op_type: "MaxPool".to_string(),
        domain: "".to_string(),
        inputs: vec!["relu_output".to_string()], // Takes output from node1
        outputs: vec!["output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let graph = Graph {
        name: "multi_node_graph".to_string(),
        nodes: vec![node2, node0, node1], // Intentionally out of order
        inputs: vec![
            onnx_parser::model::TensorInfo {
                name: "input".to_string(),
                shape: vec![1, 3, 224, 224],
                data_type: DataType::Float,
                doc_string: "".to_string(),
            }
        ],
        outputs: vec![
            onnx_parser::model::TensorInfo {
                name: "output".to_string(),
                shape: vec![1, 3, 112, 112],
                data_type: DataType::Float,
                doc_string: "".to_string(),
            }
        ],
        initializers: vec![],
        doc_string: "".to_string(),
    };
    
    let mut opset_imports = HashMap::new();
    opset_imports.insert("".to_string(), 13);
    
    OnnxModel {
        metadata,
        graph,
        opset_imports,
    }
}