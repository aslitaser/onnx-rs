// Include the generated protobuf code
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

// Re-export commonly used proto types
pub use onnx::{
    // Core model structure
    ModelProto,
    GraphProto,
    NodeProto,
    
    // Tensor definitions
    TensorProto,
    SparseTensorProto,
    ValueInfoProto,
    
    // Attribute and type information
    AttributeProto,
    TensorShapeProto,
    TypeProto,
    
    // Operator and function definitions
    OperatorSetIdProto,
    FunctionProto,
    
    // Training and metadata information
    TrainingInfoProto,
    StringStringEntryProto,
    
    // Quantization 
    QuantizationAnnotation,
    TensorAnnotation,
};