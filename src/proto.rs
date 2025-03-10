// Include the generated protobuf code
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

// Re-export commonly used proto types
pub use onnx::{
    ModelProto,
    GraphProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    AttributeProto,
    TensorShapeProto,
    TypeProto,
    OperatorSetIdProto,
};