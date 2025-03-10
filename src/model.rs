use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Unique identifier for a node in the graph
pub type NodeId = usize;

/// Metadata about the ONNX model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph_name: String,
    pub ir_version: i64,
}

/// Information about a tensor
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: DataType,
    pub doc_string: String,
}

/// ONNX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Undefined,
    Float,
    Double,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    String,
    Bool,
    Float16,
    Complex64,
    Complex128,
    BFloat16,
}

impl DataType {
    pub fn from_proto(proto_type: i32) -> Self {
        match proto_type {
            0 => DataType::Undefined,
            1 => DataType::Float,
            2 => DataType::Uint8,
            3 => DataType::Int8,
            4 => DataType::Uint16,
            5 => DataType::Int16,
            6 => DataType::Int32,
            7 => DataType::Int64,
            8 => DataType::String,
            9 => DataType::Bool,
            10 => DataType::Float16,
            11 => DataType::Double,
            12 => DataType::Uint32,
            13 => DataType::Uint64,
            14 => DataType::Complex64,
            15 => DataType::Complex128,
            16 => DataType::BFloat16,
            _ => DataType::Undefined,
        }
    }
}

/// Tensor data
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<i64>,
    pub data: Vec<u8>,
    pub doc_string: String,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Attribute>,
    pub doc_string: String,
}

/// Node attribute
#[derive(Debug, Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
}

/// Graph structure containing nodes and tensors
#[derive(Debug, Clone)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub initializers: Vec<Tensor>,
    pub doc_string: String,
}

/// The complete ONNX model
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub metadata: ModelMetadata,
    pub graph: Graph,
    pub opset_imports: HashMap<String, i64>,
}

/// Optimized execution graph
#[derive(Debug)]
pub struct ExecutionGraph {
    pub nodes: Vec<Node>,
    pub input_nodes: Vec<NodeId>,
    pub output_nodes: Vec<NodeId>,
    pub dependencies: HashMap<NodeId, Vec<NodeId>>,
}

/// Subgraph within the model
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub nodes: Vec<NodeId>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Operator schema for validation
#[derive(Debug, Clone)]
pub struct OpSchema {
    pub name: String,
    pub domain: String,
    pub since_version: i64,
    pub inputs: Vec<FormalParameter>,
    pub outputs: Vec<FormalParameter>,
    pub attributes: HashMap<String, AttributeProto>,
}

/// Formal parameter for operator schema
#[derive(Debug, Clone)]
pub struct FormalParameter {
    pub name: String,
    pub description: String,
    pub type_constraints: Option<Vec<DataType>>,
    pub optional: bool,
    pub variadic: bool,
}

/// Attribute prototype for operator schema
#[derive(Debug, Clone)]
pub struct AttributeProto {
    pub name: String,
    pub description: String,
    pub type_: AttributeType,
    pub required: bool,
}

/// Attribute type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    Float,
    Int,
    String,
    Tensor,
    Graph,
    Floats,
    Ints,
    Strings,
    Tensors,
    Graphs,
}