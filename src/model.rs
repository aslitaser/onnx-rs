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
    // New 8-bit float types added in ONNX 1.13
    Float8E4M3FN,     // 8-bit floating-point, exponent 4, mantissa 3
    Float8E4M3FNUZ,   // 8-bit floating-point, exponent 4, mantissa 3, unsigned zero
    Float8E5M2,       // 8-bit floating-point, exponent 5, mantissa 2
    Float8E5M2FNUZ,   // 8-bit floating-point, exponent 5, mantissa 2, unsigned zero
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
            17 => DataType::Float8E4M3FN,
            18 => DataType::Float8E4M3FNUZ,
            19 => DataType::Float8E5M2,
            20 => DataType::Float8E5M2FNUZ,
            _ => DataType::Undefined,
        }
    }
    
    /// Check if the data type is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            DataType::Float
                | DataType::Double
                | DataType::Float16
                | DataType::BFloat16
                | DataType::Float8E4M3FN
                | DataType::Float8E4M3FNUZ
                | DataType::Float8E5M2
                | DataType::Float8E5M2FNUZ
        )
    }
    
    /// Check if the data type is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint8
                | DataType::Uint16
                | DataType::Uint32
                | DataType::Uint64
        )
    }
    
    /// Check if the data type is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(self, DataType::Complex64 | DataType::Complex128)
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
    // Optional external data reference for large tensors
    pub external_data: Option<ExternalDataInfo>,
    // Optional quantization information
    pub quant_info: Option<QuantInfo>,
}

/// External data information for large tensors
#[derive(Debug, Clone)]
pub struct ExternalDataInfo {
    pub data_location: String,
    pub offset: i64,
    pub length: i64,
    pub checksum: String,
    pub kvp_data: HashMap<String, String>,
}

/// Quantization information for tensors
#[derive(Debug, Clone)]
pub struct QuantInfo {
    pub scale: f32,
    pub zero_point: i64,
}

/// Sparse tensor representation
#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<i64>,
    pub indices: Tensor,  // Indices of non-zero values
    pub values: Tensor,   // Values of non-zero elements
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
    Graph(Graph),
    SparseTensor(SparseTensor),
    TypeProto(TypeInfo),
    
    // List types
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    Graphs(Vec<Graph>),
    SparseTensors(Vec<SparseTensor>),
    TypeProtos(Vec<TypeInfo>),
}

/// Type information for values
#[derive(Debug, Clone)]
pub enum TypeInfo {
    Tensor {
        elem_type: DataType,
        shape: Option<Vec<Dimension>>,
    },
    Sequence {
        elem_type: Box<TypeInfo>,
    },
    Map {
        key_type: DataType,
        value_type: Box<TypeInfo>,
    },
    Optional {
        elem_type: Box<TypeInfo>,
    },
    SparseTensor {
        elem_type: DataType,
        shape: Option<Vec<Dimension>>,
    },
}

/// Dimension information for tensor shapes
#[derive(Debug, Clone)]
pub enum Dimension {
    Value(i64),
    Param(String),
}

/// Graph structure containing nodes and tensors
#[derive(Debug, Clone)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub initializers: Vec<Tensor>,
    pub sparse_initializers: Vec<SparseTensor>,
    pub value_info: Vec<TensorInfo>,   // Intermediate values
    pub quantization_annotations: Vec<QuantizationAnnotation>,
    pub tensor_annotations: Vec<TensorAnnotation>,
    pub doc_string: String,
}

/// Quantization annotation for tensors
#[derive(Debug, Clone)]
pub struct QuantizationAnnotation {
    pub tensor_name: String,
    pub quant_parameter_tensor_name: Option<String>,
    pub axis: Option<i64>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i64>,
}

/// Tensor annotation for providing metadata about tensors
#[derive(Debug, Clone)]
pub struct TensorAnnotation {
    pub tensor_name: String,
    pub quant_parameter_tensor_names: HashMap<String, String>,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub domain: String,
    pub since_version: i64,
    pub doc_string: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<String>,
    pub nodes: Vec<Node>,
    pub opset_imports: HashMap<String, i64>,
    pub is_tensor_containment: bool,
}

/// Training info for the model
#[derive(Debug, Clone)]
pub struct TrainingInfo {
    pub algorithm: Graph,            // Training algorithm
    pub initialization: Option<Graph>, // Initialization algorithm
    pub inputs: Vec<TensorInfo>,      // Training inputs
    pub outputs: Vec<TensorInfo>,     // Training outputs
    pub metadata: HashMap<String, String>, // Metadata for training
}

/// The complete ONNX model
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub metadata: ModelMetadata,
    pub graph: Graph,
    pub opset_imports: HashMap<String, i64>,
    pub functions: Vec<Function>,
    pub training_info: Option<TrainingInfo>,
    pub metadata_props: HashMap<String, String>,
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
    Undefined,
    Float,
    Int,
    String,
    Tensor,
    Graph,
    SparseTensor,
    TypeProto,
    
    // List types
    Floats,
    Ints,
    Strings,
    Tensors,
    Graphs,
    SparseTensors,
    TypeProtos,
}