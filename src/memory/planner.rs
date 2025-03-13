use std::collections::{HashMap, HashSet};
use std::cmp;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::model::{ExecutionGraph, NodeId, Node, Dimension, TypeInfo, DataType as ModelDataType, Attribute};
use crate::ops::tensor::DataType;
use crate::memory::allocator::{MemoryAllocator, MemoryBlock};

/// Unique identifier for a tensor
pub type TensorId = usize;

/// Information about a tensor's memory requirements
#[derive(Debug, Clone)]
pub struct TensorMemoryInfo {
    /// Tensor ID
    pub id: TensorId,
    /// Tensor name
    pub name: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Data type
    pub data_type: DataType,
    /// Alignment requirement
    pub alignment: usize,
    /// Whether this tensor can be reused for in-place operations
    pub allow_inplace: bool,
}

/// In-place operation opportunity
#[derive(Debug, Clone)]
pub struct InplaceOpportunity {
    /// Node ID where the in-place operation can occur
    pub node_id: NodeId,
    /// Input tensor ID that can be overwritten
    pub input_id: TensorId,
    /// Output tensor ID that can share memory with the input
    pub output_id: TensorId,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Buffer sharing opportunity
#[derive(Debug, Clone)]
pub struct SharingOpportunity {
    /// First tensor ID
    pub first_id: TensorId,
    /// Second tensor ID
    pub second_id: TensorId,
    /// Size in bytes that can be shared
    pub size_bytes: usize,
}

/// Memory allocation plan for a tensor
#[derive(Debug, Clone)]
pub struct TensorAllocation {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Offset in the buffer
    pub offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Buffer index (which buffer this tensor is allocated in)
    pub buffer_index: usize,
}

/// Complete memory plan for the execution graph
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Tensor allocations
    pub allocations: HashMap<TensorId, TensorAllocation>,
    /// Tensor information
    pub tensor_info: HashMap<TensorId, TensorMemoryInfo>,
    /// Tensor lifetimes (first_use, last_use)
    pub lifetimes: HashMap<TensorId, (usize, usize)>,
    /// Buffer sizes
    pub buffer_sizes: Vec<usize>,
    /// In-place opportunities that were used
    pub inplace_ops: Vec<InplaceOpportunity>,
    /// Total memory requirement in bytes
    pub total_memory_bytes: usize,
    /// Execution order for nodes
    pub execution_order: Vec<NodeId>,
}

/// Mapping from tensor ID to memory block
pub type BufferMap = HashMap<TensorId, MemoryBlock>;

/// Represents a tensor dimension which may be static or dynamic
#[derive(Debug, Clone)]
pub enum ShapeDim {
    /// Static dimension with known size
    Static(usize),
    /// Dynamic dimension (size unknown at planning time)
    Dynamic,
    /// Symbolic dimension with a name
    Symbolic(String),
}

impl ShapeDim {
    /// Convert to a concrete size, using default_value for dynamic dimensions
    pub fn to_size(&self, default_value: usize) -> usize {
        match self {
            ShapeDim::Static(size) => *size,
            ShapeDim::Dynamic | ShapeDim::Symbolic(_) => default_value,
        }
    }
    
    /// Check if this dimension has a static size
    pub fn is_static(&self) -> bool {
        matches!(self, ShapeDim::Static(_))
    }
    
    /// Check if this dimension is dynamic
    pub fn is_dynamic(&self) -> bool {
        matches!(self, ShapeDim::Dynamic)
    }
    
    /// Check if this dimension is symbolic
    pub fn is_symbolic(&self) -> bool {
        matches!(self, ShapeDim::Symbolic(_))
    }
    
    /// Get static value, if available
    pub fn static_value(&self) -> Option<usize> {
        match self {
            ShapeDim::Static(size) => Some(*size),
            _ => None,
        }
    }
}

/// Tensor shape with potentially dynamic dimensions
#[derive(Debug, Clone)]
pub struct TensorShape {
    /// Dimensions of the tensor
    pub dims: Vec<ShapeDim>,
}

impl TensorShape {
    /// Create a new static shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(ShapeDim::Static).collect(),
        }
    }
    
    /// Create a shape with dynamic dimensions
    pub fn with_dynamic_dims(dims: Vec<Option<usize>>) -> Self {
        Self {
            dims: dims.into_iter()
                .map(|d| match d {
                    Some(size) => ShapeDim::Static(size),
                    None => ShapeDim::Dynamic,
                })
                .collect(),
        }
    }
    
    /// Create a shape from model dimensions
    pub fn from_model_dims(dims: &[Dimension]) -> Self {
        Self {
            dims: dims.iter()
                .map(|dim| match dim {
                    Dimension::Value(size) if *size >= 0 => ShapeDim::Static(*size as usize),
                    Dimension::Value(_) => ShapeDim::Dynamic, // negative value indicates dynamic
                    Dimension::Param(name) => ShapeDim::Symbolic(name.clone()),
                })
                .collect(),
        }
    }
    
    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    
    /// Get dimensions as concrete sizes, using default_value for dynamic dimensions
    pub fn concrete_dims(&self, default_value: usize) -> Vec<usize> {
        self.dims.iter().map(|d| d.to_size(default_value)).collect()
    }
    
    /// Calculate the total number of elements
    pub fn num_elements(&self, default_value: usize) -> usize {
        self.concrete_dims(default_value).iter().product()
    }
    
    /// Check if all dimensions are static
    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }
    
    /// Try to convert from model TypeInfo
    pub fn from_type_info(type_info: &TypeInfo) -> Option<Self> {
        match type_info {
            TypeInfo::Tensor { shape, .. } => {
                shape.as_ref().map(|dims| Self::from_model_dims(dims))
            },
            TypeInfo::SparseTensor { shape, .. } => {
                shape.as_ref().map(|dims| Self::from_model_dims(dims))
            },
            _ => None,
        }
    }
}

/// Shape and type inference helper for the memory planner
#[derive(Debug)]
pub struct ShapeInferenceSystem {
    /// Cache of inferred shapes for tensors
    shape_cache: RwLock<HashMap<String, TensorShape>>,
    /// Cache of data types for tensors
    data_type_cache: RwLock<HashMap<String, DataType>>,
}

impl ShapeInferenceSystem {
    /// Create a new shape and type inference system
    pub fn new() -> Self {
        Self {
            shape_cache: RwLock::new(HashMap::new()),
            data_type_cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// Clear all caches
    pub fn clear_caches(&self) {
        if let Ok(mut cache) = self.shape_cache.write() {
            cache.clear();
        }
        if let Ok(mut cache) = self.data_type_cache.write() {
            cache.clear();
        }
    }
    
    /// Get the operator schema type constraints
    fn get_operator_type_constraints(&self, op_type: &str) -> HashMap<String, TypeConstraint> {
        let mut constraints = HashMap::new();
        
        match op_type {
            "Conv" | "ConvTranspose" => {
                // Convolution typically supports float types and selected integer types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32,
                        DataType::UInt8, DataType::BFloat16,
                    ],
                    description: "Input and output types for convolution".to_string(),
                });
            },
            "MatMul" | "Gemm" => {
                // Matrix multiplication supports float and integer types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::BFloat16,
                    ],
                    description: "Input and output types for matrix multiplication".to_string(),
                });
            },
            "Cast" => {
                // Input can be any data type
                constraints.insert("T1".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::Bool, DataType::String, DataType::BFloat16,
                        DataType::Complex64, DataType::Complex128,
                    ],
                    description: "Input type for cast".to_string(),
                });
                
                // Output can be any data type
                constraints.insert("T2".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::Bool, DataType::String, DataType::BFloat16,
                        DataType::Complex64, DataType::Complex128,
                    ],
                    description: "Output type for cast".to_string(),
                });
            },
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" | "Elu" | "Softmax" => {
                // Activation functions typically support float types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::BFloat16,
                    ],
                    description: "Input and output types for activation functions".to_string(),
                });
            },
            "MaxPool" | "AveragePool" | "GlobalMaxPool" | "GlobalAveragePool" => {
                // Pooling operators typically support float types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::UInt8, DataType::BFloat16,
                    ],
                    description: "Input and output types for pooling".to_string(),
                });
                
                // Indices output for MaxPool should be int64
                if op_type == "MaxPool" {
                    constraints.insert("I".to_string(), TypeConstraint {
                        allowed_types: vec![DataType::Int64],
                        description: "Type for maxpool indices output".to_string(),
                    });
                }
            },
            "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
                // Element-wise operators support numeric types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::BFloat16,
                    ],
                    description: "Input and output types for element-wise operations".to_string(),
                });
            },
            "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" => {
                // Comparison operators support numeric types for input
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::BFloat16, DataType::String, DataType::Bool,
                    ],
                    description: "Input types for comparison operations".to_string(),
                });
                
                // Output is always boolean
                constraints.insert("T1".to_string(), TypeConstraint {
                    allowed_types: vec![DataType::Bool],
                    description: "Output type for comparison operations".to_string(),
                });
            },
            "Concat" | "Slice" | "Reshape" | "Transpose" | "Flatten" | "Squeeze" | "Unsqueeze" => {
                // Shape manipulation operators support all types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::Bool, DataType::String, DataType::BFloat16,
                        DataType::Complex64, DataType::Complex128,
                    ],
                    description: "Input and output types for shape operations".to_string(),
                });
                
                // Some shape operators use indices
                if op_type == "Slice" || op_type == "Squeeze" || op_type == "Unsqueeze" {
                    constraints.insert("Tind".to_string(), TypeConstraint {
                        allowed_types: vec![DataType::Int32, DataType::Int64],
                        description: "Index type for shape operations".to_string(),
                    });
                }
            },
            "BatchNormalization" => {
                // BatchNorm supports float types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::BFloat16,
                    ],
                    description: "Input and output types for batch normalization".to_string(),
                });
            },
            "Resize" => {
                // Resize supports float and integer types for input/output
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::BFloat16,
                    ],
                    description: "Input and output types for resize".to_string(),
                });
                
                // Scales should be float
                constraints.insert("T1".to_string(), TypeConstraint {
                    allowed_types: vec![DataType::Float32, DataType::Float64],
                    description: "Scale type for resize".to_string(),
                });
            },
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
                // Reduction operators support numeric types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::BFloat16,
                    ],
                    description: "Input and output types for reduction operations".to_string(),
                });
            },
            // Default case for any other operator
            _ => {
                // Generic constraint covering most types
                constraints.insert("T".to_string(), TypeConstraint {
                    allowed_types: vec![
                        DataType::Float16, DataType::Float32, DataType::Float64,
                        DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                        DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
                        DataType::Bool, DataType::BFloat16,
                    ],
                    description: "Generic type constraint".to_string(),
                });
            }
        }
        
        constraints
    }
    
    /// Get the type inference rule for an operator
    fn get_type_inference_rule(&self, op_type: &str) -> TypeInferenceResult {
        match op_type {
            // Identity operators (output has same type as input)
            "Identity" | "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" | "Elu" | "Softmax" |
            "LogSoftmax" | "Softplus" | "MaxPool" | "AveragePool" | "GlobalMaxPool" |
            "GlobalAveragePool" | "Dropout" | "Reshape" | "Flatten" | "Squeeze" |
            "Unsqueeze" | "Transpose" | "Slice" | "Pad" | "BatchNormalization" |
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
                TypeInferenceResult::SameAsInput
            },
            
            // Element-wise binary operators (promote input types)
            "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    if input_types.len() < 2 || input_types[0].is_none() || input_types[1].is_none() {
                        return Ok(vec![DataType::Float32]); // Default if we can't determine
                    }
                    
                    let type1 = input_types[0].unwrap();
                    let type2 = input_types[1].unwrap();
                    
                    // Check if types are compatible
                    if !are_types_compatible(type1, type2, true) {
                        return Err(Error::InvalidModel(format!(
                            "Incompatible types for binary operation: {:?} and {:?}",
                            type1, type2
                        )));
                    }
                    
                    // Promote types according to standard rules
                    let result_type = promote_types(type1, type2, TypePromotion::HighestPrecision);
                    Ok(vec![result_type])
                })
            },
            
            // Comparison operators (output is always boolean)
            "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" => {
                TypeInferenceResult::SpecificType(DataType::Bool)
            },
            
            // Cast operator (output type depends on attributes)
            "Cast" => {
                TypeInferenceResult::Function(|input_types, node| {
                    // Get the 'to' attribute to determine the output type
                    let to_attribute = node.attributes
                        .get("to")
                        .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i as i32) } else { None });
                    
                    if let Some(to_value) = to_attribute {
                        // Map the 'to' value to a DataType
                        match to_value {
                            1 => Ok(vec![DataType::Float32]),
                            2 => Ok(vec![DataType::UInt8]),
                            3 => Ok(vec![DataType::Int8]),
                            4 => Ok(vec![DataType::UInt16]),
                            5 => Ok(vec![DataType::Int16]),
                            6 => Ok(vec![DataType::Int32]),
                            7 => Ok(vec![DataType::Int64]),
                            8 => Ok(vec![DataType::String]),
                            9 => Ok(vec![DataType::Bool]),
                            10 => Ok(vec![DataType::Float16]),
                            11 => Ok(vec![DataType::Float64]),
                            12 => Ok(vec![DataType::UInt32]),
                            13 => Ok(vec![DataType::UInt64]),
                            14 => Ok(vec![DataType::Complex64]),
                            15 => Ok(vec![DataType::Complex128]),
                            16 => Ok(vec![DataType::BFloat16]),
                            _ => Err(Error::InvalidModel(format!(
                                "Unsupported 'to' attribute value for Cast: {}", to_value
                            ))),
                        }
                    } else {
                        // If the 'to' attribute is missing, return an error
                        Err(Error::InvalidModel("Missing 'to' attribute for Cast operator".to_string()))
                    }
                })
            },
            
            // Concat operator (use the type of the first input)
            "Concat" => TypeInferenceResult::SameAsInput,
            
            // MatMul operator (output type depends on input types)
            "MatMul" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    if input_types.len() < 2 || input_types[0].is_none() || input_types[1].is_none() {
                        return Ok(vec![DataType::Float32]); // Default if we can't determine
                    }
                    
                    let type1 = input_types[0].unwrap();
                    let type2 = input_types[1].unwrap();
                    
                    // For MatMul, check if both inputs are numeric
                    if !is_floating_point(type1) && !is_integer(type1) {
                        return Err(Error::InvalidModel(format!(
                            "MatMul input A has non-numeric type: {:?}", type1
                        )));
                    }
                    
                    if !is_floating_point(type2) && !is_integer(type2) {
                        return Err(Error::InvalidModel(format!(
                            "MatMul input B has non-numeric type: {:?}", type2
                        )));
                    }
                    
                    // Promote types according to standard rules
                    let result_type = promote_types(type1, type2, TypePromotion::HighestPrecision);
                    Ok(vec![result_type])
                })
            },
            
            // Gemm operator (similar to MatMul)
            "Gemm" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    if input_types.len() < 2 || input_types[0].is_none() || input_types[1].is_none() {
                        return Ok(vec![DataType::Float32]); // Default if we can't determine
                    }
                    
                    let type1 = input_types[0].unwrap();
                    let type2 = input_types[1].unwrap();
                    
                    // For Gemm, check if both inputs are numeric
                    if !is_floating_point(type1) && !is_integer(type1) {
                        return Err(Error::InvalidModel(format!(
                            "Gemm input A has non-numeric type: {:?}", type1
                        )));
                    }
                    
                    if !is_floating_point(type2) && !is_integer(type2) {
                        return Err(Error::InvalidModel(format!(
                            "Gemm input B has non-numeric type: {:?}", type2
                        )));
                    }
                    
                    // If there's a C input, consider its type as well
                    if input_types.len() > 2 && input_types[2].is_some() {
                        let type3 = input_types[2].unwrap();
                        
                        if !is_floating_point(type3) && !is_integer(type3) {
                            return Err(Error::InvalidModel(format!(
                                "Gemm input C has non-numeric type: {:?}", type3
                            )));
                        }
                        
                        // Result type is highest precision among A, B, and C
                        let result_type = promote_types(
                            promote_types(type1, type2, TypePromotion::HighestPrecision),
                            type3,
                            TypePromotion::HighestPrecision
                        );
                        return Ok(vec![result_type]);
                    }
                    
                    // If there's no C input, result type is highest precision of A and B
                    let result_type = promote_types(type1, type2, TypePromotion::HighestPrecision);
                    Ok(vec![result_type])
                })
            },
            
            // Conv operator (similar to MatMul but with more constraints)
            "Conv" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    if input_types.len() < 2 || input_types[0].is_none() || input_types[1].is_none() {
                        return Ok(vec![DataType::Float32]); // Default if we can't determine
                    }
                    
                    let type1 = input_types[0].unwrap();
                    let type2 = input_types[1].unwrap();
                    
                    // For Conv, check if both inputs are numeric
                    if !is_floating_point(type1) && !is_integer(type1) {
                        return Err(Error::InvalidModel(format!(
                            "Conv input X has non-numeric type: {:?}", type1
                        )));
                    }
                    
                    if !is_floating_point(type2) && !is_integer(type2) {
                        return Err(Error::InvalidModel(format!(
                            "Conv input W has non-numeric type: {:?}", type2
                        )));
                    }
                    
                    // If there's a bias input, consider its type as well
                    if input_types.len() > 2 && input_types[2].is_some() {
                        let type3 = input_types[2].unwrap();
                        
                        if !is_floating_point(type3) && !is_integer(type3) {
                            return Err(Error::InvalidModel(format!(
                                "Conv input B has non-numeric type: {:?}", type3
                            )));
                        }
                        
                        // Result type is highest precision among X, W, and B
                        let result_type = promote_types(
                            promote_types(type1, type2, TypePromotion::HighestPrecision),
                            type3,
                            TypePromotion::HighestPrecision
                        );
                        return Ok(vec![result_type]);
                    }
                    
                    // If there's no bias input, result type is highest precision of X and W
                    let result_type = promote_types(type1, type2, TypePromotion::HighestPrecision);
                    Ok(vec![result_type])
                })
            },
            
            // QuantizeLinear and DequantizeLinear operators
            "QuantizeLinear" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    // Input is float, output is int (typically int8)
                    // The default quantized type is INT8
                    Ok(vec![DataType::Int8])
                })
            },
            
            "DequantizeLinear" => {
                TypeInferenceResult::Function(|input_types, _node| {
                    // Input is int, output is float (typically float32)
                    Ok(vec![DataType::Float32])
                })
            },
            
            // Split operator (all outputs have same type as input)
            "Split" => TypeInferenceResult::SameAsInput,
            
            // Resize operator (output has same type as input)
            "Resize" => TypeInferenceResult::SameAsInput,
            
            // Default case for any other operator
            _ => {
                // We don't have specific type inference rules for this operator
                // Default to preserving the first input's type
                TypeInferenceResult::SameAsInput
            }
        }
    }
    
    /// Validate type constraints for a node
    fn validate_type_constraints(&self, node: &Node, input_types: &[Option<DataType>]) -> Result<()> {
        let constraints = self.get_operator_type_constraints(&node.op_type);
        
        // If we don't have constraints for this operator, skip validation
        if constraints.is_empty() {
            return Ok(());
        }
        
        // Check if all inputs satisfy their constraints
        let mut constraint_key = "T";
        
        for (i, input_type_opt) in input_types.iter().enumerate() {
            if let Some(input_type) = input_type_opt {
                // For some operators, different inputs may have different constraints
                if node.op_type == "Cast" && i == 0 {
                    constraint_key = "T1";
                } else if node.op_type == "Cast" && i == 1 {
                    constraint_key = "T2";
                } else if (node.op_type == "MaxPool" || node.op_type == "ArgMax" || node.op_type == "ArgMin") && i == 1 {
                    constraint_key = "I";
                }
                
                if let Some(constraint) = constraints.get(constraint_key) {
                    if !constraint.allowed_types.contains(input_type) {
                        return Err(Error::InvalidModel(format!(
                            "Input {} of type {:?} for operator {} does not meet constraint {}: {:?}",
                            i, input_type, node.op_type, constraint_key, constraint.allowed_types
                        )));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Initialize shape information from a graph
    pub fn initialize_from_graph(&self, graph: &ExecutionGraph) -> Result<()> {
        if let Ok(mut shape_cache) = self.shape_cache.write() {
            if let Ok(mut data_type_cache) = self.data_type_cache.write() {
                // Initialize shapes and data types from model inputs/outputs
                if let Some(onnx_model) = &graph.onnx_model {
                    // Add shapes from model inputs
                    for input in &onnx_model.graph.inputs {
                        if let Some(model_type) = &input.type_info {
                            // Get tensor shape from type info
                            if let Some(shape) = TensorShape::from_type_info(model_type) {
                                shape_cache.insert(input.name.clone(), shape);
                            }
                            
                            // Get data type from type info
                            if let TypeInfo::Tensor { elem_type, .. } = model_type {
                                let data_type = convert_model_data_type(*elem_type);
                                data_type_cache.insert(input.name.clone(), data_type);
                            }
                        }
                    }
                    
                    // Add shapes from value_info (intermediate tensors)
                    for value in &onnx_model.graph.value_info {
                        if let Some(model_type) = &value.type_info {
                            if let Some(shape) = TensorShape::from_type_info(model_type) {
                                shape_cache.insert(value.name.clone(), shape);
                            }
                            
                            if let TypeInfo::Tensor { elem_type, .. } = model_type {
                                let data_type = convert_model_data_type(*elem_type);
                                data_type_cache.insert(value.name.clone(), data_type);
                            }
                        }
                    }
                    
                    // Add shapes from model outputs
                    for output in &onnx_model.graph.outputs {
                        if let Some(model_type) = &output.type_info {
                            if let Some(shape) = TensorShape::from_type_info(model_type) {
                                shape_cache.insert(output.name.clone(), shape);
                            }
                            
                            if let TypeInfo::Tensor { elem_type, .. } = model_type {
                                let data_type = convert_model_data_type(*elem_type);
                                data_type_cache.insert(output.name.clone(), data_type);
                            }
                        }
                    }
                    
                    // Add initializers 
                    for initializer in &onnx_model.graph.initializers {
                        // Convert i64 dims to usize dims
                        let dims: Vec<usize> = initializer.dims
                            .iter()
                            .map(|&dim| if dim >= 0 { dim as usize } else { 0 })
                            .collect();
                        
                        shape_cache.insert(initializer.name.clone(), TensorShape::new(dims));
                        data_type_cache.insert(
                            initializer.name.clone(), 
                            convert_model_data_type(initializer.data_type)
                        );
                    }
                }
                
                Ok(())
            } else {
                Err(Error::InvalidModel("Failed to acquire write lock for data_type_cache".to_string()))
            }
        } else {
            Err(Error::InvalidModel("Failed to acquire write lock for shape_cache".to_string()))
        }
    }
    
    /// Infer shapes and types for all tensors in a graph
    pub fn infer_shapes(&self, graph: &ExecutionGraph) -> Result<()> {
        // Make sure we have initialized shapes from the model
        self.initialize_from_graph(graph)?;
        
        // To ensure we infer shapes in the correct order, we use the topological sort
        let execution_order = self.topological_sort(graph)?;
        
        // Process nodes in topological order
        for node_id in execution_order {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;
            
            // Infer output shapes for this node
            self.infer_node_output_shapes(graph, node)?;
            
            // Infer output types for this node
            self.infer_node_output_types(graph, node)?;
        }
        
        Ok(())
    }
    
    /// Infer data types for all tensors in a graph
    pub fn infer_types(&self, graph: &ExecutionGraph) -> Result<()> {
        // Make sure we have initialized types from the model
        self.initialize_from_graph(graph)?;
        
        // To ensure we infer types in the correct order, we use the topological sort
        let execution_order = self.topological_sort(graph)?;
        
        // Process nodes in topological order
        for node_id in execution_order {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;
            
            // Infer output types for this node
            self.infer_node_output_types(graph, node)?;
        }
        
        Ok(())
    }
    
    /// Infer output types for a single node
    pub fn infer_node_output_types(&self, graph: &ExecutionGraph, node: &Node) -> Result<()> {
        // Get input types
        let mut input_types = Vec::new();
        for input_name in &node.inputs {
            if input_name.is_empty() {
                // Optional input, use None
                input_types.push(None);
            } else {
                // Try to get type from cache
                if let Ok(type_cache) = self.data_type_cache.read() {
                    let data_type = type_cache.get(input_name).cloned();
                    input_types.push(data_type);
                }
            }
        }
        
        // Validate that input types meet operator constraints
        self.validate_type_constraints(node, &input_types)?;
        
        // Get the type inference rule for this operation
        let type_rule = self.get_type_inference_rule(&node.op_type);
        
        // Infer output types
        let output_types = match type_rule {
            TypeInferenceResult::SameAsInput => {
                // Use the type of the first input for all outputs
                if input_types.is_empty() || input_types[0].is_none() {
                    // Default to Float32
                    vec![DataType::Float32; node.outputs.len()]
                } else {
                    // All outputs have the same type as the first input
                    vec![input_types[0].unwrap(); node.outputs.len()]
                }
            }
            TypeInferenceResult::SpecificType(dtype) => {
                // All outputs have the specified type
                vec![dtype; node.outputs.len()]
            }
            TypeInferenceResult::MultipleTypes(types) => {
                // Each output has a specific type
                if types.len() >= node.outputs.len() {
                    types[0..node.outputs.len()].to_vec()
                } else {
                    // Not enough types specified, repeat the last one
                    let mut result = types.clone();
                    if let Some(last_type) = types.last() {
                        result.resize(node.outputs.len(), *last_type);
                    } else {
                        // No types specified, default to Float32
                        result = vec![DataType::Float32; node.outputs.len()];
                    }
                    result
                }
            }
            TypeInferenceResult::Function(func) => {
                // Call the function to determine output types
                match func(&input_types, node) {
                    Ok(types) => {
                        if types.len() >= node.outputs.len() {
                            types[0..node.outputs.len()].to_vec()
                        } else {
                            // Not enough types returned, use Float32 for missing ones
                            let mut result = types;
                            result.resize(node.outputs.len(), DataType::Float32);
                            result
                        }
                    }
                    Err(e) => {
                        // Type inference failed, log the error and use Float32
                        eprintln!("Type inference failed for node {}: {}", node.id, e);
                        vec![DataType::Float32; node.outputs.len()]
                    }
                }
            }
        };
        
        // Update cache with inferred types
        if let Ok(mut type_cache) = self.data_type_cache.write() {
            for (i, output_name) in node.outputs.iter().enumerate() {
                if i < output_types.len() {
                    type_cache.insert(output_name.clone(), output_types[i]);
                }
            }
        }
        
        Ok(())
    }
    
    /// Infer output shapes for a single node
    pub fn infer_node_output_shapes(&self, graph: &ExecutionGraph, node: &Node) -> Result<()> {
        // Get input shapes
        let mut input_shapes = Vec::new();
        for input_name in &node.inputs {
            if input_name.is_empty() {
                // Optional input, use None
                input_shapes.push(None);
            } else {
                // Try to get shape from cache
                if let Ok(shape_cache) = self.shape_cache.read() {
                    let shape = shape_cache.get(input_name).cloned();
                    input_shapes.push(shape);
                }
            }
        }
        
        // Based on the operation type, infer output shapes
        let output_shapes = match node.op_type.as_str() {
            "Conv" => self.infer_conv_shape(node, &input_shapes)?,
            "MaxPool" | "AveragePool" | "LpPool" => self.infer_pool_shape(node, &input_shapes)?,
            "GlobalMaxPool" | "GlobalAveragePool" | "GlobalLpPool" => 
                self.infer_global_pool_shape(node, &input_shapes)?,
            "MatMul" => self.infer_matmul_shape(node, &input_shapes)?,
            "Gemm" => self.infer_gemm_shape(node, &input_shapes)?,
            "Add" | "Sub" | "Mul" | "Div" | "Pow" => 
                self.infer_binary_elementwise_shape(node, &input_shapes)?,
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" | "Softmax" | "LogSoftmax" => 
                self.infer_unary_elementwise_shape(node, &input_shapes)?,
            "Concat" => self.infer_concat_shape(node, &input_shapes)?,
            "Reshape" => self.infer_reshape_shape(node, &input_shapes)?,
            "Transpose" => self.infer_transpose_shape(node, &input_shapes)?,
            "Flatten" => self.infer_flatten_shape(node, &input_shapes)?,
            "Slice" => self.infer_slice_shape(node, &input_shapes)?,
            "Squeeze" => self.infer_squeeze_shape(node, &input_shapes)?,
            "Unsqueeze" => self.infer_unsqueeze_shape(node, &input_shapes)?,
            "Resize" => self.infer_resize_shape(node, &input_shapes)?,
            // If we don't have a specific handler, try to infer by using the first input shape
            _ => self.infer_default_shape(node, &input_shapes)?,
        };
        
        // Update cache with inferred shapes
        if let Ok(mut shape_cache) = self.shape_cache.write() {
            for (i, output_name) in node.outputs.iter().enumerate() {
                if i < output_shapes.len() {
                    if let Some(shape) = &output_shapes[i] {
                        shape_cache.insert(output_name.clone(), shape.clone());
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Infer output shapes for Conv operator
    fn infer_conv_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let x_shape = input_shapes[0].as_ref().unwrap();
        
        // Need at least X and W inputs
        if input_shapes.len() < 2 || input_shapes[1].is_none() {
            return Ok(vec![None]);
        }
        
        let w_shape = input_shapes[1].as_ref().unwrap();
        
        // Validate input ranks
        if x_shape.rank() < 3 {
            return Err(Error::InvalidModel(format!(
                "Conv input X must have at least 3 dimensions, got {}", x_shape.rank()
            )));
        }
        
        if w_shape.rank() < 3 {
            return Err(Error::InvalidModel(format!(
                "Conv weight W must have at least 3 dimensions, got {}", w_shape.rank()
            )));
        }
        
        // Get attributes
        let auto_pad = node.attributes
            .get("auto_pad")
            .and_then(|attr| if let Attribute::String(s) = attr { Some(s.as_str()) } else { None })
            .unwrap_or("NOTSET");
            
        let mut pads = node.attributes
            .get("pads")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![0; (x_shape.rank() - 2) * 2]);
            
        let strides = node.attributes
            .get("strides")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![1; x_shape.rank() - 2]);
            
        let dilations = node.attributes
            .get("dilations")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![1; x_shape.rank() - 2]);
            
        let group = node.attributes
            .get("group")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i as usize) } else { None })
            .unwrap_or(1);
            
        let kernel_shape = node.attributes
            .get("kernel_shape")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| {
                // If kernel_shape is not specified, infer from weight tensor shape
                w_shape.dims.iter()
                    .skip(2)  // Skip output channels and input channels/group
                    .filter_map(|d| d.static_value())
                    .collect()
            });
            
        // Make sure pads has the right length
        if pads.len() < (x_shape.rank() - 2) * 2 {
            pads.resize((x_shape.rank() - 2) * 2, 0);
        }
        
        // Handle auto_pad
        if auto_pad != "NOTSET" {
            // Reset pads for auto padding
            pads = vec![0; (x_shape.rank() - 2) * 2];
            
            // For each spatial dimension, compute padding
            for i in 0..(x_shape.rank() - 2) {
                if let (Some(input_size), Some(kernel_size)) = (
                    x_shape.dims[i + 2].static_value(),
                    kernel_shape.get(i).copied()
                ) {
                    let dilated_kernel_size = 1 + (kernel_size - 1) * dilations[i];
                    let stride = strides[i];
                    
                    let output_size = (input_size + stride - 1) / stride;
                    let pad_needed = std::cmp::max(
                        0, 
                        (output_size - 1) * stride + dilated_kernel_size - input_size
                    );
                    
                    let pad_head = pad_needed / 2;
                    let pad_tail = pad_needed - pad_head;
                    
                    if auto_pad == "SAME_UPPER" {
                        pads[i] = pad_head;
                        pads[i + (x_shape.rank() - 2)] = pad_tail;
                    } else if auto_pad == "SAME_LOWER" {
                        pads[i] = pad_tail;
                        pads[i + (x_shape.rank() - 2)] = pad_head;
                    }
                    // For "VALID", padding is 0 (already set)
                }
            }
        }
        
        // Compute output shape
        let mut output_dims = Vec::with_capacity(x_shape.rank());
        
        // Batch size remains the same
        output_dims.push(x_shape.dims[0].clone());
        
        // Output channels come from the weight tensor
        if let Some(out_channels) = w_shape.dims[0].static_value() {
            output_dims.push(ShapeDim::Static(out_channels));
        } else {
            output_dims.push(ShapeDim::Dynamic);
        }
        
        // Compute each spatial dimension
        for i in 0..(x_shape.rank() - 2) {
            if let (Some(input_size), Some(kernel_size)) = (
                x_shape.dims[i + 2].static_value(),
                kernel_shape.get(i).copied()
            ) {
                let dilated_kernel_size = 1 + (kernel_size - 1) * dilations[i];
                let pad_head = pads[i];
                let pad_tail = pads[i + (x_shape.rank() - 2)];
                let stride = strides[i];
                
                let output_size = (input_size + pad_head + pad_tail - dilated_kernel_size) / stride + 1;
                output_dims.push(ShapeDim::Static(output_size));
            } else {
                output_dims.push(ShapeDim::Dynamic);
            }
        }
        
        let output_shape = TensorShape { dims: output_dims };
        Ok(vec![Some(output_shape)])
    }
    
    /// Infer output shapes for MaxPool, AveragePool, LpPool operators
    fn infer_pool_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let x_shape = input_shapes[0].as_ref().unwrap();
        
        // Validate input rank
        if x_shape.rank() < 3 {
            return Err(Error::InvalidModel(format!(
                "Pool input X must have at least 3 dimensions, got {}", x_shape.rank()
            )));
        }
        
        // Get attributes
        let auto_pad = node.attributes
            .get("auto_pad")
            .and_then(|attr| if let Attribute::String(s) = attr { Some(s.as_str()) } else { None })
            .unwrap_or("NOTSET");
            
        let mut pads = node.attributes
            .get("pads")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![0; (x_shape.rank() - 2) * 2]);
            
        let strides = node.attributes
            .get("strides")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![1; x_shape.rank() - 2]);
            
        let dilations = node.attributes
            .get("dilations")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None })
            .unwrap_or_else(|| vec![1; x_shape.rank() - 2]);
            
        let kernel_shape = node.attributes
            .get("kernel_shape")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None });
            
        if kernel_shape.is_none() {
            return Err(Error::InvalidModel("kernel_shape attribute is required for Pool operators".to_string()));
        }
        
        let kernel_shape = kernel_shape.unwrap();
        
        // Make sure pads has the right length
        if pads.len() < (x_shape.rank() - 2) * 2 {
            pads.resize((x_shape.rank() - 2) * 2, 0);
        }
        
        // Handle auto_pad
        if auto_pad != "NOTSET" {
            // Reset pads for auto padding
            pads = vec![0; (x_shape.rank() - 2) * 2];
            
            // For each spatial dimension, compute padding
            for i in 0..(x_shape.rank() - 2) {
                if let (Some(input_size), Some(kernel_size)) = (
                    x_shape.dims[i + 2].static_value(),
                    kernel_shape.get(i).copied()
                ) {
                    let dilated_kernel_size = 1 + (kernel_size - 1) * dilations[i];
                    let stride = strides[i];
                    
                    let output_size = (input_size + stride - 1) / stride;
                    let pad_needed = std::cmp::max(
                        0, 
                        (output_size - 1) * stride + dilated_kernel_size - input_size
                    );
                    
                    let pad_head = pad_needed / 2;
                    let pad_tail = pad_needed - pad_head;
                    
                    if auto_pad == "SAME_UPPER" {
                        pads[i] = pad_head;
                        pads[i + (x_shape.rank() - 2)] = pad_tail;
                    } else if auto_pad == "SAME_LOWER" {
                        pads[i] = pad_tail;
                        pads[i + (x_shape.rank() - 2)] = pad_head;
                    }
                    // For "VALID", padding is 0 (already set)
                }
            }
        }
        
        // Compute output shape
        let mut output_dims = Vec::with_capacity(x_shape.rank());
        
        // Batch size and channels remain the same
        output_dims.push(x_shape.dims[0].clone());
        output_dims.push(x_shape.dims[1].clone());
        
        // Compute each spatial dimension
        for i in 0..(x_shape.rank() - 2) {
            if let (Some(input_size), Some(kernel_size)) = (
                x_shape.dims[i + 2].static_value(),
                kernel_shape.get(i).copied()
            ) {
                let dilated_kernel_size = 1 + (kernel_size - 1) * dilations[i];
                let pad_head = pads[i];
                let pad_tail = pads[i + (x_shape.rank() - 2)];
                let stride = strides[i];
                
                let output_size = (input_size + pad_head + pad_tail - dilated_kernel_size) / stride + 1;
                output_dims.push(ShapeDim::Static(output_size));
            } else {
                output_dims.push(ShapeDim::Dynamic);
            }
        }
        
        // Check if we need to output indices tensor (for MaxPool)
        let ceil_mode = node.attributes
            .get("ceil_mode")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i != 0) } else { None })
            .unwrap_or(false);
            
        // If ceil_mode is true, we need to ceil instead of floor the output size
        if ceil_mode {
            for i in 2..output_dims.len() {
                if let (Some(input_size), Some(kernel_size)) = (
                    x_shape.dims[i].static_value(),
                    kernel_shape.get(i - 2).copied()
                ) {
                    let dilated_kernel_size = 1 + (kernel_size - 1) * dilations[i - 2];
                    let pad_head = pads[i - 2];
                    let pad_tail = pads[i - 2 + (x_shape.rank() - 2)];
                    let stride = strides[i - 2];
                    
                    let numerator = input_size + pad_head + pad_tail - dilated_kernel_size;
                    let output_size = (numerator + stride - 1) / stride + 1;
                    output_dims[i] = ShapeDim::Static(output_size);
                }
            }
        }
        
        let output_shape = TensorShape { dims: output_dims };
        
        // Check if we need to output indices tensor (for MaxPool)
        let indices_output = node.op_type == "MaxPool" && node.outputs.len() > 1;
        if indices_output {
            // Indices tensor has the same shape as the output tensor
            Ok(vec![Some(output_shape.clone()), Some(output_shape)])
        } else {
            Ok(vec![Some(output_shape)])
        }
    }
    
    /// Infer output shapes for GlobalMaxPool, GlobalAveragePool, GlobalLpPool operators
    fn infer_global_pool_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let x_shape = input_shapes[0].as_ref().unwrap();
        
        // Validate input rank
        if x_shape.rank() < 3 {
            return Err(Error::InvalidModel(format!(
                "GlobalPool input X must have at least 3 dimensions, got {}", x_shape.rank()
            )));
        }
        
        // Compute output shape - spatial dimensions are reduced to 1
        let mut output_dims = Vec::with_capacity(x_shape.rank());
        
        // Batch size and channels remain the same
        output_dims.push(x_shape.dims[0].clone());
        output_dims.push(x_shape.dims[1].clone());
        
        // All spatial dimensions become 1
        for _ in 2..x_shape.rank() {
            output_dims.push(ShapeDim::Static(1));
        }
        
        let output_shape = TensorShape { dims: output_dims };
        
        // Check if we need to output indices tensor (for GlobalMaxPool)
        let indices_output = node.op_type == "GlobalMaxPool" && node.outputs.len() > 1;
        if indices_output {
            // Indices tensor has the same shape as the output tensor
            Ok(vec![Some(output_shape.clone()), Some(output_shape)])
        } else {
            Ok(vec![Some(output_shape)])
        }
    }
    
    /// Infer output shapes for MatMul operator
    fn infer_matmul_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.len() < 2 || input_shapes[0].is_none() || input_shapes[1].is_none() {
            return Ok(vec![None]);
        }
        
        let a_shape = input_shapes[0].as_ref().unwrap();
        let b_shape = input_shapes[1].as_ref().unwrap();
        
        // Handle special cases for vector-vector, matrix-vector, and vector-matrix
        if a_shape.rank() == 1 && b_shape.rank() == 1 {
            // Vector-vector dot product: output is a scalar
            return Ok(vec![Some(TensorShape::new(vec![]))]);
        }
        
        if a_shape.rank() == 1 {
            // Vector-matrix: treat vector as 1xK
            let mut result_dims = Vec::new();
            
            // If B is 2D, result is a vector with B's columns
            if b_shape.rank() == 2 {
                if let Some(n) = b_shape.dims[1].static_value() {
                    result_dims.push(ShapeDim::Static(n));
                } else {
                    result_dims.push(ShapeDim::Dynamic);
                }
            } else {
                // B has batch dimensions
                // Take all but the last two dims from B
                result_dims.extend_from_slice(&b_shape.dims[0..b_shape.rank() - 2]);
                
                // Add the last dimension from B
                if let Some(n) = b_shape.dims[b_shape.rank() - 1].static_value() {
                    result_dims.push(ShapeDim::Static(n));
                } else {
                    result_dims.push(ShapeDim::Dynamic);
                }
            }
            
            return Ok(vec![Some(TensorShape { dims: result_dims })]);
        }
        
        if b_shape.rank() == 1 {
            // Matrix-vector: treat vector as Kx1
            let mut result_dims = Vec::new();
            
            // If A is 2D, result is a vector with A's rows
            if a_shape.rank() == 2 {
                if let Some(m) = a_shape.dims[0].static_value() {
                    result_dims.push(ShapeDim::Static(m));
                } else {
                    result_dims.push(ShapeDim::Dynamic);
                }
            } else {
                // A has batch dimensions
                // Take all but the last two dims from A
                result_dims.extend_from_slice(&a_shape.dims[0..a_shape.rank() - 2]);
                
                // Add the second-to-last dimension from A
                if let Some(m) = a_shape.dims[a_shape.rank() - 2].static_value() {
                    result_dims.push(ShapeDim::Static(m));
                } else {
                    result_dims.push(ShapeDim::Dynamic);
                }
            }
            
            return Ok(vec![Some(TensorShape { dims: result_dims })]);
        }
        
        // Handle general case with broadcasting
        let mut result_dims = Vec::new();
        
        // Get the matrix multiply dimensions M, N, K
        let a_rows = if let Some(m) = a_shape.dims[a_shape.rank() - 2].static_value() {
            m
        } else {
            // If we can't determine the rows, result shape is dynamic
            return Ok(vec![Some(TensorShape { 
                dims: vec![ShapeDim::Dynamic; a_shape.rank().max(b_shape.rank())]
            })]);
        };
        
        let b_cols = if let Some(n) = b_shape.dims[b_shape.rank() - 1].static_value() {
            n
        } else {
            // If we can't determine the columns, result shape is dynamic
            return Ok(vec![Some(TensorShape { 
                dims: vec![ShapeDim::Dynamic; a_shape.rank().max(b_shape.rank())]
            })]);
        };
        
        // Broadcast batch dimensions
        let a_batch_dims = &a_shape.dims[0..a_shape.rank() - 2];
        let b_batch_dims = &b_shape.dims[0..b_shape.rank() - 2];
        
        // Compute the broadcasted batch dimensions
        let broadcast_batch_dims = self.broadcast_dimensions(a_batch_dims, b_batch_dims)?;
        
        // Combine batch dimensions with matrix multiply dimensions
        result_dims.extend_from_slice(&broadcast_batch_dims);
        result_dims.push(ShapeDim::Static(a_rows));
        result_dims.push(ShapeDim::Static(b_cols));
        
        Ok(vec![Some(TensorShape { dims: result_dims })])
    }
    
    /// Infer output shapes for Gemm operator
    fn infer_gemm_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.len() < 2 || input_shapes[0].is_none() || input_shapes[1].is_none() {
            return Ok(vec![None]);
        }
        
        let a_shape = input_shapes[0].as_ref().unwrap();
        let b_shape = input_shapes[1].as_ref().unwrap();
        
        // Validate input ranks
        if a_shape.rank() != 2 || b_shape.rank() != 2 {
            return Err(Error::InvalidModel(format!(
                "Gemm inputs A and B must be 2D matrices, got shapes of rank {} and {}", 
                a_shape.rank(), b_shape.rank()
            )));
        }
        
        // Get attributes
        let trans_a = node.attributes
            .get("transA")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i != 0) } else { None })
            .unwrap_or(false);
            
        let trans_b = node.attributes
            .get("transB")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i != 0) } else { None })
            .unwrap_or(false);
        
        // Compute M and N dimensions
        let m = if let Some(dim) = a_shape.dims[if trans_a { 1 } else { 0 }].static_value() {
            ShapeDim::Static(dim)
        } else {
            ShapeDim::Dynamic
        };
        
        let n = if let Some(dim) = b_shape.dims[if trans_b { 0 } else { 1 }].static_value() {
            ShapeDim::Static(dim)
        } else {
            ShapeDim::Dynamic
        };
        
        // Output shape is [M, N]
        Ok(vec![Some(TensorShape { dims: vec![m, n] })])
    }
    
    /// Infer output shapes for binary elementwise operators (Add, Sub, Mul, Div, etc.)
    fn infer_binary_elementwise_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.len() < 2 || input_shapes[0].is_none() || input_shapes[1].is_none() {
            return Ok(vec![None]);
        }
        
        let a_shape = input_shapes[0].as_ref().unwrap();
        let b_shape = input_shapes[1].as_ref().unwrap();
        
        // Broadcast the shapes
        let broadcast_dims = self.broadcast_dimensions(&a_shape.dims, &b_shape.dims)?;
        
        Ok(vec![Some(TensorShape { dims: broadcast_dims })])
    }
    
    /// Infer output shapes for unary elementwise operators (Relu, Sigmoid, etc.)
    fn infer_unary_elementwise_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        // Unary operators preserve shape
        Ok(vec![input_shapes[0].clone()])
    }
    
    /// Infer output shapes for Concat operator
    fn infer_concat_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        // Need at least one input
        if input_shapes.is_empty() || input_shapes.iter().all(|s| s.is_none()) {
            return Ok(vec![None]);
        }
        
        // Get axis attribute
        let axis = node.attributes
            .get("axis")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i as i32) } else { None })
            .unwrap_or(0);
            
        // Find the first non-None shape to use as a base
        let first_shape = input_shapes.iter()
            .filter_map(|s| s.as_ref())
            .next()
            .ok_or_else(|| Error::InvalidModel("All Concat inputs have unknown shapes".to_string()))?;
            
        let rank = first_shape.rank();
        
        // Convert negative axis to positive
        let axis = if axis < 0 { (axis + rank as i32) as usize } else { axis as usize };
        
        if axis >= rank {
            return Err(Error::InvalidModel(format!(
                "Concat axis {} is out of bounds for tensor of rank {}", axis, rank
            )));
        }
        
        // Validate all inputs have compatible shapes
        for (i, shape) in input_shapes.iter().enumerate() {
            if let Some(shape) = shape {
                if shape.rank() != rank {
                    return Err(Error::InvalidModel(format!(
                        "Concat input {} has rank {}, expected {}", i, shape.rank(), rank
                    )));
                }
                
                // Check that all non-concat dimensions match
                for dim_idx in 0..rank {
                    if dim_idx != axis {
                        match (&first_shape.dims[dim_idx], &shape.dims[dim_idx]) {
                            (ShapeDim::Static(d1), ShapeDim::Static(d2)) if d1 != d2 => {
                                return Err(Error::InvalidModel(format!(
                                    "Concat input {} dimension {} has size {}, expected {}",
                                    i, dim_idx, d2, d1
                                )));
                            },
                            _ => {} // Dynamic dimensions or matching sizes are OK
                        }
                    }
                }
            }
        }
        
        // Compute the size of the concat dimension
        let mut concat_size: Option<usize> = Some(0);
        for shape in input_shapes.iter().filter_map(|s| s.as_ref()) {
            if let ShapeDim::Static(size) = &shape.dims[axis] {
                concat_size = concat_size.map(|s| s + size);
            } else {
                concat_size = None;
                break;
            }
        }
        
        // Create output shape
        let mut output_dims = first_shape.dims.clone();
        output_dims[axis] = if let Some(size) = concat_size {
            ShapeDim::Static(size)
        } else {
            ShapeDim::Dynamic
        };
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Reshape operator
    fn infer_reshape_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.len() < 2 || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let data_shape = input_shapes[0].as_ref().unwrap();
        
        // For Reshape, the second input is the shape tensor
        // If it's a constant/initializer, we can get its value
        let new_shape: Option<Vec<i64>> = if let Some(shape_tensor_name) = node.inputs.get(1) {
            if let Ok(data_type_cache) = self.data_type_cache.read() {
                if let Ok(shape_cache) = self.shape_cache.read() {
                    if let (Some(shape_tensor_shape), Some(shape_tensor_type)) = (
                        shape_cache.get(shape_tensor_name),
                        data_type_cache.get(shape_tensor_name)
                    ) {
                        // If shape tensor is an initializer, we might be able to get its value
                        if let Some(graph) = &node.graph {
                            if let Some(initializer) = graph.initializers.iter().find(|init| &init.name == shape_tensor_name) {
                                // Extract shape values from initializer data
                                match shape_tensor_type {
                                    DataType::Int32 => {
                                        let data = initializer.data.as_slice();
                                        let int_data = unsafe {
                                            std::slice::from_raw_parts(
                                                data.as_ptr() as *const i32,
                                                data.len() / std::mem::size_of::<i32>()
                                            )
                                        };
                                        Some(int_data.iter().map(|&x| x as i64).collect())
                                    },
                                    DataType::Int64 => {
                                        let data = initializer.data.as_slice();
                                        let int_data = unsafe {
                                            std::slice::from_raw_parts(
                                                data.as_ptr() as *const i64,
                                                data.len() / std::mem::size_of::<i64>()
                                            )
                                        };
                                        Some(int_data.to_vec())
                                    },
                                    _ => None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // If we couldn't determine the new shape, return unknown shape
        if new_shape.is_none() {
            return Ok(vec![None]);
        }
        
        let new_shape = new_shape.unwrap();
        
        // Calculate total elements in data tensor
        let data_size = if data_shape.is_fully_static() {
            Some(data_shape.num_elements(0))
        } else {
            None
        };
        
        // Create output shape dimensions
        let mut output_dims = Vec::with_capacity(new_shape.len());
        let mut inferred_idx = None;
        let mut output_size = 1;
        
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == 0 {
                // Copy from input shape
                if i < data_shape.rank() {
                    output_dims.push(data_shape.dims[i].clone());
                    if let Some(size) = data_shape.dims[i].static_value() {
                        output_size *= size;
                    }
                } else {
                    output_dims.push(ShapeDim::Dynamic);
                }
            } else if dim == -1 {
                // To be inferred
                if inferred_idx.is_some() {
                    return Err(Error::InvalidModel(
                        "Only one dimension can be inferred in Reshape".to_string()
                    ));
                }
                inferred_idx = Some(i);
                output_dims.push(ShapeDim::Dynamic);
            } else if dim > 0 {
                // Static dimension
                output_dims.push(ShapeDim::Static(dim as usize));
                output_size *= dim as usize;
            } else {
                return Err(Error::InvalidModel(format!(
                    "Invalid dimension value {} in Reshape shape", dim
                )));
            }
        }
        
        // Infer the value of -1 dimension if needed
        if let (Some(idx), Some(total_size)) = (inferred_idx, data_size) {
            if output_size == 0 {
                // Handle edge case where one of the dimensions is 0
                output_dims[idx] = ShapeDim::Static(0);
            } else if total_size % output_size == 0 {
                let inferred_size = total_size / output_size;
                output_dims[idx] = ShapeDim::Static(inferred_size);
            } else {
                return Err(Error::InvalidModel(format!(
                    "Cannot reshape tensor with {} elements into shape where known dimensions have {} elements",
                    total_size, output_size
                )));
            }
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Transpose operator
    fn infer_transpose_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let input_shape = input_shapes[0].as_ref().unwrap();
        let rank = input_shape.rank();
        
        // Get perm attribute (permutation of axes)
        let perm = node.attributes
            .get("perm")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
            } else { None });
            
        // If perm is not specified, reverse the dimensions
        let perm = match perm {
            Some(p) => {
                if p.len() != rank {
                    return Err(Error::InvalidModel(format!(
                        "Transpose perm attribute has length {}, expected {}", p.len(), rank
                    )));
                }
                p
            },
            None => (0..rank).rev().collect(),
        };
        
        // Create output shape by permuting the input dimensions
        let mut output_dims = vec![ShapeDim::Dynamic; rank];
        for (i, &axis) in perm.iter().enumerate() {
            if axis >= rank {
                return Err(Error::InvalidModel(format!(
                    "Transpose perm attribute contains invalid axis {}", axis
                )));
            }
            output_dims[i] = input_shape.dims[axis].clone();
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Flatten operator
    fn infer_flatten_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let input_shape = input_shapes[0].as_ref().unwrap();
        let rank = input_shape.rank();
        
        // Get axis attribute
        let axis = node.attributes
            .get("axis")
            .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i as i32) } else { None })
            .unwrap_or(1);
            
        // Convert negative axis to positive
        let axis = if axis < 0 { (axis + rank as i32) as usize } else { axis as usize };
        
        if axis > rank {
            return Err(Error::InvalidModel(format!(
                "Flatten axis {} is out of bounds for tensor of rank {}", axis, rank
            )));
        }
        
        // Compute the first dimension (product of dimensions before axis)
        let mut first_dim = 1;
        let mut first_dim_static = true;
        
        for i in 0..axis {
            if let ShapeDim::Static(size) = input_shape.dims[i] {
                first_dim *= size;
            } else {
                first_dim_static = false;
                break;
            }
        }
        
        // Compute the second dimension (product of dimensions from axis onwards)
        let mut second_dim = 1;
        let mut second_dim_static = true;
        
        for i in axis..rank {
            if let ShapeDim::Static(size) = input_shape.dims[i] {
                second_dim *= size;
            } else {
                second_dim_static = false;
                break;
            }
        }
        
        // Create output shape
        let mut output_dims = Vec::with_capacity(2);
        
        if first_dim_static {
            output_dims.push(ShapeDim::Static(first_dim));
        } else {
            output_dims.push(ShapeDim::Dynamic);
        }
        
        if second_dim_static {
            output_dims.push(ShapeDim::Static(second_dim));
        } else {
            output_dims.push(ShapeDim::Dynamic);
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Slice operator
    fn infer_slice_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let data_shape = input_shapes[0].as_ref().unwrap();
        let rank = data_shape.rank();
        
        // For Slice, we need starts, ends, and optionally axes and steps
        // These are provided as input tensors in newer ONNX versions, or as attributes in older versions
        
        // First, try to get from attributes (older ONNX versions)
        let starts_attr = node.attributes
            .get("starts")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.clone()) 
            } else { None });
            
        let ends_attr = node.attributes
            .get("ends")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.clone()) 
            } else { None });
            
        let axes_attr = node.attributes
            .get("axes")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.clone()) 
            } else { None });
            
        // If attributes are not present, we need to get from inputs (newer ONNX versions)
        // This would require accessing the actual tensor values, which may not be available
        // during shape inference time, so we'll just return a shape with the same rank
        // but all dimensions marked as dynamic
        
        if starts_attr.is_none() || ends_attr.is_none() {
            // Return a shape with the same rank but all dimensions dynamic
            return Ok(vec![Some(TensorShape {
                dims: vec![ShapeDim::Dynamic; rank]
            })]);
        }
        
        let starts = starts_attr.unwrap();
        let ends = ends_attr.unwrap();
        
        if starts.len() != ends.len() {
            return Err(Error::InvalidModel(format!(
                "Slice starts and ends must have the same length, got {} and {}", 
                starts.len(), ends.len()
            )));
        }
        
        // Determine which axes to slice
        let axes = if let Some(axes) = axes_attr {
            if axes.len() != starts.len() {
                return Err(Error::InvalidModel(format!(
                    "Slice axes must have the same length as starts, got {} and {}", 
                    axes.len(), starts.len()
                )));
            }
            axes
        } else {
            // Default is to slice the first N dimensions
            (0..starts.len() as i64).collect()
        };
        
        // Create output shape by computing slice sizes
        let mut output_dims = data_shape.dims.clone();
        
        for i in 0..axes.len() {
            let axis = if axes[i] < 0 { axes[i] + rank as i64 } else { axes[i] } as usize;
            
            if axis >= rank {
                return Err(Error::InvalidModel(format!(
                    "Slice axis {} is out of bounds for tensor of rank {}", axis, rank
                )));
            }
            
            // Compute slice size for this dimension
            if let ShapeDim::Static(dim_size) = data_shape.dims[axis] {
                let start = starts[i];
                let end = ends[i];
                
                // Handle negative indices and clamping
                let start = if start < 0 { start + dim_size as i64 } else { start }
                    .max(0)
                    .min(dim_size as i64);
                    
                let end = if end < 0 { end + dim_size as i64 } else { end }
                    .max(0)
                    .min(dim_size as i64);
                    
                // Calculate output size for this dimension
                if start < end {
                    output_dims[axis] = ShapeDim::Static((end - start) as usize);
                } else {
                    output_dims[axis] = ShapeDim::Static(0);
                }
            } else {
                // If input dimension is dynamic, output is also dynamic
                output_dims[axis] = ShapeDim::Dynamic;
            }
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Squeeze operator
    fn infer_squeeze_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let input_shape = input_shapes[0].as_ref().unwrap();
        let rank = input_shape.rank();
        
        // Get axes to squeeze
        let axes_attr = node.attributes
            .get("axes")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as i32).collect::<Vec<_>>()) 
            } else { None });
            
        // If axes is not specified and there's a second input, it's the axes tensor
        // But we may not be able to evaluate it at shape inference time
        if axes_attr.is_none() && node.inputs.len() <= 1 {
            // If no axes specified, squeeze all dimensions of size 1
            let mut output_dims = Vec::new();
            
            for dim in &input_shape.dims {
                match dim {
                    ShapeDim::Static(1) => {}, // Skip dimensions of size 1
                    _ => output_dims.push(dim.clone()),
                }
            }
            
            return Ok(vec![Some(TensorShape { dims: output_dims })]);
        }
        
        if axes_attr.is_none() {
            // Can't determine axes at compile time, return unknown shape
            return Ok(vec![None]);
        }
        
        let axes = axes_attr.unwrap();
        
        // Convert negative axes to positive and sort
        let mut positive_axes: Vec<usize> = axes.iter()
            .map(|&axis| if axis < 0 { (axis + rank as i32) as usize } else { axis as usize })
            .collect();
            
        positive_axes.sort_unstable();
        
        // Create output shape by removing squeezed dimensions
        let mut output_dims = Vec::new();
        
        for (i, dim) in input_shape.dims.iter().enumerate() {
            if positive_axes.contains(&i) {
                // This dimension should be squeezed
                match dim {
                    ShapeDim::Static(1) => {}, // Skip this dimension
                    ShapeDim::Static(size) => {
                        return Err(Error::InvalidModel(format!(
                            "Cannot squeeze dimension {} with size {}", i, size
                        )));
                    },
                    ShapeDim::Dynamic | ShapeDim::Symbolic(_) => {
                        // Can't determine at compile time, assume it's valid
                        // But we need to keep this dimension in case it's not 1
                        output_dims.push(dim.clone());
                    },
                }
            } else {
                // Keep this dimension
                output_dims.push(dim.clone());
            }
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Unsqueeze operator
    fn infer_unsqueeze_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let input_shape = input_shapes[0].as_ref().unwrap();
        let rank = input_shape.rank();
        
        // Get axes to unsqueeze
        let axes_attr = node.attributes
            .get("axes")
            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                Some(ints.iter().map(|&i| i as i32).collect::<Vec<_>>()) 
            } else { None });
            
        // If axes is not specified and there's a second input, it's the axes tensor
        // But we may not be able to evaluate it at shape inference time
        if axes_attr.is_none() {
            // Can't determine axes at compile time, return unknown shape
            return Ok(vec![None]);
        }
        
        let axes = axes_attr.unwrap();
        
        // Convert negative axes to positive and sort
        let mut positive_axes: Vec<usize> = axes.iter()
            .map(|&axis| if axis < 0 { (axis + rank as i32 + axes.len() as i32) as usize } else { axis as usize })
            .collect();
            
        positive_axes.sort_unstable();
        
        // Create output shape by inserting dimensions of size 1
        let mut output_dims = Vec::with_capacity(rank + axes.len());
        
        let mut input_idx = 0;
        for output_idx in 0..(rank + axes.len()) {
            if positive_axes.contains(&output_idx) {
                // Insert a dimension of size 1
                output_dims.push(ShapeDim::Static(1));
            } else {
                // Copy from input shape
                if input_idx < rank {
                    output_dims.push(input_shape.dims[input_idx].clone());
                    input_idx += 1;
                }
            }
        }
        
        Ok(vec![Some(TensorShape { dims: output_dims })])
    }
    
    /// Infer output shapes for Resize operator
    fn infer_resize_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let x_shape = input_shapes[0].as_ref().unwrap();
        
        // Resize requires multiple inputs:
        // 1. X: input tensor
        // 2. roi (optional): regions of interest (used for coordinate transformation)
        // 3. scales (optional): scaling factors
        // 4. sizes (optional): target sizes
        
        // If we have the sizes input (index 3), we can use it directly
        if input_shapes.len() > 3 && input_shapes[3].is_some() {
            // Sizes input should be a 1D tensor with the same number of elements as input rank
            // But we may not be able to evaluate it at shape inference time
            // Just return unknown shape
            return Ok(vec![None]);
        }
        
        // If we have the scales input (index 2), we can try to use it
        if input_shapes.len() > 2 && input_shapes[2].is_some() {
            // Scales input should be a 1D tensor with the same number of elements as input rank
            // But we may not be able to evaluate it at shape inference time
            // Just return unknown shape
            return Ok(vec![None]);
        }
        
        // If we can't determine the output sizes at compile time, return a shape
        // with the same rank but all dimensions dynamic
        Ok(vec![Some(TensorShape {
            dims: vec![ShapeDim::Dynamic; x_shape.rank()]
        })])
    }
    
    /// Default shape inference for operators without specific handlers
    fn infer_default_shape(&self, node: &Node, input_shapes: &[Option<TensorShape>]) -> Result<Vec<Option<TensorShape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            // If we don't have input shapes, we can't infer output shapes
            return Ok(vec![None; node.outputs.len()]);
        }
        
        // By default, assume the first output has the same shape as the first input
        // and all other outputs have unknown shapes
        let mut output_shapes = vec![None; node.outputs.len()];
        if !output_shapes.is_empty() {
            output_shapes[0] = input_shapes[0].clone();
        }
        
        Ok(output_shapes)
    }
    
    /// Broadcast two sets of dimensions according to broadcasting rules
    fn broadcast_dimensions(&self, a_dims: &[ShapeDim], b_dims: &[ShapeDim]) -> Result<Vec<ShapeDim>> {
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();
        let max_rank = a_rank.max(b_rank);
        
        let mut result = Vec::with_capacity(max_rank);
        
        // Handle broadcasting with right-alignment
        for i in 0..max_rank {
            let a_idx = if i >= max_rank - a_rank { i - (max_rank - a_rank) } else { 0 };
            let b_idx = if i >= max_rank - b_rank { i - (max_rank - b_rank) } else { 0 };
            
            let a_dim = if i >= max_rank - a_rank { &a_dims[a_idx] } else { &ShapeDim::Static(1) };
            let b_dim = if i >= max_rank - b_rank { &b_dims[b_idx] } else { &ShapeDim::Static(1) };
            
            match (a_dim, b_dim) {
                (ShapeDim::Static(1), dim) => result.push(dim.clone()),
                (dim, ShapeDim::Static(1)) => result.push(dim.clone()),
                (ShapeDim::Static(a), ShapeDim::Static(b)) => {
                    if a == b {
                        result.push(ShapeDim::Static(*a));
                    } else {
                        return Err(Error::InvalidModel(format!(
                            "Cannot broadcast dimensions {} and {}", a, b
                        )));
                    }
                },
                (ShapeDim::Dynamic, _) | (_, ShapeDim::Dynamic) => result.push(ShapeDim::Dynamic),
                (ShapeDim::Symbolic(_), _) | (_, ShapeDim::Symbolic(_)) => result.push(ShapeDim::Dynamic),
            }
        }
        
        Ok(result)
    }
    
    /// Perform topological sort on the graph
    fn topological_sort(&self, graph: &ExecutionGraph) -> Result<Vec<NodeId>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        
        // Start from input nodes
        for &node_id in &graph.input_nodes {
            self.topological_sort_visit(
                node_id, 
                graph, 
                &mut visited, 
                &mut visiting, 
                &mut result
            )?;
        }
        
        Ok(result)
    }
    
    /// Helper for topological sort
    fn topological_sort_visit(
        &self,
        node_id: NodeId,
        graph: &ExecutionGraph,
        visited: &mut HashSet<NodeId>,
        visiting: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        
        if visiting.contains(&node_id) {
            return Err(Error::InvalidGraph(format!(
                "Cycle detected in graph at node {}", node_id
            )));
        }
        
        visiting.insert(node_id);
        
        // Visit dependencies
        if let Some(deps) = graph.dependencies.get(&node_id) {
            for &dep_id in deps {
                self.topological_sort_visit(dep_id, graph, visited, visiting, result)?;
            }
        }
        
        visiting.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);
        
        Ok(())
    }
}

/// Data type category for determining compatible operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCategory {
    /// Floating point types (FP32, FP16, BF16, etc.)
    Float,
    /// Integer types (INT8, INT32, INT64, etc.)
    Integer,
    /// Boolean type
    Boolean,
    /// String type
    String,
    /// Complex number types
    Complex,
    /// Unknown or undefined type
    Undefined,
}

/// Type promotion rules for binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypePromotion {
    /// Use the highest precision type
    HighestPrecision,
    /// Use the first input's type
    FirstInput,
    /// Use the second input's type
    SecondInput,
    /// Convert both inputs to a specific type
    SpecificType(DataType),
    /// No promotion needed (same type)
    None,
}

/// Type constraint for an operator
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    /// Allowed data types for this constraint
    pub allowed_types: Vec<DataType>,
    /// Description of the constraint
    pub description: String,
}

/// Type inference result for an operator's outputs
#[derive(Debug, Clone)]
pub enum TypeInferenceResult {
    /// All outputs have the same type as the first input
    SameAsInput,
    /// All outputs have the specified type
    SpecificType(DataType),
    /// Each output has a specific type
    MultipleTypes(Vec<DataType>),
    /// Output types are determined by a function of the input types
    Function(fn(&[Option<DataType>], &Node) -> Result<Vec<DataType>>),
}

/// Convert from model data type to tensor data type
fn convert_model_data_type(model_type: ModelDataType) -> DataType {
    match model_type {
        ModelDataType::Float => DataType::Float32,
        ModelDataType::Double => DataType::Float64,
        ModelDataType::Int8 => DataType::Int8,
        ModelDataType::Int16 => DataType::Int16,
        ModelDataType::Int32 => DataType::Int32,
        ModelDataType::Int64 => DataType::Int64,
        ModelDataType::Uint8 => DataType::UInt8,
        ModelDataType::Uint16 => DataType::UInt16,
        ModelDataType::Uint32 => DataType::UInt32,
        ModelDataType::Uint64 => DataType::UInt64,
        ModelDataType::Bool => DataType::Bool,
        ModelDataType::Float16 => DataType::Float16,
        ModelDataType::BFloat16 => DataType::BFloat16,
        ModelDataType::String => DataType::String,
        ModelDataType::Complex64 => DataType::Complex64,
        ModelDataType::Complex128 => DataType::Complex128,
        ModelDataType::Float8E4M3FN => DataType::Float8E4M3FN,
        ModelDataType::Float8E4M3FNUZ => DataType::Float8E4M3FNUZ,
        ModelDataType::Float8E5M2 => DataType::Float8E5M2,
        ModelDataType::Float8E5M2FNUZ => DataType::Float8E5M2FNUZ,
        _ => DataType::Float32,  // Default to Float32 for unsupported types
    }
}

/// Get the category of a data type
fn get_type_category(data_type: DataType) -> TypeCategory {
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 | 
        DataType::BFloat16 | DataType::Float8E4M3FN | DataType::Float8E4M3FNUZ |
        DataType::Float8E5M2 | DataType::Float8E5M2FNUZ => TypeCategory::Float,
        
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => TypeCategory::Integer,
        
        DataType::Bool => TypeCategory::Boolean,
        DataType::String => TypeCategory::String,
        DataType::Complex64 | DataType::Complex128 => TypeCategory::Complex,
        _ => TypeCategory::Undefined,
    }
}

/// Get the bit width of a data type
fn get_type_bit_width(data_type: DataType) -> usize {
    match data_type {
        DataType::Float8E4M3FN | DataType::Float8E4M3FNUZ |
        DataType::Float8E5M2 | DataType::Float8E5M2FNUZ |
        DataType::Int8 | DataType::UInt8 => 8,
        
        DataType::Float16 | DataType::BFloat16 |
        DataType::Int16 | DataType::UInt16 => 16,
        
        DataType::Float32 | DataType::Int32 | DataType::UInt32 => 32,
        DataType::Float64 | DataType::Int64 | DataType::UInt64 => 64,
        DataType::Complex64 => 64,
        DataType::Complex128 => 128,
        
        // For other types, use a reasonable default
        DataType::Bool => 1,
        DataType::String => 8,  // Variable size, but using 8 bits for comparison
        _ => 32,  // Default to 32 bits for undefined types
    }
}

/// Determine if a data type is a floating point type
fn is_floating_point(data_type: DataType) -> bool {
    matches!(get_type_category(data_type), TypeCategory::Float)
}

/// Determine if a data type is an integer type
fn is_integer(data_type: DataType) -> bool {
    matches!(get_type_category(data_type), TypeCategory::Integer)
}

/// Determine if a data type is a signed integer type
fn is_signed_integer(data_type: DataType) -> bool {
    matches!(data_type, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64)
}

/// Determine if a data type is an unsigned integer type
fn is_unsigned_integer(data_type: DataType) -> bool {
    matches!(data_type, DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64)
}

/// Determine if a data type is a complex number type
fn is_complex(data_type: DataType) -> bool {
    matches!(get_type_category(data_type), TypeCategory::Complex)
}

/// Promote types for a binary operation using standard promotion rules
fn promote_types(type1: DataType, type2: DataType, promotion_rule: TypePromotion) -> DataType {
    match promotion_rule {
        TypePromotion::FirstInput => type1,
        TypePromotion::SecondInput => type2,
        TypePromotion::SpecificType(dtype) => dtype,
        TypePromotion::None => type1,  // Assumes type1 == type2
        TypePromotion::HighestPrecision => {
            // If types are in different categories, use a default promotion
            if get_type_category(type1) != get_type_category(type2) {
                // Mixed float and integer promotes to float
                if is_floating_point(type1) && is_integer(type2) {
                    return type1;
                } else if is_integer(type1) && is_floating_point(type2) {
                    return type2;
                }
                
                // For other mixed types, prefer the first type
                return type1;
            }
            
            // Within the same category, use the highest bit width
            if get_type_bit_width(type1) >= get_type_bit_width(type2) {
                type1
            } else {
                type2
            }
        }
    }
}

/// Determine if two types are compatible for an operation
fn are_types_compatible(type1: DataType, type2: DataType, allow_mixed_categories: bool) -> bool {
    if type1 == type2 {
        return true;
    }
    
    if allow_mixed_categories {
        // Some operations allow mixed types (e.g., float and int)
        return true;
    }
    
    // Otherwise, types should be in the same category
    get_type_category(type1) == get_type_category(type2)
}

/// Get the result type for a cast operation
fn get_cast_result_type(to_type: &str) -> Option<DataType> {
    match to_type {
        "FLOAT" => Some(DataType::Float32),
        "FLOAT16" => Some(DataType::Float16),
        "DOUBLE" => Some(DataType::Float64),
        "INT8" => Some(DataType::Int8),
        "INT16" => Some(DataType::Int16),
        "INT32" => Some(DataType::Int32),
        "INT64" => Some(DataType::Int64),
        "UINT8" => Some(DataType::UInt8),
        "UINT16" => Some(DataType::UInt16),
        "UINT32" => Some(DataType::UInt32),
        "UINT64" => Some(DataType::UInt64),
        "BOOL" => Some(DataType::Bool),
        "STRING" => Some(DataType::String),
        "BFLOAT16" => Some(DataType::BFloat16),
        "COMPLEX64" => Some(DataType::Complex64),
        "COMPLEX128" => Some(DataType::Complex128),
        _ => None,
    }
}

/// Memory planner for optimizing memory usage during execution
pub struct MemoryPlanner {
    /// Current execution graph being analyzed
    current_graph: Option<Arc<ExecutionGraph>>,
    /// Cache of tensor information
    tensor_info_cache: RwLock<HashMap<TensorId, TensorMemoryInfo>>,
    /// Cache of tensor ID mappings
    tensor_id_map_cache: RwLock<HashMap<String, TensorId>>,
    /// Atomic counter for generating unique tensor IDs
    next_tensor_id: AtomicUsize,
    /// Shape inference system
    shape_inference: ShapeInferenceSystem,
}

impl MemoryPlanner {
    /// Create a new memory planner
    pub fn new() -> Self {
        Self {
            current_graph: None,
            tensor_info_cache: RwLock::new(HashMap::new()),
            tensor_id_map_cache: RwLock::new(HashMap::new()),
            next_tensor_id: AtomicUsize::new(1), // Start IDs from 1 to reserve 0 for special cases
            shape_inference: ShapeInferenceSystem::new(),
        }
    }
    
    /// Generate a new unique tensor ID
    fn generate_tensor_id(&self) -> TensorId {
        self.next_tensor_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Plan memory usage for an execution graph
    pub fn plan_memory_usage(
        &mut self,
        graph: &ExecutionGraph,
    ) -> Result<MemoryPlan> {
        // Store the current graph for reference
        self.current_graph = Some(Arc::new(graph.clone()));
        
        // Clear caches for fresh analysis
        if let Ok(mut cache) = self.tensor_info_cache.write() {
            cache.clear();
        } else {
            return Err(Error::InvalidModel("Failed to acquire write lock for tensor_info_cache".to_string()));
        }
        
        if let Ok(mut cache) = self.tensor_id_map_cache.write() {
            cache.clear();
        } else {
            return Err(Error::InvalidModel("Failed to acquire write lock for tensor_id_map_cache".to_string()));
        }
        
        // Reset the tensor ID counter
        self.next_tensor_id.store(1, Ordering::SeqCst);
        
        // Clear the shape and type inference caches
        self.shape_inference.clear_caches();
        
        // First, run type inference to determine tensor data types throughout the graph
        if let Err(e) = self.shape_inference.infer_types(graph) {
            // Log the error but continue - we'll use fallback data types
            eprintln!("Type inference failed: {}. Using fallback data types.", e);
        }
        
        // Then, run shape inference to determine tensor shapes throughout the graph
        // This ensures that shapes are inferred with the correct data types
        if let Err(e) = self.shape_inference.infer_shapes(graph) {
            // Log the error but continue - we'll use fallback shapes
            eprintln!("Shape inference failed: {}. Using fallback shapes.", e);
        }
        
        // Determine execution order
        let execution_order = self.determine_execution_order(graph)?;

        // Compute tensor lifetimes
        let lifetimes = self.compute_tensor_lifetimes(graph, &execution_order)?;

        // Gather tensor information using the inferred shapes and types
        let tensor_info = self.gather_tensor_info(graph)?;
        
        // Cache the tensor info for later use
        if let Ok(mut cache) = self.tensor_info_cache.write() {
            *cache = tensor_info.clone();
        }

        // Find in-place operation opportunities
        let inplace_ops = self.inplace_operations_analysis(graph)?;

        // Apply in-place optimizations
        let (tensor_info, lifetimes) = self.apply_inplace_optimizations(
            tensor_info,
            lifetimes.clone(),
            inplace_ops.clone(),
        )?;

        // Create initial memory plan
        let mut plan = MemoryPlan {
            allocations: HashMap::new(),
            tensor_info,
            lifetimes,
            buffer_sizes: vec![0],
            inplace_ops,
            total_memory_bytes: 0,
            execution_order,
        };

        // Optimize memory layout based on accurate tensor sizes
        self.optimize_memory_layout(&mut plan)?;

        // Verify the memory plan for type safety
        self.verify_memory_plan(&plan)?;

        Ok(plan)
    }
    
    /// Verify that the memory plan is type-safe
    fn verify_memory_plan(&self, plan: &MemoryPlan) -> Result<()> {
        // Check for any overlapping tensors with incompatible types
        let mut overlapping_pairs = Vec::new();
        
        // Collect all tensor allocations
        let allocations: Vec<(&TensorId, &TensorAllocation)> = plan.allocations.iter().collect();
        
        // Check each pair of allocations for potential overlaps
        for i in 0..allocations.len() {
            let (id1, alloc1) = allocations[i];
            let info1 = plan.tensor_info.get(id1).ok_or_else(|| {
                Error::InvalidModel(format!("Missing tensor info for tensor ID {}", id1))
            })?;
            
            for j in i+1..allocations.len() {
                let (id2, alloc2) = allocations[j];
                let info2 = plan.tensor_info.get(id2).ok_or_else(|| {
                    Error::InvalidModel(format!("Missing tensor info for tensor ID {}", id2))
                })?;
                
                // Check if the allocations are in the same buffer
                if alloc1.buffer_index == alloc2.buffer_index {
                    // Check if the memory regions overlap
                    let start1 = alloc1.offset;
                    let end1 = start1 + alloc1.size_bytes;
                    let start2 = alloc2.offset;
                    let end2 = start2 + alloc2.size_bytes;
                    
                    let regions_overlap = !(end1 <= start2 || start1 >= end2);
                    
                    if regions_overlap {
                        // Check if the lifetimes overlap
                        let lifetime1 = plan.lifetimes.get(id1).ok_or_else(|| {
                            Error::InvalidModel(format!("Missing lifetime for tensor ID {}", id1))
                        })?;
                        
                        let lifetime2 = plan.lifetimes.get(id2).ok_or_else(|| {
                            Error::InvalidModel(format!("Missing lifetime for tensor ID {}", id2))
                        })?;
                        
                        let lifetimes_overlap = !(lifetime1.1 < lifetime2.0 || lifetime2.1 < lifetime1.0);
                        
                        if lifetimes_overlap {
                            // Check if the types are compatible
                            if info1.data_type != info2.data_type {
                                overlapping_pairs.push((info1.clone(), info2.clone()));
                            }
                        }
                    }
                }
            }
        }
        
        // If we found any overlapping pairs with incompatible types, report an error
        if !overlapping_pairs.is_empty() {
            let error_msg = overlapping_pairs.iter()
                .map(|(info1, info2)| format!(
                    "Tensors '{}' ({:?}) and '{}' ({:?}) overlap in memory but have incompatible types",
                    info1.name, info1.data_type, info2.name, info2.data_type
                ))
                .collect::<Vec<_>>()
                .join("\n");
            
            return Err(Error::InvalidModel(format!(
                "Memory plan contains type-unsafe tensor overlaps:\n{}", error_msg
            )));
        }
        
        Ok(())
    }

    /// Determine execution order for nodes in the graph
    fn determine_execution_order(&self, graph: &ExecutionGraph) -> Result<Vec<NodeId>> {
        // This is a simplified version - in practice, you would use
        // the topological sort from the execution engine
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        for &node_id in &graph.input_nodes {
            self.topological_sort(node_id, graph, &mut visited, &mut visiting, &mut result)?;
        }

        Ok(result)
    }

    /// Helper for topological sort
    fn topological_sort(
        &self,
        node_id: NodeId,
        graph: &ExecutionGraph,
        visited: &mut HashSet<NodeId>,
        visiting: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        if visiting.contains(&node_id) {
            return Err(Error::InvalidGraph(format!(
                "Cycle detected in graph at node {}", node_id
            )));
        }

        visiting.insert(node_id);

        // Visit dependencies
        if let Some(deps) = graph.dependencies.get(&node_id) {
            for &dep_id in deps {
                self.topological_sort(dep_id, graph, visited, visiting, result)?;
            }
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);

        Ok(())
    }

    /// Compute tensor lifetimes based on execution order
    pub fn compute_tensor_lifetimes(
        &self,
        graph: &ExecutionGraph,
        execution_order: &[NodeId],
    ) -> Result<HashMap<TensorId, (usize, usize)>> {
        let mut lifetimes = HashMap::new();
        let mut tensor_producers = HashMap::new();
        let mut tensor_consumers = HashMap::new();
        
        // Create a tensor ID map for this analysis
        let tensor_id_map = self.create_tensor_id_map(graph)?;

        // Build tensor producer and consumer maps
        for (idx, &node_id) in execution_order.iter().enumerate() {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;

            // Record outputs (produced by this node)
            for output_name in &node.outputs {
                tensor_producers.insert(output_name.clone(), idx);
            }

            // Record inputs (consumed by this node)
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue; // Skip empty inputs (optional)
                }

                tensor_consumers
                    .entry(input_name.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        // Compute lifetimes
        for (tensor_name, &producer_idx) in &tensor_producers {
            let first_use = producer_idx;

            // Find the last consumer, or use the end of execution if this is an output
            let is_output = graph.output_nodes.iter().any(|&node_id| {
                let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                    Error::InvalidGraph(format!("Node with ID {} not found", node_id))
                }).unwrap_or_else(|_| panic!("Invalid graph: Node not found"));
                node.outputs.contains(tensor_name)
            });

            let last_use = if is_output {
                execution_order.len() // Keep until the end
            } else if let Some(consumers) = tensor_consumers.get(tensor_name) {
                consumers.iter().cloned().max().unwrap_or(first_use)
            } else {
                first_use // No consumers, tensor is used only once
            };

            // Get the tensor ID from the tensor_id_map
            let tensor_id = *tensor_id_map.get(tensor_name).ok_or_else(|| {
                Error::InvalidModel(format!("Missing ID for tensor {}", tensor_name))
            })?;

            lifetimes.insert(tensor_id, (first_use, last_use));
        }

        Ok(lifetimes)
    }

    /// Gather information about tensors in the graph
    fn gather_tensor_info(&self, graph: &ExecutionGraph) -> Result<HashMap<TensorId, TensorMemoryInfo>> {
        let mut tensor_info = HashMap::new();
        let mut tensor_id_map = HashMap::new();

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = self.generate_tensor_id();
            tensor_id_map.insert(name, id);
        }

        // Process all nodes in the graph
        for node in &graph.nodes {
            // Process outputs from this node
            for output_name in &node.outputs {
                if output_name.is_empty() {
                    continue;
                }
                
                // Get the tensor ID from the map
                let tensor_id = *tensor_id_map.get(output_name).ok_or_else(|| {
                    Error::InvalidModel(format!("Missing ID for tensor {}", output_name))
                })?;

                // Determine if this is a model output
                let is_model_output = graph.output_nodes.iter().any(|&node_id| {
                    let node = graph.nodes.iter().find(|n| n.id == node_id).unwrap();
                    node.outputs.contains(output_name)
                });

                // Determine tensor shape and data type based on operation
                let (shape, data_type) = self.infer_tensor_info(graph, node, output_name)?;
                
                // Calculate size in bytes based on shape and data type
                let element_size = data_type.size_in_bytes();
                let total_elements: usize = shape.iter().product();
                
                // Handle potential overflow
                let size_bytes = total_elements.checked_mul(element_size).ok_or_else(|| {
                    Error::InvalidModel(format!(
                        "Integer overflow calculating size for tensor {} with {} elements of size {} bytes",
                        output_name, total_elements, element_size
                    ))
                })?;
                
                // Determine appropriate alignment for the data type
                // For SIMD operations, align to cache line boundaries (64 bytes) for float32/64 tensors
                // and tensors with dimensions divisible by SIMD vector lengths
                let alignment = if data_type == DataType::Float32 || data_type == DataType::Float64 {
                    let is_simd_friendly = shape.iter().any(|&dim| dim % 8 == 0 || dim % 16 == 0);
                    if is_simd_friendly && size_bytes >= 64 {
                        64 // Cache line size for SIMD-friendly operations
                    } else if size_bytes >= 32 {
                        32 // For smaller but still SIMD-usable tensors
                    } else {
                        16 // Minimum alignment for float tensors
                    }
                } else {
                    // For other data types, use a reasonable alignment based on the element size
                    std::cmp::max(element_size, 8) // At least 8-byte alignment for all tensors
                };

                tensor_info.insert(
                    tensor_id,
                    TensorMemoryInfo {
                        id: tensor_id,
                        name: output_name.clone(),
                        size_bytes,
                        data_type,
                        alignment,
                        allow_inplace: !is_model_output, // Outputs can be overwritten unless they're model outputs
                    },
                );
            }

            // Process inputs as well
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue; // Skip empty inputs (optional)
                }

                // Get the tensor ID from the map
                let tensor_id = *tensor_id_map.get(input_name).ok_or_else(|| {
                    Error::InvalidModel(format!("Missing ID for tensor {}", input_name))
                })?;

                // Only add if not already present
                if !tensor_info.contains_key(&tensor_id) {
                    // For inputs, try to find the tensor info from the graph's inputs
                    let (shape, data_type) = self.get_input_tensor_info(graph, input_name)?;
                    
                    // Calculate size based on shape and data type
                    let element_size = data_type.size_in_bytes();
                    let total_elements: usize = shape.iter().product();
                    
                    // Handle potential overflow
                    let size_bytes = total_elements.checked_mul(element_size).ok_or_else(|| {
                        Error::InvalidModel(format!(
                            "Integer overflow calculating size for tensor {} with {} elements of size {} bytes",
                            input_name, total_elements, element_size
                        ))
                    })?;
                    
                    // Determine appropriate alignment
                    let alignment = if data_type == DataType::Float32 || data_type == DataType::Float64 {
                        let is_simd_friendly = shape.iter().any(|&dim| dim % 8 == 0 || dim % 16 == 0);
                        if is_simd_friendly && size_bytes >= 64 {
                            64 // Cache line size for SIMD-friendly operations
                        } else {
                            16 // Minimum alignment for float tensors
                        }
                    } else {
                        // For other data types, use a reasonable alignment based on the element size
                        std::cmp::max(element_size, 8) // At least 8-byte alignment for all tensors
                    };

                    tensor_info.insert(
                        tensor_id,
                        TensorMemoryInfo {
                            id: tensor_id,
                            name: input_name.clone(),
                            size_bytes,
                            data_type,
                            alignment,
                            allow_inplace: false, // Inputs should not be overwritten
                        },
                    );
                }
            }
        }

        Ok(tensor_info)
    }
    
    /// Infer tensor shape and data type based on operation and inputs
    fn infer_tensor_info(&self, graph: &ExecutionGraph, node: &Node, tensor_name: &str) -> Result<(Vec<usize>, DataType)> {
        // First, try to get the shape from the shape inference system
        if let Ok(shape_cache) = self.shape_inference.shape_cache.read() {
            if let Ok(data_type_cache) = self.shape_inference.data_type_cache.read() {
                // Check if we have the shape in the cache
                if let Some(shape) = shape_cache.get(tensor_name) {
                    // Get the data type if available, or use Float32 as default
                    let data_type = data_type_cache.get(tensor_name).cloned().unwrap_or(DataType::Float32);
                    
                    // Use a reasonable default size (64) for dynamic dimensions
                    let concrete_dims = shape.concrete_dims(64);
                    
                    return Ok((concrete_dims, data_type));
                }
            }
        }
        
        // If shape inference didn't provide a shape, use the legacy method
        // This is a fallback for compatibility with older models or when shape inference fails
        match node.op_type.as_str() {
            "Conv" => {
                // For convolution, we need to get the input shape, kernel shape, etc.
                // This is simplified - real implementation would parse attributes
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    // Output shape depends on input shape, kernel size, stride, padding, etc.
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let kernel_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    if input_shape.len() == 4 && kernel_shape.len() == 4 {
                        // NCHW layout: [batch, channels, height, width]
                        let batch_size = input_shape[0];
                        let out_channels = kernel_shape[0];
                        
                        // Get attributes to properly calculate output dimensions
                        let strides = node.attributes
                            .get("strides")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![1, 1]);
                            
                        let pads = node.attributes
                            .get("pads")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![0, 0, 0, 0]);
                            
                        let dilations = node.attributes
                            .get("dilations")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![1, 1]);
                        
                        // Calculate output height and width
                        let out_h = (input_shape[2] + pads[0] + pads[2] - ((kernel_shape[2] - 1) * dilations[0] + 1)) / strides[0] + 1;
                        let out_w = (input_shape[3] + pads[1] + pads[3] - ((kernel_shape[3] - 1) * dilations[1] + 1)) / strides[1] + 1;
                        
                        return Ok((vec![batch_size, out_channels, out_h, out_w], DataType::Float32));
                    }
                }
            },
            "MatMul" => {
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    let a_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let b_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    if a_shape.len() >= 2 && b_shape.len() >= 2 {
                        // For MatMul: [M,K] * [K,N] -> [M,N]
                        let m = a_shape[a_shape.len() - 2];
                        let n = b_shape[b_shape.len() - 1];
                        
                        // Handle broadcasting if tensors have more than 2 dimensions
                        let mut result_shape = Vec::new();
                        
                        if a_shape.len() > 2 || b_shape.len() > 2 {
                            // Broadcast the batch dimensions
                            let a_batch_dims = &a_shape[0..a_shape.len() - 2];
                            let b_batch_dims = &b_shape[0..b_shape.len() - 2];
                            
                            // Compute the broadcasted batch dimensions
                            let batch_dims = self.broadcast_shapes(a_batch_dims, b_batch_dims)?;
                            result_shape.extend_from_slice(&batch_dims);
                        }
                        
                        // Add the matrix multiplication result dimensions
                        result_shape.push(m);
                        result_shape.push(n);
                        
                        return Ok((result_shape, DataType::Float32));
                    }
                }
            },
            // Add cases for common operations: Add, Mul, Relu, etc.
            "Add" | "Sub" | "Mul" | "Div" => {
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    let a_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let b_shape = self.get_tensor_shape(graph, &node.inputs[1])?;
                    
                    // For elementwise ops, output shape is the broadcasted shape
                    let output_shape = self.broadcast_shapes(&a_shape, &b_shape)?;
                    
                    // Output data type is the same as input (assuming same type inputs)
                    let data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((output_shape, data_type));
                }
            },
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" => {
                if node.inputs.len() >= 1 && tensor_name == &node.outputs[0] {
                    // These activation functions preserve shape and data type
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let input_data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((input_shape, input_data_type));
                }
            },
            "MaxPool" | "AveragePool" => {
                if node.inputs.len() >= 1 && tensor_name == &node.outputs[0] {
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    
                    if input_shape.len() >= 4 {
                        // Get attributes to properly calculate output dimensions
                        let kernel_shape = node.attributes
                            .get("kernel_shape")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![2, 2]);
                            
                        let strides = node.attributes
                            .get("strides")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![1, 1]);
                            
                        let pads = node.attributes
                            .get("pads")
                            .and_then(|attr| if let Attribute::Ints(ints) = attr { 
                                Some(ints.iter().map(|&i| i as usize).collect::<Vec<_>>()) 
                            } else { None })
                            .unwrap_or_else(|| vec![0, 0, 0, 0]);
                        
                        // Calculate output height and width
                        let out_h = (input_shape[2] + pads[0] + pads[2] - kernel_shape[0]) / strides[0] + 1;
                        let out_w = (input_shape[3] + pads[1] + pads[3] - kernel_shape[1]) / strides[1] + 1;
                        
                        // For pooling, N and C dimensions are preserved
                        return Ok((vec![input_shape[0], input_shape[1], out_h, out_w], DataType::Float32));
                    }
                }
            },
            "GlobalAveragePool" | "GlobalMaxPool" => {
                if node.inputs.len() >= 1 && tensor_name == &node.outputs[0] {
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    
                    if input_shape.len() >= 4 {
                        // Global pooling reduces spatial dimensions to 1
                        let mut output_shape = input_shape.clone();
                        for i in 2..output_shape.len() {
                            output_shape[i] = 1;
                        }
                        
                        return Ok((output_shape, DataType::Float32));
                    }
                }
            },
            "Reshape" => {
                if node.inputs.len() >= 2 && tensor_name == &node.outputs[0] {
                    // Reshaping requires knowing the target shape
                    // In most cases, this would come from a constant tensor
                    // For simplicity, we'll use the input's total size and guess a reasonable shape
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    
                    // If we can't determine the reshape parameters at this point,
                    // we'll use a fallback shape (later in this function)
                    // In a production system, you would extract the actual shape from the second input
                    
                    // Use the data type from the input
                    let data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    // Try to estimate a reasonable reshape based on the total number of elements
                    let total_elements: usize = input_shape.iter().product();
                    
                    // Flatten to 2D as a reasonable default
                    let batch_size = input_shape[0];
                    let feature_size = total_elements / batch_size;
                    
                    return Ok((vec![batch_size, feature_size], data_type));
                }
            },
            "Concat" => {
                if !node.inputs.is_empty() && tensor_name == &node.outputs[0] {
                    // Get the axis attribute
                    let axis = node.attributes
                        .get("axis")
                        .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i as i32) } else { None })
                        .unwrap_or(0);
                    
                    // Start with the first input's shape
                    let mut output_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    
                    // Adjust the dimension at the concatenation axis
                    let pos_axis = if axis < 0 { (axis + output_shape.len() as i32) as usize } else { axis as usize };
                    
                    // Sum up the sizes along the concatenation axis
                    if pos_axis < output_shape.len() {
                        let mut axis_size = output_shape[pos_axis];
                        
                        for i in 1..node.inputs.len() {
                            let shape = self.get_tensor_shape(graph, &node.inputs[i])?;
                            if pos_axis < shape.len() {
                                axis_size += shape[pos_axis];
                            }
                        }
                        
                        output_shape[pos_axis] = axis_size;
                    }
                    
                    let data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    return Ok((output_shape, data_type));
                }
            },
            // Default case for unknown operations
            _ => {
                // For unknown operations, attempt to copy shape from first input
                if !node.inputs.is_empty() && tensor_name == &node.outputs[0] {
                    let input_shape = self.get_tensor_shape(graph, &node.inputs[0])?;
                    let input_data_type = self.get_tensor_data_type(graph, &node.inputs[0])?;
                    
                    return Ok((input_shape, input_data_type));
                }
            }
        }
        
        // If we couldn't determine the shape through any method, log a warning
        eprintln!("Warning: Could not infer shape for tensor {}. Using default shape.", tensor_name);
        
        // Fallback: If we can't infer the shape, use a reasonable default
        // We'll generate a more informative error message
        Ok((vec![1, 1, 64, 64], DataType::Float32))
    }
    
    /// Get the shape of a tensor from the graph
    fn get_tensor_shape(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<Vec<usize>> {
        // First, try to get the shape from the shape inference system
        if let Ok(shape_cache) = self.shape_inference.shape_cache.read() {
            if let Some(shape) = shape_cache.get(tensor_name) {
                // Use a reasonable default (64) for dynamic dimensions
                return Ok(shape.concrete_dims(64));
            }
        }
        
        // If shape is not available from the inference system, look for shape information
        // in the graph (from TensorInfo)
        if let Some(onnx_model) = &graph.onnx_model {
            // Check model inputs
            for input in &onnx_model.graph.inputs {
                if input.name == tensor_name {
                    // Convert i64 dims to usize dims, using 64 for dynamic dimensions (negative values)
                    let shape: Vec<usize> = input.shape.iter()
                        .map(|&dim| if dim >= 0 { dim as usize } else { 64 })
                        .collect();
                    return Ok(shape);
                }
            }
            
            // Check model outputs
            for output in &onnx_model.graph.outputs {
                if output.name == tensor_name {
                    let shape: Vec<usize> = output.shape.iter()
                        .map(|&dim| if dim >= 0 { dim as usize } else { 64 })
                        .collect();
                    return Ok(shape);
                }
            }
            
            // Check intermediate values
            for value in &onnx_model.graph.value_info {
                if value.name == tensor_name {
                    let shape: Vec<usize> = value.shape.iter()
                        .map(|&dim| if dim >= 0 { dim as usize } else { 64 })
                        .collect();
                    return Ok(shape);
                }
            }
            
            // Check initializers
            for initializer in &onnx_model.graph.initializers {
                if initializer.name == tensor_name {
                    let shape: Vec<usize> = initializer.dims.iter()
                        .map(|&dim| if dim >= 0 { dim as usize } else { 64 })
                        .collect();
                    return Ok(shape);
                }
            }
        }
            
        // If the shape is not available from any of the above sources,
        // use heuristics based on tensor name patterns
        
        if tensor_name.contains("weight") || tensor_name.contains("kernel") {
            // Weights for conv are typically [out_channels, in_channels, kernel_h, kernel_w]
            Ok(vec![64, 3, 3, 3])
        } else if tensor_name.contains("bias") {
            // Bias is typically [out_channels]
            Ok(vec![64])
        } else if tensor_name.contains("input") || tensor_name.contains("image") {
            // Input tensors are typically [batch_size, channels, height, width]
            Ok(vec![1, 3, 224, 224])
        } else if tensor_name.contains("pool") || tensor_name.contains("feature") {
            // Feature maps often have shape [batch_size, channels, height, width]
            Ok(vec![1, 64, 112, 112])
        } else if tensor_name.contains("fc") || tensor_name.contains("dense") {
            // Fully connected layers typically have shape [batch_size, features]
            Ok(vec![1, 1024])
        } else if tensor_name.contains("output") || tensor_name.contains("logits") {
            // Output tensors often have shape [batch_size, num_classes]
            Ok(vec![1, 1000])
        } else {
            // Default shape for unknown tensors
            Ok(vec![1, 64, 64, 64])
        }
    }
    
    /// Get the data type of a tensor from the graph
    fn get_tensor_data_type(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<DataType> {
        // First, try to get the data type from the shape inference system
        if let Ok(data_type_cache) = self.shape_inference.data_type_cache.read() {
            if let Some(data_type) = data_type_cache.get(tensor_name) {
                return Ok(*data_type);
            }
        }
        
        // If data type is not available from the inference system, look for type information
        // in the graph (from TensorInfo)
        if let Some(onnx_model) = &graph.onnx_model {
            // Check model inputs
            for input in &onnx_model.graph.inputs {
                if input.name == tensor_name {
                    return Ok(convert_model_data_type(input.data_type));
                }
            }
            
            // Check model outputs
            for output in &onnx_model.graph.outputs {
                if output.name == tensor_name {
                    return Ok(convert_model_data_type(output.data_type));
                }
            }
            
            // Check intermediate values
            for value in &onnx_model.graph.value_info {
                if value.name == tensor_name {
                    return Ok(convert_model_data_type(value.data_type));
                }
            }
            
            // Check initializers
            for initializer in &onnx_model.graph.initializers {
                if initializer.name == tensor_name {
                    return Ok(convert_model_data_type(initializer.data_type));
                }
            }
        }
        
        // Default to Float32 if we couldn't determine the data type
        Ok(DataType::Float32)
    }
    
    /// Get tensor info for input tensors
    fn get_input_tensor_info(&self, graph: &ExecutionGraph, tensor_name: &str) -> Result<(Vec<usize>, DataType)> {
        // Leverage existing methods that now use the shape inference system
        let shape = self.get_tensor_shape(graph, tensor_name)?;
        let data_type = self.get_tensor_data_type(graph, tensor_name)?;
        
        Ok((shape, data_type))
    }
    
    /// Broadcast two shapes according to broadcasting rules
    fn broadcast_shapes(&self, shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
        let rank1 = shape1.len();
        let rank2 = shape2.len();
        let result_rank = std::cmp::max(rank1, rank2);
        
        let mut result_shape = Vec::with_capacity(result_rank);
        
        for i in 0..result_rank {
            let dim1 = if i >= result_rank - rank1 {
                shape1[i - (result_rank - rank1)]
            } else {
                1
            };
            
            let dim2 = if i >= result_rank - rank2 {
                shape2[i - (result_rank - rank2)]
            } else {
                1
            };
            
            if dim1 == 1 {
                result_shape.push(dim2);
            } else if dim2 == 1 {
                result_shape.push(dim1);
            } else if dim1 == dim2 {
                result_shape.push(dim1);
            } else {
                return Err(Error::InvalidModel(format!(
                    "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions at index {}",
                    shape1, shape2, i
                )));
            }
        }
        
        Ok(result_shape)
    }
    
    /// Collect all tensor names from the graph
    fn collect_tensor_names(&self, graph: &ExecutionGraph) -> HashSet<String> {
        let mut tensor_names = HashSet::new();
        
        // Add all inputs and outputs from all nodes
        for node in &graph.nodes {
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    tensor_names.insert(input_name.clone());
                }
            }
            
            for output_name in &node.outputs {
                if !output_name.is_empty() {
                    tensor_names.insert(output_name.clone());
                }
            }
        }
        
        tensor_names
    }

    /// Analyze opportunities for in-place operations
    pub fn inplace_operations_analysis(&self, graph: &ExecutionGraph) -> Result<Vec<InplaceOpportunity>> {
        let mut opportunities = Vec::new();
        let tensor_info = self.gather_tensor_info(graph)?;
        let tensor_id_map = self.create_tensor_id_map(graph)?;

        for node in &graph.nodes {
            // Different operations have different in-place capabilities
            match node.op_type.as_str() {
                // Unary operations that can generally be done in-place
                "Relu" | "LeakyRelu" | "Sigmoid" | "Tanh" | "Abs" | "Exp" | "Log" | "Sqrt" | "Neg" => {
                if !node.inputs.is_empty() && !node.inputs[0].is_empty() && !node.outputs.is_empty() {
                    let input_name = &node.inputs[0];
                    let output_name = &node.outputs[0];
                    
                    // Get tensor IDs from the map
                    if let (Some(&input_id), Some(&output_id)) = (
                        tensor_id_map.get(input_name),
                        tensor_id_map.get(output_name)
                    ) {
                        // Check if input can be overwritten
                        if let Some(input_info) = tensor_info.get(&input_id) {
                            if input_info.allow_inplace {
                                // Check that input is only used once (in this operation)
                                if self.is_last_use_of_tensor(graph, input_name, node.id)? {
                                    // Get the size of the tensor
                                    let size_bytes = input_info.size_bytes;
                                    
                                    opportunities.push(InplaceOpportunity {
                                        node_id: node.id,
                                        input_id,
                                        output_id,
                                        size_bytes,
                                    });
                                }
                            }
                        }
                    }
                }
            },
            _ => {}
        }
    }
        
        Ok(opportunities)
    }
    
    /// Create a mapping from tensor names to tensor IDs
    fn create_tensor_id_map(&self, graph: &ExecutionGraph) -> Result<HashMap<String, TensorId>> {
        let mut tensor_id_map = HashMap::new();

        // First, check if we have a cached mapping
        if let Ok(cache) = self.tensor_id_map_cache.read() {
            if !cache.is_empty() {
                return Ok(cache.clone());
            }
        }

        // Create a mapping from tensor names to unique IDs
        let tensor_names = self.collect_tensor_names(graph);
        for name in tensor_names {
            let id = self.generate_tensor_id();
            tensor_id_map.insert(name, id);
        }
        
        // Update cache
        if let Ok(mut cache) = self.tensor_id_map_cache.write() {
            *cache = tensor_id_map.clone();
        }
        
        Ok(tensor_id_map)
    }
    
    /// Check if this is the last use of a tensor in the graph
    fn is_last_use_of_tensor(&self, graph: &ExecutionGraph, tensor_name: &str, current_node_id: NodeId) -> Result<bool> {
        // Get the execution order
        let execution_order = self.determine_execution_order(graph)?;
        
        // Find the position of the current node
        let current_pos = execution_order.iter().position(|&id| id == current_node_id).ok_or_else(|| {
            Error::InvalidGraph(format!("Node with ID {} not found in execution order", current_node_id))
        })?;
        
        // Check all nodes after the current one
        for &node_id in &execution_order[current_pos + 1..] {
            let node = graph.nodes.iter().find(|n| n.id == node_id).ok_or_else(|| {
                Error::InvalidGraph(format!("Node with ID {} not found", node_id))
            })?;
            
            // If any node uses this tensor as input, it's not the last use
            if node.inputs.contains(&tensor_name) {
                return Ok(false);
            }
        }
        
        // Check if the tensor is a model output
        let is_model_output = graph.output_nodes.iter().any(|&node_id| {
            graph.nodes.iter()
                .find(|n| n.id == node_id)
                .map(|node| node.outputs.contains(&tensor_name))
                .unwrap_or_else(|| {
                    // Log the missing node but continue with a reasonable default
                    eprintln!("Warning: Node with ID {} not found in graph", node_id);
                    false
                })
        });
        
        // If it's a model output, it cannot be used in-place
        if is_model_output {
            return Ok(false);
        }
        
        // If we haven't found any further uses, it's the last use
        Ok(true)
    }

    /// Apply in-place optimizations to tensor info and lifetimes
    fn apply_inplace_optimizations(
        &self,
        mut tensor_info: HashMap<TensorId, TensorMemoryInfo>,
        mut lifetimes: HashMap<TensorId, (usize, usize)>,
        inplace_ops: Vec<InplaceOpportunity>,
    ) -> Result<(HashMap<TensorId, TensorMemoryInfo>, HashMap<TensorId, (usize, usize)>)> {
        for op in &inplace_ops {
            // Check if both tensors exist in our maps
            if let (Some(input_info), Some(output_info)) = (
                tensor_info.get(&op.input_id),
                tensor_info.get(&op.output_id),
            ) {
                // Check if the input tensor allows in-place operations
                if input_info.allow_inplace
                    // Check data types match
                    && input_info.data_type == output_info.data_type
                    // Check sizes are compatible
                    && input_info.size_bytes >= output_info.size_bytes
                {
                    // Extend the lifetime of the input tensor to cover the output tensor
                    if let (Some(&(input_first, input_last)), Some(&(output_first, output_last))) = (
                        lifetimes.get(&op.input_id),
                        lifetimes.get(&op.output_id),
                    ) {
                        // Update the input tensor's lifetime to be the union of both
                        let new_lifetime = (
                            cmp::min(input_first, output_first),
                            cmp::max(input_last, output_last),
                        );
                        lifetimes.insert(op.input_id, new_lifetime);

                        // Remove the output tensor from planning
                        // (it will share memory with the input)
                        lifetimes.remove(&op.output_id);
                        tensor_info.remove(&op.output_id);

                        // In a real implementation, you would add the output tensor
                        // to a map that tracks which tensors share memory
                    }
                }
            }
        }

        Ok((tensor_info, lifetimes))
    }

    /// Analyze opportunities for buffer sharing
    pub fn buffer_sharing_analysis(
        &self,
        lifetimes: &HashMap<TensorId, (usize, usize)>,
    ) -> Result<Vec<SharingOpportunity>> {
        let mut opportunities = Vec::new();

        // Create a list of tensors with their sizes
        let mut tensors_with_sizes: Vec<(TensorId, usize, (usize, usize))> = Vec::new();
        
        // Thread-safe access to tensor info cache
        let tensor_info = match self.tensor_info_cache.read() {
            Ok(cache) => cache,
            Err(_) => return Err(Error::InvalidModel("Failed to acquire read lock for tensor_info_cache".to_string())),
        };
        
        for (&tensor_id, &lifetime) in lifetimes.iter() {
            if let Some(info) = tensor_info.get(&tensor_id) {
                tensors_with_sizes.push((tensor_id, info.size_bytes, lifetime));
            }
        }

        // Sort tensors by size (largest first)
        tensors_with_sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Check each pair of tensors for sharing opportunities
        for i in 0..tensors_with_sizes.len() {
            let (id1, size1, (first1, last1)) = tensors_with_sizes[i];
            
            // Skip tensors that require special alignment if they're very small
            // (sharing very small tensors often doesn't save much memory)
            if size1 < 64 {
                continue;
            }
            
            for j in i + 1..tensors_with_sizes.len() {
                let (id2, size2, (first2, last2)) = tensors_with_sizes[j];

                // Check if lifetimes don't overlap
                if last1 < first2 || last2 < first1 {
                    // Tensors don't overlap in time, check alignment compatibility
                    if let (Some(info1), Some(info2)) = (
                        tensor_info.get(&id1), 
                        tensor_info.get(&id2)
                    ) {
                        // Check if they have compatible alignment requirements
                        // For simplicity, we'll only share buffers with the same alignment
                        // A more sophisticated implementation would adjust offsets to satisfy
                        // both alignment requirements
                        if info1.alignment == info2.alignment {
                            // Check data type compatibility
                            // For some data types, we might require additional padding
                            let compatible_types = (info1.data_type == info2.data_type) ||
                                               (info1.data_type.size_in_bytes() == info2.data_type.size_in_bytes());
                            
                            if compatible_types {
                                // Use the smaller of the two sizes for the amount saved
                                let shared_size = std::cmp::min(size1, size2);
                                
                                opportunities.push(SharingOpportunity {
                                    first_id: id1,
                                    second_id: id2,
                                    size_bytes: shared_size,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Apply heuristic to limit the number of sharing opportunities
        // Too many sharing edges can make allocation complex
        if opportunities.len() > 100 {
            // Sort by size saved (largest first) and take the top 100
            opportunities.sort_unstable_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
            opportunities.truncate(100);
        }

        Ok(opportunities)
    }

    /// Compute the optimal allocation order for tensors
    pub fn compute_optimal_allocation_order(
        &self,
        sizes: &[(TensorId, usize)],
        lifetimes: &HashMap<TensorId, (usize, usize)>,
    ) -> Vec<TensorId> {
        // Sort tensors by size (largest first)
        let mut ids_with_sizes = sizes.to_vec();
        ids_with_sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        
        // Extract just the tensor IDs
        let mut result: Vec<_> = ids_with_sizes.iter().map(|(id, _)| *id).collect();
        
        // Further sort by lifetime length (longest first) for tensors of the same size
        result.sort_by(|&a, &b| {
            let size_a = ids_with_sizes.iter().find(|(id, _)| *id == a).unwrap().1;
            let size_b = ids_with_sizes.iter().find(|(id, _)| *id == b).unwrap().1;
            
            if size_a == size_b {
                let lifetime_a = lifetimes.get(&a).map(|(first, last)| last - first).unwrap_or(0);
                let lifetime_b = lifetimes.get(&b).map(|(first, last)| last - first).unwrap_or(0);
                lifetime_b.cmp(&lifetime_a)
            } else {
                size_b.cmp(&size_a)
            }
        });
        
        result
    }
    
    /// Optimize memory layout for a memory plan
    pub fn optimize_memory_layout(&self, plan: &mut MemoryPlan) -> Result<usize> {
        // Start with all tensors in a single buffer
        let mut buffer_index = 0;
        let mut current_offset = 0;
        let mut max_offset = 0;
        let mut bytes_saved = 0;
        
        // Get sharing opportunities
        let sharing_opportunities = self.buffer_sharing_analysis(&plan.lifetimes)?;
        
        // Create a map of tensor sizes
        let tensor_sizes: Vec<_> = plan.tensor_info.iter()
            .map(|(id, info)| (*id, info.size_bytes))
            .collect();
        
        // Compute optimal allocation order
        let allocation_order = self.compute_optimal_allocation_order(
            &tensor_sizes,
            &plan.lifetimes,
        );
        
        // Track allocated tensors and their buffer ranges
        let mut allocated_ranges = Vec::new();
        
        // Allocate tensors in optimal order
        for tensor_id in allocation_order {
            // Skip if already allocated
            if plan.allocations.contains_key(&tensor_id) {
                continue;
            }
            
            // Get tensor info
            let info = if let Some(info) = plan.tensor_info.get(&tensor_id) {
                info
            } else {
                continue; // Skip if no info (e.g., in-place outputs)
            };
            
            // Get tensor lifetime
            let lifetime = if let Some(&lifetime) = plan.lifetimes.get(&tensor_id) {
                lifetime
            } else {
                continue; // Skip if no lifetime
            };
            
            // Try to find a space in the buffer where this tensor can fit
            // without overlapping with other tensors' lifetimes
            let alignment = info.alignment;
            let size = info.size_bytes;
            
            // Find the earliest position where we can allocate this tensor
            let mut can_allocate_at = 0;
            let mut found_space = false;
            
            // Align to required alignment
            let align_offset = |offset: usize, alignment: usize| -> usize {
                (offset + alignment - 1) & !(alignment - 1)
            };
            
            while !found_space {
                // Align the offset
                can_allocate_at = align_offset(can_allocate_at, alignment);
                
                // Check if this offset overlaps with any allocated tensor
                let end_offset = can_allocate_at + size;
                let mut overlaps = false;
                
                for &(other_id, other_offset, other_size) in &allocated_ranges {
                    // Get the other tensor's lifetime
                    let other_lifetime = if let Some(&lifetime) = plan.lifetimes.get(&other_id) {
                        lifetime
                    } else {
                        continue;
                    };
                    
                    // Check if the lifetimes overlap
                    let lifetimes_overlap = !(lifetime.1 < other_lifetime.0 || other_lifetime.1 < lifetime.0);
                    
                    // Check if the memory regions overlap
                    let regions_overlap = !(end_offset <= other_offset || can_allocate_at >= other_offset + other_size);
                    
                    if lifetimes_overlap && regions_overlap {
                        overlaps = true;
                        can_allocate_at = other_offset + other_size;
                        break;
                    }
                }
                
                if !overlaps {
                    found_space = true;
                }
            }
            
            // Allocate the tensor at the found offset
            let allocation = TensorAllocation {
                tensor_id,
                offset: can_allocate_at,
                size_bytes: size,
                buffer_index,
            };
            
            // Update tracking structures
            plan.allocations.insert(tensor_id, allocation);
            allocated_ranges.push((tensor_id, can_allocate_at, size));
            
            // Update the maximum offset
            max_offset = cmp::max(max_offset, can_allocate_at + size);
        }
        
        // Apply buffer sharing optimizations
        for op in &sharing_opportunities {
            // If both tensors are allocated, we've already optimized them
            if plan.allocations.contains_key(&op.first_id) && plan.allocations.contains_key(&op.second_id) {
                continue;
            }
            
            // If one is allocated and the other isn't, it means the other is
            // part of an in-place operation and has been removed
            if plan.allocations.contains_key(&op.first_id) && !plan.allocations.contains_key(&op.second_id) {
                bytes_saved += op.size_bytes;
            } else if !plan.allocations.contains_key(&op.first_id) && plan.allocations.contains_key(&op.second_id) {
                bytes_saved += op.size_bytes;
            }
            // If neither is allocated, both might be part of in-place operations
        }
        
        // Update buffer sizes
        plan.buffer_sizes = vec![max_offset];
        plan.total_memory_bytes = max_offset;
        
        Ok(bytes_saved)
    }
    
    /// Allocate memory buffers according to the memory plan
    pub fn allocate_buffers_from_plan(
        &self,
        plan: &MemoryPlan,
        allocator: &mut dyn MemoryAllocator,
    ) -> Result<BufferMap> {
        let mut buffer_map = HashMap::new();
        
        // Allocate each buffer
        let mut buffer_blocks = Vec::new();
        for (i, &size) in plan.buffer_sizes.iter().enumerate() {
            // For simplicity, we'll use a common alignment for all buffers
            // In practice, this would depend on the tensors' requirements
            let alignment = 64;
            
            let block = allocator.allocate(size, alignment)?;
            buffer_blocks.push(block);
        }
        
        // Assign memory blocks to tensors based on allocations
        for (tensor_id, allocation) in &plan.allocations {
            let buffer_index = allocation.buffer_index;
            if buffer_index >= buffer_blocks.len() {
                return Err(Error::InvalidModel(format!(
                    "Buffer index {} out of bounds (only {} buffers allocated)",
                    buffer_index, buffer_blocks.len()
                )));
            }
            
            // Get the base memory block
            let base_block = &buffer_blocks[buffer_index];
            
            // Create a sub-block for this tensor
            // Calculate pointer with bounds checking
            if allocation.offset > base_block.size() {
                return Err(Error::InvalidModel(format!(
                    "Invalid allocation offset {} exceeds base block size {}",
                    allocation.offset, base_block.size()
                )));
            }
            
            let remaining_size = base_block.size() - allocation.offset;
            if allocation.size_bytes > remaining_size {
                return Err(Error::InvalidModel(format!(
                    "Invalid allocation size {} exceeds remaining space {} at offset {}",
                    allocation.size_bytes, remaining_size, allocation.offset
                )));
            }
            
            let ptr = unsafe {
                let base_ptr = base_block.ptr().as_ptr();
                let tensor_ptr = base_ptr.add(allocation.offset);
                NonNull::new_unchecked(tensor_ptr)
            };
            
            let tensor_block = MemoryBlock::new(
                ptr,
                allocation.size_bytes,
                base_block.alignment(),
                allocation.offset,
            );
            
            buffer_map.insert(*tensor_id, tensor_block);
        }
        
        Ok(buffer_map)
    }
}

// Needed for creating a sub-block
use std::ptr::NonNull;