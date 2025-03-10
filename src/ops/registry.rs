use std::collections::HashMap;
use crate::error::{Error, Result};
use crate::model::{Node, DataType as ModelDataType};
use super::tensor::{Tensor, Shape};
use std::fmt::Debug;

/// Execution context for operators
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    pub options: HashMap<String, String>,
}

/// Trait for implementing ONNX operators
pub trait Operator: Send + Sync + Debug {
    /// Compute the operation
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], context: &ExecutionContext) -> Result<()>;
    
    /// Infer input shapes
    fn input_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        Ok(input_shapes.to_vec())
    }
    
    /// Infer output shapes
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>>;
    
    /// Validate the operator against a node
    fn validate(&self, node: &Node) -> Result<()>;
}

/// Registry for ONNX operators
#[derive(Debug, Default)]
pub struct OperatorRegistry {
    operators: HashMap<(String, String, i64), Box<dyn Operator>>,
}

impl OperatorRegistry {
    /// Create a new operator registry
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }
    
    /// Register an operator
    pub fn register_operator(&mut self, name: &str, domain: &str, version: i64, op: Box<dyn Operator>) -> Result<()> {
        let key = (name.to_string(), domain.to_string(), version);
        
        if self.operators.contains_key(&key) {
            return Err(Error::InvalidOperator(format!(
                "Operator {}.{} (version {}) is already registered",
                domain, name, version
            )));
        }
        
        self.operators.insert(key, op);
        Ok(())
    }
    
    /// Get an operator by name, domain, and version
    pub fn get_operator(&self, name: &str, domain: &str, version: i64) -> Option<&dyn Operator> {
        let key = (name.to_string(), domain.to_string(), version);
        self.operators.get(&key).map(|op| op.as_ref())
    }
    
    /// Initialize the registry with standard operators
    pub fn initialize_standard_operators() -> Self {
        use crate::ops::math::matmul::MatMul;
        use crate::ops::math::gemm::Gemm;
        use crate::ops::nn::conv::Conv;
        use crate::ops::nn::pool::{MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool};
        use crate::ops::activations::{Relu, LeakyRelu, Sigmoid, Tanh, Elu};
        
        let mut registry = Self::new();
        
        // Register math operators
        registry.register_operator("MatMul", "", 9, Box::new(MatMul::default())).unwrap();
        registry.register_operator("Gemm", "", 9, Box::new(Gemm::default())).unwrap();
        
        // Register neural network operators
        registry.register_operator("Conv", "", 11, Box::new(Conv::default())).unwrap();
        registry.register_operator("MaxPool", "", 11, Box::new(MaxPool::default())).unwrap();
        registry.register_operator("AveragePool", "", 11, Box::new(AveragePool::default())).unwrap();
        registry.register_operator("GlobalAveragePool", "", 1, Box::new(GlobalAveragePool::default())).unwrap();
        registry.register_operator("GlobalMaxPool", "", 1, Box::new(GlobalMaxPool::default())).unwrap();
        
        // Register activation operators
        registry.register_operator("Relu", "", 6, Box::new(Relu::default())).unwrap();
        registry.register_operator("LeakyRelu", "", 6, Box::new(LeakyRelu::default())).unwrap();
        registry.register_operator("Sigmoid", "", 6, Box::new(Sigmoid::default())).unwrap();
        registry.register_operator("Tanh", "", 6, Box::new(Tanh::default())).unwrap();
        registry.register_operator("Elu", "", 6, Box::new(Elu::default())).unwrap();
        
        registry
    }
    
    /// Create an operator for a node
    pub fn create_operator_for_node(&self, node: &Node) -> Result<Box<dyn Operator>> {
        let domain = if node.domain.is_empty() { "" } else { &node.domain };
        
        // First try to get the exact version
        for (version, _) in node.op_type.clone().chars().zip(0..) {
            if let Some(op) = self.get_operator(&node.op_type, domain, version as i64) {
                // Create a new instance of the operator
                return match node.op_type.as_str() {
                    "MatMul" => Ok(Box::new(crate::ops::math::matmul::MatMul::default())),
                    "Gemm" => Ok(Box::new(crate::ops::math::gemm::Gemm::default())),
                    "Conv" => Ok(Box::new(crate::ops::nn::conv::Conv::default())),
                    "MaxPool" => Ok(Box::new(crate::ops::nn::pool::MaxPool::default())),
                    "AveragePool" => Ok(Box::new(crate::ops::nn::pool::AveragePool::default())),
                    "GlobalAveragePool" => Ok(Box::new(crate::ops::nn::pool::GlobalAveragePool::default())),
                    "GlobalMaxPool" => Ok(Box::new(crate::ops::nn::pool::GlobalMaxPool::default())),
                    "Relu" => Ok(Box::new(crate::ops::activations::Relu::default())),
                    "LeakyRelu" => Ok(Box::new(crate::ops::activations::LeakyRelu::default())),
                    "Sigmoid" => Ok(Box::new(crate::ops::activations::Sigmoid::default())),
                    "Tanh" => Ok(Box::new(crate::ops::activations::Tanh::default())),
                    "Elu" => Ok(Box::new(crate::ops::activations::Elu::default())),
                    _ => Err(Error::UnsupportedFeature(format!(
                        "Operator {}.{} not supported",
                        domain, node.op_type
                    ))),
                };
            }
        }
        
        Err(Error::UnsupportedFeature(format!(
            "Operator {}.{} not found in registry",
            domain, node.op_type
        )))
    }
}