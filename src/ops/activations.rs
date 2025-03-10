use crate::error::{Error, Result};
use crate::model::{Node, Attribute};
use crate::ops::registry::{Operator, ExecutionContext};
use crate::ops::tensor::{Tensor, Shape, element_wise_unary_op};

/// Base struct for simple activation operators
#[derive(Debug, Clone)]
pub struct ActivationBase {
    activation_fn: fn(f32) -> f32,
    name: &'static str,
}

/// ReLU activation operator
#[derive(Debug, Clone, Default)]
pub struct Relu;

/// LeakyReLU activation operator
#[derive(Debug, Clone, Default)]
pub struct LeakyRelu {
    alpha: f32,
}

/// Sigmoid activation operator
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

/// Tanh activation operator
#[derive(Debug, Clone, Default)]
pub struct Tanh;

/// ELU activation operator
#[derive(Debug, Clone, Default)]
pub struct Elu {
    alpha: f32,
}

// Implement the base activation operator functionality
impl ActivationBase {
    fn new(activation_fn: fn(f32) -> f32, name: &'static str) -> Self {
        Self { activation_fn, name }
    }
    
    fn compute_impl(&self, inputs: &[&Tensor], outputs: &mut [Tensor]) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("{} requires at least 1 input, got {}", self.name, inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("{} requires at least 1 output, got {}", self.name, outputs.len())
            ));
        }
        
        let x = inputs[0];
        let result = element_wise_unary_op(x, self.activation_fn)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes_impl(&self, _node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        if input_shapes.is_empty() {
            return Err(Error::ValidationError(
                format!("{} requires at least 1 input shape", self.name)
            ));
        }
        
        // Activation functions preserve input shape
        Ok(vec![input_shapes[0].clone()])
    }
    
    fn validate_impl(&self, node: &Node) -> Result<()> {
        if node.inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("{} operator requires at least 1 input, got {}", self.name, node.inputs.len())
            ));
        }
        
        if node.outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("{} operator requires at least 1 output, got {}", self.name, node.outputs.len())
            ));
        }
        
        Ok(())
    }
}

// Implement the Operator trait for ReLU
impl Operator for Relu {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        let base = ActivationBase::new(|x| if x > 0.0 { x } else { 0.0 }, "Relu");
        base.compute_impl(inputs, outputs)
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        let base = ActivationBase::new(|_| 0.0, "Relu");
        base.output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        let base = ActivationBase::new(|_| 0.0, "Relu");
        base.validate_impl(node)
    }
}

// Implement the Operator trait for LeakyRelu
impl Operator for LeakyRelu {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("LeakyRelu requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("LeakyRelu requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        // Default alpha is 0.01
        let alpha = 0.01;
        
        let x = inputs[0];
        let result = leaky_relu(x, alpha)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        let base = ActivationBase::new(|_| 0.0, "LeakyRelu");
        base.output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        let base = ActivationBase::new(|_| 0.0, "LeakyRelu");
        let result = base.validate_impl(node);
        
        // Check for alpha attribute
        if let Ok(()) = result {
            if let Some(attr) = node.attributes.get("alpha") {
                if !matches!(attr, Attribute::Float(_)) {
                    return Err(Error::ValidationError(
                        "LeakyRelu alpha attribute must be a float".to_string()
                    ));
                }
            }
        }
        
        result
    }
}

// Implement the Operator trait for Sigmoid
impl Operator for Sigmoid {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        let base = ActivationBase::new(|x| 1.0 / (1.0 + (-x).exp()), "Sigmoid");
        base.compute_impl(inputs, outputs)
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        let base = ActivationBase::new(|_| 0.0, "Sigmoid");
        base.output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        let base = ActivationBase::new(|_| 0.0, "Sigmoid");
        base.validate_impl(node)
    }
}

// Implement the Operator trait for Tanh
impl Operator for Tanh {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        let base = ActivationBase::new(|x| x.tanh(), "Tanh");
        base.compute_impl(inputs, outputs)
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        let base = ActivationBase::new(|_| 0.0, "Tanh");
        base.output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        let base = ActivationBase::new(|_| 0.0, "Tanh");
        base.validate_impl(node)
    }
}

// Implement the Operator trait for Elu
impl Operator for Elu {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Elu requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Elu requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        // Default alpha is 1.0
        let alpha = 1.0;
        
        let x = inputs[0];
        let result = elu(x, alpha)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        let base = ActivationBase::new(|_| 0.0, "Elu");
        base.output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        let base = ActivationBase::new(|_| 0.0, "Elu");
        let result = base.validate_impl(node);
        
        // Check for alpha attribute
        if let Ok(()) = result {
            if let Some(attr) = node.attributes.get("alpha") {
                if !matches!(attr, Attribute::Float(_)) {
                    return Err(Error::ValidationError(
                        "Elu alpha attribute must be a float".to_string()
                    ));
                }
            }
        }
        
        result
    }
}

/// ReLU activation function
pub fn relu(x: &Tensor) -> Result<Tensor> {
    element_wise_unary_op(x, |val| if val > 0.0 { val } else { 0.0 })
}

/// LeakyReLU activation function
pub fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    element_wise_unary_op(x, |val| if val > 0.0 { val } else { alpha * val })
}

/// Sigmoid activation function
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    element_wise_unary_op(x, |val| 1.0 / (1.0 + (-val).exp()))
}

/// Tanh activation function
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    element_wise_unary_op(x, |val| val.tanh())
}

/// ELU activation function
pub fn elu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    element_wise_unary_op(x, |val| if val > 0.0 { val } else { alpha * (val.exp() - 1.0) })
}