use crate::error::{Error, Result};
use crate::model::{Node, Attribute};
use crate::ops::registry::{Operator, ExecutionContext};
use crate::ops::tensor::{Tensor, Shape};
use ndarray::{ArrayViewD, ArrayViewMutD};

/// GEMM (General Matrix Multiplication) operator
/// Y = alpha * (A @ B) + beta * C
/// A can be optionally transposed (transA)
/// B can be optionally transposed (transB)
#[derive(Debug, Clone, Default)]
pub struct Gemm {
    // Configuration options can be added here
}

impl Operator for Gemm {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 2 {
            return Err(Error::ValidationError(
                format!("Gemm requires at least 2 inputs, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Gemm requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let a = inputs[0];
        let b = inputs[1];
        let c = if inputs.len() > 2 { Some(inputs[2]) } else { None };
        
        // Default values according to ONNX specification
        let alpha = 1.0;
        let beta = 1.0;
        let transpose_a = false;
        let transpose_b = false;
        
        let result = compute_gemm(a, b, c, alpha, beta, transpose_a, transpose_b)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        if input_shapes.len() < 2 {
            return Err(Error::ValidationError(
                format!("Gemm requires at least 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        // Extract attributes
        let transpose_a = node.attributes.get("transA")
            .and_then(|attr| {
                if let Attribute::Int(val) = attr {
                    Some(*val != 0)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        
        let transpose_b = node.attributes.get("transB")
            .and_then(|attr| {
                if let Attribute::Int(val) = attr {
                    Some(*val != 0)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        
        // Unwrap input shapes
        match (&input_shapes[0], &input_shapes[1]) {
            (Some(a_shape), Some(b_shape)) => {
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    return Err(Error::ValidationError(
                        format!("Gemm inputs must be 2D matrices, got shapes {:?} and {:?}", a_shape, b_shape)
                    ));
                }
                
                let (m, k1) = if transpose_a {
                    (a_shape[1], a_shape[0])
                } else {
                    (a_shape[0], a_shape[1])
                };
                
                let (k2, n) = if transpose_b {
                    (b_shape[1], b_shape[0])
                } else {
                    (b_shape[0], b_shape[1])
                };
                
                if k1 != k2 {
                    return Err(Error::ValidationError(
                        format!("Incompatible dimensions for Gemm: {} and {}", k1, k2)
                    ));
                }
                
                // Check C shape if provided
                if input_shapes.len() > 2 && input_shapes[2].is_some() {
                    let c_shape = input_shapes[2].as_ref().unwrap();
                    
                    if c_shape.len() != 2 || c_shape[0] != m || c_shape[1] != n {
                        return Err(Error::ValidationError(
                            format!("Incompatible C matrix shape for Gemm: expected [{}, {}], got {:?}", m, n, c_shape)
                        ));
                    }
                }
                
                Ok(vec![Some(vec![m, n])])
            },
            _ => {
                // If either shape is unknown, we can't infer the output shape
                Ok(vec![None])
            }
        }
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        if node.inputs.len() < 2 {
            return Err(Error::ValidationError(
                format!("Gemm operator requires at least 2 inputs, got {}", node.inputs.len())
            ));
        }
        
        if node.outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Gemm operator requires at least 1 output, got {}", node.outputs.len())
            ));
        }
        
        // Check attributes
        for (name, attr) in &node.attributes {
            match name.as_str() {
                "alpha" | "beta" => {
                    if !matches!(attr, Attribute::Float(_)) {
                        return Err(Error::ValidationError(
                            format!("Gemm attribute {} must be a float", name)
                        ));
                    }
                },
                "transA" | "transB" => {
                    if !matches!(attr, Attribute::Int(_)) {
                        return Err(Error::ValidationError(
                            format!("Gemm attribute {} must be an int", name)
                        ));
                    }
                },
                _ => {
                    return Err(Error::ValidationError(
                        format!("Unknown attribute for Gemm: {}", name)
                    ));
                }
            }
        }
        
        Ok(())
    }
}

/// Compute strategy for GEMM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GemmStrategy {
    Direct,       // Standard matrix multiplication
    Blocked,      // Block-based multiplication for cache efficiency
    Vectorized,   // Use SIMD if available
}

/// Compute the GEMM operation
pub fn compute_gemm(
    a: &Tensor, 
    b: &Tensor, 
    c: Option<&Tensor>, 
    alpha: f32, 
    beta: f32, 
    transpose_a: bool, 
    transpose_b: bool
) -> Result<Tensor> {
    // Validate input shapes
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(Error::ValidationError(
            format!("Gemm inputs must be 2D matrices, got shapes {:?} and {:?}", a.shape, b.shape)
        ));
    }
    
    // Get dimensions
    let (m, k1) = if transpose_a {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    
    let (k2, n) = if transpose_b {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };
    
    if k1 != k2 {
        return Err(Error::ValidationError(
            format!("Incompatible dimensions for Gemm: {} and {}", k1, k2)
        ));
    }
    
    // Check C shape if provided
    if let Some(c_tensor) = c {
        if c_tensor.shape.len() != 2 {
            return Err(Error::ValidationError(
                format!("Gemm C input must be a 2D matrix, got shape {:?}", c_tensor.shape)
            ));
        }
        
        if c_tensor.shape[0] != m || c_tensor.shape[1] != n {
            return Err(Error::ValidationError(
                format!("Incompatible C matrix shape for Gemm: expected [{}, {}], got {:?}", 
                       m, n, c_tensor.shape)
            ));
        }
    }
    
    // Create output tensor
    let mut result = Tensor::new(&[m, n], a.data_type);
    
    // Choose the best strategy
    let strategy = optimize_gemm_for_shapes(&a.shape, &b.shape, transpose_a, transpose_b);
    
    // Handle various cases based on strategy
    match strategy {
        GemmStrategy::Direct => {
            // Direct implementation using ndarray
            
            // Create views and handle transposition
            let a_view = if transpose_a {
                a.data.view().reversed_axes()
            } else {
                a.data.view()
            };
            
            let b_view = if transpose_b {
                b.data.view().reversed_axes()
            } else {
                b.data.view()
            };
            
            // Perform matrix multiplication: alpha * (A @ B)
            ndarray::linalg::general_mat_mul(
                alpha, 
                &a_view.into_dimensionality().unwrap(), 
                &b_view.into_dimensionality().unwrap(), 
                0.0, 
                &mut result.data.view_mut().into_dimensionality().unwrap()
            );
            
            // Add beta * C if provided
            if let Some(c_tensor) = c {
                if beta != 0.0 {
                    result.data.zip_mut_with(&c_tensor.data, |y, &c_val| {
                        *y += beta * c_val;
                    });
                }
            }
        },
        
        GemmStrategy::Blocked => {
            // Use blocking for better cache efficiency
            gemm_blocked(
                &a.data.view(), 
                &b.data.view(), 
                c.map(|t| t.data.view()), 
                &mut result.data.view_mut(),
                alpha, beta, transpose_a, transpose_b
            );
        },
        
        GemmStrategy::Vectorized => {
            // For simplicity, fallback to blocked implementation
            // In a real implementation, this would use SIMD instructions
            gemm_blocked(
                &a.data.view(), 
                &b.data.view(), 
                c.map(|t| t.data.view()), 
                &mut result.data.view_mut(),
                alpha, beta, transpose_a, transpose_b
            );
        },
    }
    
    Ok(result)
}

/// Determine the best strategy for GEMM computation
pub fn optimize_gemm_for_shapes(
    a_shape: &[usize], 
    b_shape: &[usize], 
    transpose_a: bool, 
    transpose_b: bool
) -> GemmStrategy {
    // This is a simple heuristic based on matrix size
    // A real implementation would consider more factors
    let a_elements = a_shape.iter().product::<usize>();
    let b_elements = b_shape.iter().product::<usize>();
    
    if a_elements < 1000 || b_elements < 1000 {
        // Small matrices - use direct approach
        GemmStrategy::Direct
    } else if a_elements > 100000 || b_elements > 100000 {
        // Large matrices - try vectorized approach if available
        // Otherwise fallback to blocked
        GemmStrategy::Vectorized
    } else {
        // Medium matrices - use blocking for cache efficiency
        GemmStrategy::Blocked
    }
}

/// Blocked GEMM implementation for better cache efficiency
fn gemm_blocked(
    a: &ArrayViewD<f32>,
    b: &ArrayViewD<f32>,
    c: Option<ArrayViewD<f32>>,
    result: &mut ArrayViewMutD<f32>,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool
) {
    // Get dimensions
    let (m, k) = if transpose_a {
        (a.shape()[1], a.shape()[0])
    } else {
        (a.shape()[0], a.shape()[1])
    };
    
    let n = if transpose_b {
        b.shape()[0]
    } else {
        b.shape()[1]
    };
    
    // Choose block sizes
    const BLOCK_SIZE: usize = 64;  // Adjust based on cache size
    
    // Initialize result tensor
    if let Some(c_view) = c {
        if beta != 0.0 {
            // Copy C * beta to result
            for i in 0..m {
                for j in 0..n {
                    result[[i, j]] = beta * c_view[[i, j]];
                }
            }
        } else {
            // Zero the result
            for i in 0..m {
                for j in 0..n {
                    result[[i, j]] = 0.0;
                }
            }
        }
    } else {
        // Zero the result
        for i in 0..m {
            for j in 0..n {
                result[[i, j]] = 0.0;
            }
        }
    }
    
    // Blocked matrix multiplication
    for i in (0..m).step_by(BLOCK_SIZE) {
        let i_end = std::cmp::min(i + BLOCK_SIZE, m);
        
        for j in (0..n).step_by(BLOCK_SIZE) {
            let j_end = std::cmp::min(j + BLOCK_SIZE, n);
            
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                
                // Compute on the current blocks
                for ii in i..i_end {
                    for kk in k_block..k_end {
                        let a_val = if transpose_a {
                            a[[kk, ii]]
                        } else {
                            a[[ii, kk]]
                        };
                        
                        let a_val = alpha * a_val;
                        
                        for jj in j..j_end {
                            let b_val = if transpose_b {
                                b[[jj, kk]]
                            } else {
                                b[[kk, jj]]
                            };
                            
                            result[[ii, jj]] += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
}