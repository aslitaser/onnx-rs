use crate::error::{Error, Result};
use crate::model::Node;
use crate::ops::registry::{Operator, ExecutionContext};
use crate::ops::tensor::{Tensor, Shape, DataType};
use ndarray::{ArrayViewD, ArrayViewMutD, Axis};
use std::cmp::max;

#[derive(Debug, Clone, Default)]
pub struct MatMul {
    // Configuration options could be added here
}

impl Operator for MatMul {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 2 {
            return Err(Error::ValidationError(
                format!("MatMul requires 2 inputs, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("MatMul requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let a = inputs[0];
        let b = inputs[1];
        
        let result = compute_matmul(a, b)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, _node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        if input_shapes.len() < 2 {
            return Err(Error::ValidationError(
                format!("MatMul requires 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        // Unwrap input shapes
        let a_shape_opt = &input_shapes[0];
        let b_shape_opt = &input_shapes[1];
        
        match (a_shape_opt, b_shape_opt) {
            (Some(a_shape), Some(b_shape)) => {
                // Validate shapes for matmul
                let output_shape = validate_matmul_shapes(a_shape, b_shape)?;
                Ok(vec![Some(output_shape)])
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
                format!("MatMul operator requires 2 inputs, got {}", node.inputs.len())
            ));
        }
        
        if node.outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("MatMul operator requires at least 1 output, got {}", node.outputs.len())
            ));
        }
        
        Ok(())
    }
}

/// Compute the matrix multiplication of two tensors
pub fn compute_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate shapes
    let output_shape = validate_matmul_shapes(&a.shape, &b.shape)?;
    
    // Handle broadcasting for batch dimensions
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    
    if a_rank < 2 || b_rank < 2 {
        return Err(Error::ValidationError(
            format!("MatMul inputs must have at least rank 2, got {} and {}", a_rank, b_rank)
        ));
    }
    
    // Extract the matrix dimensions
    let a_rows = a.shape[a_rank - 2];
    let a_cols = a.shape[a_rank - 1];
    let b_rows = b.shape[b_rank - 2];
    let b_cols = b.shape[b_rank - 1];
    
    if a_cols != b_rows {
        return Err(Error::ValidationError(
            format!("Incompatible matrix dimensions for MatMul: {}x{} and {}x{}", 
                   a_rows, a_cols, b_rows, b_cols)
        ));
    }
    
    // Create result tensor
    let mut result = Tensor::new(&output_shape, a.data_type);
    
    // Handle 2D case (simple matrix multiplication)
    if a_rank == 2 && b_rank == 2 {
        // Use optimized matmul for 2D case
        let a_view = a.data.view();
        let b_view = b.data.view();
        let mut c_view = result.data.view_mut();
        
        optimize_matmul_for_cpu(
            &a_view,
            &b_view,
            &mut c_view
        );
        
        return Ok(result);
    }
    
    // Handle batch matmul with broadcasting
    // This is a simplified implementation that handles common cases
    
    // Broadcast batch dimensions if needed
    let batch_dims_a = &a.shape[0..a_rank-2];
    let batch_dims_b = &b.shape[0..b_rank-2];
    
    // Calculate common batch dimensions
    let batch_dims = if batch_dims_a.len() >= batch_dims_b.len() {
        batch_dims_a
    } else {
        batch_dims_b
    };
    
    // For now, just implement the simple case without full broadcasting
    // A complete implementation would handle more complex broadcasting scenarios
    if batch_dims_a == batch_dims_b {
        // Same batch dimensions - no broadcasting needed
        let batch_size: usize = batch_dims.iter().product();
        
        // Reshape to 3D: [batch_size, rows, cols]
        let a_reshaped = a.data.clone().into_shape((batch_size, a_rows, a_cols)).unwrap();
        let b_reshaped = b.data.clone().into_shape((batch_size, b_rows, b_cols)).unwrap();
        let mut c_reshaped = result.data.clone().into_shape((batch_size, a_rows, b_cols)).unwrap();
        
        // Perform batch matmul
        for i in 0..batch_size {
            let a_mat = a_reshaped.slice(ndarray::s![i, .., ..]);
            let b_mat = b_reshaped.slice(ndarray::s![i, .., ..]);
            let mut c_mat = c_reshaped.slice_mut(ndarray::s![i, .., ..]);
            
            // Use matrix multiplication for this batch
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
        }
        
        // Reshape result back to the correct shape
        result.data = c_reshaped.into_shape(IxDyn(&output_shape)).unwrap();
    } else {
        // For more complex broadcasting, we'd need a more sophisticated implementation
        return Err(Error::UnsupportedFeature(
            "Complex broadcasting in batch MatMul is not implemented yet".to_string()
        ));
    }
    
    Ok(result)
}

/// Validate shapes for matrix multiplication and return the output shape
pub fn validate_matmul_shapes(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    if a_shape.is_empty() || b_shape.is_empty() {
        return Err(Error::ValidationError(
            "MatMul inputs cannot be scalars".to_string()
        ));
    }
    
    // Get the ranks
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();
    
    // Special case for vector inputs
    if a_rank == 1 && b_rank == 1 {
        // Vector dot product: [M] x [M] -> scalar
        if a_shape[0] != b_shape[0] {
            return Err(Error::ValidationError(format!(
                "Incompatible vector dimensions for MatMul: {} and {}", 
                a_shape[0], b_shape[0]
            )));
        }
        return Ok(vec![]);  // Scalar output
    }
    
    // Handle vector * matrix: [M] x [M, N] -> [N]
    if a_rank == 1 {
        if a_shape[0] != b_shape[b_rank - 2] {
            return Err(Error::ValidationError(format!(
                "Incompatible dimensions for vector*matrix MatMul: {} and [{}, {}]", 
                a_shape[0], b_shape[b_rank - 2], b_shape[b_rank - 1]
            )));
        }
        
        let mut output_shape = Vec::new();
        // Prepend batch dimensions
        output_shape.extend_from_slice(&b_shape[0..b_rank-2]);
        // Add final dimension
        output_shape.push(b_shape[b_rank - 1]);
        return Ok(output_shape);
    }
    
    // Handle matrix * vector: [M, N] x [N] -> [M]
    if b_rank == 1 {
        if a_shape[a_rank - 1] != b_shape[0] {
            return Err(Error::ValidationError(format!(
                "Incompatible dimensions for matrix*vector MatMul: [{}, {}] and {}", 
                a_shape[a_rank - 2], a_shape[a_rank - 1], b_shape[0]
            )));
        }
        
        let mut output_shape = Vec::new();
        // Prepend batch dimensions
        output_shape.extend_from_slice(&a_shape[0..a_rank-2]);
        // Add final dimension
        output_shape.push(a_shape[a_rank - 2]);
        return Ok(output_shape);
    }
    
    // Standard matrix multiplication case
    // Check that the contracting dimensions match
    if a_shape[a_rank - 1] != b_shape[b_rank - 2] {
        return Err(Error::ValidationError(format!(
            "Incompatible matrix dimensions for MatMul: {}x{} and {}x{}", 
            a_shape[a_rank - 2], a_shape[a_rank - 1], 
            b_shape[b_rank - 2], b_shape[b_rank - 1]
        )));
    }
    
    // Calculate batch dimensions
    let batch_a = &a_shape[0..a_rank-2];
    let batch_b = &b_shape[0..b_rank-2];
    
    // Calculate the broadcast batch dimensions
    let batch_output = if batch_a.is_empty() && batch_b.is_empty() {
        Vec::new()
    } else {
        // Broadcast batch dimensions
        broadcast_batch_dims(batch_a, batch_b)?
    };
    
    // Construct output shape
    let mut output_shape = batch_output;
    output_shape.push(a_shape[a_rank - 2]);  // Output rows
    output_shape.push(b_shape[b_rank - 1]);  // Output columns
    
    Ok(output_shape)
}

/// Helper function to broadcast batch dimensions
fn broadcast_batch_dims(a_batch: &[usize], b_batch: &[usize]) -> Result<Vec<usize>> {
    // Pad shorter batch with ones
    let max_batch_dims = max(a_batch.len(), b_batch.len());
    let mut padded_a = vec![1; max_batch_dims];
    let mut padded_b = vec![1; max_batch_dims];
    
    // Fill in actual dimensions
    for (i, &dim) in a_batch.iter().rev().enumerate() {
        padded_a[max_batch_dims - 1 - i] = dim;
    }
    
    for (i, &dim) in b_batch.iter().rev().enumerate() {
        padded_b[max_batch_dims - 1 - i] = dim;
    }
    
    // Compute broadcast dimensions
    let mut result = Vec::with_capacity(max_batch_dims);
    for (dim_a, dim_b) in padded_a.iter().zip(padded_b.iter()) {
        if *dim_a == 1 {
            result.push(*dim_b);
        } else if *dim_b == 1 {
            result.push(*dim_a);
        } else if dim_a == dim_b {
            result.push(*dim_a);
        } else {
            return Err(Error::ValidationError(format!(
                "Cannot broadcast batch dimensions {} and {}", dim_a, dim_b
            )));
        }
    }
    
    Ok(result)
}

/// Optimize matrix multiplication for CPU using blocking
pub fn optimize_matmul_for_cpu(
    a: &ArrayViewD<f32>,
    b: &ArrayViewD<f32>,
    c: &mut ArrayViewMutD<f32>,
) {
    // Use ndarray's built-in matrix multiplication for small matrices
    if a.len() < 10000 || b.len() < 10000 {
        // Simple matrix multiplication
        ndarray::linalg::general_mat_mul(1.0, 
            &a.view().into_dimensionality().unwrap(), 
            &b.view().into_dimensionality().unwrap(), 
            0.0, 
            &mut c.view_mut().into_dimensionality().unwrap()
        );
        return;
    }
    
    // For larger matrices, use a blocking strategy
    // Extract dimensions
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    
    // Choose block sizes
    const BLOCK_SIZE: usize = 64;  // Adjust based on cache size
    
    // Blocked matrix multiplication for better cache efficiency
    for i in (0..m).step_by(BLOCK_SIZE) {
        let i_end = std::cmp::min(i + BLOCK_SIZE, m);
        
        for j in (0..n).step_by(BLOCK_SIZE) {
            let j_end = std::cmp::min(j + BLOCK_SIZE, n);
            
            // Initialize the block to zeros
            for ii in i..i_end {
                for jj in j..j_end {
                    c[[ii, jj]] = 0.0;
                }
            }
            
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                
                // Compute on the current blocks
                for ii in i..i_end {
                    for kk in k_block..k_end {
                        let a_val = a[[ii, kk]];
                        
                        for jj in j..j_end {
                            c[[ii, jj]] += a_val * b[[kk, jj]];
                        }
                    }
                }
            }
        }
    }
}

// Import for IxDyn
use ndarray::IxDyn;