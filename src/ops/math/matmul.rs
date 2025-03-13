// Matrix multiplication operation with complex broadcasting
//
// This module implements the ONNX MatMul operator, including support for:
// - N-dimensional batch matrix multiplication
// - Broadcasting of batch dimensions
// - Vector-matrix and matrix-vector multiplication
// - Vector-vector dot products
//
// Broadcasting Rules:
// ------------------
// Broadcasting in MatMul follows the ONNX broadcasting rules:
//
// 1. Right-alignment: When tensors have different ranks, dimensions are right-aligned.
//    Example: [5, 2, 3] and [3, 2] -> the [3, 2] is aligned to the rightmost dims of [5, 2, 3]
//
// 2. Singleton expansion: Dimensions of size 1 can be broadcast to any size.
//    Example: [5, 1, 3] and [5, 4, 3] -> the middle dim of the first tensor (1) is broadcast to 4
//
// 3. Dimension compatibility: Non-singleton dimensions must match exactly.
//    Example: [5, 6, 3] and [5, 7, 3] -> Error, because 6 != 7
//
// 4. Output shape: For each dimension, the output size is the broadcasted size.
//    Example: [5, 1, 3] and [1, 4, 3] -> output batch shape is [5, 4]
//
// Special Cases:
// -------------
// 1. Vector-Vector: [M] × [M] -> scalar output []
// 2. Matrix-Vector: [M, K] × [K] -> [M]
// 3. Vector-Matrix: [K] × [K, N] -> [N]
// 4. Matrix-Matrix: [M, K] × [K, N] -> [M, N]
// 5. Batch matrix multiplication: 
//    [B1, B2, ..., Bn, M, K] × [B1, B2, ..., Bn, K, N] -> [B1, B2, ..., Bn, M, N]
//
// Performance Optimizations:
// -------------------------
// 1. Direct use of BLAS for 2D matrices and small batches
// 2. Cache-optimized blocking for large matrices
// 3. Parallelization of batch dimensions for large batches
// 4. Avoidance of unnecessary memory allocation during broadcasting
// 5. Efficient broadcasting implementation with index mapping instead of data copying
//
// Error Handling:
// --------------
// The implementation validates shapes according to ONNX rules and returns appropriate
// error messages for incompatible dimensions or other issues.

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
    
    // Special case for 1D inputs
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    
    if a_rank == 1 || b_rank == 1 {
        return compute_vector_matmul(a, b, &output_shape);
    }
    
    // Extract the matrix dimensions for standard case
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
    let batch_dims_a = &a.shape[0..a_rank-2];
    let batch_dims_b = &b.shape[0..b_rank-2];
    
    // Get broadcast information
    let (batch_output_shape, a_needs_broadcast, b_needs_broadcast) = 
        broadcast_batch_dims(batch_dims_a, batch_dims_b)?;
    
    // Compute output size and create result tensor if not already created
    let output_batch_size: usize = batch_output_shape.iter().product();
    
    // If no broadcasting is needed, use the simplified approach
    if batch_dims_a == batch_dims_b {
        // Same batch dimensions - no broadcasting needed
        let batch_size: usize = batch_dims_a.iter().product();
        
        // Reshape to 3D: [batch_size, rows, cols]
        let a_reshaped = a.data.clone().into_shape((batch_size, a_rows, a_cols)).unwrap();
        let b_reshaped = b.data.clone().into_shape((batch_size, b_rows, b_cols)).unwrap();
        let mut c_reshaped = result.data.clone().into_shape((batch_size, a_rows, b_cols)).unwrap();
        
        // Perform batch matmul - use parallel execution for large batches
        if batch_size > 8 {
            use rayon::prelude::*;
            
            (0..batch_size).into_par_iter().for_each(|i| {
                let a_mat = a_reshaped.slice(ndarray::s![i, .., ..]);
                let b_mat = b_reshaped.slice(ndarray::s![i, .., ..]);
                let mut c_view = c_reshaped.slice_mut(ndarray::s![i, .., ..]);
                
                // Use matrix multiplication for this batch
                ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_view);
            });
        } else {
            // Sequential execution for small batches
            for i in 0..batch_size {
                let a_mat = a_reshaped.slice(ndarray::s![i, .., ..]);
                let b_mat = b_reshaped.slice(ndarray::s![i, .., ..]);
                let mut c_mat = c_reshaped.slice_mut(ndarray::s![i, .., ..]);
                
                // Use matrix multiplication for this batch
                ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
            }
        }
        
        // Reshape result back to the correct shape
        result.data = c_reshaped.into_shape(IxDyn(&output_shape)).unwrap();
        
        return Ok(result);
    }
    
    // Handle complex broadcasting case
    // We need to perform broadcasting for each output batch index
    
    // First, we'll create a flat view of our output array
    let mut result_flat = result.data.clone().into_shape((output_batch_size, a_rows, b_cols)).unwrap();
    
    // Calculate strides for each input tensor
    let a_batch_strides = calculate_broadcast_strides(batch_dims_a, &batch_output_shape, &a_needs_broadcast);
    let b_batch_strides = calculate_broadcast_strides(batch_dims_b, &batch_output_shape, &b_needs_broadcast);
    
    // Calculate total batch sizes
    let a_batch_size: usize = batch_dims_a.iter().product();
    let b_batch_size: usize = batch_dims_b.iter().product();
    
    // Reshape input tensors to 3D for easier batch access
    let a_reshaped = a.data.clone().into_shape((a_batch_size, a_rows, a_cols)).unwrap();
    let b_reshaped = b.data.clone().into_shape((b_batch_size, b_rows, b_cols)).unwrap();
    
    // Create index mappings for broadcasting
    let a_indices = create_broadcast_indices(&batch_output_shape, batch_dims_a, &a_needs_broadcast);
    let b_indices = create_broadcast_indices(&batch_output_shape, batch_dims_b, &b_needs_broadcast);
    
    // Process batches - use parallel execution for large output batches
    if output_batch_size > 8 {
        use rayon::prelude::*;
        
        (0..output_batch_size).into_par_iter().for_each(|i| {
            // Map output index to input indices
            let a_idx = a_indices[i];
            let b_idx = b_indices[i];
            
            // Get the corresponding slices
            let a_mat = a_reshaped.slice(ndarray::s![a_idx, .., ..]);
            let b_mat = b_reshaped.slice(ndarray::s![b_idx, .., ..]);
            let mut c_mat = result_flat.slice_mut(ndarray::s![i, .., ..]);
            
            // Compute this batch
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
        });
    } else {
        // Sequential execution for small output batches
        for i in 0..output_batch_size {
            // Map output index to input indices
            let a_idx = a_indices[i];
            let b_idx = b_indices[i];
            
            // Get the corresponding slices
            let a_mat = a_reshaped.slice(ndarray::s![a_idx, .., ..]);
            let b_mat = b_reshaped.slice(ndarray::s![b_idx, .., ..]);
            let mut c_mat = result_flat.slice_mut(ndarray::s![i, .., ..]);
            
            // Compute this batch
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
        }
    }
    
    // Reshape result back to the correct shape
    result.data = result_flat.into_shape(IxDyn(&output_shape)).unwrap();
    
    Ok(result)
}

/// Compute matrix multiplication involving vectors (1D tensors)
fn compute_vector_matmul(a: &Tensor, b: &Tensor, output_shape: &[usize]) -> Result<Tensor> {
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    
    // Create result tensor
    let mut result = Tensor::new(output_shape, a.data_type);
    
    // Vector dot product: [M] x [M] -> scalar
    if a_rank == 1 && b_rank == 1 {
        let a_view = a.data.view();
        let b_view = b.data.view();
        
        // Compute dot product
        let dot_product = (0..a.shape[0])
            .map(|i| a_view[[i]] * b_view[[i]])
            .sum();
        
        // Assign to result (which should be a scalar)
        result.data[[]] = dot_product;
        return Ok(result);
    }
    
    // Vector-matrix multiplication: [M] x [M, N] -> [N]
    if a_rank == 1 {
        let a_view = a.data.view();
        let b_view = b.data.view();
        let mut result_view = result.data.view_mut();
        
        // Handle batch dimensions if present
        if b_rank > 2 {
            // There are batch dimensions in B
            let batch_dims = &b.shape[0..b_rank-2];
            let batch_size: usize = batch_dims.iter().product();
            let b_rows = b.shape[b_rank - 2];
            let b_cols = b.shape[b_rank - 1];
            
            // Reshape to handle batches
            let b_reshaped = b_view.into_shape((batch_size, b_rows, b_cols)).unwrap();
            let mut result_reshaped = result_view.into_shape((batch_size, b_cols)).unwrap();
            
            // Process each batch
            for i in 0..batch_size {
                let b_mat = b_reshaped.slice(ndarray::s![i, .., ..]);
                let mut r_vec = result_reshaped.slice_mut(ndarray::s![i, ..]);
                
                // Compute vector-matrix product
                for j in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..b_rows {
                        sum += a_view[[k]] * b_mat[[k, j]];
                    }
                    r_vec[[j]] = sum;
                }
            }
        } else {
            // Simple case: just one matrix
            for j in 0..b.shape[1] {
                let mut sum = 0.0;
                for k in 0..a.shape[0] {
                    sum += a_view[[k]] * b_view[[k, j]];
                }
                result_view[[j]] = sum;
            }
        }
        
        return Ok(result);
    }
    
    // Matrix-vector multiplication: [M, N] x [N] -> [M]
    if b_rank == 1 {
        let a_view = a.data.view();
        let b_view = b.data.view();
        let mut result_view = result.data.view_mut();
        
        // Handle batch dimensions if present
        if a_rank > 2 {
            // There are batch dimensions in A
            let batch_dims = &a.shape[0..a_rank-2];
            let batch_size: usize = batch_dims.iter().product();
            let a_rows = a.shape[a_rank - 2];
            let a_cols = a.shape[a_rank - 1];
            
            // Reshape to handle batches
            let a_reshaped = a_view.into_shape((batch_size, a_rows, a_cols)).unwrap();
            let mut result_reshaped = result_view.into_shape((batch_size, a_rows)).unwrap();
            
            // Process each batch
            for i in 0..batch_size {
                let a_mat = a_reshaped.slice(ndarray::s![i, .., ..]);
                let mut r_vec = result_reshaped.slice_mut(ndarray::s![i, ..]);
                
                // Compute matrix-vector product
                for j in 0..a_rows {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a_mat[[j, k]] * b_view[[k]];
                    }
                    r_vec[[j]] = sum;
                }
            }
        } else {
            // Simple case: just one matrix
            for i in 0..a.shape[0] {
                let mut sum = 0.0;
                for k in 0..a.shape[1] {
                    sum += a_view[[i, k]] * b_view[[k]];
                }
                result_view[[i]] = sum;
            }
        }
        
        return Ok(result);
    }
    
    // This should not happen since we check in the calling function
    Err(Error::ValidationError("Invalid shapes for vector-matrix multiplication".to_string()))
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
        // Broadcast batch dimensions - we only need the output shape for validation
        let (output_batch, _, _) = broadcast_batch_dims(batch_a, batch_b)?;
        output_batch
    };
    
    // Construct output shape
    let mut output_shape = batch_output;
    output_shape.push(a_shape[a_rank - 2]);  // Output rows
    output_shape.push(b_shape[b_rank - 1]);  // Output columns
    
    Ok(output_shape)
}

/// Helper function to broadcast batch dimensions
/// 
/// This implements the ONNX broadcasting rules for batch dimensions in matmul operations:
/// 1. If a dimension is size 1, it can be broadcast to any size
/// 2. If dimensions match, they remain unchanged
/// 3. Right-alignment is used for tensors with different numbers of dimensions
/// 
/// Returns the output batch shape and information needed for efficient broadcasting
fn broadcast_batch_dims(a_batch: &[usize], b_batch: &[usize]) -> Result<(Vec<usize>, Vec<bool>, Vec<bool>)> {
    // Pad shorter batch with ones
    let max_batch_dims = max(a_batch.len(), b_batch.len());
    let mut padded_a = vec![1; max_batch_dims];
    let mut padded_b = vec![1; max_batch_dims];
    
    // Fill in actual dimensions with right-alignment
    for (i, &dim) in a_batch.iter().rev().enumerate() {
        padded_a[max_batch_dims - 1 - i] = dim;
    }
    
    for (i, &dim) in b_batch.iter().rev().enumerate() {
        padded_b[max_batch_dims - 1 - i] = dim;
    }
    
    // Compute broadcast dimensions
    let mut result = Vec::with_capacity(max_batch_dims);
    
    // Keep track of which dimensions need broadcasting
    let mut a_needs_broadcast = vec![false; max_batch_dims];
    let mut b_needs_broadcast = vec![false; max_batch_dims];
    
    for i in 0..max_batch_dims {
        let dim_a = padded_a[i];
        let dim_b = padded_b[i];
        
        if dim_a == 1 && dim_b == 1 {
            // Both are 1, output is 1
            result.push(1);
        } else if dim_a == 1 {
            // A needs broadcasting
            result.push(dim_b);
            a_needs_broadcast[i] = true;
        } else if dim_b == 1 {
            // B needs broadcasting
            result.push(dim_a);
            b_needs_broadcast[i] = true;
        } else if dim_a == dim_b {
            // Dimensions match
            result.push(dim_a);
        } else {
            // Incompatible dimensions
            return Err(Error::ValidationError(format!(
                "Cannot broadcast batch dimensions {} and {}", dim_a, dim_b
            )));
        }
    }
    
    Ok((result, a_needs_broadcast, b_needs_broadcast))
}

/// Calculate the strides for broadcasting
/// 
/// Given the input dimensions and the output dimensions, calculate
/// the strides needed for efficient broadcasting without copying data
fn calculate_broadcast_strides(
    input_dims: &[usize], 
    output_dims: &[usize], 
    needs_broadcast: &[bool]
) -> Vec<usize> {
    let mut strides = Vec::with_capacity(output_dims.len());
    let mut current_stride = 1;
    
    // Calculate strides from right to left (row-major order)
    for i in (0..output_dims.len()).rev() {
        if i >= output_dims.len() - input_dims.len() {
            // This dimension exists in the input
            let input_idx = i - (output_dims.len() - input_dims.len());
            if needs_broadcast[i] {
                // This dimension is broadcast (size 1)
                strides.push(0); // Stride of 0 for broadcast dimensions
            } else {
                strides.push(current_stride);
                current_stride *= input_dims[input_idx];
            }
        } else {
            // This dimension doesn't exist in the input
            strides.push(0); // Stride of 0 for broadcast dimensions
        }
    }
    
    // Reverse to match the order of dimensions
    strides.reverse();
    strides
}

/// Create mapping from output indices to input indices for broadcasting
///
/// Given the output shape and input information, creates a mapping from
/// flattened output batch indices to flattened input batch indices.
fn create_broadcast_indices(
    output_shape: &[usize],
    input_shape: &[usize],
    needs_broadcast: &[bool]
) -> Vec<usize> {
    // Calculate total number of output elements
    let output_size: usize = output_shape.iter().product();
    
    // Prepare the output index mapping
    let mut indices = vec![0; output_size];
    
    // For each output index, calculate corresponding input index
    if output_size == 0 {
        return indices; // Empty case
    }
    
    // Calculate input size
    let input_size: usize = input_shape.iter().product();
    
    // Handle different batch dimension counts
    let right_align_offset = output_shape.len() - input_shape.len();
    
    // For each output element
    for out_idx in 0..output_size {
        // Convert flat index to multidimensional coordinates
        let mut remaining = out_idx;
        let mut in_idx = 0;
        let mut in_stride = 1;
        
        // Process dimensions from right to left
        for (i, &dim) in output_shape.iter().enumerate().rev() {
            if dim == 0 {
                continue; // Skip empty dimensions
            }
            
            let coord = remaining % dim;
            remaining /= dim;
            
            // Only process dimensions that exist in the input shape
            if i >= right_align_offset {
                let input_dim_idx = i - right_align_offset;
                
                // Use coordinate only if this dimension isn't broadcast in input
                if !needs_broadcast[i] {
                    in_idx += coord * in_stride;
                    in_stride *= input_shape[input_dim_idx];
                }
                // For broadcast dimensions (size 1), we always use index 0
            }
        }
        
        // Store the mapped index, ensuring it's within bounds
        indices[out_idx] = in_idx % input_size;
    }
    
    indices
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis, Dim};
    
    /// Helper function to create a tensor with specified shape and sequential values
    fn create_test_tensor(shape: &[usize], data_type: DataType) -> Tensor {
        let total_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);
        
        // Fill with sequential values for easy verification
        for i in 0..total_elements {
            data.push(i as f32);
        }
        
        let array = Array::from_shape_vec(IxDyn(shape), data).unwrap();
        Tensor::from_ndarray_simple(array, data_type).unwrap()
    }
    
    #[test]
    fn test_simple_matmul() {
        // Simple 2D matrix multiplication
        let a = create_test_tensor(&[2, 3], DataType::Float32);
        let b = create_test_tensor(&[3, 2], DataType::Float32);
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Expected result:
        // [0, 1, 2] * [0, 1] = [5, 14]
        // [3, 4, 5] * [2, 3] = [14, 50]
        //           * [4, 5]
        assert_eq!(result.shape, vec![2, 2]);
        
        let result_view = result.data.view();
        assert_eq!(result_view[[0, 0]], 5.0);
        assert_eq!(result_view[[0, 1]], 14.0);
        assert_eq!(result_view[[1, 0]], 14.0);
        assert_eq!(result_view[[1, 1]], 50.0);
    }
    
    #[test]
    fn test_vector_vector_matmul() {
        // Vector dot product
        let a = create_test_tensor(&[3], DataType::Float32);
        let b = create_test_tensor(&[3], DataType::Float32);
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Expected: 0*0 + 1*1 + 2*2 = 0 + 1 + 4 = 5
        assert_eq!(result.shape, Vec::<usize>::new()); // Scalar output
        assert_eq!(result.data[[]], 5.0);
    }
    
    #[test]
    fn test_vector_matrix_matmul() {
        // Vector-matrix multiplication
        let a = create_test_tensor(&[2], DataType::Float32);
        let b = create_test_tensor(&[2, 3], DataType::Float32);
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Expected:
        // [0, 1] * [0, 1, 2] = [1, 3, 5]
        //          [3, 4, 5]
        assert_eq!(result.shape, vec![3]);
        
        let result_view = result.data.view();
        assert_eq!(result_view[[0]], 1.0);
        assert_eq!(result_view[[1]], 3.0);
        assert_eq!(result_view[[2]], 5.0);
    }
    
    #[test]
    fn test_matrix_vector_matmul() {
        // Matrix-vector multiplication
        let a = create_test_tensor(&[2, 3], DataType::Float32);
        let b = create_test_tensor(&[3], DataType::Float32);
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Expected:
        // [0, 1, 2] * [0] = [5]
        // [3, 4, 5] * [1] = [14]
        //            [2]
        assert_eq!(result.shape, vec![2]);
        
        let result_view = result.data.view();
        assert_eq!(result_view[[0]], 5.0);
        assert_eq!(result_view[[1]], 14.0);
    }
    
    #[test]
    fn test_batch_matmul_no_broadcast() {
        // Batch matrix multiplication without broadcasting
        let a = create_test_tensor(&[2, 2, 3], DataType::Float32);
        let b = create_test_tensor(&[2, 3, 2], DataType::Float32);
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Check shape
        assert_eq!(result.shape, vec![2, 2, 2]);
        
        // Verify a few values (full verification would be verbose)
        let result_view = result.data.view();
        
        // First batch, first row
        // [0, 1, 2] * [0, 1] = [5, 14]
        //           * [2, 3]
        //           * [4, 5]
        assert_eq!(result_view[[0, 0, 0]], 5.0);
        assert_eq!(result_view[[0, 0, 1]], 14.0);
        
        // Second batch values
        assert_eq!(result_view[[1, 0, 0]], 149.0);
        assert_eq!(result_view[[1, 1, 1]], 446.0);
    }
    
    #[test]
    fn test_batch_matmul_with_broadcast() {
        // Test broadcasting a single matrix against a batch
        let a = create_test_tensor(&[2, 3], DataType::Float32);  // Single matrix
        let b = create_test_tensor(&[2, 3, 2], DataType::Float32);  // Batch of matrices
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Check shape (2 batch dimension from b)
        assert_eq!(result.shape, vec![2, 2, 2]);
        
        // Verify that the single matrix a was broadcasted to both batches
        let result_view = result.data.view();
        
        // First batch
        assert_eq!(result_view[[0, 0, 0]], 5.0);
        assert_eq!(result_view[[0, 0, 1]], 14.0);
        assert_eq!(result_view[[0, 1, 0]], 14.0);
        assert_eq!(result_view[[0, 1, 1]], 50.0);
        
        // Second batch (should use same matrix from a but different b)
        assert_eq!(result_view[[1, 0, 0]], 41.0);
        assert_eq!(result_view[[1, 0, 1]], 50.0);
        assert_eq!(result_view[[1, 1, 0]], 122.0);
        assert_eq!(result_view[[1, 1, 1]], 158.0);
    }
    
    #[test]
    fn test_complex_batch_broadcasting() {
        // Test complex broadcasting with different batch shapes
        let a = create_test_tensor(&[2, 1, 2, 3], DataType::Float32);  // [2, 1, 2, 3]
        let b = create_test_tensor(&[3, 3, 2], DataType::Float32);     // [3, 3, 2]
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Output should be [2, 3, 2, 2] after broadcasting
        assert_eq!(result.shape, vec![2, 3, 2, 2]);
        
        // Spot-check a few values (comprehensive verification would be verbose)
        let result_view = result.data.view();
        
        // Check the first batch
        assert_eq!(result_view[[0, 0, 0, 0]], 5.0);
        assert_eq!(result_view[[0, 1, 0, 0]], 41.0);
        
        // Check the second batch
        assert_eq!(result_view[[1, 0, 0, 0]], 5.0);
        assert_eq!(result_view[[1, 2, 1, 1]], 446.0);
    }
    
    #[test]
    fn test_singleton_dimension_broadcasting() {
        // Test broadcasting with singleton dimensions
        let a = create_test_tensor(&[2, 1, 2, 3], DataType::Float32);  // [2, 1, 2, 3]
        let b = create_test_tensor(&[1, 3, 2], DataType::Float32);     // [1, 3, 2]
        
        let result = compute_matmul(&a, &b).unwrap();
        
        // Output should be [2, 1, 2, 2] after broadcasting
        assert_eq!(result.shape, vec![2, 1, 2, 2]);
        
        // Spot-check values
        let result_view = result.data.view();
        assert_eq!(result_view[[0, 0, 0, 0]], 5.0);
        assert_eq!(result_view[[1, 0, 1, 1]], 50.0);
    }
    
    #[test]
    fn test_validate_matmul_shapes() {
        // Test shape validation with various shapes
        
        // Simple 2D case
        let result = validate_matmul_shapes(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(result, vec![2, 4]);
        
        // Vector dot product
        let result = validate_matmul_shapes(&[3], &[3]).unwrap();
        assert_eq!(result, vec![]);
        
        // Vector-matrix
        let result = validate_matmul_shapes(&[3], &[3, 2]).unwrap();
        assert_eq!(result, vec![2]);
        
        // Matrix-vector
        let result = validate_matmul_shapes(&[2, 3], &[3]).unwrap();
        assert_eq!(result, vec![2]);
        
        // Batch with no broadcast
        let result = validate_matmul_shapes(&[2, 3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(result, vec![2, 3, 5]);
        
        // Batch with broadcast (singleton dimension)
        let result = validate_matmul_shapes(&[2, 1, 3, 4], &[2, 5, 4, 5]).unwrap();
        assert_eq!(result, vec![2, 5, 3, 5]);
        
        // Different batch ranks
        let result = validate_matmul_shapes(&[3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(result, vec![2, 3, 5]);
        
        // Empty batch
        let result = validate_matmul_shapes(&[3, 4], &[4, 5]).unwrap();
        assert_eq!(result, vec![3, 5]);
    }
    
    #[test]
    fn test_invalid_matmul_shapes() {
        // Test invalid shapes
        
        // Incompatible matrix dimensions
        let result = validate_matmul_shapes(&[2, 3], &[4, 5]);
        assert!(result.is_err());
        
        // Incompatible vector dot product
        let result = validate_matmul_shapes(&[2], &[3]);
        assert!(result.is_err());
        
        // Incompatible vector-matrix
        let result = validate_matmul_shapes(&[2], &[3, 4]);
        assert!(result.is_err());
        
        // Incompatible batch dimensions
        let result = validate_matmul_shapes(&[2, 3, 4], &[3, 4, 5]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_broadcast_performance() {
        // Test performance of large batch broadcasting
        // This is more of a benchmark than a functional test
        
        // Create larger tensors
        let a = create_test_tensor(&[5, 10, 20], DataType::Float32);
        let b = create_test_tensor(&[10, 20, 30], DataType::Float32);
        
        // Time the operation
        let start = std::time::Instant::now();
        let result = compute_matmul(&a, &b).unwrap();
        let duration = start.elapsed();
        
        // Just verify shape and that it completes without error
        assert_eq!(result.shape, vec![10, 5, 30]);
        
        println!("Broadcast matmul with shapes [{:?}] * [{:?}] completed in {:?}", 
                 a.shape, b.shape, duration);
    }
}