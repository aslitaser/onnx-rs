use onnx_parser::ops::tensor::{Tensor, DataType};
use onnx_parser::ops::math::matmul::compute_matmul;
use ndarray::{Array, ArrayD, IxDyn};

fn main() -> onnx_parser::error::Result<()> {
    println!("ONNX Batch MatMul Broadcasting Example");
    println!("=====================================\n");
    
    // Example 1: Simple matrix multiplication
    println!("Example 1: Simple matrix multiplication");
    let a_shape = vec![2, 3];
    let a_data = Array::from_shape_vec(
        IxDyn(&a_shape),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ).unwrap();
    
    let b_shape = vec![3, 2];
    let b_data = Array::from_shape_vec(
        IxDyn(&b_shape),
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).unwrap();
    
    // Create tensors
    let a = Tensor::from_ndarray_simple(a_data, DataType::Float32)?;
    let b = Tensor::from_ndarray_simple(b_data, DataType::Float32)?;
    
    println!("A: shape {:?}", a.shape);
    println!("B: shape {:?}", b.shape);
    
    let result = compute_matmul(&a, &b)?;
    println!("Result: shape {:?}", result.shape);
    println!("Data: {}\n", result.data);
    
    // Example 2: Broadcasting a single matrix against a batch
    println!("Example 2: Broadcasting with batch dimensions");
    let c_shape = vec![2, 2, 3];  // Batch of 2 matrices
    let c_data = Array::from_shape_vec(
        IxDyn(&c_shape),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).unwrap();
    
    let d_shape = vec![3, 2];  // Single matrix
    let d_data = Array::from_shape_vec(
        IxDyn(&d_shape),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ).unwrap();
    
    // Create tensors
    let c = Tensor::from_ndarray_simple(c_data, DataType::Float32)?;
    let d = Tensor::from_ndarray_simple(d_data, DataType::Float32)?;
    
    println!("C: shape {:?}", c.shape);
    println!("D: shape {:?}", d.shape);
    
    let result = compute_matmul(&c, &d)?;
    println!("Result: shape {:?}", result.shape);
    println!("Data: {}\n", result.data);
    
    // Example 3: Complex broadcasting with singleton dimensions
    println!("Example 3: Complex broadcasting with singleton dimensions");
    let e_shape = vec![2, 1, 2, 3];  // Note the singleton dimension
    let e_data = Array::from_shape_vec(
        IxDyn(&e_shape),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).unwrap();
    
    let f_shape = vec![3, 3, 2];
    let f_data = Array::from_shape_vec(
        IxDyn(&f_shape),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
    ).unwrap();
    
    // Create tensors
    let e = Tensor::from_ndarray_simple(e_data, DataType::Float32)?;
    let f = Tensor::from_ndarray_simple(f_data, DataType::Float32)?;
    
    println!("E: shape {:?}", e.shape);
    println!("F: shape {:?}", f.shape);
    
    let result = compute_matmul(&e, &f)?;
    println!("Result: shape {:?}", result.shape);
    println!("Data shape: {:?}\n", result.data.shape());
    
    // Example 4: Vector and matrix multiplication
    println!("Example 4: Vector-matrix multiplication");
    let g_shape = vec![3];  // Vector
    let g_data = Array::from_shape_vec(
        IxDyn(&g_shape),
        vec![1.0, 2.0, 3.0],
    ).unwrap();
    
    let h_shape = vec![3, 4];  // Matrix
    let h_data = Array::from_shape_vec(
        IxDyn(&h_shape),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).unwrap();
    
    // Create tensors
    let g = Tensor::from_ndarray_simple(g_data, DataType::Float32)?;
    let h = Tensor::from_ndarray_simple(h_data, DataType::Float32)?;
    
    println!("G: shape {:?}", g.shape);
    println!("H: shape {:?}", h.shape);
    
    let result = compute_matmul(&g, &h)?;
    println!("Result: shape {:?}", result.shape);
    println!("Data: {}", result.data);
    
    // Example 5: Performance comparison with different batch sizes
    println!("\nExample 5: Performance comparison with different batch sizes");
    
    // Test with small batches
    let small_batch = 5;
    let small_a = create_random_tensor(&[small_batch, 128, 64], DataType::Float32)?;
    let small_b = create_random_tensor(&[small_batch, 64, 256], DataType::Float32)?;
    
    let start = std::time::Instant::now();
    let small_result = compute_matmul(&small_a, &small_b)?;
    let small_duration = start.elapsed();
    
    println!("Small batch ({}) matmul completed in {:?}", small_batch, small_duration);
    println!("Result shape: {:?}", small_result.shape);
    
    // Test with large batches
    let large_batch = 50;
    let large_a = create_random_tensor(&[large_batch, 128, 64], DataType::Float32)?;
    let large_b = create_random_tensor(&[large_batch, 64, 256], DataType::Float32)?;
    
    let start = std::time::Instant::now();
    let large_result = compute_matmul(&large_a, &large_b)?;
    let large_duration = start.elapsed();
    
    println!("Large batch ({}) matmul completed in {:?}", large_batch, large_duration);
    println!("Result shape: {:?}", large_result.shape);
    
    // Compare with broadcasting
    println!("\nBroadcasting performance:");
    let broadcast_a = create_random_tensor(&[1, 128, 64], DataType::Float32)?;
    let broadcast_b = create_random_tensor(&[large_batch, 64, 256], DataType::Float32)?;
    
    let start = std::time::Instant::now();
    let broadcast_result = compute_matmul(&broadcast_a, &broadcast_b)?;
    let broadcast_duration = start.elapsed();
    
    println!("Broadcasting matmul (1 -> {}) completed in {:?}", large_batch, broadcast_duration);
    println!("Result shape: {:?}", broadcast_result.shape);
    
    Ok(())
}

// Helper function to create a tensor filled with random values
fn create_random_tensor(shape: &[usize], data_type: DataType) -> onnx_parser::error::Result<Tensor> {
    let total_elements: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_elements);
    
    // Fill with random values between 0 and 1
    for _ in 0..total_elements {
        data.push(rand::random::<f32>());
    }
    
    let array = Array::from_shape_vec(IxDyn(shape), data).unwrap();
    Tensor::from_ndarray_simple(array, data_type)
}