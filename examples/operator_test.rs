use onnx_parser::ops::registry::{OperatorRegistry, ExecutionContext};
use onnx_parser::ops::tensor::{Tensor, DataType};
use onnx_parser::ops::math::matmul::MatMul;
use onnx_parser::ops::activations::Relu;
use ndarray::{Array, ArrayD, IxDyn};

fn main() -> onnx_parser::error::Result<()> {
    println!("Testing ONNX operators...");
    
    // Create input tensors for matrix multiplication
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
    let a = Tensor::from_ndarray(a_data, DataType::Float32)?;
    let b = Tensor::from_ndarray(b_data, DataType::Float32)?;
    
    println!("Input A: {:?}", a);
    println!("Input B: {:?}", b);
    
    // Test MatMul
    let matmul_op = MatMul::default();
    let mut c = Tensor::new(&[2, 2], DataType::Float32);
    
    matmul_op.compute(&[&a, &b], &mut [c.clone()], &ExecutionContext::default())?;
    
    println!("Output C (MatMul): {:?}", c);
    println!("C data: {}", c.data);
    
    // Test ReLU
    let relu_op = Relu::default();
    let neg_data = Array::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![-1.0, 2.0, -3.0, 4.0],
    ).unwrap();
    
    let neg_tensor = Tensor::from_ndarray(neg_data, DataType::Float32)?;
    let mut relu_output = Tensor::new(&[2, 2], DataType::Float32);
    
    relu_op.compute(&[&neg_tensor], &mut [relu_output.clone()], &ExecutionContext::default())?;
    
    println!("Input: {:?}", neg_tensor);
    println!("Output (ReLU): {:?}", relu_output);
    println!("ReLU data: {}", relu_output.data);
    
    // Test a simple computation graph: C = ReLU(A @ B)
    let mut relu_on_matmul = Tensor::new(&[2, 2], DataType::Float32);
    
    matmul_op.compute(&[&a, &b], &mut [c.clone()], &ExecutionContext::default())?;
    relu_op.compute(&[&c], &mut [relu_on_matmul.clone()], &ExecutionContext::default())?;
    
    println!("Output (ReLU(MatMul)): {:?}", relu_on_matmul);
    println!("ReLU(MatMul) data: {}", relu_on_matmul.data);
    
    Ok(())
}