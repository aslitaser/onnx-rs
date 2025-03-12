use onnx::{
    TensorLayout,
    error::Result,
};

fn main() -> Result<()> {
    println!("Tensor Layout Example");
    println!("-------------------");
    
    // 1. Creating layouts
    println!("\n1. Creating Different Layouts");
    println!("----------------------------");
    
    // Create a contiguous layout for a 3D tensor (batch, channels, height, width)
    let shape = [2, 3, 4, 5];
    let contiguous = TensorLayout::contiguous_layout(&shape);
    println!("Contiguous layout for shape {:?}:", shape);
    println!("  {:?}", contiguous);
    println!("  Is contiguous: {}", contiguous.is_contiguous());
    println!("  Size in bytes (for f32): {}", contiguous.size_in_bytes(4));
    
    // Create a layout with custom strides
    let custom_strides = [60, 20, 5, 1]; // Custom strides
    let custom = TensorLayout::new(&shape, Some(&custom_strides), 0);
    println!("\nCustom layout with strides {:?}:", custom_strides);
    println!("  {:?}", custom);
    println!("  Is contiguous: {}", custom.is_contiguous());
    
    // 2. Transpose a layout
    println!("\n2. Transposing a Layout");
    println!("---------------------");
    
    let transpose_axes = [3, 2, 1, 0]; // NCHW -> WHCN
    let transposed = TensorLayout::transposed_layout(&contiguous, &transpose_axes)?;
    println!("Transposed layout (axes {:?}):", transpose_axes);
    println!("  Original: {:?}", contiguous);
    println!("  Transposed: {:?}", transposed);
    println!("  Is contiguous: {}", transposed.is_contiguous());
    
    // 3. Broadcasting layouts
    println!("\n3. Broadcasting Layouts");
    println!("---------------------");
    
    let small_shape = [1, 3, 1];
    let small_layout = TensorLayout::contiguous_layout(&small_shape);
    let target_shape = [2, 3, 4];
    
    let broadcasted = TensorLayout::broadcasted_layout(&small_layout, &target_shape)?;
    println!("Broadcasting from shape {:?} to {:?}:", small_shape, target_shape);
    println!("  Original: {:?}", small_layout);
    println!("  Broadcasted: {:?}", broadcasted);
    
    // 4. Index calculation
    println!("\n4. Index Calculation");
    println!("------------------");
    
    let indices = [1, 2, 3];
    let linear_index = contiguous.index_of(&indices[0..3]);
    println!("Linear index for {:?} in layout {:?}: {}", indices, contiguous, linear_index);
    
    // Calculate a few more indices
    println!("Computing linear indices for various positions:");
    
    let layout3d = TensorLayout::contiguous_layout(&[3, 4, 5]);
    println!("  3D layout: {:?}", layout3d);
    println!("  Index at [0,0,0]: {}", layout3d.index_of(&[0, 0, 0]));
    println!("  Index at [1,0,0]: {}", layout3d.index_of(&[1, 0, 0]));
    println!("  Index at [0,1,0]: {}", layout3d.index_of(&[0, 1, 0]));
    println!("  Index at [0,0,1]: {}", layout3d.index_of(&[0, 0, 1]));
    println!("  Index at [2,3,4]: {}", layout3d.index_of(&[2, 3, 4]));
    
    // 5. Layout optimization for operations
    println!("\n5. Layout Optimization for Operations");
    println!("----------------------------------");
    
    // Set up layouts for a matrix multiplication
    let a_shape = [64, 128];
    let b_shape = [128, 256];
    let a_layout = TensorLayout::contiguous_layout(&a_shape);
    let b_layout = TensorLayout::contiguous_layout(&b_shape);
    
    println!("Original layouts for MatMul:");
    println!("  A: {:?}", a_layout);
    println!("  B: {:?}", b_layout);
    
    let optimized = TensorLayout::optimize_for_operation(&[&a_layout, &b_layout], "MatMul");
    println!("Optimized layouts for MatMul:");
    println!("  A: {:?}", optimized[0]);
    println!("  B: {:?}", optimized[1]);
    
    // 6. In-place operation analysis
    println!("\n6. In-place Operation Analysis");
    println!("----------------------------");
    
    let input_shape = [32, 64, 128];
    let input_layout = TensorLayout::contiguous_layout(&input_shape);
    let output_layout = TensorLayout::contiguous_layout(&input_shape);
    
    println!("Can use in-place operation?");
    println!("  Input: {:?}", input_layout);
    println!("  Output: {:?}", output_layout);
    println!("  Result: {}", TensorLayout::can_use_inplace(&input_layout, &output_layout));
    
    // Add an offset to the output layout and check again
    let offset_output = output_layout.with_offset(10);
    println!("\nWith offset in output:");
    println!("  Input: {:?}", input_layout);
    println!("  Output with offset: {:?}", offset_output);
    println!("  Result: {}", TensorLayout::can_use_inplace(&input_layout, &offset_output));
    
    // 7. Advanced layout manipulations
    println!("\n7. Advanced Layout Manipulations");
    println!("------------------------------");
    
    let layout = TensorLayout::contiguous_layout(&[2, 3, 4, 5]);
    
    // Keep only certain dimensions
    let keep_axes = [0, 2]; // Keep only batch and height dimensions
    let reduced = layout.keep_dims(&keep_axes)?;
    println!("Keeping only dimensions {:?}:", keep_axes);
    println!("  Original: {:?}", layout);
    println!("  Result: {:?}", reduced);
    
    // 8. Squeeze and unsqueeze
    println!("\n8. Squeeze and Unsqueeze");
    println!("-----------------------");
    
    let shape_with_ones = [2, 1, 3, 1, 4];
    let layout_with_ones = TensorLayout::contiguous_layout(&shape_with_ones);
    
    // Squeeze out the dimensions with size 1
    let squeeze_axes = [1, 3]; // Squeeze out the two dimensions with size 1
    let squeezed = layout_with_ones.squeeze(&squeeze_axes)?;
    println!("Squeezing out dimensions with size 1:");
    println!("  Original: {:?}", layout_with_ones);
    println!("  After squeezing {:?}: {:?}", squeeze_axes, squeezed);
    
    // Unsqueeze to add new dimensions of size 1
    let unsqueeze_axes = [0, 3]; // Add new dimensions at positions 0 and 3
    let unsqueezed = squeezed.unsqueeze(&unsqueeze_axes)?;
    println!("\nUnsqueezing to add new dimensions of size 1:");
    println!("  Original: {:?}", squeezed);
    println!("  After unsqueezing {:?}: {:?}", unsqueeze_axes, unsqueezed);
    
    Ok(())
}