use crate::error::{Error, Result};
use crate::model::{Node, Attribute};
use crate::ops::registry::{Operator, ExecutionContext};
use crate::ops::tensor::{Tensor, Shape};
use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn};
use std::cmp::{max, min};

/// Pooling type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolType {
    Max,
    Average,
    GlobalMax,
    GlobalAverage,
}

/// Base struct for pooling operators
#[derive(Debug, Clone)]
pub struct PoolingBase {
    pool_type: PoolType,
}

/// MaxPool operator
#[derive(Debug, Clone, Default)]
pub struct MaxPool;

/// AveragePool operator
#[derive(Debug, Clone, Default)]
pub struct AveragePool;

/// GlobalMaxPool operator
#[derive(Debug, Clone, Default)]
pub struct GlobalMaxPool;

/// GlobalAveragePool operator
#[derive(Debug, Clone, Default)]
pub struct GlobalAveragePool;

// Implement validation and shape inference for pooling operators
impl PoolingBase {
    fn new(pool_type: PoolType) -> Self {
        Self { pool_type }
    }
    
    fn validate_impl(&self, node: &Node) -> Result<()> {
        if node.inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Pool operator requires at least 1 input, got {}", node.inputs.len())
            ));
        }
        
        if node.outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Pool operator requires at least 1 output, got {}", node.outputs.len())
            ));
        }
        
        // Check attributes for non-global pooling
        if self.pool_type == PoolType::Max || self.pool_type == PoolType::Average {
            for (name, attr) in &node.attributes {
                match name.as_str() {
                    "kernel_shape" | "strides" | "pads" | "dilations" => {
                        if !matches!(attr, Attribute::Ints(_)) {
                            return Err(Error::ValidationError(
                                format!("Pool attribute {} must be an array of ints", name)
                            ));
                        }
                    },
                    "auto_pad" => {
                        if !matches!(attr, Attribute::String(_)) {
                            return Err(Error::ValidationError(
                                format!("Pool attribute {} must be a string", name)
                            ));
                        }
                    },
                    "count_include_pad" => {
                        if !matches!(attr, Attribute::Int(_)) {
                            return Err(Error::ValidationError(
                                format!("Pool attribute {} must be an int", name)
                            ));
                        }
                    },
                    _ => {
                        // Ignore unknown attributes for now
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn output_shapes_impl(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        if input_shapes.is_empty() || input_shapes[0].is_none() {
            return Ok(vec![None]);
        }
        
        let input_shape = input_shapes[0].as_ref().unwrap();
        
        // Check input shape
        if input_shape.len() < 3 {
            return Err(Error::ValidationError(
                format!("Pool input must have at least 3 dimensions, got {}", input_shape.len())
            ));
        }
        
        // For global pooling, the output shape is the input shape with spatial dimensions = 1
        if self.pool_type == PoolType::GlobalMax || self.pool_type == PoolType::GlobalAverage {
            let mut output_shape = input_shape.clone();
            for i in 2..output_shape.len() {
                output_shape[i] = 1;
            }
            return Ok(vec![Some(output_shape)]);
        }
        
        // For regular pooling, calculate output shape based on kernel, stride, padding
        // In a complete implementation, extract these from node.attributes
        let kernel_shape = vec![2, 2];  // Default 2x2 kernel
        let strides = vec![2, 2];       // Default stride of 2
        let pads = vec![0, 0, 0, 0];    // Default no padding
        let auto_pad = "NOTSET";        // Default padding mode
        
        // Batch size and channels (remain unchanged)
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        
        // Calculate output spatial dimensions
        let mut output_shape = vec![batch_size, channels];
        let spatial_dims = input_shape.len() - 2;
        
        for i in 0..spatial_dims {
            let input_size = input_shape[i + 2];
            let kernel_size = if i < kernel_shape.len() { kernel_shape[i] } else { 1 };
            let stride = if i < strides.len() { strides[i] } else { 1 };
            
            let pad_head = if i < pads.len() { pads[i] } else { 0 };
            let pad_tail = if i + spatial_dims < pads.len() { pads[i + spatial_dims] } else { pad_head };
            
            // Handle different padding modes
            let output_size = match auto_pad {
                "NOTSET" => {
                    // Custom padding
                    (input_size + pad_head + pad_tail - kernel_size) / stride + 1
                },
                "SAME_UPPER" | "SAME_LOWER" => {
                    // Padding to maintain input size / stride
                    (input_size + stride - 1) / stride
                },
                "VALID" => {
                    // No padding
                    (input_size - kernel_size) / stride + 1
                },
                _ => {
                    return Err(Error::ValidationError(
                        format!("Unknown auto_pad mode: {}", auto_pad)
                    ));
                }
            };
            
            output_shape.push(output_size);
        }
        
        Ok(vec![Some(output_shape)])
    }
}

// Implement the Operator trait for MaxPool
impl Operator for MaxPool {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("MaxPool requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("MaxPool requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let x = inputs[0];
        
        // Default attributes according to ONNX spec
        let kernel_shape = vec![2, 2];  // Default 2x2 kernel
        let strides = vec![2, 2];       // Default stride of 2
        let padding = vec![0, 0, 0, 0]; // Default no padding
        
        let result = max_pool(x, &kernel_shape, &strides, &padding)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        PoolingBase::new(PoolType::Max).output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        PoolingBase::new(PoolType::Max).validate_impl(node)
    }
}

// Implement the Operator trait for AveragePool
impl Operator for AveragePool {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("AveragePool requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("AveragePool requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let x = inputs[0];
        
        // Default attributes according to ONNX spec
        let kernel_shape = vec![2, 2];  // Default 2x2 kernel
        let strides = vec![2, 2];       // Default stride of 2
        let padding = vec![0, 0, 0, 0]; // Default no padding
        let count_include_pad = false;  // Default exclude padding from averaging
        
        let result = average_pool(x, &kernel_shape, &strides, &padding, count_include_pad)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        PoolingBase::new(PoolType::Average).output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        PoolingBase::new(PoolType::Average).validate_impl(node)
    }
}

// Implement the Operator trait for GlobalMaxPool
impl Operator for GlobalMaxPool {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("GlobalMaxPool requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("GlobalMaxPool requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let x = inputs[0];
        
        let result = global_pool(x, PoolType::GlobalMax)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        PoolingBase::new(PoolType::GlobalMax).output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        PoolingBase::new(PoolType::GlobalMax).validate_impl(node)
    }
}

// Implement the Operator trait for GlobalAveragePool
impl Operator for GlobalAveragePool {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("GlobalAveragePool requires at least 1 input, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("GlobalAveragePool requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        let x = inputs[0];
        
        let result = global_pool(x, PoolType::GlobalAverage)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        PoolingBase::new(PoolType::GlobalAverage).output_shapes_impl(node, input_shapes)
    }
    
    fn validate(&self, node: &Node) -> Result<()> {
        PoolingBase::new(PoolType::GlobalAverage).validate_impl(node)
    }
}

/// Max pooling implementation
pub fn max_pool(
    x: &Tensor,
    kernel_shape: &[usize],
    strides: &[usize],
    padding: &[usize]
) -> Result<Tensor> {
    // Validate input shape
    if x.shape.len() < 3 {
        return Err(Error::ValidationError(
            format!("Pool input must have at least 3 dimensions, got {}", x.shape.len())
        ));
    }
    
    // This implementation focuses on 2D pooling (NCHW format)
    // For a complete implementation, we'd handle 1D and 3D as well
    
    // Extract dimensions
    let batch_size = x.shape[0];
    let channels = x.shape[1];
    let input_height = x.shape[2];
    let input_width = if x.shape.len() > 3 { x.shape[3] } else { 1 };
    
    // Extract kernel size, stride, and padding
    let kernel_h = kernel_shape.get(0).copied().unwrap_or(1);
    let kernel_w = kernel_shape.get(1).copied().unwrap_or(1);
    
    let stride_h = strides.get(0).copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(1);
    
    let pad_top = padding.get(0).copied().unwrap_or(0);
    let pad_left = padding.get(1).copied().unwrap_or(0);
    let pad_bottom = padding.get(2).copied().unwrap_or(pad_top);
    let pad_right = padding.get(3).copied().unwrap_or(pad_left);
    
    // Calculate output dimensions
    let output_height = (input_height + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    let output_width = (input_width + pad_left + pad_right - kernel_w) / stride_w + 1;
    
    // Create output tensor
    let output_shape = if x.shape.len() > 3 {
        vec![batch_size, channels, output_height, output_width]
    } else {
        vec![batch_size, channels, output_height]
    };
    
    let mut result = Tensor::new(&output_shape, x.data_type);
    
    // Perform max pooling
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Initialize with the minimum possible value
                    let mut max_val = std::f32::MIN;
                    
                    // Scan the kernel window
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh as isize * stride_h as isize - pad_top as isize + kh as isize;
                            let iw = ow as isize * stride_w as isize - pad_left as isize + kw as isize;
                            
                            // Skip if input position is outside (implicit zero padding)
                            if ih >= 0 && ih < input_height as isize && iw >= 0 && iw < input_width as isize {
                                let input_val = if x.shape.len() > 3 {
                                    x.data[[b, c, ih as usize, iw as usize]]
                                } else {
                                    x.data[[b, c, ih as usize]]
                                };
                                
                                max_val = max_val.max(input_val);
                            }
                        }
                    }
                    
                    // Set output value
                    if result.shape.len() > 3 {
                        result.data[[b, c, oh, ow]] = max_val;
                    } else {
                        result.data[[b, c, oh]] = max_val;
                    }
                }
            }
        }
    }
    
    Ok(result)
}

/// Average pooling implementation
pub fn average_pool(
    x: &Tensor,
    kernel_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    count_include_pad: bool
) -> Result<Tensor> {
    // Validate input shape
    if x.shape.len() < 3 {
        return Err(Error::ValidationError(
            format!("Pool input must have at least 3 dimensions, got {}", x.shape.len())
        ));
    }
    
    // This implementation focuses on 2D pooling (NCHW format)
    // For a complete implementation, we'd handle 1D and 3D as well
    
    // Extract dimensions
    let batch_size = x.shape[0];
    let channels = x.shape[1];
    let input_height = x.shape[2];
    let input_width = if x.shape.len() > 3 { x.shape[3] } else { 1 };
    
    // Extract kernel size, stride, and padding
    let kernel_h = kernel_shape.get(0).copied().unwrap_or(1);
    let kernel_w = kernel_shape.get(1).copied().unwrap_or(1);
    
    let stride_h = strides.get(0).copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(1);
    
    let pad_top = padding.get(0).copied().unwrap_or(0);
    let pad_left = padding.get(1).copied().unwrap_or(0);
    let pad_bottom = padding.get(2).copied().unwrap_or(pad_top);
    let pad_right = padding.get(3).copied().unwrap_or(pad_left);
    
    // Calculate output dimensions
    let output_height = (input_height + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    let output_width = (input_width + pad_left + pad_right - kernel_w) / stride_w + 1;
    
    // Create output tensor
    let output_shape = if x.shape.len() > 3 {
        vec![batch_size, channels, output_height, output_width]
    } else {
        vec![batch_size, channels, output_height]
    };
    
    let mut result = Tensor::new(&output_shape, x.data_type);
    
    // Perform average pooling
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Initialize sum and count
                    let mut sum = 0.0;
                    let mut count = 0;
                    
                    // Scan the kernel window
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh as isize * stride_h as isize - pad_top as isize + kh as isize;
                            let iw = ow as isize * stride_w as isize - pad_left as isize + kw as isize;
                            
                            if ih >= 0 && ih < input_height as isize && iw >= 0 && iw < input_width as isize {
                                // Valid input position
                                let input_val = if x.shape.len() > 3 {
                                    x.data[[b, c, ih as usize, iw as usize]]
                                } else {
                                    x.data[[b, c, ih as usize]]
                                };
                                
                                sum += input_val;
                                count += 1;
                            } else if count_include_pad {
                                // Include padding in count (with value 0)
                                count += 1;
                            }
                        }
                    }
                    
                    // Set output value (avoid division by zero)
                    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                    
                    if result.shape.len() > 3 {
                        result.data[[b, c, oh, ow]] = avg;
                    } else {
                        result.data[[b, c, oh]] = avg;
                    }
                }
            }
        }
    }
    
    Ok(result)
}

/// Global pooling implementation (max or average)
pub fn global_pool(
    x: &Tensor,
    pool_type: PoolType
) -> Result<Tensor> {
    // Validate input shape
    if x.shape.len() < 3 {
        return Err(Error::ValidationError(
            format!("Global pool input must have at least 3 dimensions, got {}", x.shape.len())
        ));
    }
    
    // Extract dimensions
    let batch_size = x.shape[0];
    let channels = x.shape[1];
    
    // Create output tensor with spatial dimensions = 1
    let mut output_shape = x.shape.clone();
    for i in 2..output_shape.len() {
        output_shape[i] = 1;
    }
    
    let mut result = Tensor::new(&output_shape, x.data_type);
    
    // Use a generic approach for different input shapes
    for b in 0..batch_size {
        for c in 0..channels {
            // Initialize values
            let mut max_val = std::f32::MIN;
            let mut sum = 0.0;
            let mut count = 0;
            
            // Create a view for this batch and channel
            // The shape will be [D1, D2, ..., Dn] for spatial dimensions
            let mut indices = vec![b, c];
            for _ in 2..x.shape.len() {
                indices.push(0);
            }
            
            // Recursive scan over all spatial dimensions
            scan_spatial_dimensions(
                &x.data, &mut indices, 2,
                &mut max_val, &mut sum, &mut count,
                pool_type
            );
            
            // Set output value
            let output_val = match pool_type {
                PoolType::GlobalMax => max_val,
                PoolType::GlobalAverage => if count > 0 { sum / count as f32 } else { 0.0 },
                _ => return Err(Error::ValidationError("Unsupported pool type".to_string())),
            };
            
            // Create indices for the result
            let mut result_indices = vec![b, c];
            for _ in 2..result.shape.len() {
                result_indices.push(0);
            }
            
            // Set the result
            result.data[IxDyn(&result_indices)] = output_val;
        }
    }
    
    Ok(result)
}

/// Recursive function to scan all spatial dimensions
fn scan_spatial_dimensions(
    data: &ArrayD<f32>,
    indices: &mut Vec<usize>,
    dim: usize,
    max_val: &mut f32,
    sum: &mut f32,
    count: &mut usize,
    pool_type: PoolType
) {
    if dim >= indices.len() {
        // We've reached a leaf - process this value
        let val = data[IxDyn(indices)];
        *sum += val;
        *count += 1;
        
        if pool_type == PoolType::GlobalMax || pool_type == PoolType::Max {
            *max_val = max_val.max(val);
        }
        
        return;
    }
    
    // Recursively scan this dimension
    let dim_size = data.shape()[dim];
    for i in 0..dim_size {
        indices[dim] = i;
        scan_spatial_dimensions(data, indices, dim + 1, max_val, sum, count, pool_type);
    }
}