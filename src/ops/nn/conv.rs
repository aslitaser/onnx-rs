use crate::error::{Error, Result};
use crate::model::{Node, Attribute};
use crate::ops::registry::{Operator, ExecutionContext};
use crate::ops::tensor::{Tensor, Shape};
use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn};

/// Convolution operator
#[derive(Debug, Clone, Default)]
pub struct Conv {
    // Configuration options can be added here
}

impl Operator for Conv {
    fn compute(&self, inputs: &[&Tensor], outputs: &mut [Tensor], _context: &ExecutionContext) -> Result<()> {
        if inputs.len() < 2 {
            return Err(Error::ValidationError(
                format!("Conv requires at least 2 inputs, got {}", inputs.len())
            ));
        }
        
        if outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Conv requires at least 1 output, got {}", outputs.len())
            ));
        }
        
        // Input tensor
        let x = inputs[0];
        // Filter/kernel tensor
        let w = inputs[1];
        // Optional bias tensor
        let b = if inputs.len() > 2 { Some(inputs[2]) } else { None };
        
        // Default attributes according to ONNX spec
        let stride = vec![1, 1];
        let padding = vec![0, 0, 0, 0];  // [pad_top, pad_left, pad_bottom, pad_right]
        let dilation = vec![1, 1];
        let groups = 1;
        
        let result = compute_conv(x, w, b, &stride, &padding, &dilation, groups)?;
        outputs[0] = result;
        
        Ok(())
    }
    
    fn output_shapes(&self, node: &Node, input_shapes: &[Option<Shape>]) -> Result<Vec<Option<Shape>>> {
        if input_shapes.len() < 2 {
            return Err(Error::ValidationError(
                format!("Conv requires at least 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        // Extract attributes
        // In a complete implementation, we would extract these from node.attributes
        let dilations = vec![1, 1];
        let strides = vec![1, 1];
        let pads = vec![0, 0, 0, 0];  // [pad_top, pad_left, pad_bottom, pad_right]
        let auto_pad = "NOTSET";
        let groups = 1;
        
        match (&input_shapes[0], &input_shapes[1]) {
            (Some(x_shape), Some(w_shape)) => {
                // Validate input shapes
                if x_shape.len() < 3 {
                    return Err(Error::ValidationError(
                        format!("Conv input must be at least 3D, got shape {:?}", x_shape)
                    ));
                }
                
                if w_shape.len() != x_shape.len() {
                    return Err(Error::ValidationError(
                        format!("Conv kernel must have same rank as input, got shapes {:?} and {:?}", x_shape, w_shape)
                    ));
                }
                
                // Handle 1D, 2D, and 3D convolutions
                let spatial_dims = x_shape.len() - 2;
                if spatial_dims > 3 {
                    return Err(Error::ValidationError(
                        format!("Conv supports only 1D, 2D, and 3D convolutions, got {} spatial dimensions", spatial_dims)
                    ));
                }
                
                // Batch size and output channel
                let batch_size = x_shape[0];
                let out_channels = w_shape[0];
                
                // Calculate output spatial dimensions
                let mut output_shape = vec![batch_size, out_channels];
                
                for i in 0..spatial_dims {
                    let input_size = x_shape[i + 2];
                    let kernel_size = w_shape[i + 2];
                    let stride = if i < strides.len() { strides[i] } else { 1 };
                    let dilation = if i < dilations.len() { dilations[i] } else { 1 };
                    
                    let pad_head = if i < pads.len() { pads[i] } else { 0 };
                    let pad_tail = if i + spatial_dims < pads.len() { pads[i + spatial_dims] } else { pad_head };
                    
                    // Handle different padding modes
                    let output_size = match auto_pad {
                        "NOTSET" => {
                            // Custom padding
                            let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
                            let output = (input_size + pad_head + pad_tail - dilated_kernel_size) / stride + 1;
                            output
                        },
                        "SAME_UPPER" | "SAME_LOWER" => {
                            // Padding to maintain input size / stride
                            (input_size + stride - 1) / stride
                        },
                        "VALID" => {
                            // No padding
                            let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
                            let output = (input_size - dilated_kernel_size) / stride + 1;
                            output
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
                format!("Conv operator requires at least 2 inputs, got {}", node.inputs.len())
            ));
        }
        
        if node.outputs.len() < 1 {
            return Err(Error::ValidationError(
                format!("Conv operator requires at least 1 output, got {}", node.outputs.len())
            ));
        }
        
        // Check attributes
        for (name, attr) in &node.attributes {
            match name.as_str() {
                "dilations" | "strides" | "pads" | "kernel_shape" => {
                    if !matches!(attr, Attribute::Ints(_)) {
                        return Err(Error::ValidationError(
                            format!("Conv attribute {} must be an array of ints", name)
                        ));
                    }
                },
                "group" => {
                    if !matches!(attr, Attribute::Int(_)) {
                        return Err(Error::ValidationError(
                            format!("Conv attribute {} must be an int", name)
                        ));
                    }
                },
                "auto_pad" => {
                    if !matches!(attr, Attribute::String(_)) {
                        return Err(Error::ValidationError(
                            format!("Conv attribute {} must be a string", name)
                        ));
                    }
                },
                _ => {
                    // Ignore unknown attributes for now
                }
            }
        }
        
        Ok(())
    }
}

/// Strategy for convolution implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvStrategy {
    Direct,     // Direct convolution
    Im2Col,     // Image to column transformation followed by GEMM
    FFT,        // Fast Fourier Transform based convolution
    Winograd,   // Winograd algorithm for small kernels
}

/// Compute convolution
pub fn compute_conv(
    x: &Tensor, 
    w: &Tensor, 
    b: Option<&Tensor>, 
    stride: &[usize], 
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<Tensor> {
    // Validate input shapes
    if x.shape.len() < 3 {
        return Err(Error::ValidationError(
            format!("Conv input must be at least 3D, got shape {:?}", x.shape)
        ));
    }
    
    if w.shape.len() != x.shape.len() {
        return Err(Error::ValidationError(
            format!("Conv kernel must have same rank as input, got shapes {:?} and {:?}", x.shape, w.shape)
        ));
    }
    
    // Handle 1D, 2D, and 3D convolutions
    let spatial_dims = x.shape.len() - 2;
    if spatial_dims > 3 {
        return Err(Error::ValidationError(
            format!("Conv supports only 1D, 2D, and 3D convolutions, got {} spatial dimensions", spatial_dims)
        ));
    }
    
    // Choose the convolution strategy
    let strategy = optimize_convolution_strategy(&x.shape, &w.shape, groups);
    
    match strategy {
        ConvStrategy::Im2Col => {
            // For 2D convolution, use im2col + GEMM approach
            compute_conv_im2col(x, w, b, stride, padding, dilation, groups)
        },
        _ => {
            // For simplicity, fallback to direct convolution for other cases
            compute_conv_direct(x, w, b, stride, padding, dilation, groups)
        }
    }
}

/// Direct convolution implementation
fn compute_conv_direct(
    x: &Tensor, 
    w: &Tensor, 
    b: Option<&Tensor>, 
    stride: &[usize], 
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<Tensor> {
    // This is a simple implementation for 2D convolution
    // For a complete implementation, we'd handle 1D and 3D as well
    
    // Extract dimensions
    let batch_size = x.shape[0];
    let input_channels = x.shape[1];
    let input_height = x.shape[2];
    let input_width = if x.shape.len() > 3 { x.shape[3] } else { 1 };
    
    let output_channels = w.shape[0];
    let kernel_channels = w.shape[1];
    let kernel_height = w.shape[2];
    let kernel_width = if w.shape.len() > 3 { w.shape[3] } else { 1 };
    
    // Validate channel dimensions
    if groups != 1 {
        return Err(Error::UnsupportedFeature(
            "Grouped convolution not implemented in direct method".to_string()
        ));
    }
    
    if input_channels != kernel_channels {
        return Err(Error::ValidationError(
            format!("Inconsistent channel dimensions: input={}, kernel={}", input_channels, kernel_channels)
        ));
    }
    
    // Extract stride and padding
    let stride_h = stride.get(0).copied().unwrap_or(1);
    let stride_w = stride.get(1).copied().unwrap_or(1);
    
    let pad_top = padding.get(0).copied().unwrap_or(0);
    let pad_left = padding.get(1).copied().unwrap_or(0);
    let pad_bottom = padding.get(2).copied().unwrap_or(pad_top);
    let pad_right = padding.get(3).copied().unwrap_or(pad_left);
    
    let dilation_h = dilation.get(0).copied().unwrap_or(1);
    let dilation_w = dilation.get(1).copied().unwrap_or(1);
    
    // Calculate output dimensions
    let output_height = (input_height + pad_top + pad_bottom - 
                        (kernel_height - 1) * dilation_h - 1) / stride_h + 1;
    let output_width = (input_width + pad_left + pad_right - 
                       (kernel_width - 1) * dilation_w - 1) / stride_w + 1;
    
    // Create output tensor
    let output_shape = if x.shape.len() > 3 {
        vec![batch_size, output_channels, output_height, output_width]
    } else {
        vec![batch_size, output_channels, output_height]
    };
    
    let mut result = Tensor::new(&output_shape, x.data_type);
    
    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..output_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Initialize output pixel
                    let mut sum = 0.0;
                    
                    // Convolve at this position
                    for ic in 0..input_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh as isize * stride_h as isize - pad_top as isize + kh as isize * dilation_h as isize;
                                let iw = ow as isize * stride_w as isize - pad_left as isize + kw as isize * dilation_w as isize;
                                
                                // Skip if input position is outside (implicit zero padding)
                                if ih >= 0 && ih < input_height as isize && iw >= 0 && iw < input_width as isize {
                                    let input_val = if x.shape.len() > 3 {
                                        x.data[[b, ic, ih as usize, iw as usize]]
                                    } else {
                                        x.data[[b, ic, ih as usize]]
                                    };
                                    
                                    let kernel_val = if w.shape.len() > 3 {
                                        w.data[[oc, ic, kh, kw]]
                                    } else {
                                        w.data[[oc, ic, kh]]
                                    };
                                    
                                    sum += input_val * kernel_val;
                                }
                            }
                        }
                    }
                    
                    // Add bias if provided
                    if let Some(bias) = b {
                        sum += bias.data[[oc]];
                    }
                    
                    // Set output value
                    if result.shape.len() > 3 {
                        result.data[[b, oc, oh, ow]] = sum;
                    } else {
                        result.data[[b, oc, oh]] = sum;
                    }
                }
            }
        }
    }
    
    Ok(result)
}

/// Im2Col + GEMM based convolution for 2D
fn compute_conv_im2col(
    x: &Tensor, 
    w: &Tensor, 
    b: Option<&Tensor>, 
    stride: &[usize], 
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<Tensor> {
    // Only implement for 2D convolution (NCHW format)
    if x.shape.len() != 4 || w.shape.len() != 4 {
        return Err(Error::UnsupportedFeature(
            format!("Im2Col convolution only implemented for 2D (NCHW), got shapes {:?} and {:?}", 
                   x.shape, w.shape)
        ));
    }
    
    // Extract dimensions
    let batch_size = x.shape[0];
    let input_channels = x.shape[1];
    let input_height = x.shape[2];
    let input_width = x.shape[3];
    
    let output_channels = w.shape[0];
    let kernel_channels = w.shape[1];
    let kernel_height = w.shape[2];
    let kernel_width = w.shape[3];
    
    // Validate channel dimensions
    if (input_channels / groups) != kernel_channels {
        return Err(Error::ValidationError(
            format!("Inconsistent channel dimensions: input={}, kernel={}, groups={}", 
                   input_channels, kernel_channels, groups)
        ));
    }
    
    // Extract stride and padding
    let stride_h = stride.get(0).copied().unwrap_or(1);
    let stride_w = stride.get(1).copied().unwrap_or(1);
    
    let pad_top = padding.get(0).copied().unwrap_or(0);
    let pad_left = padding.get(1).copied().unwrap_or(0);
    let pad_bottom = padding.get(2).copied().unwrap_or(pad_top);
    let pad_right = padding.get(3).copied().unwrap_or(pad_left);
    
    let dilation_h = dilation.get(0).copied().unwrap_or(1);
    let dilation_w = dilation.get(1).copied().unwrap_or(1);
    
    // Calculate output dimensions
    let output_height = (input_height + pad_top + pad_bottom - 
                        (kernel_height - 1) * dilation_h - 1) / stride_h + 1;
    let output_width = (input_width + pad_left + pad_right - 
                       (kernel_width - 1) * dilation_w - 1) / stride_w + 1;
    
    // Create output tensor
    let output_shape = vec![batch_size, output_channels, output_height, output_width];
    let mut result = Tensor::new(&output_shape, x.data_type);
    
    // Convert image to column representation
    let col_buffer = im2col(x, &[kernel_height, kernel_width], stride, padding, dilation)?;
    
    // Reshape weights to matrix [output_channels, input_channels * kernel_height * kernel_width]
    let kernel_matrix = w.data.clone().into_shape((output_channels, kernel_channels * kernel_height * kernel_width))
        .map_err(|e| Error::ValidationError(format!("Failed to reshape kernel: {}", e)))?;
    
    // For each batch
    for b in 0..batch_size {
        // Extract the column buffer for this batch
        let col_view = col_buffer.slice(ndarray::s![b, .., ..]);
        
        // Perform matrix multiplication: kernel_matrix @ col_view
        let mut output_matrix = Array::zeros((output_channels, output_height * output_width));
        ndarray::linalg::general_mat_mul(1.0, &kernel_matrix, &col_view, 0.0, &mut output_matrix);
        
        // Add bias if provided
        if let Some(bias) = b {
            for oc in 0..output_channels {
                let bias_val = bias.data[[oc]];
                for i in 0..output_height * output_width {
                    output_matrix[[oc, i]] += bias_val;
                }
            }
        }
        
        // Reshape output to [output_channels, output_height, output_width]
        let output_view = output_matrix.into_shape((output_channels, output_height, output_width))
            .map_err(|e| Error::ValidationError(format!("Failed to reshape output: {}", e)))?;
        
        // Copy to result tensor
        for oc in 0..output_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    result.data[[b, oc, oh, ow]] = output_view[[oc, oh, ow]];
                }
            }
        }
    }
    
    Ok(result)
}

/// Convert image to column representation for convolution
pub fn im2col(
    input: &Tensor, 
    kernel_shape: &[usize], 
    stride: &[usize], 
    padding: &[usize],
    dilation: &[usize]
) -> Result<ArrayD<f32>> {
    // Only implement for 2D convolution (NCHW format)
    if input.shape.len() != 4 {
        return Err(Error::UnsupportedFeature(
            format!("Im2Col only implemented for 2D (NCHW), got shape {:?}", input.shape)
        ));
    }
    
    // Extract dimensions
    let batch_size = input.shape[0];
    let channels = input.shape[1];
    let height = input.shape[2];
    let width = input.shape[3];
    
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    
    // Extract stride and padding
    let stride_h = stride.get(0).copied().unwrap_or(1);
    let stride_w = stride.get(1).copied().unwrap_or(1);
    
    let pad_top = padding.get(0).copied().unwrap_or(0);
    let pad_left = padding.get(1).copied().unwrap_or(0);
    let pad_bottom = padding.get(2).copied().unwrap_or(pad_top);
    let pad_right = padding.get(3).copied().unwrap_or(pad_left);
    
    let dilation_h = dilation.get(0).copied().unwrap_or(1);
    let dilation_w = dilation.get(1).copied().unwrap_or(1);
    
    // Calculate output dimensions
    let output_height = (height + pad_top + pad_bottom - 
                        (kernel_height - 1) * dilation_h - 1) / stride_h + 1;
    let output_width = (width + pad_left + pad_right - 
                       (kernel_width - 1) * dilation_w - 1) / stride_w + 1;
    
    // Create column buffer [batch_size, channels * kernel_height * kernel_width, output_height * output_width]
    let mut col_buffer = Array::zeros(
        (batch_size, channels * kernel_height * kernel_width, output_height * output_width)
    );
    
    // Fill the column buffer
    for b in 0..batch_size {
        let mut col_idx = 0;
        for c in 0..channels {
            for kh in 0..kernel_height {
                for kw in 0..kernel_width {
                    let mut out_idx = 0;
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let ih = oh as isize * stride_h as isize - pad_top as isize + kh as isize * dilation_h as isize;
                            let iw = ow as isize * stride_w as isize - pad_left as isize + kw as isize * dilation_w as isize;
                            
                            let val = if ih >= 0 && ih < height as isize && iw >= 0 && iw < width as isize {
                                input.data[[b, c, ih as usize, iw as usize]]
                            } else {
                                0.0  // Zero padding
                            };
                            
                            col_buffer[[b, col_idx, out_idx]] = val;
                            out_idx += 1;
                        }
                    }
                    col_idx += 1;
                }
            }
        }
    }
    
    Ok(col_buffer)
}

/// Determine the best strategy for convolution
pub fn optimize_convolution_strategy(
    input_shape: &[usize], 
    kernel_shape: &[usize],
    groups: usize
) -> ConvStrategy {
    // Only implement simple heuristics
    // A real implementation would consider more factors
    
    if input_shape.len() != 4 || kernel_shape.len() != 4 {
        // For non-2D convolutions, use direct method
        return ConvStrategy::Direct;
    }
    
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];
    
    if kernel_height <= 3 && kernel_width <= 3 && groups == 1 {
        // For small kernels with no grouping, im2col is usually faster
        ConvStrategy::Im2Col
    } else if kernel_height > 5 || kernel_width > 5 {
        // For large kernels, FFT might be better, but we don't implement it
        // So fallback to direct
        ConvStrategy::Direct
    } else {
        // Default to im2col for moderate sized kernels
        ConvStrategy::Im2Col
    }
}