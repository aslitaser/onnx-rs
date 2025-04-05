// Utilities module for profiler
// Contains helper functions for data type conversions and calculations

use anyhow::{Result, anyhow};

use crate::ops::tensor::DataType;

/// Convert a DataType to its string representation
pub fn data_type_to_string(data_type: DataType) -> String {
    match data_type {
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Int8 => "Int8".to_string(), 
        DataType::Int16 => "Int16".to_string(),
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Uint8 => "Uint8".to_string(),
        DataType::Uint16 => "Uint16".to_string(),
        DataType::Uint32 => "Uint32".to_string(),
        DataType::Uint64 => "Uint64".to_string(),
        DataType::Bool => "Bool".to_string(),
        DataType::BFloat16 => "BFloat16".to_string(),
        DataType::Float16 => "Float16".to_string(),
        DataType::String => "String".to_string(),
        DataType::Complex64 => "Complex64".to_string(),
        DataType::Complex128 => "Complex128".to_string(),
    }
}

/// Convert a string to a DataType
pub fn data_type_from_string(type_str: &str) -> DataType {
    match type_str {
        "Float32" | "float32" | "FLOAT32" | "float" => DataType::Float32,
        "Float64" | "float64" | "FLOAT64" | "double" => DataType::Float64,
        "Int8" | "int8" | "INT8" => DataType::Int8,
        "Int16" | "int16" | "INT16" => DataType::Int16,
        "Int32" | "int32" | "INT32" => DataType::Int32,
        "Int64" | "int64" | "INT64" => DataType::Int64,
        "Uint8" | "uint8" | "UINT8" => DataType::Uint8,
        "Uint16" | "uint16" | "UINT16" => DataType::Uint16,
        "Uint32" | "uint32" | "UINT32" => DataType::Uint32,
        "Uint64" | "uint64" | "UINT64" => DataType::Uint64,
        "Bool" | "bool" | "BOOL" | "boolean" => DataType::Bool,
        "BFloat16" | "bfloat16" | "BFLOAT16" => DataType::BFloat16,
        "Float16" | "float16" | "FLOAT16" | "half" => DataType::Float16,
        "String" | "string" | "STRING" => DataType::String,
        "Complex64" | "complex64" | "COMPLEX64" => DataType::Complex64,
        "Complex128" | "complex128" | "COMPLEX128" => DataType::Complex128,
        _ => {
            // Handle common abbreviations and synonyms
            if type_str.eq_ignore_ascii_case("f32") { 
                DataType::Float32 
            } else if type_str.eq_ignore_ascii_case("f64") {
                DataType::Float64
            } else if type_str.eq_ignore_ascii_case("i8") {
                DataType::Int8
            } else if type_str.eq_ignore_ascii_case("i16") {
                DataType::Int16
            } else if type_str.eq_ignore_ascii_case("i32") {
                DataType::Int32
            } else if type_str.eq_ignore_ascii_case("i64") {
                DataType::Int64
            } else if type_str.eq_ignore_ascii_case("u8") {
                DataType::Uint8
            } else if type_str.eq_ignore_ascii_case("u16") {
                DataType::Uint16
            } else if type_str.eq_ignore_ascii_case("u32") {
                DataType::Uint32
            } else if type_str.eq_ignore_ascii_case("u64") {
                DataType::Uint64
            } else if type_str.eq_ignore_ascii_case("bf16") {
                DataType::BFloat16
            } else if type_str.eq_ignore_ascii_case("f16") {
                DataType::Float16
            } else {
                // For unknown types, default to Float32
                DataType::Float32
            }
        }
    }
}

/// Calculate memory usage for a tensor based on its data type and dimensions
/// Returns memory usage in bytes
pub fn calculate_tensor_memory_usage(data_type: DataType, dimensions: &[usize]) -> usize {
    // Calculate number of elements
    let element_count: usize = if dimensions.is_empty() {
        1 // Scalar
    } else {
        dimensions.iter().product()
    };
    
    // Memory per element based on data type
    let bytes_per_element = match data_type {
        DataType::Float32 => 4,
        DataType::Float64 => 8,
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 4,
        DataType::Int64 => 8,
        DataType::Uint8 => 1,
        DataType::Uint16 => 2,
        DataType::Uint32 => 4,
        DataType::Uint64 => 8,
        DataType::Bool => 1,
        DataType::BFloat16 => 2,
        DataType::Float16 => 2,
        DataType::Complex64 => 8,  // 2 * Float32
        DataType::Complex128 => 16, // 2 * Float64
        DataType::String => {
            // For string tensors, this is approximate as actual size depends on string content
            // We use 24 bytes as a reasonable estimate for the average string size
            // (8 bytes for pointer + average 16 bytes per string content)
            24
        }
    };
    
    // Calculate total memory usage
    element_count * bytes_per_element
}

/// Estimate memory requirements for common ONNX operations
/// Returns estimated memory usage in bytes
pub fn estimate_operation_memory(
    op_type: &str, 
    input_shapes: &[Vec<usize>], 
    input_types: &[DataType]
) -> usize {
    if input_shapes.is_empty() || input_types.is_empty() {
        return 0;
    }
    
    match op_type {
        "Conv" => {
            // For Conv, we need input, weights, bias (optional), and output
            // Output size depends on kernel size, padding, etc., but we'll estimate as same size as input
            if input_shapes.len() >= 2 {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let weights_memory = calculate_tensor_memory_usage(input_types[1], &input_shapes[1]);
                
                // Bias is optional
                let bias_memory = if input_shapes.len() > 2 {
                    calculate_tensor_memory_usage(input_types[2], &input_shapes[2])
                } else {
                    0
                };
                
                // Output memory (estimate)
                let output_memory = input_memory;
                
                // Workspace memory for some convolution algorithms
                let workspace_memory = input_memory * 2;
                
                input_memory + weights_memory + bias_memory + output_memory + workspace_memory
            } else {
                0
            }
        },
        "MatMul" | "Gemm" => {
            // Matrix multiplication memory estimation
            if input_shapes.len() >= 2 {
                let a_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                let b_memory = calculate_tensor_memory_usage(input_types[1], &input_shapes[1]);
                
                // Rough estimate for output size
                let output_memory = if input_shapes[0].len() >= 2 && input_shapes[1].len() >= 2 {
                    let m = input_shapes[0][0];
                    let n = input_shapes[1][1];
                    let element_size = match input_types[0] {
                        DataType::Float32 => 4,
                        DataType::Float64 => 8,
                        _ => 4,
                    };
                    m * n * element_size
                } else {
                    a_memory
                };
                
                // Some BLAS implementations use workspace memory
                let workspace_memory = (a_memory + b_memory) / 2;
                
                a_memory + b_memory + output_memory + workspace_memory
            } else {
                0
            }
        },
        "Relu" | "Sigmoid" | "Tanh" => {
            // Activation functions: input plus output (same size)
            if !input_shapes.is_empty() {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                input_memory * 2 // Input + output
            } else {
                0
            }
        },
        "MaxPool" | "AveragePool" => {
            // Pooling operations: input plus output (output smaller, but approximate as same size)
            if !input_shapes.is_empty() {
                let input_memory = calculate_tensor_memory_usage(input_types[0], &input_shapes[0]);
                input_memory * 2 // Input + output
            } else {
                0
            }
        },
        _ => {
            // Sum the memory of all inputs
            let mut total_input_memory = 0;
            for i in 0..input_shapes.len() {
                let data_type = if i < input_types.len() { input_types[i] } else { input_types[0] };
                total_input_memory += calculate_tensor_memory_usage(data_type, &input_shapes[i]);
            }
            
            // Estimate output as roughly same size as all inputs combined
            total_input_memory * 2
        }
    }
}

/// Parse tensor shape from a string like "[1,3,224,224]" or "(1, 3, 224, 224)"
pub fn parse_tensor_shape(shape_str: &str) -> Result<Vec<usize>> {
    // Remove brackets, parentheses, etc.
    let clean_str = shape_str
        .trim()
        .trim_start_matches(&['[', '(', '{'])
        .trim_end_matches(&[']', ')', '}']);
    
    // Split by commas and parse each dimension
    let dimensions: Result<Vec<usize>> = clean_str
        .split(',')
        .map(|dim| {
            dim.trim().parse::<usize>().map_err(|e| {
                anyhow!("Failed to parse tensor dimension '{}': {}", dim, e)
            })
        })
        .collect();
    
    dimensions
}