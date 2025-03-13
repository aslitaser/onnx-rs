use std::fmt;
use ndarray::{Array, ArrayD, IxDyn, Dimension, Axis};
use num_traits::{NumCast, ToPrimitive, FromPrimitive, AsPrimitive, Float, Bounded};
use half::{f16, bf16};
use crate::error::{Error, Result};

/// Shape of a tensor
pub type Shape = Vec<usize>;

/// Represents behavior for handling overflow/underflow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowBehavior {
    /// Saturate values to the bounds of the target type
    Saturate,
    /// Wrap around values that exceed the bounds (modular arithmetic)
    Wrap,
    /// Return an error if overflow occurs
    Error,
}

/// Represents behavior for handling NaN and Infinity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialValueBehavior {
    /// Replace NaN with zero, Infinity with max/min value
    ZeroAndSaturate,
    /// Preserve NaN and Infinity where possible, otherwise use default
    Preserve,
    /// Return an error on NaN or Infinity
    Error,
}

/// Represents rounding strategy for floating-point to integer conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingStrategy {
    /// Round to the nearest, with ties rounding away from zero
    Round,
    /// Always round toward zero (truncate)
    Truncate,
    /// Always round toward negative infinity (floor)
    Floor,
    /// Always round toward positive infinity (ceiling)
    Ceiling,
}

/// Conversion options for data type conversions
#[derive(Debug, Clone, Copy)]
pub struct ConversionOptions {
    pub overflow_behavior: OverflowBehavior,
    pub special_value_behavior: SpecialValueBehavior,
    pub rounding_strategy: RoundingStrategy,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        ConversionOptions {
            overflow_behavior: OverflowBehavior::Saturate,
            special_value_behavior: SpecialValueBehavior::ZeroAndSaturate,
            rounding_strategy: RoundingStrategy::Round,
        }
    }
}

/// Data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    BFloat16,
    Float16,
    String,
    Complex64,
    Complex128,
}

impl DataType {
    /// Convert from model data type
    pub fn from_model_type(data_type: crate::model::DataType) -> Result<Self> {
        match data_type {
            crate::model::DataType::Float => Ok(DataType::Float32),
            crate::model::DataType::Double => Ok(DataType::Float64),
            crate::model::DataType::Int8 => Ok(DataType::Int8),
            crate::model::DataType::Int16 => Ok(DataType::Int16),
            crate::model::DataType::Int32 => Ok(DataType::Int32),
            crate::model::DataType::Int64 => Ok(DataType::Int64),
            crate::model::DataType::Uint8 => Ok(DataType::Uint8),
            crate::model::DataType::Uint16 => Ok(DataType::Uint16),
            crate::model::DataType::Uint32 => Ok(DataType::Uint32),
            crate::model::DataType::Uint64 => Ok(DataType::Uint64),
            crate::model::DataType::Bool => Ok(DataType::Bool),
            crate::model::DataType::BFloat16 => Ok(DataType::BFloat16),
            crate::model::DataType::Float16 => Ok(DataType::Float16),
            crate::model::DataType::String => Ok(DataType::String),
            crate::model::DataType::Complex64 => Ok(DataType::Complex64),
            crate::model::DataType::Complex128 => Ok(DataType::Complex128),
            _ => Err(Error::UnsupportedFeature(format!("Unsupported data type: {:?}", data_type))),
        }
    }
    
    /// Get the size in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
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
            DataType::String => std::mem::size_of::<usize>(), // Size of pointer/reference
            DataType::Complex64 => 8,  // 2 * Float32
            DataType::Complex128 => 16, // 2 * Float64
        }
    }
    
    /// Check if data type is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(self, 
            DataType::Float32 | 
            DataType::Float64 | 
            DataType::BFloat16 | 
            DataType::Float16
        )
    }
    
    /// Check if data type is signed integer
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, 
            DataType::Int8 | 
            DataType::Int16 | 
            DataType::Int32 | 
            DataType::Int64
        )
    }
    
    /// Check if data type is unsigned integer
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, 
            DataType::Uint8 | 
            DataType::Uint16 | 
            DataType::Uint32 | 
            DataType::Uint64
        )
    }
    
    /// Check if data type is integer (signed or unsigned)
    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }
    
    /// Check if data type is boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, DataType::Bool)
    }
    
    /// Check if data type is string
    pub fn is_string(&self) -> bool {
        matches!(self, DataType::String)
    }
    
    /// Check if data type is complex
    pub fn is_complex(&self) -> bool {
        matches!(self, DataType::Complex64 | DataType::Complex128)
    }
    
    /// Get minimum and maximum value for the data type (as f64)
    pub fn get_value_range(&self) -> (f64, f64) {
        match self {
            DataType::Float32 => (f32::MIN as f64, f32::MAX as f64),
            DataType::Float64 => (f64::MIN, f64::MAX),
            DataType::Int8 => (i8::MIN as f64, i8::MAX as f64),
            DataType::Int16 => (i16::MIN as f64, i16::MAX as f64),
            DataType::Int32 => (i32::MIN as f64, i32::MAX as f64),
            DataType::Int64 => (i64::MIN as f64, i64::MAX as f64),
            DataType::Uint8 => (0.0, u8::MAX as f64),
            DataType::Uint16 => (0.0, u16::MAX as f64),
            DataType::Uint32 => (0.0, u32::MAX as f64),
            DataType::Uint64 => (0.0, u64::MAX as f64),
            DataType::Bool => (0.0, 1.0),
            DataType::BFloat16 => (-3.38e+38, 3.38e+38), // Approximate BFloat16 range
            DataType::Float16 => (-65504.0, 65504.0),    // Float16 range
            // Complex numbers have same range as their float components for real/imaginary parts
            DataType::Complex64 => (f32::MIN as f64, f32::MAX as f64),
            DataType::Complex128 => (f64::MIN, f64::MAX),
            // String doesn't have a numeric range
            DataType::String => (0.0, 0.0),
        }
    }
}

/// Tensor struct for runtime computation
#[derive(Clone)]
pub struct Tensor {
    pub name: Option<String>,
    pub data_type: DataType,
    pub shape: Shape,
    // Store data as f32 for internal computation by default
    // For string and complex types, we use special storage
    pub data: ArrayD<f32>,
    // Store string data separately
    pub string_data: Option<Vec<String>>,
    // Store complex data as pairs of real and imaginary components
    pub complex_data: Option<(ArrayD<f32>, ArrayD<f32>)>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{ name: {:?}, data_type: {:?}, shape: {:?}, data: [shape: {:?}] }}",
               self.name, self.data_type, self.shape, self.data.shape())
    }
}

impl Tensor {
    /// Return true if tensor contains string data
    pub fn has_string_data(&self) -> bool {
        self.string_data.is_some()
    }
    
    /// Return true if tensor contains complex data
    pub fn has_complex_data(&self) -> bool {
        self.complex_data.is_some()
    }
    
    /// Get a reference to the real part of complex data
    pub fn real_part(&self) -> Option<&ArrayD<f32>> {
        self.complex_data.as_ref().map(|(real, _)| real)
    }
    
    /// Get a reference to the imaginary part of complex data
    pub fn imag_part(&self) -> Option<&ArrayD<f32>> {
        self.complex_data.as_ref().map(|(_, imag)| imag)
    }
}

impl Tensor {
    /// Create a new tensor with empty data
    pub fn new(shape: &[usize], data_type: DataType) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        
        // Create complex data if needed
        let complex_data = if data_type.is_complex() {
            Some((
                ArrayD::zeros(IxDyn(shape)),  // Real part
                ArrayD::zeros(IxDyn(shape))   // Imaginary part
            ))
        } else {
            None
        };
        
        // Create string data if needed
        let string_data = if data_type.is_string() {
            let total_elements = shape.iter().product();
            Some(vec![String::new(); total_elements])
        } else {
            None
        };
        
        Self {
            name: None,
            data_type,
            shape: shape.to_vec(),
            data,
            string_data,
            complex_data,
        }
    }
    
    /// Create a tensor from ndarray with conversion options
    pub fn from_ndarray<T: NumCast + Copy>(
        arr: ArrayD<T>, 
        data_type: DataType,
        options: Option<ConversionOptions>
    ) -> Result<Self> {
        let options = options.unwrap_or_default();
        let shape = arr.shape().to_vec();
        
        // Handle different target types
        match data_type {
            DataType::String => {
                // Convert values to strings
                let total_elements = shape.iter().product();
                let mut string_data = Vec::with_capacity(total_elements);
                
                for &val in arr.iter() {
                    if let Some(f) = NumCast::from(val) {
                        string_data.push(f.to_string());
                    } else {
                        string_data.push("0".to_string());
                    }
                }
                
                // Create a placeholder f32 array
                Ok(Self {
                    name: None,
                    data_type,
                    shape,
                    data: ArrayD::zeros(IxDyn(&shape)),
                    string_data: Some(string_data),
                    complex_data: None,
                })
            },
            DataType::Complex64 | DataType::Complex128 => {
                // For complex data, we assume the source is either already complex 
                // or we'll create a complex number with zero imaginary part
                
                // Convert to f32 for the real part
                let real_data = arr.mapv(|x| {
                    NumCast::from(x).unwrap_or(0.0)
                });
                
                // Create zero imaginary part
                let imag_data = ArrayD::zeros(IxDyn(&shape));
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape,
                    data: ArrayD::zeros(IxDyn(&shape)), // Placeholder
                    string_data: None,
                    complex_data: Some((real_data, imag_data)),
                })
            },
            _ => {
                // Regular numeric conversion with overflow handling
                let data = convert_array_with_options(arr, options)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape,
                    data,
                    string_data: None,
                    complex_data: None,
                })
            }
        }
    }
    
    /// Create a tensor from ndarray (simplified version with default options)
    pub fn from_ndarray_simple<T: NumCast + Copy>(arr: ArrayD<T>, data_type: DataType) -> Result<Self> {
        Self::from_ndarray(arr, data_type, None)
    }
    
    /// Convert tensor to ndarray of specific type with conversion options
    pub fn to_ndarray<T: NumCast + Copy>(&self, options: Option<ConversionOptions>) -> Result<ArrayD<T>> {
        let options = options.unwrap_or_default();
        
        // Handle different source types
        match self.data_type {
            DataType::String => {
                if let Some(string_data) = &self.string_data {
                    // Try to parse strings to target numeric type
                    let mut values = Vec::with_capacity(string_data.len());
                    
                    for s in string_data {
                        // Try to parse as f64 first (most flexible)
                        if let Ok(val) = s.parse::<f64>() {
                            if let Some(target_val) = NumCast::from(val) {
                                values.push(target_val);
                            } else {
                                // Fallback to zero if conversion fails
                                values.push(T::from(0).unwrap());
                            }
                        } else {
                            // String couldn't be parsed as number
                            values.push(T::from(0).unwrap());
                        }
                    }
                    
                    ArrayD::from_shape_vec(IxDyn(&self.shape), values)
                        .map_err(|e| Error::ExecutionError(format!("Shape mismatch in conversion: {}", e)))
                } else {
                    Err(Error::ExecutionError(
                        "String tensor missing string_data".to_string()
                    ))
                }
            },
            DataType::Complex64 | DataType::Complex128 => {
                if let Some((real, _imag)) = &self.complex_data {
                    // For complex to non-complex conversion, we only use the real part
                    let result = real.mapv(|x| {
                        match NumCast::from(x) {
                            Some(val) => val,
                            None => T::from(0).unwrap(),
                        }
                    });
                    
                    Ok(result)
                } else {
                    Err(Error::ExecutionError(
                        "Complex tensor missing complex_data".to_string()
                    ))
                }
            },
            _ => {
                // Regular numeric conversion
                convert_from_array_with_options(&self.data, options)
            }
        }
    }
    
    /// Convert tensor to ndarray of specific type (simplified version with default options)
    pub fn to_ndarray_simple<T: NumCast + Copy>(&self) -> Result<ArrayD<T>> {
        self.to_ndarray(None)
    }
    
    /// Create a tensor from raw data with comprehensive type support
    pub fn from_raw_data(data: &[u8], shape: &[usize], data_type: DataType) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        let expected_bytes = total_elements * data_type.size_in_bytes();
        
        if data.len() != expected_bytes && !data_type.is_string() {
            return Err(Error::InvalidModel(format!(
                "Tensor data size mismatch. Expected {} bytes but got {}",
                expected_bytes, data.len()
            )));
        }
        
        // Create array and convert to f32
        match data_type {
            DataType::Float32 => {
                let mut float_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let value = f32::from_le_bytes(bytes);
                    float_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), float_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Float64 => {
                let mut float_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(8) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3], 
                                 chunk[4], chunk[5], chunk[6], chunk[7]];
                    let value = f64::from_le_bytes(bytes) as f32;
                    float_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), float_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Int8 => {
                let mut int_data = Vec::with_capacity(total_elements);
                for &byte in data {
                    let value = i8::from_le_bytes([byte]) as f32;
                    int_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), int_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Int16 => {
                let mut int_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let value = i16::from_le_bytes(bytes) as f32;
                    int_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), int_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Int32 => {
                let mut int_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let value = i32::from_le_bytes(bytes) as f32;
                    int_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), int_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Int64 => {
                let mut int_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(8) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3], 
                                 chunk[4], chunk[5], chunk[6], chunk[7]];
                    let value = i64::from_le_bytes(bytes) as f32;
                    int_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), int_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Uint8 => {
                let mut uint_data = Vec::with_capacity(total_elements);
                for &byte in data {
                    let value = byte as f32;
                    uint_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), uint_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Uint16 => {
                let mut uint_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let value = u16::from_le_bytes(bytes) as f32;
                    uint_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), uint_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Uint32 => {
                let mut uint_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let value = u32::from_le_bytes(bytes) as f32;
                    uint_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), uint_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Uint64 => {
                let mut uint_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(8) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3], 
                                 chunk[4], chunk[5], chunk[6], chunk[7]];
                    let value = u64::from_le_bytes(bytes) as f32;
                    uint_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), uint_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Bool => {
                let mut bool_data = Vec::with_capacity(total_elements);
                for &byte in data {
                    let value = if byte != 0 { 1.0 } else { 0.0 };
                    bool_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), bool_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Float16 => {
                let mut float_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let bits = u16::from_le_bytes(bytes);
                    let value = f16::from_bits(bits).to_f32();
                    float_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), float_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::BFloat16 => {
                let mut float_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let bits = u16::from_le_bytes(bytes);
                    let value = bf16::from_bits(bits).to_f32();
                    float_data.push(value);
                }
                let data_array = ArrayD::from_shape_vec(IxDyn(shape), float_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: data_array,
                    string_data: None,
                    complex_data: None,
                })
            },
            DataType::Complex64 => {
                let mut real_data = Vec::with_capacity(total_elements);
                let mut imag_data = Vec::with_capacity(total_elements);
                
                // Each complex64 is a pair of f32 (real, imag)
                for chunk in data.chunks_exact(8) {
                    let real_bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let imag_bytes = [chunk[4], chunk[5], chunk[6], chunk[7]];
                    
                    let real = f32::from_le_bytes(real_bytes);
                    let imag = f32::from_le_bytes(imag_bytes);
                    
                    real_data.push(real);
                    imag_data.push(imag);
                }
                
                let real_array = ArrayD::from_shape_vec(IxDyn(shape), real_data)?;
                let imag_array = ArrayD::from_shape_vec(IxDyn(shape), imag_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: ArrayD::zeros(IxDyn(shape)), // Placeholder
                    string_data: None,
                    complex_data: Some((real_array, imag_array)),
                })
            },
            DataType::Complex128 => {
                let mut real_data = Vec::with_capacity(total_elements);
                let mut imag_data = Vec::with_capacity(total_elements);
                
                // Each complex128 is a pair of f64 (real, imag)
                for chunk in data.chunks_exact(16) {
                    let real_bytes = [chunk[0], chunk[1], chunk[2], chunk[3], 
                                    chunk[4], chunk[5], chunk[6], chunk[7]];
                    let imag_bytes = [chunk[8], chunk[9], chunk[10], chunk[11], 
                                    chunk[12], chunk[13], chunk[14], chunk[15]];
                    
                    let real = f64::from_le_bytes(real_bytes) as f32;
                    let imag = f64::from_le_bytes(imag_bytes) as f32;
                    
                    real_data.push(real);
                    imag_data.push(imag);
                }
                
                let real_array = ArrayD::from_shape_vec(IxDyn(shape), real_data)?;
                let imag_array = ArrayD::from_shape_vec(IxDyn(shape), imag_data)?;
                
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: ArrayD::zeros(IxDyn(shape)), // Placeholder
                    string_data: None,
                    complex_data: Some((real_array, imag_array)),
                })
            },
            DataType::String => {
                // String tensors require special handling as they are variable length
                // We need to parse the bytes based on the ONNX string tensor format
                // Format: [num_bytes (4 bytes), string_data (num_bytes)]
                
                let mut strings = Vec::with_capacity(total_elements);
                let mut offset = 0;
                
                for _ in 0..total_elements {
                    if offset + 4 > data.len() {
                        break;
                    }
                    
                    // Read string length (4 bytes)
                    let len_bytes = [data[offset], data[offset+1], data[offset+2], data[offset+3]];
                    let string_len = u32::from_le_bytes(len_bytes) as usize;
                    offset += 4;
                    
                    if offset + string_len > data.len() {
                        return Err(Error::InvalidModel(
                            "Invalid string data format in tensor".to_string()
                        ));
                    }
                    
                    // Extract string bytes
                    let string_bytes = &data[offset..offset+string_len];
                    offset += string_len;
                    
                    // Convert to UTF-8 string
                    match std::str::from_utf8(string_bytes) {
                        Ok(s) => strings.push(s.to_string()),
                        Err(_) => return Err(Error::InvalidModel(
                            "Invalid UTF-8 string in tensor".to_string()
                        )),
                    }
                }
                
                // Create a placeholder f32 array
                Ok(Self {
                    name: None,
                    data_type,
                    shape: shape.to_vec(),
                    data: ArrayD::zeros(IxDyn(shape)),
                    string_data: Some(strings),
                    complex_data: None,
                })
            },
        }
    }
}
    
    /// Create a tensor from model tensor
    pub fn from_model_tensor(tensor: &crate::model::Tensor) -> Result<Self> {
        let mut shape = Vec::new();
        for &dim in &tensor.dims {
            if dim < 0 {
                return Err(Error::InvalidModel(
                    format!("Negative dimension {} in tensor {}", dim, tensor.name)
                ));
            }
            shape.push(dim as usize);
        }
        
        let data_type = DataType::from_model_type(tensor.data_type)?;
        let result = Self::from_raw_data(&tensor.data, &shape, data_type)?;
        Ok(Self {
            name: Some(tensor.name.clone()),
            ..result
        })
    }
    
    /// Reshape the tensor
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let new_total_elements: usize = shape.iter().product();
        let current_total_elements: usize = self.shape.iter().product();
        
        if new_total_elements != current_total_elements {
            return Err(Error::InvalidModel(format!(
                "Cannot reshape tensor from {:?} to {:?}: total elements don't match",
                self.shape, shape
            )));
        }
        
        // Reshape regular numeric data
        let reshaped_data = match self.data.clone().into_shape(IxDyn(shape)) {
            Ok(array) => array,
            Err(e) => return Err(Error::InvalidModel(format!(
                "Reshape error: {}", e
            ))),
        };
        
        // Reshape complex data if present
        let complex_data = if let Some((real, imag)) = &self.complex_data {
            let real_reshaped = match real.clone().into_shape(IxDyn(shape)) {
                Ok(array) => array,
                Err(e) => return Err(Error::InvalidModel(format!(
                    "Reshape error (complex real part): {}", e
                ))),
            };
            
            let imag_reshaped = match imag.clone().into_shape(IxDyn(shape)) {
                Ok(array) => array,
                Err(e) => return Err(Error::InvalidModel(format!(
                    "Reshape error (complex imaginary part): {}", e
                ))),
            };
            
            Some((real_reshaped, imag_reshaped))
        } else {
            None
        };
        
        // Handle string data
        let string_data = self.string_data.clone();
        
        Ok(Tensor {
            name: self.name.clone(),
            data_type: self.data_type,
            shape: shape.to_vec(),
            data: reshaped_data,
            string_data,
            complex_data,
        })
    }
    
    /// Cast tensor to another data type with conversion options
    pub fn cast_to(&self, target_type: DataType, options: Option<ConversionOptions>) -> Result<Tensor> {
        let options = options.unwrap_or_default();
        
        if self.data_type == target_type {
            return Ok(self.clone());
        }
        
        // Handle different source and target types
        match (self.data_type, target_type) {
            // Convert from string to any numeric type
            (DataType::String, _) => {
                if target_type.is_string() {
                    return Ok(self.clone());
                }
                
                if let Some(string_data) = &self.string_data {
                    let mut converted_data = Vec::with_capacity(string_data.len());
                    
                    for s in string_data {
                        // Try to parse as f64 first for maximum precision
                        if let Ok(val) = s.parse::<f64>() {
                            // Apply conversion options
                            let converted = convert_value_with_options(val, target_type, &options)?;
                            converted_data.push(converted);
                        } else {
                            converted_data.push(0.0); // Default for non-numeric strings
                        }
                    }
                    
                    let data_array = ArrayD::from_shape_vec(IxDyn(&self.shape), converted_data)?;
                    
                    Ok(Tensor {
                        name: self.name.clone(),
                        data_type: target_type,
                        shape: self.shape.clone(),
                        data: data_array,
                        string_data: None,
                        complex_data: None,
                    })
                } else {
                    Err(Error::ExecutionError("String tensor missing string_data".to_string()))
                }
            },
            
            // Convert from any type to string
            (_, DataType::String) => {
                let total_elements = self.shape.iter().product();
                let mut string_data = Vec::with_capacity(total_elements);
                
                // Convert each element to string
                for &val in self.data.iter() {
                    string_data.push(val.to_string());
                }
                
                Ok(Tensor {
                    name: self.name.clone(),
                    data_type: target_type,
                    shape: self.shape.clone(),
                    data: ArrayD::zeros(IxDyn(&self.shape)), // Placeholder
                    string_data: Some(string_data),
                    complex_data: None,
                })
            },
            
            // Convert from complex to complex
            (DataType::Complex64, DataType::Complex128) | 
            (DataType::Complex128, DataType::Complex64) => {
                if let Some((real, imag)) = &self.complex_data {
                    // Just change the type for now - data is already stored as f32
                    Ok(Tensor {
                        name: self.name.clone(),
                        data_type: target_type,
                        shape: self.shape.clone(),
                        data: ArrayD::zeros(IxDyn(&self.shape)), // Placeholder
                        string_data: None,
                        complex_data: Some((real.clone(), imag.clone())),
                    })
                } else {
                    Err(Error::ExecutionError("Complex tensor missing complex_data".to_string()))
                }
            },
            
            // Convert from complex to real
            (DataType::Complex64, _) | (DataType::Complex128, _) => {
                if let Some((real, _)) = &self.complex_data {
                    // Use only the real part
                    let converted = convert_array_with_numeric_options(real, target_type, &options)?;
                    
                    Ok(Tensor {
                        name: self.name.clone(),
                        data_type: target_type,
                        shape: self.shape.clone(),
                        data: converted,
                        string_data: None,
                        complex_data: None,
                    })
                } else {
                    Err(Error::ExecutionError("Complex tensor missing complex_data".to_string()))
                }
            },
            
            // Convert from real to complex
            (_, DataType::Complex64) | (_, DataType::Complex128) => {
                // Use the real data as real part, zeros for imaginary
                let imag_data = ArrayD::zeros(IxDyn(&self.shape));
                
                Ok(Tensor {
                    name: self.name.clone(),
                    data_type: target_type,
                    shape: self.shape.clone(),
                    data: ArrayD::zeros(IxDyn(&self.shape)), // Placeholder
                    string_data: None,
                    complex_data: Some((self.data.clone(), imag_data)),
                })
            },
            
            // Handle regular numeric conversions
            _ => {
                let converted = convert_array_with_numeric_options(&self.data, target_type, &options)?;
                
                Ok(Tensor {
                    name: self.name.clone(),
                    data_type: target_type,
                    shape: self.shape.clone(),
                    data: converted,
                    string_data: None,
                    complex_data: None,
                })
            }
        }
    }
    
    /// Cast tensor to another data type (simplified version with default options)
    pub fn cast_to_simple(&self, target_type: DataType) -> Result<Tensor> {
        self.cast_to(target_type, None)
    }
    
    /// Broadcast tensor to target shape
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        if self.shape == shape {
            return Ok(self.clone());
        }
        
        // Check if broadcasting is possible
        if !can_broadcast(&self.shape, shape) {
            return Err(Error::InvalidModel(format!(
                "Cannot broadcast tensor of shape {:?} to {:?}",
                self.shape, shape
            )));
        }
        
        // Handle different data types
        match self.data_type {
            DataType::String => {
                if let Some(string_data) = &self.string_data {
                    // Broadcast string data
                    let mut output_strings = vec![String::new(); shape.iter().product()];
                    let mut output_indices = ArrayD::<usize>::zeros(IxDyn(shape));
                    
                    // Initialize output indices
                    for (idx, val) in output_indices.iter_mut().enumerate() {
                        *val = idx;
                    }
                    
                    // Map each output index to input index
                    let broadcast_indices = calculate_broadcast_indices(&self.shape, shape, &output_indices)?;
                    
                    // Copy strings based on mapping
                    for (out_idx, in_idx) in broadcast_indices.iter().enumerate() {
                        if *in_idx < string_data.len() {
                            output_strings[out_idx] = string_data[*in_idx].clone();
                        }
                    }
                    
                    Ok(Tensor {
                        name: self.name.clone(),
                        data_type: self.data_type,
                        shape: shape.to_vec(),
                        data: ArrayD::zeros(IxDyn(shape)), // Placeholder
                        string_data: Some(output_strings),
                        complex_data: None,
                    })
                } else {
                    Err(Error::ExecutionError("String tensor missing string_data".to_string()))
                }
            },
            DataType::Complex64 | DataType::Complex128 => {
                if let Some((real, imag)) = &self.complex_data {
                    // Create output arrays for real and imaginary parts
                    let mut output_real = ArrayD::<f32>::zeros(IxDyn(shape));
                    let mut output_imag = ArrayD::<f32>::zeros(IxDyn(shape));
                    
                    // Broadcast real and imaginary parts separately
                    broadcast_copy(&real.view(), &mut output_real.view_mut())?;
                    broadcast_copy(&imag.view(), &mut output_imag.view_mut())?;
                    
                    Ok(Tensor {
                        name: self.name.clone(),
                        data_type: self.data_type,
                        shape: shape.to_vec(),
                        data: ArrayD::zeros(IxDyn(shape)), // Placeholder
                        string_data: None,
                        complex_data: Some((output_real, output_imag)),
                    })
                } else {
                    Err(Error::ExecutionError("Complex tensor missing complex_data".to_string()))
                }
            },
            _ => {
                // Create output array and perform the broadcast for regular numeric data
                let mut output = ArrayD::<f32>::zeros(IxDyn(shape));
                
                // Perform broadcasting
                broadcast_copy(&self.data.view(), &mut output.view_mut())?;
                
                Ok(Tensor {
                    name: self.name.clone(),
                    data_type: self.data_type,
                    shape: shape.to_vec(),
                    data: output,
                    string_data: None,
                    complex_data: None,
                })
            }
        }
    }
    
    /// Calculate broadcast indices mapping
    fn calculate_broadcast_indices(from_shape: &[usize], to_shape: &[usize], indices: &ArrayD<usize>) -> Result<Vec<usize>> {
        let rank_diff = to_shape.len() - from_shape.len();
        let mut mapping = Vec::with_capacity(indices.len());
        
        // For each output index, calculate the corresponding input index
        for out_idx in 0..indices.len() {
            // Calculate multi-dimensional index
            let mut remaining = out_idx;
            let mut multi_idx = Vec::with_capacity(to_shape.len());
            
            for &dim in to_shape.iter().rev() {
                multi_idx.push(remaining % dim);
                remaining /= dim;
            }
            
            multi_idx.reverse();
            
            // Calculate input index
            let mut in_idx = 0;
            let mut stride = 1;
            
            for i in (0..from_shape.len()).rev() {
                let out_dim_idx = i + rank_diff;
                let input_dim = from_shape[i];
                let mapped_idx = if input_dim == 1 { 0 } else { multi_idx[out_dim_idx] };
                in_idx += mapped_idx * stride;
                stride *= input_dim;
            }
            
            mapping.push(in_idx);
        }
        
        Ok(mapping)
    }
    
    /// Cast tensor to another data type (legacy method for backwards compatibility)
    pub fn cast_to(&self, target_type: DataType) -> Result<Tensor> {
        self.cast_to_simple(target_type)
    }

/// Element-wise binary operation
pub fn element_wise_binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32 + Copy,
{
    // Verify compatible types
    if a.data_type.is_string() || b.data_type.is_string() {
        return Err(Error::ExecutionError(
            "Element-wise operations not supported for string tensors".to_string()
        ));
    }
    
    // Get broadcast shape
    let output_shape = get_broadcast_shape(&a.shape, &b.shape)?;
    
    // Broadcast tensors if needed
    let a_broadcast = a.broadcast_to(&output_shape)?;
    let b_broadcast = b.broadcast_to(&output_shape)?;
    
    // Handle complex data
    if a.data_type.is_complex() || b.data_type.is_complex() {
        // Determine the output complex type (use the higher precision one)
        let output_type = if a.data_type == DataType::Complex128 || b.data_type == DataType::Complex128 {
            DataType::Complex128
        } else {
            DataType::Complex64
        };
        
        // For complex operations, we need to extract real and imaginary parts
        // Apply operation element-wise on the real and imaginary parts separately
        // This is a simplified approach - for a full implementation, proper complex arithmetic would be needed
        
        // Get real parts (default to regular data if not complex)
        let a_real = match &a_broadcast.complex_data {
            Some((real, _)) => real,
            None => &a_broadcast.data,
        };
        
        let b_real = match &b_broadcast.complex_data {
            Some((real, _)) => real,
            None => &b_broadcast.data,
        };
        
        // Get imaginary parts (default to zeros if not complex)
        let a_imag = match &a_broadcast.complex_data {
            Some((_, imag)) => imag,
            None => &ArrayD::zeros(a_broadcast.data.raw_dim()),
        };
        
        let b_imag = match &b_broadcast.complex_data {
            Some((_, imag)) => imag,
            None => &ArrayD::zeros(b_broadcast.data.raw_dim()),
        };
        
        // Apply operation to real parts
        let result_real = a_real.zip_map(b_real, |x, y| op(x, y));
        
        // Apply operation to imaginary parts 
        let result_imag = a_imag.zip_map(b_imag, |x, y| op(x, y));
        
        // Create result tensor
        return Ok(Tensor {
            name: None,
            data_type: output_type,
            shape: output_shape,
            data: ArrayD::zeros(IxDyn(&output_shape)), // Placeholder
            string_data: None,
            complex_data: Some((result_real, result_imag)),
        });
    }
    
    // Handle regular numeric data
    let result_data = a_broadcast.data.zip_map(&b_broadcast.data, |x, y| op(x, y));
    
    // Determine the output type (usually inherit from first operand)
    let output_type = if a.data_type.is_floating_point() || b.data_type.is_floating_point() {
        // Floating point has precedence over integer
        if a.data_type == DataType::Float64 || b.data_type == DataType::Float64 {
            DataType::Float64
        } else {
            DataType::Float32
        }
    } else {
        // For integer types, we keep the original type
        a.data_type
    };
    
    Ok(Tensor {
        name: None,
        data_type: output_type,
        shape: output_shape,
        data: result_data,
        string_data: None,
        complex_data: None,
    })
}

/// Element-wise unary operation
pub fn element_wise_unary_op<F>(a: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32) -> f32 + Copy,
{
    // Verify compatible types
    if a.data_type.is_string() {
        return Err(Error::ExecutionError(
            "Element-wise operations not supported for string tensors".to_string()
        ));
    }
    
    // Handle complex data
    if a.data_type.is_complex() {
        if let Some((real, imag)) = &a.complex_data {
            // Apply operation to real and imaginary parts
            let result_real = real.mapv(op);
            let result_imag = imag.mapv(op);
            
            return Ok(Tensor {
                name: None,
                data_type: a.data_type,
                shape: a.shape.clone(),
                data: ArrayD::zeros(IxDyn(&a.shape)), // Placeholder
                string_data: None,
                complex_data: Some((result_real, result_imag)),
            });
        } else {
            return Err(Error::ExecutionError(
                "Complex tensor missing complex_data".to_string()
            ));
        }
    }
    
    // Handle regular numeric data
    let result_data = a.data.mapv(op);
    
    Ok(Tensor {
        name: None,
        data_type: a.data_type,
        shape: a.shape.clone(),
        data: result_data,
        string_data: None,
        complex_data: None,
    })
}

/// Transpose a tensor
pub fn transpose(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    let rank = tensor.shape.len();
    
    // If axes not provided, reverse all dimensions
    let axes = match axes {
        Some(axes) => {
            if axes.len() != rank {
                return Err(Error::InvalidModel(format!(
                    "Transpose axes must have the same length as tensor rank. Got {} axes for rank {}",
                    axes.len(), rank
                )));
            }
            axes.to_vec()
        },
        None => (0..rank).rev().collect(),
    };
    
    // Validate axes
    for &axis in &axes {
        if axis >= rank {
            return Err(Error::InvalidModel(format!(
                "Transpose axis {} out of bounds for tensor of rank {}", axis, rank
            )));
        }
    }
    
    // Calculate new shape
    let mut new_shape = vec![0; rank];
    for (i, &axis) in axes.iter().enumerate() {
        new_shape[i] = tensor.shape[axis];
    }
    
    // Handle different data types
    match tensor.data_type {
        DataType::String => {
            if let Some(string_data) = &tensor.string_data {
                // For string tensors, we need to reorder the data manually
                let mut indices = ArrayD::<usize>::zeros(IxDyn(&tensor.shape));
                
                // Initialize with linear indices
                for (idx, val) in indices.iter_mut().enumerate() {
                    *val = idx;
                }
                
                // Transpose the indices
                let transposed_indices = indices.permuted_axes(IxDyn(&axes));
                
                // Create a new string vector with transposed order
                let mut new_string_data = vec![String::new(); string_data.len()];
                for (new_idx, &old_idx) in transposed_indices.iter().enumerate() {
                    if old_idx < string_data.len() {
                        new_string_data[new_idx] = string_data[old_idx].clone();
                    }
                }
                
                Ok(Tensor {
                    name: tensor.name.clone(),
                    data_type: tensor.data_type,
                    shape: new_shape,
                    data: ArrayD::zeros(IxDyn(&new_shape)), // Placeholder
                    string_data: Some(new_string_data),
                    complex_data: None,
                })
            } else {
                Err(Error::ExecutionError("String tensor missing string_data".to_string()))
            }
        },
        DataType::Complex64 | DataType::Complex128 => {
            if let Some((real, imag)) = &tensor.complex_data {
                // Transpose both real and imaginary parts
                let transposed_real = real.permuted_axes(IxDyn(&axes));
                let transposed_imag = imag.permuted_axes(IxDyn(&axes));
                
                Ok(Tensor {
                    name: tensor.name.clone(),
                    data_type: tensor.data_type,
                    shape: new_shape,
                    data: ArrayD::zeros(IxDyn(&new_shape)), // Placeholder
                    string_data: None,
                    complex_data: Some((transposed_real, transposed_imag)),
                })
            } else {
                Err(Error::ExecutionError("Complex tensor missing complex_data".to_string()))
            }
        },
        _ => {
            // Regular numeric data
            let transposed_data = tensor.data.permuted_axes(IxDyn(&axes));
            
            Ok(Tensor {
                name: tensor.name.clone(),
                data_type: tensor.data_type,
                shape: new_shape,
                data: transposed_data,
                string_data: None,
                complex_data: None,
            })
        }
    }
}

/// Concatenate tensors along specified axis
pub fn concatenate(tensors: &[&Tensor], axis: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::InvalidModel("Cannot concatenate empty list of tensors".to_string()));
    }
    
    let rank = tensors[0].shape.len();
    if axis >= rank {
        return Err(Error::InvalidModel(format!(
            "Concatenation axis {} out of bounds for tensor of rank {}", axis, rank
        )));
    }
    
    // Check that all tensors have same rank, compatible shapes, and same data type
    let data_type = tensors[0].data_type;
    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        if tensor.shape.len() != rank {
            return Err(Error::InvalidModel(format!(
                "All tensors must have the same rank for concatenation. Tensor 0 has rank {} but tensor {} has rank {}",
                rank, i, tensor.shape.len()
            )));
        }
        
        if tensor.data_type != data_type {
            return Err(Error::InvalidModel(format!(
                "All tensors must have the same data type for concatenation. Tensor 0 has type {:?} but tensor {} has type {:?}",
                data_type, i, tensor.data_type
            )));
        }
        
        for (dim, (&s1, &s2)) in tensors[0].shape.iter().zip(tensor.shape.iter()).enumerate() {
            if dim != axis && s1 != s2 {
                return Err(Error::InvalidModel(format!(
                    "Incompatible shapes for concatenation: tensors 0 and {} differ in dimension {}",
                    i, dim
                )));
            }
        }
    }
    
    // Calculate output shape
    let mut output_shape = tensors[0].shape.clone();
    output_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();
    
    // Handle different data types
    match data_type {
        DataType::String => {
            // For string tensors, we need to concatenate the string vectors
            let mut result_strings = Vec::new();
            
            for tensor in tensors {
                if let Some(strings) = &tensor.string_data {
                    result_strings.extend_from_slice(strings);
                } else {
                    return Err(Error::ExecutionError("String tensor missing string_data".to_string()));
                }
            }
            
            // Create output tensor
            Ok(Tensor {
                name: None,
                data_type,
                shape: output_shape,
                data: ArrayD::zeros(IxDyn(&output_shape)), // Placeholder
                string_data: Some(result_strings),
                complex_data: None,
            })
        },
        DataType::Complex64 | DataType::Complex128 => {
            // For complex tensors, we need to concatenate both real and imaginary parts
            
            // Calculate total elements in output
            let total_elements: usize = output_shape.iter().product();
            
            // Create output arrays for real and imaginary parts
            let mut result_real = ArrayD::zeros(IxDyn(&output_shape));
            let mut result_imag = ArrayD::zeros(IxDyn(&output_shape));
            
            // Copy data from input tensors
            let mut offset = 0;
            for tensor in tensors {
                if let Some((real, imag)) = &tensor.complex_data {
                    let slice_len = tensor.shape[axis];
                    
                    // Create a slice in the output tensor for this input tensor
                    let mut indices: Vec<_> = (0..rank).map(|_| ndarray::SliceInfo::new()).collect();
                    indices[axis] = ndarray::SliceInfo::<_, ndarray::SliceInfoElem>::from(
                        ndarray::Slice::from(offset..offset+slice_len)
                    );
                    
                    // Copy real part
                    let mut output_real_view = result_real.slice_each_axis_mut(|ax| {
                        indices[ax.axis.index()].clone()
                    });
                    output_real_view.assign(real);
                    
                    // Copy imaginary part
                    let mut output_imag_view = result_imag.slice_each_axis_mut(|ax| {
                        indices[ax.axis.index()].clone()
                    });
                    output_imag_view.assign(imag);
                    
                    offset += slice_len;
                } else {
                    return Err(Error::ExecutionError("Complex tensor missing complex_data".to_string()));
                }
            }
            
            // Create output tensor
            Ok(Tensor {
                name: None,
                data_type,
                shape: output_shape,
                data: ArrayD::zeros(IxDyn(&output_shape)), // Placeholder
                string_data: None,
                complex_data: Some((result_real, result_imag)),
            })
        },
        _ => {
            // Regular numeric data
            
            // Create output tensor
            let mut result = Tensor::new(&output_shape, data_type);
            
            // Copy data from input tensors
            let mut offset = 0;
            for tensor in tensors {
                let slice_len = tensor.shape[axis];
                
                // Create a slice in the output tensor for this input tensor
                let mut indices: Vec<_> = (0..rank).map(|_| ndarray::SliceInfo::new()).collect();
                indices[axis] = ndarray::SliceInfo::<_, ndarray::SliceInfoElem>::from(
                    ndarray::Slice::from(offset..offset+slice_len)
                );
                
                // Copy data
                let mut output_view = result.data.slice_each_axis_mut(|ax| {
                    indices[ax.axis.index()].clone()
                });
                output_view.assign(&tensor.data);
                
                offset += slice_len;
            }
            
            Ok(result)
        }
    }
}

/// Check if a shape can be broadcast to another shape
fn can_broadcast(from_shape: &[usize], to_shape: &[usize]) -> bool {
    if from_shape.len() > to_shape.len() {
        return false;
    }
    
    // Check each dimension for compatibility
    let offset = to_shape.len() - from_shape.len();
    for (i, &dim) in from_shape.iter().enumerate() {
        if dim != 1 && dim != to_shape[i + offset] {
            return false;
        }
    }
    
    true
}

/// Calculate broadcast shape for two tensors
fn get_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let rank1 = shape1.len();
    let rank2 = shape2.len();
    let result_rank = std::cmp::max(rank1, rank2);
    
    let mut result_shape = Vec::with_capacity(result_rank);
    
    for i in 0..result_rank {
        let dim1 = if i >= result_rank - rank1 {
            shape1[i - (result_rank - rank1)]
        } else {
            1
        };
        
        let dim2 = if i >= result_rank - rank2 {
            shape2[i - (result_rank - rank2)]
        } else {
            1
        };
        
        if dim1 == 1 {
            result_shape.push(dim2);
        } else if dim2 == 1 {
            result_shape.push(dim1);
        } else if dim1 == dim2 {
            result_shape.push(dim1);
        } else {
            return Err(Error::InvalidModel(format!(
                "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions at index {}",
                shape1, shape2, i
            )));
        }
    }
    
    Ok(result_shape)
}

/// Convert a single value with conversion options
pub fn convert_value_with_options(value: f64, target_type: DataType, options: &ConversionOptions) -> Result<f32> {
    // Handle special values (NaN, Infinity)
    if value.is_nan() || value.is_infinite() {
        match options.special_value_behavior {
            SpecialValueBehavior::ZeroAndSaturate => {
                if value.is_nan() {
                    return Ok(0.0);
                } else if value.is_infinite() {
                    // Return max or min value based on sign
                    let (min, max) = target_type.get_value_range();
                    return if value.is_sign_positive() {
                        Ok(max as f32)
                    } else {
                        Ok(min as f32)
                    };
                }
            },
            SpecialValueBehavior::Preserve => {
                // For float types, we can preserve special values
                if target_type.is_floating_point() {
                    if value.is_nan() {
                        return Ok(f32::NAN);
                    } else if value.is_infinite() {
                        return if value.is_sign_positive() {
                            Ok(f32::INFINITY)
                        } else {
                            Ok(f32::NEG_INFINITY)
                        };
                    }
                } else {
                    // For non-float types, default to zero
                    return Ok(0.0);
                }
            },
            SpecialValueBehavior::Error => {
                if value.is_nan() {
                    return Err(Error::ExecutionError("Cannot convert NaN value".to_string()));
                } else if value.is_infinite() {
                    return Err(Error::ExecutionError("Cannot convert infinite value".to_string()));
                }
            }
        }
    }
    
    // Apply rounding for floating-point to integer conversions
    let rounded_value = if target_type.is_integer() && !value.fract().is_zero() {
        match options.rounding_strategy {
            RoundingStrategy::Round => value.round(),
            RoundingStrategy::Truncate => value.trunc(),
            RoundingStrategy::Floor => value.floor(),
            RoundingStrategy::Ceiling => value.ceil(),
        }
    } else {
        value
    };
    
    // Check overflow/underflow based on target type
    let (min, max) = target_type.get_value_range();
    
    if rounded_value < min || rounded_value > max {
        match options.overflow_behavior {
            OverflowBehavior::Saturate => {
                if rounded_value < min {
                    Ok(min as f32)
                } else {
                    Ok(max as f32)
                }
            },
            OverflowBehavior::Wrap => {
                // Implement wrapping for integer types
                if target_type.is_integer() {
                    // For signed integers
                    if target_type.is_signed_integer() {
                        match target_type {
                            DataType::Int8 => Ok((rounded_value as i8 as i32) as f32),
                            DataType::Int16 => Ok((rounded_value as i16 as i32) as f32),
                            DataType::Int32 => Ok(rounded_value as i32 as f32),
                            DataType::Int64 => Ok((rounded_value as i64) as f32),
                            _ => unreachable!()
                        }
                    } 
                    // For unsigned integers
                    else {
                        match target_type {
                            DataType::Uint8 => Ok((rounded_value as u8) as f32),
                            DataType::Uint16 => Ok((rounded_value as u16) as f32),
                            DataType::Uint32 => Ok((rounded_value as u32) as f32),
                            DataType::Uint64 => Ok((rounded_value as u64) as f32),
                            _ => unreachable!()
                        }
                    }
                } else {
                    // No wrapping for floating point
                    Ok(rounded_value as f32)
                }
            },
            OverflowBehavior::Error => {
                Err(Error::ExecutionError(format!(
                    "Value {} is outside the valid range [{}, {}] for type {:?}",
                    value, min, max, target_type
                )))
            }
        }
    } else {
        // Value is in range, just convert
        Ok(rounded_value as f32)
    }
}

/// Convert a NumCast array to f32 array with options
pub fn convert_array_with_options<T: NumCast + Copy>(
    arr: ArrayD<T>, 
    options: ConversionOptions
) -> Result<ArrayD<f32>> {
    let mut result = ArrayD::zeros(arr.raw_dim());
    
    for (i, &val) in arr.iter().enumerate() {
        if let Some(float_val) = NumCast::from(val) {
            // We convert to f64 for maximum precision in intermediate calculations
            let converted = convert_value_with_options(float_val, DataType::Float32, &options)?;
            result.as_slice_mut().unwrap()[i] = converted;
        } else {
            result.as_slice_mut().unwrap()[i] = 0.0;
        }
    }
    
    Ok(result)
}

/// Convert array to target data type with numeric options
pub fn convert_array_with_numeric_options(
    arr: &ArrayD<f32>, 
    target_type: DataType,
    options: &ConversionOptions
) -> Result<ArrayD<f32>> {
    let mut result = ArrayD::zeros(arr.raw_dim());
    
    for (i, &val) in arr.iter().enumerate() {
        let converted = convert_value_with_options(val as f64, target_type, options)?;
        result.as_slice_mut().unwrap()[i] = converted;
    }
    
    Ok(result)
}

/// Convert from f32 array to array of type T with options
pub fn convert_from_array_with_options<T: NumCast + Copy>(
    arr: &ArrayD<f32>,
    options: ConversionOptions
) -> Result<ArrayD<T>> {
    let mut result_vec = Vec::with_capacity(arr.len());
    
    for &val in arr.iter() {
        // For non-f32 target types, we might need to apply conversion rules
        let target_val = match NumCast::from(val) {
            Some(v) => v,
            None => {
                // Default to zero for failed conversions
                match NumCast::from(0.0) {
                    Some(zero) => zero,
                    None => return Err(Error::ExecutionError(
                        "Failed to convert tensor to target type".to_string()
                    )),
                }
            }
        };
        
        result_vec.push(target_val);
    }
    
    ArrayD::from_shape_vec(arr.raw_dim(), result_vec)
        .map_err(|e| Error::ExecutionError(format!("Error creating array: {}", e)))
}

/// Recursive function to copy data with broadcasting
fn broadcast_copy(
    src: &ndarray::ArrayView<f32, IxDyn>,
    dst: &mut ndarray::ArrayViewMut<f32, IxDyn>,
) -> Result<()> {
    if src.shape() == dst.shape() {
        // Direct copy if shapes match
        dst.assign(src);
        return Ok(());
    }
    
    // Handle broadcasting through a comprehensive implementation
    if src.ndim() == 0 {
        // Scalar broadcasting
        dst.fill(src.first().unwrap_or(&0.0).clone());
    } else if src.ndim() < dst.ndim() {
        // First handle dimension mismatch by effectively padding with 1s at front
        let dim_diff = dst.ndim() - src.ndim();
        
        // We need to broadcast the source array to all slices of the destination
        for i in 0..dst.shape()[0] {
            let mut dst_subview = dst.slice_mut(ndarray::s![i, ..]);
            
            if dim_diff == 1 {
                // If difference is 1, we can broadcast the source directly
                broadcast_copy(&src.view(), &mut dst_subview)?;
            } else {
                // Otherwise, we need to recursively broadcast
                // First, create a view with the same number of dimensions
                let mut new_shape = vec![1; dst.ndim()];
                for (i, &dim) in src.shape().iter().enumerate() {
                    new_shape[i + dim_diff] = dim;
                }
                
                // Create a reshaped view of the source array
                let reshaped_src = match src.clone().into_shape(IxDyn(&new_shape)) {
                    Ok(arr) => arr,
                    Err(_) => return Err(Error::InvalidModel(
                        "Failed to reshape array for broadcasting".to_string()
                    )),
                };
                
                // Now broadcast this reshaped view
                broadcast_copy(&reshaped_src.view(), &mut dst_subview)?;
            }
        }
    } else {
        // Dimensions match, but some dimensions might be 1 in the source
        assert_eq!(src.ndim(), dst.ndim());
        
        // Find the first dimension where src has size 1 but dst has size > 1
        let mut broadcast_dim = None;
        for (i, (&s, &d)) in src.shape().iter().zip(dst.shape().iter()).enumerate() {
            if s == 1 && d > 1 {
                broadcast_dim = Some(i);
                break;
            }
        }
        
        match broadcast_dim {
            Some(dim) => {
                // We need to broadcast along this dimension
                let src_slice = src.slice(ndarray::s![0, ..]);
                
                for i in 0..dst.shape()[dim] {
                    // Create a slice for each index in the broadcast dimension
                    let mut index = vec![ndarray::SliceInfoElem::from(ndarray::SliceInfo::new())];
                    index[dim] = ndarray::SliceInfoElem::from(ndarray::Slice::from(i..i+1));
                    
                    let mut dst_slice = dst.slice_each_axis_mut(|ax| {
                        if ax.axis.index() == dim {
                            ndarray::SliceInfoElem::from(ndarray::Slice::from(i..i+1))
                        } else {
                            ndarray::SliceInfoElem::from(ndarray::SliceInfo::new())
                        }
                    });
                    
                    broadcast_copy(&src_slice, &mut dst_slice)?;
                }
            },
            None => {
                // If we get here, shapes should be the same or we have an error
                if src.shape() == dst.shape() {
                    dst.assign(src);
                } else {
                    return Err(Error::InvalidModel(format!(
                        "Cannot broadcast shapes {:?} to {:?}", 
                        src.shape(), dst.shape()
                    )));
                }
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};
    use half::{f16, bf16};
    use std::f32;
    use std::f64;
    
    #[test]
    fn test_numeric_type_conversions() {
        // Test integer conversions
        let options = ConversionOptions {
            overflow_behavior: OverflowBehavior::Saturate,
            special_value_behavior: SpecialValueBehavior::ZeroAndSaturate,
            rounding_strategy: RoundingStrategy::Round,
        };
        
        // Float to int with rounding
        assert_eq!(
            convert_value_with_options(3.7, DataType::Int32, &options).unwrap(),
            4.0 // Rounded up
        );
        
        assert_eq!(
            convert_value_with_options(3.2, DataType::Int32, &options).unwrap(),
            3.0 // Rounded down
        );
        
        // Test overflow with saturation
        assert_eq!(
            convert_value_with_options(300.0, DataType::Int8, &options).unwrap(),
            127.0 // Saturated to i8::MAX
        );
        
        assert_eq!(
            convert_value_with_options(-300.0, DataType::Int8, &options).unwrap(),
            -128.0 // Saturated to i8::MIN
        );
        
        // Test special values
        assert_eq!(
            convert_value_with_options(f64::NAN, DataType::Int32, &options).unwrap(),
            0.0 // NaN converted to 0
        );
        
        assert_eq!(
            convert_value_with_options(f64::INFINITY, DataType::Int32, &options).unwrap(),
            DataType::Int32.get_value_range().1 as f32 // Infinity converted to max
        );
        
        // Test with error behavior
        let error_options = ConversionOptions {
            overflow_behavior: OverflowBehavior::Error,
            special_value_behavior: SpecialValueBehavior::Error,
            rounding_strategy: RoundingStrategy::Round,
        };
        
        // Should error on overflow
        assert!(convert_value_with_options(300.0, DataType::Int8, &error_options).is_err());
        
        // Should error on NaN
        assert!(convert_value_with_options(f64::NAN, DataType::Int32, &error_options).is_err());
    }
    
    #[test]
    fn test_tensor_type_conversions() {
        // Create a tensor with float data
        let data = vec![1.5, 2.7, -3.2, 4.0, 5.9];
        let shape = vec![5];
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data.clone()).unwrap();
        
        // Create a tensor
        let tensor = Tensor::from_ndarray_simple(array, DataType::Float32).unwrap();
        
        // Test casting to different types
        
        // Float32 to Int32 with rounding
        let int_tensor = tensor.cast_to_simple(DataType::Int32).unwrap();
        assert_eq!(int_tensor.data_type, DataType::Int32);
        
        // Check values were rounded correctly
        let expected_ints = vec![2.0, 3.0, -3.0, 4.0, 6.0];
        for (i, &expected) in expected_ints.iter().enumerate() {
            assert_eq!(int_tensor.data.as_slice().unwrap()[i], expected);
        }
        
        // Test string conversion
        let string_tensor = tensor.cast_to_simple(DataType::String).unwrap();
        assert_eq!(string_tensor.data_type, DataType::String);
        assert!(string_tensor.has_string_data());
        
        // Check string values
        if let Some(strings) = &string_tensor.string_data {
            assert_eq!(strings[0], "1.5");
            assert_eq!(strings[1], "2.7");
            assert_eq!(strings[2], "-3.2");
        } else {
            panic!("String data not found");
        }
        
        // Test complex number conversion
        let complex_tensor = tensor.cast_to_simple(DataType::Complex64).unwrap();
        assert_eq!(complex_tensor.data_type, DataType::Complex64);
        assert!(complex_tensor.has_complex_data());
        
        // Check real part contains original values and imaginary part is zeros
        if let Some((real, imag)) = &complex_tensor.complex_data {
            for (i, &val) in data.iter().enumerate() {
                assert_eq!(real.as_slice().unwrap()[i], val as f32);
                assert_eq!(imag.as_slice().unwrap()[i], 0.0);
            }
        } else {
            panic!("Complex data not found");
        }
    }
    
    #[test]
    fn test_raw_data_conversions() {
        // Test different raw data formats
        
        // Float32
        let float_data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3F,  // 1.0 in IEEE-754
            0x00, 0x00, 0x00, 0x40,  // 2.0 in IEEE-754
        ];
        
        let tensor = Tensor::from_raw_data(&float_data, &[2], DataType::Float32).unwrap();
        assert_eq!(tensor.data.as_slice().unwrap()[0], 1.0);
        assert_eq!(tensor.data.as_slice().unwrap()[1], 2.0);
        
        // Int8
        let int8_data: Vec<u8> = vec![10, 20, 30];
        let tensor = Tensor::from_raw_data(&int8_data, &[3], DataType::Int8).unwrap();
        assert_eq!(tensor.data.as_slice().unwrap()[0], 10.0);
        assert_eq!(tensor.data.as_slice().unwrap()[1], 20.0);
        assert_eq!(tensor.data.as_slice().unwrap()[2], 30.0);
        
        // Bool
        let bool_data: Vec<u8> = vec![1, 0, 1];
        let tensor = Tensor::from_raw_data(&bool_data, &[3], DataType::Bool).unwrap();
        assert_eq!(tensor.data.as_slice().unwrap()[0], 1.0);
        assert_eq!(tensor.data.as_slice().unwrap()[1], 0.0);
        assert_eq!(tensor.data.as_slice().unwrap()[2], 1.0);
    }
    
    #[test]
    fn test_broadcasting() {
        // Create a tensor with shape [1, 3]
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![1, 3];
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
        let tensor = Tensor::from_ndarray_simple(array, DataType::Float32).unwrap();
        
        // Broadcast to [2, 3]
        let broadcasted = tensor.broadcast_to(&[2, 3]).unwrap();
        
        // Check that values were correctly broadcasted
        assert_eq!(broadcasted.shape, vec![2, 3]);
        
        // First row
        assert_eq!(broadcasted.data[[0, 0]], 1.0);
        assert_eq!(broadcasted.data[[0, 1]], 2.0);
        assert_eq!(broadcasted.data[[0, 2]], 3.0);
        
        // Second row (broadcasted)
        assert_eq!(broadcasted.data[[1, 0]], 1.0);
        assert_eq!(broadcasted.data[[1, 1]], 2.0);
        assert_eq!(broadcasted.data[[1, 2]], 3.0);
    }
    
    #[test]
    fn test_complex_operations() {
        // Create a complex tensor
        let mut tensor = Tensor::new(&[2], DataType::Complex64);
        
        // Set complex values directly
        if let Some((real, imag)) = &mut tensor.complex_data {
            real.as_slice_mut().unwrap()[0] = 1.0;
            real.as_slice_mut().unwrap()[1] = 2.0;
            imag.as_slice_mut().unwrap()[0] = 3.0;
            imag.as_slice_mut().unwrap()[1] = 4.0;
        }
        
        // Test conversion to float (should keep only real part)
        let float_tensor = tensor.cast_to_simple(DataType::Float32).unwrap();
        assert_eq!(float_tensor.data.as_slice().unwrap()[0], 1.0);
        assert_eq!(float_tensor.data.as_slice().unwrap()[1], 2.0);
    }
    
    #[test]
    fn test_string_operations() {
        // Create a string tensor
        let mut tensor = Tensor::new(&[3], DataType::String);
        
        // Set string values directly
        if let Some(strings) = &mut tensor.string_data {
            strings[0] = "hello".to_string();
            strings[1] = "world".to_string();
            strings[2] = "123".to_string();
        }
        
        // Test conversion to float (non-numeric strings should be 0.0)
        let float_tensor = tensor.cast_to_simple(DataType::Float32).unwrap();
        assert_eq!(float_tensor.data.as_slice().unwrap()[0], 0.0); // "hello" -> 0.0
        assert_eq!(float_tensor.data.as_slice().unwrap()[1], 0.0); // "world" -> 0.0
        assert_eq!(float_tensor.data.as_slice().unwrap()[2], 123.0); // "123" -> 123.0
    }
}