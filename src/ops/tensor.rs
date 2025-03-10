use std::fmt;
use ndarray::{Array, ArrayD, IxDyn, Dimension, Axis};
use num_traits::NumCast;
use crate::error::{Error, Result};

/// Shape of a tensor
pub type Shape = Vec<usize>;

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
        }
    }
}

/// Tensor struct for runtime computation
#[derive(Clone)]
pub struct Tensor {
    pub name: Option<String>,
    pub data_type: DataType,
    pub shape: Shape,
    // Store data as f32 for internal computation
    pub data: ArrayD<f32>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{ name: {:?}, data_type: {:?}, shape: {:?}, data: [shape: {:?}] }}",
               self.name, self.data_type, self.shape, self.data.shape())
    }
}

impl Tensor {
    /// Create a new tensor with empty data
    pub fn new(shape: &[usize], data_type: DataType) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self {
            name: None,
            data_type,
            shape: shape.to_vec(),
            data,
        }
    }
    
    /// Create a tensor from ndarray
    pub fn from_ndarray<T: NumCast>(arr: ArrayD<T>, data_type: DataType) -> Result<Self> {
        let shape = arr.shape().to_vec();
        // Convert the array to f32
        let data = arr.mapv(|x| {
            NumCast::from(x).unwrap_or(0.0)
        });
        
        Ok(Self {
            name: None,
            data_type,
            shape,
            data,
        })
    }
    
    /// Convert tensor to ndarray of specific type
    pub fn to_ndarray<T: NumCast>(&self) -> Result<ArrayD<T>> {
        let result = self.data.mapv(|x| {
            match NumCast::from(x) {
                Some(val) => val,
                None => NumCast::from(0.0).unwrap(),
            }
        });
        
        Ok(result)
    }
    
    /// Create a tensor from raw data
    pub fn from_raw_data(data: &[u8], shape: &[usize], data_type: DataType) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        let expected_bytes = total_elements * data_type.size_in_bytes();
        
        if data.len() != expected_bytes {
            return Err(Error::InvalidModel(format!(
                "Tensor data size mismatch. Expected {} bytes but got {}",
                expected_bytes, data.len()
            )));
        }
        
        // Create array and convert to f32
        let data_array = match data_type {
            DataType::Float32 => {
                let mut float_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let value = f32::from_le_bytes(bytes);
                    float_data.push(value);
                }
                ArrayD::from_shape_vec(IxDyn(shape), float_data)?
            },
            DataType::Int32 => {
                let mut int_data = Vec::with_capacity(total_elements);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let value = i32::from_le_bytes(bytes);
                    int_data.push(value as f32);
                }
                ArrayD::from_shape_vec(IxDyn(shape), int_data)?
            },
            // Handle other data types similarly...
            _ => return Err(Error::UnsupportedFeature(format!(
                "Conversion from {:?} raw data not implemented yet", data_type
            ))),
        };
        
        Ok(Self {
            name: None,
            data_type,
            shape: shape.to_vec(),
            data: data_array,
        })
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
        
        let reshaped_data = match self.data.clone().into_shape(IxDyn(shape)) {
            Ok(array) => array,
            Err(e) => return Err(Error::InvalidModel(format!(
                "Reshape error: {}", e
            ))),
        };
        
        Ok(Tensor {
            name: self.name.clone(),
            data_type: self.data_type,
            shape: shape.to_vec(),
            data: reshaped_data,
        })
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
        
        // Create output array and perform the broadcast
        let mut output = ArrayD::<f32>::zeros(IxDyn(shape));
        
        // This is a simplified broadcasting implementation
        // A full implementation would need to handle all broadcasting rules
        broadcast_copy(&self.data.view(), &mut output.view_mut())?;
        
        Ok(Tensor {
            name: self.name.clone(),
            data_type: self.data_type,
            shape: shape.to_vec(),
            data: output,
        })
    }
    
    /// Cast tensor to another data type
    pub fn cast_to(&self, target_type: DataType) -> Result<Tensor> {
        if self.data_type == target_type {
            return Ok(self.clone());
        }
        
        // Just change the type for now - data is already stored as f32
        Ok(Tensor {
            name: self.name.clone(),
            data_type: target_type,
            shape: self.shape.clone(),
            data: self.data.clone(),
        })
    }
}

/// Element-wise binary operation
pub fn element_wise_binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32 + Copy,
{
    // Get broadcast shape
    let output_shape = get_broadcast_shape(&a.shape, &b.shape)?;
    
    // Broadcast tensors if needed
    let a_broadcast = a.broadcast_to(&output_shape)?;
    let b_broadcast = b.broadcast_to(&output_shape)?;
    
    // Perform operation
    let result_data = a_broadcast.data.mapv(|x| x);
    let result_data = a_broadcast.data.zip_map(&b_broadcast.data, |x, y| op(x, y));
    
    Ok(Tensor {
        name: None,
        data_type: a.data_type,
        shape: output_shape,
        data: result_data,
    })
}

/// Element-wise unary operation
pub fn element_wise_unary_op<F>(a: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32) -> f32 + Copy,
{
    let result_data = a.data.mapv(op);
    
    Ok(Tensor {
        name: None,
        data_type: a.data_type,
        shape: a.shape.clone(),
        data: result_data,
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
    
    // Create permutation array
    let mut permutation = Vec::with_capacity(rank);
    for &axis in &axes {
        permutation.push(Axis(axis));
    }
    
    // Perform transpose
    let transposed_data = tensor.data.permuted_axes(IxDyn(&axes));
    
    // Calculate new shape
    let mut new_shape = vec![0; rank];
    for (i, &axis) in axes.iter().enumerate() {
        new_shape[i] = tensor.shape[axis];
    }
    
    Ok(Tensor {
        name: tensor.name.clone(),
        data_type: tensor.data_type,
        shape: new_shape,
        data: transposed_data,
    })
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
    
    // Check that all tensors have same rank and compatible shapes
    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        if tensor.shape.len() != rank {
            return Err(Error::InvalidModel(format!(
                "All tensors must have the same rank for concatenation. Tensor 0 has rank {} but tensor {} has rank {}",
                rank, i, tensor.shape.len()
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
    
    // Create output tensor
    let mut result = Tensor::new(&output_shape, tensors[0].data_type);
    
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
    
    // Implement broadcasting using recursion
    // This is a simplified version; a complete implementation would be more complex
    if src.ndim() == 0 {
        // Scalar broadcasting
        dst.fill(src.first().unwrap_or(&0.0).clone());
    } else if src.ndim() < dst.ndim() {
        // First handle dimension mismatch by effectively padding with 1s at front
        let dim_diff = dst.ndim() - src.ndim();
        
        for i in 0..dst.shape()[0] {
            let mut dst_subview = dst.slice_mut(ndarray::s![i, ..]);
            let src_view = if dim_diff == 1 {
                src.view()
            } else {
                broadcast_copy(src, &mut dst_subview)?;
                continue;
            };
            
            broadcast_copy(&src_view, &mut dst_subview)?;
        }
    } else {
        // Handle dimension by dimension
        for i in 0..dst.ndim() {
            if src.shape()[i] == 1 && dst.shape()[i] > 1 {
                // Broadcast this dimension
                let src_subview = src.slice(ndarray::s![0, ..]);
                for j in 0..dst.shape()[i] {
                    let mut dst_subview = dst.slice_mut(ndarray::s![j, ..]);
                    broadcast_copy(&src_subview, &mut dst_subview)?;
                }
                return Ok(());
            }
        }
        
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
    
    Ok(())
}