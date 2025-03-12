use std::fmt;

use crate::error::{Error, Result};

/// A tensor layout defines the shape, strides, and offset of a tensor in memory.
/// It describes how the logical indices of a tensor are mapped to physical memory locations.
#[derive(Clone, PartialEq, Eq)]
pub struct TensorLayout {
    /// The shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// The strides of the tensor (number of elements to skip in each dimension)
    strides: Vec<usize>,
    /// The offset from the start of the memory buffer
    offset: usize,
}

impl TensorLayout {
    /// Create a new tensor layout with the given shape, strides, and offset.
    /// If strides are not provided, defaults to C-contiguous (row-major) layout.
    pub fn new(shape: &[usize], strides: Option<&[usize]>, offset: usize) -> Self {
        let strides = if let Some(strides) = strides {
            assert_eq!(shape.len(), strides.len(), "Shape and strides must have the same length");
            strides.to_vec()
        } else {
            Self::compute_contiguous_strides(shape)
        };

        Self {
            shape: shape.to_vec(),
            strides,
            offset,
        }
    }

    /// Create a standard contiguous (row-major) layout for the given shape.
    pub fn contiguous_layout(shape: &[usize]) -> Self {
        let strides = Self::compute_contiguous_strides(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Create a transposed layout by permuting dimensions according to the given axes.
    pub fn transposed_layout(layout: &TensorLayout, axes: &[usize]) -> Result<Self> {
        // Validate axes
        if axes.len() != layout.shape.len() {
            return Err(Error::InvalidModel(format!(
                "Transpose axes must have the same length as tensor rank. Got {} axes for rank {}",
                axes.len(),
                layout.shape.len()
            )));
        }

        // Check axes for validity and uniqueness
        let mut seen = vec![false; axes.len()];
        for &axis in axes {
            if axis >= layout.shape.len() {
                return Err(Error::InvalidModel(format!(
                    "Transpose axis {} out of bounds for tensor of rank {}",
                    axis,
                    layout.shape.len()
                )));
            }
            if seen[axis] {
                return Err(Error::InvalidModel(format!(
                    "Duplicate axis {} in transpose axes",
                    axis
                )));
            }
            seen[axis] = true;
        }

        // Create the new shape and strides
        let mut new_shape = vec![0; layout.shape.len()];
        let mut new_strides = vec![0; layout.strides.len()];

        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = layout.shape[axis];
            new_strides[i] = layout.strides[axis];
        }

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: layout.offset,
        })
    }

    /// Create a broadcasted layout that expands the given layout to the target shape.
    /// Broadcasting follows numpy/ONNX broadcasting rules.
    pub fn broadcasted_layout(layout: &TensorLayout, target_shape: &[usize]) -> Result<Self> {
        // Check if broadcasting is possible
        if !Self::can_broadcast(&layout.shape, target_shape) {
            return Err(Error::InvalidModel(format!(
                "Cannot broadcast tensor of shape {:?} to {:?}",
                layout.shape, target_shape
            )));
        }

        // Compute the number of dimensions to prepend
        let prepend_dims = target_shape.len() - layout.shape.len();

        // Create new shape and strides
        let mut new_shape = vec![0; target_shape.len()];
        let mut new_strides = vec![0; target_shape.len()];

        // Fill in the prepended dimensions
        for i in 0..prepend_dims {
            new_shape[i] = target_shape[i];
            new_strides[i] = 0; // Stride is 0 for broadcasted dimensions
        }

        // Fill in the existing dimensions
        for i in 0..layout.shape.len() {
            let target_dim = i + prepend_dims;
            new_shape[target_dim] = target_shape[target_dim];

            if layout.shape[i] == 1 && target_shape[target_dim] > 1 {
                // Broadcasting from size 1 to larger size
                new_strides[target_dim] = 0;
            } else {
                // Keep the original stride
                new_strides[target_dim] = layout.strides[i];
            }
        }

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: layout.offset,
        })
    }

    /// Calculate contiguous strides for a given shape in row-major (C-style) order.
    /// e.g., for shape [2, 3, 4], strides would be [12, 4, 1]
    fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Calculate Fortran-style (column-major) contiguous strides.
    /// e.g., for shape [2, 3, 4], strides would be [1, 2, 6]
    fn compute_fortran_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in 1..shape.len() {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
        strides
    }

    /// Check if this layout is contiguous (elements are stored without gaps).
    pub fn is_contiguous(&self) -> bool {
        // Empty tensor is considered contiguous
        if self.shape.is_empty() {
            return true;
        }

        // Check if C-contiguous
        let c_strides = Self::compute_contiguous_strides(&self.shape);
        let is_c_contiguous = self.strides == c_strides && self.offset == 0;
        if is_c_contiguous {
            return true;
        }

        // Check if F-contiguous (column-major)
        let f_strides = Self::compute_fortran_strides(&self.shape);
        let is_f_contiguous = self.strides == f_strides && self.offset == 0;
        
        is_f_contiguous
    }

    /// Calculate the total size in bytes, accounting for strides.
    pub fn size_in_bytes(&self, element_size: usize) -> usize {
        if self.shape.is_empty() {
            return 0;
        }

        // Calculate the maximum index
        let mut max_index = self.offset;
        for (dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if *dim > 0 {
                max_index += (*dim - 1) * stride;
            }
        }

        // Add 1 to get the total number of elements needed
        (max_index + 1) * element_size
    }

    /// Calculate the linear index in memory for the given multi-dimensional indices.
    pub fn index_of(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Number of indices must match tensor rank");
        }

        let mut index = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                panic!("Index out of bounds");
            }
            index += idx * self.strides[i];
        }
        index
    }

    /// Check if input layout can be used for in-place operation with output layout.
    pub fn can_use_inplace(input_layout: &TensorLayout, output_layout: &TensorLayout) -> bool {
        // For in-place operation:
        // 1. Shapes must be identical or broadcastable
        // 2. Memory layout must allow elements to be written without overlap
        
        // Check if shapes are identical
        if input_layout.shape == output_layout.shape {
            // If strides and offset are identical, can use in-place
            return input_layout.strides == output_layout.strides 
                && input_layout.offset == output_layout.offset;
        }
        
        // More complex cases like broadcasting would need careful analysis
        // For simplicity, we'll be conservative and only allow exact matches
        false
    }

    /// Optimize tensor layouts for a specific operation type.
    pub fn optimize_for_operation(
        input_layouts: &[&TensorLayout], 
        op_type: &str
    ) -> Vec<TensorLayout> {
        // This is a placeholder for more sophisticated layout optimization.
        // In a real implementation, this would:
        // 1. Analyze the operation type and input layouts
        // 2. Determine optimal layouts for the operation
        // 3. Return optimized layouts for inputs and outputs
        
        match op_type {
            // For element-wise operations, we want aligned and contiguous layouts if possible
            "Add" | "Mul" | "Sub" | "Div" | "Relu" | "Sigmoid" => {
                // For element-wise ops, ensure all inputs have the same layout if possible
                if input_layouts.len() >= 2 {
                    let max_rank = input_layouts.iter()
                        .map(|layout| layout.shape.len())
                        .max()
                        .unwrap_or(0);
                    
                    // Try to make all tensors share the same layout
                    let mut result = Vec::new();
                    for &layout in input_layouts {
                        // If the layout is already optimal, keep it
                        if layout.is_contiguous() && layout.shape.len() == max_rank {
                            result.push(layout.clone());
                        } else {
                            // Otherwise, create a new contiguous layout matching the max rank
                            let mut shape = vec![1; max_rank];
                            let offset = layout.shape.len();
                            for i in 0..layout.shape.len() {
                                shape[max_rank - offset + i] = layout.shape[i];
                            }
                            result.push(Self::contiguous_layout(&shape));
                        }
                    }
                    result
                } else {
                    // Just one input, return contiguous layout
                    input_layouts.iter().map(|&l| {
                        if l.is_contiguous() {
                            l.clone()
                        } else {
                            Self::contiguous_layout(&l.shape)
                        }
                    }).collect()
                }
            },
            
            // For matrix multiplication, we might want column-major layouts for one input
            "MatMul" | "Gemm" => {
                if input_layouts.len() == 2 {
                    let mut result = Vec::new();
                    
                    // A × B: Keep A row-major, make B column-major
                    // Keep first input (A) as is if contiguous
                    if input_layouts[0].is_contiguous() {
                        result.push(input_layouts[0].clone());
                    } else {
                        result.push(Self::contiguous_layout(&input_layouts[0].shape));
                    }
                    
                    // For B, column-major would be more efficient for some BLAS implementations
                    let b_shape = &input_layouts[1].shape;
                    let mut b_strides = Self::compute_fortran_strides(b_shape);
                    result.push(TensorLayout {
                        shape: b_shape.clone(),
                        strides: b_strides,
                        offset: 0,
                    });
                    
                    result
                } else {
                    // Default to contiguous layouts
                    input_layouts.iter().map(|&l| Self::contiguous_layout(&l.shape)).collect()
                }
            },
            
            // Convolution: align filters for efficient SIMD
            "Conv" => {
                if input_layouts.len() >= 2 {
                    let mut result = Vec::new();
                    
                    // Input tensor: keep as is if contiguous
                    if input_layouts[0].is_contiguous() {
                        result.push(input_layouts[0].clone());
                    } else {
                        result.push(Self::contiguous_layout(&input_layouts[0].shape));
                    }
                    
                    // Filter tensor: align to cache lines if possible
                    if input_layouts[1].is_contiguous() {
                        result.push(input_layouts[1].clone());
                    } else {
                        // For filters, we want the output channel dimension to be outermost
                        // for efficient SIMD operations
                        result.push(Self::contiguous_layout(&input_layouts[1].shape));
                    }
                    
                    result
                } else {
                    // Default to contiguous layouts
                    input_layouts.iter().map(|&l| Self::contiguous_layout(&l.shape)).collect()
                }
            },
            
            // Default: use contiguous layouts
            _ => input_layouts.iter().map(|&l| Self::contiguous_layout(&l.shape)).collect(),
        }
    }

    /// Check if a shape can be broadcast to a target shape.
    fn can_broadcast(from_shape: &[usize], to_shape: &[usize]) -> bool {
        // Can't broadcast to fewer dimensions
        if from_shape.len() > to_shape.len() {
            return false;
        }

        // Check each dimension starting from the end
        let offset = to_shape.len() - from_shape.len();
        for (i, &dim) in from_shape.iter().enumerate() {
            let target_dim = to_shape[i + offset];
            // A dimension can be broadcast if it's 1 or equal to the target
            if dim != 1 && dim != target_dim {
                return false;
            }
        }

        true
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the strides of the tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Return the offset of the tensor.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is scalar (0-dimensional).
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Check if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }

    /// Create a view with a new offset.
    pub fn with_offset(&self, offset: usize) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset,
        }
    }

    /// Create a view with a subset of dimensions, keeping the specified axes.
    pub fn keep_dims(&self, axes: &[usize]) -> Result<Self> {
        let rank = self.shape.len();
        
        // Validate axes
        for &axis in axes {
            if axis >= rank {
                return Err(Error::InvalidModel(format!(
                    "Axis {} is out of bounds for tensor of rank {}",
                    axis, rank
                )));
            }
        }
        
        // Create new shape and strides
        let mut new_shape = Vec::with_capacity(axes.len());
        let mut new_strides = Vec::with_capacity(axes.len());
        
        for &axis in axes {
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }
        
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
    
    /// Create a view with specific dimensions squeezed out (removed).
    pub fn squeeze(&self, axes: &[usize]) -> Result<Self> {
        let rank = self.shape.len();
        
        // Validate axes
        for &axis in axes {
            if axis >= rank {
                return Err(Error::InvalidModel(format!(
                    "Axis {} is out of bounds for tensor of rank {}",
                    axis, rank
                )));
            }
            
            if self.shape[axis] != 1 {
                return Err(Error::InvalidModel(format!(
                    "Cannot squeeze dimension {} with size {}",
                    axis, self.shape[axis]
                )));
            }
        }
        
        // Create new shape and strides
        let mut new_shape = Vec::with_capacity(rank - axes.len());
        let mut new_strides = Vec::with_capacity(rank - axes.len());
        
        for i in 0..rank {
            if !axes.contains(&i) {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }
        
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
    
    /// Create a view with extra dimensions of size 1 inserted.
    pub fn unsqueeze(&self, axes: &[usize]) -> Result<Self> {
        let new_rank = self.shape.len() + axes.len();
        
        // Validate axes
        for &axis in axes {
            if axis > new_rank {
                return Err(Error::InvalidModel(format!(
                    "Axis {} is out of bounds for new tensor of rank {}",
                    axis, new_rank
                )));
            }
        }
        
        // Create new shape and strides
        let mut new_shape = vec![0; new_rank];
        let mut new_strides = vec![0; new_rank];
        
        let mut src_idx = 0;
        for dst_idx in 0..new_rank {
            if axes.contains(&dst_idx) {
                // Insert a dimension of size 1
                new_shape[dst_idx] = 1;
                // Stride doesn't matter for dim 1, use 0
                new_strides[dst_idx] = 0;
            } else {
                // Copy from original tensor
                new_shape[dst_idx] = self.shape[src_idx];
                new_strides[dst_idx] = self.strides[src_idx];
                src_idx += 1;
            }
        }
        
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
}

impl fmt::Debug for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorLayout {{ shape: {:?}, strides: {:?}, offset: {} }}",
            self.shape, self.strides, self.offset
        )
    }
}