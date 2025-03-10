pub mod registry;
pub mod tensor;
pub mod math;
pub mod nn;
pub mod activations;

pub mod prelude {
    pub use super::registry::{Operator, OperatorRegistry};
    pub use super::tensor::{Tensor, Shape, DataType};
}

pub use registry::{Operator, OperatorRegistry};
pub use tensor::{Tensor, Shape, DataType};

// Module files for math subdirectory
pub mod math {
    pub mod matmul;
    pub mod gemm;
}

// Module files for nn subdirectory
pub mod nn {
    pub mod conv;
    pub mod pool;
}