use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::model::{OnnxModel, Graph, Node, OpSchema, DataType, FormalParameter, AttributeProto, AttributeType};

/// ONNX schema validator responsible for validating model against operator schemas
pub struct SchemaValidator {
    // Registry of operator schemas
    schemas: HashMap<String, HashMap<i64, OpSchema>>,
    
    // Minimum supported IR version
    min_ir_version: i64,
    
    // Maximum supported IR version
    max_ir_version: i64,
}

impl SchemaValidator {
    /// Create a new schema validator with defaults
    pub fn new() -> Self {
        let mut validator = Self {
            schemas: HashMap::new(),
            min_ir_version: 3, // ONNX IR version 3
            max_ir_version: 8, // ONNX IR version 8
        };
        
        validator.register_default_schemas();
        validator
    }
    
    /// Register built-in operator schemas
    fn register_default_schemas(&mut self) {
        // In a real implementation, this would load schemas from an
        // operator registry. Here we'll just implement a few common ones as examples.
        
        // Register Conv schema
        self.register_schema(Self::create_conv_schema());
        
        // Register MatMul schema
        self.register_schema(Self::create_matmul_schema());
        
        // Register Relu schema
        self.register_schema(Self::create_relu_schema());
        
        // More operators would be registered here...
    }
    
    /// Register a schema in the registry
    fn register_schema(&mut self, schema: OpSchema) {
        let domain_map = self.schemas
            .entry(schema.domain.clone())
            .or_insert_with(HashMap::new);
        
        domain_map.insert(schema.since_version, schema);
    }
    
    /// Example schema for Conv operator
    fn create_conv_schema() -> OpSchema {
        let mut inputs = Vec::new();
        inputs.push(FormalParameter {
            name: "X".to_string(),
            description: "Input tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: false,
            variadic: false,
        });
        inputs.push(FormalParameter {
            name: "W".to_string(),
            description: "Filter tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: false,
            variadic: false,
        });
        inputs.push(FormalParameter {
            name: "B".to_string(),
            description: "Bias tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: true,
            variadic: false,
        });
        
        let mut outputs = Vec::new();
        outputs.push(FormalParameter {
            name: "Y".to_string(),
            description: "Output tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: false,
            variadic: false,
        });
        
        let mut attributes = HashMap::new();
        attributes.insert("kernel_shape".to_string(), AttributeProto {
            name: "kernel_shape".to_string(),
            description: "Shape of the convolution kernels".to_string(),
            type_: AttributeType::Ints,
            required: false,
        });
        
        attributes.insert("strides".to_string(), AttributeProto {
            name: "strides".to_string(),
            description: "Stride along each spatial axis".to_string(),
            type_: AttributeType::Ints,
            required: false,
        });
        
        OpSchema {
            name: "Conv".to_string(),
            domain: "".to_string(),
            since_version: 1,
            inputs,
            outputs,
            attributes,
        }
    }
    
    /// Example schema for MatMul operator
    fn create_matmul_schema() -> OpSchema {
        let mut inputs = Vec::new();
        inputs.push(FormalParameter {
            name: "A".to_string(),
            description: "First input matrix".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double, DataType::Int32, DataType::Int64]),
            optional: false,
            variadic: false,
        });
        inputs.push(FormalParameter {
            name: "B".to_string(),
            description: "Second input matrix".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double, DataType::Int32, DataType::Int64]),
            optional: false,
            variadic: false,
        });
        
        let mut outputs = Vec::new();
        outputs.push(FormalParameter {
            name: "Y".to_string(),
            description: "Output matrix".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double, DataType::Int32, DataType::Int64]),
            optional: false,
            variadic: false,
        });
        
        OpSchema {
            name: "MatMul".to_string(),
            domain: "".to_string(),
            since_version: 1,
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }
    
    /// Example schema for Relu operator
    fn create_relu_schema() -> OpSchema {
        let mut inputs = Vec::new();
        inputs.push(FormalParameter {
            name: "X".to_string(),
            description: "Input tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: false,
            variadic: false,
        });
        
        let mut outputs = Vec::new();
        outputs.push(FormalParameter {
            name: "Y".to_string(),
            description: "Output tensor".to_string(),
            type_constraints: Some(vec![DataType::Float, DataType::Float16, DataType::Double]),
            optional: false,
            variadic: false,
        });
        
        OpSchema {
            name: "Relu".to_string(),
            domain: "".to_string(),
            since_version: 1,
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }
    
    /// Validate the ONNX model
    pub fn validate_model(&self, model: &OnnxModel) -> Result<()> {
        // Check IR version compatibility
        self.check_version_compatibility(model)?;
        
        // Validate the model graph
        self.validate_graph(&model.graph)?;
        
        // Validate tensor shapes and types
        self.validate_tensor_shapes(model)?;
        
        Ok(())
    }
    
    /// Check if the model IR version is compatible
    pub fn check_version_compatibility(&self, model: &OnnxModel) -> Result<()> {
        let ir_version = model.metadata.ir_version;
        
        if ir_version < self.min_ir_version || ir_version > self.max_ir_version {
            return Err(Error::VersionIncompatible(
                format!("IR version {} is not supported (min: {}, max: {})",
                        ir_version, self.min_ir_version, self.max_ir_version)
            ));
        }
        
        Ok(())
    }
    
    /// Validate the graph structure
    pub fn validate_graph(&self, graph: &Graph) -> Result<()> {
        // Check for duplicate node names if names are provided
        let mut node_names = HashSet::new();
        for node in &graph.nodes {
            if !node.name.is_empty() {
                if !node_names.insert(&node.name) {
                    return Err(Error::InvalidGraph(format!("Duplicate node name: {}", node.name)));
                }
            }
        }
        
        // Check for duplicate output names
        let mut output_names = HashSet::new();
        for node in &graph.nodes {
            for output in &node.outputs {
                if !output_names.insert(output) {
                    return Err(Error::InvalidGraph(format!("Duplicate output name: {}", output)));
                }
            }
        }
        
        // Validate each node
        for node in &graph.nodes {
            // Get the opset version for this node's domain
            let domain = if node.domain.is_empty() { "".to_string() } else { node.domain.clone() };
            let opset_version = match model.opset_imports.get(&domain) {
                Some(version) => *version,
                None => return Err(Error::InvalidOperator(
                    format!("Unknown operator domain: {}", domain)
                )),
            };
            
            self.validate_node(node, opset_version)?;
        }
        
        Ok(())
    }
    
    /// Validate a single node
    pub fn validate_node(&self, node: &Node, opset_version: i64) -> Result<()> {
        let domain = if node.domain.is_empty() { "".to_string() } else { node.domain.clone() };
        
        // Get schema for this operator
        let schema = self.get_operator_schema(&node.op_type, &domain, opset_version)
            .ok_or_else(|| Error::InvalidOperator(
                format!("Unknown operator: {}:{} (version {})", 
                        domain, node.op_type, opset_version)
            ))?;
        
        // Check inputs
        self.check_input_types(node, &schema)?;
        
        // Check outputs
        self.check_output_types(node, &schema)?;
        
        // Check attributes
        self.check_attributes(node, &schema)?;
        
        Ok(())
    }
    
    /// Get operator schema for a specific operator type, domain, and version
    pub fn get_operator_schema(&self, op_type: &str, domain: &str, version: i64) -> Option<OpSchema> {
        // Get schemas for this domain
        let domain_schemas = match self.schemas.get(domain) {
            Some(schemas) => schemas,
            None => return None,
        };
        
        // Find the highest version that's <= the requested version
        let mut compatible_version = None;
        let mut schema = None;
        
        for (&schema_version, schema_def) in domain_schemas {
            if schema_version <= version && 
               (compatible_version.is_none() || schema_version > compatible_version.unwrap()) &&
               schema_def.name == op_type {
                compatible_version = Some(schema_version);
                schema = Some(schema_def);
            }
        }
        
        schema.cloned()
    }
    
    /// Check if node inputs match schema
    pub fn check_input_types(&self, node: &Node, schema: &OpSchema) -> Result<()> {
        let mut required_inputs = 0;
        
        for (i, param) in schema.inputs.iter().enumerate() {
            if !param.optional && !param.variadic {
                required_inputs += 1;
            }
            
            // Check if we have enough inputs
            if i < node.inputs.len() {
                // Would check type constraints here if we had type info
            } else if !param.optional {
                return Err(Error::ValidationError(
                    format!("Node {} is missing required input {}", node.name, param.name)
                ));
            }
        }
        
        // Check if we have too many inputs
        let has_variadic = schema.inputs.iter().any(|p| p.variadic);
        if node.inputs.len() > schema.inputs.len() && !has_variadic {
            return Err(Error::ValidationError(
                format!("Node {} has too many inputs", node.name)
            ));
        }
        
        // Check if we have all required inputs
        if node.inputs.len() < required_inputs {
            return Err(Error::ValidationError(
                format!("Node {} doesn't have enough inputs. Required: {}, Found: {}", 
                        node.name, required_inputs, node.inputs.len())
            ));
        }
        
        Ok(())
    }
    
    /// Check if node outputs match schema
    pub fn check_output_types(&self, node: &Node, schema: &OpSchema) -> Result<()> {
        let mut required_outputs = 0;
        
        for (i, param) in schema.outputs.iter().enumerate() {
            if !param.optional && !param.variadic {
                required_outputs += 1;
            }
            
            // Check if we have enough outputs
            if i < node.outputs.len() {
                // Would check type constraints here if we had type info
            } else if !param.optional {
                return Err(Error::ValidationError(
                    format!("Node {} is missing required output {}", node.name, param.name)
                ));
            }
        }
        
        // Check if we have too many outputs
        let has_variadic = schema.outputs.iter().any(|p| p.variadic);
        if node.outputs.len() > schema.outputs.len() && !has_variadic {
            return Err(Error::ValidationError(
                format!("Node {} has too many outputs", node.name)
            ));
        }
        
        // Check if we have all required outputs
        if node.outputs.len() < required_outputs {
            return Err(Error::ValidationError(
                format!("Node {} doesn't have enough outputs. Required: {}, Found: {}", 
                        node.name, required_outputs, node.outputs.len())
            ));
        }
        
        Ok(())
    }
    
    /// Check if node attributes match schema
    fn check_attributes(&self, node: &Node, schema: &OpSchema) -> Result<()> {
        // Check required attributes
        for (name, attr_proto) in &schema.attributes {
            if attr_proto.required && !node.attributes.contains_key(name) {
                return Err(Error::ValidationError(
                    format!("Node {} is missing required attribute {}", node.name, name)
                ));
            }
        }
        
        // Check attribute types
        for (name, attr) in &node.attributes {
            if let Some(attr_proto) = schema.attributes.get(name) {
                // Check if the attribute type matches the schema
                let attr_type = match attr {
                    crate::model::Attribute::Float(_) => AttributeType::Float,
                    crate::model::Attribute::Int(_) => AttributeType::Int,
                    crate::model::Attribute::String(_) => AttributeType::String,
                    crate::model::Attribute::Tensor(_) => AttributeType::Tensor,
                    crate::model::Attribute::Floats(_) => AttributeType::Floats,
                    crate::model::Attribute::Ints(_) => AttributeType::Ints,
                    crate::model::Attribute::Strings(_) => AttributeType::Strings,
                    crate::model::Attribute::Tensors(_) => AttributeType::Tensors,
                };
                
                if attr_type != attr_proto.type_ {
                    return Err(Error::ValidationError(
                        format!("Attribute {} type mismatch in node {}", name, node.name)
                    ));
                }
            }
            // Unknown attributes are allowed in ONNX
        }
        
        Ok(())
    }
    
    /// Validate tensor shapes and types consistency
    pub fn validate_tensor_shapes(&self, model: &OnnxModel) -> Result<()> {
        // Build a map of tensor names to their info
        let mut tensor_info = HashMap::new();
        
        // Add model inputs
        for input in &model.graph.inputs {
            tensor_info.insert(input.name.clone(), input);
        }
        
        // Add initializers
        for initializer in &model.graph.initializers {
            if let Some(existing) = tensor_info.get(&initializer.name) {
                // Verify type consistency
                if existing.data_type != initializer.data_type {
                    return Err(Error::ValidationError(
                        format!("Type mismatch for tensor {}: {:?} vs {:?}",
                                initializer.name, existing.data_type, initializer.data_type)
                    ));
                }
                
                // Verify shape consistency
                if existing.shape.len() != initializer.dims.len() {
                    return Err(Error::ValidationError(
                        format!("Shape mismatch for tensor {}: dimensions don't match",
                                initializer.name)
                    ));
                }
                
                for (i, (&a, &b)) in existing.shape.iter().zip(initializer.dims.iter()).enumerate() {
                    if a != -1 && a != b {
                        return Err(Error::ValidationError(
                            format!("Shape mismatch for tensor {} at dimension {}: {} vs {}",
                                    initializer.name, i, a, b)
                        ));
                    }
                }
            }
        }
        
        // In a real implementation, we would check shape consistency through the graph
        // by propagating shapes through operators, but that's beyond the scope here.
        
        Ok(())
    }
}