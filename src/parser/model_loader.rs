use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use prost::Message;

use crate::error::{Error, Result};
use crate::model::{
    OnnxModel, ModelMetadata, TensorInfo, DataType, Tensor, Node, Graph, Attribute,
    SparseTensor, ExternalDataInfo, QuantInfo, Function, TrainingInfo, TypeInfo,
    Dimension, QuantizationAnnotation, TensorAnnotation,
};
use crate::proto::{
    ModelProto, GraphProto, NodeProto, TensorProto, SparseTensorProto, ValueInfoProto, 
    AttributeProto, OperatorSetIdProto, FunctionProto, TrainingInfoProto,
    QuantizationAnnotation as ProtoQuantizationAnnotation, 
    TensorAnnotation as ProtoTensorAnnotation,
    StringStringEntryProto,
};

/// ONNX model loader responsible for parsing and loading ONNX models
pub struct OnnxModelLoader;

impl OnnxModelLoader {
    /// Load an ONNX model from a file path
    pub fn load_model(path: &Path) -> Result<OnnxModel> {
        let mut file = File::open(path).map_err(|e| {
            Error::ModelLoadError(path.to_path_buf(), format!("Failed to open file: {}", e))
        })?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| {
            Error::ModelLoadError(path.to_path_buf(), format!("Failed to read file: {}", e))
        })?;
        
        Self::load_model_from_bytes(&buffer)
    }
    
    /// Load an ONNX model from bytes
    pub fn load_model_from_bytes(data: &[u8]) -> Result<OnnxModel> {
        let model_proto = Self::deserialize_model_proto(data)?;
        Self::convert_proto_to_internal(model_proto)
    }
    
    /// Deserialize protobuf bytes into a ModelProto
    pub fn deserialize_model_proto(bytes: &[u8]) -> Result<ModelProto> {
        ModelProto::decode(bytes).map_err(Error::ProtobufError)
    }
    
    /// Convert protobuf model to internal representation
    pub fn convert_proto_to_internal(proto: ModelProto) -> Result<OnnxModel> {
        // Validate required fields
        if !proto.has_graph() {
            return Err(Error::MissingField("Model is missing graph".to_string()));
        }
        
        let opset_imports = Self::handle_opset_imports(&proto.opset_import)?;
        
        let metadata = Self::extract_model_metadata(&proto);
        let graph = Self::convert_graph_proto(proto.graph.unwrap())?;
        
        // Convert functions if present
        let functions = if !proto.functions.is_empty() {
            proto.functions.iter()
                .map(Self::convert_function_proto)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        
        // Convert training info if present
        let training_info = if proto.has_training_info() {
            Some(Self::convert_training_info_proto(&proto.training_info.unwrap())?)
        } else {
            None
        };
        
        // Extract metadata properties
        let metadata_props = Self::extract_metadata_props(&proto.metadata_props);
        
        Ok(OnnxModel {
            metadata,
            graph,
            opset_imports,
            functions,
            training_info,
            metadata_props,
        })
    }
    
    /// Extract metadata properties from the model
    fn extract_metadata_props(metadata_props: &[StringStringEntryProto]) -> HashMap<String, String> {
        let mut props = HashMap::new();
        for prop in metadata_props {
            if let (Some(key), Some(value)) = (&prop.key, &prop.value) {
                props.insert(key.clone(), value.clone());
            }
        }
        props
    }
    
    /// Convert a FunctionProto to internal Function representation
    fn convert_function_proto(function_proto: &FunctionProto) -> Result<Function> {
        let nodes = function_proto.node.iter().enumerate()
            .map(|(id, node)| Self::convert_node_proto(node, id))
            .collect::<Result<Vec<_>>>()?;
        
        let opset_imports = if !function_proto.opset_import.is_empty() {
            Self::handle_opset_imports(&function_proto.opset_import)?
        } else {
            HashMap::new()
        };
        
        Ok(Function {
            name: function_proto.name.clone(),
            domain: function_proto.domain.clone(),
            since_version: function_proto.since_version,
            doc_string: function_proto.doc_string.clone(),
            inputs: function_proto.input.clone(),
            outputs: function_proto.output.clone(),
            attributes: function_proto.attribute.clone(),
            nodes,
            opset_imports,
            is_tensor_containment: function_proto.is_tensor_containment,
        })
    }
    
    /// Convert a TrainingInfoProto to internal TrainingInfo representation
    fn convert_training_info_proto(training_proto: &TrainingInfoProto) -> Result<TrainingInfo> {
        let algorithm = if training_proto.has_algorithm() {
            Self::convert_graph_proto(training_proto.algorithm.clone().unwrap())?
        } else {
            return Err(Error::MissingField("Training info missing algorithm".to_string()));
        };
        
        let initialization = if training_proto.has_initialization() {
            Some(Self::convert_graph_proto(training_proto.initialization.clone().unwrap())?)
        } else {
            None
        };
        
        let inputs = training_proto.input.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let outputs = training_proto.output.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let metadata = Self::extract_metadata_props(&training_proto.metadata_props);
        
        Ok(TrainingInfo {
            algorithm,
            initialization,
            inputs,
            outputs,
            metadata,
        })
    }
    
    /// Extract model metadata from protobuf
    pub fn extract_model_metadata(proto: &ModelProto) -> ModelMetadata {
        ModelMetadata {
            producer_name: proto.producer_name.clone(),
            producer_version: proto.producer_version.clone(),
            domain: proto.domain.clone(),
            model_version: proto.model_version,
            doc_string: proto.doc_string.clone(),
            graph_name: proto.graph.as_ref()
                .map(|g| g.name.clone())
                .unwrap_or_default(),
            ir_version: proto.ir_version,
        }
    }
    
    /// Process opset imports
    pub fn handle_opset_imports(imports: &[OperatorSetIdProto]) -> Result<HashMap<String, i64>> {
        let mut opset_map = HashMap::new();
        
        // Default opset for the empty domain
        opset_map.insert("".to_string(), 1);
        
        for import in imports {
            let domain = import.domain.clone().unwrap_or_default();
            let version = import.version.unwrap_or(1);
            
            opset_map.insert(domain, version);
        }
        
        Ok(opset_map)
    }
    
    /// Extract input tensor information from model
    pub fn get_input_info(model: &OnnxModel) -> Vec<TensorInfo> {
        model.graph.inputs.clone()
    }
    
    /// Extract output tensor information from model
    pub fn get_output_info(model: &OnnxModel) -> Vec<TensorInfo> {
        model.graph.outputs.clone()
    }
    
    /// Convert a GraphProto to internal Graph representation
    fn convert_graph_proto(graph_proto: GraphProto) -> Result<Graph> {
        let initializers = graph_proto.initializer.iter()
            .map(Self::convert_tensor_proto)
            .collect::<Result<Vec<_>>>()?;
        
        // Convert sparse initializers if present
        let sparse_initializers = graph_proto.sparse_initializer.iter()
            .map(Self::convert_sparse_tensor_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let inputs = graph_proto.input.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let outputs = graph_proto.output.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        // Add value_info for intermediate values
        let value_info = graph_proto.value_info.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let nodes = graph_proto.node.iter().enumerate()
            .map(|(id, node)| Self::convert_node_proto(node, id))
            .collect::<Result<Vec<_>>>()?;
        
        // Convert quantization annotations if present
        let quantization_annotations = graph_proto.quantization_annotation.iter()
            .map(Self::convert_quantization_annotation)
            .collect::<Result<Vec<_>>>()?;
        
        // Convert tensor annotations if present
        let tensor_annotations = graph_proto.tensor_annotation.iter()
            .map(Self::convert_tensor_annotation)
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Graph {
            name: graph_proto.name,
            nodes,
            inputs,
            outputs,
            initializers,
            sparse_initializers,
            value_info,
            quantization_annotations,
            tensor_annotations,
            doc_string: graph_proto.doc_string,
        })
    }
    
    /// Convert a SparseTensorProto to internal SparseTensor representation
    fn convert_sparse_tensor_proto(tensor_proto: &SparseTensorProto) -> Result<SparseTensor> {
        let data_type = tensor_proto
            .data_type
            .map(DataType::from_proto)
            .unwrap_or(DataType::Undefined);
        
        // Convert indices tensor
        let indices = if tensor_proto.has_indices() {
            Self::convert_tensor_proto(&tensor_proto.indices.clone().unwrap())?
        } else {
            return Err(Error::MissingField("Sparse tensor missing indices".to_string()));
        };
        
        // Convert values tensor
        let values = if tensor_proto.has_values() {
            Self::convert_tensor_proto(&tensor_proto.values.clone().unwrap())?
        } else {
            return Err(Error::MissingField("Sparse tensor missing values".to_string()));
        };
        
        Ok(SparseTensor {
            name: values.name.clone(),
            data_type,
            dims: tensor_proto.dims.clone(),
            indices,
            values,
            doc_string: String::new(), // SparseTensor doesn't have doc_string field
        })
    }
    
    /// Convert a QuantizationAnnotation to internal struct
    fn convert_quantization_annotation(
        annotation: &ProtoQuantizationAnnotation
    ) -> Result<QuantizationAnnotation> {
        let tensor_name = annotation.tensor_name.clone();
        
        Ok(QuantizationAnnotation {
            tensor_name,
            quant_parameter_tensor_name: annotation.quant_parameter_tensor_name.clone(),
            axis: annotation.axis,
            scales: annotation.scale.clone(),
            zero_points: annotation.zero_point.clone(),
        })
    }
    
    /// Convert a TensorAnnotation to internal struct
    fn convert_tensor_annotation(
        annotation: &ProtoTensorAnnotation
    ) -> Result<TensorAnnotation> {
        let tensor_name = annotation.tensor_name.clone();
        let mut quant_parameter_tensor_names = HashMap::new();
        
        for prop in &annotation.quant_parameter_tensor_names {
            if let (Some(key), Some(value)) = (&prop.key, &prop.value) {
                quant_parameter_tensor_names.insert(key.clone(), value.clone());
            }
        }
        
        Ok(TensorAnnotation {
            tensor_name,
            quant_parameter_tensor_names,
        })
    }
    
    /// Convert a NodeProto to internal Node representation
    fn convert_node_proto(node_proto: &NodeProto, id: usize) -> Result<Node> {
        let mut attributes = HashMap::new();
        
        for attr in &node_proto.attribute {
            let name = attr.name.clone();
            let value = Self::convert_attribute_proto(attr)?;
            attributes.insert(name, value);
        }
        
        Ok(Node {
            id,
            name: node_proto.name.clone(),
            op_type: node_proto.op_type.clone(),
            domain: node_proto.domain.clone(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            attributes,
            doc_string: node_proto.doc_string.clone(),
        })
    }
    
    /// Convert a TensorProto to internal Tensor representation
    fn convert_tensor_proto(tensor_proto: &TensorProto) -> Result<Tensor> {
        let data_type = tensor_proto
            .data_type
            .map(DataType::from_proto)
            .unwrap_or(DataType::Undefined);
        
        let mut raw_data = Vec::new();
        
        // Extract data based on data type
        if !tensor_proto.raw_data.is_empty() {
            raw_data = tensor_proto.raw_data.clone();
        } else {
            // Handle different data types 
            match data_type {
                DataType::Float => {
                    let mut bytes = Vec::with_capacity(tensor_proto.float_data.len() * 4);
                    for &val in &tensor_proto.float_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Int32 => {
                    let mut bytes = Vec::with_capacity(tensor_proto.int32_data.len() * 4);
                    for &val in &tensor_proto.int32_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Int64 => {
                    let mut bytes = Vec::with_capacity(tensor_proto.int64_data.len() * 8);
                    for &val in &tensor_proto.int64_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Double => {
                    let mut bytes = Vec::with_capacity(tensor_proto.double_data.len() * 8);
                    for &val in &tensor_proto.double_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Uint32 => {
                    let mut bytes = Vec::with_capacity(tensor_proto.uint32_data.len() * 4);
                    for &val in &tensor_proto.uint32_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Uint64 => {
                    let mut bytes = Vec::with_capacity(tensor_proto.uint64_data.len() * 8);
                    for &val in &tensor_proto.uint64_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::String => {
                    // Handle string data as concatenated binary for now
                    for s in &tensor_proto.string_data {
                        raw_data.extend_from_slice(s);
                    }
                }
                // Other data types would need more specialized handling
                _ => {}
            }
        }
        
        // Convert external data information if present
        let external_data = if tensor_proto.has_external_data() {
            let ext_data = tensor_proto.external_data.as_ref().unwrap();
            let mut kvp_data = HashMap::new();
            
            for kvp in &ext_data.kvp_data {
                if let (Some(key), Some(value)) = (&kvp.key, &kvp.value) {
                    kvp_data.insert(key.clone(), value.clone());
                }
            }
            
            Some(ExternalDataInfo {
                data_location: ext_data.data_location.clone().unwrap_or_default(),
                offset: ext_data.offset.unwrap_or_default(),
                length: ext_data.length.unwrap_or_default(),
                checksum: ext_data.checksum.clone().unwrap_or_default(),
                kvp_data,
            })
        } else {
            None
        };
        
        // Convert quantization info if present
        let quant_info = if tensor_proto.has_quant_info() {
            let qi = tensor_proto.quant_info.as_ref().unwrap();
            Some(QuantInfo {
                scale: qi.scale.unwrap_or(1.0),
                zero_point: qi.zero_point.unwrap_or(0),
            })
        } else {
            None
        };
        
        Ok(Tensor {
            name: tensor_proto.name.clone(),
            data_type,
            dims: tensor_proto.dims.clone(),
            data: raw_data,
            doc_string: tensor_proto.doc_string.clone(),
            external_data,
            quant_info,
        })
    }
    
    /// Convert a ValueInfoProto to internal TensorInfo representation
    fn convert_value_info_proto(value_info: &ValueInfoProto) -> Result<TensorInfo> {
        let name = value_info.name.clone();
        let doc_string = value_info.doc_string.clone();
        
        let type_proto = value_info.r#type.as_ref()
            .ok_or_else(|| Error::MissingField(format!("Missing type for value info: {}", name)))?;
        
        // We currently only support tensor types in our TensorInfo structure
        // For other types we'd need more complex handling, but we extract what we can
        
        let (elem_type, shape) = Self::extract_type_info(type_proto, &name)?;
        
        Ok(TensorInfo {
            name,
            shape,
            data_type: elem_type.map(DataType::from_proto).unwrap_or(DataType::Undefined),
            doc_string,
        })
    }
    
    /// Extract type information from a TypeProto
    fn extract_type_info(type_proto: &TypeProto, name: &str) -> Result<(Option<i32>, Vec<i64>)> {
        if let Some(value) = &type_proto.value {
            match value {
                // Tensor type
                prost::Oneof::First(tensor_type) => {
                    let elem_type = tensor_type.elem_type;
                    let shape = if let Some(shape_proto) = &tensor_type.shape {
                        shape_proto.dim.iter()
                            .map(|dim| match &dim.value {
                                Some(prost::Oneof::First(val)) => Ok(*val),
                                Some(prost::Oneof::Second(_)) => Ok(-1), // Dynamic dimension
                                None => Err(Error::MissingField("Missing dimension value".to_string())),
                            })
                            .collect::<Result<Vec<i64>>>()?
                    } else {
                        Vec::new()
                    };
                    Ok((elem_type, shape))
                },
                // Sequence type - we represent as a 1D tensor of unknown size
                prost::Oneof::Second(_) => {
                    // For sequence, we can't know the element type through this API
                    Ok((Some(0), vec![-1]))
                },
                // Map type - we represent as a 1D tensor of unknown size
                prost::Oneof::Third(_) => {
                    // For map, we can't know the element types through this API
                    Ok((Some(0), vec![-1]))
                },
                // Optional type - we use the elem_type but treat as scalar
                prost::Oneof::Fourth(_) => {
                    // For optional, we can't know the element type through this API
                    Ok((Some(0), Vec::new()))
                },
                // SparseTensor type
                prost::Oneof::Fifth(sparse_tensor_type) => {
                    let elem_type = sparse_tensor_type.elem_type;
                    let shape = if let Some(shape_proto) = &sparse_tensor_type.shape {
                        shape_proto.dim.iter()
                            .map(|dim| match &dim.value {
                                Some(prost::Oneof::First(val)) => Ok(*val),
                                Some(prost::Oneof::Second(_)) => Ok(-1), // Dynamic dimension
                                None => Err(Error::MissingField("Missing dimension value".to_string())),
                            })
                            .collect::<Result<Vec<i64>>>()?
                    } else {
                        Vec::new()
                    };
                    Ok((elem_type, shape))
                },
                // Unknown type
                _ => Err(Error::InvalidModel(format!("Unsupported value type for: {}", name))),
            }
        } else {
            Err(Error::MissingField(format!("Missing type for: {}", name)))
        }
    }
    
    /// Convert an AttributeProto to internal Attribute representation
    fn convert_attribute_proto(attr: &AttributeProto) -> Result<Attribute> {
        let attr_type = attr.r#type();
        
        match attr_type {
            0 => Err(Error::InvalidModel("Undefined attribute type".to_string())),
            
            // Single value attributes
            1 => Ok(Attribute::Float(attr.f)),
            2 => Ok(Attribute::Int(attr.i)),
            3 => Ok(Attribute::String(String::from_utf8_lossy(&attr.s).to_string())),
            4 => {
                if let Some(t) = &attr.t {
                    Ok(Attribute::Tensor(Self::convert_tensor_proto(t)?))
                } else {
                    Err(Error::MissingField(format!("Missing tensor in attribute {}", attr.name)))
                }
            },
            5 => {
                if let Some(g) = &attr.g {
                    Ok(Attribute::Graph(Self::convert_graph_proto(g.clone())?))
                } else {
                    Err(Error::MissingField(format!("Missing graph in attribute {}", attr.name)))
                }
            },
            11 => {
                if let Some(st) = &attr.sparse_tensor {
                    Ok(Attribute::SparseTensor(Self::convert_sparse_tensor_proto(st)?))
                } else {
                    Err(Error::MissingField(format!("Missing sparse tensor in attribute {}", attr.name)))
                }
            },
            13 => {
                if let Some(tp) = &attr.tp {
                    let (elem_type, shape) = Self::extract_type_info(tp, &attr.name)?;
                    
                    let type_info = match tp.value.as_ref() {
                        Some(prost::Oneof::First(tensor_type)) => {
                            // Convert shape to dimensions
                            let shape_info = if let Some(shape_proto) = &tensor_type.shape {
                                let dims = shape_proto.dim.iter().map(|dim| {
                                    match &dim.value {
                                        Some(prost::Oneof::First(val)) => Dimension::Value(*val),
                                        Some(prost::Oneof::Second(param)) => Dimension::Param(param.clone()),
                                        None => Dimension::Value(-1), // Unknown dimension
                                    }
                                }).collect();
                                Some(dims)
                            } else {
                                None
                            };
                            
                            let elem_type_val = tensor_type.elem_type
                                .map(DataType::from_proto)
                                .unwrap_or(DataType::Undefined);
                            
                            TypeInfo::Tensor {
                                elem_type: elem_type_val,
                                shape: shape_info,
                            }
                        },
                        // For other types, we'll use simpler representations for now
                        Some(prost::Oneof::Second(_)) => TypeInfo::Sequence {
                            elem_type: Box::new(TypeInfo::Tensor { 
                                elem_type: DataType::Undefined, 
                                shape: None 
                            }),
                        },
                        Some(prost::Oneof::Third(_)) => TypeInfo::Map {
                            key_type: DataType::Undefined,
                            value_type: Box::new(TypeInfo::Tensor { 
                                elem_type: DataType::Undefined, 
                                shape: None 
                            }),
                        },
                        Some(prost::Oneof::Fourth(_)) => TypeInfo::Optional {
                            elem_type: Box::new(TypeInfo::Tensor { 
                                elem_type: DataType::Undefined, 
                                shape: None 
                            }),
                        },
                        Some(prost::Oneof::Fifth(_)) => TypeInfo::SparseTensor {
                            elem_type: DataType::Undefined,
                            shape: None,
                        },
                        _ => TypeInfo::Tensor { 
                            elem_type: DataType::Undefined, 
                            shape: None 
                        },
                    };
                    
                    Ok(Attribute::TypeProto(type_info))
                } else {
                    Err(Error::MissingField(format!("Missing type proto in attribute {}", attr.name)))
                }
            },
            
            // List attributes
            6 => Ok(Attribute::Floats(attr.floats.clone())),
            7 => Ok(Attribute::Ints(attr.ints.clone())),
            8 => {
                let strings = attr.strings.iter()
                    .map(|s| String::from_utf8_lossy(s).to_string())
                    .collect();
                Ok(Attribute::Strings(strings))
            },
            9 => {
                let tensors = attr.tensors.iter()
                    .map(Self::convert_tensor_proto)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Attribute::Tensors(tensors))
            },
            10 => {
                let graphs = attr.graphs.iter()
                    .map(|g| Self::convert_graph_proto(g.clone()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Attribute::Graphs(graphs))
            },
            12 => {
                let sparse_tensors = attr.sparse_tensors.iter()
                    .map(Self::convert_sparse_tensor_proto)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Attribute::SparseTensors(sparse_tensors))
            },
            14 => {
                // Type protos are more complex to convert
                // For now, we'll create a simple placeholder implementation
                Ok(Attribute::TypeProtos(vec![TypeInfo::Tensor { 
                    elem_type: DataType::Undefined, 
                    shape: None 
                }]))
            },
            
            // Unknown attribute type
            _ => Err(Error::InvalidModel(format!("Unknown attribute type: {}", attr_type))),
        }
    }
}