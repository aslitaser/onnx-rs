use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use prost::Message;

use crate::error::{Error, Result};
use crate::model::{OnnxModel, ModelMetadata, TensorInfo, DataType, Tensor, Node, Graph, Attribute};
use crate::proto::{ModelProto, GraphProto, NodeProto, TensorProto, ValueInfoProto, AttributeProto, OperatorSetIdProto};

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
        
        Ok(OnnxModel {
            metadata,
            graph,
            opset_imports,
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
        
        let inputs = graph_proto.input.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let outputs = graph_proto.output.iter()
            .map(Self::convert_value_info_proto)
            .collect::<Result<Vec<_>>>()?;
        
        let nodes = graph_proto.node.iter().enumerate()
            .map(|(id, node)| Self::convert_node_proto(node, id))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Graph {
            name: graph_proto.name,
            nodes,
            inputs,
            outputs,
            initializers,
            doc_string: graph_proto.doc_string,
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
            // Handle different data types (simplified for brevity)
            match data_type {
                DataType::Float => {
                    let mut bytes = Vec::with_capacity(tensor_proto.float_data.len() * 4);
                    for &val in &tensor_proto.float_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                DataType::Int32 | DataType::Int64 => {
                    let mut bytes = Vec::with_capacity(tensor_proto.int64_data.len() * 8);
                    for &val in &tensor_proto.int64_data {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    raw_data = bytes;
                }
                // Other data types would be handled similarly
                _ => {}
            }
        }
        
        Ok(Tensor {
            name: tensor_proto.name.clone(),
            data_type,
            dims: tensor_proto.dims.clone(),
            data: raw_data,
            doc_string: tensor_proto.doc_string.clone(),
        })
    }
    
    /// Convert a ValueInfoProto to internal TensorInfo representation
    fn convert_value_info_proto(value_info: &ValueInfoProto) -> Result<TensorInfo> {
        let name = value_info.name.clone();
        let doc_string = value_info.doc_string.clone();
        
        let type_proto = value_info.r#type.as_ref()
            .ok_or_else(|| Error::MissingField(format!("Missing type for value info: {}", name)))?;
        
        let tensor_type = type_proto.value.as_ref()
            .ok_or_else(|| Error::MissingField(format!("Missing tensor type for value info: {}", name)))?;
        
        let elem_type = match tensor_type {
            prost::Oneof::First(tensor) => tensor.elem_type,
            _ => return Err(Error::InvalidModel(format!("Unsupported type for value info: {}", name))),
        };
        
        let shape = match tensor_type {
            prost::Oneof::First(tensor) => {
                if let Some(shape) = &tensor.shape {
                    shape.dim.iter()
                        .map(|dim| match &dim.value {
                            Some(prost::Oneof::First(val)) => Ok(*val),
                            Some(prost::Oneof::Second(_)) => Ok(-1), // Dynamic dimension
                            None => Err(Error::MissingField("Missing dimension value".to_string())),
                        })
                        .collect::<Result<Vec<i64>>>()?
                } else {
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };
        
        Ok(TensorInfo {
            name,
            shape,
            data_type: elem_type.map(DataType::from_proto).unwrap_or(DataType::Undefined),
            doc_string,
        })
    }
    
    /// Convert an AttributeProto to internal Attribute representation
    fn convert_attribute_proto(attr: &AttributeProto) -> Result<Attribute> {
        let attr_type = attr.r#type();
        
        match attr_type {
            0 => Err(Error::InvalidModel("Undefined attribute type".to_string())),
            1 => Ok(Attribute::Float(attr.f)),
            2 => Ok(Attribute::Int(attr.i)),
            3 => Ok(Attribute::String(attr.s.clone())),
            4 => {
                if let Some(t) = &attr.t {
                    Ok(Attribute::Tensor(Self::convert_tensor_proto(t)?))
                } else {
                    Err(Error::MissingField(format!("Missing tensor in attribute {}", attr.name)))
                }
            },
            5 => { /* Graph attribute - not implemented */ 
                Err(Error::UnsupportedFeature("Graph attributes not supported".to_string()))
            },
            6 => Ok(Attribute::Floats(attr.floats.clone())),
            7 => Ok(Attribute::Ints(attr.ints.clone())),
            8 => Ok(Attribute::Strings(attr.strings.clone())),
            9 => {
                let tensors = attr.tensors.iter()
                    .map(Self::convert_tensor_proto)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Attribute::Tensors(tensors))
            },
            10 => { /* Graphs attribute - not implemented */
                Err(Error::UnsupportedFeature("Graph attributes not supported".to_string()))
            },
            _ => Err(Error::InvalidModel(format!("Unknown attribute type: {}", attr_type))),
        }
    }
}