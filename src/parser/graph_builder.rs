use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{Error, Result};
use crate::model::{OnnxModel, Node, NodeId, Tensor, ExecutionGraph, Subgraph};
use crate::proto::GraphProto;

/// Graph builder responsible for constructing the computational graph
pub struct GraphBuilder;

impl GraphBuilder {
    /// Build an execution graph from the ONNX model
    pub fn build_graph(model: &OnnxModel) -> Result<ExecutionGraph> {
        // Create nodes from the graph
        let mut nodes = model.graph.nodes.clone();
        
        // Resolve inputs and outputs
        Self::resolve_inputs_outputs(&mut nodes, &model.graph.initializers)?;
        
        // Sort nodes topologically
        let sorted_nodes = Self::topological_sort(nodes)?;
        
        // Build dependency map
        let dependencies = Self::build_dependency_map(&sorted_nodes);
        
        // Create execution graph
        let graph = ExecutionGraph {
            nodes: sorted_nodes.clone(),
            input_nodes: Self::find_input_nodes(&sorted_nodes),
            output_nodes: Self::find_output_nodes(&sorted_nodes, &model.graph.outputs),
            dependencies,
        };
        
        // Validate graph connectivity
        Self::validate_graph_connectivity(&graph)?;
        
        Ok(graph)
    }
    
    /// Create nodes from a GraphProto
    pub fn create_nodes(graph_proto: &GraphProto) -> Result<Vec<Node>> {
        let mut nodes = Vec::new();
        
        for (i, node_proto) in graph_proto.node.iter().enumerate() {
            let domain = node_proto.domain.clone();
            let name = if node_proto.name.is_empty() {
                format!("node_{}", i)
            } else {
                node_proto.name.clone()
            };
            
            let node = Node {
                id: i,
                name,
                op_type: node_proto.op_type.clone(),
                domain,
                inputs: node_proto.input.clone(),
                outputs: node_proto.output.clone(),
                attributes: HashMap::new(), // Will be filled by model_loader
                doc_string: node_proto.doc_string.clone(),
            };
            
            nodes.push(node);
        }
        
        Ok(nodes)
    }
    
    /// Resolve inputs and outputs connections
    pub fn resolve_inputs_outputs(nodes: &mut [Node], initializers: &[Tensor]) -> Result<()> {
        // Create a map of all initializers by name
        let initializer_map: HashMap<String, &Tensor> = initializers
            .iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        
        // Create a map of tensor producers
        let mut tensor_producers: HashMap<String, NodeId> = HashMap::new();
        
        // First pass: record all outputs
        for node in nodes.iter() {
            for output in &node.outputs {
                if !output.is_empty() {
                    tensor_producers.insert(output.clone(), node.id);
                }
            }
        }
        
        // Nothing to do in the second pass, since we're not modifying the nodes
        // In a real implementation, we might update the node structure to include
        // references to producer nodes rather than just tensor names
        
        Ok(())
    }
    
    /// Sort nodes topologically
    pub fn topological_sort(nodes: Vec<Node>) -> Result<Vec<Node>> {
        let node_count = nodes.len();
        
        // Build a map of nodes by ID for easy lookup
        let node_map: HashMap<NodeId, &Node> = nodes
            .iter()
            .map(|n| (n.id, n))
            .collect();
        
        // Build the adjacency list representation of the graph
        let mut adjacency_list: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        
        // Initialize in-degree counts to 0
        for node in &nodes {
            in_degree.insert(node.id, 0);
            adjacency_list.insert(node.id, Vec::new());
        }
        
        // Create a map of tensor producers
        let mut tensor_producers: HashMap<String, NodeId> = HashMap::new();
        
        // Identify the producer of each tensor
        for node in &nodes {
            for output in &node.outputs {
                if !output.is_empty() {
                    tensor_producers.insert(output.clone(), node.id);
                }
            }
        }
        
        // Build the graph edges and count in-degrees
        for node in &nodes {
            for input in &node.inputs {
                if !input.is_empty() {
                    if let Some(&producer_id) = tensor_producers.get(input) {
                        // Add edge from producer to consumer
                        if let Some(edges) = adjacency_list.get_mut(&producer_id) {
                            edges.push(node.id);
                        }
                        
                        // Increment in-degree of the consumer
                        *in_degree.entry(node.id).or_insert(0) += 1;
                    }
                    // If the tensor doesn't have a producer, it's an external input
                }
            }
        }
        
        // Perform topological sort using Kahn's algorithm
        let mut sorted = Vec::with_capacity(node_count);
        let mut queue = VecDeque::new();
        
        // Start with nodes that have no dependencies (in-degree = 0)
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }
        
        while let Some(node_id) = queue.pop_front() {
            // Add node to sorted result
            sorted.push(node_id);
            
            // Reduce in-degree for all dependent nodes
            if let Some(edges) = adjacency_list.get(&node_id) {
                for &dependent_id in edges {
                    let new_degree = in_degree.get(&dependent_id).cloned().unwrap_or(0) - 1;
                    in_degree.insert(dependent_id, new_degree);
                    
                    if new_degree == 0 {
                        queue.push_back(dependent_id);
                    }
                }
            }
        }
        
        // Check if we have a valid topological sort
        if sorted.len() != node_count {
            return Err(Error::InvalidGraph("Graph contains cycles".to_string()));
        }
        
        // Create the sorted node list
        let sorted_nodes = sorted
            .into_iter()
            .map(|id| nodes.iter().find(|n| n.id == id).unwrap().clone())
            .collect();
        
        Ok(sorted_nodes)
    }
    
    /// Detect subgraphs in the node list
    pub fn detect_subgraphs(nodes: &[Node]) -> Vec<Subgraph> {
        // In a real implementation, this would identify common patterns in the graph
        // that could be optimized, such as fusing multiple operations.
        // For simplicity, we'll return just one subgraph representing the entire graph.
        
        let mut subgraph = Subgraph {
            nodes: nodes.iter().map(|n| n.id).collect(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        
        // Find external inputs (inputs not produced within the graph)
        let mut tensor_producers: HashMap<String, NodeId> = HashMap::new();
        
        // Record all tensor producers
        for node in nodes {
            for output in &node.outputs {
                if !output.is_empty() {
                    tensor_producers.insert(output.clone(), node.id);
                }
            }
        }
        
        // Find graph inputs and outputs
        let mut input_tensors = HashSet::new();
        let mut all_inputs = HashSet::new();
        let mut output_tensors = HashSet::new();
        
        // Collect all inputs
        for node in nodes {
            for input in &node.inputs {
                if !input.is_empty() {
                    all_inputs.insert(input.clone());
                    if !tensor_producers.contains_key(input) {
                        input_tensors.insert(input.clone());
                    }
                }
            }
        }
        
        // Collect outputs (tensors that are not consumed within the graph)
        for node in nodes {
            for output in &node.outputs {
                if !output.is_empty() {
                    let is_consumed = nodes.iter().any(|n| n.inputs.contains(output));
                    if !is_consumed {
                        output_tensors.insert(output.clone());
                    }
                }
            }
        }
        
        subgraph.inputs = input_tensors.into_iter().collect();
        subgraph.outputs = output_tensors.into_iter().collect();
        
        vec![subgraph]
    }
    
    /// Find input nodes in the execution graph
    pub fn find_input_nodes(nodes: &[Node]) -> Vec<NodeId> {
        // Create a map of all tensor producers
        let mut tensor_producers = HashMap::new();
        
        for node in nodes {
            for output in &node.outputs {
                tensor_producers.insert(output.clone(), node.id);
            }
        }
        
        // Find nodes that take external inputs
        let mut input_nodes = HashSet::new();
        
        for node in nodes {
            for input in &node.inputs {
                if !input.is_empty() && !tensor_producers.contains_key(input) {
                    input_nodes.insert(node.id);
                }
            }
        }
        
        input_nodes.into_iter().collect()
    }
    
    /// Find output nodes in the execution graph
    pub fn find_output_nodes(nodes: &[Node], model_outputs: &[crate::model::TensorInfo]) -> Vec<NodeId> {
        // Create a set of model output names
        let model_output_names: HashSet<String> = model_outputs.iter()
            .map(|o| o.name.clone())
            .collect();
        
        // Find nodes that produce graph outputs
        let mut output_nodes = HashSet::new();
        
        for node in nodes {
            for output in &node.outputs {
                if model_output_names.contains(output) {
                    output_nodes.insert(node.id);
                }
            }
        }
        
        output_nodes.into_iter().collect()
    }
    
    /// Build a dependency map for all nodes
    pub fn build_dependency_map(nodes: &[Node]) -> HashMap<NodeId, Vec<NodeId>> {
        let mut dependency_map = HashMap::new();
        
        // Create a map of tensor producers
        let mut tensor_producers = HashMap::new();
        
        for node in nodes {
            for output in &node.outputs {
                tensor_producers.insert(output.clone(), node.id);
            }
        }
        
        // For each node, find its dependencies
        for node in nodes {
            let mut dependencies = Vec::new();
            
            for input in &node.inputs {
                if let Some(&producer_id) = tensor_producers.get(input) {
                    if !dependencies.contains(&producer_id) {
                        dependencies.push(producer_id);
                    }
                }
            }
            
            dependency_map.insert(node.id, dependencies);
        }
        
        dependency_map
    }
    
    /// Validate graph connectivity
    pub fn validate_graph_connectivity(graph: &ExecutionGraph) -> Result<()> {
        // Check that all nodes are reachable from inputs
        let node_ids: HashSet<NodeId> = graph.nodes.iter().map(|n| n.id).collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start with input nodes
        for &input_id in &graph.input_nodes {
            queue.push_back(input_id);
        }
        
        // Breadth-first traversal
        while let Some(node_id) = queue.pop_front() {
            if visited.insert(node_id) {
                // Find all nodes that depend on this one
                for &next_id in node_ids.iter() {
                    if let Some(deps) = graph.dependencies.get(&next_id) {
                        if deps.contains(&node_id) && !visited.contains(&next_id) {
                            queue.push_back(next_id);
                        }
                    }
                }
                
                // Also find nodes this one depends on
                if let Some(deps) = graph.dependencies.get(&node_id) {
                    for &dep_id in deps {
                        if !visited.contains(&dep_id) {
                            queue.push_back(dep_id);
                        }
                    }
                }
            }
        }
        
        // Make sure we've visited all nodes
        if visited.len() != node_ids.len() {
            let unreachable: Vec<NodeId> = node_ids.difference(&visited)
                .cloned()
                .collect();
                
            return Err(Error::InvalidGraph(
                format!("Graph contains unreachable nodes: {:?}", unreachable)
            ));
        }
        
        // Check that there are no dangling outputs (outputs that aren't consumed or model outputs)
        let mut tensor_consumers = HashMap::new();
        
        // Map each input tensor to nodes that consume it
        for node in &graph.nodes {
            for input in &node.inputs {
                tensor_consumers
                    .entry(input.clone())
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }
        
        // Check that all outputs are either consumed or model outputs
        let model_output_nodes: HashSet<NodeId> = graph.output_nodes.iter().cloned().collect();
        
        for node in &graph.nodes {
            // Skip output nodes, they are allowed to have unconsumed outputs
            if model_output_nodes.contains(&node.id) {
                continue;
            }
            
            for output in &node.outputs {
                if !tensor_consumers.contains_key(output) {
                    // This output is not consumed anywhere
                    
                    // In a real implementation, we might warn about unused outputs
                    // but not treat it as an error
                    // return Err(Error::InvalidGraph(
                    //     format!("Node {} produces unused output {}", node.name, output)
                    // ));
                }
            }
        }
        
        Ok(())
    }
}