use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::error::Result;
use crate::model::{ExecutionGraph, Node, NodeId, Attribute};
use crate::optimization::graph_optimizer::{OptimizationPass, PassResult};

/// Candidate for operator fusion
#[derive(Debug)]
pub struct FusionCandidate {
    /// Pattern name
    pub pattern_name: String,
    /// Nodes involved in the fusion
    pub nodes: Vec<NodeId>,
    /// New node that will replace the fused nodes
    pub fused_node: Node,
}

/// Operator fusion pass
pub struct OperatorFusion {
    /// Name of the pass
    name: String,
}

impl OperatorFusion {
    /// Create a new operator fusion pass
    pub fn new() -> Self {
        Self {
            name: "OperatorFusion".to_string(),
        }
    }
    
    /// Find fusion patterns in the graph
    pub fn find_fusion_patterns(&self, graph: &ExecutionGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        
        // Find Conv + Bias + Activation pattern
        let conv_bias_act = self.find_conv_bias_activation_patterns(graph);
        candidates.extend(conv_bias_act);
        
        // Find MatMul + Add pattern
        let matmul_add = self.find_matmul_add_patterns(graph);
        candidates.extend(matmul_add);
        
        // Find consecutive Transpose patterns
        let transpose = self.find_consecutive_transpose_patterns(graph);
        candidates.extend(transpose);
        
        candidates
    }
    
    /// Apply fusion to a graph
    pub fn apply_fusion(&self, graph: &mut ExecutionGraph, candidates: Vec<FusionCandidate>) -> Result<usize> {
        let mut count = 0;
        
        // Sort candidates to fuse more complex patterns first
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by_key(|c| -(c.nodes.len() as isize));
        
        // Track nodes that have been fused, so we don't try to fuse them again
        let mut fused_nodes = HashSet::new();
        
        for candidate in sorted_candidates {
            // Skip if any of the nodes have already been fused
            if candidate.nodes.iter().any(|node_id| fused_nodes.contains(node_id)) {
                continue;
            }
            
            // Apply the fusion
            match candidate.pattern_name.as_str() {
                "conv_bias_activation" => {
                    self.apply_conv_bias_activation_fusion(graph, &candidate)?;
                    count += 1;
                },
                "matmul_add" => {
                    self.apply_matmul_add_fusion(graph, &candidate)?;
                    count += 1;
                },
                "consecutive_transposes" => {
                    self.apply_consecutive_transposes_fusion(graph, &candidate)?;
                    count += 1;
                },
                _ => {
                    // Unknown pattern
                    continue;
                }
            }
            
            // Mark the nodes as fused
            for &node_id in &candidate.nodes {
                fused_nodes.insert(node_id);
            }
        }
        
        Ok(count)
    }
    
    /// Find Conv + Bias + Activation patterns
    fn find_conv_bias_activation_patterns(&self, graph: &ExecutionGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        let nodes = &graph.nodes;
        
        // Map from output tensor name to producing node
        let mut output_map: HashMap<String, (NodeId, &Node)> = HashMap::new();
        for node in nodes {
            for output in &node.outputs {
                output_map.insert(output.clone(), (node.id, node));
            }
        }
        
        // Find Conv nodes
        for conv_node in nodes.iter().filter(|n| n.op_type == "Conv") {
            // Skip if the Conv already has a bias attribute
            if conv_node.attributes.contains_key("B") {
                continue;
            }
            
            // Check if the Conv output feeds into an Add node (for bias)
            let conv_output = match conv_node.outputs.first() {
                Some(output) => output,
                None => continue,
            };
            
            // Find Add node that uses the Conv output
            let add_node = nodes.iter().find(|n| {
                n.op_type == "Add" && n.inputs.first().map_or(false, |input| input == conv_output)
            });
            
            let add_node = match add_node {
                Some(node) => node,
                None => continue,
            };
            
            // Get the Add output
            let add_output = match add_node.outputs.first() {
                Some(output) => output,
                None => continue,
            };
            
            // Check if the Add output feeds into an activation function
            let activation_candidates = ["Relu", "LeakyRelu", "Sigmoid", "Tanh", "Elu"];
            let activation_node = nodes.iter().find(|n| {
                activation_candidates.contains(&n.op_type.as_str()) && 
                n.inputs.first().map_or(false, |input| input == add_output)
            });
            
            // Attempt to fuse either Conv+Add+Activation or just Conv+Add
            if let Some(act_node) = activation_node {
                // Create a fused node Conv + Add + Activation
                let mut fused_node = conv_node.clone();
                fused_node.id = conv_node.id; // Keep the Conv node ID
                fused_node.name = format!("{}_fused", conv_node.name);
                
                // Add bias parameter from Add node
                let bias_index = if add_node.inputs[0] == *conv_output { 1 } else { 0 };
                let bias_name = &add_node.inputs[bias_index];
                
                // Try to find the bias value
                let bias_tensor = graph.nodes.iter()
                    .filter(|n| n.op_type == "Constant")
                    .find_map(|n| {
                        if n.outputs.first().map_or(false, |out| out == bias_name) {
                            n.attributes.get("value").cloned()
                        } else {
                            None
                        }
                    });
                
                if let Some(Attribute::Tensor(bias)) = bias_tensor {
                    fused_node.attributes.insert("bias".to_string(), Attribute::Tensor(bias));
                }
                
                // Add activation type attribute
                fused_node.attributes.insert("activation_type".to_string(), 
                    Attribute::String(act_node.op_type.clone()));
                
                // Update the outputs
                fused_node.outputs = act_node.outputs.clone();
                
                // Create fusion candidate
                candidates.push(FusionCandidate {
                    pattern_name: "conv_bias_activation".to_string(),
                    nodes: vec![conv_node.id, add_node.id, act_node.id],
                    fused_node,
                });
            } else {
                // Create a fused node Conv + Add
                let mut fused_node = conv_node.clone();
                fused_node.id = conv_node.id; // Keep the Conv node ID
                fused_node.name = format!("{}_with_bias", conv_node.name);
                
                // Add bias parameter from Add node
                let bias_index = if add_node.inputs[0] == *conv_output { 1 } else { 0 };
                let bias_name = &add_node.inputs[bias_index];
                
                // Try to find the bias value
                let bias_tensor = graph.nodes.iter()
                    .filter(|n| n.op_type == "Constant")
                    .find_map(|n| {
                        if n.outputs.first().map_or(false, |out| out == bias_name) {
                            n.attributes.get("value").cloned()
                        } else {
                            None
                        }
                    });
                
                if let Some(Attribute::Tensor(bias)) = bias_tensor {
                    fused_node.attributes.insert("bias".to_string(), Attribute::Tensor(bias));
                }
                
                // Update the outputs
                fused_node.outputs = add_node.outputs.clone();
                
                // Create fusion candidate
                candidates.push(FusionCandidate {
                    pattern_name: "conv_bias_activation".to_string(),
                    nodes: vec![conv_node.id, add_node.id],
                    fused_node,
                });
            }
        }
        
        candidates
    }
    
    /// Find MatMul + Add patterns
    fn find_matmul_add_patterns(&self, graph: &ExecutionGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        let nodes = &graph.nodes;
        
        // Map from output tensor name to producing node
        let mut output_map: HashMap<String, (NodeId, &Node)> = HashMap::new();
        for node in nodes {
            for output in &node.outputs {
                output_map.insert(output.clone(), (node.id, node));
            }
        }
        
        // Find MatMul nodes
        for matmul_node in nodes.iter().filter(|n| n.op_type == "MatMul") {
            // Check if the MatMul output feeds into an Add node
            let matmul_output = match matmul_node.outputs.first() {
                Some(output) => output,
                None => continue,
            };
            
            // Find Add node that uses the MatMul output
            let add_node = nodes.iter().find(|n| {
                n.op_type == "Add" && n.inputs.first().map_or(false, |input| input == matmul_output)
            });
            
            let add_node = match add_node {
                Some(node) => node,
                None => continue,
            };
            
            // Create a fused node MatMul + Add (Gemm)
            let mut fused_node = Node {
                id: matmul_node.id, // Keep the MatMul node ID
                name: format!("{}_fused_gemm", matmul_node.name),
                op_type: "Gemm".to_string(),
                domain: "".to_string(),
                inputs: matmul_node.inputs.clone(),
                outputs: add_node.outputs.clone(),
                attributes: HashMap::new(),
                doc_string: format!("Fused MatMul+Add from {} and {}", matmul_node.name, add_node.name),
            };
            
            // Add bias input
            let bias_index = if add_node.inputs[0] == *matmul_output { 1 } else { 0 };
            fused_node.inputs.push(add_node.inputs[bias_index].clone());
            
            // Add Gemm attributes
            fused_node.attributes.insert("alpha".to_string(), Attribute::Float(1.0));
            fused_node.attributes.insert("beta".to_string(), Attribute::Float(1.0));
            fused_node.attributes.insert("transA".to_string(), Attribute::Int(0));
            fused_node.attributes.insert("transB".to_string(), Attribute::Int(0));
            
            // Create fusion candidate
            candidates.push(FusionCandidate {
                pattern_name: "matmul_add".to_string(),
                nodes: vec![matmul_node.id, add_node.id],
                fused_node,
            });
        }
        
        candidates
    }
    
    /// Find consecutive Transpose patterns
    fn find_consecutive_transpose_patterns(&self, graph: &ExecutionGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        let nodes = &graph.nodes;
        
        // Map from output tensor name to producing node
        let mut output_map: HashMap<String, (NodeId, &Node)> = HashMap::new();
        for node in nodes {
            for output in &node.outputs {
                output_map.insert(output.clone(), (node.id, node));
            }
        }
        
        // Find Transpose nodes
        for transpose1 in nodes.iter().filter(|n| n.op_type == "Transpose") {
            // Get the output of the first transpose
            let transpose1_output = match transpose1.outputs.first() {
                Some(output) => output,
                None => continue,
            };
            
            // Find another Transpose node that uses this output
            let transpose2 = nodes.iter().find(|n| {
                n.op_type == "Transpose" && 
                n.inputs.first().map_or(false, |input| input == transpose1_output)
            });
            
            let transpose2 = match transpose2 {
                Some(node) => node,
                None => continue,
            };
            
            // Get the perm attributes from both transposes
            let perm1 = match transpose1.attributes.get("perm") {
                Some(Attribute::Ints(perm)) => perm,
                _ => continue,
            };
            
            let perm2 = match transpose2.attributes.get("perm") {
                Some(Attribute::Ints(perm)) => perm,
                _ => continue,
            };
            
            // Check if these transposes cancel each other out or can be combined
            let can_optimize = self.can_optimize_transposes(perm1, perm2);
            
            if can_optimize {
                // Create an identity node or optimized transpose
                let input = transpose1.inputs.first().cloned().unwrap_or_default();
                let output = transpose2.outputs.first().cloned().unwrap_or_default();
                
                let fused_node = Node {
                    id: transpose1.id, // Keep the first transpose node ID
                    name: format!("{}_optimized", transpose1.name),
                    op_type: "Identity".to_string(), // Or optimized Transpose
                    domain: "".to_string(),
                    inputs: vec![input],
                    outputs: vec![output],
                    attributes: HashMap::new(),
                    doc_string: format!("Optimized transposes from {} and {}", transpose1.name, transpose2.name),
                };
                
                // Create fusion candidate
                candidates.push(FusionCandidate {
                    pattern_name: "consecutive_transposes".to_string(),
                    nodes: vec![transpose1.id, transpose2.id],
                    fused_node,
                });
            }
        }
        
        candidates
    }
    
    /// Check if two transpose operations can be optimized
    fn can_optimize_transposes(&self, perm1: &[i64], perm2: &[i64]) -> bool {
        // Special case: if they're the same length, check if they cancel each other out
        if perm1.len() == perm2.len() {
            let mut result = vec![0; perm1.len()];
            for (i, &p) in perm1.iter().enumerate() {
                if p >= 0 && p < perm2.len() as i64 {
                    result[i] = perm2[p as usize];
                } else {
                    return false;
                }
            }
            
            // Check if result is the identity permutation [0, 1, 2, ...]
            for (i, &r) in result.iter().enumerate() {
                if r != i as i64 {
                    return false;
                }
            }
            
            return true;
        }
        
        false
    }
    
    /// Apply Conv + Bias + Activation fusion
    fn apply_conv_bias_activation_fusion(&self, graph: &mut ExecutionGraph, candidate: &FusionCandidate) -> Result<()> {
        // Replace the old nodes with the fused node
        let fused_node = candidate.fused_node.clone();
        
        // Update nodes list
        // Remove old nodes
        graph.nodes.retain(|node| !candidate.nodes.contains(&node.id));
        
        // Add the new fused node
        graph.nodes.push(fused_node);
        
        // Update dependencies
        self.update_dependencies(graph, &candidate.nodes, candidate.fused_node.id)?;
        
        Ok(())
    }
    
    /// Apply MatMul + Add fusion
    fn apply_matmul_add_fusion(&self, graph: &mut ExecutionGraph, candidate: &FusionCandidate) -> Result<()> {
        // Replace the old nodes with the fused node
        let fused_node = candidate.fused_node.clone();
        
        // Update nodes list
        // Remove old nodes
        graph.nodes.retain(|node| !candidate.nodes.contains(&node.id));
        
        // Add the new fused node
        graph.nodes.push(fused_node);
        
        // Update dependencies
        self.update_dependencies(graph, &candidate.nodes, candidate.fused_node.id)?;
        
        Ok(())
    }
    
    /// Apply consecutive Transposes fusion
    fn apply_consecutive_transposes_fusion(&self, graph: &mut ExecutionGraph, candidate: &FusionCandidate) -> Result<()> {
        // Replace the old nodes with the fused node
        let fused_node = candidate.fused_node.clone();
        
        // Update nodes list
        // Remove old nodes
        graph.nodes.retain(|node| !candidate.nodes.contains(&node.id));
        
        // Add the new fused node
        graph.nodes.push(fused_node);
        
        // Update dependencies
        self.update_dependencies(graph, &candidate.nodes, candidate.fused_node.id)?;
        
        Ok(())
    }
    
    /// Update graph dependencies after fusion
    fn update_dependencies(&self, graph: &mut ExecutionGraph, old_nodes: &[NodeId], new_node_id: NodeId) -> Result<()> {
        // Map of inputs and outputs of all the old nodes
        let mut all_inputs = HashSet::new();
        let mut all_outputs = HashSet::new();
        
        for &node_id in old_nodes {
            if let Some(node) = graph.nodes.iter().find(|n| n.id == node_id) {
                all_inputs.extend(node.inputs.iter().cloned());
                all_outputs.extend(node.outputs.iter().cloned());
            }
        }
        
        // Update dependencies
        let mut new_dependencies = HashMap::new();
        
        for (node_id, deps) in &graph.dependencies {
            if old_nodes.contains(node_id) {
                // Skip old nodes
                continue;
            }
            
            let new_deps: Vec<NodeId> = deps.iter()
                .filter(|&&dep_id| !old_nodes.contains(&dep_id))
                .cloned()
                .collect();
            
            // Check if this node depends on any of the outputs from old nodes
            let node = graph.nodes.iter().find(|n| &n.id == node_id).unwrap();
            let depends_on_outputs = node.inputs.iter().any(|input| all_outputs.contains(input));
            
            if depends_on_outputs {
                // This node now depends on the new fused node
                let mut updated_deps = new_deps.clone();
                updated_deps.push(new_node_id);
                new_dependencies.insert(*node_id, updated_deps);
            } else {
                new_dependencies.insert(*node_id, new_deps);
            }
        }
        
        // Add dependencies for the new node
        let new_node = graph.nodes.iter().find(|n| n.id == new_node_id).unwrap();
        let new_node_deps: Vec<NodeId> = graph.nodes.iter()
            .filter(|n| n.id != new_node_id && new_node.inputs.iter().any(|input| n.outputs.contains(input)))
            .map(|n| n.id)
            .collect();
        
        new_dependencies.insert(new_node_id, new_node_deps);
        
        // Update input and output nodes
        let mut input_nodes: Vec<NodeId> = graph.input_nodes.iter()
            .filter(|&&id| !old_nodes.contains(&id))
            .cloned()
            .collect();
        
        // Check if the new node is an input node
        let new_node = graph.nodes.iter().find(|n| n.id == new_node_id).unwrap();
        for input in &new_node.inputs {
            if !graph.nodes.iter().any(|n| n.outputs.contains(input)) {
                // This input is not produced by any node, so the new node is an input node
                input_nodes.push(new_node_id);
                break;
            }
        }
        
        let mut output_nodes: Vec<NodeId> = graph.output_nodes.iter()
            .filter(|&&id| !old_nodes.contains(&id))
            .cloned()
            .collect();
        
        // Check if any of the old nodes were output nodes
        let old_nodes_were_outputs = old_nodes.iter().any(|id| graph.output_nodes.contains(id));
        if old_nodes_were_outputs {
            // The new node is now an output node
            output_nodes.push(new_node_id);
        }
        
        // Update the graph
        graph.dependencies = new_dependencies;
        graph.input_nodes = input_nodes;
        graph.output_nodes = output_nodes;
        
        Ok(())
    }
}

impl OptimizationPass for OperatorFusion {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&self, graph: &mut ExecutionGraph) -> Result<PassResult> {
        let start_time = Instant::now();
        
        // Find fusion candidates
        let candidates = self.find_fusion_patterns(graph);
        
        // Apply fusions
        let count = self.apply_fusion(graph, candidates)?;
        
        let duration = start_time.elapsed();
        
        Ok(PassResult {
            name: self.name.clone(),
            optimizations_applied: count,
            duration,
            changed: count > 0,
        })
    }
    
    fn dependencies(&self) -> Vec<&str> {
        // This pass should run after constant folding
        vec!["ConstantFolding"]
    }
}