use std::collections::HashMap;
use onnx::{
    memory::{
        MemoryPlanner, MemoryPlan, SystemAllocator, MemoryAllocator,
        TensorId, TensorMemoryInfo, InplaceOpportunity, SharingOpportunity, TensorAllocation
    },
    model::{ExecutionGraph, Node, NodeId},
    ops::tensor::DataType,
    error::Result,
};

// Create a mock execution graph for testing
fn create_mock_graph() -> ExecutionGraph {
    let node1 = Node {
        id: 1,
        name: "node1".to_string(),
        op_type: "Conv".to_string(),
        domain: "".to_string(),
        inputs: vec!["input".to_string(), "weight".to_string()],
        outputs: vec!["conv_output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let node2 = Node {
        id: 2,
        name: "node2".to_string(),
        op_type: "Relu".to_string(),
        domain: "".to_string(),
        inputs: vec!["conv_output".to_string()],
        outputs: vec!["relu_output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let node3 = Node {
        id: 3,
        name: "node3".to_string(),
        op_type: "MaxPool".to_string(),
        domain: "".to_string(),
        inputs: vec!["relu_output".to_string()],
        outputs: vec!["pool_output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let node4 = Node {
        id: 4,
        name: "node4".to_string(),
        op_type: "Gemm".to_string(),
        domain: "".to_string(),
        inputs: vec!["pool_output".to_string(), "fc_weight".to_string()],
        outputs: vec!["output".to_string()],
        attributes: HashMap::new(),
        doc_string: "".to_string(),
    };
    
    let mut dependencies = HashMap::new();
    dependencies.insert(1, vec![]);
    dependencies.insert(2, vec![1]);
    dependencies.insert(3, vec![2]);
    dependencies.insert(4, vec![3]);
    
    ExecutionGraph {
        nodes: vec![node1, node2, node3, node4],
        input_nodes: vec![1],
        output_nodes: vec![4],
        dependencies,
    }
}

// Mock implementation for lifetimes
fn create_mock_lifetimes() -> HashMap<TensorId, (usize, usize)> {
    let mut lifetimes = HashMap::new();
    
    // Input tensors
    lifetimes.insert(hash_string("input"), (0, 1));
    lifetimes.insert(hash_string("weight"), (0, 1));
    lifetimes.insert(hash_string("fc_weight"), (0, 4));
    
    // Intermediate tensors
    lifetimes.insert(hash_string("conv_output"), (1, 2));
    lifetimes.insert(hash_string("relu_output"), (2, 3));
    lifetimes.insert(hash_string("pool_output"), (3, 4));
    
    // Output tensor
    lifetimes.insert(hash_string("output"), (4, 5));
    
    lifetimes
}

// Mock implementation for tensor info
fn create_mock_tensor_info() -> HashMap<TensorId, TensorMemoryInfo> {
    let mut tensor_info = HashMap::new();
    
    // Input tensors
    tensor_info.insert(
        hash_string("input"),
        TensorMemoryInfo {
            id: hash_string("input"),
            name: "input".to_string(),
            size_bytes: 4 * 1 * 224 * 224, // Batch x Channels x Height x Width (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: false,
        },
    );
    
    tensor_info.insert(
        hash_string("weight"),
        TensorMemoryInfo {
            id: hash_string("weight"),
            name: "weight".to_string(),
            size_bytes: 4 * 64 * 1 * 3 * 3, // OutChannels x InChannels x KernelH x KernelW (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: false,
        },
    );
    
    tensor_info.insert(
        hash_string("fc_weight"),
        TensorMemoryInfo {
            id: hash_string("fc_weight"),
            name: "fc_weight".to_string(),
            size_bytes: 4 * 1000 * 64 * 56 * 56, // OutputDim x InputDim (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: false,
        },
    );
    
    // Intermediate tensors
    tensor_info.insert(
        hash_string("conv_output"),
        TensorMemoryInfo {
            id: hash_string("conv_output"),
            name: "conv_output".to_string(),
            size_bytes: 4 * 1 * 64 * 224 * 224, // Batch x Channels x Height x Width (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: true,
        },
    );
    
    tensor_info.insert(
        hash_string("relu_output"),
        TensorMemoryInfo {
            id: hash_string("relu_output"),
            name: "relu_output".to_string(),
            size_bytes: 4 * 1 * 64 * 224 * 224, // Same as conv_output (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: true,
        },
    );
    
    tensor_info.insert(
        hash_string("pool_output"),
        TensorMemoryInfo {
            id: hash_string("pool_output"),
            name: "pool_output".to_string(),
            size_bytes: 4 * 1 * 64 * 112 * 112, // Batch x Channels x Height/2 x Width/2 (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: true,
        },
    );
    
    // Output tensor
    tensor_info.insert(
        hash_string("output"),
        TensorMemoryInfo {
            id: hash_string("output"),
            name: "output".to_string(),
            size_bytes: 4 * 1 * 1000, // Batch x OutputDim (float32)
            data_type: DataType::Float32,
            alignment: 64,
            allow_inplace: false,
        },
    );
    
    tensor_info
}

// Create a mock memory plan
fn create_mock_memory_plan() -> MemoryPlan {
    let lifetimes = create_mock_lifetimes();
    let tensor_info = create_mock_tensor_info();
    
    // Create a simplified memory plan
    let mut allocations = HashMap::new();
    let mut offset = 0;
    
    for (tensor_id, info) in &tensor_info {
        // In a real system, you would optimize the layout
        allocations.insert(
            *tensor_id,
            TensorAllocation {
                tensor_id: *tensor_id,
                offset,
                size_bytes: info.size_bytes,
                buffer_index: 0,
            },
        );
        
        offset += info.size_bytes;
    }
    
    // Create some mock in-place opportunities
    let inplace_ops = vec![
        InplaceOpportunity {
            node_id: 2, // Relu
            input_id: hash_string("conv_output"),
            output_id: hash_string("relu_output"),
            size_bytes: 4 * 1 * 64 * 224 * 224,
        },
    ];
    
    MemoryPlan {
        allocations,
        tensor_info,
        lifetimes,
        buffer_sizes: vec![offset],
        inplace_ops,
        total_memory_bytes: offset,
        execution_order: vec![1, 2, 3, 4],
    }
}

// Simple hash function to create TensorId from string
fn hash_string(s: &str) -> TensorId {
    s.as_bytes().iter().sum::<u8>() as usize
}

fn main() -> Result<()> {
    println!("Memory Planning Example");
    println!("----------------------");
    
    // Create a mock execution graph
    let graph = create_mock_graph();
    println!("Created mock execution graph with {} nodes", graph.nodes.len());
    
    // Create a memory planner
    let planner = MemoryPlanner::new();
    
    // Manually call the functions to demonstrate the API
    // In a real system, you would call plan_memory_usage directly
    
    println!("\n1. Computing Tensor Lifetimes");
    println!("--------------------------");
    let execution_order = vec![1, 2, 3, 4]; // Mock execution order
    let lifetimes = planner.compute_tensor_lifetimes(&graph, &execution_order)?;
    
    println!("Tensor lifetimes (tensor_id, first_use, last_use):");
    for (tensor_id, (first, last)) in &lifetimes {
        // Look up the tensor name for display
        let tensor_name = if tensor_id == &hash_string("input") {
            "input"
        } else if tensor_id == &hash_string("weight") {
            "weight"
        } else if tensor_id == &hash_string("fc_weight") {
            "fc_weight"
        } else if tensor_id == &hash_string("conv_output") {
            "conv_output"
        } else if tensor_id == &hash_string("relu_output") {
            "relu_output"
        } else if tensor_id == &hash_string("pool_output") {
            "pool_output"
        } else if tensor_id == &hash_string("output") {
            "output"
        } else {
            "unknown"
        };
        
        println!("  {} ({}): ({}, {})", tensor_name, tensor_id, first, last);
    }
    
    println!("\n2. Analyzing In-place Opportunities");
    println!("--------------------------------");
    let inplace_ops = planner.inplace_operations_analysis(&graph)?;
    
    println!("In-place operation opportunities:");
    for op in &inplace_ops {
        // Look up the tensor names for display
        let input_name = if op.input_id == hash_string("conv_output") {
            "conv_output"
        } else if op.input_id == hash_string("relu_output") {
            "relu_output"
        } else {
            "unknown"
        };
        
        let output_name = if op.output_id == hash_string("relu_output") {
            "relu_output"
        } else if op.output_id == hash_string("pool_output") {
            "pool_output"
        } else {
            "unknown"
        };
        
        println!("  Node {}: {} -> {} ({} bytes)", op.node_id, input_name, output_name, op.size_bytes);
    }
    
    println!("\n3. Analyzing Buffer Sharing Opportunities");
    println!("---------------------------------------");
    let sharing_ops = planner.buffer_sharing_analysis(&lifetimes);
    
    println!("Buffer sharing opportunities:");
    for op in &sharing_ops {
        // Look up the tensor names for display
        let first_name = if op.first_id == hash_string("input") {
            "input"
        } else if op.first_id == hash_string("conv_output") {
            "conv_output"
        } else {
            "unknown"
        };
        
        let second_name = if op.second_id == hash_string("output") {
            "output"
        } else if op.second_id == hash_string("pool_output") {
            "pool_output"
        } else {
            "unknown"
        };
        
        println!("  {} and {} can share {} bytes", first_name, second_name, op.size_bytes);
    }
    
    println!("\n4. Creating Memory Plan");
    println!("---------------------");
    // Instead of computing, use a mock plan for demonstration
    let mut plan = create_mock_memory_plan();
    
    println!("Initial memory plan:");
    println!("  Total memory required: {} bytes", plan.total_memory_bytes);
    println!("  Number of tensors: {}", plan.tensor_info.len());
    println!("  Number of allocations: {}", plan.allocations.len());
    
    println!("\n5. Optimizing Memory Layout");
    println!("--------------------------");
    let bytes_saved = planner.optimize_memory_layout(&mut plan)?;
    
    println!("Optimized memory plan:");
    println!("  Total memory required: {} bytes", plan.total_memory_bytes);
    println!("  Bytes saved: {} bytes", bytes_saved);
    println!("  Number of buffers: {}", plan.buffer_sizes.len());
    
    println!("\n6. Allocating Buffers");
    println!("--------------------");
    let mut allocator = SystemAllocator::new(None);
    let buffer_map = planner.allocate_buffers_from_plan(&plan, &mut allocator)?;
    
    println!("Allocated buffers:");
    println!("  Number of tensors with allocated memory: {}", buffer_map.len());
    println!("  Total allocator memory: {} bytes", allocator.allocated_memory());
    
    // Show the memory allocation for each tensor
    println!("\nTensor Memory Allocations:");
    for (tensor_id, allocation) in &plan.allocations {
        // Look up the tensor name for display
        let tensor_name = plan.tensor_info.get(tensor_id)
            .map(|info| info.name.as_str())
            .unwrap_or("unknown");
        
        println!("  {} ({}): offset={}, size={} bytes, buffer={}",
            tensor_name, tensor_id, allocation.offset, allocation.size_bytes, allocation.buffer_index);
    }
    
    Ok(())
}