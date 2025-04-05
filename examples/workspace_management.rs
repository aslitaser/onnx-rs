use onnx::{
    memory::{
        WorkspaceManager, WorkspaceGuard, ScopedWorkspace,
        SystemAllocator, create_default_allocator
    },
    error::Result,
};

fn main() -> Result<()> {
    println!("Workspace Management and Profiling Example");
    println!("---------------------------------------");
    
    // Part 1: Basic Workspace Management
    basic_workspace_management()?;
    
    // Part 2: Workspace Profiling
    workspace_profiling()?;
    
    Ok(())
}

// Basic workspace management demonstrations
fn basic_workspace_management() -> Result<()> {
    println!("\nPart 1: Basic Workspace Management");
    println!("---------------------------------");
    
    // Create a memory allocator
    let allocator = create_default_allocator(None);
    
    // Create a workspace manager with 1MB initial buffer
    let mut workspace_manager = WorkspaceManager::new(allocator, 1024 * 1024);
    
    println!("Initial state:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    println!("\nAllocating workspaces of different sizes...");
    
    // Allocate a small workspace (should use primary buffer)
    let small_size = 32 * 1024; // 32KB
    let mut small_workspace = workspace_manager.get_workspace(small_size)?;
    
    // Fill it with some data
    fill_workspace(&mut small_workspace, 0xA);
    
    println!("After small allocation:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    // Allocate a medium workspace (should still use primary buffer)
    let medium_size = 256 * 1024; // 256KB
    let mut medium_workspace = workspace_manager.get_workspace(medium_size)?;
    
    // Fill it with some data
    fill_workspace(&mut medium_workspace, 0xB);
    
    println!("After medium allocation:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    // Allocate a large workspace (might use auxiliary allocation)
    let large_size = 8 * 1024 * 1024; // 8MB
    let mut large_workspace = workspace_manager.get_workspace(large_size)?;
    
    // Fill it with some data
    fill_workspace(&mut large_workspace, 0xC);
    
    println!("After large allocation:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    // Drop the workspaces in reverse order
    println!("\nFreeing workspaces...");
    
    drop(large_workspace);
    println!("After dropping large workspace:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    drop(medium_workspace);
    println!("After dropping medium workspace:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    drop(small_workspace);
    println!("After dropping small workspace:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    // Reset the workspace manager
    println!("\nResetting workspace manager...");
    workspace_manager.reset()?;
    
    println!("After reset:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    println!("\nUsing ScopedWorkspace");
    println!("-------------------");
    
    // Create a scoped workspace block
    {
        println!("Creating scoped workspace...");
        let mut scoped = ScopedWorkspace::new(&mut workspace_manager, 1024 * 1024)?;
        
        // Fill it with data
        fill_workspace_slice(scoped.as_mut_slice(), 0xD);
        
        println!("Within scope:");
        println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
        println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
        
        // Resize the workspace
        println!("Resizing scoped workspace...");
        scoped.resize(2 * 1024 * 1024)?;
        
        println!("After resize:");
        println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
        println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
        
        println!("Scope ending, workspace will be automatically freed");
    }
    
    println!("After scope exit:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    println!("\nDemonstrating multiple concurrent workspaces");
    println!("---------------------------------------");
    
    // Allocate multiple workspaces for different tasks
    let workspace1 = workspace_manager.get_workspace(100 * 1024)?; // 100KB for task 1
    let workspace2 = workspace_manager.get_workspace(200 * 1024)?; // 200KB for task 2
    let workspace3 = workspace_manager.get_workspace(150 * 1024)?; // 150KB for task 3
    
    println!("After allocating multiple workspaces:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    // Let them go out of scope automatically
    drop(workspace1);
    drop(workspace2);
    drop(workspace3);
    
    println!("After freeing all workspaces:");
    println!("  Current usage: {} bytes", workspace_manager.current_usage()?);
    println!("  Peak usage: {} bytes", workspace_manager.peak_usage()?);
    
    Ok(())
}

// Demonstrate workspace profiling capabilities
fn workspace_profiling() -> Result<()> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    
    use onnx::{
        execution::{ExecutionEngine, context::ExecutionOptions},
        model::OnnxModel,
        tools::profiler::{
            core as profiler_core,
            types::{ProfileEventType, WorkspaceAllocationEvent}
        },
    };
    
    println!("\nPart 2: Workspace Memory Profiling");
    println!("--------------------------------");
    
    println!("Simulating model execution with workspace tracking...");
    
    // Create a simple mock execution engine (in a real application, this would load a model)
    let options = ExecutionOptions::default();
    let mock_model = OnnxModel::default(); // This would be a real model in practice
    let engine = ExecutionEngine::new(mock_model, options)?;
    
    // Create a thread-safe collection to store allocation events for demonstration
    let allocation_events = Arc::new(Mutex::new(Vec::<WorkspaceAllocationEvent>::new()));
    let allocation_events_clone = allocation_events.clone();
    
    // Record workspace usage
    let mut peak_workspace = 0;
    let mut current_workspace = 0;
    let peak_workspace_ptr = Arc::new(Mutex::new(&mut peak_workspace));
    let current_workspace_ptr = Arc::new(Mutex::new(&mut current_workspace));
    
    // Track usage per operator
    let usage_per_operator = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
    let usage_per_operator_clone = usage_per_operator.clone();
    
    // Set up workspace allocation callback
    engine.set_workspace_allocation_callback(Box::new(move |size, node_id, op_type| {
        println!("Workspace allocated: {} bytes for operator '{}'", size, op_type);
        
        // Update current and peak usage
        {
            let mut current = current_workspace_ptr.lock().unwrap();
            **current += size;
            
            let mut peak = peak_workspace_ptr.lock().unwrap();
            **peak = std::cmp::max(**peak, **current);
        }
        
        // Track per-operator usage
        {
            let mut usage_map = usage_per_operator_clone.lock().unwrap();
            *usage_map.entry(op_type.clone()).or_insert(0) += size;
        }
        
        // Record allocation event
        let event = WorkspaceAllocationEvent {
            allocation_time: Instant::now().elapsed().as_nanos() as u64,
            deallocation_time: None, // Will be updated when deallocated
            size_bytes: size,
            node_id,
            op_type: op_type.clone(),
        };
        
        allocation_events_clone.lock().unwrap().push(event);
    }));
    
    // Simulate workspace allocations for different operators
    println!("\nSimulating workspace allocations for various operators:");
    
    // Simulate ConvOp workspace allocation
    {
        println!("Allocating workspace for Conv operation...");
        let workspace1 = engine.allocate_workspace(2 * 1024 * 1024)?; // 2MB for Conv
        
        // In a real scenario, this would be allocated during operation execution
        // Do some work with the workspace
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        // Deallocate (happens automatically when workspace is dropped)
        drop(workspace1);
    }
    
    // Simulate MatMulOp workspace allocation
    {
        println!("Allocating workspace for MatMul operation...");
        let workspace2 = engine.allocate_workspace(1 * 1024 * 1024)?; // 1MB for MatMul
        
        // Do some work with the workspace
        std::thread::sleep(std::time::Duration::from_millis(50));
        
        // Deallocate
        drop(workspace2);
    }
    
    // Simulate PoolOp workspace allocation
    {
        println!("Allocating workspace for Pool operation...");
        let workspace3 = engine.allocate_workspace(512 * 1024)?; // 512KB for Pool
        
        // Do some work with the workspace
        std::thread::sleep(std::time::Duration::from_millis(25));
        
        // Deallocate
        drop(workspace3);
    }
    
    // Simulate multiple concurrent allocations
    println!("\nSimulating concurrent workspace allocations:");
    {
        println!("Allocating multiple workspaces concurrently...");
        let workspace4 = engine.allocate_workspace(300 * 1024)?; // 300KB
        let workspace5 = engine.allocate_workspace(200 * 1024)?; // 200KB
        
        // Do some work with the workspaces
        std::thread::sleep(std::time::Duration::from_millis(75));
        
        // Deallocate in reverse order
        drop(workspace5);
        drop(workspace4);
    }
    
    // Print profiling results
    println!("\nWorkspace Profiling Results:");
    println!("---------------------------");
    println!("Peak workspace memory usage: {} bytes", peak_workspace);
    
    println!("\nWorkspace usage per operator:");
    {
        let usage_map = usage_per_operator.lock().unwrap();
        let mut usage_vec: Vec<_> = usage_map.iter().collect();
        usage_vec.sort_by(|a, b| b.1.cmp(a.1)); // Sort by usage (descending)
        
        for (op_type, usage) in usage_vec {
            println!("  {}: {} bytes", op_type, usage);
        }
    }
    
    println!("\nWorkspace allocation events:");
    {
        let events = allocation_events.lock().unwrap();
        for (i, event) in events.iter().enumerate() {
            println!("  Event #{}: {} bytes for '{}'", 
                i + 1, 
                event.size_bytes, 
                event.op_type
            );
        }
    }
    
    Ok(())
}

// Helper function to fill a workspace with a pattern
fn fill_workspace(workspace: &mut WorkspaceGuard, pattern: u8) {
    let slice = workspace.as_mut_slice();
    fill_workspace_slice(slice, pattern);
}

// Helper function to fill a slice with a pattern
fn fill_workspace_slice(slice: &mut [u8], pattern: u8) {
    // Fill the first few bytes for demonstration
    let fill_size = std::cmp::min(slice.len(), 1024);
    for i in 0..fill_size {
        slice[i] = pattern;
    }
    
    // Print the first few bytes
    println!("  Filled workspace with pattern 0x{:X}", pattern);
    println!("  First 8 bytes: {:?}", &slice[0..std::cmp::min(8, slice.len())]);
}