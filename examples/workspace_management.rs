use onnx::{
    memory::{
        WorkspaceManager, WorkspaceGuard, ScopedWorkspace,
        SystemAllocator, create_default_allocator
    },
    error::Result,
};

fn main() -> Result<()> {
    println!("Workspace Management Example");
    println!("---------------------------");
    
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