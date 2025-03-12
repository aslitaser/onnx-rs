use onnx::{
    memory::{SystemAllocator, ArenaAllocator, PoolAllocator, MemoryAllocator, MemoryBlock},
    error::Result,
};

fn main() -> Result<()> {
    // Example 1: System allocator
    println!("Testing SystemAllocator...");
    test_system_allocator()?;

    // Example 2: Arena allocator
    println!("\nTesting ArenaAllocator...");
    test_arena_allocator()?;

    // Example 3: Pool allocator
    println!("\nTesting PoolAllocator...");
    test_pool_allocator()?;

    Ok(())
}

fn test_system_allocator() -> Result<()> {
    let mut allocator = SystemAllocator::new(Some(1024 * 1024)); // 1MB limit
    
    println!("Initial state:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Allocate some memory
    let block1 = allocator.allocate(1024, 8)?;
    let block2 = allocator.allocate(2048, 16)?;
    
    println!("After allocations:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Use the memory blocks
    unsafe {
        let slice1 = block1.as_slice_mut();
        let slice2 = block2.as_slice_mut();
        
        // Fill with some data
        for (i, byte) in slice1.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        
        for (i, byte) in slice2.iter_mut().enumerate() {
            *byte = ((i + 128) % 256) as u8;
        }
        
        // Verify first few bytes
        println!("  Block1 first bytes: {:?}", &slice1[0..4]);
        println!("  Block2 first bytes: {:?}", &slice2[0..4]);
    }
    
    // Deallocate memory
    allocator.deallocate(block1);
    allocator.deallocate(block2);
    
    println!("After deallocation:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    Ok(())
}

fn test_arena_allocator() -> Result<()> {
    let mut allocator = ArenaAllocator::new(4096, 8)?;
    
    println!("Initial state:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Allocate some memory
    let block1 = allocator.allocate(1024, 8)?;
    let block2 = allocator.allocate(2048, 16)?;
    
    println!("After allocations:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Use the memory blocks
    unsafe {
        let slice1 = block1.as_slice_mut();
        let slice2 = block2.as_slice_mut();
        
        // Fill with some data
        for (i, byte) in slice1.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        
        for (i, byte) in slice2.iter_mut().enumerate() {
            *byte = ((i + 128) % 256) as u8;
        }
        
        // Verify first few bytes
        println!("  Block1 first bytes: {:?}", &slice1[0..4]);
        println!("  Block2 first bytes: {:?}", &slice2[0..4]);
    }
    
    // In arena allocator, individual deallocations don't actually free memory
    // Instead we reset the entire arena
    println!("Resetting arena...");
    allocator.reset();
    
    println!("After reset:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    Ok(())
}

fn test_pool_allocator() -> Result<()> {
    let mut allocator = PoolAllocator::new(1024, 8, Some(4096));
    
    println!("Initial state:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Allocate some memory
    let block1 = allocator.allocate(1024, 8)?;
    let block2 = allocator.allocate(1024, 8)?;
    let block3 = allocator.allocate(1024, 8)?;
    
    println!("After allocations:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Use one memory block
    unsafe {
        let slice = block1.as_slice_mut();
        
        // Fill with some data
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        
        // Verify first few bytes
        println!("  Block1 first bytes: {:?}", &slice[0..4]);
    }
    
    // Deallocate one block
    println!("Deallocating one block...");
    allocator.deallocate(block2);
    
    println!("After partial deallocation:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Allocate again - should reuse the deallocated block
    println!("Allocating again...");
    let block4 = allocator.allocate(1024, 8)?;
    
    println!("After reallocation:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    // Reset the allocator
    println!("Resetting pool...");
    allocator.reset();
    
    println!("After reset:");
    println!("  Available memory: {} bytes", allocator.available_memory());
    println!("  Allocated memory: {} bytes", allocator.allocated_memory());
    
    Ok(())
}