use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

use crate::error::{Error, Result};

/// A memory block representing an allocation
#[derive(Debug)]
pub struct MemoryBlock {
    /// Pointer to the memory block
    ptr: NonNull<u8>,
    /// Size of the memory block in bytes
    size: usize,
    /// Alignment of the memory block
    alignment: usize,
    /// Offset from the original allocation (for suballocations)
    offset: usize,
    /// Reference to parent block (if this is a sub-allocation)
    _parent: Option<Arc<MemoryBlock>>,
}

impl MemoryBlock {
    /// Create a new memory block
    pub fn new(ptr: NonNull<u8>, size: usize, alignment: usize, offset: usize) -> Self {
        Self {
            ptr,
            size,
            alignment,
            offset,
            _parent: None,
        }
    }

    /// Create a subblock from a parent block
    pub fn new_subblock(
        ptr: NonNull<u8>, 
        size: usize, 
        alignment: usize, 
        offset: usize,
        parent: Arc<MemoryBlock>,
    ) -> Self {
        Self {
            ptr,
            size,
            alignment,
            offset,
            _parent: Some(parent),
        }
    }

    /// Get a pointer to the memory block
    pub fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Get the size of the memory block
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the alignment of the memory block
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get the offset from the original allocation
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get a slice to the memory block
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
    }

    /// Get a mutable slice to the memory block
    pub unsafe fn as_slice_mut(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
    }
}

// We need to implement Clone manually to handle the Arc reference correctly
impl Clone for MemoryBlock {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            size: self.size,
            alignment: self.alignment,
            offset: self.offset,
            _parent: self._parent.clone(),
        }
    }
}

/// Memory allocator trait for ONNX runtime
pub trait MemoryAllocator: Send + Sync {
    /// Allocate a block of memory with the specified size and alignment
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock>;

    /// Deallocate a memory block
    fn deallocate(&mut self, block: MemoryBlock);

    /// Reset the allocator, deallocating all memory
    fn reset(&mut self);

    /// Get the amount of available memory
    fn available_memory(&self) -> usize;

    /// Get the amount of allocated memory
    fn allocated_memory(&self) -> usize;
}

/// Thread-safe system allocator that uses the Rust allocator
pub struct SystemAllocator {
    allocated: AtomicUsize,
    memory_limit: Option<usize>,
    allocations: Mutex<HashMap<usize, Layout>>,
}

impl SystemAllocator {
    /// Create a new system allocator
    pub fn new(memory_limit: Option<usize>) -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            memory_limit,
            allocations: Mutex::new(HashMap::new()),
        }
    }
}

impl MemoryAllocator for SystemAllocator {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        // Ensure size is not zero
        let size = std::cmp::max(1, size);

        // Check memory limit before allocation
        if let Some(limit) = self.memory_limit {
            let current = self.allocated.load(Ordering::Relaxed);
            if current.checked_add(size).map_or(true, |total| total > limit) {
                return Err(Error::InvalidModel(format!(
                    "Memory limit of {} bytes exceeded with allocation of {} bytes (current: {})",
                    limit, size, current
                )));
            }
        }

        // Create layout
        let layout = match Layout::from_size_align(size, alignment) {
            Ok(layout) => layout,
            Err(e) => return Err(Error::InvalidModel(format!(
                "Invalid memory layout: size={}, alignment={}, error={}",
                size, alignment, e
            ))),
        };

        // Allocate memory
        let ptr = unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                return Err(Error::InvalidModel(format!(
                    "Failed to allocate memory: size={}, alignment={}",
                    size, alignment
                )));
            }
            NonNull::new_unchecked(ptr)
        };

        // Track allocation
        self.allocated.fetch_add(size, Ordering::Relaxed);
        
        // Lock the mutex to update allocations map
        let mut allocations = match self.allocations.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If lock fails, we need to free the memory to avoid leaks
                unsafe {
                    alloc::dealloc(ptr.as_ptr(), layout);
                }
                return Err(Error::InvalidModel(
                    "Failed to lock allocations mutex".to_string()
                ));
            }
        };
        
        allocations.insert(ptr.as_ptr() as usize, layout);

        // Return block
        Ok(MemoryBlock::new(ptr, size, alignment, 0))
    }

    fn deallocate(&mut self, block: MemoryBlock) {
        // Lock the mutex to update allocations map
        let mut allocations = match self.allocations.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If we can't lock, we can't safely free, but log the issue
                // This is a serious error and might lead to memory leaks
                // In a real system, this would be logged and the process might need to be restarted
                return;
            }
        };
        
        let ptr = block.ptr().as_ptr() as usize;
        if let Some(layout) = allocations.remove(&ptr) {
            unsafe {
                alloc::dealloc(block.ptr().as_ptr(), layout);
            }
            // Only decrease counter if we actually freed something
            self.allocated.fetch_sub(block.size(), Ordering::Relaxed);
        }
    }

    fn reset(&mut self) {
        // Lock the mutex to update allocations map
        let allocations_opt = self.allocations.lock().ok().map(|allocations| {
            std::mem::take(&mut *allocations)
        });
        
        if let Some(allocations) = allocations_opt {
            // Deallocate all memory
            for (ptr, layout) in allocations {
                unsafe {
                    alloc::dealloc(ptr as *mut u8, layout);
                }
            }
            // Reset the counter
            self.allocated.store(0, Ordering::Relaxed);
        }
    }

    fn available_memory(&self) -> usize {
        match self.memory_limit {
            Some(limit) => limit.saturating_sub(self.allocated.load(Ordering::Relaxed)),
            None => usize::MAX - self.allocated.load(Ordering::Relaxed),
        }
    }

    fn allocated_memory(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

impl Drop for SystemAllocator {
    fn drop(&mut self) {
        self.reset();
    }
}

/// Arena allocator that pre-allocates a block of memory and manages sub-allocations
pub struct ArenaAllocator {
    memory_block: Option<Arc<MemoryBlock>>,
    capacity: usize,
    allocated: AtomicUsize,
    offset: Mutex<usize>,
    alignment: usize,
}

impl ArenaAllocator {
    /// Create a new arena allocator with the specified capacity
    pub fn new(capacity: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(capacity, alignment)
            .map_err(|e| Error::InvalidModel(format!("Invalid arena layout: {}", e)))?;

        let ptr = unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                return Err(Error::InvalidModel(format!(
                    "Failed to allocate arena memory: size={}, alignment={}",
                    capacity, alignment
                )));
            }
            NonNull::new_unchecked(ptr)
        };

        let memory_block = Arc::new(MemoryBlock::new(ptr, capacity, alignment, 0));

        Ok(Self {
            memory_block: Some(memory_block),
            capacity,
            allocated: AtomicUsize::new(0),
            offset: Mutex::new(0),
            alignment,
        })
    }

    /// Align the offset to the specified alignment
    fn align_offset(&self, offset: usize, alignment: usize) -> usize {
        (offset + alignment - 1) & !(alignment - 1)
    }
}

impl MemoryAllocator for ArenaAllocator {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        let memory_block = self.memory_block.as_ref().ok_or_else(|| {
            Error::InvalidModel("Arena allocator has been reset".to_string())
        })?;

        // Lock the offset mutex
        let mut offset_guard = match self.offset.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel(
                "Failed to lock offset mutex".to_string()
            )),
        };

        // Align offset
        let aligned_offset = self.align_offset(*offset_guard, std::cmp::max(alignment, self.alignment));
        let end_offset = aligned_offset.checked_add(size).ok_or_else(|| {
            Error::InvalidModel("Integer overflow calculating end offset".to_string())
        })?;

        // Check if we have enough space
        if end_offset > self.capacity {
            return Err(Error::InvalidModel(format!(
                "Arena allocator out of memory: capacity={}, requested={}, offset={}",
                self.capacity, size, aligned_offset
            )));
        }

        // Calculate pointer for this allocation
        let ptr = unsafe {
            let base_ptr = memory_block.ptr().as_ptr();
            NonNull::new_unchecked(base_ptr.add(aligned_offset))
        };

        // Update state
        *offset_guard = end_offset;
        self.allocated.fetch_add(size, Ordering::Relaxed);

        // Create a subblock referencing the parent
        let parent_ref = memory_block.clone();
        Ok(MemoryBlock::new_subblock(
            ptr, size, alignment, aligned_offset, parent_ref
        ))
    }

    fn deallocate(&mut self, _block: MemoryBlock) {
        // Individual deallocations are not supported in the arena allocator
        // Memory will be reclaimed on reset() or drop()
        
        // Do not decrement allocated counter, as we can't reclaim the space
        
        // This behavior should be clearly documented with a warning that
        // deallocate does not actually free memory in ArenaAllocator
    }

    fn reset(&mut self) {
        // Reset allocator state without deallocating the underlying memory
        if let Ok(mut offset) = self.offset.lock() {
            *offset = 0;
        }
        self.allocated.store(0, Ordering::Relaxed);
    }

    fn available_memory(&self) -> usize {
        let current_offset = self.offset.lock().map(|o| *o).unwrap_or(self.capacity);
        self.capacity.saturating_sub(current_offset)
    }

    fn allocated_memory(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        if let Some(block) = self.memory_block.take() {
            // Only deallocate if we are the only reference to this block
            if Arc::strong_count(&block) == 1 {
                let layout = Layout::from_size_align(self.capacity, self.alignment)
                    .expect("Invalid layout in arena allocator drop");
                unsafe {
                    alloc::dealloc(block.ptr().as_ptr(), layout);
                }
            }
        }
    }
}

/// A memory block wrapper with reference counting for the pool allocator
struct PoolBlock {
    /// The memory block (never changes once created)
    block: MemoryBlock,
    /// Reference count (how many allocations are using this block)
    ref_count: AtomicUsize,
    /// Whether this block is currently in the pool
    in_pool: AtomicBool,
}

impl PoolBlock {
    /// Create a new pool block
    fn new(block: MemoryBlock) -> Self {
        Self {
            block,
            ref_count: AtomicUsize::new(0),
            in_pool: AtomicBool::new(true),
        }
    }

    /// Increment the reference count and return whether this was successful
    /// Returns false if the block is not in the pool
    fn acquire(&self) -> bool {
        // First check if the block is in the pool
        if !self.in_pool.load(Ordering::Acquire) {
            return false;
        }

        // Try to increment reference count, starting from 0
        let result = self.ref_count.compare_exchange(
            0, 1, 
            Ordering::AcqRel, 
            Ordering::Relaxed
        );

        // If successful, mark as not in pool
        if result.is_ok() {
            self.in_pool.store(false, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Increment the reference count (for when a block is shared)
    fn add_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement the reference count and return the new count
    fn release(&self) -> usize {
        let prev = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        prev - 1
    }

    /// Return the block to the pool
    fn return_to_pool(&self) {
        // Make sure reference count is zero
        debug_assert_eq!(self.ref_count.load(Ordering::Relaxed), 0);
        
        // Mark as in pool
        self.in_pool.store(true, Ordering::Release);
    }

    /// Get a reference to the underlying memory block
    fn get_block(&self) -> MemoryBlock {
        self.block.clone()
    }
}

/// Pool allocator that maintains pools of fixed-size blocks
pub struct PoolAllocator {
    /// Block size for this pool
    block_size: usize,
    /// Alignment for allocations
    alignment: usize,
    /// Memory limit
    memory_limit: Option<usize>,
    /// Total capacity
    capacity: AtomicUsize,
    /// Total allocated memory
    allocated: AtomicUsize,
    /// Pool of memory blocks protected by a mutex
    pool: Arc<Mutex<Vec<Arc<PoolBlock>>>>,
    /// Map of allocated blocks to their pool blocks
    block_map: Arc<Mutex<HashMap<usize, Arc<PoolBlock>>>>,
}

impl PoolAllocator {
    /// Create a new pool allocator
    pub fn new(block_size: usize, alignment: usize, memory_limit: Option<usize>) -> Self {
        Self {
            block_size,
            alignment,
            memory_limit,
            capacity: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
            pool: Arc::new(Mutex::new(Vec::new())),
            block_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Allocate a new block from the system
    fn allocate_new_block(&self) -> Result<Arc<PoolBlock>> {
        // Check memory limit with overflow protection
        if let Some(limit) = self.memory_limit {
            let current = self.capacity.load(Ordering::Relaxed);
            let new_capacity = current.checked_add(self.block_size).ok_or_else(|| {
                Error::InvalidModel("Integer overflow calculating capacity".to_string())
            })?;
            
            if new_capacity > limit {
                return Err(Error::InvalidModel(format!(
                    "Pool allocator memory limit of {} bytes exceeded (current: {}, requested: {})",
                    limit, current, self.block_size
                )));
            }
        }

        // Create layout with overflow protection
        let layout = Layout::from_size_align(self.block_size, self.alignment)
            .map_err(|e| Error::InvalidModel(format!("Invalid pool layout: {}", e)))?;

        // Allocate memory
        let ptr = unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                return Err(Error::InvalidModel(format!(
                    "Failed to allocate pool memory: size={}, alignment={}",
                    self.block_size, self.alignment
                )));
            }
            NonNull::new_unchecked(ptr)
        };

        // Update capacity
        self.capacity.fetch_add(self.block_size, Ordering::Relaxed);

        // Create and return block wrapped in Arc for shared ownership
        let memory_block = MemoryBlock::new(ptr, self.block_size, self.alignment, 0);
        let pool_block = Arc::new(PoolBlock::new(memory_block));
        
        Ok(pool_block)
    }
}

impl MemoryAllocator for PoolAllocator {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        // Check if requested size fits in our blocks with clear error message
        if size > self.block_size {
            return Err(Error::InvalidModel(format!(
                "Requested allocation size {} bytes exceeds pool block size {} bytes",
                size, self.block_size
            )));
        }

        // Check if requested alignment is compatible with clear error message
        if alignment > self.alignment {
            return Err(Error::InvalidModel(format!(
                "Requested alignment {} exceeds pool alignment {}",
                alignment, self.alignment
            )));
        }

        // First try to reuse a block from the pool
        let mut allocated_pool_block: Option<Arc<PoolBlock>> = None;
        
        // Acquire pool lock
        let mut pool_guard = match self.pool.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel(
                "Failed to lock pool mutex".to_string()
            )),
        };
        
        // Look for an available block in the pool
        let mut i = 0;
        while i < pool_guard.len() {
            let pool_block = &pool_guard[i];
            
            // Try to acquire this block
            if pool_block.acquire() {
                // Successfully acquired the block
                allocated_pool_block = Some(pool_block.clone());
                break;
            }
            
            i += 1;
        }
        
        // If no block was found in the pool, allocate a new one
        if allocated_pool_block.is_none() {
            // Release the pool lock before potentially lengthy allocation
            drop(pool_guard);
            
            // Allocate a new block
            let new_block = self.allocate_new_block()?;
            
            // Mark it as acquired
            new_block.acquire();
            
            // Add it to the pool
            match self.pool.lock() {
                Ok(mut guard) => {
                    guard.push(new_block.clone());
                },
                Err(_) => {
                    // If we can't add to the pool, we'll still use the block,
                    // but it will leak when deallocated since it won't be in the pool
                    // Log warning in a real implementation
                }
            }
            
            allocated_pool_block = Some(new_block);
        }
        
        // Get the allocated block
        let pool_block = allocated_pool_block.unwrap();
        
        // Update tracking 
        self.allocated.fetch_add(self.block_size, Ordering::Relaxed);
        
        // Update block map for deallocation lookup
        let ptr_key = pool_block.block.ptr().as_ptr() as usize;
        match self.block_map.lock() {
            Ok(mut guard) => {
                guard.insert(ptr_key, pool_block.clone());
            },
            Err(_) => {
                // If we can't update the map, return an error
                // This would make deallocate impossible, which would lead to a memory leak
                pool_block.release(); // Decrement the reference count
                // If the ref count becomes 0, we should return it to the pool
                if pool_block.ref_count.load(Ordering::Relaxed) == 0 {
                    pool_block.return_to_pool();
                }
                self.allocated.fetch_sub(self.block_size, Ordering::Relaxed);
                return Err(Error::InvalidModel(
                    "Failed to lock block map mutex".to_string()
                ));
            }
        };
        
        // Get the memory block to return
        let memory_block = pool_block.get_block();
        
        // Set parent reference to ensure the pool block stays alive as long as the memory block
        Ok(MemoryBlock::new_subblock(
            memory_block.ptr(),
            memory_block.size(),
            memory_block.alignment(),
            memory_block.offset(),
            Arc::new(memory_block)
        ))
    }

    fn deallocate(&mut self, block: MemoryBlock) {
        let ptr_key = block.ptr().as_ptr() as usize;
        
        // Look up the pool block in our map
        let pool_block = match self.block_map.lock() {
            Ok(mut guard) => guard.remove(&ptr_key),
            Err(_) => {
                // Can't access the map, log error in real implementation
                // We can't deallocate properly, which might cause a memory leak
                return;
            }
        };
        
        if let Some(pool_block) = pool_block {
            // Update allocation tracking
            self.allocated.fetch_sub(self.block_size, Ordering::Relaxed);
            
            // Release the reference and check if it's the last one
            let new_count = pool_block.release();
            
            // If the reference count is zero, return it to the pool
            if new_count == 0 {
                pool_block.return_to_pool();
            }
        }
    }

    fn reset(&mut self) {
        // Acquire locks for both collections
        let block_map = match self.block_map.lock() {
            Ok(guard) => guard,
            Err(_) => return, // Can't reset if we can't lock
        };
        
        // We'll return blocks to the pool only if their ref count is 0
        // Get all blocks from the map
        let block_map_arc = self.block_map.clone();
        let pool_arc = self.pool.clone();
        
        // Use a background thread to avoid deadlocks when acquiring multiple locks
        std::thread::spawn(move || {
            if let Ok(mut block_map) = block_map_arc.lock() {
                // Clear the map, moving all blocks out
                let blocks = std::mem::take(&mut *block_map);
                
                // For each block, check ref count and return to pool if zero
                for (_, block) in blocks {
                    if block.ref_count.load(Ordering::Relaxed) == 0 {
                        block.return_to_pool();
                    } else {
                        // Re-insert into map if still referenced
                        let ptr_key = block.block.ptr().as_ptr() as usize;
                        block_map.insert(ptr_key, block);
                    }
                }
            }
        });
        
        // Reset allocation counter (actual count is maintained by reference counts)
        self.allocated.store(0, Ordering::Relaxed);
    }

    fn available_memory(&self) -> usize {
        match self.memory_limit {
            Some(limit) => limit.saturating_sub(self.allocated.load(Ordering::Relaxed)),
            None => usize::MAX - self.allocated.load(Ordering::Relaxed),
        }
    }

    fn allocated_memory(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

impl Drop for PoolAllocator {
    fn drop(&mut self) {
        // This is a critical section to ensure all memory is properly deallocated
        // to prevent memory leaks
        
        // First, try to get all blocks from the pool
        let pool_blocks = match self.pool.lock() {
            Ok(guard) => {
                // Take all blocks from the pool
                std::mem::take(&mut *guard.clone())
            },
            Err(_) => {
                // If we can't lock the pool, we can't clean up properly
                // In a real implementation, this would be logged as an error
                Vec::new()
            }
        };
        
        // Next, try to get all blocks from the block map
        let map_blocks = match self.block_map.lock() {
            Ok(guard) => {
                // Take all blocks from the map
                let mut blocks = Vec::new();
                for (_, block) in guard.clone().iter() {
                    blocks.push(block.clone());
                }
                blocks
            },
            Err(_) => {
                // If we can't lock the map, we can't clean up properly
                // In a real implementation, this would be logged as an error
                Vec::new()
            }
        };
        
        // Combine all blocks from both collections
        let mut all_blocks = pool_blocks;
        all_blocks.extend(map_blocks);
        
        // Create a set to deduplicate blocks
        let mut unique_blocks = std::collections::HashSet::new();
        
        // Deallocate all unique blocks where we have the only reference
        for block in all_blocks {
            // Use the pointer as a unique key for deduplication
            let ptr = block.block.ptr().as_ptr() as usize;
            
            // Only process each unique block once
            if unique_blocks.insert(ptr) {
                // Check if this is the only strong reference to the block
                // This is a critical check to avoid double-free errors
                if Arc::strong_count(&block) == 1 {
                    unsafe {
                        let layout = Layout::from_size_align_unchecked(self.block_size, self.alignment);
                        alloc::dealloc(block.block.ptr().as_ptr(), layout);
                    }
                }
            }
        }
    }
}

/// Create a default memory allocator
pub fn create_default_allocator(memory_limit: Option<usize>) -> Box<dyn MemoryAllocator> {
    Box::new(SystemAllocator::new(memory_limit))
}