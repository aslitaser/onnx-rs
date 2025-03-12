use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    /// Free blocks that can be reused
    free_blocks: Mutex<Vec<Arc<MemoryBlock>>>,
    /// Allocated blocks tracked with weak references
    allocated_blocks: Mutex<HashMap<usize, Arc<MemoryBlock>>>,
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
            free_blocks: Mutex::new(Vec::new()),
            allocated_blocks: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate a new block from the system
    fn allocate_new_block(&self) -> Result<Arc<MemoryBlock>> {
        // Check memory limit
        if let Some(limit) = self.memory_limit {
            let current = self.capacity.load(Ordering::Relaxed);
            let new_capacity = current.checked_add(self.block_size).ok_or_else(|| {
                Error::InvalidModel("Integer overflow calculating capacity".to_string())
            })?;
            
            if new_capacity > limit {
                return Err(Error::InvalidModel(format!(
                    "Pool allocator memory limit of {} bytes exceeded",
                    limit
                )));
            }
        }

        // Create layout
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

        // Return block wrapped in Arc for shared ownership
        Ok(Arc::new(MemoryBlock::new(ptr, self.block_size, self.alignment, 0)))
    }
}

impl MemoryAllocator for PoolAllocator {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        // Check if requested size fits in our blocks
        if size > self.block_size {
            return Err(Error::InvalidModel(format!(
                "Requested allocation size {} exceeds pool block size {}",
                size, self.block_size
            )));
        }

        // Check if requested alignment is compatible
        if alignment > self.alignment {
            return Err(Error::InvalidModel(format!(
                "Requested alignment {} exceeds pool alignment {}",
                alignment, self.alignment
            )));
        }

        // Try to get a block from the free list
        let mut free_blocks = match self.free_blocks.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::InvalidModel(
                "Failed to lock free blocks mutex".to_string()
            )),
        };
        
        let block = if let Some(block) = free_blocks.pop() {
            block
        } else {
            // Allocate a new block
            drop(free_blocks); // Release the lock before potentially lengthy allocation
            self.allocate_new_block()?
        };

        // Update allocation tracking
        self.allocated.fetch_add(self.block_size, Ordering::Relaxed);
        
        // Add to allocated blocks map
        let mut allocated_blocks = match self.allocated_blocks.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If we can't track it, put it back in the free list to avoid leak
                if let Ok(mut free_blocks) = self.free_blocks.lock() {
                    free_blocks.push(block.clone());
                }
                self.allocated.fetch_sub(self.block_size, Ordering::Relaxed);
                return Err(Error::InvalidModel(
                    "Failed to lock allocated blocks mutex".to_string()
                ));
            }
        };
        
        allocated_blocks.insert(block.ptr().as_ptr() as usize, block.clone());
        
        // Return a clone of the block
        Ok(MemoryBlock {
            ptr: block.ptr(),
            size: block.size(),
            alignment: block.alignment(),
            offset: block.offset(),
            _parent: Some(block),
        })
    }

    fn deallocate(&mut self, block: MemoryBlock) {
        let ptr = block.ptr().as_ptr() as usize;
        
        // Remove from allocated blocks map
        let maybe_block = self.allocated_blocks.lock().ok().and_then(|mut map| {
            map.remove(&ptr)
        });
        
        if let Some(block) = maybe_block {
            // If we found and removed the block from allocated set
            self.allocated.fetch_sub(self.block_size, Ordering::Relaxed);
            
            // Add back to free list if we're the only reference
            if Arc::strong_count(&block) == 1 {
                if let Ok(mut free_blocks) = self.free_blocks.lock() {
                    free_blocks.push(block);
                }
            }
        }
    }

    fn reset(&mut self) {
        // Move all allocated blocks to free list
        let allocated_opt = self.allocated_blocks.lock().ok().map(|mut allocated| {
            std::mem::take(&mut *allocated)
        });
        
        if let Some(allocated) = allocated_opt {
            let mut free_opt = self.free_blocks.lock().ok();
            if let Some(ref mut free) = free_opt {
                for (_, block) in allocated {
                    // Only put back in free list if we're the only reference
                    if Arc::strong_count(&block) == 1 {
                        free.push(block);
                    }
                }
            }
            // Reset allocated counter
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

impl Drop for PoolAllocator {
    fn drop(&mut self) {
        // Take all blocks from free list
        let mut free_blocks = self.free_blocks.lock().map(|guard| {
            std::mem::take(&mut *guard)
        }).unwrap_or_default();
        
        // Take all blocks from allocated map
        let mut allocated_blocks = self.allocated_blocks.lock().map(|guard| {
            std::mem::take(&mut *guard)
        }).unwrap_or_default();
        
        // Add allocated blocks to the free list
        for (_, block) in allocated_blocks {
            free_blocks.push(block);
        }
        
        // Deallocate all unique blocks where we have the only reference
        for block in free_blocks {
            if Arc::strong_count(&block) == 1 {
                unsafe {
                    let layout = Layout::from_size_align_unchecked(self.block_size, self.alignment);
                    alloc::dealloc(block.ptr().as_ptr(), layout);
                }
            }
        }
    }
}

/// Create a default memory allocator
pub fn create_default_allocator(memory_limit: Option<usize>) -> Box<dyn MemoryAllocator> {
    Box::new(SystemAllocator::new(memory_limit))
}