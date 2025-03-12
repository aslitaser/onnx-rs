use std::sync::{Arc, Mutex};
use std::cell::UnsafeCell;

use crate::error::{Error, Result};
use crate::memory::allocator::{MemoryAllocator, MemoryBlock};

/// Guard for a workspace allocation
pub struct WorkspaceGuard {
    /// The actual memory block
    block: Option<MemoryBlock>,
    /// Block size in bytes
    size: usize,
    /// Reference to the workspace manager for deallocation
    manager: Arc<Mutex<WorkspaceManagerState>>,
}

impl WorkspaceGuard {
    /// Create a new workspace guard
    fn new(block: MemoryBlock, size: usize, manager: Arc<Mutex<WorkspaceManagerState>>) -> Self {
        Self {
            block: Some(block),
            size,
            manager,
        }
    }

    /// Get a slice to the workspace memory
    pub fn as_slice(&self) -> &[u8] {
        if let Some(block) = &self.block {
            unsafe { block.as_slice() }
        } else {
            &[]
        }
    }

    /// Get a mutable slice to the workspace memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Some(block) = &mut self.block {
            unsafe { block.as_slice_mut() }
        } else {
            &mut []
        }
    }

    /// Get a raw pointer to the workspace memory
    pub fn as_ptr(&self) -> *const u8 {
        if let Some(block) = &self.block {
            block.ptr().as_ptr()
        } else {
            std::ptr::null()
        }
    }

    /// Get a raw mutable pointer to the workspace memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        if let Some(block) = &mut self.block {
            block.ptr().as_ptr()
        } else {
            std::ptr::null_mut()
        }
    }

    /// Get the size of the workspace
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            // Return the block to the workspace manager
            if let Ok(mut manager) = self.manager.lock() {
                manager.current_usage -= self.size;
                manager.free_block(block);
            }
        }
    }
}

/// Internal state for the workspace manager
struct WorkspaceManagerState {
    /// Memory allocator
    allocator: Box<dyn MemoryAllocator>,
    /// Current workspace usage in bytes
    current_usage: usize,
    /// Peak workspace usage in bytes
    peak_usage: usize,
    /// Primary buffer (if allocated)
    primary_buffer: Option<MemoryBlock>,
    /// Current size of the primary buffer
    primary_buffer_size: usize,
    /// Auxiliary allocations for requests larger than the primary buffer
    auxiliary_blocks: Vec<MemoryBlock>,
}

impl WorkspaceManagerState {
    /// Create a new workspace manager state
    fn new(allocator: Box<dyn MemoryAllocator>, initial_size: usize) -> Self {
        Self {
            allocator,
            current_usage: 0,
            peak_usage: 0,
            primary_buffer: None,
            primary_buffer_size: initial_size,
            auxiliary_blocks: Vec::new(),
        }
    }

    /// Allocate or expand the primary buffer
    fn ensure_primary_buffer(&mut self, required_size: usize) -> Result<()> {
        // If we already have a primary buffer that's big enough, return it
        if let Some(buffer) = &self.primary_buffer {
            if self.primary_buffer_size >= required_size {
                return Ok(());
            }

            // Otherwise, deallocate it to allocate a bigger one
            self.allocator.deallocate(buffer.clone());
        }

        // Determine new size (grow exponentially)
        let new_size = if self.primary_buffer_size == 0 {
            required_size
        } else {
            let mut size = self.primary_buffer_size;
            while size < required_size {
                size *= 2;
            }
            size
        };

        // Allocate new buffer
        match self.allocator.allocate(new_size, 64) {
            Ok(block) => {
                self.primary_buffer = Some(block);
                self.primary_buffer_size = new_size;
                Ok(())
            }
            Err(e) => {
                self.primary_buffer = None;
                self.primary_buffer_size = 0;
                Err(e)
            }
        }
    }

    /// Allocate a workspace of the specified size
    fn allocate_workspace(&mut self, size_bytes: usize) -> Result<MemoryBlock> {
        // Check if requested size fits in primary buffer
        if size_bytes <= self.primary_buffer_size || size_bytes <= 1024 * 1024 {
            // Try to allocate or expand the primary buffer
            self.ensure_primary_buffer(size_bytes)?;

            // Create a view into the primary buffer
            if let Some(buffer) = &self.primary_buffer {
                let ptr = unsafe {
                    let base_ptr = buffer.ptr().as_ptr();
                    std::ptr::NonNull::new_unchecked(base_ptr)
                };

                return Ok(MemoryBlock::new(ptr, size_bytes, 64, 0));
            }
        }

        // For very large allocations, create a separate block
        let block = self.allocator.allocate(size_bytes, 64)?;
        self.auxiliary_blocks.push(block.clone());
        Ok(block)
    }

    /// Return a block to the workspace manager
    fn free_block(&mut self, block: MemoryBlock) {
        // For now, we just deallocate auxiliary blocks
        // Primary buffer allocations are just views, not actual allocations
        
        // Check if this is an auxiliary block
        if let Some(index) = self.auxiliary_blocks.iter().position(|b| {
            b.ptr().as_ptr() == block.ptr().as_ptr() && b.size() == block.size()
        }) {
            let block = self.auxiliary_blocks.remove(index);
            self.allocator.deallocate(block);
        }
    }

    /// Reset the workspace manager
    fn reset(&mut self) {
        // Deallocate auxiliary blocks
        for block in self.auxiliary_blocks.drain(..) {
            self.allocator.deallocate(block);
        }
        
        // Reset usage counters
        self.current_usage = 0;
    }
}

/// Thread-safe workspace memory manager
pub struct WorkspaceManager {
    /// Internal state
    state: Arc<Mutex<WorkspaceManagerState>>,
}

impl WorkspaceManager {
    /// Create a new workspace manager
    pub fn new(allocator: Box<dyn MemoryAllocator>, initial_size: usize) -> Self {
        let state = Arc::new(Mutex::new(WorkspaceManagerState::new(allocator, initial_size)));
        Self { state }
    }

    /// Get a workspace of the specified size
    pub fn get_workspace(&mut self, size_bytes: usize) -> Result<WorkspaceGuard> {
        // Lock the state
        let mut state = self.state.lock().map_err(|_| {
            Error::InvalidModel("Failed to lock workspace manager state".to_string())
        })?;

        // Allocate the workspace
        let block = state.allocate_workspace(size_bytes)?;

        // Update usage statistics
        state.current_usage += size_bytes;
        state.peak_usage = std::cmp::max(state.peak_usage, state.current_usage);

        // Create and return the guard
        Ok(WorkspaceGuard::new(block, size_bytes, self.state.clone()))
    }

    /// Reset the workspace manager
    pub fn reset(&mut self) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            Error::InvalidModel("Failed to lock workspace manager state".to_string())
        })?;

        state.reset();
        Ok(())
    }

    /// Get the current memory usage
    pub fn current_usage(&self) -> Result<usize> {
        let state = self.state.lock().map_err(|_| {
            Error::InvalidModel("Failed to lock workspace manager state".to_string())
        })?;

        Ok(state.current_usage)
    }

    /// Get the peak memory usage
    pub fn peak_usage(&self) -> Result<usize> {
        let state = self.state.lock().map_err(|_| {
            Error::InvalidModel("Failed to lock workspace manager state".to_string())
        })?;

        Ok(state.peak_usage)
    }
}

/// Scoped workspace allocation
/// 
/// This allows for allocating a workspace that is automatically 
/// deallocated when it goes out of scope, similar to a scoped lock.
pub struct ScopedWorkspace<'a> {
    /// The workspace guard
    guard: Option<WorkspaceGuard>,
    /// Reference to the workspace manager
    manager: &'a mut WorkspaceManager,
}

impl<'a> ScopedWorkspace<'a> {
    /// Create a new scoped workspace
    pub fn new(manager: &'a mut WorkspaceManager, size_bytes: usize) -> Result<Self> {
        let guard = manager.get_workspace(size_bytes)?;
        Ok(Self {
            guard: Some(guard),
            manager,
        })
    }

    /// Get a slice to the workspace memory
    pub fn as_slice(&self) -> &[u8] {
        if let Some(guard) = &self.guard {
            guard.as_slice()
        } else {
            &[]
        }
    }

    /// Get a mutable slice to the workspace memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Some(guard) = &mut self.guard {
            guard.as_mut_slice()
        } else {
            &mut []
        }
    }

    /// Get the size of the workspace
    pub fn size(&self) -> usize {
        if let Some(guard) = &self.guard {
            guard.size()
        } else {
            0
        }
    }

    /// Resize the workspace
    pub fn resize(&mut self, new_size_bytes: usize) -> Result<()> {
        // Drop the current guard
        self.guard = None;
        
        // Allocate a new workspace
        self.guard = Some(self.manager.get_workspace(new_size_bytes)?);
        Ok(())
    }
}

impl<'a> Drop for ScopedWorkspace<'a> {
    fn drop(&mut self) {
        // The guard will be dropped automatically
        self.guard = None;
    }
}

unsafe impl Send for WorkspaceManager {}
unsafe impl Sync for WorkspaceManager {}