use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::execution::context::OptimizationLevel;
use crate::model::ExecutionGraph;

/// Result of running an optimization pass
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Name of the pass
    pub name: String,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Duration of the pass
    pub duration: Duration,
    /// Whether the pass made any changes
    pub changed: bool,
}

/// Statistics from running optimization passes
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Results from each pass
    pub pass_results: Vec<PassResult>,
    /// Total number of optimizations applied
    pub total_optimizations: usize,
    /// Total time spent optimizing
    pub total_duration: Duration,
}

impl OptimizationStats {
    /// Create a new optimization stats object
    pub fn new() -> Self {
        Self {
            pass_results: Vec::new(),
            total_optimizations: 0,
            total_duration: Duration::new(0, 0),
        }
    }
    
    /// Add a pass result
    pub fn add_pass_result(&mut self, result: PassResult) {
        self.total_optimizations += result.optimizations_applied;
        self.total_duration += result.duration;
        self.pass_results.push(result);
    }
}

/// Trait for graph optimization passes
pub trait OptimizationPass: Send + Sync {
    /// Name of the pass
    fn name(&self) -> &str;
    
    /// Run the pass on the graph
    fn run(&self, graph: &mut ExecutionGraph) -> Result<PassResult>;
    
    /// Whether this pass requires fixed input shapes
    fn requires_fixed_input_shapes(&self) -> bool {
        false
    }
    
    /// Dependencies of this pass (names of passes that must run before this one)
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }
}

/// Graph optimizer that applies optimization passes
pub struct GraphOptimizer {
    /// Registered optimization passes
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl GraphOptimizer {
    /// Create a new graph optimizer
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }
    
    /// Register an optimization pass
    pub fn register_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }
    
    /// Register default passes
    pub fn with_default_passes(mut self) -> Self {
        for pass in Self::create_default_passes(OptimizationLevel::Standard) {
            self.register_pass(pass);
        }
        self
    }
    
    /// Create default passes for the given optimization level
    pub fn create_default_passes(level: OptimizationLevel) -> Vec<Box<dyn OptimizationPass>> {
        use crate::optimization::passes::constant_folding::ConstantFolding;
        use crate::optimization::passes::fusion::OperatorFusion;
        
        match level {
            OptimizationLevel::None => Vec::new(),
            OptimizationLevel::Basic => {
                vec![
                    Box::new(ConstantFolding::new()) as Box<dyn OptimizationPass>,
                ]
            },
            OptimizationLevel::Standard => {
                vec![
                    Box::new(ConstantFolding::new()) as Box<dyn OptimizationPass>,
                    Box::new(OperatorFusion::new()) as Box<dyn OptimizationPass>,
                ]
            },
            OptimizationLevel::Aggressive => {
                vec![
                    Box::new(ConstantFolding::new()) as Box<dyn OptimizationPass>,
                    Box::new(OperatorFusion::new()) as Box<dyn OptimizationPass>,
                    // More aggressive passes would go here
                ]
            },
        }
    }
    
    /// Optimize the graph with the given optimization level
    pub fn optimize(&self, graph: &mut ExecutionGraph, level: OptimizationLevel) -> Result<OptimizationStats> {
        let passes = Self::create_default_passes(level);
        self.run_passes(graph, &passes)
    }
    
    /// Run multiple passes in order
    pub fn run_passes(&self, graph: &mut ExecutionGraph, passes: &[Box<dyn OptimizationPass>]) -> Result<OptimizationStats> {
        let mut stats = OptimizationStats::new();
        
        // Sort passes by dependencies
        let sorted_passes = self.sort_passes_by_dependencies(passes)?;
        
        // Run passes in order
        for pass in sorted_passes {
            let result = pass.run(graph)?;
            stats.add_pass_result(result);
        }
        
        Ok(stats)
    }
    
    /// Sort passes by dependencies
    fn sort_passes_by_dependencies<'a>(&self, passes: &'a [Box<dyn OptimizationPass>]) -> Result<Vec<&'a Box<dyn OptimizationPass>>> {
        let mut sorted_passes = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        
        // Build dependency graph
        let mut dep_graph: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut pass_map: HashMap<&str, &Box<dyn OptimizationPass>> = HashMap::new();
        
        for pass in passes {
            let name = pass.name();
            dep_graph.insert(name, pass.dependencies());
            pass_map.insert(name, pass);
        }
        
        // Topological sort
        for &pass in passes {
            let name = pass.name();
            if !visited.contains(name) {
                self.dfs_sort(name, &dep_graph, &pass_map, &mut visited, &mut visiting, &mut sorted_passes)?;
            }
        }
        
        Ok(sorted_passes)
    }
    
    /// DFS helper for topological sort
    fn dfs_sort<'a>(
        &self,
        pass_name: &str,
        dep_graph: &HashMap<&str, Vec<&str>>,
        pass_map: &HashMap<&str, &'a Box<dyn OptimizationPass>>,
        visited: &mut HashSet<&str>,
        visiting: &mut HashSet<&str>,
        sorted_passes: &mut Vec<&'a Box<dyn OptimizationPass>>,
    ) -> Result<()> {
        if visited.contains(pass_name) {
            return Ok(());
        }
        
        if visiting.contains(pass_name) {
            return Err(Error::InvalidGraph(format!(
                "Cycle detected in optimization pass dependencies: {}", pass_name
            )));
        }
        
        visiting.insert(pass_name);
        
        // Visit dependencies
        if let Some(deps) = dep_graph.get(pass_name) {
            for &dep in deps {
                if !pass_map.contains_key(dep) {
                    return Err(Error::InvalidGraph(format!(
                        "Optimization pass '{}' depends on '{}', which is not registered",
                        pass_name, dep
                    )));
                }
                
                self.dfs_sort(dep, dep_graph, pass_map, visited, visiting, sorted_passes)?;
            }
        }
        
        visiting.remove(pass_name);
        visited.insert(pass_name);
        
        if let Some(&pass) = pass_map.get(pass_name) {
            sorted_passes.push(pass);
        }
        
        Ok(())
    }
}