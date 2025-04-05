// Export module for profiler
// Contains functionality for exporting profiling data in various formats

use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::collections::HashMap;

use serde::{Serialize, Deserialize};
use serde_json::json;
use anyhow::{Result, anyhow};

use crate::model::NodeId;

use super::types::{
    ProfileResults, ProfileEventType, ProfileEvent, ExportFormat
};

/// Export profiling data to various formats
pub fn export_profile_data(profile_results: &ProfileResults, format: ExportFormat) -> Result<Vec<u8>> {
    match format {
        ExportFormat::Json => {
            serde_json::to_vec_pretty(profile_results)
                .map_err(|e| anyhow!("Failed to serialize profile data to JSON: {}", e))
        }
        ExportFormat::Csv => {
            // Implement CSV export functionality
            let mut output = Vec::new();
            // Headers
            writeln!(&mut output, "Operation,Type,Time (ms),Percentage")?;
            
            // Operation times
            let op_percentages = profile_results.performance.operation_time_percentage();
            for (op_name, time_ns) in &profile_results.performance.per_op_type_time_ns {
                let time_ms = *time_ns as f64 / 1_000_000.0;
                let percentage = op_percentages.get(op_name).copied().unwrap_or(0.0);
                    
                writeln!(&mut output, "{},{:.2},{:.2}%", op_name, time_ms, percentage)?;
            }
            
            Ok(output)
        }
        ExportFormat::ChromeTrace => {
            // Implement Chrome Trace Format (for chrome://tracing)
            let mut trace_events = Vec::new();
            
            // Convert profile events to chrome trace format
            for event in &profile_results.timeline {
                if let Some(duration) = event.duration {
                    let start_time_micros = 0; // This should actually be calculated from the start of profiling
                    
                    let trace_event = json!({
                        "name": event.name,
                        "cat": format!("{:?}", event.event_type),
                        "ph": "X", // Complete event (with duration)
                        "ts": start_time_micros,
                        "dur": duration.as_micros(),
                        "pid": 1, // Process ID
                        "tid": event.node_id.map(|id| id.0).unwrap_or(0), // Thread ID
                        "args": {
                            "node_id": event.node_id.map(|id| id.0),
                            "tensor_id": event.tensor_id.map(|id| id.0),
                        }
                    });
                    
                    trace_events.push(trace_event);
                }
            }
            
            serde_json::to_vec_pretty(&trace_events)
                .map_err(|e| anyhow!("Failed to serialize profile data to Chrome Trace format: {}", e))
        }
        ExportFormat::Markdown => {
            // Implement Markdown export
            let mut output = Vec::new();
            
            // Title
            writeln!(&mut output, "# ONNX Runtime Profile Report\n")?;
            
            // Summary
            writeln!(&mut output, "## Summary\n")?;
            writeln!(&mut output, "- **Total Execution Time**: {:.2} ms", 
                profile_results.performance.total_execution_time_ns as f64 / 1_000_000.0)?;
            writeln!(&mut output, "- **Peak Memory Usage**: {:.2} MB", 
                profile_results.performance.peak_memory_bytes as f64 / (1024.0 * 1024.0))?;
            writeln!(&mut output, "- **Model Operations**: {}", 
                profile_results.model_info.op_count)?;
            
            // Operation Breakdown
            writeln!(&mut output, "\n## Operation Time Breakdown\n")?;
            writeln!(&mut output, "| Operation | Time (ms) | Percentage |")?;
            writeln!(&mut output, "|-----------|-----------|------------|")?;
            
            let op_percentages = profile_results.performance.operation_time_percentage();
            let mut ops: Vec<(String, u64)> = profile_results.performance.per_op_type_time_ns.iter()
                .map(|(name, &time)| (name.clone(), time))
                .collect();
            
            ops.sort_by(|a, b| b.1.cmp(&a.1));
            
            for (op_name, time_ns) in ops {
                let time_ms = time_ns as f64 / 1_000_000.0;
                let percentage = op_percentages.get(&op_name).copied().unwrap_or(0.0);
                writeln!(&mut output, "| {} | {:.2} | {:.2}% |", op_name, time_ms, percentage)?;
            }
            
            // Critical Path
            if !profile_results.performance.critical_path.is_empty() {
                writeln!(&mut output, "\n## Critical Path\n")?;
                writeln!(&mut output, "| Node | Operation | Time (ms) |")?;
                writeln!(&mut output, "|------|-----------|-----------|")?;
                
                for &node_id in &profile_results.performance.critical_path {
                    let node_time = profile_results.performance.per_op_instance_time_ns
                        .get(&node_id)
                        .map(|&ns| ns as f64 / 1_000_000.0)
                        .unwrap_or(0.0);
                    
                    let node_name = format!("Node {}", node_id.0);
                    
                    writeln!(&mut output, "| {} | {} | {:.2} |", 
                        node_id.0, node_name, node_time)?;
                }
            }
            
            Ok(output)
        }
        ExportFormat::Protobuf => {
            // Placeholder for protobuf export
            Err(anyhow!("Protobuf export not yet implemented"))
        }
    }
}

/// Generate a flamegraph from profiling data
pub fn generate_flamegraph(profile_results: &ProfileResults, output_path: &Path) -> Result<()> {
    // Make sure the directory exists
    if let Some(parent) = output_path.parent() {
        create_dir_all(parent)?;
    }
    
    // Generate folded stack format for flamegraph
    let mut folded_output = Vec::new();
    
    // Process timeline events to create stack traces
    for event in &profile_results.timeline {
        if event.event_type == ProfileEventType::OpExecution && event.duration.is_some() {
            let duration_micros = event.duration.unwrap().as_micros();
            
            // Build stack trace
            let mut stack = Vec::new();
            
            // Add operation type (extracted from name)
            if let Some(op_type) = event.name.split_whitespace().next() {
                stack.push(op_type.to_string());
            }
            
            // Add node ID if available
            if let Some(node_id) = event.node_id {
                stack.push(format!("Node_{}", node_id.0));
            }
            
            // Convert stack to folded format: stack_frame1;stack_frame2;... duration
            let stack_str = stack.join(";");
            writeln!(&mut folded_output, "{} {}", stack_str, duration_micros)?;
        }
    }
    
    // Write folded output to intermediate file
    let folded_path = output_path.with_extension("folded");
    let mut folded_file = File::create(&folded_path)?;
    folded_file.write_all(&folded_output)?;
    
    // Generate flamegraph using external flamegraph tool if available
    // For simplicity, we'll just write the folded format and suggest using the flamegraph.pl script
    
    println!("Folded stack format written to: {}", folded_path.display());
    println!("To generate a flamegraph, use flamegraph.pl:");
    println!("  cat {} | flamegraph.pl > {}", folded_path.display(), output_path.display());
    
    Ok(())
}

/// Converts a series of profiling events to a flamegraph-compatible stack trace
fn convert_events_to_stacks(events: &[ProfileEvent]) -> HashMap<String, u64> {
    let mut stacks = HashMap::new();
    
    // Process events with parent-child relationships
    for event in events {
        if event.duration.is_none() {
            continue;
        }
        
        let mut stack_parts = Vec::new();
        let mut current_event = Some(event);
        
        // Build stack from leaf to root
        while let Some(evt) = current_event {
            stack_parts.push(evt.name.clone());
            
            // Find parent event
            if let Some(parent_id) = evt.parent_id {
                current_event = events.iter().find(|e| e.id == parent_id);
            } else {
                current_event = None;
            }
        }
        
        // Reverse to get root-to-leaf order
        stack_parts.reverse();
        
        // Join stack frames and add to map
        let stack_str = stack_parts.join(";");
        let duration = event.duration.unwrap().as_micros() as u64;
        
        *stacks.entry(stack_str).or_insert(0) += duration;
    }
    
    stacks
}