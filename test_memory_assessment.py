"""
Simplified memory management test without full model download.
Tests the memory management components and strategy selection.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.memory import memory_manager, memory_profiler, LoadStrategy


def test_memory_profiler():
    """Test memory profiling capabilities."""
    print("\n=== Testing Memory Profiler ===")
    
    # Take baseline snapshot
    baseline = memory_profiler.take_snapshot("baseline")
    print(f"Baseline snapshot taken")
    print(f"  Process RAM: {baseline.process_ram_mb:.1f} MB")
    print(f"  System RAM: {baseline.system_ram_percent:.1f}%")
    
    # Test memory estimation
    model_config = {
        "parameter_count": 8000000000,  # 8B parameters
        "hidden_size": 4096,
        "num_layers": 32
    }
    
    estimates = memory_profiler.estimate_model_memory(model_config)
    print(f"\nMemory estimates for 8B parameter model:")
    for strategy, memory_mb in estimates.items():
        print(f"  {strategy}: {memory_mb:.1f} MB")
    
    return baseline, estimates


def test_loading_configs():
    """Test loading configuration creation."""
    print("\n=== Testing Loading Configurations ===")
    
    print(f"Device: {memory_manager.device}")
    print(f"Available strategies: {list(memory_manager.loading_configs.keys())}")
    
    for strategy, config in memory_manager.loading_configs.items():
        print(f"\n{strategy.value}:")
        print(f"  torch_dtype: {config.torch_dtype}")
        print(f"  low_cpu_mem_usage: {config.low_cpu_mem_usage}")
        print(f"  enable_attention_slicing: {config.enable_attention_slicing}")
        print(f"  enable_xformers: {config.enable_xformers}")
        print(f"  enable_model_cpu_offload: {config.enable_model_cpu_offload}")
        print(f"  enable_sequential_cpu_offload: {config.enable_sequential_cpu_offload}")
        if config.quantization_config:
            print(f"  quantization: {type(config.quantization_config).__name__}")
    
    return memory_manager.loading_configs


def test_strategy_recommendations():
    """Test strategy recommendations based on available memory."""
    print("\n=== Testing Strategy Recommendations ===")
    
    available = memory_manager.get_available_memory_mb()
    print(f"Available GPU memory: {available:.1f} MB")
    
    estimates = memory_manager.estimate_required_memory("Qwen-Image-Edit-2511")
    print(f"Memory estimates:")
    for strategy, memory_mb in estimates.items():
        print(f"  {strategy}: {memory_mb:.1f} MB")
    
    recommendations = memory_manager.get_recommended_strategy(available)
    print(f"Recommended strategies: {[s.value for s in recommendations]}")
    
    # Test with different memory scenarios
    scenarios = [4000, 8000, 16000, 24000, 32000]
    print(f"\nStrategy recommendations by available memory:")
    for memory_mb in scenarios:
        recs = memory_manager.get_recommended_strategy(memory_mb)
        print(f"  {memory_mb:4d} MB: {[s.value for s in recs]}")
    
    return recommendations


def test_oom_detection():
    """Test OOM error detection."""
    print("\n=== Testing OOM Detection ===")
    
    # Test various error messages
    test_errors = [
        "CUDA out of memory. Tried to allocate 256.00 MiB",
        "out of memory",
        "DefaultCPUAllocator: not enough memory",
        "Ran out of memory with size 1024.00 MiB",
        "Some other error message"
    ]
    
    for error_msg in test_errors:
        is_oom = memory_manager.is_oom_error(Exception(error_msg))
        print(f"  '{error_msg[:50]}...' -> OOM: {is_oom}")
    
    return True


def test_memory_cleanup():
    """Test memory cleanup functionality."""
    print("\n=== Testing Memory Cleanup ===")
    
    # Take snapshot before cleanup
    before = memory_profiler.take_snapshot("before_cleanup")
    
    # Perform cleanup
    memory_manager.cleanup_memory(aggressive=False)
    print("Standard cleanup performed")
    
    # Take snapshot after cleanup
    after = memory_profiler.take_snapshot("after_cleanup")
    
    # Perform aggressive cleanup
    memory_manager.cleanup_memory(aggressive=True)
    print("Aggressive cleanup performed")
    
    final = memory_profiler.take_snapshot("after_aggressive")
    
    print(f"Memory changes:")
    print(f"  Process RAM: {before.process_ram_mb:.1f} -> {after.process_ram_mb:.1f} -> {final.process_ram_mb:.1f} MB")
    print(f"  System RAM: {before.system_ram_percent:.1f}% -> {after.system_ram_percent:.1f}% -> {final.system_ram_percent:.1f}%")
    
    return final


def test_fallback_chain():
    """Test fallback chain ordering and availability."""
    print("\n=== Testing Fallback Chain ===")
    
    print(f"Default fallback chain:")
    for i, strategy in enumerate(memory_manager.fallback_chain):
        available = strategy in memory_manager.loading_configs
        print(f"  {i+1}. {strategy.value} - {'Available' if available else 'Not Available'}")
    
    # Test fallback chain filtering
    available_strategies = [s for s in memory_manager.fallback_chain if s in memory_manager.loading_configs]
    print(f"\nAvailable strategies for fallback: {[s.value for s in available_strategies]}")
    
    return available_strategies


def generate_memory_report():
    """Generate a comprehensive memory management report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MEMORY MANAGEMENT REPORT")
    print("="*60)
    
    # Run all tests
    baseline, estimates = test_memory_profiler()
    configs = test_loading_configs()
    recommendations = test_strategy_recommendations()
    test_oom_detection()
    final_snapshot = test_memory_cleanup()
    available_strategies = test_fallback_chain()
    
    # Compile report
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "device": memory_manager.device,
        "system_info": {
            "available_gpu_memory_mb": memory_manager.get_available_memory_mb(),
            "baseline_process_ram_mb": baseline.process_ram_mb,
            "baseline_system_ram_percent": baseline.system_ram_percent,
            "final_process_ram_mb": final_snapshot.process_ram_mb,
            "final_system_ram_percent": final_snapshot.system_ram_percent,
        },
        "memory_estimates": estimates,
        "available_strategies": [s.value for s in available_strategies],
        "configurations": {
            strategy.value: {
                "torch_dtype": config.torch_dtype,
                "quantization": type(config.quantization_config).__name__ if config.quantization_config else None,
                "cpu_offload": config.enable_model_cpu_offload,
                "sequential_offload": config.enable_sequential_cpu_offload,
                "attention_slicing": config.enable_attention_slicing,
                "xformers": config.enable_xformers,
            }
            for strategy, config in configs.items()
        },
        "recommendations": {
            "current_memory_mb": memory_manager.get_available_memory_mb(),
            "recommended_strategies": [s.value for s in recommendations],
            "minimum_memory_mb": min(estimates.values()) if estimates else None,
        },
        "viability_assessment": assess_24gb_viability(estimates, memory_manager.get_available_memory_mb())
    }
    
    # Save report
    report_dir = project_root / "memory_test_reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"memory_assessment_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Device: {report['device']}")
    print(f"Available GPU Memory: {report['system_info']['available_gpu_memory_mb']:.1f} MB")
    print(f"Process RAM Usage: {report['system_info']['final_process_ram_mb']:.1f} MB")
    print(f"Available Strategies: {len(report['available_strategies'])}")
    print(f"24GB Viability: {report['viability_assessment']['status']}")
    print(f"Report saved to: {report_file}")
    
    return report


def assess_24gb_viability(estimates, available_memory):
    """Assess viability on 24GB GPU systems."""
    max_required = max(estimates.values()) if estimates else float('inf')
    
    if max_required <= 20000:  # 20GB
        status = "VIABLE"
        confidence = "High"
        recommended_config = "fp16_full"
    elif max_required <= 24000:  # 24GB
        status = "MARGINAL"
        confidence = "Medium"
        recommended_config = "nf4_quantized or cpu_offload"
    else:
        status = "NOT_VIABLE"
        confidence = "Low"
        recommended_config = "Consider smaller models or cloud inference"
    
    return {
        "status": status,
        "confidence": confidence,
        "max_required_mb": max_required,
        "recommended_config": recommended_config,
        "notes": f"Requires {max_required:.1f} MB, have {available_memory:.1f} MB available"
    }


if __name__ == "__main__":
    try:
        report = generate_memory_report()
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        assessment = report["viability_assessment"]
        print(f"Status: {assessment['status']}")
        print(f"Confidence: {assessment['confidence']}")
        print(f"Recommended Configuration: {assessment['recommended_config']}")
        print(f"Notes: {assessment['notes']}")
        
        if report["recommendations"]["recommended_strategies"]:
            print(f"\nFor this system, recommended strategies in order:")
            for strategy in report["recommendations"]["recommended_strategies"]:
                print(f"  1. {strategy}")
        
    except Exception as e:
        print(f"Assessment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        memory_manager.cleanup_memory(aggressive=True)
        print("\nAssessment completed.")