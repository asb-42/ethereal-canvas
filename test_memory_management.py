"""
Test script to evaluate different memory management strategies for Qwen-Image-Edit-2511.
This script tests loading strategies, memory usage, and performance on 24GB GPU systems.
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
from modules.backends.enhanced_image_edit import EnhancedImageEditBackend
from modules.runtime.paths import OUTPUTS_DIR, ensure_runtime_dirs


def test_memory_estimations():
    """Test memory requirement estimations."""
    print("\n=== Testing Memory Estimations ===")
    
    estimates = memory_manager.estimate_required_memory("Qwen-Image-Edit-2511")
    
    print("Estimated memory requirements:")
    for strategy, memory_mb in estimates.items():
        print(f"  {strategy}: {memory_mb:.1f} MB")
    
    available = memory_manager.get_available_memory_mb()
    print(f"\nAvailable GPU memory: {available:.1f} MB")
    
    recommendations = memory_manager.get_recommended_strategy(available)
    print(f"Recommended strategies: {[s.value for s in recommendations]}")
    
    return estimates, available


def test_loading_strategies():
    """Test different loading strategies."""
    print("\n=== Testing Loading Strategies ===")
    
    backend = EnhancedImageEditBackend()
    
    # Test each strategy individually
    results = {}
    
    for strategy in memory_manager.fallback_chain:
        if strategy not in memory_manager.loading_configs:
            continue
        
        print(f"\n--- Testing {strategy.value} ---")
        
        try:
            # Cleanup before each test
            backend.cleanup()
            memory_manager.cleanup_memory(aggressive=True)
            
            # Try to load with this specific strategy
            start_time = time.time()
            success = backend.switch_strategy(strategy)
            load_time = time.time() - start_time
            
            if success:
                memory_info = backend.get_memory_info()
                
                results[strategy.value] = {
                    "success": True,
                    "load_time": load_time,
                    "peak_memory_mb": memory_info.get("gpu_allocated_mb", 0),
                    "vram_usage_percent": (memory_info.get("gpu_allocated_mb", 0) / 
                                        memory_info.get("total_mb", 1)) * 100
                }
                
                print(f"✓ Success - Load time: {load_time:.2f}s")
                print(f"  Memory: {memory_info.get('gpu_allocated_mb', 0):.1f} MB")
            else:
                results[strategy.value] = {
                    "success": False,
                    "load_time": load_time,
                    "error": "Failed to load"
                }
                
                print(f"✗ Failed to load")
        
        except Exception as e:
            results[strategy.value] = {
                "success": False,
                "error": str(e)
            }
            
            print(f"✗ Error: {e}")
    
    return results


def test_inference_performance():
    """Test inference performance with different strategies."""
    print("\n=== Testing Inference Performance ===")
    
    # Find or create a test image
    ensure_runtime_dirs()
    test_image_path = OUTPUTS_DIR / "test_input.png"
    
    if not test_image_path.exists():
        print(f"Creating test image at {test_image_path}")
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='lightgray')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 400], fill='darkblue', outline='black', width=2)
        draw.text((150, 250), "TEST IMAGE", fill='white')
        
        test_image_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(test_image_path)
    
    test_prompt = "transform the blue rectangle into a green circle"
    
    backend = EnhancedImageEditBackend()
    inference_results = {}
    
    # Test only successful strategies from loading tests
    for strategy in [LoadStrategy.FP16_FULL, LoadStrategy.NF4_QUANTIZED, LoadStrategy.CPU_OFFLOAD]:
        if strategy not in memory_manager.loading_configs:
            continue
        
        print(f"\n--- Inference test with {strategy.value} ---")
        
        try:
            # Load with strategy
            if not backend.switch_strategy(strategy):
                print(f"Skipping {strategy.value} - failed to load")
                continue
            
            # Run inference multiple times for average
            times = []
            memory_usage = []
            
            for i in range(3):
                start_time = time.time()
                
                result_path = backend.edit(
                    test_prompt, 
                    str(test_image_path),
                    num_inference_steps=8  # Use fewer steps for faster testing
                )
                
                inference_time = time.time() - start_time
                
                if result_path:
                    times.append(inference_time)
                    
                    memory_info = backend.get_memory_info()
                    memory_usage.append(memory_info.get("gpu_allocated_mb", 0))
                    
                    print(f"  Run {i+1}: {inference_time:.2f}s, Memory: {memory_info.get('gpu_allocated_mb', 0):.1f} MB")
                
                # Small delay between runs
                time.sleep(0.5)
            
            if times:
                avg_time = sum(times) / len(times)
                avg_memory = sum(memory_usage) / len(memory_usage)
                
                inference_results[strategy.value] = {
                    "success": True,
                    "avg_inference_time": avg_time,
                    "avg_memory_usage": avg_memory,
                    "runs_completed": len(times)
                }
                
                print(f"✓ Average: {avg_time:.2f}s, Memory: {avg_memory:.1f} MB")
            else:
                inference_results[strategy.value] = {
                    "success": False,
                    "error": "No successful inferences"
                }
        
        except Exception as e:
            inference_results[strategy.value] = {
                "success": False,
                "error": str(e)
            }
            print(f"✗ Inference failed: {e}")
    
    return inference_results


def test_automatic_fallback():
    """Test the automatic fallback mechanism."""
    print("\n=== Testing Automatic Fallback ===")
    
    # Try with unrealistic memory requirements to trigger fallback
    low_memory_backend = EnhancedImageEditBackend()
    
    # Simulate low memory scenario by starting with the most demanding strategy
    print("Testing fallback from most demanding to least demanding strategies...")
    
    try:
        success = low_memory_backend.load()
        
        if success:
            info = low_memory_backend.get_memory_info()
            final_strategy = info.get("strategy", "Unknown")
            memory_used = info.get("gpu_allocated_mb", 0)
            
            print(f"✓ Fallback successful - Final strategy: {final_strategy}")
            print(f"  Memory used: {memory_used:.1f} MB")
            
            return {
                "success": True,
                "final_strategy": final_strategy,
                "memory_used": memory_used,
                "loading_history": info.get("loading_history", [])
            }
        else:
            print("✗ All fallback strategies failed")
            return {"success": False}
    
    except Exception as e:
        print(f"✗ Fallback test failed: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        low_memory_backend.cleanup()


def run_comprehensive_test():
    """Run all tests and generate a comprehensive report."""
    print("=== Comprehensive Memory Management Test ===")
    print(f"Device: {memory_manager.device}")
    print(f"Python: {sys.version}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    estimates, available = test_memory_estimations()
    loading_results = test_loading_strategies()
    inference_results = test_inference_performance()
    fallback_results = test_automatic_fallback()
    
    # Compile comprehensive report
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "device": memory_manager.device,
        "available_memory_mb": available,
        "memory_estimates": estimates,
        "loading_tests": loading_results,
        "inference_tests": inference_results,
        "fallback_test": fallback_results,
        "summary": {
            "successful_strategies": [
                strategy for strategy, result in loading_results.items() 
                if result.get("success", False)
            ],
            "recommended_strategy": None,
            "minimum_memory_requirement": None,
            "viability_assessment": "Unknown"
        }
    }
    
    # Determine recommendations
    successful_strategies = report["summary"]["successful_strategies"]
    
    if successful_strategies:
        # Find the least memory-intensive successful strategy
        strategy_memory = {}
        for strategy in successful_strategies:
            if strategy in loading_results and loading_results[strategy].get("peak_memory_mb"):
                strategy_memory[strategy] = loading_results[strategy]["peak_memory_mb"]
        
        if strategy_memory:
            best_strategy = min(strategy_memory, key=strategy_memory.get)
            report["summary"]["recommended_strategy"] = best_strategy
            report["summary"]["minimum_memory_requirement"] = strategy_memory[best_strategy]
        
        # Assess viability for 24GB GPU
        max_required = max(strategy_memory.values()) if strategy_memory else 0
        if max_required < 20000:  # 20GB
            report["summary"]["viability_assessment"] = "VIABLE"
        elif max_required < 24000:  # 24GB
            report["summary"]["viability_assessment"] = "MARGINAL"
        else:
            report["summary"]["viability_assessment"] = "NOT_VIABLE"
    
    # Save report
    report_dir = project_root / "memory_test_reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"memory_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Available GPU Memory: {available:.1f} MB")
    print(f"Successful Strategies: {successful_strategies}")
    
    if report["summary"]["recommended_strategy"]:
        print(f"Recommended Strategy: {report['summary']['recommended_strategy']}")
        print(f"Minimum Memory Required: {report['summary']['minimum_memory_requirement']:.1f} MB")
    
    print(f"24GB Viability: {report['summary']['viability_assessment']}")
    print(f"Report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    try:
        report = run_comprehensive_test()
        
        # Print final recommendation
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)
        
        if report["summary"]["viability_assessment"] == "VIABLE":
            print("✓ Qwen-Image-Edit-2511 is VIABLE on this system")
            print(f"Recommended strategy: {report['summary']['recommended_strategy']}")
        elif report["summary"]["viability_assessment"] == "MARGINAL":
            print("⚠ Qwen-Image-Edit-2511 is MARGINAL on this system")
            print("May work with careful memory management")
        else:
            print("✗ Qwen-Image-Edit-2511 is NOT VIABLE on this system")
            print("Consider using FP8/Lightning optimizations or smaller models")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        memory_manager.cleanup_memory(aggressive=True)
        print("\nTest completed and memory cleaned up.")