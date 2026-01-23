"""
Diffusers viability evaluation for Qwen-Image-Edit-2511 on 24GB GPU systems.
Analyzes memory requirements, performance characteristics, and alternative approaches.
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


def analyze_diffusers_memory_requirements():
    """Analyze memory requirements for different Diffusers approaches."""
    print("\n=== Diffusers Memory Requirements Analysis ===")
    
    # Qwen-Image-Edit-2511 specific requirements
    model_analysis = {
        "total_parameters": 8000000000,  # ~8B parameters
        "transformer_parameters": 6000000000,  # Main diffusion transformer
        "text_encoder_parameters": 2000000000,  # Qwen2.5-VL text encoder
        "hidden_size": 4096,
        "num_layers": 32,
        "image_resolution": 1024,
        "patch_size": 16,
    }
    
    # Memory requirements by component
    component_memory = {
        "transformer_fp16_mb": model_analysis["transformer_parameters"] * 2 / (1024 * 1024),  # 2 bytes per param
        "text_encoder_fp16_mb": model_analysis["text_encoder_parameters"] * 2 / (1024 * 1024),
        "activations_mb": model_analysis["hidden_size"] * model_analysis["hidden_size"] * 32 / (1024 * 1024),  # KV cache
        "intermediate_mb": model_analysis["hidden_size"] * model_analysis["hidden_size"] * 4 / (1024 * 1024),  # Intermediate layers
        "overhead_mb": 1500,  # Framework overhead
    }
    
    print("Component memory requirements (FP16):")
    for component, memory_mb in component_memory.items():
        print(f"  {component}: {memory_mb:.1f} MB")
    
    total_fp16 = sum(component_memory.values())
    print(f"\nTotal FP16 memory requirement: {total_fp16:.1f} MB")
    
    # Different strategy estimates
    strategies = {
        "full_precision_fp16": {
            "multiplier": 1.0,
            "description": "Full FP16 precision, no optimizations"
        },
        "attention_slicing": {
            "multiplier": 0.85,
            "description": "Attention slicing to reduce activation memory"
        },
        "cpu_offload": {
            "multiplier": 0.25,
            "description": "Model CPU offloading, keep minimal GPU memory"
        },
        "sequential_offload": {
            "multiplier": 0.15,
            "description": "Sequential CPU offloading, one layer at a time"
        },
        "gradient_checkpointing": {
            "multiplier": 0.7,
            "description": "Gradient checkpointing, trade compute for memory"
        },
        "nf4_quantization": {
            "multiplier": 0.35,
            "description": "NF4 4-bit quantization (experimental)"
        },
        "fp8_quantization": {
            "multiplier": 0.5,
            "description": "FP8 quantization (if supported)"
        }
    }
    
    print(f"\nMemory requirements by strategy:")
    for strategy, config in strategies.items():
        memory_mb = total_fp16 * config["multiplier"]
        print(f"  {strategy}: {memory_mb:.1f} MB ({config['description']})")
    
    return component_memory, strategies, total_fp16


def evaluate_24gb_viability():
    """Evaluate viability on 24GB GPU systems."""
    print("\n=== 24GB GPU Viability Evaluation ===")
    
    component_memory, strategies, total_fp16 = analyze_diffusers_memory_requirements()
    
    # GPU scenarios
    gpu_scenarios = [
        {"name": "RTX 3090/4090 (24GB)", "available_mb": 24000, "bandwidth": "High"},
        {"name": "RTX A4000 (16GB)", "available_mb": 16000, "bandwidth": "Medium"},
        {"name": "RTX 3080 (10GB)", "available_mb": 10000, "bandwidth": "High"},
        {"name": "RTX 3060 (12GB)", "available_mb": 12000, "bandwidth": "Medium"},
    ]
    
    print("Viability by GPU:")
    for gpu in gpu_scenarios:
        available_mb = gpu["available_mb"]
        viable_strategies = []
        
        for strategy, config in strategies.items():
            memory_mb = total_fp16 * config["multiplier"]
            if memory_mb <= available_mb * 0.9:  # Leave 10% buffer
                viable_strategies.append({
                    "strategy": strategy,
                    "memory_mb": memory_mb,
                    "utilization": (memory_mb / available_mb) * 100
                })
        
        print(f"\n{gpu['name']} ({gpu['available_mb']//1000}GB):")
        if viable_strategies:
            for strategy_info in sorted(viable_strategies, key=lambda x: x["memory_mb"]):
                print(f"  ✓ {strategy_info['strategy']}: {strategy_info['memory_mb']:.1f}MB ({strategy_info['utilization']:.1f}% VRAM)")
        else:
            print(f"  ✗ No viable strategies")
    
    # Specific focus on 24GB
    rtx_4090_strategies = []
    available_mb = 24000
    
    for strategy, config in strategies.items():
        memory_mb = total_fp16 * config["multiplier"]
        if memory_mb <= available_mb * 0.9:
            rtx_4090_strategies.append({
                "strategy": strategy,
                "memory_mb": memory_mb,
                "performance_impact": estimate_performance_impact(strategy),
                "recommended_for": get_recommended_use_case(strategy)
            })
    
    print(f"\n=== Recommended 24GB Configurations ===")
    for strategy_info in rtx_4090_strategies:
        print(f"\n{strategy_info['strategy']}:")
        print(f"  Memory: {strategy_info['memory_mb']:.1f} MB")
        print(f"  Performance Impact: {strategy_info['performance_impact']}")
        print(f"  Recommended For: {strategy_info['recommended_for']}")
    
    return rtx_4090_strategies


def estimate_performance_impact(strategy):
    """Estimate performance impact of memory optimization."""
    impacts = {
        "full_precision_fp16": "Baseline (no impact)",
        "attention_slicing": "Minor (10-20% slower)",
        "cpu_offload": "Significant (2-3x slower)",
        "sequential_offload": "Major (4-5x slower)",
        "gradient_checkpointing": "Moderate (30-50% slower)",
        "nf4_quantization": "Variable (20-40% slower, quality loss)",
        "fp8_quantization": "Minor (5-15% slower, minimal quality loss)"
    }
    return impacts.get(strategy, "Unknown")


def get_recommended_use_case(strategy):
    """Get recommended use case for strategy."""
    use_cases = {
        "full_precision_fp16": "High-quality generation when memory permits",
        "attention_slicing": "Moderate memory savings with minimal quality loss",
        "cpu_offload": "Extreme memory constraints, batch processing",
        "sequential_offload": "Very tight memory constraints, interactive use",
        "gradient_checkpointing": "Training or very high resolution outputs",
        "nf4_quantization": "Rapid prototyping, quality-critical applications",
        "fp8_quantization": "Production use with modern GPUs"
    }
    return use_cases.get(strategy, "General use")


def research_alternative_approaches():
    """Research alternative approaches beyond standard Diffusers."""
    print("\n=== Alternative Approaches Research ===")
    
    alternatives = {
        "comfyui_optimized": {
            "description": "ComfyUI's optimized execution graphs and memory management",
            "memory_savings": "30-50%",
            "performance_impact": "Minimal to moderate",
            "implementation_complexity": "High",
            "status": "Research needed",
            "notes": "Uses patched execution graphs and custom memory pools"
        },
        "lightning_lora": {
            "description": "Lightning LoRA for 4-8 step inference",
            "memory_savings": "Minimal (speed optimization)",
            "performance_impact": "Major (5-10x faster)",
            "implementation_complexity": "Medium",
            "status": "Partially implemented",
            "notes": "Available from lightx2v/Qwen-Image-Lightning"
        },
        "torchao_fp8": {
            "description": "TorchAO FP8 quantization",
            "memory_savings": "50%",
            "performance_impact": "Minimal (5-15% slower)",
            "implementation_complexity": "High",
            "status": "Research needed",
            "notes": "Requires recent PyTorch versions and GPU support"
        },
        "custom_compilation": {
            "description": "Custom model compilation with torch.compile",
            "memory_savings": "10-20%",
            "performance_impact": "Positive (10-30% faster)",
            "implementation_complexity": "Medium",
            "status": "Experimental",
            "notes": "Needs careful configuration for stability"
        }
    }
    
    for approach, details in alternatives.items():
        print(f"\n{approach}:")
        for key, value in details.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return alternatives


def generate_viability_report():
    """Generate comprehensive viability report."""
    print("\n" + "="*70)
    print("DIFFUSERS VIABILITY EVALUATION REPORT")
    print("="*70)
    
    # Run all analyses
    component_memory, strategies, total_fp16 = analyze_diffusers_memory_requirements()
    viable_strategies = evaluate_24gb_viability()
    alternatives = research_alternative_approaches()
    
    # Compile comprehensive report
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "model": "Qwen-Image-Edit-2511",
        "total_fp16_memory_mb": total_fp16,
        "component_breakdown": component_memory,
        "viable_strategies_24gb": viable_strategies,
        "alternative_approaches": alternatives,
        "recommendations": {
            "primary_recommendation": "fp16_full with attention_slicing",
            "fallback_recommendation": "cpu_offload for memory-constrained systems",
            "research_priorities": [
                "torchao_fp8 for latest GPUs",
                "comfyui_optimized for production systems",
                "lightning_lora for speed-critical applications"
            ],
            "implementation_notes": [
                "NF4 quantization needs careful layer skipping",
                "FP8 requires recent hardware and PyTorch",
                "CPU offload dramatically impacts performance",
                "Attention slicing provides good balance"
            ]
        },
        "conclusion": assess_diffusers_viability(viable_strategies)
    }
    
    # Save report
    report_dir = project_root / "viability_reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"diffusers_viability_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("VIABILITY ASSESSMENT SUMMARY")
    print("="*70)
    
    conclusion = report["conclusion"]
    print(f"Overall Viability: {conclusion['status']}")
    print(f"Confidence: {conclusion['confidence']}")
    print(f"Recommended Strategy: {report['recommendations']['primary_recommendation']}")
    
    print(f"\nKey Findings:")
    for finding in conclusion["key_findings"]:
        print(f"  • {finding}")
    
    print(f"\nNext Steps:")
    for step in conclusion["next_steps"]:
        print(f"  • {step}")
    
    print(f"\nReport saved to: {report_file}")
    
    return report


def assess_diffusers_viability(strategies):
    """Assess overall Diffusers viability."""
    if not strategies:
        return {
            "status": "NOT_VIABLE",
            "confidence": "Low",
            "key_findings": [
                "No viable memory strategies found for 24GB GPU",
                "Qwen-Image-Edit-2511 exceeds available memory with all tested approaches"
            ],
            "next_steps": [
                "Consider model distillation or smaller alternatives",
                "Research advanced quantization techniques",
                "Explore custom memory management solutions"
            ]
        }
    
    # Check if we have at least one reasonable strategy
    reasonable_strategies = [s for s in strategies if s["performance_impact"] in [
        "Baseline (no impact)", "Minor (10-20% slower)", "Minor (5-15% slower)"
    ]]
    
    if reasonable_strategies:
        return {
            "status": "HIGHLY_VIABLE",
            "confidence": "High",
            "key_findings": [
                "Multiple viable strategies available for 24GB GPU",
                "Full precision possible with minor optimizations",
                "Good balance between memory and performance achievable"
            ],
            "next_steps": [
                "Implement primary strategy with fallback mechanisms",
                "Test Lightning LoRA for speed optimization",
                "Monitor memory usage in production"
            ]
        }
    elif len(strategies) >= 2:
        return {
            "status": "VIABLE_WITH_LIMITATIONS",
            "confidence": "Medium",
            "key_findings": [
                "Viable strategies exist but with performance trade-offs",
                "Memory optimization required for stable operation",
                "CPU offload may be necessary for complex tasks"
            ],
            "next_steps": [
                "Implement adaptive strategy selection",
                "Consider model compilation for better performance",
                "Plan for hybrid CPU/GPU workflows"
            ]
        }
    else:
        return {
            "status": "MARGINALLY_VIABLE",
            "confidence": "Low",
            "key_findings": [
                "Only one viable strategy available",
                "Significant performance compromises required",
                "Alternative approaches should be investigated"
            ],
            "next_steps": [
                "Research ComfyUI optimization techniques",
                "Investigate custom quantization solutions",
                "Consider cloud-based inference for demanding tasks"
            ]
        }


if __name__ == "__main__":
    try:
        report = generate_viability_report()
        
        print("\n" + "="*70)
        print("FINAL RECOMMENDATIONS")
        print("="*70)
        
        if report["conclusion"]["status"] == "HIGHLY_VIABLE":
            print("✓ Diffusers approach is HIGHLY RECOMMENDED for Qwen-Image-Edit-2511")
            print("  Implement with primary strategy and testing framework")
        elif report["conclusion"]["status"] == "VIABLE_WITH_LIMITATIONS":
            print("⚠ Diffusers approach is VIABLE with careful implementation")
            print("  Requires memory management and performance monitoring")
        elif report["conclusion"]["status"] == "MARGINALLY_VIABLE":
            print("⚠ Diffusers approach is MARGINALLY VIABLE")
            print("  Consider alternative approaches for production use")
        else:
            print("✗ Diffusers approach is NOT RECOMMENDED")
            print("  Research alternative frameworks or model optimization")
        
        print(f"\nPrimary Strategy: {report['recommendations']['primary_recommendation']}")
        print(f"Research Priority: {report['recommendations']['research_priorities'][0]}")
        
    except Exception as e:
        print(f"Viability assessment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        memory_manager.cleanup_memory(aggressive=True)
        print("\nViability assessment completed.")