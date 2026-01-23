"""
Memory profiling and monitoring utilities for Qwen-Image-Edit-2511 model.
Provides runtime memory inspection and analysis capabilities.
"""

import gc
import os
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    try:
        nvml.nvmlInit()
    except:
        NVML_AVAILABLE = False
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0
    system_ram_mb: float = 0.0
    system_ram_percent: float = 0.0
    process_ram_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    vram_used_mb: float = 0.0


@dataclass
class ModelMemoryProfile:
    """Comprehensive memory profile for a model."""
    model_name: str
    load_strategy: str
    peak_memory_mb: float
    stable_memory_mb: float
    load_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    snapshots: List[MemorySnapshot] = None
    
    def __post_init__(self):
        if self.snapshots is None:
            self.snapshots = []


class MemoryProfiler:
    """Main memory profiling utility."""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.profiles: List[ModelMemoryProfile] = []
        self.device_count = 0
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
        elif NVML_AVAILABLE:
            try:
                self.device_count = nvml.nvmlDeviceGetCount()
            except:
                self.device_count = 0
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        snapshot = MemorySnapshot(timestamp=time.time())
        
        # System RAM
        memory = psutil.virtual_memory()
        snapshot.system_ram_mb = memory.total / (1024 * 1024)
        snapshot.system_ram_percent = memory.percent
        
        process = psutil.Process(os.getpid())
        snapshot.process_ram_mb = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_allocated = torch.cuda.max_memory_allocated()
                
                snapshot.gpu_allocated_mb = allocated / (1024 * 1024)
                snapshot.gpu_reserved_mb = reserved / (1024 * 1024)
                snapshot.gpu_max_allocated_mb = max_allocated / (1024 * 1024)
            except:
                pass
        
        # VRAM via NVML
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                snapshot.vram_total_mb = info.total / (1024 * 1024)
                snapshot.vram_free_mb = info.free / (1024 * 1024)
                snapshot.vram_used_mb = info.used / (1024 * 1024)
            except:
                pass
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def profile_model_loading(self, model_name: str, load_strategy: str, 
                            load_fn, *args, **kwargs) -> ModelMemoryProfile:
        """
        Profile model loading with comprehensive memory tracking.
        
        Args:
            model_name: Name of the model being loaded
            load_strategy: Description of loading strategy
            load_fn: Function that loads the model
            *args, **kwargs: Arguments to pass to load_fn
        
        Returns:
            ModelMemoryProfile with detailed memory usage data
        """
        print(f"Profiling model load: {model_name} with strategy: {load_strategy}")
        
        profile = ModelMemoryProfile(
            model_name=model_name,
            load_strategy=load_strategy,
            peak_memory_mb=0.0,
            stable_memory_mb=0.0,
            load_time_seconds=0.0,
            success=False
        )
        
        # Clear memory before profiling
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Baseline snapshot
        baseline = self.take_snapshot("baseline")
        
        start_time = time.time()
        peak_memory = 0.0
        
        try:
            # Load model with monitoring
            model = load_fn(*args, **kwargs)
            
            # Monitor memory for a few seconds to get stable reading
            stable_snapshots = []
            for i in range(5):
                time.sleep(0.5)
                snapshot = self.take_snapshot(f"stable_{i}")
                stable_snapshots.append(snapshot.gpu_allocated_mb)
                
                # Track peak memory
                if snapshot.gpu_allocated_mb > peak_memory:
                    peak_memory = snapshot.gpu_allocated_mb
            
            load_time = time.time() - start_time
            
            # Calculate stable memory (average of last 3 snapshots)
            stable_memory = sum(stable_snapshots[-3:]) / len(stable_snapshots[-3:])
            
            profile.peak_memory_mb = peak_memory
            profile.stable_memory_mb = stable_memory
            profile.load_time_seconds = load_time
            profile.success = True
            profile.snapshots = self.snapshots[baseline:]  # Only snapshots from this profile
            
            print(f"✓ Load successful - Peak: {peak_memory:.1f}MB, Stable: {stable_memory:.1f}MB, Time: {load_time:.2f}s")
            
            return model, profile
            
        except Exception as e:
            load_time = time.time() - start_time
            profile.load_time_seconds = load_time
            profile.success = False
            profile.error_message = str(e)
            
            # Final snapshot even on failure
            final_snapshot = self.take_snapshot("error")
            if final_snapshot.gpu_allocated_mb > peak_memory:
                peak_memory = final_snapshot.gpu_allocated_mb
            profile.peak_memory_mb = peak_memory
            
            print(f"✗ Load failed after {load_time:.2f}s: {e}")
            
            raise e
    
    def get_vram_info(self) -> Dict[str, float]:
        """Get current VRAM information."""
        info = {
            "total_mb": 0.0,
            "free_mb": 0.0,
            "used_mb": 0.0,
            "available_mb": 0.0
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                
                info["total_mb"] = total / (1024 * 1024)
                info["used_mb"] = allocated / (1024 * 1024)
                info["available_mb"] = (total - allocated) / (1024 * 1024)
                info["free_mb"] = (total - reserved) / (1024 * 1024)
            except:
                pass
        
        return info
    
    def estimate_model_memory(self, model_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate memory requirements for a model based on configuration.
        
        Args:
            model_config: Model configuration dictionary
        
        Returns:
            Estimated memory requirements in MB
        """
        # Base estimation for transformer models
        params = model_config.get("parameter_count", 0)
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_layers", 32)
        
        # Memory estimates for different components
        transformer_memory = (params * 2) / (1024 * 1024)  # 2 bytes per parameter for FP16
        text_encoder_memory = hidden_size * num_layers * 8 / (1024 * 1024)  # Rough estimate
        activation_memory = hidden_size * hidden_size * 32 / (1024 * 1024)  # Attention KV cache
        
        # Different loading strategies
        estimates = {
            "fp16_full": transformer_memory + text_encoder_memory + activation_memory,
            "fp8_optimized": transformer_memory * 0.5 + text_encoder_memory * 0.5 + activation_memory,
            "nf4_quantized": transformer_memory * 0.25 + text_encoder_memory * 0.25 + activation_memory,
            "cpu_offload": transformer_memory * 0.1 + text_encoder_memory * 0.1 + activation_memory * 0.5
        }
        
        return estimates
    
    def print_memory_summary(self, profile: Optional[ModelMemoryProfile] = None):
        """Print a formatted memory summary."""
        if profile:
            print(f"\n=== Memory Profile: {profile.model_name} ===")
            print(f"Strategy: {profile.load_strategy}")
            print(f"Success: {profile.success}")
            if profile.error_message:
                print(f"Error: {profile.error_message}")
            print(f"Load Time: {profile.load_time_seconds:.2f}s")
            print(f"Peak Memory: {profile.peak_memory_mb:.1f} MB")
            print(f"Stable Memory: {profile.stable_memory_mb:.1f} MB")
            
            if profile.snapshots:
                baseline = profile.snapshots[0]
                peak = profile.snapshots[-1]
                print(f"Memory Increase: {peak.gpu_allocated_mb - baseline.gpu_allocated_mb:.1f} MB")
        
        # Current system state
        print(f"\n=== Current System State ===")
        vram_info = self.get_vram_info()
        if vram_info["total_mb"] > 0:
            print(f"GPU Memory: {vram_info['used_mb']:.1f}/{vram_info['total_mb']:.1f} MB ({vram_info['used_mb']/vram_info['total_mb']*100:.1f}%)")
        
        current_snapshot = self.take_snapshot("current")
        print(f"Process RAM: {current_snapshot.process_ram_mb:.1f} MB")
        print(f"System RAM: {current_snapshot.system_ram_percent:.1f}%")
    
    def save_profile(self, profile: ModelMemoryProfile, filepath: str):
        """Save a memory profile to file."""
        import json
        
        # Convert to dict for JSON serialization
        data = asdict(profile)
        data["snapshots"] = [asdict(s) for s in profile.snapshots]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Profile saved to {filepath}")
    
    def compare_profiles(self, profiles: List[ModelMemoryProfile]) -> Dict[str, Any]:
        """Compare multiple memory profiles."""
        if not profiles:
            return {}
        
        comparison = {
            "model_name": profiles[0].model_name,
            "strategies": []
        }
        
        for profile in profiles:
            comparison["strategies"].append({
                "strategy": profile.load_strategy,
                "success": profile.success,
                "peak_memory_mb": profile.peak_memory_mb,
                "stable_memory_mb": profile.stable_memory_mb,
                "load_time_seconds": profile.load_time_seconds,
                "memory_efficiency": profile.peak_memory_mb / profile.stable_memory_mb if profile.stable_memory_mb > 0 else float('inf')
            })
        
        # Sort by peak memory usage
        comparison["strategies"].sort(key=lambda x: x["peak_memory_mb"] if x["success"] else float('inf'))
        
        return comparison


# Global instance
memory_profiler = MemoryProfiler()


def profile_memory_usage(label: str = ""):
    """Decorator to profile function memory usage."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n=== Memory Profiling: {label or func.__name__} ===")
            
            # Start snapshot
            start_snapshot = memory_profiler.take_snapshot("start")
            
            try:
                result = func(*args, **kwargs)
                
                # End snapshot
                end_snapshot = memory_profiler.take_snapshot("end")
                
                memory_delta = end_snapshot.gpu_allocated_mb - start_snapshot.gpu_allocated_mb
                print(f"Function completed. Memory change: {memory_delta:+.1f} MB")
                
                return result
                
            except Exception as e:
                error_snapshot = memory_profiler.take_snapshot("error")
                memory_delta = error_snapshot.gpu_allocated_mb - start_snapshot.gpu_allocated_mb
                print(f"Function failed. Memory change: {memory_delta:+.1f} MB")
                raise e
        
        return wrapper
    return decorator