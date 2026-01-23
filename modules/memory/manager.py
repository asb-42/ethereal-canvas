"""
Memory management strategies and OOM prevention for Qwen-Image-Edit-2511.
Implements automatic retry with different configurations and backend switching.
"""

import gc
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import DiffusionPipeline, BitsAndBytesConfig, TorchAoConfig
    from diffusers.utils import logging as diffusers_logging
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .profiler import memory_profiler, profile_memory_usage


class LoadStrategy(Enum):
    """Available model loading strategies."""
    FP16_FULL = "fp16_full"
    FP8_OPTIMIZED = "fp8_optimized"
    NF4_QUANTIZED = "nf4_quantized"
    CPU_OFFLOAD = "cpu_offload"
    SEQUENTIAL_OFFLOAD = "sequential_offload"
    TINY_GRADIENT = "tiny_gradient"


@dataclass
class LoadingConfig:
    """Configuration for model loading strategy."""
    strategy: LoadStrategy
    torch_dtype: Optional[str] = None
    quantization_config: Optional[Any] = None
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = True
    enable_sequential_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_xformers: bool = False
    custom_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_kwargs is None:
            self.custom_kwargs = {}


class MemoryManager:
    """
    Central memory management component.
    Handles OOM detection, retry strategies, and backend switching.
    """
    
    def __init__(self):
        self.device = self._detect_device()
        self.loading_configs = self._create_loading_configs()
        self.fallback_chain = [
            LoadStrategy.FP16_FULL,
            LoadStrategy.FP8_OPTIMIZED,
            LoadStrategy.NF4_QUANTIZED,
            LoadStrategy.CPU_OFFLOAD,
            LoadStrategy.SEQUENTIAL_OFFLOAD,
        ]
        self.current_config = None
        
    def _detect_device(self) -> str:
        """Detect available compute device."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _create_loading_configs(self) -> Dict[LoadStrategy, LoadingConfig]:
        """Create predefined loading configurations."""
        configs = {}
        
        # FP16 Full Precision
        configs[LoadStrategy.FP16_FULL] = LoadingConfig(
            strategy=LoadStrategy.FP16_FULL,
            torch_dtype="float16" if self.device == "cuda" else "float32",
            low_cpu_mem_usage=True,
            enable_attention_slicing=False,
            enable_xformers=True
        )
        
        # FP8 Optimized
        if TORCH_AVAILABLE and self.device == "cuda":
            try:
                configs[LoadStrategy.FP8_OPTIMIZED] = LoadingConfig(
                    strategy=LoadStrategy.FP8_OPTIMIZED,
                    torch_dtype="float8_e5m2",
                    low_cpu_mem_usage=True,
                    enable_attention_slicing=True,
                    enable_xformers=True,
                    custom_kwargs={"variant": "fp8"}
                )
            except Exception as e:
                print(f"Warning: FP8 not supported on this hardware: {e}")
        else:
            # FP8 fallback to FP16 for CPU or when not available
            configs[LoadStrategy.FP8_OPTIMIZED] = LoadingConfig(
                strategy=LoadStrategy.FP8_OPTIMIZED,
                torch_dtype="float32",
                low_cpu_mem_usage=True,
                enable_attention_slicing=True,
                enable_xformers=False
            )
        
        # NF4 Quantized
        if DIFFUSERS_AVAILABLE and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                nf4_config = DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                configs[LoadStrategy.NF4_QUANTIZED] = LoadingConfig(
                    strategy=LoadStrategy.NF4_QUANTIZED,
                    torch_dtype="bfloat16" if self.device == "cuda" else "float32",
                    quantization_config=nf4_config,
                    low_cpu_mem_usage=True,
                    enable_attention_slicing=True,
                    enable_xformers=False
                )
            except Exception as e:
                print(f"Warning: Could not create NF4 quantization config: {e}")
        
        # CPU Offload
        configs[LoadStrategy.CPU_OFFLOAD] = LoadingConfig(
            strategy=LoadStrategy.CPU_OFFLOAD,
            torch_dtype="float16" if self.device == "cuda" else "float32",
            low_cpu_mem_usage=True,
            enable_model_cpu_offload=True,
            enable_attention_slicing=True,
            enable_xformers=False
        )
        
        # Sequential CPU Offload
        configs[LoadStrategy.SEQUENTIAL_OFFLOAD] = LoadingConfig(
            strategy=LoadStrategy.SEQUENTIAL_OFFLOAD,
            torch_dtype="float16" if self.device == "cuda" else "float32",
            low_cpu_mem_usage=True,
            enable_sequential_cpu_offload=True,
            enable_attention_slicing=True,
            enable_xformers=False
        )
        
        return configs
    
    def get_available_memory_mb(self) -> float:
        """Get available GPU memory in MB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            return (total - allocated) / (1024 * 1024)
        except:
            return 0.0
    
    def estimate_required_memory(self, model_name: str) -> Dict[str, float]:
        """Estimate memory requirements for different strategies."""
        # Base estimates for Qwen-Image-Edit-2511 (approximately)
        base_requirements = {
            "transformer": 8000,  # MB for main transformer
            "text_encoder": 2000,  # MB for text encoder
            "vae": 1000,  # MB for VAE if present
            "activations": 4000,  # MB for intermediate activations
            "overhead": 1000,  # MB for framework overhead
        }
        
        total = sum(base_requirements.values())
        
        strategy_multipliers = {
            LoadStrategy.FP16_FULL: 1.0,
            LoadStrategy.FP8_OPTIMIZED: 0.6,
            LoadStrategy.NF4_QUANTIZED: 0.35,
            LoadStrategy.CPU_OFFLOAD: 0.2,
            LoadStrategy.SEQUENTIAL_OFFLOAD: 0.15,
        }
        
        estimates = {}
        for strategy, multiplier in strategy_multipliers.items():
            if strategy in self.loading_configs:
                estimates[strategy.value] = total * multiplier
        
        return estimates
    
    def is_oom_error(self, error: Exception) -> bool:
        """Check if error is an OOM (Out of Memory) error."""
        error_str = str(error).lower()
        oom_indicators = [
            "out of memory",
            "cuda out of memory",
            "memory allocation failed",
            "cuda error: out of memory",
            "ran out of memory",
            "defaultcpuallocator: not enough memory"
        ]
        return any(indicator in error_str for indicator in oom_indicators)
    
    def cleanup_memory(self, aggressive: bool = False):
        """Clean up memory and reset caches."""
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
    
    def load_model_with_fallback(self, model_name: str, load_fn: Callable, 
                                preferred_strategies: Optional[List[LoadStrategy]] = None,
                                max_retries: int = 3) -> Tuple[Any, LoadingConfig]:
        """
        Load model with automatic fallback on OOM errors.
        
        Args:
            model_name: Name of the model to load
            load_fn: Function that loads the model
            preferred_strategies: Preferred strategies to try first
            max_retries: Maximum retries per strategy
        
        Returns:
            Tuple of (loaded_model, successful_config)
        """
        if preferred_strategies is None:
            strategies_to_try = self.fallback_chain
        else:
            strategies_to_try = preferred_strategies + [s for s in self.fallback_chain if s not in preferred_strategies]
        
        last_error = None
        
        for strategy in strategies_to_try:
            if strategy not in self.loading_configs:
                continue
            
            config = self.loading_configs[strategy]
            print(f"\n=== Attempting load with strategy: {strategy.value} ===")
            
            for attempt in range(max_retries):
                try:
                    # Cleanup before each attempt
                    self.cleanup_memory(aggressive=(attempt > 0))
                    
                    # Prepare loading arguments
                    load_kwargs = self._prepare_load_kwargs(config, model_name)
                    
                    # Profile the loading attempt
                    @profile_memory_usage(f"{strategy.value}_attempt_{attempt + 1}")
                    def load_with_profiling():
                        return load_fn(**load_kwargs)
                    
                    model = load_with_profiling()
                    
                    print(f"✓ Successfully loaded with {strategy.value} on attempt {attempt + 1}")
                    self.current_config = config
                    return model, config
                    
                except Exception as e:
                    last_error = e
                    
                    if self.is_oom_error(e):
                        print(f"✗ OOM with {strategy.value} on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            print(f"  Retrying with more aggressive cleanup...")
                            time.sleep(1)  # Brief pause
                        else:
                            print(f"  Giving up on {strategy.value}, trying next strategy...")
                    else:
                        print(f"✗ Non-OOM error with {strategy.value}: {e}")
                        # For non-OOM errors, don't retry other strategies
                        raise e
        
        # All strategies failed
        memory_info = self.get_available_memory_mb()
        raise RuntimeError(
            f"Failed to load {model_name} with all strategies. "
            f"Last error: {last_error}. "
            f"Available GPU memory: {memory_info:.1f} MB"
        )
    
    def _prepare_load_kwargs(self, config: LoadingConfig, model_name: str) -> Dict[str, Any]:
        """Prepare loading arguments based on configuration."""
        kwargs = {}
        
        # Basic arguments
        if config.torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            
            if TORCH_AVAILABLE and hasattr(torch, "float8_e5m2"):
                dtype_map["float8_e5m2"] = torch.float8_e5m2
            
            if config.torch_dtype in dtype_map:
                kwargs["torch_dtype"] = dtype_map[config.torch_dtype]
        
        if config.quantization_config:
            kwargs["quantization_config"] = config.quantization_config
        
        if config.device_map:
            kwargs["device_map"] = config.device_map
        
        if config.low_cpu_mem_usage:
            kwargs["low_cpu_mem_usage"] = True
        
        # Include custom kwargs
        kwargs.update(config.custom_kwargs)
        
        return kwargs
    
    def apply_optimizations(self, pipeline, config: LoadingConfig):
        """Apply post-loading optimizations to pipeline."""
        if not pipeline or not config:
            return
        
        # CPU offload optimizations
        if config.enable_model_cpu_offload and hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
            print("✓ Enabled model CPU offload")
        
        if config.enable_sequential_cpu_offload and hasattr(pipeline, 'enable_sequential_cpu_offload'):
            pipeline.enable_sequential_cpu_offload()
            print("✓ Enabled sequential CPU offload")
        
        # Attention slicing
        if config.enable_attention_slicing and hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
            print("✓ Enabled attention slicing")
        
        # Memory-efficient attention
        if config.enable_xformers and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✓ Enabled xFormers memory efficient attention")
            except Exception as e:
                print(f"  xFormers not available: {e}")
    
    def get_recommended_strategy(self, available_memory_mb: float) -> List[LoadStrategy]:
        """Get recommended strategies based on available memory."""
        estimates = self.estimate_required_memory("Qwen-Image-Edit-2511")
        recommendations = []
        
        for strategy in self.fallback_chain:
            if strategy.value in estimates:
                required = estimates[strategy.value]
                if required <= available_memory_mb * 0.9:  # Leave 10% buffer
                    recommendations.append(strategy)
        
        # Add the lowest memory option as fallback
        if not recommendations:
            recommendations = [LoadStrategy.SEQUENTIAL_OFFLOAD]
        
        return recommendations
    
    def monitor_inference_memory(self, inference_fn: Callable, *args, **kwargs):
        """Monitor memory usage during inference and handle OOM."""
        try:
            # Pre-inference snapshot
            pre_snapshot = memory_profiler.take_snapshot("pre_inference")
            
            result = inference_fn(*args, **kwargs)
            
            # Post-inference snapshot
            post_snapshot = memory_profiler.take_snapshot("post_inference")
            
            memory_delta = post_snapshot.gpu_allocated_mb - pre_snapshot.gpu_allocated_mb
            if memory_delta > 0:
                print(f"Inference memory increase: {memory_delta:.1f} MB")
            
            return result
            
        except Exception as e:
            if self.is_oom_error(e):
                print(f"✗ Inference OOM detected: {e}")
                self.cleanup_memory(aggressive=True)
                
                # Try with more aggressive settings
                if self.current_config and not self.current_config.enable_attention_slicing:
                    print("Retrying with attention slicing...")
                    if hasattr(self.current_config, 'enable_attention_slicing'):
                        self.current_config.enable_attention_slicing = True
                    
                    return inference_fn(*args, **kwargs)
            
            raise e


# Global memory manager instance
memory_manager = MemoryManager()


def retry_on_oom(max_attempts: int = 3, cleanup_delay: float = 1.0):
    """Decorator to retry function on OOM errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if memory_manager.is_oom_error(e) and attempt < max_attempts - 1:
                        print(f"OOM detected, cleaning up and retrying (attempt {attempt + 2}/{max_attempts})")
                        memory_manager.cleanup_memory(aggressive=True)
                        time.sleep(cleanup_delay)
                    else:
                        raise e
            return None
        return wrapper
    return decorator