"""
Enhanced image editing backend with comprehensive memory management.
Integrates the memory management layer with Qwen-Image-Edit-2511.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from modules.memory import memory_manager, memory_profiler, LoadStrategy, profile_memory_usage, retry_on_oom
from modules.img_read.reader import read_image
from modules.img_write.writer import write_image


class EnhancedImageEditBackend:
    """
    Enhanced image editing backend with comprehensive memory management.
    Automatically selects optimal loading strategies and handles OOM errors.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2511", 
                 cache_dir: Optional[str] = None,
                 preferred_strategies: Optional[list] = None):
        self.model_name = model_name
        self.loaded = False
        self.pipeline = None
        self.config = None
        
        # Set up model cache directory
        app_root = Path(__file__).parent.parent.parent
        self.cache_dir = cache_dir or str(app_root / "models" / "Qwen-Image-Edit-2511")
        
        # User preferences for loading strategies
        self.preferred_strategies = preferred_strategies
        
        # Memory management
        self.device = memory_manager.device
        self.loading_history = []
        
        print(f"Enhanced Image Edit Backend initialized")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Cache: {self.cache_dir}")
    
    def _load_model_with_diffusers(self, **kwargs):
        """Load model using diffusers pipeline."""
        try:
            from diffusers import QwenImageEditPipeline
        except ImportError:
            raise ImportError("diffusers library is required. Install with: pip install diffusers")
        
        return QwenImageEditPipeline.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **kwargs
        )
    
    def _load_model_with_components(self, **kwargs):
        """Load model using individual components for better memory control."""
        try:
            from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise ImportError("diffusers and transformers libraries are required")
        
        # Load transformer separately for quantization control
        if 'quantization_config' in kwargs:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                **kwargs
            )
            transformer = transformer.to("cpu")
            
            # Load text encoder separately if needed
            text_encoder = None
            if 'quantization_config' in kwargs:
                from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
                if hasattr(kwargs['quantization_config'], 'load_in_4bit') and kwargs['quantization_config'].load_in_4bit:
                    tf_config = TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=kwargs.get('torch_dtype', torch.float16),
                    )
                    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        subfolder="text_encoder",
                        quantization_config=tf_config,
                        torch_dtype=kwargs.get('torch_dtype', torch.float16),
                    )
                    text_encoder = text_encoder.to("cpu")
            
            # Create pipeline with loaded components
            pipeline_kwargs = {'transformer': transformer}
            if text_encoder is not None:
                pipeline_kwargs['text_encoder'] = text_encoder
            
            pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_name,
                **pipeline_kwargs
            )
        else:
            # Fallback to simple loading
            pipeline = self._load_model_with_diffusers(**kwargs)
        
        return pipeline
    
    def load(self) -> bool:
        """
        Load the model with automatic fallback strategies.
        Returns True if successful, False otherwise.
        """
        if self.loaded:
            return True
        
        print(f"\n=== Loading {self.model_name} ===")
        
        # Check available memory and get recommendations
        available_memory = memory_manager.get_available_memory_mb()
        print(f"Available GPU memory: {available_memory:.1f} MB")
        
        recommendations = memory_manager.get_recommended_strategy(available_memory)
        print(f"Recommended strategies: {[s.value for s in recommendations]}")
        
        try:
            # Try loading with fallback strategies
            self.pipeline, self.config = memory_manager.load_model_with_fallback(
                model_name=self.model_name,
                load_fn=self._load_model_with_components,
                preferred_strategies=self.preferred_strategies or recommendations
            )
            
            # Apply post-loading optimizations
            memory_manager.apply_optimizations(self.pipeline, self.config)
            
            # Move to device if not already handled by offloading
            if self.device == "cuda" and not (self.config.enable_model_cpu_offload or self.config.enable_sequential_cpu_offload):
                self.pipeline = self.pipeline.to("cuda")
            
            self.loaded = True
            self.loading_history.append({
                "strategy": self.config.strategy.value,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"✓ Model loaded successfully with strategy: {self.config.strategy.value}")
            memory_profiler.print_memory_summary()
            
            return True
            
        except Exception as e:
            self.loading_history.append({
                "strategy": "failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"✗ Failed to load model: {e}")
            return False
    
    @retry_on_oom(max_attempts=3)
    @profile_memory_usage("image_editing")
    def edit(self, prompt: str, input_path: str, **kwargs) -> Optional[str]:
        """
        Edit image with memory-managed inference.
        
        Args:
            prompt: Editing prompt
            input_path: Path to input image
            **kwargs: Additional inference parameters
        
        Returns:
            Path to edited image or None if failed
        """
        if not self.loaded:
            if not self.load():
                return None
        
        # If pipeline failed to load, use stub
        if not self.pipeline:
            return self._create_stub_edit(prompt, input_path)
        
        try:
            print(f"Editing image: {input_path}")
            print(f"Edit prompt: {prompt[:50]}...")
            
            # Load input image
            input_image = read_image(input_path)
            
            # Default inference parameters
            inference_params = {
                "image": input_image,
                "prompt": prompt,
                "num_inference_steps": kwargs.get("num_inference_steps", 20),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "num_images_per_prompt": 1,
            }
            
            # Add optional parameters
            optional_params = ["generator", "true_cfg_scale", "negative_prompt"]
            for param in optional_params:
                if param in kwargs:
                    inference_params[param] = kwargs[param]
            
            # Monitor and run inference
            def run_inference():
                return self.pipeline(**inference_params)
            
            result = memory_manager.monitor_inference_memory(run_inference)
            edited_image = result.images[0]
            
            # Save edited image
            from modules.runtime.paths import OUTPUTS_DIR, timestamp
            output_path = OUTPUTS_DIR / f"edited_{timestamp()}.png"
            write_image(edited_image, str(output_path))
            
            print(f"✓ Image edited: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Failed to edit image: {e}")
            return self._create_stub_edit(prompt, input_path, error=str(e))
    
    def _create_stub_edit(self, prompt: str, input_path: str, error: Optional[str] = None) -> str:
        """Create a stub edited image when real editing fails."""
        from modules.runtime.paths import OUTPUTS_DIR, timestamp
        import hashlib
        
        hash_input = (prompt + input_path + (error or "")).encode('utf-8')
        hash_digest = hashlib.md5(hash_input).hexdigest()
        output_path = OUTPUTS_DIR / f"stub_edit_{hash_digest[:8]}_{timestamp()}.png"
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (512, 512), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Try to use a simple font
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Draw information text
            text_lines = [
                f"ENHANCED STUB EDIT",
                f"Strategy: {self.config.strategy.value if self.config else 'None'}",
                f"Prompt: {prompt[:30]}...",
                f"Input: {input_path.split('/')[-1] if '/' in input_path else input_path}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]
            
            if error:
                text_lines.append(f"Error: {error[:40]}...")
            
            y_position = 30
            for line in text_lines:
                if font:
                    draw.text((30, y_position), line, fill='black', font=font)
                else:
                    draw.text((30, y_position), line, fill='black')
                y_position += 25
            
            draw.rectangle([10, 10, 502, 502], outline='black', width=2)
            
            img.save(output_path)
            return str(output_path)
            
        except Exception as e:
            print(f"Stub edit failed: {e}")
            return str(output_path)
    
    def cleanup(self):
        """Clean up resources with memory management."""
        if self.pipeline:
            try:
                # Apply cleanup strategies based on current config
                if self.config and self.config.enable_model_cpu_offload:
                    # Model is already on CPU, just clear cache
                    pass
                elif self.device == "cuda":
                    # Move to CPU first, then clear
                    self.pipeline = self.pipeline.to("cpu")
                
                memory_manager.cleanup_memory(aggressive=True)
                
            except Exception as e:
                print(f"[EnhancedImageEditBackend] Warning: Failed to cleanup: {e}")
        
        self.loaded = False
        self.pipeline = None
        self.config = None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        info = {
            "loaded": self.loaded,
            "device": self.device,
            "strategy": self.config.strategy.value if self.config else None,
            "loading_history": self.loading_history
        }
        
        if self.loaded:
            vram_info = memory_manager.get_vram_info()
            info.update(vram_info)
            
            current_snapshot = memory_profiler.take_snapshot("status")
            info["gpu_allocated_mb"] = current_snapshot.gpu_allocated_mb
            info["gpu_reserved_mb"] = current_snapshot.gpu_reserved_mb
        
        return info
    
    def switch_strategy(self, new_strategy: LoadStrategy) -> bool:
        """
        Switch to a different loading strategy.
        Requires unloading and reloading the model.
        """
        print(f"Switching from {self.config.strategy.value if self.config else 'None'} to {new_strategy.value}")
        
        # Cleanup current model
        self.cleanup()
        
        # Set new preferred strategy and reload
        self.preferred_strategies = [new_strategy]
        return self.load()
    
    def benchmark_strategies(self, test_image_path: str, test_prompt: str = "test edit") -> Dict[str, Any]:
        """
        Benchmark different loading strategies with a test case.
        Returns performance and memory usage comparison.
        """
        print(f"\n=== Benchmarking Loading Strategies ===")
        
        results = {}
        
        for strategy in memory_manager.fallback_chain:
            if strategy not in memory_manager.loading_configs:
                continue
            
            print(f"\nTesting {strategy.value}...")
            
            # Cleanup before each test
            self.cleanup()
            memory_manager.cleanup_memory(aggressive=True)
            
            # Load with this strategy
            start_time = time.time()
            load_success = self.switch_strategy(strategy)
            load_time = time.time() - start_time
            
            if not load_success:
                results[strategy.value] = {
                    "load_success": False,
                    "load_time": load_time,
                    "inference_success": False
                }
                continue
            
            # Test inference
            start_time = time.time()
            try:
                result_path = self.edit(test_prompt, test_image_path, num_inference_steps=4)
                inference_time = time.time() - start_time
                inference_success = result_path is not None
            except Exception as e:
                inference_time = time.time() - start_time
                inference_success = False
                print(f"Inference failed: {e}")
            
            # Get memory info
            memory_info = self.get_memory_info()
            
            results[strategy.value] = {
                "load_success": True,
                "load_time": load_time,
                "inference_success": inference_success,
                "inference_time": inference_time,
                "peak_memory_mb": memory_info.get("gpu_allocated_mb", 0),
                "vram_total_mb": memory_info.get("total_mb", 0),
                "memory_efficiency": memory_info.get("gpu_allocated_mb", 0) / memory_info.get("total_mb", 1)
            }
        
        return results
    
    def __str__(self):
        return f"EnhancedImageEditBackend({self.model_name}, device={self.device}, loaded={self.loaded})"


# Factory function for easy instantiation
def create_enhanced_edit_backend(prefer_low_memory: bool = False) -> EnhancedImageEditBackend:
    """
    Factory function to create an enhanced backend with sensible defaults.
    
    Args:
        prefer_low_memory: If True, prioritize memory-efficient strategies
    """
    if prefer_low_memory:
        preferred = [LoadStrategy.NF4_QUANTIZED, LoadStrategy.CPU_OFFLOAD, LoadStrategy.SEQUENTIAL_OFFLOAD]
    else:
        preferred = None  # Use automatic recommendations
    
    return EnhancedImageEditBackend(preferred_strategies=preferred)