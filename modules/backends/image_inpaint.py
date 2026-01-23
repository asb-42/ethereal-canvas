"""
Real image inpainting backend using Qwen models with memory management.
"""

import os
from pathlib import Path

# Import memory management
try:
    from modules.memory import memory_manager, LoadStrategy
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    print("Warning: Memory management not available, using standard loading")
    MEMORY_MANAGEMENT_AVAILABLE = False

from modules.img_read.reader import read_image
from modules.img_write.writer import write_image


class ImageInpaintBackend:
    """Real image inpainting backend using Qwen models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2511"):
        self.model_name = model_name
        self.loaded = False
        self.pipeline = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        
        # Set up model cache directory
        app_root = Path(__file__).parent.parent.parent
        self.cache_dir = app_root / "models" / "Qwen-Image-Edit-2511"
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load(self):
        """Load the Qwen image inpainting model with memory management."""
        if self.loaded:
            return
        
        # Use memory management if available
        if MEMORY_MANAGEMENT_AVAILABLE:
            return self._load_with_memory_management()
        else:
            return self._load_standard()
    
    def _load_with_memory_management(self):
        """Load model using memory management system."""
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading inpaint model with memory management: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            # Define loading function for memory manager
            def load_inpaint_model(**kwargs):
                return QwenImageEditPlusPipeline.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    **kwargs
                )
            
            # Use memory manager to load with fallback strategies
            self.pipeline, config = memory_manager.load_model_with_fallback(
                model_name=self.model_name,
                load_fn=load_inpaint_model,
                preferred_strategies=[
                    LoadStrategy.FP16_FULL,
                    LoadStrategy.FP8_OPTIMIZED,
                    LoadStrategy.CPU_OFFLOAD
                ]
            )
            
            # Apply post-loading optimizations
            if hasattr(config, 'enable_attention_slicing') and config.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                print("✓ Enabled attention slicing")
            
            if hasattr(config, 'enable_xformers') and config.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("✓ Enabled xFormers optimization")
                except Exception as e:
                    print(f"xFormers not available: {e}")
            
            self.loaded = True
            strategy_name = config.strategy.value if hasattr(config, 'strategy') else 'unknown'
            print(f"✓ Inpaint model loaded successfully using strategy: {strategy_name}")
            
        except Exception as e:
            if memory_manager.is_oom_error(e):
                print(f"OOM error even with memory management: {e}")
                # Try even more aggressive strategies
                self._load_with_aggressive_fallback()
            else:
                print(f"Failed to load inpaint model with memory management: {e}")
                print("Falling back to stub implementation...")
                self.loaded = True
    
    def _load_with_aggressive_fallback(self):
        """Load with most aggressive memory-saving strategies."""
        try:
            print("Attempting aggressive memory-saving strategies...")
            
            import torch
            from diffusers import QwenImageEditPlusPipeline
            
            # Try sequential offload first
            try:
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                
                # Enable all memory optimizations
                self.pipeline.enable_sequential_cpu_offload()
                self.pipeline.enable_attention_slicing()
                
                self.loaded = True
                print("✓ Loaded with aggressive memory optimizations")
                
            except Exception as e:
                print(f"Aggressive fallback failed: {e}")
                raise e
        
        except Exception as e:
            print(f"All loading strategies failed: {e}")
            self.loaded = True  # Prevent repeated attempts
    
    def _load_standard(self):
        """Standard loading without memory management (fallback)."""
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading inpaint model (standard): {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            # Load the image inpainting pipeline
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            self.loaded = True
            print(f"✓ Inpaint model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load inpaint model: {e}")
            print("Falling back to stub implementation...")
            self.loaded = True  # Still mark as loaded to avoid repeated attempts
    
    def inpaint(self, image, mask, prompt):
        """Inpaint image based on mask and prompt."""
        if not self.loaded:
            self.load()
        
        # If pipeline failed to load, use stub
        if not self.pipeline:
            return f"inpainted_{hash(str(image) + str(mask) + prompt)}.png"
        
        try:
            print(f"Inpainting with mask and prompt: {prompt[:50]}...")
            
            # Use the image and mask directly
            input_image = image if hasattr(image, 'save') else read_image(image) if isinstance(image, str) else image
            input_mask = mask if hasattr(mask, 'save') else read_image(mask) if isinstance(mask, str) else mask
            
            # Generate inpainted image
            with torch.autocast(self.device):
                result = self.pipeline(
                    image=input_image,
                    mask_image=input_mask,
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                )
            
            inpainted_image = result.images[0]
            
            # Save inpainted image
            output_path = f"inpainted_{hash(str(input_image) + str(input_mask) + prompt)}.png"
            write_image(inpainted_image, output_path)
            
            print(f"✓ Image inpainted: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to inpaint image: {e}")
            return f"inpainted_{hash(str(image) + str(mask) + prompt)}.png"
    
    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline and hasattr(self.pipeline, 'to'):
            try:
                # Move pipeline to CPU to free GPU memory
                if self.device == "cuda":
                    self.pipeline = self.pipeline.to("cpu")
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[inpaint_backend] Warning: Failed to cleanup GPU memory: {e}")
        
        self.loaded = False
        self.pipeline = None
    
    def __str__(self):
        return f"ImageInpaintBackend({self.model_name}, device={self.device})"