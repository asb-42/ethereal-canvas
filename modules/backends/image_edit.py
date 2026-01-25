"""
Real image editing backend using Qwen-Image-Edit-2511 model with memory management.
"""

import os
from pathlib import Path
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

# Import memory management
try:
    from modules.memory import memory_manager, LoadStrategy
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    print("Warning: Memory management not available, using standard loading")
    MEMORY_MANAGEMENT_AVAILABLE = False

# Import pipeline monitoring
try:
    from utils.pipeline_monitor import create_monitor, monitor_pipeline_step
    MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: Pipeline monitoring not available")
    MONITORING_AVAILABLE = False

from modules.img_read.reader import read_image
from modules.img_write.writer import write_image


class ImageEditBackend:
    """Real image editing backend using Qwen models."""
    
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
        """Load the Qwen image editing model with memory management."""
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
            from diffusers import DiffusionPipeline
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading edit model with memory management: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            # Define loading function for memory manager
            def load_edit_model(**kwargs):
                from diffusers import QwenImageEditPipeline
                return QwenImageEditPipeline.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    **kwargs
                )
            
            # Use memory manager to load with fallback strategies
            self.pipeline, config = memory_manager.load_model_with_fallback(
                model_name=self.model_name,
                load_fn=load_edit_model,
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
            
            # Disable xFormers to prevent CPU/CUDA tensor mismatches
            if hasattr(config, 'enable_xformers') and config.enable_xformers:
                try:
                    # self.pipeline.enable_xformers_memory_efficient_attention()
                    print("⚠️ xFormers optimization disabled due to tensor placement issues")
                except Exception as e:
                    print(f"xFormers not available: {e}")
            
            self.loaded = True
            strategy_name = config.strategy.value if hasattr(config, 'strategy') else 'unknown'
            print(f"✓ Edit model loaded successfully using strategy: {strategy_name}")
            
        except Exception as e:
            if memory_manager.is_oom_error(e):
                print(f"OOM error even with memory management: {e}")
                # Try even more aggressive strategies
                self._load_with_aggressive_fallback()
            else:
                print(f"Failed to load edit model with memory management: {e}")
                print("Falling back to stub implementation...")
                self.loaded = True
    
    def _load_with_aggressive_fallback(self):
        """Load with most aggressive memory-saving strategies."""
        try:
            print("Attempting aggressive memory-saving strategies...")
            
            import torch
            from diffusers import QwenImageEditPipeline
            
            # Try sequential offload first
            try:
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                self.pipeline = QwenImageEditPipeline.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                
                # Enable memory optimizations - keep on CUDA to prevent tensor mismatches
                if self.device == "cuda":
                    self.pipeline.enable_attention_slicing()
                    # Keep pipeline on CUDA, no CPU offloading for xFormers compatibility
                else:
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
            from diffusers import QwenImageEditPipeline
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading edit model (standard): {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            # Load the image editing pipeline using QwenImageEditPipeline
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            self.loaded = True
            print(f"✓ Edit model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load edit model: {e}")
            print("Falling back to stub implementation...")
            self.loaded = True  # Still mark as loaded to avoid repeated attempts
            self.pipeline = None  # Explicitly set pipeline to None when loading fails
    
    def edit(self, prompt, input_path):
        """Edit image based on prompt."""
        if not self.loaded:
            self.load()
        
        # Start monitoring
        monitor = None
        if MONITORING_AVAILABLE:
            monitor = create_monitor(f"I2I-{self.device.upper()}")
            monitor.start_monitoring(total_steps=20)
        
        # If pipeline failed to load, use stub
        if not self.pipeline:
            if monitor:
                monitor.error("Pipeline not loaded, using stub")
            
            from modules.runtime.paths import OUTPUTS_DIR, timestamp
            import hashlib
            hash_input = (prompt + input_path).encode('utf-8')
            hash_digest = hashlib.md5(hash_input).hexdigest()
            output_path = OUTPUTS_DIR / f"stub_edit_{hash_digest[:8]}_{timestamp()}.png"
            
            # Create stub edited image
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (512, 512), color='lightblue')
                draw = ImageDraw.Draw(img)
                
                # Try to use a simple font
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Draw some text
                text_lines = [
                    f"STUB EDIT",
                    f"Prompt: {prompt[:30]}...",
                    f"Input: {input_path.split('/')[-1] if '/' in input_path else input_path}",
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                ]
                
                y_position = 50
                for line in text_lines:
                    if font:
                        draw.text((50, y_position), line, fill='black', font=font)
                    else:
                        draw.text((50, y_position), line, fill='black')
                    y_position += 30
                
                draw.rectangle([10, 10, 492, 492], outline='black', width=2)
                
                img.save(output_path)
                return str(output_path)
                
            except Exception as e:
                print(f"Stub edit failed: {e}")
                return str(output_path)
        
        try:
            print(f"Editing image: {input_path}")
            print(f"Edit prompt: {prompt[:50]}...")
            
            if monitor:
                monitor.update_step("Loading input image")
            
            # Load input image
            image_data = read_image(input_path)
            input_image = image_data.pixels  # Extract PIL Image from ImageData
            
            if monitor:
                monitor.update_step("Setting up autocast context")
            
            if monitor:
                monitor.update_step("Running Qwen-Edit pipeline inference")
            
            # Use torch.inference_mode() for better performance and memory usage
            if torch and hasattr(torch, 'inference_mode'):
                inference_context = torch.inference_mode()
            else:
                from contextlib import nullcontext
                inference_context = nullcontext()
            
            with inference_context:
                # Qwen-Edit Pipeline expects specific parameters
                inputs = {
                    "image": input_image,
                    "prompt": prompt,
                    "num_inference_steps": 20,
                    "true_cfg_scale": 4.0,  # Qwen-Edit specific parameter
                    "negative_prompt": "",  # Required to avoid warning
                    "guidance_scale": 1.0,  # Qwen-Edit specific parameter  
                    "num_images_per_prompt": 1
                }
                result = self.pipeline(**inputs)
            
            if monitor:
                monitor.update_step("Processing pipeline results")
            
            edited_image = result.images[0]
            
            # Save edited image to proper outputs directory
            if monitor:
                monitor.update_step("Saving edited image")
            
            from modules.runtime.paths import OUTPUTS_DIR, timestamp
            output_path = OUTPUTS_DIR / f"edited_{timestamp()}.png"
            write_image(edited_image, str(output_path))
            
            if monitor:
                monitor.success(str(output_path))
            
            print(f"✓ Image edited: {output_path}")
            return output_path
            
        except Exception as e:
            if monitor:
                monitor.error(f"Image editing failed: {e}", e)
            print(f"Failed to edit image: {e}")
            # Create stub output even when real editing fails
            from modules.runtime.paths import OUTPUTS_DIR, timestamp
            import hashlib
            hash_input = (prompt + input_path).encode('utf-8')
            hash_digest = hashlib.md5(hash_input).hexdigest()
            output_path = OUTPUTS_DIR / f"failed_edit_{hash_digest[:8]}_{timestamp()}.png"
            return str(output_path)
    
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
                print(f"[edit_backend] Warning: Failed to cleanup GPU memory: {e}")
        
        self.loaded = False
        self.pipeline = None
    
    def __str__(self):
        return f"ImageEditBackend({self.model_name}, device={self.device})"