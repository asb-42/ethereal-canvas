"""
Real text-to-image backend using Qwen-Image-2512 model.
Runtime-compliant with proper path management.
"""

import os
import sys
import time
from pathlib import Path
import torch
from datetime import datetime

# Import runtime utilities
try:
    from modules.runtime.paths import (
        RUNTIME_ROOT, LOGS_DIR, OUTPUTS_DIR, CACHE_DIR, TMP_DIR,
        QWEN_T2I_CACHE, timestamp
    )
except ImportError:
    print("Warning: Runtime paths not available, using fallback")
    RUNTIME_ROOT = Path(".")
    LOGS_DIR = RUNTIME_ROOT / "logs"
    OUTPUTS_DIR = RUNTIME_ROOT / "outputs"
    CACHE_DIR = RUNTIME_ROOT / ".." / "models"
    QWEN_T2I_CACHE = CACHE_DIR / "Qwen-Image-2512"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Fallback logger class for when runtime utils are not available
class SimpleLogger:
    def __init__(self, name: str):
        self.name = name
    
    def info(self, message: str, **kwargs):
        print(f"[{self.name}] INFO: {message}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def warning(self, message: str, **kwargs):
        print(f"[{self.name}] WARNING: {message}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def error(self, message: str, **kwargs):
        print(f"[{self.name}] ERROR: {message}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def success(self, operation: str, **kwargs):
        print(f"[{self.name}] SUCCESS: {operation}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def info(self, message: str, **kwargs):
        print(f"[{self.name}] INFO: {message}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def warning(self, message: str, **kwargs):
        print(f"[{self.name}] WARNING: {message}")
        for key, value in kwargs.items():
            print(f"[{self.name}] {key}: {value}")
    
    def error(self, message: str, **kwargs):
        print(f"[{self.name}] ERROR: {message}")
        for key, value in kwargs.items():
            print(f"[self.name}] {key}: {value}")
    
    def success(self, operation: str, **kwargs):
        print(f"[{self.name}] SUCCESS: {operation}")
        for key, value in kwargs.items():
            print(f"[self.name}] {key}: {value}")

class TextToImageBackend(GenerationBackend):
    """Real text-to-image backend using Qwen models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-2512"):
        self.model_name = model_name
        self.loaded = False
        self.pipeline = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        
        # Initialize logger
        try:
            self.logger = RuntimeLogger("t2i_backend")
        except ImportError:
            self.logger = None
        
        print(f"🎨 TextToImageBackend initialized: {model_name}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load(self) -> None:
        """Load Qwen text-to-image model."""
        if self.loaded:
            self.logger.info("Model already loaded")
            return
        
        start_time = time.time()
        
        try:
            # Load dependencies
            import torch
            from diffusers import DiffusionPipeline
            
            self.logger.info(f"Loading T2I model: {self.model_name}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Cache directory: {QWEN_T2I_CACHE}")
            
            # Load pipeline with proper caching
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                cache_dir=str(QWEN_T2I_CACHE),
                torch_dtype=torch.float16,
                safety_checker=None
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            self.loaded = True
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ T2I model loaded successfully in {load_time:.2f}s")
            
        except ImportError as e:
            self.logger.error(f"Failed to import required dependencies: {e}")
            self.logger.info("Falling back to stub implementation...")
            self.loaded = True  # Still mark as loaded to avoid repeated attempts
        
        except Exception as e:
            self.logger.error(f"Failed to load T2I model: {e}")
            self.logger.info("Falling back to stub implementation...")
            self.loaded = True
    
    def generate(self, prompt: str) -> str:
        """Generate image from text prompt."""
        if not self.loaded:
            self.logger.warning("Model not loaded, using stub implementation")
            return self._generate_stub(prompt)
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating image for prompt: {prompt[:50]}...")
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                )
            
            image = result.images[0]
            
            # Save image to runtime/outputs/
            output_path = OUTPUTS_DIR / f"t2i_{timestamp()}.png"
            image.save(output_path)
            
            generation_time = time.time() - start_time
            
            self.logger.success("image_generation", {
                "prompt": prompt[:50],
                "output_path": str(output_path),
                "generation_time": f"{generation_time:.2f}s"
            })
            
            self.logger.info(f"✓ Image generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate image: {e}")
            return self._generate_stub(prompt)
        
        finally:
            pass
    
    def _generate_stub(self, prompt: str) -> str:
        """Fallback stub implementation when models are not available."""
        try:
            # Simple deterministic stub
            import hashlib
            hash_input = (prompt + str(time.time())).encode('utf-8')
            hash_digest = hashlib.md5(hash_input).hexdigest()
            output_path = OUTPUTS_DIR / f"stub_t2i_{hash_digest[:8]}.png"
            
            # Create a simple colored text image using PIL
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a simple font
            try:
                font = ImageFont.load_default()
                font_size = 30
            except:
                font = ImageFont.load_default()
                font_size = 20
            
            # Draw some text
            text_lines = [
                f"STUB GENERATION",
                f"Prompt: {prompt[:30]}...",
                f"Model: {self.model_name}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]
            
            y_position = 50
            for line in text_lines:
                text_width = draw.textlength(line) if line else 0
                draw.text((50 - text_width) // 2, y_position, text, fill='black')
                y_position += 30
            
            draw.rectangle([10, 10, 492, 492], outline='black', width=2)
            
            img.save(output_path)
            
            if self.logger:
                self.logger.info(f"Generated stub image: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stub generation failed: {e}")
            return f"error_stub_{hash(prompt)}.png"
    
    def get_model_info(self) -> dict:
        """Get model information."""
        cache_stats = get_cache_usage() if 'get_cache_usage' in globals() else {}
        
        model_info = {
            "model_name": self.model_name,
            "loaded": self.loaded,
            "device": self.device,
            "pipeline_type": "DiffusionPipeline" if self.pipeline else "stub",
            "cache_stats": cache_stats.get("Qwen-Image-2512", {"size_bytes": 0, "file_count": 0}),
            "runtime_paths": {
                "output_dir": str(OUTPUTS_DIR),
                "cache_dir": str(CACHE_DIR),
                "logs_dir": str(LOGS_DIR),
                "tmp_dir": str(TMP_DIR)
            }
        }
        
        return model_info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.pipeline and hasattr(self.pipeline, 'cpu'):
            try:
                # Move pipeline to CPU to free GPU memory
                self.pipeline = self.pipeline.to("cpu")
                torch.cuda.empty_cache()
            except:
                pass
        
        if self.logger:
            self.logger.info("T2I backend cleanup completed")
        
        self.loaded = False
        self.pipeline = None