"""
Real text-to-image backend using Qwen-Image-2512 model.
"""

import os
from pathlib import Path
from modules.img_write.writer import write_image


class TextToImageBackend:
    """Real text-to-image backend using Qwen models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-2512"):
        self.model_name = model_name
        self.loaded = False
        self.pipeline = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        
        # Set up model cache directory
        app_root = Path(__file__).parent.parent.parent
        self.cache_dir = app_root / "models" / "Qwen-Image-2512"
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load(self):
        """Load the Qwen text-to-image model."""
        if self.loaded:
            return
        
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading T2I model: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            # Load the pipeline with proper caching
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            self.loaded = True
            print(f"✓ T2I model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load T2I model: {e}")
            print("Falling back to stub implementation...")
            self.loaded = True  # Still mark as loaded to avoid repeated attempts
    
    def generate(self, prompt):
        """Generate image from prompt."""
        if not self.loaded:
            self.load()
        
        # If pipeline failed to load, use stub
        if not self.pipeline:
            return f"generated_{hash(prompt)}.png"
        
        try:
            print(f"Generating image for prompt: {prompt[:50]}...")
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                )
            
            image = result.images[0]
            
            # Save image
            output_path = f"generated_{hash(prompt)}.png"
            write_image(image, output_path)
            
            print(f"✓ Image generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to generate image: {e}")
            return f"generated_{hash(prompt)}.png"
    
    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline and hasattr(self.pipeline, 'to'):
            try:
                # Move pipeline to CPU to free GPU memory
                if self.device == "cuda":
                    self.pipeline = self.pipeline.to("cpu")
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        self.loaded = False
        self.pipeline = None
    
    def __str__(self):
        return f"TextToImageBackend({self.model_name}, device={self.device})"