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
from modules.runtime.paths import output_inpaint_path
from modules.runtime import model_access


# Pipeline classes, in the order they are worth trying.

#

# Only MASK_CAPABLE_CLASS exposes an inpainting mask on __call__; the plain Qwen

# edit pipelines accept prompt-embedding masks only, so calling them with a

# mask raises on the keyword and the mask never reaches the model.

# CHECKPOINT_CLASS is what this checkpoint declares for itself in its

# model_index.json. GENERIC_CLASS is the base class: it can load, but can never

# consume a mask, so it stays a last resort only.

MASK_CAPABLE_CLASS = "QwenImageEditInpaintPipeline"

CHECKPOINT_CLASS = "QwenImageEditPlusPipeline"

GENERIC_CLASS = "DiffusionPipeline"



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
    
    def _declared_pipeline_class(self):

        """The pipeline class this checkpoint declares for itself, if found."""

        try:

            import json

            for card in sorted(Path(self.cache_dir).rglob("model_index.json")):

                try:

                    declared = json.loads(card.read_text()).get("_class_name")

                except Exception:

                    continue

                if declared:

                    return declared

        except Exception:

            pass

        return None

    

    def _candidate_classes(self):

        """Pipeline class names to try, most capable first, de-duplicated."""

        order = [MASK_CAPABLE_CLASS, self._declared_pipeline_class(), CHECKPOINT_CLASS, GENERIC_CLASS]

        seen, out = set(), []

        for name in order:

            if name and name not in seen:

                seen.add(name)

                out.append(name)

        return out

    

    def _load_pretrained(self, **kwargs):

        """Load the checkpoint with the first pipeline class that accepts it."""
        # Weights belong to the upstream and are licensed by them: refuse before any fetch.
        model_access.require(self.model_name)

        import diffusers

        

        attempts = []

        for name in self._candidate_classes():

            cls = getattr(diffusers, name, None)

            if cls is None:

                attempts.append(f"{name}: not present in diffusers {diffusers.__version__}")

                continue

            try:

                pipeline = cls.from_pretrained(

                    self.model_name,

                    cache_dir=str(self.cache_dir),

                    **kwargs

                )

            except Exception as e:

                attempts.append(f"{name}: {e}")

                print(f"[inpaint] {name} could not load this checkpoint: {e}")

                continue

            

            if name == GENERIC_CLASS:

                print(f"[inpaint] WARNING: loaded through {GENERIC_CLASS}, which cannot consume "

                      "an inpainting mask; inpaint() will refuse rather than ignore the mask")

            else:

                print(f"[inpaint] loaded {name} for {self.model_name}")

            return pipeline

        

        raise RuntimeError(

            f"No pipeline could load {self.model_name}. Attempts: " + " | ".join(attempts)

        )

    

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
        
        # The memory-managed path is opt-in via EC_ENABLE_MEMORY_MANAGEMENT=1. It was
        # pinned off wholesale while debugging a load hang, which also left
        # _load_with_memory_management() and _load_with_aggressive_fallback()
        # unreachable; the flag keeps today's behaviour the default while leaving that
        # path reachable and testable.
        if MEMORY_MANAGEMENT_AVAILABLE and os.environ.get("EC_ENABLE_MEMORY_MANAGEMENT") == "1":
            return self._load_with_memory_management()
        return self._load_standard()
    
    def _load_with_memory_management(self):
        """Load model using memory management system."""
        try:
            import torch
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading inpaint model with memory management: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            def load_inpaint_model(**kwargs):
                return self._load_pretrained(**kwargs)
            
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
            
            try:
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                self.pipeline = self._load_pretrained(
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
        except ImportError as e:
            print(f"Failed to import required dependencies: {e}")
            print("Using stub implementation...")
            self.loaded = True
            return
        
        print(f"Loading inpaint model (standard): {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        try:
            self.pipeline = self._load_pretrained(
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
    
    def _accepts_mask(self) -> bool:
        """Report whether the loaded pipeline can actually consume a mask.
        
        Only the mask-capable Qwen edit pipelines expose ``mask_image``; the
        plain edit pipelines accept prompt-embedding masks only, so a call
        through them would raise on the keyword and silently drop the mask.
        """
        try:
            import inspect
            params = inspect.signature(type(self.pipeline).__call__).parameters
        except (TypeError, ValueError):
            return True  # Not introspectable: do not block the call.
        return 'mask_image' in params or 'mask' in params
    
    def inpaint(self, image, mask, prompt):
        """Inpaint the masked region of ``image``.
        
        Failures raise. The previous version returned a fabricated
        ``inpainted_<hash>.png`` name whenever the pipeline was missing or the
        call failed, so callers saw success against a file that was never
        written and the UI rendered a broken image with no error anywhere.
        """
        if not self.loaded:
            self.load()
        
        if self.pipeline is None:
            raise RuntimeError(
                "Inpaint backend has no pipeline loaded, so inpainting cannot run. "
                f"Check that '{self.model_name}' is available under {self.cache_dir}."
            )
        
        if not self._accepts_mask():
            raise NotImplementedError(
                f"{type(self.pipeline).__name__} accepts no inpainting mask argument, so the "
                "mask would be ignored. Load this checkpoint with a mask-capable pipeline "
                "(QwenImageEditInpaintPipeline) before calling inpaint()."
            )
        
        print(f"Inpainting with mask and prompt: {prompt[:50]}...")
        
        input_image = image if hasattr(image, 'save') else read_image(image) if isinstance(image, str) else image
        input_mask = mask if hasattr(mask, 'save') else read_image(mask) if isinstance(mask, str) else mask
        
        import torch
        call_kwargs = {
            "image": input_image,
            "mask_image": input_mask,
            "prompt": prompt,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        }
        
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.inference_mode():
                    result = self.pipeline(**call_kwargs)
            else:
                result = self.pipeline(**call_kwargs)
            
            inpainted_image = result.images[0]
            
            # Write through the runtime paths helper: the old fixed name was
            # relative to the current working directory, not to runtime/outputs.
            output_path = output_inpaint_path("inpainted")
            write_image(inpainted_image, str(output_path))
            
            print(f"\u2713 Image inpainted: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Failed to inpaint image: {e}")
            raise
    
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