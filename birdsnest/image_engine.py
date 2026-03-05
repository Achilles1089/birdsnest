########################################################################################################
# Bird's Nest — Persistent Image Engine
# Keeps one mflux model loaded in GPU memory at a time.
# Eliminates subprocess overhead and model reload delay.
########################################################################################################

import threading
import time
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger("birdsnest.image_engine")

# ── Model Registry: maps model IDs to their mflux classes and configs ────────

MODEL_REGISTRY = {
    # ── mflux models ──────────────────────────────────────────────────────
    # Z-Image (6B)
    "z-image-turbo": {
        "engine": "mflux",
        "class_path": "mflux.models.z_image.variants.z_image.ZImage",
        "config": "z_image_turbo",
        "default_steps": 9,
    },
    # FLUX.2 Klein
    "flux2-klein-4b": {
        "engine": "mflux",
        "class_path": "mflux.models.flux2.Flux2Klein",
        "config": "flux2_klein_4b",
        "default_steps": 4,
    },
    "flux2-klein-9b": {
        "engine": "mflux",
        "class_path": "mflux.models.flux2.Flux2Klein",
        "config": "flux2_klein_9b",
        "default_steps": 20,
    },
    # FIBO
    "fibo-lite": {
        "engine": "mflux",
        "class_path": "mflux.models.fibo.variants.txt2img.fibo.FIBO",
        "config": "fibo_lite",
        "default_steps": 8,
    },
    "fibo": {
        "engine": "mflux",
        "class_path": "mflux.models.fibo.variants.txt2img.fibo.FIBO",
        "config": "fibo",
        "default_steps": 30,
    },
    # FLUX.1 (legacy)
    "schnell": {
        "engine": "mflux",
        "class_path": "mflux.models.flux.variants.txt2img.flux.Flux1",
        "config": "schnell",
        "default_steps": 4,
    },
    "dev": {
        "engine": "mflux",
        "class_path": "mflux.models.flux.variants.txt2img.flux.Flux1",
        "config": "dev",
        "default_steps": 20,
    },
    "krea-dev": {
        "engine": "mflux",
        "class_path": "mflux.models.flux.variants.txt2img.flux.Flux1",
        "config": "krea_dev",
        "default_steps": 20,
    },
    "kontext": {
        "engine": "mflux",
        "class_path": "mflux.models.flux.variants.txt2img.flux.Flux1",
        "config": "dev_kontext",
        "default_steps": 20,
    },
    # Qwen Image (20B)
    "qwen-image": {
        "engine": "mflux",
        "class_path": "mflux.models.qwen.variants.txt2img.qwen_image.QwenImage",
        "config": "qwen_image",
        "default_steps": 30,
    },

    # ── diffusers models ──────────────────────────────────────────────────
    "sdxl-turbo": {
        "engine": "diffusers",
        "hf_repo": "stabilityai/sdxl-turbo",
        "default_steps": 1,
        "default_width": 512,
        "default_height": 512,
        "guidance_scale": 0.0,
    },
    "sdxl-lightning": {
        "engine": "diffusers",
        "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_repo": "ByteDance/SDXL-Lightning",
        "lora_weight": "sdxl_lightning_4step_lora.safetensors",
        "default_steps": 4,
        "default_width": 1024,
        "default_height": 1024,
        "guidance_scale": 0.0,
    },
}


def _import_class(class_path: str):
    """Dynamically import a class from a dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ImageEngine:
    """Persistent single-model image generation engine.
    
    Keeps one mflux model loaded in GPU memory at a time.
    Swaps model when user changes selection (unload old, load new).
    Uses Q4 quantization by default for speed.
    """

    def __init__(self, quantize: int = 4):
        self._model = None
        self._model_id: Optional[str] = None
        self._quantize = quantize
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._load_time: float = 0
        self._gen_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def current_model(self) -> Optional[str]:
        return self._model_id

    @property
    def status(self) -> dict:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "model": self._model_id,
            "quantize": self._quantize,
            "load_time": round(self._load_time, 1),
            "gen_count": self._gen_count,
        }

    def load_model(self, model_id: str, force: bool = False) -> dict:
        """Load a model into GPU memory. Unloads any existing model first."""
        if model_id == self._model_id and self._ready and not force:
            return {"status": "already_loaded", "model": model_id}

        if model_id not in MODEL_REGISTRY:
            return {"status": "error", "message": f"Unknown model: {model_id}"}

        with self._lock:
            self._loading = True
            self._ready = False

            # Unload current model
            if self._model is not None:
                logger.info(f"Unloading image model: {self._model_id}")
                del self._model
                self._model = None
                self._model_id = None
                # Force garbage collection to free GPU memory
                import gc
                gc.collect()

            # Load new model
            reg = MODEL_REGISTRY[model_id]
            logger.info(f"Loading image model: {model_id} (Q{self._quantize})")
            t0 = time.time()

            try:
                ModelClass = _import_class(reg["class_path"])
                from mflux.models.common.config import ModelConfig
                config_fn = getattr(ModelConfig, reg["config"])
                model_config = config_fn()

                self._model = ModelClass(
                    model_config=model_config,
                    quantize=self._quantize,
                )
                self._model_id = model_id
                self._load_time = time.time() - t0
                self._ready = True
                self._loading = False

                logger.info(f"Image model loaded: {model_id} in {self._load_time:.1f}s")
                return {
                    "status": "loaded",
                    "model": model_id,
                    "load_time": round(self._load_time, 1),
                }
            except Exception as e:
                self._loading = False
                logger.error(f"Failed to load image model {model_id}: {e}")
                return {"status": "error", "message": str(e)}

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> dict:
        """Generate an image using the loaded model.
        
        Returns dict with status, path, and timing info.
        """
        if not self.is_ready:
            return {"status": "error", "message": "No image model loaded. Load one first."}

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        reg = MODEL_REGISTRY.get(self._model_id, {})
        if steps is None:
            steps = reg.get("default_steps", 9)

        t0 = time.time()

        try:
            with self._lock:
                # Build generation kwargs
                gen_kwargs = dict(
                    seed=seed,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                )
                if guidance is not None:
                    gen_kwargs["guidance"] = guidance

                image = self._model.generate_image(**gen_kwargs)
                image.save(path=output_path)

            elapsed = time.time() - t0
            self._gen_count += 1

            return {
                "status": "ok",
                "path": output_path,
                "elapsed": round(elapsed, 1),
                "seed": seed,
                "steps": steps,
                "model": self._model_id,
            }
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def warm(self) -> dict:
        """Run a tiny warmup generation to prime Metal kernel caches."""
        if not self.is_ready:
            return {"status": "error", "message": "No model loaded"}

        t0 = time.time()
        try:
            with self._lock:
                self._model.generate_image(
                    seed=0,
                    prompt="warmup",
                    width=64,
                    height=64,
                    num_inference_steps=1,
                )
            elapsed = time.time() - t0
            return {"status": "warm", "elapsed": round(elapsed, 1)}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class DiffusersEngine:
    """Image generation engine using HuggingFace diffusers (SDXL Turbo/Lightning).

    Lazy-loads PyTorch + diffusers only when a diffusers model is selected.
    Uses MPS (Apple Silicon GPU) when available, CPU otherwise.
    """

    def __init__(self):
        self._pipeline = None
        self._model_id: Optional[str] = None
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._load_time: float = 0
        self._gen_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self._ready and self._pipeline is not None

    @property
    def current_model(self) -> Optional[str]:
        return self._model_id

    @property
    def status(self) -> dict:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "model": self._model_id,
            "engine": "diffusers",
            "load_time": round(self._load_time, 1),
            "gen_count": self._gen_count,
        }

    def unload(self):
        """Fully unload pipeline and free GPU memory."""
        if self._pipeline is not None:
            logger.info(f"Unloading diffusers model: {self._model_id}")
            del self._pipeline
            self._pipeline = None
            self._model_id = None
            self._ready = False
            import gc
            gc.collect()

    def warm(self) -> dict:
        """Run a tiny inference to prime MPS kernel caches. Model must be loaded."""
        if not self.is_ready:
            return {"status": "error", "message": "No diffusers model loaded"}
        try:
            import torch
            t0 = time.time()
            logger.info(f"Warming diffusers model: {self._model_id}")
            with self._lock:
                # Tiny 64×64 inference to compile Metal kernels
                reg = MODEL_REGISTRY.get(self._model_id, {})
                steps = reg.get("default_steps", 1)
                guidance = reg.get("guidance_scale", 0.0)
                gen = torch.Generator(device=self._pipeline.device).manual_seed(0)
                self._pipeline(
                    prompt="warmup",
                    width=64,
                    height=64,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                )
            elapsed = time.time() - t0
            logger.info(f"Diffusers warm complete: {elapsed:.1f}s")
            return {"status": "warm", "elapsed": round(elapsed, 1)}
        except Exception as e:
            logger.error(f"Diffusers warm failed: {e}")
            return {"status": "error", "message": str(e)}

    def load_model(self, model_id: str, force: bool = False) -> dict:
        """Load an SDXL model into GPU memory."""
        if model_id == self._model_id and self._ready and not force:
            return {"status": "already_loaded", "model": model_id}

        if model_id not in MODEL_REGISTRY:
            return {"status": "error", "message": f"Unknown model: {model_id}"}

        reg = MODEL_REGISTRY[model_id]
        if reg.get("engine") != "diffusers":
            return {"status": "error", "message": f"{model_id} is not a diffusers model"}

        with self._lock:
            self._loading = True
            self._ready = False
            self.unload()

            logger.info(f"Loading diffusers model: {model_id}")
            t0 = time.time()

            try:
                import torch
                from diffusers import StableDiffusionXLPipeline

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                dtype = torch.float16 if device == "mps" else torch.float32

                # Load base pipeline (direct import avoids HunyuanDiT/MT5 dep chain)
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    reg["hf_repo"],
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                )

                # Apply Lightning LoRA if specified
                lora_repo = reg.get("lora_repo")
                if lora_repo:
                    pipe.load_lora_weights(
                        lora_repo,
                        weight_name=reg.get("lora_weight"),
                    )
                    pipe.fuse_lora()

                pipe = pipe.to(device)

                self._pipeline = pipe
                self._model_id = model_id
                self._load_time = time.time() - t0
                self._ready = True
                self._loading = False

                logger.info(f"Diffusers model loaded: {model_id} on {device} in {self._load_time:.1f}s")
                return {
                    "status": "loaded",
                    "model": model_id,
                    "device": device,
                    "load_time": round(self._load_time, 1),
                }
            except Exception as e:
                self._loading = False
                logger.error(f"Failed to load diffusers model {model_id}: {e}")
                return {"status": "error", "message": str(e)}

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> dict:
        """Generate an image using the loaded diffusers pipeline."""
        if not self.is_ready:
            return {"status": "error", "message": "No diffusers model loaded."}

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        reg = MODEL_REGISTRY.get(self._model_id, {})
        if steps is None:
            steps = reg.get("default_steps", 4)
        if guidance is None:
            guidance = reg.get("guidance_scale", 0.0)

        t0 = time.time()

        try:
            import torch
            with self._lock:
                generator = torch.Generator(device="cpu").manual_seed(seed)

                result = self._pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )

                image = result.images[0]
                image.save(output_path)

            elapsed = time.time() - t0
            self._gen_count += 1

            return {
                "status": "ok",
                "path": output_path,
                "elapsed": round(elapsed, 1),
                "seed": seed,
                "steps": steps,
                "model": self._model_id,
            }
        except Exception as e:
            logger.error(f"Diffusers generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def warm(self) -> dict:
        """Run a tiny warmup generation to prime GPU caches."""
        if not self.is_ready:
            return {"status": "error", "message": "No model loaded"}

        t0 = time.time()
        try:
            import torch
            with self._lock:
                generator = torch.Generator(device="cpu").manual_seed(0)
                self._pipeline(
                    prompt="warmup",
                    width=64,
                    height=64,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=generator,
                )
            elapsed = time.time() - t0
            return {"status": "warm", "elapsed": round(elapsed, 1)}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ── Global engine instances (lazy, only one active at a time) ───────────────

_mflux_engine: Optional[ImageEngine] = None
_diffusers_engine: Optional[DiffusersEngine] = None


def get_engine(model_id: Optional[str] = None) -> 'ImageEngine | DiffusersEngine':
    """Get the correct engine for the given model.

    - mflux models → ImageEngine
    - diffusers models → DiffusersEngine
    Fully unloads the other engine when switching types.
    """
    global _mflux_engine, _diffusers_engine

    # Determine engine type
    reg = MODEL_REGISTRY.get(model_id or "", {})
    engine_type = reg.get("engine", "mflux")

    if engine_type == "diffusers":
        # Unload mflux engine if active
        if _mflux_engine is not None and _mflux_engine.is_ready:
            logger.info("Switching to diffusers engine — unloading mflux")
            if _mflux_engine._model is not None:
                del _mflux_engine._model
                _mflux_engine._model = None
                _mflux_engine._ready = False
                _mflux_engine._model_id = None
                import gc
                gc.collect()

        if _diffusers_engine is None:
            _diffusers_engine = DiffusersEngine()
        return _diffusers_engine
    else:
        # Unload diffusers engine if active
        if _diffusers_engine is not None and _diffusers_engine.is_ready:
            logger.info("Switching to mflux engine — unloading diffusers")
            _diffusers_engine.unload()

        if _mflux_engine is None:
            _mflux_engine = ImageEngine(quantize=8)
        return _mflux_engine


def get_diffusers_engine() -> DiffusersEngine:
    """Get or create the DiffusersEngine singleton (without unloading mflux)."""
    global _diffusers_engine
    if _diffusers_engine is None:
        _diffusers_engine = DiffusersEngine()
    return _diffusers_engine
