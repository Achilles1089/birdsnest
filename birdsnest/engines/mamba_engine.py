########################################################################################################
# Bird's Nest — Mamba Engine
# HuggingFace Transformers-based inference for Mamba SSM models
########################################################################################################

import os, time, threading
from typing import Generator, Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, MambaForCausalLM, TextIteratorStreamer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from birdsnest.engine import InferenceEngine


class MambaEngine(InferenceEngine):
    """Inference engine for Mamba State Space Models via HF Transformers."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    @property
    def engine_name(self) -> str:
        return "Mamba SSM"

    def load(self, model_path: str, **kwargs) -> Dict[str, Any]:
        """Load a Mamba model from HuggingFace format."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed. Run: pip install transformers")

        t0 = time.time()
        device = self.detect_device()

        # model_path could be a local dir or HF repo ID
        repo_or_path = model_path

        # If it's a local directory, use it; otherwise treat as HF repo
        if not os.path.isdir(model_path):
            # Try to find it in the models catalog
            from birdsnest.models import MODEL_CATALOG
            for entry in MODEL_CATALOG:
                if entry.get("architecture") == "mamba" and entry["id"] in model_path:
                    repo_or_path = entry["hf_repo"]
                    break

        self.tokenizer = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)
        self.model = MambaForCausalLM.from_pretrained(
            repo_or_path,
            torch_dtype=torch.float32,  # MPS needs float32
            trust_remote_code=True,
        ).to(device)

        self.model.eval()
        self._device = device
        self.model_name = os.path.basename(model_path)
        self.is_loaded = True
        load_time = round(time.time() - t0, 1)

        # Count params
        n_params = sum(p.numel() for p in self.model.parameters())
        size_str = f"{n_params / 1e9:.1f}B" if n_params > 1e9 else f"{n_params / 1e6:.0f}M"

        return {
            "model": self.model_name,
            "device": device,
            "version": "mamba",
            "size": size_str,
            "load_time": load_time,
        }

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.is_loaded = False
        self.model_name = None

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream tokens using HF generate with TextIteratorStreamer."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "do_sample": temperature > 0.01,
            "streamer": streamer,
        }

        # Run generation in a thread so we can stream
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield text

        thread.join()

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, tokens) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "loaded": self.is_loaded,
            "model": self.model_name,
            "device": getattr(self, '_device', None),
        }
