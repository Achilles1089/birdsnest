########################################################################################################
# Bird's Nest — Model Manager
# Handles local model scanning, HuggingFace downloads, deletion, and metadata
########################################################################################################

import os, json, shutil, time
from typing import Dict, List, Optional, Any
from pathlib import Path


# ── Architecture Categories ─────────────────────────────────────────────────

ARCH_CATEGORIES = {
    "rwkv": {
        "label": "RWKV",
        "type": "Linear RNN",
        "desc": "Recurrent neural network with linear attention — constant memory, infinite context, O(1) per token",
        "color": "#7c6aef",
    },
    "mamba": {
        "label": "Mamba",
        "type": "State Space Model",
        "desc": "Selective state space model — hardware-aware, linear complexity, faster than Transformers",
        "color": "#34d399",
    },
    "xlstm": {
        "label": "xLSTM",
        "type": "Extended RNN",
        "desc": "Modernized LSTM with exponential gating and matrix memory — by Sepp Hochreiter (LSTM inventor)",
        "color": "#fbbf24",
    },
    "hyena": {
        "label": "StripedHyena",
        "type": "Conv + SSM Hybrid",
        "desc": "Gated convolutions interleaved with attention — subquadratic, long-context specialist",
        "color": "#f87171",
    },
}


# ── Curated Model Catalog ───────────────────────────────────────────────────

MODEL_CATALOG = [
    # ── RWKV-7 G1 "GooseOne" Thinking Models ──────────────
    {
        "id": "rwkv7-g1e-7.2b",
        "name": "RWKV-7 GooseOne 7.2B (G1e)",
        "display_name": "GooseOne 7.2B ⭐",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1e-7.2b-20260301-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "7.2B",
        "size_gb": 14.4,
        "context": 8192,
        "description": "★ Latest (Mar 1) — best G1 thinking model",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-13.3b",
        "name": "RWKV-7 GooseOne 13.3B",
        "display_name": "GooseOne 13.3B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-13.3b-20260131-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "13.3B",
        "size_gb": 26.5,
        "context": 8192,
        "description": "Largest thinking model — needs 32GB+ RAM",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-7.2b",
        "name": "RWKV-7 GooseOne 7.2B (G1d)",
        "display_name": "GooseOne 7.2B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-7.2b-20260131-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "7.2B",
        "size_gb": 14.4,
        "context": 8192,
        "description": "Large thinking — strong reasoning",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-2.9b",
        "name": "RWKV-7 GooseOne 2.9B",
        "display_name": "GooseOne 2.9B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-2.9b-20260131-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "2.9B",
        "size_gb": 5.9,
        "context": 8192,
        "description": "★ Recommended — best balance for Mac",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-1.5b",
        "name": "RWKV-7 GooseOne 1.5B",
        "display_name": "GooseOne 1.5B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-1.5b-20260212-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "1.5B",
        "size_gb": 3.0,
        "context": 8192,
        "description": "Medium thinking — good speed & quality",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-0.4b",
        "name": "RWKV-7 GooseOne 0.4B",
        "display_name": "GooseOne 0.4B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "0.4B",
        "size_gb": 0.9,
        "context": 8192,
        "description": "Small thinking — very fast, decent quality",
        "thinking": True,
    },
    {
        "id": "rwkv7-g1-0.1b",
        "name": "RWKV-7 GooseOne 0.1B",
        "display_name": "GooseOne 0.1B",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7-g1d-0.1b-20260129-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7-g1",
        "params": "0.1B",
        "size_gb": 0.3,
        "context": 8192,
        "description": "Tiny thinking — experiments, low RAM",
        "thinking": True,
    },
    {
        "id": "rwkv7a-g1-0.1b",
        "name": "RWKV-7a GooseOne 0.1B (DeepEmbed)",
        "display_name": "GooseOne 0.1B DeepEmbed",
        "hf_repo": "BlinkDL/rwkv7-g1",
        "hf_file": "rwkv7a-g1d-0.1b-20260212-ctx8192.pth",
        "architecture": "rwkv",
        "version": "v7a-g1",
        "params": "0.1B",
        "size_gb": 2.0,
        "context": 8192,
        "description": "DeepEmbed variant — enhanced embeddings",
        "thinking": True,
    },
    # ── RWKV-7 Goose World (Chat, Non-Thinking) ────────────
    {
        "id": "rwkv7-world-0.1b",
        "name": "RWKV-7 Goose World 0.1B",
        "display_name": "Goose World 0.1B",
        "hf_repo": "BlinkDL/rwkv-7-world",
        "hf_file": "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v7",
        "params": "0.1B",
        "size_gb": 0.3,
        "context": 4096,
        "description": "Tiny — experiments, minimal RAM",
    },
    {
        "id": "rwkv7-world-0.4b",
        "name": "RWKV-7 Goose World 0.4B",
        "display_name": "Goose World 0.4B",
        "hf_repo": "BlinkDL/rwkv-7-world",
        "hf_file": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v7",
        "params": "0.4B",
        "size_gb": 0.9,
        "context": 4096,
        "description": "Small — fast, decent multilingual",
    },
    {
        "id": "rwkv7-world-1.5b",
        "name": "RWKV-7 Goose World 1.5B",
        "display_name": "Goose World 1.5B",
        "hf_repo": "BlinkDL/rwkv-7-world",
        "hf_file": "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v7",
        "params": "1.5B",
        "size_gb": 3.0,
        "context": 4096,
        "description": "Medium — good multilingual balance",
    },
    {
        "id": "rwkv7-world-2.9b",
        "name": "RWKV-7 Goose World 2.9B",
        "display_name": "Goose World 2.9B",
        "hf_repo": "BlinkDL/rwkv-7-world",
        "hf_file": "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v7",
        "params": "2.9B",
        "size_gb": 5.5,
        "context": 4096,
        "description": "★ Recommended — 100+ languages",
    },
    # ── RWKV-6 Finch (Legacy, Non-Thinking) ───────────────
    {
        "id": "rwkv6-world-1.6b",
        "name": "RWKV-6 Finch 1.6B",
        "display_name": "Finch 1.6B",
        "hf_repo": "BlinkDL/rwkv-6-world",
        "hf_file": "RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v6",
        "params": "1.6B",
        "size_gb": 3.0,
        "context": 4096,
        "description": "Fastest — 22+ tok/s on MPS",
    },
    {
        "id": "rwkv6-world-3b",
        "name": "RWKV-6 Finch 3B",
        "display_name": "Finch 3B",
        "hf_repo": "BlinkDL/rwkv-6-world",
        "hf_file": "RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v6",
        "params": "3B",
        "size_gb": 6.2,
        "context": 4096,
        "description": "Mid-range — solid general use",
    },
    {
        "id": "rwkv6-world-7b",
        "name": "RWKV-6 Finch 7B",
        "display_name": "Finch 7B",
        "hf_repo": "BlinkDL/rwkv-6-world",
        "hf_file": "RWKV-x060-World-7B-v3-20241112-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v6",
        "params": "7B",
        "size_gb": 15.3,
        "context": 4096,
        "description": "Large — strong quality",
    },
    {
        "id": "rwkv6-world-14b",
        "name": "RWKV-6 Finch 14B",
        "display_name": "Finch 14B",
        "hf_repo": "BlinkDL/rwkv-6-world",
        "hf_file": "RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth",
        "architecture": "rwkv",
        "version": "v6",
        "params": "14B",
        "size_gb": 28.1,
        "context": 4096,
        "description": "Largest Finch — needs 32GB+ RAM",
    },
    # ── Mamba (State Space Model) ──────────────────────────
    {
        "id": "mamba-2.8b",
        "name": "Mamba 2.8B",
        "display_name": "Mamba 2.8B",
        "hf_repo": "state-spaces/mamba-2.8b-hf",
        "hf_file": None,
        "architecture": "mamba",
        "version": "v1",
        "params": "2.8B",
        "size_gb": 5.6,
        "context": 8192,
        "description": "Largest Mamba — strong performance",
    },
    {
        "id": "mamba-1.4b",
        "name": "Mamba 1.4B",
        "display_name": "Mamba 1.4B",
        "hf_repo": "state-spaces/mamba-1.4b-hf",
        "hf_file": None,
        "architecture": "mamba",
        "version": "v1",
        "params": "1.4B",
        "size_gb": 2.8,
        "context": 8192,
        "description": "Fast & efficient SSM",
    },
    # ── xLSTM (Extended RNN) ──────────────────────────────
    {
        "id": "xlstm-7b",
        "name": "xLSTM 7B",
        "display_name": "xLSTM 7B",
        "hf_repo": "NX-AI/xLSTM-7b",
        "hf_file": None,
        "architecture": "xlstm",
        "version": "v1",
        "params": "7B",
        "size_gb": 14.0,
        "context": 8192,
        "description": "Modernized LSTM with matrix memory",
    },
    # ── StripedHyena (Conv + SSM Hybrid) ──────────────────
    {
        "id": "stripedhyena-nous-7b",
        "name": "StripedHyena-Nous 7B",
        "display_name": "StripedHyena Chat 7B",
        "hf_repo": "togethercomputer/StripedHyena-Nous-7B",
        "hf_file": None,
        "architecture": "hyena",
        "version": "v1",
        "params": "7B",
        "size_gb": 14.0,
        "context": 32768,
        "description": "Chat-tuned — 32K context window",
    },
]


# ── Image Model Catalog ─────────────────────────────────────────────────────

IMAGE_MODEL_CATALOG = [
    # ═══════════════════════════════════════════════════════════════════════════
    #  Z-Image — Tongyi Lab (Alibaba), 6B params, Nov 2025
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "z-image-turbo",
        "name": "Z-Image Turbo",
        "display_name": "Z-Image Turbo ★",
        "cli_command": "mflux-generate-z-image-turbo",
        "hf_repo": "Tongyi-MAI/Z-Image-Turbo",
        "default_steps": 9,
        "size_gb": 12.0,
        "quantized_size_gb": 4.0,
        "params": "6B",
        "capabilities": ["generate", "img2img", "lora"],
        "description": "Fastest high-quality — 9 steps, distilled 6B, bilingual text rendering",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  FLUX.2 — Black Forest Labs, 4B & 9B params, Jan 2026
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "flux2-klein-4b",
        "name": "FLUX.2 Klein 4B",
        "display_name": "FLUX.2 Klein 4B ★",
        "cli_command": "mflux-generate-flux2",
        "model_version": "4b",
        "hf_repo": "black-forest-labs/FLUX.2-klein-4B",
        "default_steps": 4,
        "size_gb": 8.0,
        "quantized_size_gb": 3.0,
        "params": "4B",
        "capabilities": ["generate", "edit", "img2img", "lora"],
        "description": "Smallest & fastest FLUX.2 — 4 steps, image editing support",
    },
    {
        "id": "flux2-klein-9b",
        "name": "FLUX.2 Klein 9B",
        "display_name": "FLUX.2 Klein 9B",
        "cli_command": "mflux-generate-flux2",
        "model_version": "9b",
        "hf_repo": "black-forest-labs/FLUX.2-klein-9B",
        "default_steps": 8,
        "size_gb": 18.0,
        "quantized_size_gb": 7.0,
        "params": "9B",
        "capabilities": ["generate", "edit", "img2img", "lora"],
        "description": "Larger FLUX.2 — better quality, image editing, 9B params",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  FIBO — Bria.ai, 8B params, Oct 2025
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "fibo-lite",
        "name": "FIBO Lite",
        "display_name": "FIBO Lite",
        "cli_command": "mflux-generate-fibo",
        "model_version": "lite",
        "hf_repo": "briaai/FIBO-lite",
        "default_steps": 8,
        "size_gb": 10.0,
        "quantized_size_gb": 4.0,
        "params": "8B",
        "capabilities": ["generate"],
        "description": "Distilled FIBO — 8 steps, JSON-native prompting, ~10x faster",
    },
    {
        "id": "fibo",
        "name": "FIBO",
        "display_name": "FIBO (Base)",
        "cli_command": "mflux-generate-fibo",
        "model_version": "fibo",
        "hf_repo": "briaai/FIBO",
        "default_steps": 30,
        "size_gb": 16.0,
        "quantized_size_gb": 6.0,
        "params": "8B",
        "capabilities": ["generate", "refine", "inspire"],
        "description": "Full FIBO — JSON-native precision, VLM refine/inspire modes, 30 steps",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  SeedVR2 — ByteDance, 3B params, Jun 2025
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "seedvr2",
        "name": "SeedVR2 Upscaler",
        "display_name": "SeedVR2 Upscaler",
        "cli_command": "mflux-upscale-seedvr2",
        "hf_repo": "numz/SeedVR2_3b",
        "default_steps": 1,
        "size_gb": 6.0,
        "quantized_size_gb": 2.5,
        "params": "3B",
        "capabilities": ["upscale"],
        "description": "1-step upscale — no prompt, fast, high-fidelity super-resolution",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  Qwen Image — Alibaba, 20B params, Aug 2025
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "qwen-image",
        "name": "Qwen Image",
        "display_name": "Qwen Image 20B",
        "cli_command": "mflux-generate-qwen",
        "hf_repo": "Qwen/Qwen-Image",
        "default_steps": 30,
        "size_gb": 40.0,
        "quantized_size_gb": 15.0,
        "params": "20B",
        "capabilities": ["generate", "edit"],
        "description": "Largest model — 20B, best prompt understanding & world knowledge, slower",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  FLUX.1 — Black Forest Labs, 12B params, Aug 2024 (Legacy)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "schnell",
        "name": "FLUX.1 Schnell",
        "display_name": "FLUX.1 Schnell",
        "cli_command": "mflux-generate",
        "cli_args": ["--model", "schnell"],
        "hf_repo": "black-forest-labs/FLUX.1-schnell",
        "default_steps": 4,
        "size_gb": 24.0,
        "quantized_size_gb": 8.0,
        "params": "12B",
        "capabilities": ["generate", "lora"],
        "description": "Fast FLUX.1 — 4 steps, distilled, good general quality",
        "legacy": True,
    },
    {
        "id": "dev",
        "name": "FLUX.1 Dev",
        "display_name": "FLUX.1 Dev",
        "cli_command": "mflux-generate",
        "cli_args": ["--model", "dev"],
        "hf_repo": "black-forest-labs/FLUX.1-dev",
        "default_steps": 20,
        "size_gb": 24.0,
        "quantized_size_gb": 8.0,
        "params": "12B",
        "capabilities": ["generate", "lora", "controlnet", "depth"],
        "description": "Best FLUX.1 — 20 steps, full quality, controlnet/depth support",
        "legacy": True,
    },
    {
        "id": "krea-dev",
        "name": "FLUX.1 Krea Dev",
        "display_name": "FLUX.1 Krea Dev",
        "cli_command": "mflux-generate",
        "cli_args": ["--model", "krea-dev"],
        "hf_repo": "black-forest-labs/FLUX.1-Krea-dev",
        "default_steps": 20,
        "size_gb": 24.0,
        "quantized_size_gb": 8.0,
        "params": "12B",
        "capabilities": ["generate", "lora"],
        "description": "Enhanced photorealism — Krea AI fine-tuned FLUX.1 Dev",
        "legacy": True,
    },
    {
        "id": "kontext",
        "name": "FLUX.1 Kontext",
        "display_name": "FLUX.1 Kontext",
        "cli_command": "mflux-generate-kontext",
        "hf_repo": "black-forest-labs/FLUX.1-Kontext-dev",
        "default_steps": 20,
        "size_gb": 24.0,
        "quantized_size_gb": 8.0,
        "params": "12B",
        "capabilities": ["generate", "edit", "style-transfer"],
        "description": "Image editing via text — style transfer, character consistency, in-context gen",
        "legacy": True,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    #  SDXL — Stability AI / ByteDance, 3.5B params, Distilled for Speed
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id": "sdxl-turbo",
        "name": "SDXL Turbo",
        "display_name": "SDXL Turbo ⚡",
        "engine": "diffusers",
        "hf_repo": "stabilityai/sdxl-turbo",
        "default_steps": 1,
        "size_gb": 6.5,
        "quantized_size_gb": 6.5,
        "params": "3.5B",
        "capabilities": ["generate"],
        "description": "Fastest — 1 step, 512² only, adversarial distillation, real-time generation",
    },
    {
        "id": "sdxl-lightning",
        "name": "SDXL Lightning",
        "display_name": "SDXL Lightning ⚡⚡",
        "engine": "diffusers",
        "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 4,
        "size_gb": 6.5,
        "quantized_size_gb": 6.5,
        "params": "3.5B + LoRA",
        "capabilities": ["generate"],
        "description": "Ultra-fast 1024² — 4 steps, ByteDance Lightning LoRA, best speed/quality ratio",
    },
]


class ModelManager:
    """Manages local models folder and HuggingFace downloads."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def list_local(self) -> List[Dict[str, Any]]:
        """List all models downloaded to disk."""
        MODEL_EXTENSIONS = ('.pth', '.safetensors', '.gguf')
        models = []
        for f in sorted(os.listdir(self.models_dir)):
            path = os.path.join(self.models_dir, f)

            # Skip hidden/cache dirs
            if f.startswith('.'):
                continue

            # ── File-based models ──
            if any(f.endswith(ext) for ext in MODEL_EXTENSIONS):
                size_gb = os.path.getsize(path) / 1024**3
                # Strip extension for name
                name = f
                for ext in MODEL_EXTENSIONS:
                    name = name.replace(ext, '')

                catalog_entry = self._match_catalog(f)
                arch = catalog_entry.get("architecture", "unknown") if catalog_entry else self._guess_arch(f)
                version = catalog_entry.get("version", "?") if catalog_entry else "?"
                display_name = catalog_entry.get("display_name", name) if catalog_entry else self._clean_name(name)

                models.append({
                    "name": name,
                    "display_name": display_name,
                    "filename": f,
                    "path": path,
                    "size_gb": round(size_gb, 1),
                    "architecture": arch,
                    "version": version,
                    "catalog_id": catalog_entry["id"] if catalog_entry else None,
                    "description": catalog_entry.get("description", "") if catalog_entry else "",
                })

            # ── Directory-based models (Mamba, etc.) ──
            elif os.path.isdir(path):
                config_path = os.path.join(path, "config.json")
                if os.path.exists(config_path):
                    # Calculate total dir size
                    total_size = sum(
                        os.path.getsize(os.path.join(dp, fn))
                        for dp, _, fns in os.walk(path) for fn in fns
                    )
                    size_gb = total_size / 1024**3

                    catalog_entry = self._match_catalog(f)
                    arch = catalog_entry.get("architecture", "unknown") if catalog_entry else self._guess_arch(f)
                    version = catalog_entry.get("version", "?") if catalog_entry else "?"
                    display_name = catalog_entry.get("display_name", f) if catalog_entry else self._clean_name(f)

                    models.append({
                        "name": f,
                        "display_name": display_name,
                        "filename": f,
                        "path": path,
                        "size_gb": round(size_gb, 1),
                        "architecture": arch,
                        "version": version,
                        "catalog_id": catalog_entry["id"] if catalog_entry else None,
                        "description": catalog_entry.get("description", "") if catalog_entry else "",
                    })

        return models

    def list_available(self) -> List[Dict[str, Any]]:
        """List all models from catalog, marking which are downloaded."""
        local = {m["filename"] for m in self.list_local()}
        result = []
        for entry in MODEL_CATALOG:
            filename = entry.get("hf_file") or f"{entry['id']}.safetensors"
            result.append({
                **entry,
                "downloaded": filename in local or any(
                    entry["id"].replace("-", "").replace(".", "") in f.replace("-", "").replace(".", "")
                    for f in local
                ),
            })
        return result

    def get_model_path(self, name_or_id: str) -> Optional[str]:
        """Find a model by name or catalog ID."""
        for m in self.list_local():
            if m["name"] == name_or_id or m.get("catalog_id") == name_or_id:
                return m["path"]
            # Fuzzy match
            if name_or_id.lower() in m["name"].lower():
                return m["path"]
        return None

    def delete(self, name_or_filename: str) -> Dict[str, Any]:
        """Delete a model from disk (handles both files and directories)."""
        for m in self.list_local():
            if m["name"] == name_or_filename or m["filename"] == name_or_filename:
                path = m["path"]
                size = m["size_gb"]
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                return {"deleted": m["name"], "freed_gb": size}
        return {"error": f"Model not found: {name_or_filename}"}

    def download(self, catalog_id: str, progress_callback=None) -> Dict[str, Any]:
        """Download a model from HuggingFace."""
        entry = next((e for e in MODEL_CATALOG if e["id"] == catalog_id), None)
        if not entry:
            return {"error": f"Unknown model: {catalog_id}"}

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return {"error": "huggingface_hub not installed. Run: pip install huggingface-hub"}

        hf_repo = entry["hf_repo"]
        hf_file = entry.get("hf_file")

        if hf_file:
            # Direct file download (RWKV .pth files)
            dest = os.path.join(self.models_dir, hf_file)
            if os.path.exists(dest):
                return {"status": "already_exists", "path": dest}

            t0 = time.time()
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
            )
            elapsed = time.time() - t0
            return {
                "status": "downloaded",
                "path": downloaded,
                "time": round(elapsed, 1),
                "size_gb": entry["size_gb"],
            }
        else:
            # HF Transformers model — download entire repo
            from huggingface_hub import snapshot_download
            dest = os.path.join(self.models_dir, catalog_id)
            if os.path.exists(dest):
                return {"status": "already_exists", "path": dest}

            t0 = time.time()
            downloaded = snapshot_download(
                repo_id=hf_repo,
                local_dir=dest,
                local_dir_use_symlinks=False,
            )
            elapsed = time.time() - t0
            return {
                "status": "downloaded",
                "path": downloaded,
                "time": round(elapsed, 1),
                "size_gb": entry["size_gb"],
            }

    def disk_usage(self) -> Dict[str, Any]:
        """Get total disk usage of models directory."""
        total = 0
        count = 0
        supported_ext = ('.pth', '.safetensors')
        for f in os.listdir(self.models_dir):
            path = os.path.join(self.models_dir, f)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                total += size
                if f.endswith(supported_ext):
                    count += 1
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for ff in files:
                        total += os.path.getsize(os.path.join(root, ff))
                count += 1
        return {"total_gb": round(total / 1024**3, 1), "model_count": count}

    def _match_catalog(self, filename: str) -> Optional[Dict]:
        """Try to match a local file to a catalog entry."""
        for entry in MODEL_CATALOG:
            if entry.get("hf_file") == filename:
                return entry
        return None

    def _guess_arch(self, filename: str) -> str:
        """Guess architecture from filename."""
        fn = filename.lower()
        if 'rwkv' in fn or 'x060' in fn or 'x070' in fn:
            return "rwkv"
        if 'mamba' in fn:
            return "mamba"
        if 'xlstm' in fn:
            return "xlstm"
        if 'hyena' in fn:
            return "hyena"
        return "unknown"

    def _clean_name(self, name: str) -> str:
        """Make a clean display name from a raw filename."""
        # RWKV-x060-World-1B6-v2.1-20240328-ctx4096 → Finch 1.6B
        if 'x060' in name:
            return 'Finch ' + name.split('-')[3] if len(name.split('-')) > 3 else name
        if 'x070' in name:
            return 'Goose ' + name.split('-')[3] if len(name.split('-')) > 3 else name
        return name
