########################################################################################################
# Bird's Nest — FastAPI Server
# WebSocket streaming chat + REST model management
########################################################################################################

import os, sys, json, time, asyncio, tempfile, shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from birdsnest.models import ModelManager, ARCH_CATEGORIES
from birdsnest.engine import InferenceEngine
from birdsnest.engines.rwkv_engine import RWKVEngine
from birdsnest.engines.mamba_engine import MambaEngine
from birdsnest.engines.hf_engine import HFEngine
from birdsnest.tools import (
    get_tools, get_enabled_tools, toggle_tool, execute_tool,
    detect_tool_call, is_definitely_not_tool_call, parse_tool_call,
    build_tool_system_prompt,
)


# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent / "web"
DATA_DIR = BASE_DIR / "data"
WORKSPACE_DIR = Path.home() / "birdsnest_workspace"
IMAGES_DIR = WORKSPACE_DIR / "images"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"

# In app mode, use ~/birdsnest_models (writable); in dev mode, use repo's models/
if os.environ.get("BIRDSNEST_APP_MODE"):
    MODELS_DIR = Path.home() / "birdsnest_models"
else:
    MODELS_DIR = BASE_DIR / "models"

# ── Global State ────────────────────────────────────────────────────────────
model_manager = ModelManager(str(MODELS_DIR))
active_engine: Optional[InferenceEngine] = None
rag_enabled = False
tools_enabled = True  # Tool calling on by default

try:
    from birdsnest.rag import RAGPipeline
    rag_pipeline = RAGPipeline(str(DATA_DIR))
    rag_available = True
except Exception as e:
    print(f"⚠️  RAG unavailable: {e}")
    rag_pipeline = None
    rag_available = False

# ── Engine Registry ─────────────────────────────────────────────────────────
ENGINE_MAP = {
    "rwkv": RWKVEngine,
    "mamba": MambaEngine,
    "xlstm": HFEngine,
    "hyena": HFEngine,
    "hybrid": HFEngine,
}

def get_engine_for_arch(architecture: str) -> InferenceEngine:
    """Create the right engine for a model's architecture."""
    engine_class = ENGINE_MAP.get(architecture)
    if engine_class is None:
        raise ValueError(f"No engine for architecture: {architecture}. Available: {list(ENGINE_MAP.keys())}")
    return engine_class()


# ── FastAPI App ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("""
╔══════════════════════════════════════════════════╗
║   🪹  Bird's Nest — Non-Transformer AI Hub       ║
║   Local AI for Mac • No Cloud • No API Keys      ║
╚══════════════════════════════════════════════════╝
    """)
    print(f"📦 Models dir: {MODELS_DIR}")
    print(f"🌐 Web UI: http://localhost:7861")
    print(f"📡 API docs: http://localhost:7861/docs\n")
    yield
    # Cleanup
    global active_engine
    if active_engine and active_engine.is_loaded:
        active_engine.unload()

app = FastAPI(
    title="Bird's Nest",
    description="Non-transformer AI chat for Mac",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request Models ──────────────────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    model_name: str

class DownloadModelRequest(BaseModel):
    catalog_id: str

class ChatMessage(BaseModel):
    message: str
    temperature: float = 1.0
    top_p: float = 0.7
    max_tokens: int = 500


# ── REST API: System Stats ───────────────────────────────────────────────────

@app.get("/api/system-stats")
async def system_stats():
    """Return process RAM usage and total model disk usage."""
    import resource
    # macOS: ru_maxrss is in bytes, Linux: in kilobytes
    import sys
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        ram_bytes = rusage.ru_maxrss  # bytes on macOS
    else:
        ram_bytes = rusage.ru_maxrss * 1024  # KB on Linux

    ram_gb = round(ram_bytes / 1024**3, 2)

    # Disk usage from all models
    disk = model_manager.disk_usage()

    return {
        "ram_gb": ram_gb,
        "disk_gb": disk.get("total_gb", 0),
        "model_count": disk.get("model_count", 0),
        "loaded": active_engine.model_name if active_engine and active_engine.is_loaded else None,
    }

# ── HuggingFace Cache Helper ─────────────────────────────────────────────────

def _scan_hf_cache(prefix_filter: str):
    """Scan HF hub cache for model dirs matching a prefix. Returns list of {id, dir_name, size_gb}."""
    import pathlib
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    results = []
    if not cache_dir.exists():
        return results
    for d in cache_dir.iterdir():
        if d.is_dir() and d.name.startswith("models--") and prefix_filter in d.name:
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            size_gb = round(size_bytes / 1024**3, 2)
            # Extract model id from dir name (models--org--name → org/name)
            model_id = d.name.replace("models--", "").replace("--", "/", 1)
            results.append({"id": model_id, "dir_name": d.name, "size_gb": size_gb})
    return results


def _delete_hf_model(dir_name: str):
    """Delete a model from HF cache by directory name. Returns freed_gb."""
    import pathlib
    import shutil
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / dir_name
    if model_dir.exists():
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        freed_gb = round(size_bytes / 1024**3, 2)
        shutil.rmtree(model_dir)
        return {"success": True, "freed_gb": freed_gb}
    return {"success": False, "error": "Model directory not found"}


# ── REST API: Image Models ───────────────────────────────────────────────────

# Load active image model from config (persist across restarts)
_img_config = Path.home() / "birdsnest_workspace" / ".birdsnest_image_model"
active_image_model = _img_config.read_text().strip() if _img_config.exists() else "schnell"

from birdsnest.models import IMAGE_MODEL_CATALOG as _IMG_CATALOG

@app.get("/api/image-models")
async def list_image_models():
    """Return catalog with installed status + active model."""
    # Scan HF cache for all downloaded models
    installed_repos = set()
    installed_raw = []
    for prefix in ["black-forest-labs", "FLUX", "mflux", "Tongyi", "briaai", "ByteDance"]:
        for m in _scan_hf_cache(prefix):
            if m["dir_name"] not in {x["dir_name"] for x in installed_raw}:
                installed_raw.append(m)
                installed_repos.add(m["id"])  # e.g. "Tongyi-MAI/Z-Image-Turbo"

    # Build catalog with installed flags
    catalog = []
    for entry in _IMG_CATALOG:
        hf_repo = entry.get("hf_repo", "")
        # Check if this model's HF repo is in the cache
        # HF cache uses -- separator: "models--org--name" → "org/name"
        is_installed = any(
            hf_repo.lower() in repo_id.lower() or repo_id.lower() in hf_repo.lower()
            for repo_id in installed_repos
        )
        catalog.append({
            "id": entry["id"],
            "name": entry.get("display_name", entry["name"]),
            "desc": entry.get("description", ""),
            "size_gb": entry.get("size_gb", 0),
            "steps": entry.get("default_steps", 4),
            "params": entry.get("params", ""),
            "hf_repo": hf_repo,
            "installed": is_installed,
            "legacy": entry.get("legacy", False),
            "type": "upscaler" if "upscale" in entry.get("capabilities", []) else "generator",
        })

    return {"catalog": catalog, "installed_raw": installed_raw, "active": active_image_model}


@app.post("/api/image-models/download")
async def download_image_model(request: Request):
    """Pre-download an image model from HuggingFace."""
    data = await request.json()
    model_id = data.get("model_id", "")

    entry = next((e for e in _IMG_CATALOG if e["id"] == model_id), None)
    if not entry:
        raise HTTPException(400, f"Unknown image model: {model_id}")

    hf_repo = entry.get("hf_repo")
    if not hf_repo:
        raise HTTPException(400, f"No HF repo for model: {model_id}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise HTTPException(500, "huggingface_hub not installed")

    loop = asyncio.get_running_loop()

    def _download():
        return snapshot_download(
            repo_id=hf_repo,
            local_dir_use_symlinks=True,
        )

    t0 = time.time()
    await loop.run_in_executor(None, _download)
    elapsed = time.time() - t0

    return {
        "status": "downloaded",
        "model_id": model_id,
        "size_gb": entry.get("size_gb", 0),
        "time": round(elapsed, 1),
    }


@app.post("/api/image-models/select")
async def select_image_model(request: Request):
    global active_image_model
    data = await request.json()
    active_image_model = data.get("model", "schnell")
    config_path = Path.home() / "birdsnest_workspace" / ".birdsnest_image_model"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(active_image_model)
    return {"success": True, "model": active_image_model}


@app.delete("/api/image-models/{dir_name}")
async def delete_image_model(dir_name: str):
    """Delete a cached image model from HF hub cache."""
    return _delete_hf_model(dir_name)


# ── Image performance settings ────────────────────────────────────────────────
image_quantize = "8"   # Default: int8 quantization
image_low_ram = False
image_style_preset = "none"
image_style_intensity = 2  # 1=Subtle, 2=Normal, 3=Strong

@app.post("/api/image-settings")
async def set_image_settings(request: Request):
    global image_quantize, image_low_ram, image_style_preset, image_style_intensity
    data = await request.json()
    image_quantize = data.get("quantize", "8")
    image_low_ram = data.get("low_ram", False)
    image_style_preset = data.get("style_preset", "none")
    image_style_intensity = data.get("style_intensity", 2)
    return {"success": True, "quantize": image_quantize, "low_ram": image_low_ram,
            "style_preset": image_style_preset, "style_intensity": image_style_intensity}


# ── REST API: Image Upload & Library ─────────────────────────────────────────

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to the workspace. Returns URL for inline display."""
    # Validate file type
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported image type: {ext}. Allowed: {', '.join(allowed)}")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).stem[:40].replace(" ", "_")
    filename = f"{safe_name}_{timestamp}{ext}"
    filepath = UPLOADS_DIR / filename

    content = await file.read()
    filepath.write_bytes(content)

    size_kb = len(content) / 1024
    return {
        "filename": filename,
        "url": f"/workspace/uploads/{filename}",
        "path": str(filepath),
        "size_kb": round(size_kb, 1),
    }


@app.get("/api/image-library")
async def list_image_library():
    """List all images in workspace (generated + uploaded)."""
    images = []

    for source, dir_path in [("generated", IMAGES_DIR), ("uploaded", UPLOADS_DIR)]:
        if not dir_path.exists():
            continue
        for f in sorted(dir_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                stat = f.stat()
                images.append({
                    "filename": f.name,
                    "url": f"/workspace/{source == 'generated' and 'images' or 'uploads'}/{f.name}",
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "created": stat.st_mtime,
                    "source": source,
                })

    return {"images": images, "total": len(images)}


@app.delete("/api/image-library/{filename}")
async def delete_library_image(filename: str):
    """Delete an image from the workspace."""
    # Check both directories
    for dir_path in [IMAGES_DIR, UPLOADS_DIR]:
        filepath = dir_path / filename
        if filepath.exists():
            size_kb = round(filepath.stat().st_size / 1024, 1)
            filepath.unlink()
            return {"success": True, "filename": filename, "freed_kb": size_kb}
    raise HTTPException(404, f"Image not found: {filename}")


# ── REST API: Music Models ───────────────────────────────────────────────────

@app.get("/api/music-models")
async def list_music_models():
    """Check which MusicGen models are cached locally."""
    installed = _scan_hf_cache("facebook--musicgen")
    return {"installed": installed}


@app.post("/api/music-models/select")
async def select_music_model(request: Request):
    data = await request.json()
    model_size = data.get("model", "small")
    from pathlib import Path
    config_path = Path.home() / "birdsnest_workspace" / ".birdsnest_music_model"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(model_size)
    return {"success": True, "model": model_size}


@app.delete("/api/music-models/{dir_name}")
async def delete_music_model(dir_name: str):
    """Delete a cached MusicGen model from HF hub cache."""
    return _delete_hf_model(dir_name)


# ── REST API: Embed Models ───────────────────────────────────────────────────

EMBED_HF_PREFIXES = ["sentence-transformers", "BAAI", "nomic-ai"]

@app.get("/api/embed-models")
async def list_embed_models():
    """Check which embedding models are cached locally."""
    installed = []
    for prefix in EMBED_HF_PREFIXES:
        installed.extend(_scan_hf_cache(prefix))
    seen = set()
    unique = []
    for m in installed:
        if m["dir_name"] not in seen:
            seen.add(m["dir_name"])
            unique.append(m)
    return {"installed": unique}


@app.delete("/api/embed-models/{dir_name}")
async def delete_embed_model(dir_name: str):
    """Delete a cached embedding model from HF hub cache."""
    return _delete_hf_model(dir_name)


# ── REST API: Translation Models ─────────────────────────────────────────────

@app.get("/api/translation-models")
async def list_translation_models():
    """Check which Opus-MT translation models are cached locally."""
    import pathlib
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    installed = []
    if cache_dir.exists():
        for d in cache_dir.iterdir():
            if d.name.startswith("models--Helsinki-NLP--opus-mt-"):
                # Extract language pair from dir name
                pair = d.name.replace("models--Helsinki-NLP--opus-mt-", "").replace("--", "-")
                installed.append(pair)
    return {"installed": installed}


@app.post("/api/translation-models/download")
async def download_translation_model(request: Request):
    """Pre-download an Opus-MT translation model."""
    data = await request.json()
    pair = data.get("pair", "")
    if not pair:
        return {"success": False, "error": "No language pair specified"}

    try:
        import asyncio
        loop = asyncio.get_running_loop()

        def _download():
            from transformers import MarianMTModel, MarianTokenizer
            model_name = f"Helsinki-NLP/opus-mt-{pair}"
            MarianTokenizer.from_pretrained(model_name)
            MarianMTModel.from_pretrained(model_name)
            return True

        await loop.run_in_executor(None, _download)
        return {"success": True, "pair": pair}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/translation-models/{pair}")
async def delete_translation_model(pair: str):
    """Delete a cached Opus-MT translation model from disk."""
    import pathlib
    import shutil
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    # HF cache uses -- as separator (e.g., models--Helsinki-NLP--opus-mt-en-es)
    hf_pair = pair.replace("-", "--", 1) if pair.count("-") == 1 else pair
    model_dir = cache_dir / f"models--Helsinki-NLP--opus-mt-{hf_pair}"
    
    if not model_dir.exists():
        # Try with the pair as-is
        model_dir = cache_dir / f"models--Helsinki-NLP--opus-mt-{pair}"
    
    if model_dir.exists():
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        freed_gb = round(size_bytes / 1024**3, 2)
        shutil.rmtree(model_dir)
        return {"success": True, "pair": pair, "freed_gb": freed_gb}
    
    return {"success": False, "error": f"Model for {pair} not found in cache"}


# ── REST API: Models ────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List local and available models."""
    local = model_manager.list_local()
    available = model_manager.list_available()
    disk = model_manager.disk_usage()

    # Find loaded model's display name for chat nickname
    loaded_name = None
    loaded_nickname = None
    if active_engine and active_engine.is_loaded:
        loaded_name = active_engine.model_name
        for m in local:
            if m["name"] == loaded_name:
                loaded_nickname = m.get("display_name", loaded_name)
                break

    return {
        "local": local,
        "catalog": available,
        "categories": ARCH_CATEGORIES,
        "disk_usage": disk,
        "loaded": loaded_name,
        "loaded_nickname": loaded_nickname,
    }

@app.post("/api/models/load")
async def load_model(req: LoadModelRequest):
    """Hot-load a model into GPU memory."""
    global active_engine

    # Find model path
    path = model_manager.get_model_path(req.model_name)
    if not path:
        raise HTTPException(404, f"Model not found: {req.model_name}")

    # Determine architecture
    local = model_manager.list_local()
    arch = "rwkv"  # default
    for m in local:
        if m["path"] == path:
            arch = m["architecture"]
            break

    # Unload current model
    if active_engine and active_engine.is_loaded:
        active_engine.unload()

    # Create engine and load
    try:
        active_engine = get_engine_for_arch(arch)
        info = active_engine.load(path)
        return {"status": "loaded", "model": req.model_name, "info": info}
    except Exception as e:
        raise HTTPException(500, f"Failed to load: {str(e)}")

@app.post("/api/models/unload")
async def unload_model():
    """Unload current model and free GPU memory."""
    global active_engine
    if active_engine and active_engine.is_loaded:
        name = active_engine.model_name
        active_engine.unload()
        return {"status": "unloaded", "model": name}
    return {"status": "no_model_loaded"}

@app.post("/api/models/download")
async def download_model(req: DownloadModelRequest):
    """Download a model from HuggingFace."""
    result = model_manager.download(req.catalog_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model from disk. Auto-unloads if currently active."""
    global active_engine
    # Auto-unload if this model is currently loaded
    if active_engine and active_engine.is_loaded and active_engine.model_name == model_name:
        active_engine.unload()
    
    result = model_manager.delete(model_name)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result

@app.post("/api/models/reset")
async def reset_conversation():
    """Reset conversation state."""
    global active_engine
    if active_engine and active_engine.is_loaded:
        if hasattr(active_engine, 'reset'):
            active_engine.reset()
        return {"status": "reset"}
    raise HTTPException(400, "No model loaded")

@app.get("/api/status")
async def get_status():
    """System status."""
    import torch
    engine_status = active_engine.get_status() if active_engine else {"loaded": False}
    return {
        "engine": engine_status,
        "system": {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "disk": model_manager.disk_usage(),
    }


# ── REST API: RAG ──────────────────────────────────────────────────────────

@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """Upload and ingest a document into the RAG pipeline."""
    if not rag_available:
        raise HTTPException(503, "RAG unavailable (chromadb incompatible with Python 3.14)")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        result = rag_pipeline.ingest(tmp.name, filename=file.filename)
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result
    finally:
        os.unlink(tmp.name)

@app.get("/api/rag/documents")
async def rag_list_docs():
    """List all indexed RAG documents."""
    if not rag_available:
        return {"total_documents": 0, "total_chunks": 0, "documents": [], "rag_unavailable": True}
    return rag_pipeline.get_stats()

@app.delete("/api/rag/documents/{doc_id}")
async def rag_delete_doc(doc_id: str):
    """Delete a document from the RAG index."""
    if not rag_available:
        raise HTTPException(503, "RAG unavailable")
    result = rag_pipeline.delete_document(doc_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result

@app.post("/api/rag/query")
async def rag_query(body: dict):
    """Query the RAG index directly (debug)."""
    if not rag_available:
        raise HTTPException(503, "RAG unavailable")
    text = body.get("query", "")
    top_k = body.get("top_k", 5)
    if not text:
        raise HTTPException(400, "Query text required")
    results = rag_pipeline.query(text, top_k=top_k)
    return {"results": results}

@app.post("/api/rag/toggle")
async def rag_toggle(body: dict):
    """Toggle RAG on/off."""
    global rag_enabled
    if not rag_available:
        return {"rag_enabled": False, "error": "RAG unavailable"}
    rag_enabled = body.get("enabled", False)
    return {"rag_enabled": rag_enabled}

@app.get("/api/rag/status")
async def rag_status():
    """Get RAG status."""
    if not rag_available:
        return {"rag_enabled": False, "available": False, "stats": {"total_documents": 0, "total_chunks": 0, "documents": []}}
    return {"rag_enabled": rag_enabled, "available": True, "stats": rag_pipeline.get_stats()}


# ── REST API: Update Check ─────────────────────────────────────────────────

_update_cache = {"checked": False, "result": None}

@app.get("/api/update-check")
async def update_check():
    """Check if a newer version is available on GitHub Releases."""
    if _update_cache["checked"]:
        return _update_cache["result"]

    import urllib.request

    # Read current version
    version = os.environ.get("BIRDSNEST_VERSION", "")
    if not version:
        version_file = Path(__file__).parent.parent / "VERSION"
        if version_file.exists():
            version = version_file.read_text().strip()
        else:
            version = "dev"

    result = {
        "current_version": version,
        "update_available": False,
        "latest_version": version,
        "download_url": None,
    }

    if version == "dev":
        _update_cache["checked"] = True
        _update_cache["result"] = result
        return result

    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/Achilles1089/birdsnest/releases/latest",
            headers={"User-Agent": "BirdsNest-UpdateCheck", "Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            latest = data.get("tag_name", "").lstrip("v")
            if latest and latest != version:
                # Simple version comparison
                try:
                    from packaging.version import Version
                    if Version(latest) > Version(version):
                        result["update_available"] = True
                except Exception:
                    # Fallback: string comparison
                    if latest > version:
                        result["update_available"] = True

                if result["update_available"]:
                    result["latest_version"] = latest
                    # Find DMG asset
                    for asset in data.get("assets", []):
                        if asset["name"].endswith(".dmg"):
                            result["download_url"] = asset["browser_download_url"]
                            break
                    if not result["download_url"]:
                        result["download_url"] = data.get("html_url", "")
    except Exception:
        pass  # Silently fail — update check is best-effort

    _update_cache["checked"] = True
    _update_cache["result"] = result
    return result


# ── REST API: Tools ────────────────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    """List all registered tools and their status."""
    tools = get_tools()
    return {
        "tools_enabled": tools_enabled,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "parameters": t.parameters,
            }
            for t in tools.values()
        ],
    }

@app.post("/api/tools/toggle")
async def tools_toggle(body: dict):
    """Toggle tool calling globally or per-tool."""
    global tools_enabled
    if "enabled" in body:
        tools_enabled = body["enabled"]
    if "tool_name" in body and "tool_enabled" in body:
        toggle_tool(body["tool_name"], body["tool_enabled"])
    return {"tools_enabled": tools_enabled, "tools": [
        {"name": t.name, "enabled": t.enabled} for t in get_tools().values()
    ]}


# ── WebSocket: Streaming Chat ──────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Streaming chat via WebSocket with tool call detection."""
    await websocket.accept()
    cancel_flag = {"cancelled": False}  # Mutable dict so inner functions can set it

    try:
        while True:
            cancel_flag["cancelled"] = False  # Reset for each message
            data = await websocket.receive_text()
            msg = json.loads(data)

            # Handle cancel messages
            if msg.get("type") == "cancel":
                cancel_flag["cancelled"] = True
                continue

            prompt = msg.get("message", "")
            temperature = msg.get("temperature", 1.0)
            top_p = msg.get("top_p", 0.7)
            max_tokens = msg.get("max_tokens", 500)
            use_rag = msg.get("rag", rag_enabled)
            use_tools = msg.get("tools", tools_enabled)
            image_path = msg.get("image_path", "")  # From drag-drop/attach

            if not prompt.strip():
                continue

            # ── Server-side intent detection (runs BEFORE model check) ──
            intent_handled = False
            if use_tools and get_enabled_tools():
                from birdsnest.tools import detect_user_intent
                intent = detect_user_intent(prompt)
                if intent:
                    tool_name, tool_args = intent
                    # Inject image path into tool args if available
                    if image_path and 'image_path' not in tool_args:
                        tool_args['image_path'] = image_path

                    # Send start marker — use model name if loaded, else fallback
                    model_label = "Bird's Nest"
                    if active_engine and active_engine.is_loaded:
                        model_label = active_engine.model_name
                        for m in model_manager.list_local():
                            if m["name"] == active_engine.model_name:
                                model_label = m.get("display_name", active_engine.model_name)
                                break
                    await websocket.send_json({"type": "start", "model": model_label, "nickname": model_label})

                    t0 = time.time()

                    # Execute tool directly
                    await websocket.send_json({
                        "type": "tool_call",
                        "name": tool_name,
                        "args": tool_args,
                    })

                    # Run in thread pool so WebSocket messages flush before blocking
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, execute_tool, tool_name, tool_args)
                    await websocket.send_json({
                        "type": "tool_result",
                        "name": tool_name,
                        "result": result,
                    })

                    elapsed = time.time() - t0
                    await websocket.send_json({
                        "type": "stats",
                        "tokens": 0,
                        "tok_s": 0,
                        "time": round(elapsed, 2),
                    })
                    await websocket.send_json({"type": "done"})
                    intent_handled = True

            if intent_handled:
                continue

            # ── Model required for non-tool messages ──
            if not active_engine or not active_engine.is_loaded:
                await websocket.send_json({
                    "type": "error",
                    "content": "No model loaded. Load one from the model library. (Tools still work — try the sidebar!)"
                })
                continue



            # RAG context injection
            rag_context = ""
            if use_rag and rag_available and rag_pipeline:
                try:
                    rag_context = rag_pipeline.build_context(prompt, top_k=5)
                except Exception:
                    pass

            # Tool system prompt (tells model about its capabilities)
            tool_context = ""
            if use_tools:
                tool_context = build_tool_system_prompt()

            # Image context injection
            image_context = ""
            if image_path:
                image_context = f"[User shared an image: {image_path}] "

            # Separate system prefix (fed silently into state) from user prompt
            system_prefix = ""
            if tool_context:
                system_prefix += tool_context + "\n\n"
            if rag_context:
                system_prefix += rag_context + "\n\n"

            # User prompt is just the actual user message + image context
            user_prompt = image_context + prompt

            # Send start marker — find nickname
            nickname = active_engine.model_name
            for m in model_manager.list_local():
                if m["name"] == active_engine.model_name:
                    nickname = m.get("display_name", active_engine.model_name)
                    break
            await websocket.send_json({"type": "start", "model": active_engine.model_name, "nickname": nickname})

            t0 = time.time()
            token_count = 0

            # ── Cancel listener: runs concurrently to check for stop requests ──
            async def cancel_listener():
                """Listen for cancel messages while generation is running."""
                try:
                    while not cancel_flag["cancelled"]:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        inner_msg = json.loads(data)
                        if inner_msg.get("type") == "cancel":
                            cancel_flag["cancelled"] = True
                            return
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

            # Start cancel listener as background task
            listener_task = asyncio.create_task(cancel_listener())

            # Stream tokens — model-based tool detection as fallback
            was_cancelled = False
            try:
                if use_tools and get_enabled_tools():
                    # ── Buffered mode: detect tool calls in model output ──
                    buffer = ""
                    tool_mode_confirmed = False
                    normal_mode_confirmed = False

                    for piece in active_engine.generate_stream(
                        user_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                        system_prefix=system_prefix
                    ):
                        if cancel_flag["cancelled"]:
                            was_cancelled = True
                            break
                        token_count += 1

                        # Already confirmed this is normal text — send immediately, skip buffering
                        if normal_mode_confirmed:
                            await websocket.send_json({"type": "token", "content": piece})
                            await asyncio.sleep(0)
                            continue

                        buffer += piece

                        # Check for complete tool call
                        fmt, match = detect_tool_call(buffer)
                        if match:
                            tool_mode_confirmed = True

                            # Send any text before the tool call
                            pre_text = buffer[:match.start()].strip()
                            if pre_text:
                                await websocket.send_json({"type": "token", "content": pre_text})

                            # Parse and execute tool
                            func_name, args = parse_tool_call(fmt, match)
                            await websocket.send_json({
                                "type": "tool_call",
                                "name": func_name,
                                "args": args,
                            })

                            result = execute_tool(func_name, args)
                            await websocket.send_json({
                                "type": "tool_result",
                                "name": func_name,
                                "result": result,
                            })

                            # Stop generation — small models hallucinate after tool calls
                            break

                        # Check if we can rule out tool call
                        if is_definitely_not_tool_call(buffer):
                            normal_mode_confirmed = True
                            await websocket.send_json({"type": "token", "content": buffer})
                            buffer = ""
                            continue

                        # Still undecided — keep buffering
                        await asyncio.sleep(0)

                    # Flush any remaining buffer
                    if buffer and not tool_mode_confirmed and not was_cancelled:
                        await websocket.send_json({"type": "token", "content": buffer})

                else:
                    # ── Direct mode: no tool detection ──
                    for piece in active_engine.generate_stream(
                        user_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                        system_prefix=system_prefix
                    ):
                        if cancel_flag["cancelled"]:
                            was_cancelled = True
                            break
                        token_count += 1
                        await websocket.send_json({"type": "token", "content": piece})
                        await asyncio.sleep(0)

            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})
            finally:
                listener_task.cancel()
                try:
                    await listener_task
                except (asyncio.CancelledError, Exception):
                    pass

            elapsed = time.time() - t0
            tok_s = token_count / elapsed if elapsed > 0 else 0

            if was_cancelled:
                await websocket.send_json({
                    "type": "cancelled",
                    "stats": {
                        "tokens": token_count,
                        "time": round(elapsed, 2),
                        "tok_s": round(tok_s, 1),
                    }
                })
            else:
                await websocket.send_json({
                    "type": "done",
                    "stats": {
                        "tokens": token_count,
                        "time": round(elapsed, 2),
                        "tok_s": round(tok_s, 1),
                    }
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass


# ── Static File Serving ─────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the chat UI."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Bird's Nest API is running. Web UI not found."})

# Mount static files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Mount workspace for generated images + uploads
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("BIRDSNEST_PORT", "7861"))
    host = "127.0.0.1" if os.environ.get("BIRDSNEST_APP_MODE") else "0.0.0.0"
    uvicorn.run(
        "birdsnest.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
