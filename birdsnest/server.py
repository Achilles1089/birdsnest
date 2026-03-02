########################################################################################################
# Bird's Nest — FastAPI Server
# WebSocket streaming chat + REST model management
########################################################################################################

import os, sys, json, time, asyncio, tempfile, shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
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
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = Path(__file__).parent / "web"
DATA_DIR = BASE_DIR / "data"

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
    """Delete a model from disk."""
    global active_engine
    # Don't delete loaded model
    if active_engine and active_engine.is_loaded and active_engine.model_name == model_name:
        raise HTTPException(400, "Cannot delete currently loaded model. Unload first.")
    
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

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            prompt = msg.get("message", "")
            temperature = msg.get("temperature", 1.0)
            top_p = msg.get("top_p", 0.7)
            max_tokens = msg.get("max_tokens", 500)
            use_rag = msg.get("rag", rag_enabled)
            use_tools = msg.get("tools", tools_enabled)

            if not prompt.strip():
                continue

            # ── Server-side intent detection (runs BEFORE model check) ──
            intent_handled = False
            if use_tools and get_enabled_tools():
                from birdsnest.tools import detect_user_intent
                intent = detect_user_intent(prompt)
                if intent:
                    tool_name, tool_args = intent

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

                    result = execute_tool(tool_name, tool_args)
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

            # User prompt stays clean — just the user's message + any RAG context
            user_prompt = (rag_context or "") + prompt

            # Send start marker — find nickname
            nickname = active_engine.model_name
            for m in model_manager.list_local():
                if m["name"] == active_engine.model_name:
                    nickname = m.get("display_name", active_engine.model_name)
                    break
            await websocket.send_json({"type": "start", "model": active_engine.model_name, "nickname": nickname})

            t0 = time.time()
            token_count = 0

            # Stream tokens — model-based tool detection as fallback
            try:
                if use_tools and get_enabled_tools():
                    # ── Buffered mode: detect tool calls in model output ──
                    buffer = ""
                    tool_mode_confirmed = False
                    normal_mode_confirmed = False

                    for piece in active_engine.generate_stream(
                        user_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
                    ):
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
                    if buffer and not tool_mode_confirmed:
                        await websocket.send_json({"type": "token", "content": buffer})

                else:
                    # ── Direct mode: no tool detection ──
                    for piece in active_engine.generate_stream(
                        user_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
                    ):
                        token_count += 1
                        await websocket.send_json({"type": "token", "content": piece})
                        await asyncio.sleep(0)

            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})

            elapsed = time.time() - t0
            tok_s = token_count / elapsed if elapsed > 0 else 0

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

# Mount workspace for generated images
WORKSPACE_SERVE = Path.home() / "birdsnest_workspace"
if WORKSPACE_SERVE.exists():
    app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_SERVE)), name="workspace")


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "birdsnest.server:app",
        host="0.0.0.0",
        port=7861,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
