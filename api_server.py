#!/usr/bin/env python3
"""
FastAPI service for Arju Commander.

Run:
    python api_server.py

Then open:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import config
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from modules.ollama_engine import OllamaEngine
from modules.rag_module import RAGMemory
from modules.self_thinking import SelfThinkingEngine
from modules.vision_module import VisionModule


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    use_memory: bool = True
    store: bool = True
    mode: str = "autonomous"
    top_k: int | None = None
    image_path: str | None = None
    image_base64: str | None = None


class VisionRequest(BaseModel):
    question: str = "Describe this image clearly."
    use_memory: bool = True
    store: bool = True
    mode: str = "vlm"
    top_k: int | None = None
    capture_camera: bool = False
    image_path: str | None = None
    image_base64: str | None = None


class MemoryAddRequest(BaseModel):
    text: str = Field(..., min_length=1)
    category: str = "api"
    metadata: dict[str, Any] | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arju Commander API",
        description="Local FastAPI bridge for Arju VLM, RAG memory, and self-thinking chat.",
        version="1.0.0",
    )

    origins = getattr(config, "API_CORS_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.ai = OllamaEngine()
    app.state.rag = RAGMemory()
    app.state.vision = VisionModule(app.state.ai, app.state.rag)
    app.state.brain = SelfThinkingEngine(app.state.ai, app.state.rag)

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def site():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return {
            "service": "Arju Commander API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/api")
    async def root() -> dict[str, Any]:
        return {
            "service": "Arju Commander API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        rag = app.state.rag
        ai = app.state.ai
        return {
            "ok": True,
            "assistant": config.ASSISTANT_NAME,
            "boss": config.BOSS_NAME,
            "ollama_ready": ai.is_ready,
            "ollama_model": config.OLLAMA_MODEL,
            "memory_count": rag.count(),
        }

    @app.post("/chat")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        image_path = _resolve_image(req.image_path, req.image_base64)
        result = await asyncio.to_thread(
            app.state.brain.answer,
            req.message,
            image_path,
            req.use_memory,
            req.store,
            req.mode,
            req.top_k,
        )
        return _thought_response(result)

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        image_path = _resolve_image(req.image_path, req.image_base64)

        def events():
            iterator = app.state.brain.stream_answer(
                req.message,
                image_path=image_path,
                use_memory=req.use_memory,
                store=req.store,
                mode=req.mode,
                top_k=req.top_k,
            )
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration as done:
                    yield _sse("done", _thought_response(done.value))
                    break
                yield _sse("token", {"text": chunk})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/vision")
    async def vision(req: VisionRequest) -> dict[str, Any]:
        image_path = await _vision_image_path(app.state.vision, req)
        result = await asyncio.to_thread(
            app.state.brain.answer,
            req.question,
            image_path,
            req.use_memory,
            req.store,
            req.mode,
            req.top_k,
        )
        response = _thought_response(result)
        response["image_path"] = image_path
        return response

    @app.post("/memory")
    async def add_memory(req: MemoryAddRequest) -> dict[str, Any]:
        await asyncio.to_thread(
            app.state.rag.add,
            req.text,
            req.category,
            req.metadata,
        )
        return {"stored": True, "memory_count": app.state.rag.count()}

    @app.post("/memory/search")
    async def search_memory(req: MemorySearchRequest) -> dict[str, Any]:
        context = await asyncio.to_thread(
            app.state.rag.retrieve,
            req.query,
            req.top_k,
        )
        return {"query": req.query, "context": context}

    @app.get("/memory/count")
    async def memory_count() -> dict[str, int]:
        return {"count": app.state.rag.count()}

    @app.delete("/memory")
    async def clear_memory() -> dict[str, Any]:
        await asyncio.to_thread(app.state.rag.clear_all)
        return {"cleared": True, "memory_count": app.state.rag.count()}

    @app.websocket("/ws/chat")
    async def websocket_chat(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                payload = await ws.receive_json()
                try:
                    req = ChatRequest(**payload)
                    image_path = _resolve_image(req.image_path, req.image_base64)
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    continue

                await ws.send_json({"type": "start"})
                iterator = app.state.brain.stream_answer(
                    req.message,
                    image_path=image_path,
                    use_memory=req.use_memory,
                    store=req.store,
                    mode=req.mode,
                    top_k=req.top_k,
                )

                while True:
                    has_chunk, value = await asyncio.to_thread(_next_stream, iterator)
                    if not has_chunk:
                        await ws.send_json({"type": "done", **_thought_response(value)})
                        break
                    await ws.send_json({"type": "token", "text": value})
        except WebSocketDisconnect:
            return

    return app


def _thought_response(result) -> dict[str, Any]:
    if result is None:
        return {
            "answer": "",
            "thinking": "",
            "memory_context": "",
            "used_image": False,
            "stored": False,
        }
    return {
        "answer": result.answer,
        "thinking": result.thinking,
        "memory_context": result.memory_context,
        "used_image": result.used_image,
        "stored": result.stored,
    }


def _next_stream(iterator):
    try:
        return True, next(iterator)
    except StopIteration as done:
        return False, done.value


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def _resolve_image(image_path: str | None, image_base64: str | None) -> str | None:
    if image_base64:
        return _save_base64_image(image_base64)
    if not image_path:
        return None

    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {path}")
    return str(path)


async def _vision_image_path(vision: VisionModule, req: VisionRequest) -> str:
    image_path = _resolve_image(req.image_path, req.image_base64)
    if image_path:
        return image_path

    if req.capture_camera or not image_path:
        captured = await asyncio.to_thread(vision.capture)
        if captured:
            return str((BASE_DIR / captured).resolve())

    raise HTTPException(
        status_code=400,
        detail="Provide image_path/image_base64 or enable capture_camera.",
    )


def _save_base64_image(data: str) -> str:
    header = ""
    payload = data
    if "," in data and data.lower().startswith("data:image/"):
        header, payload = data.split(",", 1)

    ext = "jpg"
    if "image/png" in header:
        ext = "png"
    elif "image/webp" in header:
        ext = "webp"
    elif "image/jpeg" in header or "image/jpg" in header:
        ext = "jpg"

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image_base64 payload.") from exc

    upload_dir = BASE_DIR / getattr(config, "API_UPLOAD_DIR", "memory/api_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"api_image_{int(time.time() * 1000)}.{ext}"
    path.write_bytes(raw)
    return str(path)


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=getattr(config, "API_HOST", "127.0.0.1"),
        port=getattr(config, "API_PORT", 8000),
        reload=False,
    )
