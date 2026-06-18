# ============================================================
#   ARJU COMMANDER — modules/rag_module.py
#
#   FIX — ChromaDB ONNX model download blocks first retrieve
#   ──────────────────────────────────────────────────────────
#   ChromaDB downloads a 79 MB ONNX embedding model on the
#   very first query. This freezes the app mid-conversation.
#   Fix: prewarm() runs a dummy query in a background thread
#   immediately at startup so the model is downloaded/cached
#   before the user gives any command.
# ============================================================

import json
import os
import threading
import uuid
from datetime import datetime

import config

try:
    import chromadb
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False
    print("[RAG] chromadb not installed. Run: pip install chromadb")

_FALLBACK_PATH = os.path.join(config.RAG_DIR, "rag_fallback.json")


class RAGMemory:

    def __init__(self):
        os.makedirs(config.RAG_DIR, exist_ok=True)
        self._use_chroma = False
        self._col        = None
        self._fallback   = []
        self._ready      = threading.Event()
        self._lock       = threading.Lock()

        # Init in background so startup is instant
        threading.Thread(target=self._init, daemon=True,
                         name="RAG-Init").start()

    def _init(self):
        if _CHROMA_OK:
            try:
                client = chromadb.PersistentClient(path=config.RAG_DIR)
                self._col = client.get_or_create_collection(
                    name="arju_memory",
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
                count = self._col.count()
                print(f"[RAG] ChromaDB ready — {count} memories loaded.")

                # Prewarm: run a dummy query to trigger ONNX download NOW
                # (in background, so it doesn't block the user)
                if count == 0:
                    # Add and immediately delete a seed doc to warm the model
                    self._col.add(
                        documents=["Arju memory system initialized."],
                        metadatas=[{"category": "system"}],
                        ids=["_warmup"],
                    )
                    self._col.query(
                        query_texts=["hello"],
                        n_results=1,
                    )
                    self._col.delete(ids=["_warmup"])
                    print("[RAG] Embedding model warmed up.")
                else:
                    # Prewarm with existing data
                    self._col.query(query_texts=["hello"], n_results=1)
                    print("[RAG] Embedding model warmed up.")

            except Exception as e:
                print(f"[RAG] ChromaDB error: {e} — using JSON fallback.")
                self._use_chroma = False

        if not self._use_chroma:
            if os.path.exists(_FALLBACK_PATH):
                try:
                    with open(_FALLBACK_PATH) as f:
                        self._fallback = json.load(f)
                except Exception:
                    self._fallback = []
            print(f"[RAG] JSON fallback — {len(self._fallback)} memories.")

        self._ready.set()

    def _wait(self, timeout: float = 120) -> bool:
        """Block until RAG is ready. Returns True if ready."""
        return self._ready.wait(timeout=timeout)

    # ── Store ─────────────────────────────────────────────────

    def add(self, text: str, category: str = "general", metadata: dict = None):
        if not text or not text.strip():
            return
        if not self._wait(30):
            print("[RAG] Not ready yet — skipping store.")
            return

        ts   = datetime.now().isoformat()
        uid  = str(uuid.uuid4())[:8]
        meta = {"category": category, "timestamp": ts,
                "boss": config.BOSS_NAME}
        if metadata:
            meta.update(metadata)

        with self._lock:
            if self._use_chroma:
                try:
                    if self._col.count() >= config.RAG_MAX_MEMORIES:
                        old = self._col.get(limit=config.RAG_MAX_MEMORIES // 10)
                        if old["ids"]:
                            self._col.delete(ids=old["ids"])
                    self._col.add(
                        documents=[text.strip()],
                        metadatas=[meta],
                        ids=[uid],
                    )
                except Exception as e:
                    print(f"[RAG] Store error: {e}")
            else:
                self._fallback.append(
                    {"id": uid, "text": text.strip(), "meta": meta}
                )
                self._save_fallback()

    def add_qa(self, question: str, answer: str):
        self.add(f"Q: {question}\nA: {answer}", category="conversation")

    def add_correction(self, wrong: str, correct: str):
        text = (
            f"CORRECTION: When Arju said '{wrong}', "
            f"{config.BOSS_NAME} corrected it to '{correct}'. "
            f"Remember: '{correct}' is the right answer."
        )
        self.add(text, category="correction",
                 metadata={"wrong": wrong, "correct": correct})
        print(f"[RAG] Correction stored.")

    def add_preference(self, preference: str):
        self.add(
            f"USER PREFERENCE: {config.BOSS_NAME} prefers: {preference}",
            category="preference"
        )

    def add_vision_observation(self, observation: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
        text = f"CAMERA OBSERVATION at {ts}: {observation}"
        self.add(text, category="vision")

    # ── Retrieve ──────────────────────────────────────────────

    def retrieve(self, query: str, n: int = None) -> str:
        """
        Return relevant memories as a formatted string.
        Returns "" immediately if RAG is still warming up.
        """
        if not self._ready.is_set():
            return ""   # don't block; just run without context
        if not query or not query.strip():
            return ""

        n = n or config.RAG_TOP_K

        with self._lock:
            if self._use_chroma:
                return self._chroma_retrieve(query, n)
            else:
                return self._fallback_retrieve(query, n)

    def _chroma_retrieve(self, query: str, n: int) -> str:
        try:
            cnt = self._col.count()
            if cnt == 0:
                return ""
            results = self._col.query(
                query_texts=[query],
                n_results=min(n, cnt),
                include=["documents", "metadatas", "distances"],
            )
            docs  = results["documents"][0]
            dists = results["distances"][0]
            metas = results["metadatas"][0]
            out   = []
            for doc, dist, meta in zip(docs, dists, metas):
                similarity = 1 - dist
                if similarity >= config.RAG_SIMILARITY:
                    cat = meta.get("category", "")
                    out.append(f"[{cat}] {doc}" if cat else doc)
            return "\n".join(out)
        except Exception as e:
            print(f"[RAG] Retrieve error: {e}")
            return ""

    def _fallback_retrieve(self, query: str, n: int) -> str:
        words  = set(query.lower().split())
        scored = []
        for item in self._fallback:
            score = sum(1 for w in words if w in item["text"].lower())
            if score > 0:
                scored.append((score, item["text"]))
        scored.sort(reverse=True)
        return "\n".join(t for _, t in scored[:n])

    def _save_fallback(self):
        try:
            if len(self._fallback) > config.RAG_MAX_MEMORIES:
                self._fallback = self._fallback[-config.RAG_MAX_MEMORIES:]
            with open(_FALLBACK_PATH, "w") as f:
                json.dump(self._fallback, f, indent=2)
        except Exception as e:
            print(f"[RAG] Save error: {e}")

    # ── Stats / Ops ───────────────────────────────────────────

    def count(self) -> int:
        if not self._ready.is_set():
            return 0
        try:
            if self._use_chroma:
                return self._col.count()
            return len(self._fallback)
        except Exception:
            return 0

    def clear_all(self):
        if not self._wait(30):
            return
        with self._lock:
            if self._use_chroma:
                try:
                    from chromadb import PersistentClient
                    client = PersistentClient(path=config.RAG_DIR)
                    client.delete_collection("arju_memory")
                    self._col = client.get_or_create_collection("arju_memory")
                except Exception as e:
                    print(f"[RAG] Clear error: {e}")
            else:
                self._fallback = []
                self._save_fallback()
        print("[RAG] All memories cleared.")
