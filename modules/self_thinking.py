"""
Autonomous thinking layer for Arju.

This module keeps the policy for API/chat answers in one place:
retrieve memory, ask the VLM/LLM to reason with that context, return the
clean answer, and store the new learning back into RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator


@dataclass
class ThoughtResult:
    answer: str
    thinking: str
    memory_context: str
    used_image: bool
    stored: bool


class SelfThinkingEngine:
    """
    High-level brain for text and VLM API calls.

    The Ollama qwen3-vl model can emit <think>...</think> blocks. The lower
    engine strips them from spoken output; this layer preserves them for API
    clients while still storing only the useful final result in memory.
    """

    def __init__(self, ollama_engine, rag_memory):
        self.ai = ollama_engine
        self.rag = rag_memory

    def _retrieve(self, prompt: str, use_memory: bool, top_k: int | None = None) -> str:
        if not use_memory:
            return ""
        try:
            return self.rag.retrieve(prompt, n=top_k)
        except Exception as exc:
            print(f"[SelfThink] Memory retrieve failed: {exc}")
            return ""

    def _context(self, memory_context: str, image_path: str | None, mode: str) -> str:
        parts = [
            "Autonomous thinking instructions:",
            "- Understand the user's real goal before answering.",
            "- Use retrieved memory when it is relevant, but do not force it.",
            "- If an image is present, ground the answer in visible evidence.",
            "- Decide what is uncertain and say so instead of guessing.",
            "- Give a direct, useful final answer for Vishnu.",
            f"- Reasoning mode requested by client: {mode or 'autonomous'}.",
        ]
        if image_path:
            parts.append(f"- Image available to the VLM: {image_path}")
        if memory_context:
            parts.extend(["", "Retrieved memory:", memory_context])
        return "\n".join(parts)

    def answer(
        self,
        prompt: str,
        image_path: str | None = None,
        use_memory: bool = True,
        store: bool = True,
        mode: str = "autonomous",
        top_k: int | None = None,
    ) -> ThoughtResult:
        memory_context = self._retrieve(prompt, use_memory, top_k=top_k)
        context = self._context(memory_context, image_path, mode)

        thinking, answer = self.ai.think_and_reply(
            prompt,
            image_path=image_path,
            context=context,
        )

        stored = False
        if store and answer:
            stored = self._store(prompt, answer, image_path=image_path)

        return ThoughtResult(
            answer=answer,
            thinking=thinking,
            memory_context=memory_context,
            used_image=bool(image_path),
            stored=stored,
        )

    def stream_answer(
        self,
        prompt: str,
        image_path: str | None = None,
        use_memory: bool = True,
        store: bool = True,
        mode: str = "autonomous",
        top_k: int | None = None,
    ) -> Generator[str, None, ThoughtResult]:
        memory_context = self._retrieve(prompt, use_memory, top_k=top_k)
        context = self._context(memory_context, image_path, mode)

        chunks: list[str] = []
        for chunk in self.ai.stream_reply(
            prompt,
            image_path=image_path,
            context=context,
        ):
            chunks.append(chunk)
            yield chunk

        answer = " ".join(part.strip() for part in chunks if part.strip()).strip()
        stored = False
        if store and answer:
            stored = self._store(prompt, answer, image_path=image_path)

        return ThoughtResult(
            answer=answer,
            thinking="",
            memory_context=memory_context,
            used_image=bool(image_path),
            stored=stored,
        )

    def _store(self, prompt: str, answer: str, image_path: str | None = None) -> bool:
        try:
            if image_path:
                self.rag.add_vision_observation(
                    f"API VLM image={image_path} | Q: {prompt} | A: {answer[:500]}"
                )
            else:
                self.rag.add_qa(prompt, answer)
            return True
        except Exception as exc:
            print(f"[SelfThink] Memory store failed: {exc}")
            return False
