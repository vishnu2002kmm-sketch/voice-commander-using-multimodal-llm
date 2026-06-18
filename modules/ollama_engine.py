# ============================================================
#   ARJU COMMANDER — modules/ollama_engine.py
#
#   Wraps Ollama qwen3-vl:4b for:
#   • Text conversation with thinking (qwen3 <think> blocks)
#   • Vision QA — pass camera image to the model
#   • Streaming responses for real-time TTS
#   • Intent extraction — structured JSON output
#
#   qwen3-vl:4b thinking mode:
#   The model emits  <think>…</think>  before its final answer.
#   We capture that block for internal reasoning display but
#   strip it from the voice output so Vishnu only hears the
#   clean answer.
# ============================================================

import json
import os
import re
import time
import warnings
from typing import Generator

import config

warnings.filterwarnings("ignore")

try:
    import ollama as _ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False
    print("[Arju] ollama package not installed. Run: pip install ollama")


# ── Helpers ───────────────────────────────────────────────────

def _strip_think(text: str) -> tuple[str, str]:
    """
    Split model output into (thinking, answer).
    <think>reasoning here</think>  clean answer here
    Returns (think_block, clean_answer).
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    clean    = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, clean


def _json_from(text: str) -> dict | None:
    """Extract first JSON object from model text."""
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════
class OllamaEngine:
    """
    qwen3-vl:4b engine.
    All methods return plain English strings ready for TTS.
    """

    def __init__(self):
        self._model = config.OLLAMA_MODEL
        self._host  = config.OLLAMA_HOST
        self._ready = False
        self._conversation: list[dict] = []   # running chat history

        self._init_system_prompt()
        self._check_ollama()

    def _init_system_prompt(self):
        self._system = (
            f"You are {config.ASSISTANT_NAME}, a highly intelligent, multimodal "
            f"voice assistant for your boss {config.BOSS_NAME}. "
            f"You have vision (camera), memory (RAG), and reasoning (thinking) capabilities.\n\n"
            f"Personality rules:\n"
            f"• Address the user always as '{config.BOSS_NAME}' or 'boss'.\n"
            f"• Be concise — 1-3 sentences for voice replies.\n"
            f"• If you are unsure what was said, say so and ask for clarification.\n"
            f"• If you are corrected, acknowledge it, learn from it, and apologise.\n"
            f"• Think step-by-step before answering (use your internal reasoning).\n"
            f"• Never make up facts. Say 'I don't know' when uncertain.\n"
            f"• For vision queries, describe what you actually see precisely.\n"
        )

    def _check_ollama(self):
        if not _OLLAMA_OK:
            print("[Arju] Ollama not installed.")
            return
        try:
            client = self._client()
            models = client.list()
            names  = [m["model"] for m in models.get("models", [])]
            if self._model in names or any(self._model in n for n in names):
                self._ready = True
                print(f"[Arju] Ollama ready — model: {self._model}")
            else:
                print(f"[Arju] Model '{self._model}' not found.")
                print(f"[Arju] Run:  ollama pull {self._model}")
        except Exception as e:
            print(f"[Arju] Ollama connection error: {e}")
            print("[Arju] Make sure Ollama is running:  ollama serve")

    def _client(self):
        return _ollama.Client(host=self._host)

    @property
    def is_ready(self) -> bool:
        return self._ready and _OLLAMA_OK

    # ── Internal chat ─────────────────────────────────────────

    def _build_messages(self, user_text: str,
                        image_path = None,
                        context: str    = None,
                        include_history: bool = True) -> list[dict]:
        msgs = [{"role": "system", "content": self._system}]

        # Inject RAG context if provided
        if context:
            msgs.append({
                "role": "system",
                "content": f"Relevant memories from past conversations:\n{context}"
            })

        # Include recent conversation history (last 8 turns)
        if include_history and self._conversation:
            msgs.extend(self._conversation[-8:])

        # Build current user message
        user_msg: dict = {"role": "user", "content": user_text}
        if image_path:
            if isinstance(image_path, (list, tuple)):
                images = [p for p in image_path if p and os.path.exists(p)]
            else:
                images = [image_path] if os.path.exists(image_path) else []
            if images:
                user_msg["images"] = images

        msgs.append(user_msg)
        return msgs

    def _call(self, messages: list[dict], stream: bool = False):
        """Raw ollama call — returns full text or generator."""
        client = self._client()
        options = {
            "num_predict":  config.OLLAMA_MAX_TOKENS,
            "temperature":  0.7,
            "top_p":        0.9,
        }
        response = client.chat(
            model   = self._model,
            messages= messages,
            options = options,
            stream  = stream,
        )
        return response

    # ── Public APIs ───────────────────────────────────────────

    def think_and_reply(self,
                        user_text: str,
                        image_path: str = None,
                        context: str    = None,
                        include_history: bool = True) -> tuple[str, str]:
        """
        Full reasoning reply.
        Returns (thinking_block, clean_answer).
        Thinking block shown in console, clean answer spoken aloud.
        Also updates conversation history.
        """
        if not self.is_ready:
            return "", "Ollama is not ready. Please start it with: ollama serve"

        try:
            msgs     = self._build_messages(
                user_text, image_path, context, include_history=include_history
            )
            response = self._call(msgs, stream=False)
            raw      = response["message"]["content"]

            thinking, answer = _strip_think(raw)

            if thinking:
                print(f"\n[Arju THINKING]\n{thinking}\n{'─'*40}")

            # Update conversation history
            self._conversation.append({"role": "user",     "content": user_text})
            self._conversation.append({"role": "assistant", "content": answer})

            return thinking, answer

        except Exception as e:
            err = f"I encountered an error: {str(e)[:80]}"
            return "", err

    def stream_reply(self,
                     user_text: str,
                     image_path: str = None,
                     context: str    = None) -> Generator[str, None, None]:
        """
        Stream tokens for real-time TTS.
        Yields sentence-complete chunks (for smoother TTS).
        Strips <think> blocks from yielded output.
        """
        if not self.is_ready:
            yield "Ollama is not ready. Please start it with ollama serve."
            return

        try:
            msgs     = self._build_messages(user_text, image_path, context)
            stream   = self._call(msgs, stream=True)

            buffer     = ""
            in_think   = False
            full_reply = ""

            for chunk in stream:
                token = chunk["message"]["content"]
                buffer += token

                # Track think block boundaries
                if "<think>" in buffer:
                    in_think = True
                if "</think>" in buffer:
                    in_think = False
                    # strip the think block from buffer
                    buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL)

                if in_think:
                    continue     # don't yield thinking tokens

                full_reply += token

                # Yield at sentence boundaries for smooth TTS
                for sep in [".", "!", "?", "\n"]:
                    if sep in buffer and not in_think:
                        parts   = buffer.split(sep, 1)
                        sentence= parts[0].strip()
                        if sentence:
                            yield sentence + sep
                        buffer = parts[1] if len(parts) > 1 else ""
                        break

            # Yield any remaining buffer
            if buffer.strip() and not in_think:
                yield buffer.strip()

            # Update history with complete reply
            _, clean = _strip_think(full_reply)
            self._conversation.append({"role": "user",     "content": user_text})
            self._conversation.append({"role": "assistant", "content": clean})

        except Exception as e:
            yield f"Error: {str(e)[:80]}"

    def vision_query(self, image_path: str, question: str,
                     context: str = None) -> tuple[str, str]:
        """Ask a visual question about an image file (non-streaming)."""
        return self.think_and_reply(question, image_path=image_path, context=context)

    def speak_stream_vision(self, image_path, question: str,
                            voice_engine, context: str = None) -> str:
        """Streaming vision query — Arju speaks as tokens arrive."""
        return self.speak_stream(
            user_text   = question,
            voice_engine= voice_engine,
            image_path  = image_path,
            context     = context,
        )

    def speak_stream(self, user_text: str, voice_engine,
                     image_path: str = None,
                     context: str = None) -> str:
        """
        THE MAIN METHOD FOR ALL VOICE RESPONSES.

        Streams tokens from Ollama and feeds each complete sentence
        to TTS the moment it arrives — so Arju starts speaking
        within 1-2 seconds instead of waiting 30-60s for the
        full response.

        Pipeline:
          Ollama token stream
            → accumulate until sentence boundary (. ! ? \\n)
            → voice.speak_now(sentence)   ← heard immediately
            → repeat until done

        Returns the full answer text (for RAG storage).
        """
        if not self.is_ready:
            msg = "Ollama is not ready. Please run ollama serve."
            voice_engine.speak_now(msg)
            return msg

        full_text  = ""
        buffer     = ""
        in_think   = False
        think_buf  = ""

        try:
            msgs   = self._build_messages(user_text, image_path, context)
            stream = self._call(msgs, stream=True)

            for chunk in stream:
                token   = chunk["message"]["content"]
                full_text += token

                # ── Track <think> blocks ──────────────────────
                if "<think>" in token:
                    in_think = True
                if "</think>" in token:
                    in_think = False
                    # flush think buffer to console
                    if think_buf:
                        print(f"\n[Arju THINKING] {think_buf[:300]}{'...' if len(think_buf)>300 else ''}")
                        think_buf = ""
                    # strip </think> from buffer
                    buffer = re.sub(r"<think>.*?</think>", "", buffer + token,
                                    flags=re.DOTALL)
                    continue

                if in_think:
                    think_buf += token
                    continue

                # ── Accumulate answer tokens ───────────────────
                buffer += token

                # Speak at sentence boundaries
                for sep in (".", "!", "?", "\n"):
                    if sep in buffer:
                        parts    = buffer.split(sep, 1)
                        sentence = parts[0].strip()
                        if len(sentence) > 3:          # skip very short fragments
                            print(f"[{config.ASSISTANT_NAME}] {sentence}{sep}")
                            voice_engine.speak_now(sentence + sep, log=False)
                        buffer = parts[1] if len(parts) > 1 else ""
                        break

            # Speak any remaining buffer
            leftover = buffer.strip()
            if len(leftover) > 3:
                print(f"[{config.ASSISTANT_NAME}] {leftover}")
                voice_engine.speak_now(leftover, log=False)

            # Save to conversation history
            _, clean = _strip_think(full_text)
            self._conversation.append({"role": "user",     "content": user_text})
            self._conversation.append({"role": "assistant", "content": clean})
            return clean

        except Exception as e:
            err = f"I had an error, {config.BOSS_NAME}: {str(e)[:60]}"
            voice_engine.speak_now(err)
            return err

    def interpret_intent(self, raw_text: str) -> dict:
        """
        Use qwen3 to interpret a possibly unclear command.
        Returns dict with keys:
          intent     : string label
          params     : dict of extracted parameters
          confidence : "high" | "medium" | "low"
          guess      : natural language restatement of what was understood
        """
        if not self.is_ready:
            return {"intent": "unknown", "confidence": "low",
                    "guess": raw_text, "params": {}}

        prompt = (
            f"The user said: \"{raw_text}\"\n\n"
            "Interpret this as a voice command for a smart assistant. "
            "Reply ONLY with a JSON object (no extra text, no markdown) with these exact keys:\n"
            "  intent     : one of [open_app, web_search, youtube_play, volume_up, volume_down,\n"
            "               mute, unmute, screenshot, get_time, get_date, vision_identify,\n"
            "               vision_describe, vision_emotion, vision_ocr, vision_count,\n"
            "               vision_activity, vision_detect, vision_color, vision_presence,\n"
            "               vision_chat, gesture_start, joke, motivate, chat, exit, unknown]\n"
            "  params     : object with any extracted values (app_name, query, label, subject)\n"
            "  confidence : 'high' if clear, 'medium' if mostly clear, 'low' if unclear\n"
            "  guess      : one natural-language sentence restating what you understood\n"
            "Example: {\"intent\":\"open_app\",\"params\":{\"app_name\":\"chrome\"},\"confidence\":\"high\","
            "\"guess\":\"Open Google Chrome\"}"
        )

        try:
            msgs     = self._build_messages(prompt, include_history=False)
            response = self._call(msgs, stream=False)
            raw      = response["message"]["content"]
            _, clean = _strip_think(raw)
            result   = _json_from(clean)
            if result:
                return result
        except Exception as e:
            print(f"[Arju] Intent parse error: {e}")

        return {
            "intent":     "unknown",
            "confidence": "low",
            "guess":      raw_text,
            "params":     {}
        }

    def generate_clarification(self, raw_text: str, guess: str) -> str:
        """
        Generate a human-like clarification question.
        e.g. "I think you said 'describe scene' — is that right, Vishnu?"
        """
        prompt = (
            f"The user said something unclear: '{raw_text}'\n"
            f"My best guess is: '{guess}'\n"
            f"Write a very short, natural clarification question (one sentence) "
            f"asking the user to confirm. Address them as {config.BOSS_NAME}. "
            f"Do not add any explanation."
        )
        try:
            msgs     = self._build_messages(prompt, include_history=False)
            response = self._call(msgs, stream=False)
            raw      = response["message"]["content"]
            _, clean = _strip_think(raw)
            return clean.strip()
        except Exception:
            return (
                f"I think you said '{guess}' — is that right, "
                f"{config.BOSS_NAME}?"
            )

    def acknowledge_correction(self, was_wrong: str, correct: str) -> str:
        """Generate a natural acknowledgement of a correction."""
        prompt = (
            f"I said or did '{was_wrong}' but the correct thing was '{correct}'. "
            f"Write a brief, genuine apology and acknowledgement (1-2 sentences). "
            f"Address the user as {config.BOSS_NAME}."
        )
        try:
            msgs     = self._build_messages(prompt, include_history=False)
            response = self._call(msgs, stream=False)
            raw      = response["message"]["content"]
            _, clean = _strip_think(raw)
            return clean.strip()
        except Exception:
            return (
                f"I'm sorry {config.BOSS_NAME}, I was wrong. "
                f"I've noted the correction and will do better."
            )

    def clear_history(self):
        """Reset conversation history."""
        self._conversation = []

    def joke(self) -> str:
        _, ans = self.think_and_reply(
            "Tell me one short, clever, funny joke. Keep it to 2 sentences max.",
            include_history=False if hasattr(self, '_call') else True
        )
        return ans

    def motivate(self) -> str:
        _, ans = self.think_and_reply(
            f"Give {config.BOSS_NAME} a powerful one-sentence motivation right now."
        )
        return ans
