#!/usr/bin/env python3
# ============================================================
#   ARJU VOICE COMMANDER
#   Boss: Vishnu  |  Assistant: Arju  |  Engine: qwen3-vl:4b
#
#   FIXES IN THIS VERSION
#   ──────────────────────
#   FIX 1 — Greeting spoken/printed twice
#     Root cause: speak_now() on main thread + background
#     _memory_report thread calling voice.speak() while TTS
#     worker was still active — queued the greeting again.
#     Fix: All startup speech uses speak_now() (blocking).
#     Memory report waits for RAG then speaks in sequence.
#     No concurrent TTS during startup.
#
#   FIX 2 — Wake loop fires during command processing
#     Root cause: voice.set_busy() was called AFTER
#     extract_inline_command() — gap where wake loop could
#     insert a second command.
#     Fix: set_busy() called the INSTANT wake word confirmed,
#     before any command processing begins.
#
#   FIX 3 — Slow/silent ambiguity handling
#     Moved to thinking_module: handle_ambiguous() is instant,
#     no Ollama calls. Just asks Vishnu to repeat.
# ============================================================

import os, sys, random, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("memory",      exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

import config
from modules.voice_module      import VoiceEngine
from modules.ollama_engine     import OllamaEngine
from modules.rag_module        import RAGMemory
from modules.vision_module     import VisionModule
from modules.thinking_module   import ThinkingModule
from modules.gesture_module    import GestureController
from modules.command_processor import CommandProcessor
from modules.terminal_module   import TerminalNarrator

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║          A R J U   V O I C E   C O M M A N D E R           ║
║   Boss: Vishnu  |  Engine: qwen3-vl:4b  |  100% Local       ║
║   Thinking · RAG Memory · Vision · Self-Correction          ║
╚══════════════════════════════════════════════════════════════╝
"""

GUIDE = """
┌──────────────────────────────────────────────────────────────┐
│                  ARJU — COMMAND GUIDE                        │
├──────────────────────────────────────────────────────────────┤
│  WAKE WORD: Say "Arju" to activate                           │
│  (aliases: arjun, arjo, raju, orju, arjuna)                  │
│                                                              │
│  SYSTEM ─────────────────────────────────────────────────── │
│  "Arju open Chrome"          "Arju search AI on Google"      │
│  "Arju increase volume"      "Arju mute / unmute"            │
│  "Arju screenshot"           "Arju what time is it"          │
│  "Arju lock screen"                                          │
│                                                              │
│  VISION (qwen3-vl sees your camera) ───────────────────── │
│  "Arju what is this"         ← hold object up                │
│  "Arju describe scene"       ← full description              │
│  "Arju count objects"        ← count everything              │
│  "Arju detect my emotion"    ← read your face                │
│  "Arju read text"            ← OCR                           │
│  "Arju what colour is my shirt"                              │
│  "Arju can you see my laptop"                                │
│  "Arju vision chat"          ← multi-turn camera Q&A         │
│  "end vision"                ← stop vision chat              │
│                                                              │
│  THINKING & MEMORY ────────────────────────────────────── │
│  Unclear → Arju asks you to repeat (instant, no wait)        │
│  Wrong answer → say "that's wrong" → Arju learns             │
│  "Remember that I prefer dark mode"                          │
│  "How many memories do you have"                             │
│  "Clear memory"  /  "Reset conversation"                     │
│                                                              │
│  AI CONVERSATION ──────────────────────────────────────── │
│  Ask anything — Arju streams answer as it thinks             │
│                                                              │
│  GESTURE CONTROL ──────────────────────────────────────── │
│  "Start gesture control"                                     │
│  Thumb Up=Vol Up  Thumb Down=Vol Down  Palm=Mute             │
│                                                              │
│  "Arju goodbye" / "stop" / "exit"  → quit                   │
└──────────────────────────────────────────────────────────────┘
"""

_MOTIV = [
    "Think big, act bigger.",
    "Every problem is a puzzle waiting to be solved.",
    "The best time to start is right now.",
    f"You have got this, {config.BOSS_NAME}.",
    "Stay sharp. Stay focused.",
]


def get_greeting() -> str:
    h = time.localtime().tm_hour
    if   h < 12: period = "Good morning"
    elif h < 17: period = "Good afternoon"
    elif h < 21: period = "Good evening"
    else:         period = "Good night"
    return (f"{period}, {config.BOSS_NAME}! "
            "Arju is online and ready.")


def tick(name: str, ok: bool, note: str = ""):
    icon = "OK" if ok else "!!"
    note = f"  ({note})" if note else ""
    print(f"  [{icon}] {name}{note}")


def main():
    print(BANNER)
    print("[INIT] Starting Arju Commander...\n")

    # ── Init all modules ──────────────────────────────────────
    voice   = VoiceEngine()
    ai      = OllamaEngine()
    rag     = RAGMemory()
    vision  = VisionModule(ai, rag)
    thinker = ThinkingModule(ai, rag, voice)
    gesture = GestureController(voice)
    terminal = TerminalNarrator(voice)
    proc    = CommandProcessor(voice, vision, ai, rag, thinker, gesture, terminal)

    # ── Status ────────────────────────────────────────────────
    print("\n[INIT] Module status:")
    thresh = int(voice.rec.energy_threshold) if voice.rec else "N/A"
    tick("TTS",             voice.tts is not None)
    tick("STT",             voice.rec is not None, f"threshold={thresh}")
    tick("Ollama qwen3-vl", ai.is_ready,
         "run 'ollama serve' if not ready")
    tick("RAG memory",      True, "warming in background")
    tick("Vision (camera)", True)
    tick("Thinking engine", True, "instant — no Ollama calls")
    tick("Gesture control", gesture.available)
    tick("Terminal voice",   True, f"logs={config.TERMINAL_LOG_DIR}")
    print()

    # ── FIX 1: Greeting spoken ONCE, in sequence ──────────────
    # speak_now() is blocking — no concurrent TTS, no duplicates.
    greeting = get_greeting()
    print(f"[{config.ASSISTANT_NAME}] {greeting}")
    voice.speak_now(greeting, log=False)          # ← single call, no queue

    mot = random.choice(_MOTIV)
    print(f"[{config.ASSISTANT_NAME}] {mot}")
    voice.speak_now(mot, log=False)               # ← blocking, no queue

    # Memory count — spoken after RAG warms up (background thread)
    def _mem_notify():
        rag._wait(90)
        n = rag.count()
        if n > 0:
            msg = f"I remember {n} things from our past conversations."
            print(f"[{config.ASSISTANT_NAME}] {msg}")
            # Only speak if not currently speaking something else
            if not voice._is_speaking.is_set():
                voice.speak_now(msg, log=False)
        else:
            print("[RAG] Fresh start — no past memories yet.")

    threading.Thread(target=_mem_notify, daemon=True, name="MemNotify").start()

    print(GUIDE)
    print(f"[MODE] Type commands OR say 'ARJU' anytime\n")

    # ── FIX 2: Wake loop with tight busy lock ─────────────────
    _stop = threading.Event()

    def wake_loop():
        while not _stop.is_set():
            if not voice.rec:
                break

            # Skip if already processing a command
            if voice.is_busy():
                time.sleep(0.1)
                continue

            phrase = voice.listen_wake()
            if not phrase:
                continue
            if not voice.contains_wake_word(phrase):
                continue

            # ── Set busy IMMEDIATELY — before any processing ───
            # This prevents a second wake detection while we work
            voice.set_busy()
            try:
                print(f"\n[Wake] '{phrase}'")
                inline = voice.extract_inline_command(phrase)

                if inline:
                    print(f"[Wake] Command: '{inline}'")
                    cmd = inline
                else:
                    # Wake word only — listen for the command
                    cmd = voice.get_command()

                if cmd:
                    keep = proc.process(cmd)
                    if not keep:
                        _stop.set()
            except Exception as e:
                print(f"[Wake] Error: {e}")
            finally:
                voice.clear_busy()   # ← always release, even on error

    if voice.rec:
        threading.Thread(
            target=wake_loop, daemon=True, name="WakeWord"
        ).start()
    else:
        print("[WARN] No microphone — text input only.\n")

    # ── Main text input loop ──────────────────────────────────
    try:
        while not _stop.is_set():
            try:
                raw = input(f"\n{config.BOSS_NAME} ▶ ").strip()
            except EOFError:
                break
            if not raw:
                continue

            voice.set_busy()
            try:
                keep = proc.process(raw)
                if not keep:
                    break
            except Exception as e:
                print(f"[Main] Error: {e}")
                voice.speak_now(
                    f"I had an error, {config.BOSS_NAME}. Please try again."
                )
            finally:
                voice.clear_busy()

    except KeyboardInterrupt:
        print("\n[CTRL+C] Stopping...")

    finally:
        _stop.set()
        rag.add("Session ended.", category="session")
        voice.speak_now(
            f"Shutting down. Goodbye {config.BOSS_NAME}! See you soon.",
            log=False
        )
        print(f"\n[{config.ASSISTANT_NAME}] Shutting down. Goodbye {config.BOSS_NAME}!")
        gesture.stop()
        vision.release()
        voice.shutdown()
        print("[SHUTDOWN] Arju offline.")


if __name__ == "__main__":
    main()
