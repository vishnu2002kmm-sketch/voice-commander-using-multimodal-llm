# ============================================================
#   ARJU COMMANDER — modules/voice_module.py
#
#   FIXES IN THIS VERSION
#   ─────────────────────
#   FIX 1 — Energy threshold 5422 / 4249 — mic became deaf
#     Root cause: adjust_for_ambient_noise() in a noisy room
#     calibrates to background sound level, then max() with
#     SR_ENERGY=600 still allows very high values.
#     Fix: HARD CAP at 1500. Above that the mic hears nothing.
#
#   FIX 2 — "can you hear me arjun" — alias left in command
#     Root cause: extract_inline_command() only stripped alias
#     if it appeared at the START of the phrase. "arjun" at
#     the END was left in → command became "can you hear me arjun"
#     Fix: strip ALL aliases from ANYWHERE in the phrase,
#     not just from the beginning.
#
#   FIX 3 — Wake fires during Ollama inference
#     Root cause: listen_wake() inside wake_loop() picks up
#     the TTS "I'm here and listening" spoken by Arju while
#     Ollama is still computing, then fires another wake.
#     Fix: wake_loop() now checks is_busy() BEFORE listen_wake()
#     AND respects the _is_speaking flag — no listening during TTS.
# ============================================================

import queue
import re
import threading
import time
import warnings

import pyttsx3
import speech_recognition as sr

import config

warnings.filterwarnings("ignore")

# Hard cap — anything above this makes the mic effectively deaf
_ENERGY_MAX = 1500
_ENERGY_MIN = 300

_REJECT = [r"[^\x00-\x7F]", r"^\s*$"]


def _validate(text: str) -> str | None:
    if not text or len(text.strip()) < 2:
        return None
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        print(f"[Voice] Rejected non-English: '{text[:30]}'")
        return None
    for pat in _REJECT:
        if re.search(pat, text.strip()):
            return None
    return text.lower().strip()


class VoiceEngine:

    def __init__(self):
        self._q           = queue.Queue()
        self._tts_lock    = threading.Lock()
        self._is_speaking = threading.Event()
        self._busy        = threading.Event()

        self._init_tts()
        self._init_stt()
        threading.Thread(target=self._worker, daemon=True, name="TTS").start()

    # ── TTS ──────────────────────────────────────────────────

    def _init_tts(self):
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate",   config.TTS_RATE)
            self.tts.setProperty("volume", config.TTS_VOLUME)
            voices = self.tts.getProperty("voices")
            if voices:
                self.tts.setProperty(
                    "voice", voices[min(config.TTS_VOICE_INDEX, len(voices)-1)].id
                )
            print("[Voice] TTS ready.")
        except Exception as e:
            print(f"[Voice] TTS error: {e}")
            self.tts = None

    def _worker(self):
        while True:
            text = self._q.get()
            if text is None:
                break
            self._speak_blocking(text)
            self._q.task_done()

    def _speak_blocking(self, text: str):
        if not self.tts:
            return
        self._is_speaking.set()
        try:
            with self._tts_lock:
                self.tts.say(text)
                self.tts.runAndWait()
        except Exception as e:
            print(f"[Voice] TTS runtime: {e}")
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate",   config.TTS_RATE)
                self.tts.setProperty("volume", config.TTS_VOLUME)
            except Exception:
                pass
        finally:
            self._is_speaking.clear()

    def speak(self, text: str, log: bool = True):
        """Non-blocking — queued."""
        if log:
            print(f"\n[{config.ASSISTANT_NAME}] {text}")
        self._q.put(text)

    def speak_now(self, text: str, log: bool = True):
        """Blocking — speaks immediately, waits until done."""
        if log:
            print(f"\n[{config.ASSISTANT_NAME}] {text}")
        self._speak_blocking(text)

    def _wait_for_tts(self):
        """Wait until TTS finishes + silence gap before opening mic."""
        if self._is_speaking.is_set():
            self._is_speaking.wait(timeout=30)
        time.sleep(config.TTS_POST_DELAY)

    # ── STT init ─────────────────────────────────────────────

    def _init_stt(self):
        try:
            self.rec = sr.Recognizer()
            self.rec.dynamic_energy_threshold = False
            self.rec.energy_threshold         = config.SR_ENERGY
            self.rec.pause_threshold          = 0.9
            self.rec.non_speaking_duration    = 0.4
            self.mic = sr.Microphone()

            print("[Voice] Calibrating mic (stay quiet 2 s)...")
            with self.mic as src:
                self.rec.adjust_for_ambient_noise(src, duration=2.0)
                self.rec.dynamic_energy_threshold = False

                # ── FIX 1: Hard cap — prevents deafness in noisy rooms ──
                calibrated = self.rec.energy_threshold
                capped     = max(_ENERGY_MIN, min(calibrated, _ENERGY_MAX))
                self.rec.energy_threshold = capped

                if calibrated > _ENERGY_MAX:
                    print(f"[Voice] Ambient noise high ({int(calibrated)}) "
                          f"— capped to {int(capped)}.")

            print(f"[Voice] STT ready | threshold={int(self.rec.energy_threshold)} "
                  f"| lang={config.SR_LANGUAGE}")
        except Exception as e:
            print(f"[Voice] STT error: {e}")
            self.rec = None
            self.mic = None

    # ── STT core ─────────────────────────────────────────────

    def _google(self, audio) -> str | None:
        try:
            raw = self.rec.recognize_google(audio, language=config.SR_LANGUAGE)
            return _validate(raw)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[Voice] Google STT error: {e}")
            return None

    # ── Wake word listener ────────────────────────────────────

    def listen_wake(self) -> str | None:
        """
        Passive always-on listener.
        Does NOT open mic while TTS is playing — prevents echo activation.
        """
        if not self.rec or not self.mic:
            return None

        # Don't listen while Arju is speaking (TTS echo would trigger wake)
        if self._is_speaking.is_set():
            self._is_speaking.wait(timeout=30)
            time.sleep(0.3)   # brief gap after TTS

        try:
            with self.mic as src:
                audio = self.rec.listen(
                    src,
                    timeout           = config.SR_WAKE_TIMEOUT,
                    phrase_time_limit = config.SR_WAKE_PHRASE_LIMIT,
                )
            result = self._google(audio)
            if result:
                print(f"[Wake] Heard: '{result}'")
            return result
        except sr.WaitTimeoutError:
            return None
        except Exception:
            time.sleep(0.2)
            return None

    def contains_wake_word(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(a in t for a in config.WAKE_WORD_ALIASES)

    def extract_inline_command(self, phrase: str) -> str | None:
        """
        Strip ALL wake word aliases from the phrase (not just from start).
        'arju can you hear me arjun' → 'can you hear me'
        'arjun describe the scene'   → 'describe the scene'
        'arju'                       → None
        """
        t = phrase.lower().strip()

        # Remove every alias occurrence (sort longest-first to avoid partial matches)
        for alias in sorted(config.WAKE_WORD_ALIASES, key=len, reverse=True):
            # Replace whole-word occurrences only
            t = re.sub(r'\b' + re.escape(alias) + r'\b', '', t)

        # Clean up extra spaces and punctuation
        cmd = re.sub(r'\s+', ' ', t).strip(" ,.")
        return cmd if len(cmd) >= 2 else None

    # ── Command listener ──────────────────────────────────────

    def listen_command(self, timeout=None, phrase_limit=None, retries=None) -> str | None:
        if not self.rec or not self.mic:
            return None

        timeout      = timeout      or config.SR_TIMEOUT
        phrase_limit = phrase_limit or config.SR_PHRASE_LIMIT
        max_att      = (retries if retries is not None else config.SR_RETRIES) + 1

        self._wait_for_tts()

        for attempt in range(1, max_att + 1):
            print(f"[Voice] Listening... ({attempt}/{max_att})")
            try:
                with self.mic as src:
                    audio = self.rec.listen(
                        src, timeout=timeout, phrase_time_limit=phrase_limit
                    )
                result = self._google(audio)
                if result:
                    print(f"[Voice] Got: '{result}'")
                    return result
                if attempt < max_att:
                    print("[Voice] Not clear — retrying...")
                    time.sleep(0.2)
            except sr.WaitTimeoutError:
                print("[Voice] No speech.")
                return None
            except Exception as e:
                print(f"[Voice] Error: {e}")
                return None
        return None

    def listen(self, timeout=None, phrase_limit=None, retries=None):
        return self.listen_command(timeout, phrase_limit, retries)

    # ── Busy lock ─────────────────────────────────────────────
    def set_busy(self):   self._busy.set()
    def clear_busy(self): self._busy.clear()
    def is_busy(self):    return self._busy.is_set()

    def get_command(self) -> str | None:
        """Short confirmation then listen for command."""
        print("[Voice] Activated — listening for command...")
        self.speak_now("Ready.", log=False)
        return self.listen_command()

    def ask_user(self, prompt: str) -> str | None:
        self.speak_now(prompt)
        return self.listen_command()

    def shutdown(self):
        self._q.put(None)
