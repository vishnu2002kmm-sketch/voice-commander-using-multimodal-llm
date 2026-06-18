# ============================================================
#   ARJU COMMANDER — modules/command_processor.py
#
#   KEY FIXES
#   ──────────
#   FIX 1 — "goodbye" → goes to ambiguity check (WRONG)
#     Root cause: exit check was AFTER ambiguity check.
#     "goodbye" is 1 word → is_ambiguous() = True → think flow.
#     Fix: EXIT CHECK IS NOW FIRST — before everything else.
#
#   FIX 2 — AI responses not spoken in voice
#     Root cause: command_processor called think_and_reply()
#     which waits 30-60s for full response THEN returns a string.
#     The string was passed to say() but TTS had long silence.
#     Fix: ALL AI/vision calls now use ai.speak_stream() which
#     feeds each sentence to voice.speak_now() as tokens arrive.
#     Arju starts talking within 1-2 seconds.
#
#   FIX 3 — Vision responses also now streamed
#     vision_module now calls ai.speak_stream_vision() instead
#     of returning a string.
# ============================================================

import re
import config
from modules import system_commands as sc
from modules.terminal_module import TerminalNarrator

_WAKE_ALIASES = config.WAKE_WORD_ALIASES

# Exit words — must be checked FIRST before ANY other logic
_EXIT_WORDS = {
    "stop", "exit", "quit", "goodbye", "bye", "close",
    "shut down arju", "shutdown arju", "turn off"
}


def _strip_wake_anywhere(text: str, preserve_case: bool = False) -> str:
    """Remove ALL wake word aliases from anywhere in the text."""
    t = text.strip() if preserve_case else text.lower().strip()
    for alias in sorted(_WAKE_ALIASES, key=len, reverse=True):
        flags = re.IGNORECASE if preserve_case else 0
        t = re.sub(r'\b' + re.escape(alias) + r'\b', '', t, flags=flags)
    return re.sub(r'\s+', ' ', t).strip(" ,.")


class CommandProcessor:

    def __init__(self, voice, vision, ai, rag, thinker, gesture, terminal=None):
        self.voice   = voice
        self.vision  = vision
        self.ai      = ai
        self.rag     = rag
        self.thinker = thinker
        self.gesture = gesture
        self.terminal = terminal or TerminalNarrator(voice)

        self._in_vchat   = False
        self._vchat_img  = None
        self._vchat_ctx  = ""
        self._last_cmd_original = ""
        self._last_cmd_lower = ""

    def _source_cmd(self, cmd: str) -> str:
        if getattr(self, "_last_cmd_lower", "") == cmd:
            return getattr(self, "_last_cmd_original", cmd)
        return cmd

    def say(self, text: str):
        """Speak text aloud and print it."""
        print(f"\n[{config.ASSISTANT_NAME}] {text}")
        self.voice.speak_now(text, log=False)

    def stream(self, question: str, context: str = "",
               image_path: str = None) -> str:
        """
        Stream Ollama response to voice in real time.
        Arju speaks each sentence the moment it's generated.
        Returns full answer for RAG storage.
        """
        return self.ai.speak_stream(
            user_text    = question,
            voice_engine = self.voice,
            image_path   = image_path,
            context      = context,
        )

    # ═════════════════════════════════════════════════════════
    # Entry point
    # ═════════════════════════════════════════════════════════

    def process(self, raw: str) -> bool:
        if not raw or not raw.strip():
            return True

        # Strip wake word from anywhere in input
        original_cmd = _strip_wake_anywhere(raw, preserve_case=True)
        cmd = original_cmd.lower()
        self._last_cmd_original = original_cmd
        self._last_cmd_lower = cmd

        if not cmd:
            self.say(
                f"Yes {config.BOSS_NAME}, I'm listening. What would you like?"
            )
            return True

        print(f"\n[CMD] '{cmd}'")

        # ── Vision conversation mode ──────────────────────────
        if self._in_vchat:
            return self._vchat_turn(cmd)

        # ══════════════════════════════════════════════════════
        # STEP 1 — EXIT CHECK FIRST (before ambiguity check!)
        # "goodbye" is 1 word and must exit, not trigger confirm
        # ══════════════════════════════════════════════════════
        if any(w in cmd for w in _EXIT_WORDS) or cmd in _EXIT_WORDS:
            self.say(f"Goodbye {config.BOSS_NAME}! See you soon.")
            return False

        # ── Correction detection ──────────────────────────────
        if self.thinker.is_correction(cmd):
            ack = self.thinker.handle_correction(cmd)
            self.say(ack)
            self.rag.add_qa(f"[correction] {cmd}", ack)
            return True

        # ── Ambiguity check — INSTANT, zero Ollama calls ──────
        if self.thinker.is_ambiguous(cmd):
            print(f"[Think] Unclear input: '{cmd}' — asking to repeat")
            repeated = self.thinker.handle_ambiguous(cmd)
            if not repeated:
                return True
            # Strip wake word from repeated input too
            original_cmd = _strip_wake_anywhere(repeated, preserve_case=True)
            cmd = original_cmd.lower()
            self._last_cmd_original = original_cmd
            self._last_cmd_lower = cmd
            if not cmd:
                return True
            print(f"[Think] Repeated: '{cmd}'")

        # ── RAG context ───────────────────────────────────────
        context = self.rag.retrieve(cmd)

        # ── Route ─────────────────────────────────────────────
        return self._route(cmd, context)

    # ═════════════════════════════════════════════════════════
    # Routing — returns bool (keep_going)
    # ═════════════════════════════════════════════════════════

    def _route(self, cmd: str, context: str) -> bool:

        # ── Time / Date (instant) ──────────────────────────────
        if any(p in cmd for p in ["what time", "current time",
                                   "time is it", "the time"]):
            self.say(sc.get_time()); return True
        if any(p in cmd for p in ["what date", "today's date",
                                   "what day", "current date",
                                   "date is it"]):
            self.say(sc.get_date()); return True

        # ── Greeting ──────────────────────────────────────────
        if any(p in cmd for p in ["hello", "hi", "hey", "how are you",
                                   "can you hear me", "are you there",
                                   "are you listening"]):
            self.say(f"Yes {config.BOSS_NAME}, I can hear you clearly! "
                     "How can I help?")
            return True

        if any(p in cmd for p in [
            "read terminal output", "speak terminal output",
            "voice terminal output", "tell terminal output",
        ]):
            output = self.terminal.read_last()
            if not output:
                self.say("There is no saved terminal output yet.")
                return True
            self.terminal.narrate(output, intro="Reading the last terminal output.")
            return True

        source_cmd = self._source_cmd(cmd)
        if m := re.search(
            r"^(?:run\s+)?(?:terminal|shell|powershell)(?:\s+command)?\s+(.+)$",
            source_cmd,
            flags=re.IGNORECASE,
        ):
            command = m.group(1).strip()
            result = self.terminal.run(command)
            self.terminal.narrate_result(result)
            return True

        if any(p in cmd for p in [
            "continuous chat", "hands free chat", "start chat mode",
            "voice chat mode", "continue chat",
        ]):
            return self._continuous_chat(context)

        # ── Apps ──────────────────────────────────────────────
        if m := re.search(r"\bopen\s+(.+)", cmd):
            app = m.group(1).strip()
            if app.startswith(("website ", "site ", "url ")):
                target = re.sub(r"^(website|site|url)\s+", "", app).strip()
                self.say(sc.open_url(target))
            elif "youtube"  in app: self.say(sc.open_youtube())
            elif "gmail"    in app: self.say(sc.open_gmail())
            elif "whatsapp" in app: self.say(sc.open_whatsapp_web())
            else:                   self.say(sc.open_app(app))
            return True

        # ── Web ───────────────────────────────────────────────
        if m := re.search(r"\bsearch\s+(.+?)(?:\s+on\s+google)?$", cmd):
            self.say(sc.search_google(m.group(1).strip())); return True
        if m := re.search(r"\bplay\s+(.+?)\s+on\s+([a-z0-9 ._-]+)$", cmd):
            self.say(sc.play_on_site(m.group(1).strip(), m.group(2).strip()))
            return True
        if m := re.search(r"\bplay\s+(?:video|music|song)?\s*(.+)$", cmd):
            media = m.group(1).strip()
            if media:
                self.say(sc.play_on_site(media, "youtube"))
                return True

        # ── Volume ────────────────────────────────────────────
        if re.search(r"(increase|up|louder|raise).{0,10}volume|volume\s+up", cmd):
            self.say(sc.increase_volume()); return True
        if re.search(r"(decrease|down|lower|quieter|reduce).{0,10}volume|volume\s+down", cmd):
            self.say(sc.decrease_volume()); return True
        if re.search(r"\bmute\b", cmd) and "unmute" not in cmd:
            self.say(sc.mute_volume()); return True
        if "unmute" in cmd:
            self.say(sc.unmute_volume()); return True
        if m := re.search(r"set\s+volume\s+(?:to\s+)?(\d+)", cmd):
            self.say(sc.set_volume(int(m.group(1)))); return True

        # ── Brightness ────────────────────────────────────────
        if re.search(r"(increase|up|raise).{0,10}brightness", cmd):
            self.say(sc.increase_brightness()); return True
        if re.search(r"(decrease|down|lower|reduce).{0,10}brightness", cmd):
            self.say(sc.decrease_brightness()); return True

        # ── Screenshot ────────────────────────────────────────
        if "screenshot" in cmd:
            self.say(sc.screenshot()); return True

        # ── WhatsApp ─────────────────────────────────────────
        if m := re.search(
            r"send\s+whatsapp(?:\s+message)?\s+to\s+(.+?)[:;,]\s*(.+)", cmd
        ):
            self.say(sc.send_whatsapp(m.group(1).strip(),
                                       m.group(2).strip())); return True

        # ── Power ─────────────────────────────────────────────
        if "shut down" in cmd:
            self.say(sc.shutdown()); return True
        if "restart" in cmd or "reboot" in cmd:
            self.say(sc.restart()); return True
        if "lock" in cmd:
            self.say(sc.lock()); return True

        # ── Memory ops ───────────────────────────────────────
        if "clear memory" in cmd or "forget everything" in cmd:
            self.rag.clear_all()
            self.say(f"Memory cleared, {config.BOSS_NAME}.")
            return True
        if any(p in cmd for p in ["how many memories", "memory count",
                                   "what do you remember"]):
            self.say(f"I have {self.rag.count()} memories stored, "
                     f"{config.BOSS_NAME}.")
            return True
        if "reset conversation" in cmd or "clear history" in cmd:
            self.ai.clear_history()
            self.say("Conversation history reset."); return True
        if m := re.search(r"remember that (.+)", cmd):
            self.rag.add_preference(m.group(1).strip())
            self.say(f"Got it, I'll remember that, {config.BOSS_NAME}.")
            return True

        # ── Gesture ──────────────────────────────────────────
        if "start gesture" in cmd or "gesture control" in cmd:
            self.say(self.gesture.start()); return True
        if "stop gesture" in cmd:
            self.say(self.gesture.stop()); return True

        # ── Jokes / Motivation (streamed for natural pacing) ──
        if "joke" in cmd:
            self.say("Here is a joke for you.")
            answer = self.stream("Tell me one short, funny, clean joke. Two sentences max.")
            self.rag.add_qa(cmd, answer)
            return True
        if any(p in cmd for p in ["motivate", "motivation", "inspire"]):
            answer = self.stream(
                f"Give {config.BOSS_NAME} a powerful one-sentence motivation."
            )
            self.rag.add_qa(cmd, answer)
            return True

        if any(p in cmd for p in [
            "explain video", "describe video", "analyse video",
            "analyze video", "video explanation", "what happens in video",
        ]):
            return self._route_video(cmd, context)

        if any(p in cmd for p in [
            "current situation", "what is happening", "what's happening",
            "explain motion", "describe motion", "detect motion",
            "movement", "motion", "live situation",
        ]):
            self.say("Let me check the current situation and motion.")
            return self._vision_stream(
                "Explain the current situation in the camera view. "
                "Describe visible people, objects, actions, motion, and anything important happening now.",
                context
            )

        # ── Vision ────────────────────────────────────────────
        VISION_KW = [
            "what is this", "identify", "what am i", "describe", "caption",
            "look around", "what can you see", "count", "emotion", "mood",
            "how do i look", "my face", "read text", "ocr", "monitor",
            "colour", "color", "can you see", "is there", "do you see",
            "detect", "vision chat", "camera", "activity", "what's this",
            "look at this", "holding", "what do you see", "tell me what",
        ]
        if any(kw in cmd for kw in VISION_KW):
            return self._route_vision(cmd, context)

        # ── General AI conversation — STREAMED ────────────────
        print(f"[CMD] → Ollama AI (streaming)")
        self.say(f"Let me think, {config.BOSS_NAME}...")
        answer = self.stream(cmd, context=context)
        if answer:
            self.thinker.set_last_action(cmd, answer)
            self.rag.add_qa(cmd, answer)
        return True

    # ═════════════════════════════════════════════════════════
    # Vision routing — ALL streamed
    # ═════════════════════════════════════════════════════════

    def _route_vision(self, cmd: str, context: str) -> bool:
        if not self.ai.is_ready:
            self.say("Ollama is not ready. Please run: ollama serve")
            return True

        # ── Identify object ───────────────────────────────────
        if any(p in cmd for p in ["what is this", "identify", "what's this",
                                   "what am i holding", "look at this",
                                   "what is it", "show me", "holding"]):
            self.say("Let me see what that is...")
            return self._vision_stream(
                "What object is being shown to the camera? "
                "Identify it precisely: name, colour, shape, and any visible brand.",
                context
            )

        # ── Activity ──────────────────────────────────────────
        if "what am i doing" in cmd or "activity" in cmd:
            self.say("Let me see what you are doing...")
            return self._vision_stream(
                "What activity or action is the person performing? Be specific.",
                context
            )

        # ── Describe ─────────────────────────────────────────
        if any(p in cmd for p in ["describe", "caption", "what can you see",
                                   "look around", "tell me what you see",
                                   "what do you see"]):
            self.say("Let me describe what I see...")
            return self._vision_stream(
                "Describe this scene in clear, specific detail.",
                context
            )

        # ── Count ─────────────────────────────────────────────
        if "count" in cmd:
            self.say("Let me count everything...")
            return self._vision_stream(
                "List and count every distinct object visible. "
                "Say it like: 1 laptop, 2 books, 1 mug.",
                context
            )

        # ── Emotion ───────────────────────────────────────────
        if any(p in cmd for p in ["emotion", "mood", "how do i look",
                                   "my face", "expression"]):
            self.say("Let me read your expression...")
            return self._vision_stream(
                "What emotion is the person expressing? "
                "Describe the facial cues you see.",
                context
            )

        # ── OCR ───────────────────────────────────────────────
        if any(p in cmd for p in ["read text", "ocr", "read the text"]):
            self.say("Let me read that text...")
            return self._vision_stream(
                "Read and transcribe ALL visible text exactly as written. "
                "If no text is visible, say: No text found.",
                context
            )

        # ── Monitor room ─────────────────────────────────────
        if "monitor" in cmd:
            self.say("Monitoring the room for 30 seconds.")
            import time
            for i in range(1, 7):   # 6 captures × 5s = 30s
                path = self.vision.capture()
                if path:
                    self.say(f"Observation {i}:")
                    answer = self.ai.speak_stream_vision(
                        path, "Briefly describe what is happening now.",
                        self.voice, context=context
                    )
                    if answer:
                        self._log_vlm_result(
                            f"VLM MONITOR OBSERVATION {i}",
                            path,
                            "Briefly describe what is happening now.",
                            answer,
                        )
                    self.rag.add_vision_observation(answer)
                time.sleep(5)
            return True

        # ── Detect specific object ────────────────────────────
        if m := re.search(
            r"(?:detect|find|locate)\s+(?:the\s+|a\s+)?(.+?)(?:\s+in.+)?$", cmd
        ):
            label = m.group(1).strip()
            if label not in {"emotion", "activity", "scene", "text",
                              "room", "camera", "image"}:
                self.say(f"Looking for {label}...")
                return self._vision_stream(
                    f"How many '{label}' objects can you see and where?",
                    context
                )

        # ── Colour ───────────────────────────────────────────
        if m := re.search(
            r"what\s+colou?r\s+is\s+(?:my\s+|the\s+)?(.+?)[\?]?$", cmd
        ):
            self.say("Let me check...")
            return self._vision_stream(
                f"What colour is the {m.group(1).strip()}?",
                context
            )

        # ── Presence ─────────────────────────────────────────
        if m := re.search(
            r"(?:can you see|is there|do you see)\s+(?:a\s+|my\s+|the\s+)?(.+?)[\?]?$",
            cmd
        ):
            return self._vision_stream(
                f"Can you see {m.group(1).strip()} in this image? "
                "If yes, describe where it is.",
                context
            )

        # ── Vision conversation ───────────────────────────────
        if any(p in cmd for p in ["vision chat", "vision conversation",
                                   "camera chat", "start vision"]):
            return self._start_vchat(context)

        # Generic visual question — streamed
        self.say("Let me look...")
        return self._vision_stream(cmd, context)

    def _log_vlm_result(self, title: str, source: str, question: str,
                        answer: str, extra: str = "") -> str:
        body = (
            f"Source: {source}\n"
            f"Question: {question}\n"
            f"{extra.strip()}\n\n"
            f"Answer:\n{answer}"
        ).strip()
        return self.terminal.log_event(
            "vlm_result", title, body, update_last=True
        )

    def _extract_video_path(self, cmd: str) -> str:
        patterns = [
            r"(?:explain|describe|analyse|analyze)\s+video(?:\s+file)?\s+(.+)$",
            r"video\s+(?:explanation\s+)?(?:file\s+)?(.+)$",
        ]
        for pattern in patterns:
            if m := re.search(pattern, cmd, flags=re.IGNORECASE):
                candidate = m.group(1).strip(" .")
                candidate = re.sub(r"^(at|from|path|called)\s+", "", candidate, flags=re.IGNORECASE)
                candidate = re.sub(r"\s+(in detail|detailed|please)$", "", candidate, flags=re.IGNORECASE).strip()
                lower_candidate = candidate.lower()
                if any(mark in lower_candidate for mark in [":", "\\", "/", ".mp4", ".mov", ".avi", ".mkv", ".webm"]):
                    return candidate.strip('"').strip("'")
        return ""

    def _route_video(self, cmd: str, context: str) -> bool:
        if not self.ai.is_ready:
            self.say("Ollama is not ready. Please run: ollama serve")
            return True

        video_path = self._extract_video_path(self._source_cmd(cmd))
        if not video_path:
            self.say(
                "Please include the local video file path after describe video, "
                "for example: describe video C colon backslash videos backslash demo dot mp4."
            )
            return True

        meta = self.vision.sample_video(video_path)
        frames = meta.get("frame_paths", [])
        if meta.get("error"):
            self.say(meta["error"])
            return True
        if not frames:
            self.say("I could not sample frames from that video.")
            return True

        self.say(f"I sampled {len(frames)} frames. I will explain the video in detail.")
        duration = meta.get("duration_seconds", 0.0)
        question = (
            "These images are sampled frames from one video in chronological order. "
            "Explain the video in detail for voice: describe the scene, people, objects, "
            "motion, action changes, current situation, and any important events. "
            "Be specific, but do not invent things that are not visible."
        )
        answer = self.ai.speak_stream_vision(
            frames, question, self.voice, context=context
        )
        if answer:
            extra = (
                f"Video: {meta.get('video_path')}\n"
                f"Duration seconds: {duration:.2f}\n"
                f"FPS: {meta.get('fps', 0.0):.2f}\n"
                f"Frames sampled: {len(frames)}\n"
                f"Frame files: {', '.join(frames)}"
            )
            self._log_vlm_result("VIDEO VLM RESULT", meta.get("video_path", video_path),
                                 question, answer, extra=extra)
            self.rag.add_vision_observation(
                f"VIDEO: {meta.get('video_path', video_path)} | {answer[:220]}"
            )
            self.thinker.set_last_action(cmd, answer)
        return True

    def _continuous_chat(self, context: str) -> bool:
        if not self.voice.rec:
            self.say("Text chat is already continuous. Microphone chat needs speech recognition.")
            return True

        self.say("Continuous chat is on. Speak naturally. Say stop chat to end it.")
        turns = 0
        misses = 0
        max_turns = getattr(config, "CONTINUOUS_CHAT_MAX_TURNS", 20)

        while turns < max_turns:
            heard = self.voice.listen_command(timeout=10, phrase_limit=20, retries=1)
            if not heard:
                misses += 1
                if misses >= 2:
                    self.say("I did not hear anything, so I paused continuous chat.")
                    return True
                self.say("I did not hear that. Please continue.")
                continue

            misses = 0
            original_next = _strip_wake_anywhere(heard, preserve_case=True)
            next_cmd = original_next.lower()
            self._last_cmd_original = original_next
            self._last_cmd_lower = next_cmd
            if not next_cmd:
                continue
            if any(p in next_cmd for p in ["end chat", "stop chat", "exit chat", "pause chat"]):
                self.say("Continuous chat ended.")
                return True
            if any(w in next_cmd for w in _EXIT_WORDS):
                self.say(f"Goodbye {config.BOSS_NAME}! See you soon.")
                return False
            if any(p in next_cmd for p in ["continuous chat", "hands free chat", "start chat mode"]):
                self.say("Continuous chat is already on.")
                continue

            route_context = self.rag.retrieve(next_cmd)
            keep_going = self._route(next_cmd, route_context or context)
            turns += 1
            if not keep_going:
                return False

        self.say("Continuous chat paused after the maximum turns.")
        return True

    def _vision_stream(self, question: str, context: str) -> bool:
        """Capture camera and stream vision answer to voice."""
        path = self.vision.capture()
        if not path:
            self.say("I could not capture a camera image.")
            return True

        answer = self.ai.speak_stream_vision(
            path, question, self.voice, context=context
        )
        if answer:
            self._log_vlm_result("VLM RESULT", path, question, answer)
            self.rag.add_vision_observation(
                f"Q: {question[:60]} | A: {answer[:120]}"
            )
            self.thinker.set_last_action(question, answer)
        return True

    # ═════════════════════════════════════════════════════════
    # Vision conversation
    # ═════════════════════════════════════════════════════════

    def _start_vchat(self, context: str) -> bool:
        first_q = self.voice.ask_user(
            f"I will capture what the camera sees, {config.BOSS_NAME}. "
            "What would you like to ask about the scene?"
        )
        if not first_q:
            self.say("I didn't catch your question.")
            return True

        self.say("Capturing the scene now...")
        path = self.vision.capture(preview=True)
        if not path:
            self.say("Could not capture camera image.")
            return True

        # First answer — streamed
        answer = self.ai.speak_stream_vision(
            path, first_q, self.voice, context=context
        )
        if answer:
            self._log_vlm_result("VLM VISION CHAT START", path, first_q, answer)

        self._vchat_img = path
        self._vchat_ctx = f"Q: {first_q}\nA: {answer}"
        self._in_vchat  = True
        self.say("Keep asking about the scene. Say end vision to stop.")
        return True

    def _vchat_turn(self, cmd: str) -> bool:
        if any(p in cmd for p in ["end vision", "stop vision", "exit vision",
                                   "end chat", "stop chat", "end camera"]):
            self._in_vchat  = False
            self._vchat_img = None
            self._vchat_ctx = ""
            self.say("Vision conversation ended.")
            return True

        if any(w in cmd for w in _EXIT_WORDS):
            self._in_vchat = False
            self.say(f"Goodbye {config.BOSS_NAME}!")
            return False

        context = self.rag.retrieve(cmd)
        full_ctx = (
            f"Conversation so far:\n{self._vchat_ctx}\n\n"
            + (f"Memories:\n{context}" if context else "")
        )

        answer = self.ai.speak_stream_vision(
            self._vchat_img, cmd, self.voice, context=full_ctx
        )
        if answer:
            self._log_vlm_result("VLM VISION CHAT TURN", self._vchat_img, cmd, answer)
        self._vchat_ctx += f"\nQ: {cmd}\nA: {answer}"
        self.rag.add_qa(cmd, answer)
        return True
