# ============================================================
#   ARJU COMMANDER — modules/thinking_module.py
#
#   ROOT CAUSE OF SLOWNESS / SILENCE
#   ──────────────────────────────────
#   confirm_intent() was calling:
#     1. ai.interpret_intent()      → full Ollama call (~30s)
#     2. ai.generate_clarification() → another Ollama call (~30s)
#   Total: 60s of silence just to ask "did you mean X?"
#   During that time [Arju] printed blank, no voice, no response.
#
#   THE FIX
#   ────────
#   Ambiguity handling is now 100% LOCAL — zero Ollama calls.
#   • Rule-based word matching decides if unclear
#   • Instant "Please say that again clearly" response
#   • Ollama is ONLY used for actual answers, not meta-questions
#
#   Self-correction (when Vishnu says "that's wrong") still
#   calls Ollama ONCE to generate a natural apology, but that's
#   intentional and user-triggered, not automatic.
# ============================================================

import re
import config

# ── Always-clear single words (never ambiguous) ──────────────
_CLEAR = {
    "stop","exit","quit","goodbye","bye","close",
    "hello","hi","hey","yes","yeah","yep","yup","no","nope","okay","ok",
    "mute","unmute","screenshot","restart","lock","unlock",
    "joke","motivate","help","time","date",
    "describe","caption","emotion","activity","count","identify",
    "open","search","play","find","read","detect","monitor",
    # partial commands that are still meaningful alone
    "chrome","firefox","edge","notepad","calculator","paint","spotify",
    "volume","brightness",
}

# ── Filler / noise words that mean nothing alone ─────────────
_NOISE = {
    "um","uh","hmm","hm","er","ah","uhh","umm",
    "maybe","something","idk","dunno",
}

# ── Correction signals from Vishnu ───────────────────────────
_CORRECTIONS = [
    "that's wrong","that is wrong","wrong answer","not right",
    "incorrect","you're wrong","you are wrong","not what i said",
    "i didn't say that","you misunderstood","correct yourself",
    "you got it wrong","no no no","that's not right",
    "i meant","i said","i asked for",
]

# ── Yes / No ─────────────────────────────────────────────────
_YES = {"yes","yeah","yep","yup","correct","right","exactly",
        "sure","ok","okay","proceed","go ahead","do it","affirmative"}
_NO  = {"no","nope","nah","wrong","incorrect","negative","dont","stop"}


class ThinkingModule:
    """
    Arju's reasoning brain — all ambiguity decisions are instant,
    zero Ollama calls in the normal flow.
    """

    def __init__(self, ollama_engine, rag_memory, voice_engine):
        self.ai    = ollama_engine
        self.rag   = rag_memory
        self.voice = voice_engine
        self._last_cmd    = ""
        self._last_action = ""

    # ── Classification ────────────────────────────────────────

    def is_ambiguous(self, text: str) -> bool:
        """
        True only if the input is genuine gibberish/noise.
        Most single-word commands are NOT ambiguous.
        """
        if not text:
            return True
        t     = text.lower().strip()
        words = [w for w in t.split() if w]

        # Known clear single word → unambiguous
        if len(words) == 1 and words[0] in _CLEAR:
            return False

        # Pure noise word → ambiguous
        if len(words) == 1 and words[0] in _NOISE:
            return True

        # Very short, unknown single word that isn't a command
        if len(words) == 1 and len(words[0]) < 3:
            return True

        # Multiple words → always attempt to route (not ambiguous)
        # The command router will handle it or fall through to AI
        if len(words) >= 2:
            return False

        return False

    def is_correction(self, text: str) -> bool:
        t = text.lower().strip()
        return any(sig in t for sig in _CORRECTIONS)

    def is_yes(self, text: str) -> bool:
        t = text.lower().strip()
        return t in _YES or any(t.startswith(w + " ") for w in _YES)

    def is_no(self, text: str) -> bool:
        t = text.lower().strip()
        return t in _NO or any(t.startswith(w + " ") for w in _NO)

    # ── Ambiguity handling — INSTANT, no Ollama ───────────────

    def handle_ambiguous(self, raw_text: str) -> str | None:
        """
        Handle unclear input WITHOUT calling Ollama.
        Simply asks Vishnu to repeat clearly, then returns what's heard.
        Returns the repeated command, or None.
        """
        self.voice.speak_now(
            f"Sorry {config.BOSS_NAME}, I didn't catch that clearly. "
            "Please say it again."
        )
        repeated = self.voice.listen_command(timeout=8, retries=2)
        if not repeated:
            self.voice.speak_now(
                f"I still couldn't hear you, {config.BOSS_NAME}. "
                "Please try again."
            )
            return None
        return repeated

    # ── Correction flow — ONE Ollama call (user-triggered) ────

    def handle_correction(self, correction_text: str) -> str:
        """
        Vishnu says "that's wrong" — ask what was correct, store it.
        ONE Ollama call to generate a natural apology.
        """
        self.voice.speak_now(
            f"I'm sorry {config.BOSS_NAME}. "
            "What should the correct answer have been?"
        )
        correct = self.voice.listen_command(timeout=10, retries=2)
        if not correct:
            return (
                f"I'm sorry {config.BOSS_NAME}, I made a mistake. "
                "I'll do better next time."
            )

        self.rag.add_correction(
            wrong   = self._last_action or "my last response",
            correct = correct
        )

        # ONE Ollama call — user deliberately triggered this
        try:
            ack = self.ai.acknowledge_correction(
                was_wrong = self._last_action or "my last response",
                correct   = correct
            )
        except Exception:
            ack = (f"Understood, {config.BOSS_NAME}. "
                   f"The correct answer is '{correct}'. I've stored that.")

        print(f"[Think] Correction stored → '{correct}'")
        return ack

    def set_last_action(self, cmd: str, action: str):
        self._last_cmd    = cmd
        self._last_action = action
