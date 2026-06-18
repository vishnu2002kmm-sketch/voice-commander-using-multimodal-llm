# ============================================================
#   ARJU COMMANDER - modules/terminal_module.py
#
#   Captures terminal command output, writes full logs, and can
#   read captured output aloud through the existing voice engine.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import os
import re
import subprocess

import config


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_BLOCKED_PATTERNS = [
    r"\bdel\b",
    r"\berase\b",
    r"\brd\b",
    r"\brmdir\b",
    r"\bremove-item\b",
    r"\brm\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\brestart-computer\b",
    r"\bstop-computer\b",
    r"\btaskkill\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+clean\b",
]


@dataclass
class TerminalResult:
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    log_path: str
    blocked: bool = False
    error: str = ""

    @property
    def output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append("ERROR:\n" + self.stderr.strip())
        if self.error.strip():
            parts.append("ERROR:\n" + self.error.strip())
        return "\n\n".join(parts).strip() or "(no terminal output)"

    @property
    def transcript(self) -> str:
        status = "BLOCKED" if self.blocked else f"EXIT CODE: {self.exit_code}"
        return (
            f"COMMAND: {self.command}\n"
            f"{status}\n"
            f"LOG: {self.log_path}\n\n"
            f"{self.output}\n"
        )


class TerminalNarrator:
    """Run safe terminal commands, persist output, and narrate it by voice."""

    def __init__(self, voice_engine=None):
        self.voice = voice_engine
        self.log_dir = getattr(config, "TERMINAL_LOG_DIR", "terminal_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_log_path = os.path.join(self.log_dir, "last_terminal_output.txt")
        self.last_output = ""

    def _timestamp(self) -> str:
        return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _write_log(self, name: str, text: str, update_last: bool = False) -> str:
        path = os.path.join(self.log_dir, f"{name}_{self._timestamp()}.txt")
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
        if update_last:
            with open(self.last_log_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
            self.last_output = text
        return path

    def _is_blocked(self, command: str) -> str:
        lowered = command.lower()
        for pattern in _BLOCKED_PATTERNS:
            if re.search(pattern, lowered):
                return pattern
        return ""

    def run(self, command: str) -> TerminalResult:
        command = command.strip()
        if not command:
            text = "No terminal command was provided."
            path = self._write_log("terminal", text, update_last=True)
            return TerminalResult(command="", exit_code=None, stdout="", stderr="", log_path=path, error=text)

        blocked = self._is_blocked(command)
        if blocked:
            text = (
                "This terminal command was blocked because it looks destructive.\n"
                f"Blocked pattern: {blocked}\n"
                f"Command: {command}\n"
            )
            path = self._write_log("terminal", text, update_last=True)
            print(f"\n[Terminal]\n{text}")
            return TerminalResult(
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                log_path=path,
                blocked=True,
                error=text,
            )

        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=getattr(config, "TERMINAL_COMMAND_TIMEOUT", 30),
                shell=False,
            )
            result = TerminalResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                log_path="",
            )
        except subprocess.TimeoutExpired as e:
            result = TerminalResult(
                command=command,
                exit_code=None,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                log_path="",
                error="Terminal command timed out.",
            )
        except Exception as e:
            result = TerminalResult(
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                log_path="",
                error=f"Terminal command failed: {e}",
            )

        path = os.path.join(self.log_dir, f"terminal_{self._timestamp()}.txt")
        result.log_path = path
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(result.transcript)
        with open(self.last_log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(result.transcript)
        self.last_output = result.transcript

        print(f"\n[Terminal]\n{result.transcript}")
        print(f"[Terminal] Full output saved: {path}")
        return result

    def log_event(self, kind: str, title: str, body: str, update_last: bool = False) -> str:
        safe_kind = re.sub(r"[^a-zA-Z0-9_-]+", "_", kind).strip("_") or "event"
        text = f"{title}\n{'=' * len(title)}\n{body.strip()}\n"
        path = self._write_log(safe_kind, text, update_last=update_last)
        print(f"\n[{title}]\n{body.strip()}\n[{title}] Saved: {path}")
        return path

    def read_last(self) -> str:
        if self.last_output:
            return self.last_output
        if os.path.exists(self.last_log_path):
            with open(self.last_log_path, "r", encoding="utf-8", errors="replace") as f:
                self.last_output = f.read()
        return self.last_output

    def narrate(self, text: str, intro: str = "Terminal output.") -> None:
        if not self.voice:
            return

        cleaned = _ANSI_RE.sub("", text or "").strip()
        if not cleaned:
            self.voice.speak_now("There is no terminal output to read.")
            return

        limit = getattr(config, "TERMINAL_VOICE_MAX_CHARS", 6000)
        if limit and len(cleaned) > limit:
            cleaned = (
                cleaned[:limit].rstrip()
                + "\nOutput is longer. The full terminal output is printed and saved."
            )

        self.voice.speak_now(intro, log=False)

        chunk_size = getattr(config, "TERMINAL_VOICE_CHUNK_CHARS", 700)
        for chunk in self._chunks(cleaned, chunk_size):
            self.voice.speak_now(chunk, log=False)

    def narrate_result(self, result: TerminalResult) -> None:
        status = "blocked" if result.blocked else f"finished with exit code {result.exit_code}"
        intro = f"Terminal command {status}. I will read the output now."
        self.narrate(result.output, intro=intro)

    def _chunks(self, text: str, chunk_size: int):
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}".strip()
                continue
            if current:
                yield current
            if len(para) <= chunk_size:
                current = para
            else:
                for i in range(0, len(para), chunk_size):
                    yield para[i:i + chunk_size]
                current = ""
        if current:
            yield current
