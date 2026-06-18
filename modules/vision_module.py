# ============================================================
#   ARJU COMMANDER — modules/vision_module.py
#
#   Camera capture + qwen3-vl:4b vision queries.
#   All vision results are spoken aloud AND stored in RAG.
# ============================================================

import os
import time
import threading

import cv2
from PIL import Image

import config


class VisionModule:

    def __init__(self, ollama_engine, rag_memory):
        self.ai  = ollama_engine
        self.rag = rag_memory

        self._cam      = None
        self._cam_lock = threading.Lock()

        os.makedirs(os.path.dirname(config.CAM_CAPTURE_PATH), exist_ok=True)

    # ── Camera ───────────────────────────────────────────────

    def _open_cam(self) -> bool:
        with self._cam_lock:
            if self._cam and self._cam.isOpened():
                return True
            cap = cv2.VideoCapture(config.CAM_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)
            if not cap.isOpened():
                print(f"[Vision] Cannot open camera {config.CAM_INDEX}")
                return False
            self._cam = cap
            return True

    def capture(self, preview: bool = False) -> str | None:
        """Capture frame and return saved path (for command_processor)."""
        return self._capture_path(preview=preview)

    def sample_video(self, video_path: str, max_frames: int = None) -> dict:
        """
        Sample representative frames from a local video file.
        Returns metadata plus saved frame paths for VLM analysis.
        """
        max_frames = max_frames or getattr(config, "VIDEO_MAX_FRAMES", 6)
        raw_path = (video_path or "").strip().strip('"').strip("'")
        raw_path = os.path.expandvars(os.path.expanduser(raw_path))
        if not os.path.isabs(raw_path):
            raw_path = os.path.abspath(raw_path)

        meta = {
            "video_path": raw_path,
            "frame_paths": [],
            "duration_seconds": 0.0,
            "fps": 0.0,
            "frame_count": 0,
            "error": "",
        }

        if not os.path.exists(raw_path):
            meta["error"] = f"Video file not found: {raw_path}"
            print(f"[Vision] {meta['error']}")
            return meta

        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            meta["error"] = f"Could not open video: {raw_path}"
            print(f"[Vision] {meta['error']}")
            return meta

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            duration = (frame_count / fps) if frame_count and fps else 0.0
            meta.update({
                "duration_seconds": duration,
                "fps": fps,
                "frame_count": frame_count,
            })

            out_dir = getattr(config, "VIDEO_FRAME_DIR", "memory/video_frames")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")

            if frame_count > 0:
                count = min(max_frames, frame_count)
                if count == 1:
                    indices = [0]
                else:
                    indices = [
                        int(round(i * (frame_count - 1) / (count - 1)))
                        for i in range(count)
                    ]
            else:
                indices = list(range(max_frames))

            for n, frame_index in enumerate(indices, start=1):
                if frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_name = f"video_{stamp}_{n:02d}.jpg"
                frame_path = os.path.join(out_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                meta["frame_paths"].append(frame_path)

            print(
                f"[Vision] Sampled {len(meta['frame_paths'])} video frames "
                f"from {raw_path}"
            )
            return meta
        finally:
            cap.release()

    def _capture_path(self, preview: bool = False,
                      save_path: str = None) -> str | None:
        """
        Capture one frame, save as JPEG.
        Returns the saved file path, or None on failure.
        """
        if not self._open_cam():
            return None

        save_path = save_path or config.CAM_CAPTURE_PATH

        # Flush warm-up frames
        deadline = time.time() + config.CAM_WARMUP
        with self._cam_lock:
            while time.time() < deadline:
                self._cam.read()
            ok, frame = self._cam.read()

        if not ok:
            print("[Vision] Frame capture failed.")
            return None

        if preview:
            cv2.imshow("Arju captured — auto-closes", frame)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path, frame)
        print(f"[Vision] Frame saved: {save_path}")
        return save_path

    def release(self):
        with self._cam_lock:
            if self._cam:
                self._cam.release()
                self._cam = None

    # ── Vision queries ────────────────────────────────────────

    def ask(self, question: str, context: str = "") -> str:
        """Capture and answer any visual question."""
        path = self.capture()
        if not path:
            return "I could not capture a camera image."

        _, answer = self.ai.vision_query(path, question, context=context)

        # Store observation in RAG
        self.rag.add_vision_observation(f"{question} → {answer}")
        return answer

    def identify_object(self) -> str:
        return self.ask(
            "What object is being held up or shown to the camera? "
            "Identify it precisely: name, colour, shape, size, brand or label if visible."
        )

    def describe_scene(self) -> str:
        return self.ask(
            "Describe this scene in 2-3 clear, specific sentences."
        )

    def detect_activity(self) -> str:
        return self.ask(
            "What is the person in this image doing? "
            "Describe the action or activity specifically."
        )

    def count_objects(self) -> str:
        return self.ask(
            "List every distinct object visible and how many of each. "
            "Be specific: '1 laptop, 2 coffee mugs, 3 books'."
        )

    def detect_emotion(self) -> str:
        return self.ask(
            "What emotion is the person expressing? "
            "Describe the facial cues — eyes, mouth, eyebrows."
        )

    def read_text(self) -> str:
        return self.ask(
            "Read and transcribe ALL visible text exactly as written. "
            "If no text is visible say: No text found."
        )

    def color_of(self, subject: str) -> str:
        return self.ask(f"What colour is the {subject}?")

    def is_present(self, subject: str) -> str:
        return self.ask(
            f"Can you see {subject} in this image? "
            "If yes, describe where it is."
        )

    def detect_objects(self, label: str) -> str:
        return self.ask(
            f"How many '{label}' objects can you see? "
            "Where are they located in the frame?"
        )

    def monitor(self, duration: int = 30, interval: int = 5) -> list[str]:
        results, start, n = [], time.time(), 0
        while time.time() - start < duration:
            n  += 1
            path = self.capture()
            if path:
                _, ans = self.ai.vision_query(
                    path,
                    "Briefly describe what is happening right now."
                )
                results.append(f"[{n}] {ans}")
                self.rag.add_vision_observation(ans)
            time.sleep(interval)
        return results

    # ── Multi-turn vision conversation ────────────────────────

    def start_conversation(self, first_question: str, context: str = ""):
        """
        Capture ONE frame, answer the first question.
        Returns (image_path, conversation_context, answer).
        image_path is reused for follow-ups (same frame).
        """
        print("[Vision] Capturing for conversation...")
        path = self.capture(preview=True)
        if not path:
            return None

        _, answer = self.ai.vision_query(path, first_question, context=context)
        conv_ctx  = f"Q: {first_question}\nA: {answer}"
        return path, conv_ctx, answer

    def continue_conversation(self, follow_up: str, image_path: str,
                              conv_context: str, rag_context: str = "") -> tuple[str, str]:
        """Ask a follow-up about the same captured frame."""
        combined_ctx = f"{rag_context}\n\nConversation so far:\n{conv_context}" \
                       if rag_context else f"Conversation so far:\n{conv_context}"

        _, answer = self.ai.vision_query(image_path, follow_up, context=combined_ctx)
        conv_context = conv_context + f"\nQ: {follow_up}\nA: {answer}"
        return answer, conv_context
