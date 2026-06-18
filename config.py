# ============================================================
#   ARJU VOICE COMMANDER — config.py
#   Assistant : Arju
#   Boss      : Vishnu
#   Engine    : Ollama  qwen3-vl:4b  (local, no API key)
# ============================================================

BOSS_NAME        = "Vishnu"
ASSISTANT_NAME   = "Arju"
WAKE_WORD        = "arju"

# All aliases Google STT may produce when Vishnu says "Arju"
WAKE_WORD_ALIASES = [
    "arju", "arjun", "arjo", "arjoo", "aro",
    "orju", "r2", "ar ju", "raju", "arjuna",
]

# ── Ollama (qwen3-vl:4b) ─────────────────────────────────────
OLLAMA_HOST      = "http://localhost:11434"
OLLAMA_MODEL     = "qwen3-vl:4b"
OLLAMA_TIMEOUT   = 120          # seconds per request
OLLAMA_MAX_TOKENS= 512          # max tokens in any response
# Thinking mode — qwen3 emits <think>...</think> blocks.
# We parse them internally for reasoning but strip them from voice.
OLLAMA_THINK     = True

# ── RAG (memory) ─────────────────────────────────────────────
RAG_DIR          = "memory"             # folder for persistent storage
RAG_TOP_K        = 4                    # how many memories to retrieve
RAG_SIMILARITY   = 0.35                 # minimum similarity score (0-1)
RAG_MAX_MEMORIES = 2000                 # cap stored memories

# ── Thinking / Confidence ─────────────────────────────────────
# If ASR text contains these signals, Arju will ask to confirm
# its interpretation before acting.
AMBIGUOUS_SIGNALS = [
    "maybe", "something", "um", "uh", "hm",
    "kind of", "sort of", "i think",
]
# Minimum word count to process a command without asking
MIN_COMMAND_WORDS = 2
# Arju guesses intent and asks "Did you mean X?" — max guesses
MAX_CONFIRM_TRIES = 2

# ── TTS ──────────────────────────────────────────────────────
TTS_RATE         = 165
TTS_VOLUME       = 1.0
TTS_VOICE_INDEX  = 0

# ── STT ──────────────────────────────────────────────────────
SR_LANGUAGE      = "en-IN"      # Indian English — locked
SR_ENERGY        = 600          # fixed mic threshold
SR_TIMEOUT       = 6
SR_PHRASE_LIMIT  = 12
SR_RETRIES       = 3

SR_WAKE_TIMEOUT      = 6
SR_WAKE_PHRASE_LIMIT = 6

# ── TTS → Mic gap ─────────────────────────────────────────────
TTS_POST_DELAY   = 0.8

# ── Camera ───────────────────────────────────────────────────
CAM_INDEX        = 0
CAM_W            = 640
CAM_H            = 480
CAM_WARMUP       = 1.2
CAM_CAPTURE_PATH = "memory/last_capture.jpg"   # reused across queries

# ── Gesture ──────────────────────────────────────────────────
GESTURE_VOL_STEP = 5

# ── Misc ─────────────────────────────────────────────────────
SCREENSHOT_DIR   = "screenshots"

# Terminal output voice/logging
TERMINAL_LOG_DIR          = "terminal_logs"
TERMINAL_COMMAND_TIMEOUT  = 30
TERMINAL_VOICE_MAX_CHARS  = 6000
TERMINAL_VOICE_CHUNK_CHARS= 700

# Video understanding
VIDEO_FRAME_DIR   = "memory/video_frames"
VIDEO_MAX_FRAMES  = 6

# Hands-free chat mode
CONTINUOUS_CHAT_MAX_TURNS = 20

# FastAPI service
API_HOST = "127.0.0.1"
API_PORT = 8000
API_CORS_ORIGINS = ["*"]
API_UPLOAD_DIR = "memory/api_uploads"
