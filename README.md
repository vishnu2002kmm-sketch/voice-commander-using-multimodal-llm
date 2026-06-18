# ARJU VOICE COMMANDER

Boss: Vishnu | Assistant: Arju | Engine: local Ollama VLM (`qwen3-vl:4b`)

Arju is a local multimodal voice assistant with speech input, spoken replies,
camera-based vision, persistent RAG memory, system controls, gesture controls,
terminal-output narration, media playback helpers, and local video explanation.

## Current Project Capabilities

| Feature | What it does |
|---|---|
| Voice-first assistant | Wake word, speech recognition, and pyttsx3 spoken replies. |
| Streaming AI replies | Ollama responses are spoken sentence-by-sentence as tokens arrive. |
| Vision understanding | Camera captures are sent to the VLM for object, scene, OCR, emotion, activity, and presence questions. |
| VLM terminal logs | Every VLM result is printed and saved in `terminal_logs/vlm_result_*.txt`. |
| RAG memory | ChromaDB stores conversations, preferences, corrections, and visual observations. |
| Terminal narration | `terminal ...` commands capture stdout/stderr, save logs, and read output aloud. |
| Video explanation | Local video files are sampled into frames and explained by the VLM in voice. |
| Media playback | Play music/video on YouTube, Spotify, SoundCloud, or search a named site. |
| Continuous chat | Hands-free voice chat mode keeps listening until you say `stop chat`. |
| Gesture control | MediaPipe hand gestures can control volume and mute. |

## Commands

### Voice and Chat

```text
Arju hello
Arju continuous chat
stop chat
Arju reset conversation
Arju goodbye
```

### Terminal Output in Voice

```text
Arju terminal Get-ChildItem
Arju powershell command Get-Date
Arju read terminal output
```

The complete captured output is printed in the terminal and saved to:

```text
terminal_logs/terminal_*.txt
terminal_logs/last_terminal_output.txt
```

Voice reading is capped by `TERMINAL_VOICE_MAX_CHARS` in `config.py` so one huge command cannot lock TTS for too long.

### Vision and Current Situation

```text
Arju what is this
Arju describe the scene
Arju what is happening now
Arju describe motion
Arju read text
Arju count objects
Arju detect my emotion
Arju vision chat
end vision
```

Every vision answer is printed and saved to:

```text
terminal_logs/vlm_result_*.txt
```

### Local Video Explanation

```text
Arju describe video E:\Videos\demo.mp4
Arju explain video C:\Users\Vishnu\Videos\clip.mp4 in detail
```

Arju samples frames into `memory/video_frames/`, sends those frames to the VLM in chronological order, speaks a detailed explanation, stores the result in RAG, and saves the VLM transcript under `terminal_logs/`.

### Music, Video, Websites, and Apps

```text
Arju play lo-fi music on YouTube
Arju play shape of you on Spotify
Arju play relaxing music on SoundCloud
Arju open website example.com
Arju open Chrome
Arju search multimodal AI on Google
```

### Memory

```text
Arju remember that I prefer dark mode
Arju how many memories do you have
Arju clear memory
```

## Setup

1. Install Ollama from https://ollama.com.
2. Pull the local model:

```bash
ollama pull qwen3-vl:4b
ollama serve
```

3. Install Python packages:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python main.py
```

### Run the FastAPI VLM + RAG Server

The API reuses the same local Ollama model, Chroma RAG memory, and Arju
self-thinking flow.

```bash
python api_server.py
```

Open the web UI:

```text
http://127.0.0.1:8000
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check Ollama readiness, model name, and memory count. |
| `POST /chat` | Text chat with autonomous thinking and RAG context. |
| `POST /chat/stream` | Server-sent streaming chat chunks. |
| `POST /vision` | Ask the VLM about a camera capture, image path, or base64 image. |
| `POST /memory` | Add a memory manually. |
| `POST /memory/search` | Retrieve relevant RAG memory for a query. |
| `DELETE /memory` | Clear stored memory. |
| `WS /ws/chat` | WebSocket streaming chat. |

Example chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Arju, what do you remember about me?\"}"
```

Example VLM request using the webcam:

```bash
curl -X POST http://127.0.0.1:8000/vision ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What can you see?\",\"capture_camera\":true}"
```

## Git Setup

This folder is not a Git repository by default. To publish it:

```bash
git init
git add .
git commit -m "Add Arju VLM RAG FastAPI web app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

If the remote already has code:

```bash
git remote add origin <your-github-repo-url>
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Project Structure

```text
ArjuCommander/
|-- main.py
|-- config.py
|-- requirements.txt
|-- memory/
|-- screenshots/
|-- terminal_logs/
`-- modules/
    |-- command_processor.py
    |-- gesture_module.py
    |-- ollama_engine.py
    |-- rag_module.py
    |-- system_commands.py
    |-- terminal_module.py
    |-- thinking_module.py
    |-- vision_module.py
    `-- voice_module.py
```

## Architecture Notes

The current repository is a local Python voice-assistant build. The larger full-stack direction can be added as separate services:

| Planned/Portfolio Area | Suggested implementation path |
|---|---|
| FastAPI + React frontend | Add an async FastAPI server with WebSocket streaming and a React client for text, image, and voice sessions. |
| InternVL on HuggingFace | Add an optional VLM backend adapter beside the current Ollama VLM adapter. |
| FAISS + Chroma dual-vector RAG | Keep Chroma for persistence and add FAISS as a hot in-memory ANN index. |
| WebRTC + LiveKit + Twilio | Add a real-time media service for browser voice and PSTN telephony access. |
| Cross-modal context fusion | Persist transcripts, image summaries, video summaries, and retrieved documents into a unified session context. |

## Troubleshooting

| Problem | Fix |
|---|---|
| Ollama not ready | Run `ollama serve` in another terminal. |
| Model not found | Run `ollama pull qwen3-vl:4b`. |
| No voice output | Try a different `TTS_VOICE_INDEX` in `config.py`. |
| Wake word not heard | Try aliases: `arjun`, `arjo`, `raju`, `orju`. |
| Camera not found | Change `CAM_INDEX` in `config.py`. |
| Long terminal output | Increase or reduce `TERMINAL_VOICE_MAX_CHARS` in `config.py`. |
