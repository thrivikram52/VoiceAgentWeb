# VoiceAgentWeb — Design Document

## Overview

A real-time browser-based voice agent that enables spoken conversations with a locally-hosted LLM. The user speaks into their browser mic, the server transcribes speech, generates an LLM response, synthesizes speech, and streams audio back — all with interruption (barge-in) support.

Inspired by [Deepgram Agent Playground](https://playground.deepgram.com/?endpoint=agent), but fully local with no cloud API dependencies.

## Architecture

**Pattern**: Monolithic Python server with embedded STT/TTS, external Ollama for LLM.

```
Browser (mic) → WebSocket → [Python Server: VAD → STT → LLM → TTS] → WebSocket → Browser (speaker)
```

### Components

| Component | Technology | Details |
|-----------|-----------|---------|
| Backend | Python, FastAPI, uvicorn | Single process, serves API + static frontend |
| Transport | WebSocket | Binary frames (audio) + JSON text frames (control) |
| STT | pywhispercpp, large-v3-turbo | Metal GPU acceleration |
| VAD | Silero VAD (torch) | Speech boundary detection + barge-in |
| LLM | Ollama (llama3.2:3b) | HTTP streaming via /api/chat |
| TTS | Piper (en_US-lessac-high) | ONNX, CPU-based |
| Frontend | Vanilla HTML/CSS/JS | Minimal UI, no build step |

### System Diagram

```
┌──────────────────────────────────────────────────────┐
│                      Browser                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Mic Input │  │ Audio Player │  │ Chat UI        │  │
│  │getUserMed│  │ PCM playback │  │ Transcript +   │  │
│  │ia()      │  │ via AudioCtx │  │ status         │  │
│  └─────┬────┘  └──────▲───────┘  └────────▲───────┘  │
│        │              │                    │          │
│        ▼              │                    │          │
│       WebSocket (single connection)                   │
└────────┬──────────────┼────────────────────┼──────────┘
         │              │                    │
    PCM chunks     TTS audio         JSON messages
         │              │                    │
┌────────▼──────────────┴────────────────────┴──────────┐
│            Python Server (FastAPI + WebSocket)         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │              Session Manager                     │  │
│  │  States: IDLE → LISTENING → THINKING → SPEAKING  │  │
│  │  Handles interruption (cancel pipeline)          │  │
│  └──────┬──────────────┬──────────────┬────────────┘  │
│         │              │              │               │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────┐   │
│  │ STT         │ │ LLM Client │ │ TTS           │   │
│  │ pywhisper   │ │ Ollama HTTP│ │ Piper         │   │
│  │ cpp + VAD   │ │ streaming  │ │ in-process    │   │
│  └─────────────┘ └────────────┘ └───────────────┘   │
└───────────────────────┬───────────────────────────────┘
                        │ HTTP streaming
                  ┌─────▼─────────┐
                  │ Ollama daemon  │
                  │ llama3.2:3b    │
                  └────────────────┘
```

## Model Paths

All models are stored in shared locations accessible across projects:

| Model | Path | Management |
|-------|------|------------|
| Whisper large-v3-turbo | `~/Library/Application Support/pywhispercpp/models/ggml-large-v3-turbo.bin` | Auto-managed by pywhispercpp |
| Piper en_US-lessac-high | `~/Models/en_US-lessac-high.onnx` | Shared `~/Models/` directory |
| llama3.2:3b | `~/.ollama/models/` | Auto-managed by Ollama |

## Pipeline Data Flow

### Normal Conversation

1. User speaks → Browser streams PCM chunks over WS (binary frames)
2. Server buffers audio → Silero VAD detects speech end
3. STT: whisper.cpp transcribes buffered audio → produces text
4. Server sends JSON: `{ type: "transcript", text: "...", role: "user" }`
5. LLM: Stream tokens from Ollama → accumulate response
6. Server sends JSON: `{ type: "transcript", text: "...", role: "assistant" }` (incremental)
7. TTS: As LLM tokens form sentences, Piper synthesizes → stream PCM back (binary frames)
8. Server sends JSON: `{ type: "state", state: "SPEAKING" }`
9. Browser plays TTS audio through AudioContext
10. When TTS finishes → back to LISTENING

### Interruption (Barge-in)

1. Agent is in SPEAKING state (TTS audio streaming to browser)
2. User starts talking → Browser continues sending audio chunks
3. Server VAD detects voice activity during SPEAKING state
4. Server immediately:
   - Cancels in-flight LLM generation (abort Ollama request)
   - Stops TTS synthesis
   - Sends JSON: `{ type: "interrupt" }` → Browser stops audio playback
   - Transitions to LISTENING state
   - Buffers the new speech for STT
5. Normal flow resumes

## WebSocket Protocol

### Client → Server

- **Binary frames**: Raw PCM audio (16-bit, 16kHz, mono)
- **JSON frames**: `{ type: "config", sampleRate: 16000 }` (sent on connect)

### Server → Client

- **Binary frames**: TTS audio (16-bit, 22050Hz, mono)
- **JSON frames**:

```json
{ "type": "state",      "state": "LISTENING|THINKING|SPEAKING|IDLE" }
{ "type": "transcript", "role": "user",      "text": "...", "final": true }
{ "type": "transcript", "role": "assistant",  "text": "...", "final": false }
{ "type": "transcript", "role": "assistant",  "text": "...", "final": true }
{ "type": "interrupt" }
{ "type": "error",      "message": "..." }
```

## Frontend

Vanilla HTML/CSS/JS, no build step. Served as static files by FastAPI.

### UI Layout

- Connection status indicator (Online/Offline)
- Scrollable conversation history (user + assistant messages)
- Pipeline state indicator (LISTENING, THINKING, SPEAKING)
- Mic input level visualization
- Single Start/Stop button

### Audio Handling

- `getUserMedia()` for mic capture
- `AudioWorklet` to extract PCM chunks at 16kHz
- `AudioContext` with buffered queue for TTS playback
- On `interrupt` message: flush audio queue, stop playback immediately

## Project Structure

```
VoiceAgentWeb/
├── server/
│   ├── main.py              # FastAPI app, WebSocket endpoint, static file serving
│   ├── session.py           # Session manager, pipeline state machine
│   ├── stt.py               # whisper.cpp wrapper (pywhispercpp)
│   ├── llm.py               # Ollama streaming client (httpx)
│   ├── tts.py               # Piper wrapper
│   ├── vad.py               # Silero VAD wrapper
│   └── config.py            # System prompt, model paths, settings
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js           # Main app logic, UI updates
│       ├── audio.js         # Mic capture, PCM encoding, TTS playback
│       └── websocket.js     # WS connection, message routing
├── pyproject.toml
├── .gitignore
└── docs/
    └── plans/
        └── 2026-02-27-voice-agent-web-design.md
```

## Dependencies

### Python (pyproject.toml)

```
fastapi
uvicorn[standard]
pywhispercpp>=1.2
piper-tts>=1.4
torch>=2.2
httpx
numpy>=1.26
```

### External

- Ollama daemon running with `llama3.2:3b` model pulled

## Configuration

Simple system prompt, hardcoded initially:

```
You are a helpful voice assistant. Keep responses concise (1-3 sentences) since
you are speaking aloud. Be conversational and natural. Do not use markdown formatting.
```

LLM model: `llama3.2:3b` via Ollama at `http://localhost:11434`.

## Design Decisions

1. **WebSocket over WebRTC**: Simpler implementation, negligible latency difference for local use. WebRTC can be added later if needed.
2. **Single Python process**: Keeps interruption handling trivial (cancel asyncio tasks). Microservices would add complexity for no benefit at this scale.
3. **Silero VAD**: Accurate speech boundary detection enables hands-free operation and barge-in. Pulls in PyTorch (~2GB) — can swap to webrtcvad later if size is a concern.
4. **Shared model directory**: `~/Models/` for Piper, pywhispercpp default cache for Whisper. Avoids duplicating large model files across projects.
5. **Vanilla frontend**: No build step, no framework. Keeps the project simple and focused on the voice pipeline.
