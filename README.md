# VoiceAgentWeb

A browser-based real-time voice assistant that runs entirely on local models. Speak into your mic, get spoken responses back with sub-second latency.

## Architecture

```
Browser (mic) → WebSocket → VAD → STT → LLM → TTS → WebSocket → Browser (speaker)
```

| Stage | Model | Description |
|-------|-------|-------------|
| VAD | Silero VAD | Voice activity detection to segment speech |
| STT | whisper.cpp (large-v3-turbo) | Speech-to-text transcription |
| LLM | Ollama (llama3.2:3b) | Conversational response generation |
| TTS | Piper / XTTS v2 / Qwen3 | Text-to-speech synthesis (selectable) |

## Features

- Real-time streaming voice conversation
- Three TTS engines with UI dropdown selector
- Per-component latency display (STT/LLM/TTS) on each response
- Client-side and server-side barge-in (interrupt) detection
- Streaming TTS for Qwen3 (reduced time-to-first-audio)
- Conversation history with configurable turn limit

## TTS Engines

| Engine | Speed | Quality | Notes |
|--------|-------|---------|-------|
| Piper | ~30-100ms | Good | ONNX-based, CPU-only, lightweight |
| XTTS v2 | ~500-1000ms | Great | Voice cloning from a WAV sample |
| Qwen3 Kids Voice | ~2-4s | Excellent | MLX-optimized, voice design via text description |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b`
- Piper model at `~/Models/en_US-lessac-high.onnx` (for Piper engine)
- XTTS voice sample at `~/Models/voice_sample.wav` (for XTTS engine)

### For Qwen3 TTS (Apple Silicon)

```bash
pip install mlx-audio
```

The MLX-optimized model (`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`) downloads automatically on first use.

## Setup

```bash
# Clone and install
git clone https://github.com/thrivikram52/VoiceAgentWeb.git
cd VoiceAgentWeb
uv sync

# Pull the LLM
ollama pull llama3.2:3b

# Start the server
uv run python -m server.main
```

Open http://localhost:8080 in your browser.

## Usage

1. Select a TTS model from the dropdown (loads on selection)
2. Click **Start** to begin the conversation
3. Speak into your mic -- the assistant responds with voice
4. Each assistant response shows latency badges: `STT: Xms | LLM: Xms | TTS: Xms`
5. Click **Stop** to end the session

## Project Structure

```
server/
  config.py       Configuration dataclass (models, thresholds, prompts)
  main.py         FastAPI server, WebSocket endpoint, TTS cache + REST API
  session.py      Pipeline orchestration (VAD -> STT -> LLM -> TTS)
  llm.py          Ollama LLM client with sentence-level streaming
  stt.py          whisper.cpp speech-to-text
  tts.py          Multi-engine TTS (Piper, XTTS v2, Qwen3 via MLX)
  vad.py          Silero VAD wrapper
frontend/
  index.html      UI shell
  css/style.css   Dark theme styling
  js/app.js       App wiring (socket callbacks, DOM, TTS selector)
  js/audio.js     Mic capture + TTS playback + interrupt detection
  js/websocket.js WebSocket message routing
```

## Configuration

All settings are in `server/config.py`. Key tuning parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `tts_engine` | `piper` | Default TTS engine |
| `interrupt_vad_threshold` | `0.8` | Server-side barge-in sensitivity (0-1) |
| `interrupt_consecutive_chunks` | `4` | Consecutive speech chunks to trigger interrupt |
| `qwen3_voice_instruct` | `a cute young child's voice...` | Qwen3 voice description |
| `system_prompt` | (see config) | LLM behavior instructions |
