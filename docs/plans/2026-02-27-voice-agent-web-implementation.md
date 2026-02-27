# VoiceAgentWeb Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a browser-based real-time voice agent using WebSocket transport, with local STT (whisper.cpp), LLM (Ollama), TTS (Piper), and interruption support.

**Architecture:** Single FastAPI Python server orchestrates VAD→STT→LLM→TTS pipeline. Browser captures mic audio via AudioWorklet, streams PCM over WebSocket, receives TTS audio + JSON control messages back. Session state machine manages pipeline stages and barge-in.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, pywhispercpp, piper-tts, Silero VAD (torch), httpx, vanilla HTML/CSS/JS frontend.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `server/__init__.py`
- Create: `server/config.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voice-agent-web"
version = "0.1.0"
description = "Browser-based real-time voice agent with local models"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pywhispercpp>=1.2",
    "piper-tts>=1.4",
    "torch>=2.2",
    "httpx>=0.28",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
]
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
models/
.env
```

**Step 3: Create server/__init__.py**

Empty file.

**Step 4: Create server/config.py**

```python
"""Configuration for VoiceAgentWeb."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """All configuration in one place."""

    # STT
    whisper_model: str = "large-v3-turbo"

    # LLM
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    max_conversation_turns: int = 10
    system_prompt: str = (
        "You are a helpful voice assistant. Keep responses concise "
        "(1-3 sentences) since you are speaking aloud. Be conversational "
        "and natural. Do not use markdown formatting."
    )

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500

    # Barge-in
    interrupt_vad_threshold: float = 0.85
    interrupt_consecutive_chunks: int = 3

    # TTS
    piper_model_path: str = str(Path.home() / "Models" / "en_US-lessac-high.onnx")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
```

**Step 5: Initialize venv and install deps**

Run: `cd /Users/thrivikram/Workspace/HobbyProjects/VoiceAgentWeb && uv venv && uv pip install -e ".[dev]"`

**Step 6: Commit**

```bash
git add pyproject.toml .gitignore server/__init__.py server/config.py
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: VAD Module

**Files:**
- Create: `server/vad.py`
- Create: `tests/test_vad.py`

**Step 1: Write the failing test**

```python
"""Tests for VAD module."""

import numpy as np
import pytest

from server.vad import VoiceActivityDetector, VADState


class TestVADState:
    def test_initial_state_is_idle(self):
        vad = VoiceActivityDetector()
        assert vad.state == VADState.IDLE

    def test_reset_clears_state(self):
        vad = VoiceActivityDetector()
        vad._audio_buffer.append(np.zeros(512, dtype=np.float32))
        vad._silence_chunks = 5
        vad.state = VADState.SPEAKING
        vad.reset()
        assert vad.state == VADState.IDLE
        assert vad._audio_buffer == []
        assert vad._silence_chunks == 0


class TestVADProcessChunk:
    def test_silence_returns_none(self):
        vad = VoiceActivityDetector()
        silence = np.zeros(512, dtype=np.float32)
        result = vad.process_chunk(silence)
        assert result is None
        assert vad.state == VADState.IDLE

    def test_is_speech_returns_bool(self):
        vad = VoiceActivityDetector()
        silence = np.zeros(512, dtype=np.float32)
        result = vad.is_speech(silence)
        assert isinstance(result, bool)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thrivikram/Workspace/HobbyProjects/VoiceAgentWeb && uv run pytest tests/test_vad.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.vad'`

**Step 3: Write server/vad.py**

```python
"""Voice Activity Detection using Silero VAD."""

import enum

import numpy as np
import torch


class VADState(enum.Enum):
    """States for the voice activity detector."""
    IDLE = "idle"
    SPEAKING = "speaking"


class VoiceActivityDetector:
    """Detects speech in audio chunks using Silero VAD.

    Accumulates audio while speech is detected. Returns the complete
    audio segment when speech ends (silence exceeds duration threshold).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 500,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ):
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.state = VADState.IDLE

        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )

        self._audio_buffer: list[np.ndarray] = []
        self._silence_chunks = 0
        self._silence_chunks_threshold = int(
            (silence_duration_ms / 1000) * sample_rate / chunk_size
        )

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Process an audio chunk and detect speech boundaries.

        Returns complete audio segment when speech ends, None otherwise.
        """
        tensor = torch.from_numpy(chunk)
        speech_prob = self._model(tensor, self.sample_rate).item()

        if self.state == VADState.IDLE:
            if speech_prob >= self.threshold:
                self.state = VADState.SPEAKING
                self._audio_buffer.append(chunk)
                self._silence_chunks = 0
            return None

        self._audio_buffer.append(chunk)

        if speech_prob < self.threshold:
            self._silence_chunks += 1
            if self._silence_chunks >= self._silence_chunks_threshold:
                audio = np.concatenate(self._audio_buffer)
                self.reset()
                return audio
        else:
            self._silence_chunks = 0

        return None

    def is_speech(self, chunk: np.ndarray, threshold: float | None = None) -> bool:
        """Check if a chunk contains speech without affecting state."""
        tensor = torch.from_numpy(chunk)
        speech_prob = self._model(tensor, self.sample_rate).item()
        return speech_prob >= (threshold if threshold is not None else self.threshold)

    def reset(self) -> None:
        """Reset the VAD state and clear buffers."""
        self.state = VADState.IDLE
        self._audio_buffer = []
        self._silence_chunks = 0
        self._model.reset_states()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vad.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/vad.py tests/test_vad.py
git commit -m "feat: add VAD module with Silero speech detection"
```

---

### Task 3: STT Module

**Files:**
- Create: `server/stt.py`
- Create: `tests/test_stt.py`

**Step 1: Write the failing test**

```python
"""Tests for STT module."""

import numpy as np
import pytest

from server.stt import SpeechToText, TranscriptionResult


class TestSpeechToText:
    def test_empty_audio_returns_none(self):
        stt = SpeechToText()
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = stt.transcribe(silence)
        assert result is None

    def test_transcription_result_has_fields(self):
        result = TranscriptionResult(text="hello", language="en")
        assert result.text == "hello"
        assert result.language == "en"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stt.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write server/stt.py**

```python
"""Speech-to-Text using pywhispercpp (whisper.cpp)."""

from dataclasses import dataclass

import numpy as np
from pywhispercpp.model import Model


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str


class SpeechToText:
    """Transcribes audio using whisper.cpp via pywhispercpp."""

    def __init__(self, model_name: str = "large-v3-turbo"):
        self._model = Model(model_name)

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Transcribe audio segment to text.

        Args:
            audio: float32 numpy array, 16kHz mono.

        Returns:
            TranscriptionResult or None if empty.
        """
        segments = self._model.transcribe(audio)

        if not segments:
            return None

        text = "".join(seg.text for seg in segments).strip()

        if len(text) < 2:
            return None

        language = getattr(self._model, "language", "en")
        return TranscriptionResult(text=text, language=language)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stt.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/stt.py tests/test_stt.py
git commit -m "feat: add STT module with whisper.cpp"
```

---

### Task 4: LLM Module (Async)

**Files:**
- Create: `server/llm.py`
- Create: `tests/test_llm.py`

**Step 1: Write the failing test**

```python
"""Tests for LLM module."""

import pytest

from server.llm import LLMClient


class TestSentenceChunking:
    """Test sentence splitting logic (no Ollama needed)."""

    def test_single_sentence(self):
        llm = LLMClient()
        tokens = ["Hello", " world", "."]
        sentences = list(llm._chunk_sentences(tokens))
        assert sentences == ["Hello world."]

    def test_multiple_sentences(self):
        llm = LLMClient()
        tokens = ["Hello", ".", " How", " are", " you", "?"]
        sentences = list(llm._chunk_sentences(tokens))
        assert len(sentences) >= 1

    def test_empty_tokens(self):
        llm = LLMClient()
        sentences = list(llm._chunk_sentences([]))
        assert sentences == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write server/llm.py**

This version uses `httpx` for async streaming instead of the synchronous `ollama` library, since the server is async (FastAPI).

```python
"""LLM client using Ollama via httpx (async)."""

import re
from collections.abc import AsyncGenerator, Generator

import httpx


class LLMClient:
    """Manages conversation with an LLM via Ollama.

    Streams responses and yields complete sentences for TTS.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        system_prompt: str = "",
        max_turns: int = 10,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    async def check_available(self) -> None:
        """Check if Ollama is running and the model is available."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Start it with: ollama serve"
                ) from e

            data = resp.json()
            model_names = [m["name"] for m in data.get("models", [])]
            if self.model not in model_names:
                raise ValueError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}\n"
                    f"Available: {model_names}"
                )

    async def chat(self, user_message: str) -> AsyncGenerator[str, None]:
        """Send a message and yield complete sentences.

        Args:
            user_message: The user's transcribed speech.

        Yields:
            Complete sentences as they become available.
        """
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._history,
        ]

        full_response: list[str] = []
        buffer = ""

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if not token:
                        continue

                    buffer += token
                    sentences, buffer = self._extract_sentences(buffer)
                    for sentence in sentences:
                        full_response.append(sentence)
                        yield sentence

        # Yield remaining buffer
        if buffer.strip():
            full_response.append(buffer.strip())
            yield buffer.strip()

        assistant_message = " ".join(full_response)
        if assistant_message:
            self._history.append({"role": "assistant", "content": assistant_message})
        else:
            self._history.pop()

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def _extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Extract complete sentences from buffer.

        Returns:
            Tuple of (complete_sentences, remaining_buffer).
        """
        parts = self._SENTENCE_SPLIT_RE.split(buffer)
        if len(parts) > 1:
            sentences = [p.strip() for p in parts[:-1] if p.strip()]
            return sentences, parts[-1]
        return [], buffer

    def _chunk_sentences(
        self, tokens: list[str] | Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Sync version for testing: accumulate tokens, yield sentences."""
        buffer = ""
        for token in tokens:
            buffer += token
            sentences, buffer = self._extract_sentences(buffer)
            yield from sentences
        if buffer.strip():
            yield buffer.strip()

    def _trim_history(self) -> None:
        """Trim conversation history to max_turns."""
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/llm.py tests/test_llm.py
git commit -m "feat: add async LLM client with sentence chunking"
```

---

### Task 5: TTS Module

**Files:**
- Create: `server/tts.py`
- Create: `tests/test_tts.py`

**Step 1: Write the failing test**

```python
"""Tests for TTS module."""

import numpy as np
import pytest

from server.tts import TextToSpeech


class TestTextToSpeech:
    def test_empty_text_returns_none(self):
        tts = TextToSpeech()
        result = tts.synthesize("")
        assert result is None

    def test_whitespace_returns_none(self):
        tts = TextToSpeech()
        result = tts.synthesize("   ")
        assert result is None

    def test_synthesize_returns_audio_and_sample_rate(self):
        tts = TextToSpeech()
        result = tts.synthesize("Hello world.")
        assert result is not None
        audio, sr = result
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert sr > 0
        assert len(audio) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts.py -v`
Expected: FAIL

**Step 3: Write server/tts.py**

Note: This version returns int16 audio (not float32) since we send raw PCM bytes over WebSocket.

```python
"""Text-to-Speech using Piper."""

import io
import wave
from pathlib import Path

import numpy as np


class TextToSpeech:
    """TTS using Piper (ONNX, CPU-based)."""

    _DEFAULT_MODEL = str(Path.home() / "Models" / "en_US-lessac-high.onnx")

    def __init__(self, model_path: str | None = None):
        from piper import PiperVoice

        path = model_path or self._DEFAULT_MODEL
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Piper model not found at '{path}'. Download it to ~/Models/"
            )
        self._voice = PiperVoice.load(path)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize text to audio.

        Returns:
            Tuple of (int16 numpy array, sample_rate) or None if empty.
        """
        if not text or not text.strip():
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        return audio, self.sample_rate
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/tts.py tests/test_tts.py
git commit -m "feat: add TTS module with Piper"
```

---

### Task 6: Session Manager (Pipeline State Machine)

**Files:**
- Create: `server/session.py`
- Create: `tests/test_session.py`

**Step 1: Write the failing test**

```python
"""Tests for session state machine."""

import pytest

from server.session import SessionState, SessionManager


class TestSessionState:
    def test_initial_state_is_idle(self):
        sm = SessionManager.__new__(SessionManager)
        sm.state = SessionState.IDLE
        assert sm.state == SessionState.IDLE

    def test_all_states_exist(self):
        assert SessionState.IDLE
        assert SessionState.LISTENING
        assert SessionState.THINKING
        assert SessionState.SPEAKING
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL

**Step 3: Write server/session.py**

```python
"""Session manager — orchestrates the VAD→STT→LLM→TTS pipeline per WebSocket connection."""

import asyncio
import enum
import json
import logging
from collections.abc import Callable

import numpy as np
from fastapi import WebSocket

from server.config import Config
from server.llm import LLMClient
from server.stt import SpeechToText
from server.tts import TextToSpeech
from server.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class SessionManager:
    """Manages one voice conversation session (one per WebSocket connection).

    Coordinates the pipeline: incoming audio → VAD → STT → LLM → TTS → outgoing audio.
    Handles barge-in by cancelling in-flight LLM/TTS when speech detected during SPEAKING.
    """

    def __init__(
        self,
        ws: WebSocket,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        llm: LLMClient,
        tts: TextToSpeech,
        config: Config,
    ):
        self.ws = ws
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.config = config
        self.state = SessionState.IDLE
        self._pipeline_task: asyncio.Task | None = None
        self._interrupt_event = asyncio.Event()
        self._interrupt_count = 0

    async def _send_state(self, state: SessionState) -> None:
        """Send state change to client."""
        self.state = state
        await self.ws.send_json({"type": "state", "state": state.value})

    async def _send_transcript(self, role: str, text: str, final: bool = True) -> None:
        """Send transcript to client."""
        await self.ws.send_json({
            "type": "transcript",
            "role": role,
            "text": text,
            "final": final,
        })

    async def _send_interrupt(self) -> None:
        """Tell client to stop audio playback."""
        await self.ws.send_json({"type": "interrupt"})

    async def start(self) -> None:
        """Start listening for audio."""
        await self._send_state(SessionState.LISTENING)

    async def handle_audio(self, audio_bytes: bytes) -> None:
        """Process incoming audio chunk from the browser.

        Converts bytes to float32 numpy array and feeds to VAD.
        """
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Pad or trim to chunk_size for VAD
        if len(chunk) < self.config.chunk_size:
            chunk = np.pad(chunk, (0, self.config.chunk_size - len(chunk)))
        elif len(chunk) > self.config.chunk_size:
            # Process in chunk_size segments
            for i in range(0, len(chunk), self.config.chunk_size):
                segment = chunk[i:i + self.config.chunk_size]
                if len(segment) == self.config.chunk_size:
                    await self._process_vad_chunk(segment)
            return

        await self._process_vad_chunk(chunk)

    async def _process_vad_chunk(self, chunk: np.ndarray) -> None:
        """Process a single VAD-sized chunk."""
        if self.state == SessionState.SPEAKING:
            # Check for barge-in
            is_speech = self.vad.is_speech(
                chunk, threshold=self.config.interrupt_vad_threshold
            )
            if is_speech:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 0

            if self._interrupt_count >= self.config.interrupt_consecutive_chunks:
                logger.info("Barge-in detected — interrupting pipeline")
                self._interrupt_event.set()
                if self._pipeline_task and not self._pipeline_task.done():
                    self._pipeline_task.cancel()
                self._interrupt_count = 0
                self.vad.reset()
                await self._send_interrupt()
                await self._send_state(SessionState.LISTENING)
            return

        if self.state != SessionState.LISTENING:
            return

        # Normal VAD processing
        audio_segment = self.vad.process_chunk(chunk)
        if audio_segment is not None:
            # Speech ended — start pipeline
            self._interrupt_event.clear()
            self._pipeline_task = asyncio.create_task(
                self._run_pipeline(audio_segment)
            )

    async def _run_pipeline(self, audio: np.ndarray) -> None:
        """Run STT → LLM → TTS pipeline for a complete audio segment."""
        try:
            # --- STT ---
            await self._send_state(SessionState.THINKING)

            result = await asyncio.to_thread(self.stt.transcribe, audio)
            if result is None:
                await self._send_state(SessionState.LISTENING)
                return

            logger.info(f"STT: {result.text}")
            await self._send_transcript("user", result.text)

            # --- LLM → TTS streaming ---
            await self._send_state(SessionState.SPEAKING)

            full_response: list[str] = []
            async for sentence in self.llm.chat(result.text):
                if self._interrupt_event.is_set():
                    break

                full_response.append(sentence)
                await self._send_transcript(
                    "assistant", " ".join(full_response), final=False
                )

                # TTS for this sentence
                tts_result = await asyncio.to_thread(self.tts.synthesize, sentence)
                if tts_result is not None and not self._interrupt_event.is_set():
                    audio_data, _sr = tts_result
                    await self.ws.send_bytes(audio_data.tobytes())

            if not self._interrupt_event.is_set():
                if full_response:
                    await self._send_transcript(
                        "assistant", " ".join(full_response), final=True
                    )
                await self._send_state(SessionState.LISTENING)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled (barge-in)")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.ws.send_json({"type": "error", "message": str(e)})
            await self._send_state(SessionState.LISTENING)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_session.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/session.py tests/test_session.py
git commit -m "feat: add session manager with pipeline state machine and barge-in"
```

---

### Task 7: FastAPI Server (main.py)

**Files:**
- Create: `server/main.py`

**Step 1: Write server/main.py**

```python
"""FastAPI server — WebSocket endpoint and static file serving."""

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import Config
from server.llm import LLMClient
from server.session import SessionManager
from server.stt import SpeechToText
from server.tts import TextToSpeech
from server.vad import VoiceActivityDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
app = FastAPI(title="VoiceAgentWeb")

# Shared model instances (loaded once at startup)
vad: VoiceActivityDetector | None = None
stt: SpeechToText | None = None
tts: TextToSpeech | None = None


@app.on_event("startup")
async def startup():
    global vad, stt, tts
    logger.info("Loading models...")

    logger.info("  Loading VAD (Silero)...")
    vad = await asyncio.to_thread(
        VoiceActivityDetector,
        threshold=config.vad_threshold,
        silence_duration_ms=config.silence_duration_ms,
        sample_rate=config.sample_rate,
        chunk_size=config.chunk_size,
    )

    logger.info("  Loading STT (whisper.cpp)...")
    stt = await asyncio.to_thread(SpeechToText, model_name=config.whisper_model)

    logger.info("  Loading TTS (Piper)...")
    tts = await asyncio.to_thread(TextToSpeech, model_path=config.piper_model_path)

    logger.info("  Checking Ollama...")
    llm = LLMClient(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        system_prompt=config.system_prompt,
        max_turns=config.max_conversation_turns,
    )
    await llm.check_available()

    logger.info("All models loaded. Server ready.")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # Each connection gets its own LLM client (separate conversation history)
    # but shares VAD/STT/TTS model instances.
    # NOTE: VAD has internal state, so each session needs its own instance.
    session_vad = await asyncio.to_thread(
        VoiceActivityDetector,
        threshold=config.vad_threshold,
        silence_duration_ms=config.silence_duration_ms,
        sample_rate=config.sample_rate,
        chunk_size=config.chunk_size,
    )

    session_llm = LLMClient(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        system_prompt=config.system_prompt,
        max_turns=config.max_conversation_turns,
    )

    session = SessionManager(
        ws=ws,
        vad=session_vad,
        stt=stt,
        llm=session_llm,
        tts=tts,
        config=config,
    )

    await session.start()

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    await session.handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    import json
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        logger.info(f"Client config: {data}")
    except WebSocketDisconnect:
        logger.info("Client disconnected")


# Serve frontend static files (must be last — catches all remaining routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
```

**Step 2: Verify server starts**

Run: `uv run python -m server.main`
Expected: Server starts, logs model loading, listens on port 8080. Ctrl+C to stop.

**Step 3: Commit**

```bash
git add server/main.py
git commit -m "feat: add FastAPI server with WebSocket endpoint"
```

---

### Task 8: Frontend — HTML + CSS

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/css/style.css`

**Step 1: Create frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Voice Agent</h1>
            <span id="status" class="status offline">Offline</span>
        </header>

        <div id="conversation" class="conversation"></div>

        <div class="controls">
            <div id="state-indicator" class="state-indicator">IDLE</div>
            <div id="audio-level" class="audio-level">
                <div id="audio-level-bar" class="audio-level-bar"></div>
            </div>
            <button id="mic-btn" class="mic-btn" disabled>Start</button>
        </div>
    </div>

    <script src="/js/audio.js"></script>
    <script src="/js/websocket.js"></script>
    <script src="/js/app.js"></script>
</body>
</html>
```

**Step 2: Create frontend/css/style.css**

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    width: 100%;
    max-width: 640px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 1rem;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
    border-bottom: 1px solid #222;
}

header h1 {
    font-size: 1.2rem;
    font-weight: 600;
}

.status {
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
}

.status.online {
    background: #0d3320;
    color: #4ade80;
}

.status.offline {
    background: #3b1c1c;
    color: #f87171;
}

.conversation {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.message {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 85%;
    line-height: 1.5;
    font-size: 0.95rem;
}

.message.user {
    align-self: flex-end;
    background: #1e3a5f;
    color: #bfdbfe;
}

.message.assistant {
    align-self: flex-start;
    background: #1a1a2e;
    color: #d1d5db;
}

.message.assistant.streaming {
    border-left: 2px solid #6366f1;
}

.controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding-top: 1rem;
    border-top: 1px solid #222;
}

.state-indicator {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
}

.state-indicator.listening {
    color: #4ade80;
}

.state-indicator.thinking {
    color: #facc15;
}

.state-indicator.speaking {
    color: #6366f1;
}

.audio-level {
    width: 200px;
    height: 4px;
    background: #222;
    border-radius: 2px;
    overflow: hidden;
}

.audio-level-bar {
    height: 100%;
    width: 0%;
    background: #4ade80;
    transition: width 50ms ease;
}

.mic-btn {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 2px solid #333;
    background: #1a1a1a;
    color: #e0e0e0;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
}

.mic-btn:hover:not(:disabled) {
    border-color: #4ade80;
    background: #0d3320;
}

.mic-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.mic-btn.active {
    border-color: #f87171;
    background: #3b1c1c;
    color: #f87171;
}
```

**Step 3: Commit**

```bash
git add frontend/index.html frontend/css/style.css
git commit -m "feat: add frontend HTML and CSS"
```

---

### Task 9: Frontend — Audio Module (Mic Capture + TTS Playback)

**Files:**
- Create: `frontend/js/audio.js`

**Step 1: Create frontend/js/audio.js**

```javascript
/**
 * Audio module — mic capture (PCM 16-bit 16kHz) and TTS playback.
 */

class AudioManager {
    constructor() {
        this.audioContext = null;
        this.stream = null;
        this.workletNode = null;
        this.onAudioChunk = null; // callback: (Int16Array) => void
        this.onAudioLevel = null; // callback: (0..1) => void

        // TTS playback queue
        this._playbackQueue = [];
        this._isPlaying = false;
        this._playbackSampleRate = 22050; // Piper default
    }

    async startMic() {
        this.audioContext = new AudioContext({ sampleRate: 16000 });

        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // Load AudioWorklet for PCM extraction
        await this.audioContext.audioWorklet.addModule("/js/pcm-worklet.js");
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-processor");

        this.workletNode.port.onmessage = (event) => {
            const { pcm, level } = event.data;
            if (this.onAudioChunk) {
                this.onAudioChunk(new Int16Array(pcm));
            }
            if (this.onAudioLevel) {
                this.onAudioLevel(level);
            }
        };

        source.connect(this.workletNode);
        // Don't connect to destination — we don't want mic echoed to speakers
    }

    stopMic() {
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
            this.stream = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }

    /**
     * Queue TTS audio (Int16 PCM) for playback.
     * @param {ArrayBuffer} pcmBuffer - Raw PCM bytes (int16)
     */
    queueTTSAudio(pcmBuffer) {
        this._playbackQueue.push(pcmBuffer);
        if (!this._isPlaying) {
            this._playNext();
        }
    }

    async _playNext() {
        if (this._playbackQueue.length === 0) {
            this._isPlaying = false;
            return;
        }

        this._isPlaying = true;
        const pcmBuffer = this._playbackQueue.shift();

        // Convert Int16 PCM to Float32 for Web Audio API
        const int16 = new Int16Array(pcmBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        // Create AudioBuffer and play
        const playbackCtx = this.audioContext || new AudioContext();
        const audioBuffer = playbackCtx.createBuffer(
            1,
            float32.length,
            this._playbackSampleRate
        );
        audioBuffer.getChannelData(0).set(float32);

        const source = playbackCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackCtx.destination);
        source.onended = () => this._playNext();
        source.start();
        this._currentSource = source;
    }

    /**
     * Stop all TTS playback immediately (for barge-in).
     */
    flushPlayback() {
        this._playbackQueue = [];
        this._isPlaying = false;
        if (this._currentSource) {
            try {
                this._currentSource.stop();
            } catch (e) {
                // Already stopped
            }
            this._currentSource = null;
        }
    }
}
```

**Step 2: Create the AudioWorklet processor**

Create: `frontend/js/pcm-worklet.js`

```javascript
/**
 * AudioWorklet processor — converts float32 audio to int16 PCM chunks.
 * Runs in a separate thread for low-latency processing.
 */

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(0);
        // Send chunks of ~32ms at 16kHz = 512 samples
        this._chunkSize = 512;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];

        // Append to buffer
        const newBuffer = new Float32Array(
            this._buffer.length + channelData.length
        );
        newBuffer.set(this._buffer);
        newBuffer.set(channelData, this._buffer.length);
        this._buffer = newBuffer;

        // Send complete chunks
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);

            // Convert to Int16
            const pcm = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }

            // Calculate RMS level for visualization
            let sum = 0;
            for (let i = 0; i < chunk.length; i++) {
                sum += chunk[i] * chunk[i];
            }
            const level = Math.sqrt(sum / chunk.length);

            this.port.postMessage(
                { pcm: pcm.buffer, level },
                [pcm.buffer]
            );
        }

        return true;
    }
}

registerProcessor("pcm-processor", PCMProcessor);
```

**Step 3: Commit**

```bash
git add frontend/js/audio.js frontend/js/pcm-worklet.js
git commit -m "feat: add audio module with mic capture and TTS playback"
```

---

### Task 10: Frontend — WebSocket Module

**Files:**
- Create: `frontend/js/websocket.js`

**Step 1: Create frontend/js/websocket.js**

```javascript
/**
 * WebSocket module — connection management and message routing.
 */

class VoiceWebSocket {
    constructor() {
        this.ws = null;
        this.onStateChange = null;   // (state: string) => void
        this.onTranscript = null;    // (role, text, final) => void
        this.onInterrupt = null;     // () => void
        this.onTTSAudio = null;      // (ArrayBuffer) => void
        this.onConnectionChange = null; // (connected: boolean) => void
        this.onError = null;         // (message: string) => void
    }

    connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${protocol}//${location.host}/ws`;

        this.ws = new WebSocket(url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
            console.log("WebSocket connected");
            // Send config
            this.ws.send(JSON.stringify({
                type: "config",
                sampleRate: 16000,
            }));
            if (this.onConnectionChange) this.onConnectionChange(true);
        };

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary = TTS audio
                if (this.onTTSAudio) this.onTTSAudio(event.data);
                return;
            }

            // JSON control message
            const msg = JSON.parse(event.data);

            switch (msg.type) {
                case "state":
                    if (this.onStateChange) this.onStateChange(msg.state);
                    break;
                case "transcript":
                    if (this.onTranscript)
                        this.onTranscript(msg.role, msg.text, msg.final);
                    break;
                case "interrupt":
                    if (this.onInterrupt) this.onInterrupt();
                    break;
                case "error":
                    if (this.onError) this.onError(msg.message);
                    break;
            }
        };

        this.ws.onclose = () => {
            console.log("WebSocket disconnected");
            if (this.onConnectionChange) this.onConnectionChange(false);
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    sendAudio(int16Array) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(int16Array.buffer);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
```

**Step 2: Commit**

```bash
git add frontend/js/websocket.js
git commit -m "feat: add WebSocket module with message routing"
```

---

### Task 11: Frontend — App Module (Wiring It All Together)

**Files:**
- Create: `frontend/js/app.js`

**Step 1: Create frontend/js/app.js**

```javascript
/**
 * Main app — wires AudioManager, VoiceWebSocket, and DOM together.
 */

const audio = new AudioManager();
const socket = new VoiceWebSocket();

// DOM elements
const micBtn = document.getElementById("mic-btn");
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state-indicator");
const convoEl = document.getElementById("conversation");
const levelBar = document.getElementById("audio-level-bar");

let isActive = false;
let currentAssistantEl = null;

// --- Socket callbacks ---

socket.onConnectionChange = (connected) => {
    statusEl.textContent = connected ? "Online" : "Offline";
    statusEl.className = `status ${connected ? "online" : "offline"}`;
    micBtn.disabled = !connected;
};

socket.onStateChange = (state) => {
    stateEl.textContent = state;
    stateEl.className = `state-indicator ${state.toLowerCase()}`;
};

socket.onTranscript = (role, text, final) => {
    if (role === "user") {
        addMessage("user", text);
    } else if (role === "assistant") {
        if (!currentAssistantEl) {
            currentAssistantEl = addMessage("assistant", text, !final);
        } else {
            currentAssistantEl.textContent = text;
            if (final) {
                currentAssistantEl.classList.remove("streaming");
                currentAssistantEl = null;
            }
        }
    }
};

socket.onTTSAudio = (buffer) => {
    audio.queueTTSAudio(buffer);
};

socket.onInterrupt = () => {
    audio.flushPlayback();
    if (currentAssistantEl) {
        currentAssistantEl.classList.remove("streaming");
        currentAssistantEl = null;
    }
};

socket.onError = (msg) => {
    console.error("Server error:", msg);
};

// --- Audio callbacks ---

audio.onAudioChunk = (int16Array) => {
    socket.sendAudio(int16Array);
};

audio.onAudioLevel = (level) => {
    const pct = Math.min(level * 500, 100); // Scale for visibility
    levelBar.style.width = `${pct}%`;
};

// --- UI ---

function addMessage(role, text, streaming = false) {
    const el = document.createElement("div");
    el.className = `message ${role}${streaming ? " streaming" : ""}`;
    el.textContent = text;
    convoEl.appendChild(el);
    convoEl.scrollTop = convoEl.scrollHeight;
    return el;
}

micBtn.addEventListener("click", async () => {
    if (!isActive) {
        try {
            await audio.startMic();
            socket.connect();
            micBtn.textContent = "Stop";
            micBtn.classList.add("active");
            isActive = true;
        } catch (err) {
            console.error("Failed to start:", err);
            alert("Microphone access required.");
        }
    } else {
        audio.stopMic();
        socket.disconnect();
        micBtn.textContent = "Start";
        micBtn.classList.remove("active");
        isActive = false;
        levelBar.style.width = "0%";
        stateEl.textContent = "IDLE";
        stateEl.className = "state-indicator";
    }
});
```

**Step 2: Commit**

```bash
git add frontend/js/app.js
git commit -m "feat: add main app module wiring audio, websocket, and UI"
```

---

### Task 12: Create tests/__init__.py and End-to-End Smoke Test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_server.py`

**Step 1: Create tests/__init__.py**

Empty file.

**Step 2: Write smoke test**

```python
"""Smoke test — verify server starts and serves frontend."""

import pytest
from fastapi.testclient import TestClient


class TestServerSmoke:
    def test_frontend_served(self):
        """Verify the frontend index.html is served at /."""
        from server.main import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Voice Agent" in resp.text

    def test_websocket_connects(self):
        """Verify WebSocket endpoint accepts connections."""
        from server.main import app
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Should receive initial state message
            data = ws.receive_json()
            assert data["type"] == "state"
            assert data["state"] == "LISTENING"
```

**Step 3: Run smoke test**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (may be slow first time due to model loading)

**Step 4: Commit**

```bash
git add tests/__init__.py tests/test_server.py
git commit -m "test: add smoke tests for server startup and WebSocket"
```

---

### Task 13: Manual End-to-End Test

**Step 1: Start Ollama**

Run: `ollama serve` (if not already running)
Verify: `ollama list` shows `llama3.2:3b`

**Step 2: Start the server**

Run: `cd /Users/thrivikram/Workspace/HobbyProjects/VoiceAgentWeb && uv run python -m server.main`
Expected: Logs showing all models loaded, server on port 8080

**Step 3: Open browser**

Navigate to `http://localhost:8080`
Expected: Voice Agent UI with Start button

**Step 4: Test conversation flow**

1. Click "Start" — should request mic permission, status goes "Online"
2. Speak a short sentence — state should go LISTENING → THINKING → SPEAKING
3. Hear TTS response through speakers
4. Verify transcript appears in conversation history
5. Test barge-in: speak while agent is talking — audio should stop

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: VoiceAgentWeb v0.1 — complete voice agent with WebSocket transport"
```

---

## Task Summary

| Task | Component | Est. Complexity |
|------|-----------|----------------|
| 1 | Project scaffolding | Low |
| 2 | VAD module | Low (port from VoiceAgent) |
| 3 | STT module | Low (port from VoiceAgent) |
| 4 | LLM module (async) | Medium (sync→async rewrite) |
| 5 | TTS module | Low (port from VoiceAgent) |
| 6 | Session manager | High (state machine + barge-in) |
| 7 | FastAPI server | Medium (WebSocket + lifecycle) |
| 8 | Frontend HTML + CSS | Low |
| 9 | Frontend audio module | Medium (AudioWorklet + playback) |
| 10 | Frontend WebSocket | Low |
| 11 | Frontend app wiring | Low |
| 12 | Smoke tests | Low |
| 13 | Manual E2E test | Manual verification |
