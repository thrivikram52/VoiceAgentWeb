"""Session manager — orchestrates the VAD->STT->LLM->TTS pipeline per WebSocket connection."""

import asyncio
import enum
import logging
import time

import numpy as np
from fastapi import WebSocket

from server.config import Config
from server.llm import LLMClient
from server.stt import SpeechToText
from server.tts import TextToSpeech
from server.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

# Silence gap appended between TTS sentences (in samples at TTS sample rate)
_SENTENCE_GAP_MS = 250

# Minimum audio duration (seconds) to process after bot was recently speaking.
# Filters out short interrupt words ("wait", "stop") that sneak through.
_MIN_POST_SPEAK_AUDIO_SECS = 0.8


class SessionState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class SessionManager:
    """Manages one voice conversation session (one per WebSocket connection).

    Coordinates the pipeline: incoming audio -> VAD -> STT -> LLM -> TTS -> outgoing audio.
    Handles barge-in via both server-side VAD and client-side interrupt signals.
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
        self._post_interrupt_cooldown = 0.0
        self._last_speak_end = 0.0  # when bot last stopped speaking

    async def _send_state(self, state: SessionState) -> None:
        self.state = state
        await self.ws.send_json({"type": "state", "state": state.value})

    async def _send_transcript(self, role: str, text: str, final: bool = True) -> None:
        await self.ws.send_json({
            "type": "transcript", "role": role, "text": text, "final": final,
        })

    async def _send_interrupt(self) -> None:
        await self.ws.send_json({"type": "interrupt"})

    async def start(self) -> None:
        await self._send_state(SessionState.LISTENING)

    async def handle_interrupt_from_client(self) -> None:
        """Handle interrupt signal sent by the browser (client-side detection)."""
        if self.state != SessionState.SPEAKING:
            return

        logger.info("Client-side interrupt received")
        await self._do_interrupt()

    async def _do_interrupt(self) -> None:
        """Execute interrupt: cancel pipeline, reset state, notify client."""
        self._interrupt_event.set()
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
        self._interrupt_count = 0
        self.vad.reset()
        self._post_interrupt_cooldown = time.monotonic() + (
            self.config.interrupt_cooldown_ms / 1000
        )
        await self._send_interrupt()
        await self._send_state(SessionState.LISTENING)

    async def handle_audio(self, audio_bytes: bytes) -> None:
        """Process incoming audio chunk from the browser."""
        if time.monotonic() < self._post_interrupt_cooldown:
            return

        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(chunk) < self.config.chunk_size:
            chunk = np.pad(chunk, (0, self.config.chunk_size - len(chunk)))
        elif len(chunk) > self.config.chunk_size:
            for i in range(0, len(chunk), self.config.chunk_size):
                segment = chunk[i : i + self.config.chunk_size]
                if len(segment) == self.config.chunk_size:
                    await self._process_vad_chunk(segment)
            return

        await self._process_vad_chunk(chunk)

    async def _process_vad_chunk(self, chunk: np.ndarray) -> None:
        """Process a single VAD-sized chunk."""
        if self.state == SessionState.SPEAKING:
            # Server-side barge-in detection (backup — client-side is primary)
            is_speech = self.vad.is_speech(
                chunk, threshold=self.config.interrupt_vad_threshold
            )
            if is_speech:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 0

            if self._interrupt_count >= self.config.interrupt_consecutive_chunks:
                logger.info("Server-side barge-in detected")
                await self._do_interrupt()
            return

        if self.state != SessionState.LISTENING:
            return

        # Normal VAD processing
        audio_segment = self.vad.process_chunk(chunk)
        if audio_segment is not None:
            audio_duration = len(audio_segment) / self.config.sample_rate

            # Filter out short utterances right after bot was speaking
            # (likely interrupt words that leaked through cooldown)
            time_since_speak = time.monotonic() - self._last_speak_end
            if time_since_speak < 2.0 and audio_duration < _MIN_POST_SPEAK_AUDIO_SECS:
                logger.info(
                    f"Discarding short utterance ({audio_duration:.1f}s) "
                    f"shortly after bot spoke ({time_since_speak:.1f}s ago)"
                )
                return

            self._interrupt_event.clear()
            self._pipeline_task = asyncio.create_task(
                self._run_pipeline(audio_segment)
            )

    async def _run_pipeline(self, audio: np.ndarray) -> None:
        """Run STT -> LLM -> TTS pipeline for a complete audio segment."""
        full_response: list[str] = []
        try:
            # --- STT ---
            await self._send_state(SessionState.THINKING)

            result = await asyncio.to_thread(self.stt.transcribe, audio)
            if result is None:
                await self._send_state(SessionState.LISTENING)
                return

            logger.info(f"STT: {result.text}")
            await self._send_transcript("user", result.text)

            # --- LLM -> TTS streaming ---
            await self._send_state(SessionState.SPEAKING)

            is_first_sentence = True

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
                    audio_data, sr = tts_result

                    # Add silence gap between sentences for natural pacing
                    if not is_first_sentence:
                        gap_samples = int(sr * _SENTENCE_GAP_MS / 1000)
                        silence = np.zeros(gap_samples, dtype=np.int16)
                        audio_with_gap = np.concatenate([silence, audio_data])
                        await self.ws.send_bytes(audio_with_gap.tobytes())
                    else:
                        await self.ws.send_bytes(audio_data.tobytes())

                    is_first_sentence = False

            # Save response to history
            if full_response:
                self.llm.save_assistant_response(" ".join(full_response))

            if not self._interrupt_event.is_set():
                if full_response:
                    await self._send_transcript(
                        "assistant", " ".join(full_response), final=True
                    )
                self._last_speak_end = time.monotonic()
                await self._send_state(SessionState.LISTENING)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled (barge-in)")
            if full_response:
                self.llm.save_assistant_response(" ".join(full_response))
            self._last_speak_end = time.monotonic()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.ws.send_json({"type": "error", "message": str(e)})
            self._last_speak_end = time.monotonic()
            await self._send_state(SessionState.LISTENING)
