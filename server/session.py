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


class SessionState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class SessionManager:
    """Manages one voice conversation session (one per WebSocket connection).

    Coordinates the pipeline: incoming audio -> VAD -> STT -> LLM -> TTS -> outgoing audio.
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
        self._post_interrupt_cooldown = 0.0  # timestamp until which audio is ignored

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
        """Process incoming audio chunk from the browser."""
        # Post-interrupt cooldown: discard audio to avoid interrupt speech
        # being treated as new input
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
                # Cooldown: ignore audio for 800ms after interrupt to discard
                # the interrupt speech (e.g. "wait", "stop") so it doesn't
                # become the next user input
                self._post_interrupt_cooldown = time.monotonic() + (self.config.interrupt_cooldown_ms / 1000)
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
        """Run STT -> LLM -> TTS pipeline for a complete audio segment."""
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

            full_response: list[str] = []
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

            # Save partial response to LLM history even if interrupted
            if full_response:
                self.llm.save_assistant_response(" ".join(full_response))

            if not self._interrupt_event.is_set():
                if full_response:
                    await self._send_transcript(
                        "assistant", " ".join(full_response), final=True
                    )
                await self._send_state(SessionState.LISTENING)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled (barge-in)")
            # Still save whatever was generated so LLM has context
            if full_response:
                self.llm.save_assistant_response(" ".join(full_response))
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.ws.send_json({"type": "error", "message": str(e)})
            await self._send_state(SessionState.LISTENING)
