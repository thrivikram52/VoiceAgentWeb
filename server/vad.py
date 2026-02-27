"""Voice Activity Detection using Silero VAD."""

import enum

import numpy as np
import torch


class VADState(enum.Enum):
    """States for the voice activity detector."""
    IDLE = "idle"
    SPEAKING = "speaking"


class VoiceActivityDetector:
    """Detects speech in audio chunks using Silero VAD."""

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
