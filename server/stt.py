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
