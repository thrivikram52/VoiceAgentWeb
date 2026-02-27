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
