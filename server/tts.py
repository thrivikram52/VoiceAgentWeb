"""Text-to-Speech — dual engine (XTTS v2 default, Piper fallback)."""

import io
import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_XTTS_SAMPLE_RATE = 24000
_DEFAULT_PIPER_MODEL = str(Path.home() / "Models" / "en_US-lessac-high.onnx")


class TextToSpeech:
    """TTS with XTTS v2 (voice cloning) or Piper (ONNX) backend."""

    def __init__(
        self,
        engine: str = "xtts",
        *,
        # Piper params
        piper_model_path: str | None = None,
        # XTTS params
        xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_voice_sample: str = str(Path.home() / "Models" / "voice_sample.wav"),
        xtts_language: str = "en",
    ):
        self._engine = engine

        if engine == "xtts":
            self._init_xtts(xtts_model_name, xtts_voice_sample, xtts_language)
        elif engine == "piper":
            self._init_piper(piper_model_path or _DEFAULT_PIPER_MODEL)
        else:
            raise ValueError(f"Unknown TTS engine: {engine!r}. Use 'xtts' or 'piper'.")

    def _init_xtts(self, model_name: str, voice_sample: str, language: str) -> None:
        if not Path(voice_sample).exists():
            raise FileNotFoundError(
                f"Voice sample not found at '{voice_sample}'. "
                "Record a 6-30s mono WAV and place it there.\n"
                "Prep: ffmpeg -i input.wav -ac 1 -ar 22050 -sample_fmt s16 ~/Models/voice_sample.wav"
            )

        from TTS.api import TTS

        logger.info("Loading XTTS v2 model (first run downloads ~1.8GB)...")
        self._tts = TTS(model_name=model_name).to("cpu")
        self._voice_sample = voice_sample
        self._language = language
        self.sample_rate = _XTTS_SAMPLE_RATE

    def _init_piper(self, model_path: str) -> None:
        from piper import PiperVoice

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Piper model not found at '{model_path}'. Download it to ~/Models/"
            )
        self._voice = PiperVoice.load(model_path)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize text to audio.

        Returns:
            Tuple of (int16 numpy array, sample_rate) or None if empty.
        """
        if not text or not text.strip():
            return None

        if self._engine == "xtts":
            return self._synthesize_xtts(text)
        return self._synthesize_piper(text)

    def _synthesize_xtts(self, text: str) -> tuple[np.ndarray, int]:
        # XTTS returns a list of floats in [-1, 1]
        wav = self._tts.tts(
            text=text,
            speaker_wav=self._voice_sample,
            language=self._language,
        )
        float_array = np.array(wav, dtype=np.float32)
        # Convert to int16 for consistent interface
        audio = np.clip(float_array * 32767, -32768, 32767).astype(np.int16)
        return audio, self.sample_rate

    def _synthesize_piper(self, text: str) -> tuple[np.ndarray, int]:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        return audio, self.sample_rate
