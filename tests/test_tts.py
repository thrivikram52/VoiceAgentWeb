"""Tests for TTS module — dual engine (XTTS + Piper)."""

from pathlib import Path

import numpy as np
import pytest

from server.config import Config
from server.tts import TextToSpeech

# Check model/sample availability for conditional skipping
_piper_available = Path(Config().piper_model_path).exists()
_xtts_sample_available = Path(Config().xtts_voice_sample).exists()


class TestPiperEngine:
    @pytest.fixture()
    def tts(self):
        if not _piper_available:
            pytest.skip("Piper model not available")
        return TextToSpeech(engine="piper")

    def test_empty_text_returns_none(self, tts):
        assert tts.synthesize("") is None

    def test_whitespace_returns_none(self, tts):
        assert tts.synthesize("   ") is None

    def test_synthesize_returns_audio_and_sample_rate(self, tts):
        result = tts.synthesize("Hello world.")
        assert result is not None
        audio, sr = result
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert sr == 22050
        assert len(audio) > 0


class TestXTTSEngine:
    @pytest.fixture()
    def tts(self):
        if not _xtts_sample_available:
            pytest.skip("XTTS voice sample not available")
        return TextToSpeech(engine="xtts")

    def test_empty_text_returns_none(self, tts):
        assert tts.synthesize("") is None

    def test_whitespace_returns_none(self, tts):
        assert tts.synthesize("   ") is None

    def test_synthesize_returns_audio_and_sample_rate(self, tts):
        result = tts.synthesize("Hello world.")
        assert result is not None
        audio, sr = result
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert sr == 24000
        assert len(audio) > 0


class TestEngineValidation:
    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            TextToSpeech(engine="invalid")

    def test_missing_voice_sample_raises(self):
        with pytest.raises(FileNotFoundError, match="Voice sample not found"):
            TextToSpeech(
                engine="xtts",
                xtts_voice_sample="/nonexistent/voice_sample.wav",
            )

    def test_missing_piper_model_raises(self):
        with pytest.raises(FileNotFoundError, match="Piper model not found"):
            TextToSpeech(
                engine="piper",
                piper_model_path="/nonexistent/model.onnx",
            )
