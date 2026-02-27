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
