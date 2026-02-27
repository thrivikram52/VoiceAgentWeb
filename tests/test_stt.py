"""Tests for STT module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.stt import SpeechToText, TranscriptionResult


class TestTranscriptionResult:
    def test_transcription_result_has_fields(self):
        result = TranscriptionResult(text="hello", language="en")
        assert result.text == "hello"
        assert result.language == "en"


class TestSpeechToText:
    @patch("server.stt.Model")
    def test_empty_segments_returns_none(self, mock_model_cls):
        mock_model_cls.return_value.transcribe.return_value = []
        stt = SpeechToText()
        silence = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(silence)
        assert result is None

    @patch("server.stt.Model")
    def test_short_text_returns_none(self, mock_model_cls):
        seg = MagicMock()
        seg.text = " "
        mock_model_cls.return_value.transcribe.return_value = [seg]
        stt = SpeechToText()
        audio = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(audio)
        assert result is None

    @patch("server.stt.Model")
    def test_valid_transcription_returns_result(self, mock_model_cls):
        seg = MagicMock()
        seg.text = " Hello world "
        mock_model_cls.return_value.transcribe.return_value = [seg]
        mock_model_cls.return_value.language = "en"
        stt = SpeechToText()
        audio = np.random.randn(16000).astype(np.float32)
        result = stt.transcribe(audio)
        assert result is not None
        assert result.text == "Hello world"
        assert result.language == "en"

    @patch("server.stt.Model")
    def test_multiple_segments_concatenated(self, mock_model_cls):
        seg1 = MagicMock()
        seg1.text = " Hello"
        seg2 = MagicMock()
        seg2.text = " world"
        mock_model_cls.return_value.transcribe.return_value = [seg1, seg2]
        mock_model_cls.return_value.language = "en"
        stt = SpeechToText()
        audio = np.random.randn(16000).astype(np.float32)
        result = stt.transcribe(audio)
        assert result is not None
        assert result.text == "Hello world"

    @patch("server.stt.Model")
    def test_default_language_fallback(self, mock_model_cls):
        seg = MagicMock()
        seg.text = "Hello"
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = [seg]
        # Remove language attribute so getattr falls back to "en"
        del mock_instance.language
        mock_model_cls.return_value = mock_instance
        stt = SpeechToText()
        audio = np.random.randn(16000).astype(np.float32)
        result = stt.transcribe(audio)
        assert result is not None
        assert result.language == "en"
