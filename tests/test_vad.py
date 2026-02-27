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
