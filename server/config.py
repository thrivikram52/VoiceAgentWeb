"""Configuration for VoiceAgentWeb."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """All configuration in one place."""

    # STT
    whisper_model: str = "large-v3-turbo"

    # LLM
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    max_conversation_turns: int = 10
    system_prompt: str = (
        "You are a helpful voice assistant. Keep every response to 1-2 short sentences max. "
        "Be direct and concise. No filler words. No markdown. Speak naturally."
    )

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500

    # Barge-in
    interrupt_vad_threshold: float = 0.6
    interrupt_consecutive_chunks: int = 2
    interrupt_cooldown_ms: int = 800  # ignore audio after interrupt to discard interrupt speech

    # TTS
    tts_engine: str = "piper"  # "piper" or "xtts"
    piper_model_path: str = str(Path.home() / "Models" / "en_US-lessac-high.onnx")
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_voice_sample: str = str(Path.home() / "Models" / "voice_sample.wav")
    xtts_language: str = "en"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
