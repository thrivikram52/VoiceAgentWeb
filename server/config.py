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
        "You are a helpful voice assistant. Rules: "
        "Reply in 1 short sentence only. "
        "Use simple plain words. No special characters, no asterisks, no dashes, no quotes, no parentheses. "
        "No markdown, no lists, no formatting. Just plain spoken English."
    )

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500

    # Barge-in
    interrupt_vad_threshold: float = 0.8
    interrupt_consecutive_chunks: int = 4
    interrupt_cooldown_ms: int = 800  # ignore audio after interrupt to discard interrupt speech

    # TTS
    tts_engine: str = "piper"  # "piper", "xtts", or "qwen3"
    piper_model_path: str = str(Path.home() / "Models" / "en_US-lessac-high.onnx")
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_voice_sample: str = str(Path.home() / "Models" / "voice_sample.wav")
    xtts_language: str = "en"
    qwen3_model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
    qwen3_voice_instruct: str = "a cute young child's voice, bright and cheerful, speaking slowly and clearly"
    qwen3_language: str = "English"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
