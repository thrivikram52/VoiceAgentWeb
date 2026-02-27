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
        "You are a helpful voice assistant. Keep responses concise "
        "(1-3 sentences) since you are speaking aloud. Be conversational "
        "and natural. Do not use markdown formatting."
    )

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500

    # Barge-in
    interrupt_vad_threshold: float = 0.85
    interrupt_consecutive_chunks: int = 3

    # TTS
    piper_model_path: str = str(Path.home() / "Models" / "en_US-lessac-high.onnx")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
