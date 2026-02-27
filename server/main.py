"""FastAPI server — WebSocket endpoint and static file serving."""

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import Config
from server.llm import LLMClient
from server.session import SessionManager
from server.stt import SpeechToText
from server.tts import TextToSpeech
from server.vad import VoiceActivityDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
app = FastAPI(title="VoiceAgentWeb")

# Shared model instances (loaded once at startup)
vad: VoiceActivityDetector | None = None
stt: SpeechToText | None = None
tts: TextToSpeech | None = None


@app.on_event("startup")
async def startup():
    global vad, stt, tts
    logger.info("Loading models...")

    logger.info("  Loading VAD (Silero)...")
    vad = await asyncio.to_thread(
        VoiceActivityDetector,
        threshold=config.vad_threshold,
        silence_duration_ms=config.silence_duration_ms,
        sample_rate=config.sample_rate,
        chunk_size=config.chunk_size,
    )

    logger.info("  Loading STT (whisper.cpp)...")
    stt = await asyncio.to_thread(SpeechToText, model_name=config.whisper_model)

    logger.info("  Loading TTS (Piper)...")
    tts = await asyncio.to_thread(TextToSpeech, model_path=config.piper_model_path)

    logger.info("  Checking Ollama...")
    llm = LLMClient(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        system_prompt=config.system_prompt,
        max_turns=config.max_conversation_turns,
    )
    await llm.check_available()

    logger.info("All models loaded. Server ready.")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # Each connection gets its own VAD (has internal state) and LLM (separate conversation history)
    # STT and TTS are stateless and shared across connections
    session_vad = await asyncio.to_thread(
        VoiceActivityDetector,
        threshold=config.vad_threshold,
        silence_duration_ms=config.silence_duration_ms,
        sample_rate=config.sample_rate,
        chunk_size=config.chunk_size,
    )

    session_llm = LLMClient(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        system_prompt=config.system_prompt,
        max_turns=config.max_conversation_turns,
    )

    session = SessionManager(
        ws=ws,
        vad=session_vad,
        stt=stt,
        llm=session_llm,
        tts=tts,
        config=config,
    )

    await session.start()

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    await session.handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        logger.info(f"Client config: {data}")
    except WebSocketDisconnect:
        logger.info("Client disconnected")


# Serve frontend static files (must be last — catches all remaining routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
