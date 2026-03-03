"""FastAPI server — WebSocket endpoint and static file serving."""

import asyncio
import json
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
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

# Shared model instances
vad: VoiceActivityDetector | None = None
stt: SpeechToText | None = None

# TTS: cache loaded engines, track which is active
_tts_cache: dict[str, TextToSpeech] = {}
_active_tts: TextToSpeech | None = None
_active_tts_engine: str = config.tts_engine
_tts_load_lock = asyncio.Lock()


def _build_tts(engine: str) -> TextToSpeech:
    """Create a TextToSpeech instance for the given engine."""
    return TextToSpeech(
        engine=engine,
        piper_model_path=config.piper_model_path,
        xtts_model_name=config.xtts_model_name,
        xtts_voice_sample=config.xtts_voice_sample,
        xtts_language=config.xtts_language,
        qwen3_model_name=config.qwen3_model_name,
        qwen3_voice_instruct=config.qwen3_voice_instruct,
        qwen3_language=config.qwen3_language,
    )


async def _load_tts_engine(engine: str) -> TextToSpeech:
    """Load a TTS engine (cached). Returns the instance."""
    global _active_tts, _active_tts_engine

    async with _tts_load_lock:
        if engine not in _tts_cache:
            logger.info(f"Loading TTS engine: {engine}")
            _tts_cache[engine] = await asyncio.to_thread(_build_tts, engine)
            logger.info(f"TTS engine loaded: {engine}")

        _active_tts = _tts_cache[engine]
        _active_tts_engine = engine
        return _active_tts


@app.on_event("startup")
async def startup():
    global vad, stt
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

    logger.info(f"  Loading TTS (engine={config.tts_engine})...")
    await _load_tts_engine(config.tts_engine)

    logger.info("  Checking Ollama...")
    llm = LLMClient(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        system_prompt=config.system_prompt,
        max_turns=config.max_conversation_turns,
    )
    await llm.check_available()

    logger.info("All models loaded. Server ready.")


@app.post("/api/tts/load")
async def load_tts_endpoint(request: Request):
    """Load (or switch to) a TTS engine. Caches previously loaded engines."""
    data = await request.json()
    engine = data.get("engine", config.tts_engine)

    valid_engines = ("piper", "xtts", "qwen3")
    if engine not in valid_engines:
        return JSONResponse(
            {"error": f"Unknown engine: {engine!r}. Use one of {valid_engines}"},
            status_code=400,
        )

    try:
        tts_instance = await _load_tts_engine(engine)
        return JSONResponse({
            "engine": engine,
            "sampleRate": tts_instance.sample_rate,
            "cached": engine in _tts_cache,
        })
    except Exception as e:
        logger.error(f"Failed to load TTS engine {engine!r}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        tts=_active_tts,
        config=config,
    )

    await session.start()

    # Tell client what sample rate to use for TTS playback
    if _active_tts is not None:
        await ws.send_json({"type": "tts_config", "sampleRate": _active_tts.sample_rate})

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    await session.handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        logger.info(f"Client config: {data}")
                    elif data.get("type") == "interrupt":
                        await session.handle_interrupt_from_client()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        logger.info("Client disconnected")


# Serve frontend static files (must be last — catches all remaining routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
