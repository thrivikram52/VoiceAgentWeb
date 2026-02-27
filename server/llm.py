"""LLM client using Ollama via httpx (async)."""

import json
import re
from collections.abc import AsyncGenerator, Generator

import httpx


class LLMClient:
    """Manages conversation with an LLM via Ollama.

    Streams responses and yields complete sentences for TTS.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        system_prompt: str = "",
        max_turns: int = 10,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    async def check_available(self) -> None:
        """Check if Ollama is running and the model is available."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Start it with: ollama serve"
                ) from e

            data = resp.json()
            model_names = [m["name"] for m in data.get("models", [])]
            if self.model not in model_names:
                raise ValueError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}\n"
                    f"Available: {model_names}"
                )

    async def chat(self, user_message: str) -> AsyncGenerator[str, None]:
        """Send a message and yield complete sentences.

        Args:
            user_message: The user's transcribed speech.

        Yields:
            Complete sentences as they become available.

        Note:
            History is NOT auto-saved by this generator. The caller must
            call save_assistant_response() with the collected sentences,
            even if interrupted, to keep conversation context consistent.
        """
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._history,
        ]

        buffer = ""

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if not token:
                        continue

                    buffer += token
                    sentences, buffer = self._extract_sentences(buffer)
                    for sentence in sentences:
                        yield sentence

        # Yield remaining buffer
        if buffer.strip():
            yield buffer.strip()

    def save_assistant_response(self, text: str) -> None:
        """Save the assistant's response to history.

        Call this after chat() completes or is interrupted, passing the
        collected sentences. If text is empty, removes the dangling user message.
        """
        if text:
            self._history.append({"role": "assistant", "content": text})
        elif self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def _extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Extract complete sentences from buffer.

        Returns:
            Tuple of (complete_sentences, remaining_buffer).
        """
        parts = self._SENTENCE_SPLIT_RE.split(buffer)
        if len(parts) > 1:
            sentences = [p.strip() for p in parts[:-1] if p.strip()]
            return sentences, parts[-1]
        return [], buffer

    def _chunk_sentences(
        self, tokens: list[str] | Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Sync version for testing: accumulate tokens, yield sentences."""
        buffer = ""
        for token in tokens:
            buffer += token
            sentences, buffer = self._extract_sentences(buffer)
            yield from sentences
        if buffer.strip():
            yield buffer.strip()

    def _trim_history(self) -> None:
        """Trim conversation history to max_turns."""
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
