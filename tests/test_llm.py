"""Tests for LLM module."""

import pytest

from server.llm import LLMClient


class TestSentenceChunking:
    """Test sentence splitting logic (no Ollama needed)."""

    def test_single_sentence(self):
        llm = LLMClient()
        tokens = ["Hello", " world", "."]
        sentences = list(llm._chunk_sentences(tokens))
        assert sentences == ["Hello world."]

    def test_multiple_sentences(self):
        llm = LLMClient()
        tokens = ["Hello", ".", " How", " are", " you", "?"]
        sentences = list(llm._chunk_sentences(tokens))
        assert len(sentences) >= 1

    def test_empty_tokens(self):
        llm = LLMClient()
        sentences = list(llm._chunk_sentences([]))
        assert sentences == []
