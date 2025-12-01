"""LLM configuration and factory utilities for Packy."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI


class LLMManager:
    """Manage construction of language model clients."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[ChatOpenAI] = None

    def get_llm(self) -> ChatOpenAI:
        """Return a cached ChatOpenAI instance configured for the project."""
        if self._llm is None:
            self._llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        return self._llm
