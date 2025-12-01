"""Chat history utilities for Packy."""

from __future__ import annotations

from typing import List, Mapping, Tuple


class ChatHistoryManager:
    """Store and manipulate chat history for Streamlit session state."""

    def __init__(self):
        self.history: List[Mapping[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Append a user message to history."""
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to history."""
        self.history.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        """Remove all messages from history."""
        self.history.clear()

    def as_tuples(self) -> List[Tuple[str, str]]:
        """Return history as a list of (user, assistant) turns for LangChain."""
        turns: List[Tuple[str, str]] = []
        user_cache: str | None = None
        for message in self.history:
            if message["role"] == "user":
                user_cache = message["content"]
            elif message["role"] == "assistant" and user_cache is not None:
                turns.append((user_cache, message["content"]))
                user_cache = None
        return turns
