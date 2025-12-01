"""Packy agent orchestration for travel packing assistance."""

from __future__ import annotations

from typing import List, Mapping, Tuple

from langchain.chains import ConversationalRetrievalChain

from modules.llm import LLMManager
from modules.vector_store import VectorStoreManager


class PackyAgent:
    """Decide between retrieval and direct answering for Packy."""

    def __init__(self):
        self.llm_manager = LLMManager()
        self.vector_store_manager = VectorStoreManager()
        self.chain = self._build_chain()

    def _build_chain(self) -> ConversationalRetrievalChain:
        """Create a conversational retrieval chain with the configured LLM and retriever."""
        llm = self.llm_manager.get_llm()
        retriever = self.vector_store_manager.get_retriever()
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    def handle_input(
        self,
        user_input: str,
        history: List[Mapping[str, str]] | None = None,
        history_tuples: List[Tuple[str, str]] | None = None,
    ) -> str:
        """Generate a response given the user input and chat history."""
        chat_history = history_tuples or []
        result = self.chain.invoke({"question": user_input, "chat_history": chat_history})
        return result.get("answer", "죄송해요, 아직 답변을 준비 중이에요.")
