"""Vector store utilities for retrieval augmented generation."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """Load local data and expose a retriever for Packy."""

    def __init__(self, data_path: str = "data", chunk_size: int = 800, chunk_overlap: int = 200):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vector_store: FAISS | None = None

    def load_documents(self) -> List[Document]:
        """Load text documents from the data directory."""
        documents: List[Document] = []
        for file_path in self.data_path.glob("*.txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        return documents

    def build_vector_store(self) -> FAISS:
        """Create a vector store from local documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs = text_splitter.split_documents(self.load_documents())
        embeddings = OpenAIEmbeddings()
        self._vector_store = FAISS.from_documents(docs, embedding=embeddings)
        return self._vector_store

    def get_retriever(self):
        """Return a retriever built from the vector store."""
        if self._vector_store is None:
            self.build_vector_store()
        return self._vector_store.as_retriever()
