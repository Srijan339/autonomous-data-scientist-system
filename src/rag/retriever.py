from __future__ import annotations

from typing import List

from src.rag.vector_store import LocalVectorStore, RetrievedChunk


class Retriever:
    def __init__(self, store: LocalVectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        return self.store.search(query=query, top_k=top_k)
