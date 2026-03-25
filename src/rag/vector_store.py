from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float


class LocalVectorStore:
    """Simple local retrieval store backed by TF-IDF for offline reliability."""

    def __init__(self, texts: List[str], metadatas: List[Dict[str, Any]], vectorizer: TfidfVectorizer | None = None):
        self.texts = texts
        self.metadatas = metadatas
        self.vectorizer = vectorizer or TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    @classmethod
    def from_texts(cls, texts: List[str], metadatas: List[Dict[str, Any]]) -> "LocalVectorStore":
        return cls(texts=texts, metadatas=metadatas)

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(text=self.texts[index], metadata=self.metadatas[index], score=float(scores[index]))
            for index in ranked_indices
        ]

    def save(self, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "texts": self.texts,
                "metadatas": self.metadatas,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
            },
            path,
        )
        return str(path.resolve())

    @classmethod
    def load(cls, input_path: str | Path) -> "LocalVectorStore":
        payload = joblib.load(input_path)
        store = cls(texts=payload["texts"], metadatas=payload["metadatas"], vectorizer=payload["vectorizer"])
        store.matrix = payload["matrix"]
        return store
