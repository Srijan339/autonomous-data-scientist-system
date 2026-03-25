from __future__ import annotations

from typing import Any, Dict, List

from src.rag.vector_store import RetrievedChunk


class QAAssistant:
    """Answer dataset questions from stored analysis artifacts without external dependencies."""

    def answer(self, query: str, retrieved_chunks: List[RetrievedChunk], results_payload: Dict[str, Any]) -> Dict[str, Any]:
        query_lower = query.lower()
        metrics = results_payload.get("metrics", {})
        feature_importance = results_payload.get("feature_importance", [])

        if "best model" in query_lower:
            answer = f"The best model is {results_payload.get('best_model')} for a {results_payload.get('problem_type')} task."
        elif "problem type" in query_lower or "classification" in query_lower or "regression" in query_lower:
            answer = f"The system detected the task as {results_payload.get('problem_type')}."
        elif "accuracy" in query_lower or "metric" in query_lower or "performance" in query_lower:
            answer = "The latest evaluation metrics are: " + ", ".join(f"{k}={v}" for k, v in metrics.items()) + "."
        elif "feature" in query_lower or "important" in query_lower:
            top_features = ", ".join(item["feature"] for item in feature_importance[:5])
            answer = f"The strongest model signals are {top_features}."
        else:
            context = " ".join(chunk.text for chunk in retrieved_chunks[:3]).strip()
            answer = (
                "Based on the stored analysis, here is the most relevant context: "
                + context
                if context
                else "I could not find relevant analysis context yet. Train the system first and try again."
            )

        return {
            "answer": answer,
            "sources": [
                {"score": round(chunk.score, 4), "section": chunk.metadata.get("section", "knowledge")}
                for chunk in retrieved_chunks
            ],
        }
