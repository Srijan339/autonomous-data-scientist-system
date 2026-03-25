from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class InsightGenerator:
    """Create structured findings and a knowledge document from pipeline artifacts."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        metrics: Dict[str, float],
        feature_importance: List[Dict[str, float]],
        profile: Dict[str, Any],
        eda_summary: Dict[str, Any],
        output_dir: str,
        best_model: str,
    ):
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.metrics = metrics
        self.feature_importance = feature_importance
        self.profile = profile
        self.eda_summary = eda_summary
        self.best_model = best_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> Dict[str, Any]:
        insights = self._build_insights()
        knowledge_text = self._build_knowledge_text(insights)

        insights_path = self.output_dir / "insights.json"
        knowledge_path = self.output_dir / "knowledge.txt"

        insights_path.write_text(json.dumps(insights, indent=2, default=str), encoding="utf-8")
        knowledge_path.write_text(knowledge_text, encoding="utf-8")

        return {
            "insights": insights,
            "knowledge_text": knowledge_text,
            "insights_path": str(insights_path.resolve()),
            "knowledge_path": str(knowledge_path.resolve()),
        }

    def _build_insights(self) -> List[Dict[str, str]]:
        insights: List[Dict[str, str]] = []
        insights.append(
            {
                "title": "Detected learning objective",
                "detail": f"The system identified a {self.problem_type} problem for target '{self.target_column}' and selected {self.best_model} as the top model.",
            }
        )

        if self.metrics:
            metric_text = ", ".join(f"{key}={value}" for key, value in self.metrics.items())
            insights.append({"title": "Model quality", "detail": f"Holdout evaluation metrics: {metric_text}."})

        missing_values = self.eda_summary.get("missing_values", {})
        heavy_missing = [column for column, count in missing_values.items() if count and count / max(len(self.df), 1) > 0.2]
        if heavy_missing:
            insights.append(
                {
                    "title": "Data quality risk",
                    "detail": f"High-missing columns were detected: {', '.join(heavy_missing[:5])}. These fields may require stronger domain treatment.",
                }
            )

        if self.feature_importance:
            top_features = ", ".join(item["feature"] for item in self.feature_importance[:3])
            insights.append(
                {
                    "title": "Strongest drivers",
                    "detail": f"Top predictive signals from the winning model are {top_features}.",
                }
            )

        if self.problem_type == "classification":
            class_balance = self.df[self.target_column].value_counts(normalize=True, dropna=False).round(3).to_dict()
            dominant_class = max(class_balance, key=class_balance.get)
            insights.append(
                {
                    "title": "Target balance",
                    "detail": f"Class distribution shows '{dominant_class}' as the largest class at {class_balance[dominant_class]:.1%}.",
                }
            )
        else:
            target_series = self.df[self.target_column]
            insights.append(
                {
                    "title": "Target range",
                    "detail": f"The regression target spans from {target_series.min():.3f} to {target_series.max():.3f} with mean {target_series.mean():.3f}.",
                }
            )

        return insights

    def _build_knowledge_text(self, insights: List[Dict[str, str]]) -> str:
        lines = [
            "AUTOMATED DATA SCIENCE & ANALYTICS KNOWLEDGE BASE",
            "=" * 72,
            f"Target Column: {self.target_column}",
            f"Problem Type: {self.problem_type}",
            f"Best Model: {self.best_model}",
            "",
            "Metrics:",
        ]
        for key, value in self.metrics.items():
            lines.append(f"- {key}: {value}")

        lines.extend(["", "Top Feature Importance:"])
        for item in self.feature_importance[:10]:
            lines.append(f"- {item['feature']}: {item['importance']}")

        lines.extend(["", "Insights:"])
        for insight in insights:
            lines.append(f"- {insight['title']}: {insight['detail']}")

        lines.extend(["", "EDA Summary:"])
        lines.append(f"- Rows: {self.eda_summary.get('shape', {}).get('rows', 0)}")
        lines.append(f"- Columns: {self.eda_summary.get('shape', {}).get('columns', 0)}")

        return "\n".join(lines)
