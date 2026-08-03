from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_descriptive_reports(
    predictions: pd.DataFrame,
    output_dir: Path,
    layer_name: str,
    dimension_codes: list[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(predictions)
    for code in dimension_codes:
        score_column = f"{code}_score"
        active_column = f"{code}_active"
        series = predictions[score_column].astype(float)
        rows.append(
            {
                "dimension_code": code,
                "total_comments": total,
                "active_count": int(predictions[active_column].sum()),
                "active_percentage": (
                    float(predictions[active_column].mean() * 100) if total else 0.0
                ),
                "mean_score": float(series.mean()) if total else 0.0,
                "median_score": float(series.median()) if total else 0.0,
                "p95_score": float(series.quantile(0.95)) if total else 0.0,
                "max_score": float(series.max()) if total else 0.0,
            }
        )
    summary_path = output_dir / f"{layer_name}_dimension_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    primary_path = output_dir / f"{layer_name}_primary_distribution.csv"
    (
        predictions["primary_dimension"]
        .value_counts(dropna=False)
        .rename_axis("primary_dimension")
        .reset_index(name="count")
        .assign(percentage=lambda frame: frame["count"] / max(total, 1) * 100)
        .to_csv(primary_path, index=False, encoding="utf-8-sig")
    )
    return {"dimension_summary": summary_path, "primary_distribution": primary_path}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
