from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def weighted_max_mean(values: list[float], max_weight: float, mean_weight: float) -> float:
    if not values:
        return 0.0
    if abs((max_weight + mean_weight) - 1.0) > 1e-6:
        raise ValueError("Os pesos max_weight e mean_weight precisam somar 1.0.")
    return (max_weight * max(values)) + (mean_weight * mean(values))


def aggregate_hypothesis_scores(
    rows: list[dict[str, Any]],
    aggregation_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    max_weight = float(aggregation_config.get("max_weight", 0.70))
    mean_weight = float(aggregation_config.get("mean_weight", 0.30))
    counter_weight = float(aggregation_config.get("counter_weight", 0.50))
    construct_max_weight = float(
        aggregation_config.get("construct_max_weight", max_weight)
    )
    construct_mean_weight = float(
        aggregation_config.get("construct_mean_weight", mean_weight)
    )

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"positive": [], "negative": [], "neutral": []}
    )
    for row in rows:
        key = (row["dimension_code"], row["construct_code"])
        grouped[key][row["hypothesis_kind"]].append(float(row["label_probability"]))

    construct_scores: dict[str, float] = {}
    by_dimension: dict[str, list[float]] = defaultdict(list)
    for (dimension_code, construct_code), kind_scores in grouped.items():
        positive_raw = weighted_max_mean(
            kind_scores["positive"], max_weight, mean_weight
        )
        counter_candidates = kind_scores["negative"] + kind_scores["neutral"]
        counter_score = max(counter_candidates) if counter_candidates else 0.0
        adjusted = max(0.0, min(1.0, positive_raw - (counter_weight * counter_score)))
        key = f"{dimension_code}.{construct_code}"
        construct_scores[key] = adjusted
        by_dimension[dimension_code].append(adjusted)

    dimension_scores = {
        code: weighted_max_mean(values, construct_max_weight, construct_mean_weight)
        for code, values in by_dimension.items()
    }
    return dimension_scores, construct_scores

