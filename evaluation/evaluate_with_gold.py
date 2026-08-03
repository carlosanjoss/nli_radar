from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from common.reporting import write_json


TRUE_VALUES = {"1", "true", "sim", "yes", "positivo", "active", "ativa", "hate"}


def _as_binary(series: pd.Series) -> pd.Series:
    return series.map(lambda value: int(str(value).strip().lower() in TRUE_VALUES))


def evaluate_dimension(
    predictions_path: Path,
    gold_path: Path,
    prediction_id_column: str,
    gold_id_column: str,
    predicted_column: str,
    gold_column: str,
    output_path: Path,
) -> dict[str, object]:
    predictions = pd.read_csv(predictions_path)
    gold = pd.read_csv(gold_path)
    merged = predictions[[prediction_id_column, predicted_column]].merge(
        gold[[gold_id_column, gold_column]],
        left_on=prediction_id_column,
        right_on=gold_id_column,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Nenhuma linha foi associada entre as previsões e o gabarito.")

    y_pred = _as_binary(merged[predicted_column])
    y_true = _as_binary(merged[gold_column])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result: dict[str, object] = {
        "matched_rows": len(merged),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula métricas binárias quando existir gabarito humano."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--prediction-id-column", default="source_id")
    parser.add_argument("--gold-id-column", required=True)
    parser.add_argument("--predicted-column", required=True)
    parser.add_argument("--gold-column", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/gold_metrics.json"))
    args = parser.parse_args()
    result = evaluate_dimension(
        args.predictions,
        args.gold,
        args.prediction_id_column,
        args.gold_id_column,
        args.predicted_column,
        args.gold_column,
        args.output,
    )
    print(result)


if __name__ == "__main__":
    main()

