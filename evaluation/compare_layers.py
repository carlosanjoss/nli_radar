from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.reporting import write_json


LAYER4_DIMENSIONS = [
    "dehumanization",
    "authoritarianism",
    "revictimization",
    "factualization",
    "masculinism",
]


def compare_layer_outputs(
    layer3_path: Path,
    layer4_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    layer3 = pd.read_csv(layer3_path)
    layer4 = pd.read_csv(layer4_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer3_columns = [
        "row_id",
        "source_id",
        "text",
        "toxicity_score",
        "toxicity_label",
        "dehumanization_score",
    ]
    layer4_columns = ["row_id", "primary_dimension", "primary_score", "active_dimensions"]
    for code in LAYER4_DIMENSIONS:
        layer4_columns.extend([f"{code}_score", f"{code}_active"])

    merged = layer3[layer3_columns].merge(
        layer4[layer4_columns],
        on="row_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_layer3", "_layer4"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        missing = merged.loc[merged["_merge"] != "both", ["row_id", "_merge"]]
        raise ValueError(
            "As saídas das camadas não possuem as mesmas linhas. Exemplos: "
            + missing.head(10).to_dict(orient="records").__repr__()
        )
    merged.drop(columns="_merge", inplace=True)

    merged["review_layer4_active_layer3_nontoxic"] = (
        merged["active_dimensions"].fillna("").astype(str).str.len().gt(0)
        & merged["toxicity_label"].eq("nao_toxico")
    )
    merged["review_layer3_severe_without_layer4_dimension"] = (
        merged["toxicity_label"].isin(["media", "alta"])
        & merged["active_dimensions"].fillna("").astype(str).str.len().eq(0)
    )
    merged["dehumanization_score_difference"] = (
        merged["dehumanization_score_layer4"]
        - merged["dehumanization_score_layer3"]
    ).abs()
    merged["review_dehumanization_disagreement"] = (
        merged["dehumanization_score_difference"] >= 0.35
    )
    review_columns = [
        "review_layer4_active_layer3_nontoxic",
        "review_layer3_severe_without_layer4_dimension",
        "review_dehumanization_disagreement",
    ]
    merged["needs_manual_review"] = merged[review_columns].any(axis=1)

    comparison_path = output_dir / "cross_layer_verification.csv"
    merged.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "cross_layer_verification_summary.json"
    write_json(
        summary_path,
        {
            "total_rows": len(merged),
            "manual_review_candidates": int(merged["needs_manual_review"].sum()),
            "layer4_active_layer3_nontoxic": int(
                merged["review_layer4_active_layer3_nontoxic"].sum()
            ),
            "layer3_severe_without_layer4_dimension": int(
                merged["review_layer3_severe_without_layer4_dimension"].sum()
            ),
            "dehumanization_disagreement_ge_0_35": int(
                merged["review_dehumanization_disagreement"].sum()
            ),
            "interpretation": (
                "Os casos sinalizados não são erros comprovados. Sem gabarito humano, "
                "eles são candidatos prioritários para inspeção e anotação manual."
            ),
        },
    )
    return comparison_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara descritivamente as camadas 3 e 4.")
    parser.add_argument("--layer3", type=Path, required=True)
    parser.add_argument("--layer4", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    comparison, summary = compare_layer_outputs(args.layer3, args.layer4, args.output_dir)
    print(f"Verificação por comentário: {comparison}")
    print(f"Síntese da verificação: {summary}")


if __name__ == "__main__":
    main()

