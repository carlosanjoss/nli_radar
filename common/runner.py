from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # permite execução funcional sem barra de progresso
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, **_: Any):
            self.total = total

        def update(self, _: int) -> None:
            return None

        def close(self) -> None:
            return None

from common.aggregation import aggregate_hypothesis_scores
from common.config import load_yaml, project_root_from_config, resolve_project_path
from common.data import LoadedInput, load_input_csv
from common.hypotheses import Hypothesis, load_hypotheses
from common.reporting import write_descriptive_reports, write_json

if TYPE_CHECKING:
    from common.nli import NLIModel


Finalizer = Callable[[str, dict[str, float], dict[str, Any]], dict[str, Any]]
BatchCallback = Callable[
    [pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]],
    None,
]


DETAIL_FIELDS = [
    "row_id",
    "source_id",
    "dimension_code",
    "dimension_name",
    "construct_code",
    "construct_name",
    "hypothesis_code",
    "hypothesis_kind",
    "hypothesis_text",
    "entailment_probability_3way",
    "neutral_probability_3way",
    "contradiction_probability_3way",
    "label_probability",
    "inference_time_seconds",
]


def _threshold_for_dimension(config: dict[str, Any], code: str) -> float:
    thresholds = config.get("thresholds", {})
    by_dimension = thresholds.get("dimensions", {})
    return float(by_dimension.get(code, thresholds.get("default", 0.65)))


def _default_finalizer(
    _text: str,
    dimension_scores: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    active = [
        code
        for code, score in dimension_scores.items()
        if score >= _threshold_for_dimension(config, code)
    ]
    primary = max(dimension_scores, key=dimension_scores.get) if dimension_scores else ""
    return {
        "primary_dimension": primary,
        "primary_score": dimension_scores.get(primary, 0.0),
        "active_dimensions": "|".join(active),
        "active_dimension_count": len(active),
    }


def _prepare_pair_batch(
    chunk: pd.DataFrame,
    hypotheses: list[Hypothesis],
) -> tuple[list[str], list[str], list[tuple[int, Hypothesis]]]:
    premises: list[str] = []
    hypothesis_texts: list[str] = []
    metadata: list[tuple[int, Hypothesis]] = []
    for local_index, row in chunk.iterrows():
        premise = str(row["_text"])
        for hypothesis in hypotheses:
            premises.append(premise)
            hypothesis_texts.append(hypothesis.text)
            metadata.append((local_index, hypothesis))
    return premises, hypothesis_texts, metadata


def run_classifier(
    layer_name: str,
    config_path: Path,
    input_override: Path | None = None,
    output_override: Path | None = None,
    text_column_override: str | None = None,
    id_column_override: str | None = None,
    sample_limit_override: int | None = None,
    gate_column_override: str | None = None,
    overwrite: bool = False,
    finalizer: Finalizer | None = None,
    nli_model: "NLIModel | None" = None,
    loaded_input_override: LoadedInput | None = None,
    input_source_description: str | None = None,
    batch_callback: BatchCallback | None = None,
    show_progress: bool = True,
) -> tuple[Path, "NLIModel"]:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    project_root = project_root_from_config(config_path)

    hypotheses_path = resolve_project_path(project_root, config["hypotheses_file"])
    hypotheses = load_hypotheses(hypotheses_path)
    dimension_codes = list(dict.fromkeys(h.dimension_code for h in hypotheses))

    input_config = config.get("input", {})
    input_path = input_override or resolve_project_path(project_root, input_config["path"])
    output_dir = output_override or resolve_project_path(
        project_root, config.get("output_dir", f"outputs/{layer_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"{layer_name}_predictions.csv"
    detail_path = output_dir / f"{layer_name}_hypothesis_scores.csv"
    if not overwrite and (predictions_path.exists() or detail_path.exists()):
        raise FileExistsError(
            f"Já existem saídas em {output_dir}. Use --overwrite para substituí-las."
        )

    gate_config = input_config.get("gate", {}) or {}
    loaded: LoadedInput
    if loaded_input_override is not None:
        loaded = loaded_input_override
        if loaded.dataframe.empty:
            raise ValueError("Nenhum comentário foi fornecido para a inferência NLI.")
    else:
        loaded = load_input_csv(
            path=input_path,
            text_column=text_column_override or input_config.get("text_column", "auto"),
            id_column=id_column_override or input_config.get("id_column", "auto"),
            sample_limit=(
                sample_limit_override
                if sample_limit_override is not None
                else input_config.get("sample_limit")
            ),
            gate_column=gate_column_override or gate_config.get("column"),
            gate_positive_values=gate_config.get(
                "positive_values", [1, True, "hate", "sim", "positivo"]
            ),
        )

    if nli_model is None:
        from common.nli import NLIModel

        nli_model = NLIModel(config["model"])

    comment_batch_size = int(config.get("runtime", {}).get("comment_batch_size", 16))
    finalizer = finalizer or _default_finalizer
    started = perf_counter()
    final_records: list[dict[str, Any]] = []

    with detail_path.open("w", encoding="utf-8-sig", newline="") as detail_file:
        writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
        writer.writeheader()

        dataframe = loaded.dataframe
        progress = tqdm(
            total=len(dataframe),
            desc=f"{layer_name}: comentários",
            unit="comentário",
            disable=not show_progress,
        )
        try:
            for start in range(0, len(dataframe), comment_batch_size):
                chunk = dataframe.iloc[start : start + comment_batch_size]
                premises, hypothesis_texts, metadata = _prepare_pair_batch(
                    chunk, hypotheses
                )
                scores = nli_model.score_pairs(premises, hypothesis_texts)
                batch_detail_rows: list[dict[str, Any]] = []
                batch_records: list[dict[str, Any]] = []

                rows_by_index: dict[int, list[dict[str, Any]]] = {
                    int(index): [] for index in chunk.index
                }
                for (local_index, hypothesis), score in zip(
                    metadata, scores, strict=True
                ):
                    source_row = dataframe.loc[local_index]
                    detail_row = {
                        "row_id": int(source_row["_row_id"]),
                        "source_id": source_row["_source_id"],
                        "dimension_code": hypothesis.dimension_code,
                        "dimension_name": hypothesis.dimension_name,
                        "construct_code": hypothesis.construct_code,
                        "construct_name": hypothesis.construct_name,
                        "hypothesis_code": hypothesis.hypothesis_code,
                        "hypothesis_kind": hypothesis.hypothesis_kind,
                        "hypothesis_text": hypothesis.text,
                        "entailment_probability_3way": (
                            score.entailment_probability_3way
                        ),
                        "neutral_probability_3way": score.neutral_probability_3way,
                        "contradiction_probability_3way": (
                            score.contradiction_probability_3way
                        ),
                        "label_probability": score.label_probability,
                        "inference_time_seconds": score.inference_time_seconds,
                    }
                    writer.writerow(detail_row)
                    batch_detail_rows.append(detail_row)
                    rows_by_index[int(local_index)].append(detail_row)

                for local_index, row in chunk.iterrows():
                    dimension_scores, construct_scores = aggregate_hypothesis_scores(
                        rows_by_index[int(local_index)], config.get("aggregation", {})
                    )
                    extra = finalizer(str(row["_text"]), dimension_scores, config)
                    record: dict[str, Any] = {
                        "row_id": int(row["_row_id"]),
                        "source_row_index": int(row["_source_row_index"]),
                        "source_id": row["_source_id"],
                        "text": row["_text"],
                    }
                    for code in dimension_codes:
                        score = float(dimension_scores.get(code, 0.0))
                        record[f"{code}_score"] = score
                        record[f"{code}_active"] = (
                            score >= _threshold_for_dimension(config, code)
                        )
                    for key, score in construct_scores.items():
                        record[f"construct_{key}_score"] = float(score)
                    record.update(extra)
                    batch_records.append(record)

                # O callback PostgreSQL confirma o lote antes de avançar. Se a
                # transação falhar, a inferência para aqui e o lote inteiro é
                # retomado na próxima execução.
                if batch_callback is not None:
                    batch_callback(chunk, batch_records, batch_detail_rows)

                final_records.extend(batch_records)
                detail_file.flush()
                progress.update(len(chunk))
        finally:
            progress.close()

    predictions = pd.DataFrame(final_records)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    reports = write_descriptive_reports(
        predictions, output_dir, layer_name, dimension_codes
    )

    metadata_path = output_dir / f"{layer_name}_run_metadata.json"
    write_json(
        metadata_path,
        {
            "layer": layer_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": nli_model.model_name,
            "device": str(nli_model.device),
            "score_mode": nli_model.score_mode,
            "input_source": input_source_description or str(input_path),
            "input_path": None if loaded_input_override is not None else str(input_path),
            "input_encoding": loaded.encoding,
            "input_text_column": loaded.text_column,
            "input_id_column": loaded.id_column,
            "original_rows": loaded.original_rows,
            "processed_rows": len(predictions),
            "empty_rows_removed": loaded.empty_rows_removed,
            "gate_rows_removed": loaded.gate_rows_removed,
            "hypothesis_count": len(hypotheses),
            "dimension_codes": dimension_codes,
            "elapsed_seconds": perf_counter() - started,
            "outputs": {
                "predictions": str(predictions_path),
                "hypothesis_scores": str(detail_path),
                **{key: str(value) for key, value in reports.items()},
            },
            "warning": (
                "Sem coluna de gabarito, estes relatórios são descritivos; "
                "não representam acurácia, precisão, recall ou F1."
            ),
        },
    )
    return predictions_path, nli_model
