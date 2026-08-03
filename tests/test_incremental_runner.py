from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from common.data import LoadedInput
from common.runner import run_classifier
from layer3.postprocessing import finalize_layer3


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Score:
    entailment_probability_3way: float = 0.8
    neutral_probability_3way: float = 0.1
    contradiction_probability_3way: float = 0.1
    label_probability: float = 0.88
    inference_time_seconds: float = 0.001


class _FakeNLIModel:
    model_name = "fake-nli"
    device = "cpu"
    score_mode = "entailment_vs_contradiction"

    def score_pairs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[_Score]:
        assert len(premises) == len(hypotheses)
        return [_Score() for _ in premises]


def _loaded_input() -> LoadedInput:
    dataframe = pd.DataFrame(
        [
            {
                "analysis_id": 101,
                "comment_id": 1,
                "platform": "reddit",
                "external_id": "a",
                "text_hash": "h1",
                "text_snapshot": "comentário hostil um",
            },
            {
                "analysis_id": 102,
                "comment_id": 2,
                "platform": "reddit",
                "external_id": "b",
                "text_hash": "h2",
                "text_snapshot": "comentário hostil dois",
            },
            {
                "analysis_id": 103,
                "comment_id": 3,
                "platform": "reddit",
                "external_id": "c",
                "text_hash": "h3",
                "text_snapshot": "comentário hostil três",
            },
        ]
    )
    dataframe["_text"] = dataframe["text_snapshot"]
    dataframe["_source_id"] = dataframe["comment_id"].astype(str)
    dataframe["_source_row_index"] = range(len(dataframe))
    dataframe["_row_id"] = range(1, len(dataframe) + 1)
    return LoadedInput(
        dataframe=dataframe,
        text_column="text_snapshot",
        id_column="comment_id",
        original_rows=3,
        empty_rows_removed=0,
        gate_rows_removed=0,
        encoding="PostgreSQL UTF-8",
    )


def test_runner_calls_persistence_after_each_batch(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "layer3" / "config.yaml").read_text(encoding="utf-8")
    )
    config["hypotheses_file"] = str(
        (ROOT / "layer3" / "hypotheses.yaml").resolve()
    )
    config["runtime"]["comment_batch_size"] = 2
    config_path = tmp_path / "layer3" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    batches: list[tuple[int, int, int]] = []

    def persist(
        source_chunk: pd.DataFrame,
        predictions: list[dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> None:
        batches.append((len(source_chunk), len(predictions), len(scores)))

    run_classifier(
        layer_name="layer3",
        config_path=config_path,
        output_override=tmp_path / "outputs",
        finalizer=finalize_layer3,
        nli_model=_FakeNLIModel(),  # type: ignore[arg-type]
        loaded_input_override=_loaded_input(),
        batch_callback=persist,
        show_progress=False,
    )

    assert batches == [(2, 2, 80), (1, 1, 40)]
