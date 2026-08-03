from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import psycopg
from dotenv import load_dotenv
from psycopg import Connection
from psycopg.rows import dict_row

from common.data import LoadedInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = PROJECT_ROOT / "sql"
VALID_LAYERS = {"layer3", "layer4"}


def load_postgres_uri(explicit_uri: str | None = None) -> str:
    """Obtém a mesma conexão usada pelo classificador Gemma."""
    if explicit_uri:
        return explicit_uri

    for path in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / "gemma_hate_inference" / ".env",
        PROJECT_ROOT.parent / "social_data_pipeline" / ".env",
    ):
        load_dotenv(path, override=False)

    uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")
    if not uri:
        raise RuntimeError(
            "POSTGRES_URI não foi definida. Configure-a no ambiente, em .env, "
            "em ../gemma_hate_inference/.env ou em ../social_data_pipeline/.env."
        )
    return uri


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classifier_config_sha256(config: dict[str, Any]) -> str:
    """Versiona somente as opções que alteram o resultado da classificação."""
    ignored = {"input", "output_dir", "runtime", "hypotheses_file"}
    relevant = {key: value for key, value in config.items() if key not in ignored}
    serialized = json.dumps(
        relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_run_name(layer_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"radar_nli_{layer_name}_{timestamp}"


def _result_table(layer_name: str) -> str:
    if layer_name not in VALID_LAYERS:
        raise ValueError(f"Camada inválida: {layer_name}")
    return f"radar_nli_{layer_name}_results"


def _score_table(layer_name: str) -> str:
    if layer_name not in VALID_LAYERS:
        raise ValueError(f"Camada inválida: {layer_name}")
    return f"radar_nli_{layer_name}_hypothesis_scores"


class PostgresStore:
    def __init__(self, uri: str):
        self.conn: Connection = psycopg.connect(uri, row_factory=dict_row)

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not migrations:
            raise FileNotFoundError(f"Nenhuma migração encontrada em {MIGRATIONS_DIR}")
        try:
            for migration in migrations:
                self.conn.execute(migration.read_text(encoding="utf-8"))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def fetch_pending_hate_comments(
        self,
        *,
        layer_name: str,
        nli_model_name: str,
        hypotheses_hash: str,
        config_hash: str,
        platforms: Sequence[str] = (),
        gemma_model: str | None = None,
        gemma_prompt_version: str | None = None,
        limit: int | None = None,
        content_kinds: Sequence[str] = ("comment",),
    ) -> LoadedInput:
        result_table = _result_table(layer_name)
        if limit is not None and limit <= 0:
            raise ValueError("limit precisa ser maior que zero.")
        invalid_kinds = set(content_kinds) - {"comment", "post"}
        if not content_kinds or invalid_kinds:
            raise ValueError(
                "content_kinds deve conter apenas 'comment' e/ou 'post'."
            )

        filters = [
            "a.content_kind = ANY(%s)",
            "a.status = 'success'",
            "(a.comment_id IS NOT NULL OR a.post_id IS NOT NULL)",
        ]
        params: list[object] = [list(content_kinds)]
        if platforms:
            filters.append("a.platform = ANY(%s)")
            params.append(list(platforms))
        if gemma_model:
            filters.append("a.model_name = %s")
            params.append(gemma_model)
        if gemma_prompt_version:
            filters.append("a.prompt_version = %s")
            params.append(gemma_prompt_version)

        params.extend([nli_model_name, hypotheses_hash, config_hash, limit])
        sql = f"""
            WITH ranked AS (
                SELECT
                    a.analysis_id,
                    a.content_kind,
                    a.post_id,
                    a.comment_id,
                    a.platform,
                    a.external_id,
                    a.text_hash,
                    a.text_snapshot,
                    a.model_name AS gemma_model_name,
                    a.prompt_version AS gemma_prompt_version,
                    a.hate_probability,
                    a.is_hate,
                    a.analyzed_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.content_kind, COALESCE(a.comment_id, a.post_id)
                        ORDER BY a.analyzed_at DESC, a.analysis_id DESC
                    ) AS position
                FROM gemma_hate_analyses a
                WHERE {' AND '.join(filters)}
            ),
            hate_candidates AS (
                SELECT *
                FROM ranked
                WHERE position = 1
                  AND is_hate IS TRUE
                  AND NULLIF(BTRIM(text_snapshot), '') IS NOT NULL
            ),
            deduplicated AS (
                SELECT
                    candidate.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY candidate.platform, candidate.text_hash
                        ORDER BY candidate.comment_id, candidate.analysis_id
                    ) AS content_position
                FROM hate_candidates candidate
            )
            SELECT
                r.analysis_id,
                r.content_kind,
                r.post_id,
                r.comment_id,
                COALESCE(r.comment_id, r.post_id) AS source_content_id,
                r.platform,
                r.external_id,
                r.text_hash,
                r.text_snapshot,
                r.gemma_model_name,
                r.gemma_prompt_version,
                r.hate_probability
            FROM deduplicated r
            WHERE r.content_position = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM {result_table} nli
                  WHERE nli.platform = r.platform
                    AND nli.text_hash = r.text_hash
                    AND nli.nli_model_name = %s
                    AND nli.hypotheses_hash = %s
                    AND nli.config_hash = %s
              )
            ORDER BY r.comment_id
            LIMIT COALESCE(%s, 2147483647)
        """
        rows = self.conn.execute(sql, params).fetchall()
        dataframe = pd.DataFrame([dict(row) for row in rows])
        if dataframe.empty:
            dataframe = pd.DataFrame(
                columns=[
                    "analysis_id",
                    "content_kind",
                    "post_id",
                    "comment_id",
                    "source_content_id",
                    "platform",
                    "external_id",
                    "text_hash",
                    "text_snapshot",
                    "gemma_model_name",
                    "gemma_prompt_version",
                    "hate_probability",
                ]
            )

        dataframe["_text"] = dataframe["text_snapshot"].astype(str)
        dataframe["_source_id"] = (
            dataframe["content_kind"].astype(str)
            + ":"
            + dataframe["source_content_id"].astype(str)
        )
        dataframe["_source_row_index"] = range(len(dataframe))
        dataframe["_row_id"] = range(1, len(dataframe) + 1)
        return LoadedInput(
            dataframe=dataframe,
            text_column="text_snapshot",
            id_column="source_content_id",
            original_rows=len(dataframe),
            empty_rows_removed=0,
            gate_rows_removed=0,
            encoding="PostgreSQL UTF-8",
        )

    def create_run(
        self,
        *,
        layer_name: str,
        nli_model_name: str,
        hypotheses_hash: str,
        config_hash: str,
        total_selected: int,
        platforms: Sequence[str] = (),
        gemma_model: str | None = None,
        gemma_prompt_version: str | None = None,
    ) -> int:
        if layer_name not in VALID_LAYERS:
            raise ValueError(f"Camada inválida: {layer_name}")
        row = self.conn.execute(
            """
            INSERT INTO radar_nli_runs (
                run_name, layer_name, nli_model_name, hypotheses_hash, config_hash,
                source_gemma_model, source_gemma_prompt_version, source_platforms,
                total_selected
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (
                make_run_name(layer_name),
                layer_name,
                nli_model_name,
                hypotheses_hash,
                config_hash,
                gemma_model,
                gemma_prompt_version,
                list(platforms),
                total_selected,
            ),
        ).fetchone()
        self.conn.commit()
        return int(row["run_id"])

    def finish_run(self, run_id: int, processed_count: int) -> None:
        self.conn.execute(
            """
            UPDATE radar_nli_runs
            SET status = 'completed', processed_count = %s, finished_at = NOW()
            WHERE run_id = %s
            """,
            (processed_count, run_id),
        )
        self.conn.commit()

    def fail_run(self, run_id: int, error_message: str) -> None:
        self.conn.rollback()
        self.conn.execute(
            """
            UPDATE radar_nli_runs
            SET status = 'failed', finished_at = NOW(), error_message = %s
            WHERE run_id = %s
            """,
            (error_message[:4000], run_id),
        )
        self.conn.commit()

    def save_layer_batch(
        self,
        *,
        layer_name: str,
        run_id: int,
        source_chunk: pd.DataFrame,
        predictions: list[dict[str, Any]],
        hypothesis_scores: list[dict[str, Any]],
        dimension_codes: Iterable[str],
        nli_model_name: str,
        hypotheses_hash: str,
        config_hash: str,
    ) -> int:
        """Persiste um lote completo e o progresso na mesma transação."""
        result_table = _result_table(layer_name)
        score_table = _score_table(layer_name)
        dimensions = list(dimension_codes)
        source_metadata = {
            str(row["_source_id"]): row
            for row in source_chunk.to_dict(orient="records")
        }
        result_ids: dict[str, int] = {}

        try:
            for prediction in predictions:
                source_id = str(prediction["source_id"])
                source = source_metadata[source_id]
                dimension_scores = {
                    code: float(prediction[f"{code}_score"]) for code in dimensions
                }
                construct_scores = {
                    key[len("construct_") : -len("_score")]: float(value)
                    for key, value in prediction.items()
                    if key.startswith("construct_")
                    and key.endswith("_score")
                    and value not in (None, "")
                }
                active_dimensions = [
                    value
                    for value in str(
                        prediction.get("active_dimensions", "")
                    ).split("|")
                    if value
                ]
                common_values: tuple[object, ...] = (
                    run_id,
                    int(source["analysis_id"]),
                    str(source["content_kind"]),
                    _optional_int(source.get("post_id")),
                    _optional_int(source.get("comment_id")),
                    str(source["platform"]),
                    str(source["external_id"]),
                    str(source["text_hash"]),
                    str(source["text_snapshot"]),
                    nli_model_name,
                    hypotheses_hash,
                    config_hash,
                    prediction.get("primary_dimension") or None,
                    _optional_float(prediction.get("primary_score")),
                    active_dimensions,
                    json.dumps(dimension_scores, ensure_ascii=False),
                    json.dumps(construct_scores, ensure_ascii=False),
                )
                if layer_name == "layer3":
                    result_sql = f"""
                        INSERT INTO {result_table} (
                            run_id, source_gemma_analysis_id, content_kind, post_id,
                            comment_id, platform, external_id, text_hash, text_snapshot,
                            nli_model_name, hypotheses_hash, config_hash,
                            primary_dimension, primary_score, active_dimensions,
                            dimension_scores, construct_scores, toxicity_score,
                            toxicity_label
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s
                        )
                        ON CONFLICT (
                            platform, text_hash, nli_model_name,
                            hypotheses_hash, config_hash
                        ) DO NOTHING
                        RETURNING result_id
                    """
                    result_values = common_values + (
                        float(prediction["toxicity_score"]),
                        str(prediction["toxicity_label"]),
                    )
                else:
                    result_sql = f"""
                        INSERT INTO {result_table} (
                            run_id, source_gemma_analysis_id, content_kind, post_id,
                            comment_id, platform, external_id, text_hash, text_snapshot,
                            nli_model_name, hypotheses_hash, config_hash,
                            primary_dimension, primary_score, active_dimensions,
                            dimension_scores, construct_scores
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
                        )
                        ON CONFLICT (
                            platform, text_hash, nli_model_name,
                            hypotheses_hash, config_hash
                        ) DO NOTHING
                        RETURNING result_id
                    """
                    result_values = common_values

                inserted = self.conn.execute(
                    result_sql, result_values
                ).fetchone()
                if inserted:
                    result_ids[source_id] = int(inserted["result_id"])

            score_sql = f"""
                INSERT INTO {score_table} (
                    result_id, dimension_code, dimension_name, construct_code,
                    construct_name, hypothesis_code, hypothesis_kind,
                    hypothesis_text, entailment_probability_3way,
                    neutral_probability_3way, contradiction_probability_3way,
                    label_probability, inference_time_seconds
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (
                    result_id, dimension_code, construct_code, hypothesis_code
                ) DO NOTHING
            """
            score_rows: list[tuple[object, ...]] = []
            for score in hypothesis_scores:
                result_id = result_ids.get(str(score["source_id"]))
                if result_id is None:
                    continue
                score_rows.append(
                    (
                        result_id,
                        score["dimension_code"],
                        score["dimension_name"],
                        score["construct_code"],
                        score["construct_name"],
                        score["hypothesis_code"],
                        score["hypothesis_kind"],
                        score["hypothesis_text"],
                        float(score["entailment_probability_3way"]),
                        float(score["neutral_probability_3way"]),
                        float(score["contradiction_probability_3way"]),
                        float(score["label_probability"]),
                        _optional_float(score.get("inference_time_seconds")),
                    )
                )
            if score_rows:
                with self.conn.cursor() as cursor:
                    cursor.executemany(score_sql, score_rows)

            inserted_count = len(result_ids)
            progress_cursor = self.conn.execute(
                """
                UPDATE radar_nli_runs
                SET processed_count = processed_count + %s
                WHERE run_id = %s AND status = 'running'
                """,
                (inserted_count, run_id),
            )
            if progress_cursor.rowcount != 1:
                raise RuntimeError(
                    f"Run PostgreSQL {run_id} não está disponível para progresso."
                )
            self.conn.commit()
            return inserted_count
        except Exception:
            self.conn.rollback()
            raise

    def save_layer_outputs(
        self,
        *,
        layer_name: str,
        run_id: int,
        loaded: LoadedInput,
        predictions_path: Path,
        hypothesis_scores_path: Path,
        dimension_codes: Iterable[str],
        nli_model_name: str,
        hypotheses_hash: str,
        config_hash: str,
    ) -> int:
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
            predictions = list(csv.DictReader(file))
        with hypothesis_scores_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            hypothesis_scores = list(csv.DictReader(file))
        return self.save_layer_batch(
            layer_name=layer_name,
            run_id=run_id,
            source_chunk=loaded.dataframe,
            predictions=predictions,
            hypothesis_scores=hypothesis_scores,
            dimension_codes=dimension_codes,
            nli_model_name=nli_model_name,
            hypotheses_hash=hypotheses_hash,
            config_hash=config_hash,
        )


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
