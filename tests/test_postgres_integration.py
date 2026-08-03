from __future__ import annotations

from pathlib import Path

from common.postgres import PostgresStore, classifier_config_sha256


ROOT = Path(__file__).resolve().parents[1]


class _Rows:
    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows
        self.sql = ""
        self.params: list[object] = []

    def execute(self, sql: str, params: list[object]) -> _Rows:
        self.sql = sql
        self.params = params
        return _Rows(self.rows)


def test_postgres_selection_uses_latest_successful_gemma_hate() -> None:
    fake = _FakeConnection(
        [
            {
                "analysis_id": 91,
                "content_kind": "comment",
                "post_id": None,
                "comment_id": 12,
                "source_content_id": 12,
                "platform": "reddit",
                "external_id": "abc",
                "text_hash": "hash",
                "text_snapshot": "comentário de teste",
                "gemma_model_name": "gemma4:e4b",
                "gemma_prompt_version": "v1",
                "hate_probability": 0.9,
            }
        ]
    )
    store = PostgresStore.__new__(PostgresStore)
    store.conn = fake  # type: ignore[assignment]

    loaded = store.fetch_pending_hate_comments(
        layer_name="layer3",
        nli_model_name="nli-model",
        hypotheses_hash="hyp-hash",
        config_hash="config-hash",
        platforms=["reddit"],
    )

    normalized_sql = " ".join(fake.sql.split())
    assert "ROW_NUMBER() OVER" in normalized_sql
    assert "a.status = 'success'" in normalized_sql
    assert "WHERE position = 1" in normalized_sql
    assert "is_hate IS TRUE" in normalized_sql
    assert "PARTITION BY candidate.platform, candidate.text_hash" in normalized_sql
    assert "r.content_position = 1" in normalized_sql
    assert "FROM radar_nli_layer3_results nli" in normalized_sql
    assert "a.content_kind = ANY(%s)" in normalized_sql
    assert loaded.dataframe.loc[0, "_source_id"] == "comment:12"
    assert loaded.dataframe.loc[0, "_text"] == "comentário de teste"


def test_classifier_hash_ignores_io_but_tracks_thresholds() -> None:
    base = {
        "model": {"name": "model"},
        "thresholds": {"default": 0.5},
        "input": {"path": "a.csv"},
        "output_dir": "outputs/a",
        "runtime": {"comment_batch_size": 1},
    }
    same_classifier = {
        **base,
        "input": {"path": "b.csv"},
        "output_dir": "outputs/b",
        "runtime": {"comment_batch_size": 99},
    }
    changed_classifier = {**base, "thresholds": {"default": 0.7}}

    assert classifier_config_sha256(base) == classifier_config_sha256(same_classifier)
    assert classifier_config_sha256(base) != classifier_config_sha256(changed_classifier)


def test_post_selection_uses_a_distinct_source_id() -> None:
    fake = _FakeConnection(
        [
            {
                "analysis_id": 92,
                "content_kind": "post",
                "post_id": 99,
                "comment_id": None,
                "source_content_id": 99,
                "platform": "whatsapp",
                "external_id": "message-99",
                "text_hash": "post-hash",
                "text_snapshot": "mensagem de teste",
                "gemma_model_name": "gemma4:e4b",
                "gemma_prompt_version": "v1",
                "hate_probability": 0.9,
            }
        ]
    )
    store = PostgresStore.__new__(PostgresStore)
    store.conn = fake  # type: ignore[assignment]

    loaded = store.fetch_pending_hate_comments(
        layer_name="layer3",
        nli_model_name="nli-model",
        hypotheses_hash="hyp-hash",
        config_hash="config-hash",
        content_kinds=("post",),
    )

    assert loaded.dataframe.loc[0, "_source_id"] == "post:99"
    assert loaded.dataframe.loc[0, "content_kind"] == "post"
    assert loaded.dataframe.loc[0, "comment_id"] is None


def test_schema_keeps_layer3_and_layer4_in_separate_tables() -> None:
    schema = (ROOT / "sql" / "001_create_radar_nli_schema.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS radar_nli_layer3_results" in schema
    assert "CREATE TABLE IF NOT EXISTS radar_nli_layer3_hypothesis_scores" in schema
    assert "CREATE TABLE IF NOT EXISTS radar_nli_layer4_results" in schema
    assert "CREATE TABLE IF NOT EXISTS radar_nli_layer4_hypothesis_scores" in schema
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_radar_nli_layer3_content" in schema
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_radar_nli_layer4_content" in schema
