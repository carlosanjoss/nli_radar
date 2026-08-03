-- Radar Ódio: resultados NLI das camadas 3 e 4.
--
-- A entrada é sempre uma classificação bem-sucedida e positiva do Gemma para
-- um comentário. As duas camadas permanecem fisicamente separadas para evitar
-- a mistura de hipóteses, dimensões e regras de agregação.

CREATE TABLE IF NOT EXISTS radar_nli_runs (
    run_id                       BIGSERIAL PRIMARY KEY,
    run_name                     TEXT NOT NULL UNIQUE,
    layer_name                   TEXT NOT NULL CHECK (layer_name IN ('layer3', 'layer4')),
    status                       TEXT NOT NULL DEFAULT 'running'
                                 CHECK (status IN ('running', 'completed', 'failed')),
    nli_model_name               TEXT NOT NULL,
    hypotheses_hash              VARCHAR(64) NOT NULL,
    config_hash                  VARCHAR(64) NOT NULL,
    source_gemma_model           TEXT,
    source_gemma_prompt_version  TEXT,
    source_platforms             TEXT[] NOT NULL DEFAULT '{}',
    total_selected               INTEGER NOT NULL DEFAULT 0 CHECK (total_selected >= 0),
    processed_count              INTEGER NOT NULL DEFAULT 0 CHECK (processed_count >= 0),
    started_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at                  TIMESTAMPTZ,
    error_message                TEXT
);

CREATE TABLE IF NOT EXISTS radar_nli_layer3_results (
    result_id                    BIGSERIAL PRIMARY KEY,
    run_id                       BIGINT NOT NULL REFERENCES radar_nli_runs(run_id) ON DELETE RESTRICT,
    source_gemma_analysis_id     BIGINT NOT NULL REFERENCES gemma_hate_analyses(analysis_id) ON DELETE RESTRICT,
    comment_id                   INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    platform                     VARCHAR(20) NOT NULL,
    external_id                  VARCHAR(255) NOT NULL,
    text_hash                    VARCHAR(64) NOT NULL,
    text_snapshot                TEXT NOT NULL,
    nli_model_name               TEXT NOT NULL,
    hypotheses_hash              VARCHAR(64) NOT NULL,
    config_hash                  VARCHAR(64) NOT NULL,
    primary_dimension            TEXT,
    primary_score                DOUBLE PRECISION,
    active_dimensions            TEXT[] NOT NULL DEFAULT '{}',
    dimension_scores             JSONB NOT NULL,
    construct_scores             JSONB NOT NULL,
    toxicity_score               DOUBLE PRECISION NOT NULL,
    toxicity_label               TEXT NOT NULL,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_radar_nli_layer3_result
        UNIQUE (comment_id, nli_model_name, hypotheses_hash, config_hash, text_hash),
    CONSTRAINT ck_radar_nli_layer3_primary_score
        CHECK (primary_score IS NULL OR (primary_score >= 0 AND primary_score <= 1)),
    CONSTRAINT ck_radar_nli_layer3_toxicity_score
        CHECK (toxicity_score >= 0 AND toxicity_score <= 1)
);

CREATE TABLE IF NOT EXISTS radar_nli_layer3_hypothesis_scores (
    score_id                      BIGSERIAL PRIMARY KEY,
    result_id                     BIGINT NOT NULL REFERENCES radar_nli_layer3_results(result_id) ON DELETE CASCADE,
    dimension_code                TEXT NOT NULL,
    dimension_name                TEXT NOT NULL,
    construct_code                TEXT NOT NULL,
    construct_name                TEXT NOT NULL,
    hypothesis_code               TEXT NOT NULL,
    hypothesis_kind               TEXT NOT NULL CHECK (hypothesis_kind IN ('positive', 'negative', 'neutral')),
    hypothesis_text               TEXT NOT NULL,
    entailment_probability_3way    DOUBLE PRECISION NOT NULL,
    neutral_probability_3way       DOUBLE PRECISION NOT NULL,
    contradiction_probability_3way DOUBLE PRECISION NOT NULL,
    label_probability              DOUBLE PRECISION NOT NULL,
    inference_time_seconds         DOUBLE PRECISION,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_radar_nli_layer3_hypothesis_score
        UNIQUE (result_id, dimension_code, construct_code, hypothesis_code)
);

CREATE TABLE IF NOT EXISTS radar_nli_layer4_results (
    result_id                    BIGSERIAL PRIMARY KEY,
    run_id                       BIGINT NOT NULL REFERENCES radar_nli_runs(run_id) ON DELETE RESTRICT,
    source_gemma_analysis_id     BIGINT NOT NULL REFERENCES gemma_hate_analyses(analysis_id) ON DELETE RESTRICT,
    comment_id                   INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    platform                     VARCHAR(20) NOT NULL,
    external_id                  VARCHAR(255) NOT NULL,
    text_hash                    VARCHAR(64) NOT NULL,
    text_snapshot                TEXT NOT NULL,
    nli_model_name               TEXT NOT NULL,
    hypotheses_hash              VARCHAR(64) NOT NULL,
    config_hash                  VARCHAR(64) NOT NULL,
    primary_dimension            TEXT,
    primary_score                DOUBLE PRECISION,
    active_dimensions            TEXT[] NOT NULL DEFAULT '{}',
    dimension_scores             JSONB NOT NULL,
    construct_scores             JSONB NOT NULL,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_radar_nli_layer4_result
        UNIQUE (comment_id, nli_model_name, hypotheses_hash, config_hash, text_hash),
    CONSTRAINT ck_radar_nli_layer4_primary_score
        CHECK (primary_score IS NULL OR (primary_score >= 0 AND primary_score <= 1))
);

CREATE TABLE IF NOT EXISTS radar_nli_layer4_hypothesis_scores (
    score_id                      BIGSERIAL PRIMARY KEY,
    result_id                     BIGINT NOT NULL REFERENCES radar_nli_layer4_results(result_id) ON DELETE CASCADE,
    dimension_code                TEXT NOT NULL,
    dimension_name                TEXT NOT NULL,
    construct_code                TEXT NOT NULL,
    construct_name                TEXT NOT NULL,
    hypothesis_code               TEXT NOT NULL,
    hypothesis_kind               TEXT NOT NULL CHECK (hypothesis_kind IN ('positive', 'negative', 'neutral')),
    hypothesis_text               TEXT NOT NULL,
    entailment_probability_3way    DOUBLE PRECISION NOT NULL,
    neutral_probability_3way       DOUBLE PRECISION NOT NULL,
    contradiction_probability_3way DOUBLE PRECISION NOT NULL,
    label_probability              DOUBLE PRECISION NOT NULL,
    inference_time_seconds         DOUBLE PRECISION,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_radar_nli_layer4_hypothesis_score
        UNIQUE (result_id, dimension_code, construct_code, hypothesis_code)
);

CREATE INDEX IF NOT EXISTS idx_radar_nli_runs_layer_status
    ON radar_nli_runs (layer_name, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer3_gemma
    ON radar_nli_layer3_results (source_gemma_analysis_id);
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer3_comment
    ON radar_nli_layer3_results (comment_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_radar_nli_layer3_content
    ON radar_nli_layer3_results (
        platform, text_hash, nli_model_name, hypotheses_hash, config_hash
    );
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer3_scores_result
    ON radar_nli_layer3_hypothesis_scores (result_id);
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer4_gemma
    ON radar_nli_layer4_results (source_gemma_analysis_id);
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer4_comment
    ON radar_nli_layer4_results (comment_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_radar_nli_layer4_content
    ON radar_nli_layer4_results (
        platform, text_hash, nli_model_name, hypotheses_hash, config_hash
    );
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer4_scores_result
    ON radar_nli_layer4_hypothesis_scores (result_id);

DROP VIEW IF EXISTS v_radar_nli_layer3_results;
CREATE VIEW v_radar_nli_layer3_results AS
SELECT
    r.*,
    a.hate_probability AS gemma_hate_probability,
    a.model_name AS gemma_model_name,
    a.prompt_version AS gemma_prompt_version,
    a.analyzed_at AS gemma_analyzed_at
FROM radar_nli_layer3_results r
JOIN gemma_hate_analyses a ON a.analysis_id = r.source_gemma_analysis_id;

DROP VIEW IF EXISTS v_radar_nli_layer4_results;
CREATE VIEW v_radar_nli_layer4_results AS
SELECT
    r.*,
    a.hate_probability AS gemma_hate_probability,
    a.model_name AS gemma_model_name,
    a.prompt_version AS gemma_prompt_version,
    a.analyzed_at AS gemma_analyzed_at
FROM radar_nli_layer4_results r
JOIN gemma_hate_analyses a ON a.analysis_id = r.source_gemma_analysis_id;
