-- Permite que o NLI processe todos os conteúdos classificados como hate pelo
-- Gemma: comentários de redes sociais e posts (incluindo WhatsApp/Telegram).

ALTER TABLE radar_nli_layer3_results
    ADD COLUMN IF NOT EXISTS content_kind TEXT,
    ADD COLUMN IF NOT EXISTS post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE;

ALTER TABLE radar_nli_layer4_results
    ADD COLUMN IF NOT EXISTS content_kind TEXT,
    ADD COLUMN IF NOT EXISTS post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE;

UPDATE radar_nli_layer3_results
SET content_kind = 'comment'
WHERE content_kind IS NULL;

UPDATE radar_nli_layer4_results
SET content_kind = 'comment'
WHERE content_kind IS NULL;

ALTER TABLE radar_nli_layer3_results
    ALTER COLUMN content_kind SET DEFAULT 'comment',
    ALTER COLUMN content_kind SET NOT NULL,
    ALTER COLUMN comment_id DROP NOT NULL;

ALTER TABLE radar_nli_layer4_results
    ALTER COLUMN content_kind SET DEFAULT 'comment',
    ALTER COLUMN content_kind SET NOT NULL,
    ALTER COLUMN comment_id DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_radar_nli_layer3_content_reference'
    ) THEN
        ALTER TABLE radar_nli_layer3_results
            ADD CONSTRAINT ck_radar_nli_layer3_content_reference CHECK (
                (content_kind = 'comment' AND comment_id IS NOT NULL AND post_id IS NULL)
                OR
                (content_kind = 'post' AND post_id IS NOT NULL AND comment_id IS NULL)
            );
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_radar_nli_layer4_content_reference'
    ) THEN
        ALTER TABLE radar_nli_layer4_results
            ADD CONSTRAINT ck_radar_nli_layer4_content_reference CHECK (
                (content_kind = 'comment' AND comment_id IS NOT NULL AND post_id IS NULL)
                OR
                (content_kind = 'post' AND post_id IS NOT NULL AND comment_id IS NULL)
            );
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_radar_nli_layer3_post
    ON radar_nli_layer3_results (post_id);
CREATE INDEX IF NOT EXISTS idx_radar_nli_layer4_post
    ON radar_nli_layer4_results (post_id);

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
