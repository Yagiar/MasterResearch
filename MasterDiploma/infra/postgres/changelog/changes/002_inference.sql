--liquibase formatted sql

--changeset uavdet:002-inference
--comment: результаты детекторов (топик inference) — по одной строке на InferenceMsg
CREATE TABLE uavdet.inference (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    msg_id          TEXT        NOT NULL,
    schema_ver      INT         NOT NULL DEFAULT 1,
    source_id       TEXT        NOT NULL,
    ts              DOUBLE PRECISION NOT NULL,            -- момент события (unix-время, сек)
    event_time      TIMESTAMPTZ GENERATED ALWAYS AS (to_timestamp(ts)) STORED,
    modality        TEXT        NOT NULL CHECK (modality IN ('video','audio')),
    label           TEXT        NOT NULL CHECK (label IN ('drone','non-drone')),
    confidence      REAL        NOT NULL,
    bbox            REAL[],                                -- [x,y,w,h] (только modality=video)
    track_id        INT,
    model_name      TEXT        NOT NULL DEFAULT 'unknown',
    model_ver       TEXT        NOT NULL DEFAULT '0',
    det_latency_ms  REAL        NOT NULL DEFAULT 0,
    ingest_ts       DOUBLE PRECISION,                     -- когда сообщение попало в video.raw/audio.raw
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),   -- когда строка записана в БД
    CONSTRAINT uq_inference_msg_id UNIQUE (msg_id)        -- идемпотентность (at-least-once)
);
CREATE INDEX ix_inference_source_ts ON uavdet.inference (source_id, ts);
CREATE INDEX ix_inference_modality  ON uavdet.inference (modality);
--rollback DROP TABLE IF EXISTS uavdet.inference;
