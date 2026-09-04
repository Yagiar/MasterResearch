--liquibase formatted sql

--changeset uavdet:003-decisions
--comment: решения fusion-движка (топик decisions) — по одной строке на DecisionMsg
CREATE TABLE uavdet.decisions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    msg_id          TEXT        NOT NULL,
    schema_ver      INT         NOT NULL DEFAULT 1,
    source_id       TEXT        NOT NULL,
    ts              DOUBLE PRECISION NOT NULL,            -- момент принятия решения (unix-время, сек)
    event_time      TIMESTAMPTZ GENERATED ALWAYS AS (to_timestamp(ts)) STORED,
    ts_window_start DOUBLE PRECISION NOT NULL,            -- [t0, t1] окна выравнивания
    ts_window_end   DOUBLE PRECISION NOT NULL,
    mode            TEXT        NOT NULL CHECK (mode IN ('video-only','audio-only','late','hybrid')),
    decision        BOOLEAN     NOT NULL,                 -- обнаружен БПЛА / нет
    p_fused         REAL        NOT NULL,
    p_v             REAL,                                 -- вклад видео (NULL — модальность не участвовала)
    p_a             REAL,                                 -- вклад аудио
    w_v             REAL,                                 -- вес видео
    w_a             REAL,                                 -- вес аудио
    delta           REAL        NOT NULL DEFAULT 0,       -- поправка правила компенсации (late/hybrid)
    snr_audio       REAL,                                 -- gating: SNR аудио
    img_quality     REAL,                                 -- gating: качество кадра
    e2e_latency_ms  REAL        NOT NULL DEFAULT 0,
    source_msg_ids  TEXT[]      NOT NULL DEFAULT '{}',    -- msg_id детекций, вошедших в окно
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_decisions_msg_id UNIQUE (msg_id)
);
CREATE INDEX ix_decisions_source_ts ON uavdet.decisions (source_id, ts);
CREATE INDEX ix_decisions_mode      ON uavdet.decisions (mode);
CREATE INDEX ix_decisions_decision  ON uavdet.decisions (decision);
--rollback DROP TABLE IF EXISTS uavdet.decisions;
