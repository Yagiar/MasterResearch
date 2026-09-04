"""Доступ к PostgreSQL — запись результатов пайплайна (inference + decisions).

Тонкая обёртка над psycopg3 (синхронный — consumer-сервисы синхронные). Схема
БД создаётся миграциями Liquibase (`infra/postgres/changelog/`), здесь — только INSERT'ы.
Идемпотентность: `ON CONFLICT (msg_id) DO NOTHING` (at-least-once + UNIQUE-ограничение).

Импорт `psycopg` — ленивый (внутри методов), чтобы пакет ставился/тестировался без БД.
"""

from __future__ import annotations

from typing import Any

from .logging import get_logger
from .messages import DecisionMsg, InferenceMsg

log = get_logger("uavdet-db")

_INSERT_INFERENCE = """
INSERT INTO uavdet.inference
    (msg_id, schema_ver, source_id, ts, modality, label, confidence, bbox, track_id,
     model_name, model_ver, det_latency_ms, ingest_ts)
VALUES (%(msg_id)s, %(schema_ver)s, %(source_id)s, %(ts)s, %(modality)s, %(label)s,
        %(confidence)s, %(bbox)s, %(track_id)s, %(model_name)s, %(model_ver)s,
        %(det_latency_ms)s, %(ingest_ts)s)
ON CONFLICT (msg_id) DO NOTHING
"""

_INSERT_DECISION = """
INSERT INTO uavdet.decisions
    (msg_id, schema_ver, source_id, ts, ts_window_start, ts_window_end, mode, decision,
     p_fused, p_v, p_a, w_v, w_a, delta, snr_audio, img_quality, e2e_latency_ms, source_msg_ids)
VALUES (%(msg_id)s, %(schema_ver)s, %(source_id)s, %(ts)s, %(ts_window_start)s, %(ts_window_end)s,
        %(mode)s, %(decision)s, %(p_fused)s, %(p_v)s, %(p_a)s, %(w_v)s, %(w_a)s, %(delta)s,
        %(snr_audio)s, %(img_quality)s, %(e2e_latency_ms)s, %(source_msg_ids)s)
ON CONFLICT (msg_id) DO NOTHING
"""


class PgResultStore:
    """Запись InferenceMsg / DecisionMsg в PostgreSQL (autocommit, идемпотентно)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None

    def _ensure_conn(self):
        import psycopg  # noqa: PLC0415

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, autocommit=True)
            log.info("uavdet-db: подключение установлено")
        return self._conn

    @staticmethod
    def _inference_params(msg: InferenceMsg) -> dict[str, Any]:
        return {
            "msg_id": msg.msg_id,
            "schema_ver": msg.schema_ver,
            "source_id": msg.source_id,
            "ts": msg.ts,
            "modality": msg.modality,
            "label": msg.label,
            "confidence": msg.confidence,
            "bbox": list(msg.bbox) if msg.bbox is not None else None,
            "track_id": msg.track_id,
            "model_name": msg.model.name,
            "model_ver": msg.model.ver,
            "det_latency_ms": msg.det_latency_ms,
            "ingest_ts": msg.ingest_ts,
        }

    @staticmethod
    def _decision_params(msg: DecisionMsg) -> dict[str, Any]:
        t0, t1 = (msg.ts_window + [0.0, 0.0])[:2]
        c = msg.contributions
        g = msg.gating
        return {
            "msg_id": msg.msg_id,
            "schema_ver": msg.schema_ver,
            "source_id": msg.source_id,
            "ts": msg.ts,
            "ts_window_start": t0,
            "ts_window_end": t1,
            "mode": msg.mode,
            "decision": msg.decision,
            "p_fused": msg.p_fused,
            "p_v": c.p_v,
            "p_a": c.p_a,
            "w_v": c.w_v,
            "w_a": c.w_a,
            "delta": c.delta,
            "snr_audio": g.snr_audio,
            "img_quality": g.img_quality,
            "e2e_latency_ms": msg.e2e_latency_ms,
            "source_msg_ids": list(msg.source_msg_ids),
        }

    def write_inference(self, msg: InferenceMsg) -> None:
        with self._ensure_conn().cursor() as cur:
            cur.execute(_INSERT_INFERENCE, self._inference_params(msg))

    def write_decision(self, msg: DecisionMsg) -> None:
        with self._ensure_conn().cursor() as cur:
            cur.execute(_INSERT_DECISION, self._decision_params(msg))

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()


def make_dsn(
    *,
    host: str = "postgres",
    port: int = 5432,
    dbname: str = "uavdet",
    user: str = "uavdet",
    password: str = "uavdet",
) -> str:
    """Собрать DSN-строку подключения к PostgreSQL."""
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


__all__ = ["PgResultStore", "make_dsn"]
