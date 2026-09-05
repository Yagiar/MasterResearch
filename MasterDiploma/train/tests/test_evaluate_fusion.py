"""Тесты evaluate_fusion_jsonl (it-48): протокол media_ts + NORM + burn-in, метрики.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from uavtrain.evaluate import evaluate_fusion_jsonl


def _write_gt(tmp_path: Path) -> Path:
    gt = tmp_path / "gt.csv"
    # 20 секунд: 0–9 airborne=0, 10–19 airborne=1
    lines = ["second,drone_visible,airborne"]
    lines += [f"{s},1,{'1' if s >= 10 else '0'}" for s in range(20)]
    gt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gt


def _dec_line(media_ts: float, decision: bool, *, p_v: float | None, p_a: float | None,
              mode: str = "late", ts: float | None = None) -> str:
    return json.dumps({
        "media_ts": media_ts, "ts": ts if ts is not None else 1000.0 + media_ts,
        "mode": mode, "decision": int(decision), "p_fused": 0.9,
        "contributions": {"p_v": p_v, "p_a": p_a, "w_v": 0.5, "w_a": 0.5, "delta": 0.0},
    })


def test_evaluate_fusion_jsonl_metrics(tmp_path: Path) -> None:
    gt = _write_gt(tmp_path)
    dec = tmp_path / "decisions.jsonl"
    lines = [
        # perfect joint-окна: 10 положительных в airborne-сегменте, 10 отрицательных на земле
        _dec_line(i + 0.5, i >= 10, p_v=0.9 if i >= 10 else 0.1, p_a=0.9 if i >= 10 else 0.1)
        for i in range(20)
    ]
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = evaluate_fusion_jsonl(dec, gt, name="test")
    m = report.metrics
    assert m["n_decisions"] == 20
    assert m["n_skipped_no_media"] == 0
    assert m["all_joint_share"] == 1.0
    assert m["all_raw"]["f1"] == 1.0
    assert m["all_norm"]["f1"] == 1.0
    assert (report.artifacts_dir / "metrics.json").exists()


def test_evaluate_fusion_jsonl_skips_and_no_media(tmp_path: Path) -> None:
    gt = _write_gt(tmp_path)
    dec = tmp_path / "decisions.jsonl"
    lines = [_dec_line(10.5, True, p_v=0.9, p_a=0.9),
             json.dumps({"media_ts": None, "ts": 1001.0, "mode": "late", "decision": 1,
                         "contributions": {"p_v": 0.9, "p_a": None}})]
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = evaluate_fusion_jsonl(dec, gt, name="test2").metrics
    assert m["n_decisions"] == 1
    assert m["n_skipped_no_media"] == 1


def test_evaluate_fusion_jsonl_burn_in(tmp_path: Path) -> None:
    gt = _write_gt(tmp_path)
    dec = tmp_path / "decisions.jsonl"
    # ts растёт: 1000..1009 — burn-in 5 с оставит только ts >= 1005 (5 решений)
    lines = [_dec_line(0.5, True, p_v=0.9, p_a=0.9, ts=1000.0 + i) for i in range(10)]
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = evaluate_fusion_jsonl(dec, gt, name="test3", burn_in_s=5.0).metrics
    assert m["n_decisions"] == 5


def test_evaluate_fusion_jsonl_raises_without_media_ts(tmp_path: Path) -> None:
    gt = _write_gt(tmp_path)
    dec = tmp_path / "decisions.jsonl"
    dec.write_text(json.dumps({"media_ts": None, "ts": 1000.0, "mode": "late", "decision": 1}) + "\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="media_ts"):
        evaluate_fusion_jsonl(dec, gt, name="test4")
