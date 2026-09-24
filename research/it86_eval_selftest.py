#!/usr/bin/env python3
"""Самопроверка it86_sync_negative_eval на синтетике (без Docker): объединённый multi-segment
срез плеча, перезапуск медиа-шкалы, перекрывающиеся окна, wall-clock-обвязка.
Запуск: research/.venv/bin/python research/it86_eval_selftest.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "research/it86_sync_negative_eval.py"
PY = sys.executable


def rec(t0, t1, dec=False, pa=None, mts="auto"):
    m = t0 if mts == "auto" else mts
    return json.dumps({"decision": dec, "ts_window": [t0, t1], "media_ts": m,
                       "contributions": {"p_a": pa}})


def run_eval(lines):
    with tempfile.TemporaryDirectory() as td:
        jl = Path(td) / "d.jsonl"
        jl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = Path(td) / "s.txt"
        r = subprocess.run([PY, str(EVAL), f"--jsonl={jl}", f"--out={out}",
                            "V0 off-audio=1:%d" % len(lines)],
                           capture_output=True, text=True, check=True)
        return r.stdout


def parse(stdout):
    line = next(x for x in stdout.splitlines() if x.startswith("[V0 off-audio]"))
    cover = float(line.split("медиа-покрытие=")[1].split(" с")[0])
    nseg = int(line.split("сегм. ")[1].split(")")[0])
    fa = int(line.split("FA=")[1].split(" ")[0])
    return cover, nseg, fa


def main():
    # m1: два сегмента без пересчёта окон + wall-clock-обвязка (media_ts=None)
    l = [rec(1.79e9, 1.79e9 + 2, mts=None)]
    l += [rec(t, t + 10, t in (50, 150, 250), 0.1) for t in range(0, 300, 10)]
    l += [rec(t, t + 10, t in (20, 220), 0.2) for t in range(0, 320, 10)]
    c, k, fa = parse(run_eval(l))
    assert (c, k, fa) == (620.0, 2, 5), f"m1: {(c, k, fa)}"
    # m2: перекрывающиеся окна (шаг 10, длина 15): union 305+325=630, FA=2
    l = [rec(t, t + 15, t == 50, 0.1) for t in range(0, 300, 10)]
    l += [rec(t, t + 15, t == 20, 0.2) for t in range(0, 320, 10)]
    c, k, fa = parse(run_eval(l))
    assert (c, k, fa) == (630.0, 2, 2), f"m2: {(c, k, fa)}"
    # m3: регрессия одиночного монотонного сегмента (17-с sanity-случай): 17 с, 1 сегмент
    l = [rec(t / 2.0, (t + 1) / 2.0, False, 0.05) for t in range(34)]
    c, k, fa = parse(run_eval(l))
    assert (c, k, fa) == (17.0, 1, 0), f"m3: {(c, k, fa)}"
    print("it86_eval_selftest: OK (m1 рестарт-шкала, m2 union-пересечения, m3 регрессия 17с)")


if __name__ == "__main__":
    main()
