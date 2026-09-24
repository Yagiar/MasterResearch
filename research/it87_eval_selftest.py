#!/usr/bin/env python3
"""it-87 самопроверка eval-харнеса (НЕ эксперимент, НЕ боевой замер).

Назначение: боевое открытие it-87 — однократное, а позитивные пути H2 (sequence-recall,
счёт пойманных пролётов) и H3 (медиана/p90 TTD, фильтр alarm ∈ [gt_start, gt_end]) в
live pre-flight 24.09 не исполнялись (весь smoke-материал был role=neg). Скрипт строит
полностью синтетический decisions.jsonl + manifest с ЗАРАНЕЕ известными expected-значениями,
прогоняет копию боевого it87_holdout_eval.py (единственная правка копии — канонические
sha-префиксы заменяются на префиксы фиктивных весов, т.к. подбор коллизии sha256 невозможен)
и сверяет распечатанные H1/H2/H3/вердикт и CSV с ручной арифметикой.

Ожидания (конструкция данных, см. build()):
  7 сегментов (pos1..pos5, neg1, neg2), dur=110, окна 0,5 с (span=100 → ratio=1,000 OK);
  pos: pos1 TTD 2,0; pos2 TTD 10,0; pos3 TTD 0,5; pos4 TTD 4,0; pos5 — вне интервала →
       H2 = 4/5 = 80 % (красный путь гейта, ≥90 не выполнен);
  TTD [2,10,0.5,4] → медиана 3,0; p90 (nearest-rank round(0,9n)-1, при n=4 — максимум
       ряда) = 10,0 → H3 🟢;
  neg: 40/400 = 10,0 % → H1 🟢; итог-вердикт «НЕ ПОДТВЕРЖДЕНО» (из-за H2).
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "it87_holdout_eval.py"
TMP = Path("/tmp/it87_selftest")


def sha8(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def windows(start_t=0.0, n=201, alarms=(), dt=0.5):
    out = []
    for i in range(n):
        t = start_t + i * dt
        out.append({"media_ts": t, "decision": t in alarms,
                    "ts_window": [t, t + dt],
                    "contributions": {"p_v": 0.5, "p_a": 0.2, "w_v": 0.5, "w_a": 0.5, "delta": 0.0}})
    return out


def build():
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "weights").mkdir(exist_ok=True)
    old = TMP / "weights/yolov8s-uav.pt"
    new = TMP / "weights/uav-yolov8s-bg-best.pt"
    old.write_bytes(b"synthetic-old")
    new.write_bytes(b"synthetic-new")

    pos_alarms = [
        {22.0},        # pos1 TTD 2,0
        {30.0},        # pos2 TTD 10,0
        {20.5},        # pos3 TTD 0,5
        {24.0},        # pos4 TTD 4,0
        {90.0},        # pos5 — поймает TTD, но вне [20,80] → не пойман
    ]
    segs, recs, slices = [], [], []
    # neg1: 10 окон-alarms из 201; neg2: 30 (итого 40/400 ≈ 10 %)
    plan = [("pos1", "pos", pos_alarms[0]), ("pos2", "pos", pos_alarms[1]),
            ("pos3", "pos", pos_alarms[2]), ("pos4", "pos", pos_alarms[3]),
            ("pos5", "pos", pos_alarms[4]),
            ("neg1", "neg", {i * 0.5 for i in range(20, 40, 2)}),
            ("neg2", "neg", {i * 0.5 for i in range(40, 100, 2)})]
    alarms_total = sum(len(a) for _, r, a in plan if r == "neg")
    for sid, role, al in plan:
        w = windows(alarms=al if role == "pos" else al)
        # для pos alarm должен попасть в [gs,ge]; для neg интервала нет
        start_line = len(recs) + 1
        recs.extend(w)
        slices.append(f"{sid}={start_line}:{len(recs)}")
        segs.append((sid, role, al))
    jsonl = TMP / "decisions.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    man = TMP / "manifest.tsv"
    man.write_text("\n".join(
        f"{sid}\t{role}\tsynth.mp4\tsynth.wav\t110\t20\t80"
        if role == "pos" else f"{sid}\t{role}\tsynth.mp4\tsynth.wav\t110\t0\t0"
        for sid, role, _ in segs) + "\n", encoding="utf-8")

    src = EVAL.read_text(encoding="utf-8")
    copy = TMP / "it87_eval_copy.py"
    copy.write_text(
        src.replace('"41f3fd55"', f'"{sha8(old)}"').replace('"7042602a"', f'"{sha8(new)}"'),
        encoding="utf-8")
    return jsonl, man, copy, slices, alarms_total


def check(out, alarms_total):
    fails = []

    def want(pat, label):
        m = re.search(pat, out)
        if not m:
            fails.append(f"не найден вывод: {label} (/{pat}/)")
        return m

    want(r"H0 sha yolov8s-uav\.pt: [0-9a-f]{8}.*→ OK", "H0 sha old")
    want(r"H0 .*→ ПРОЙДЕН", "H0 итог")
    m = want(r"H1 \(FP на негативах\): alarm-окна (\d+)/(\d+) = ([\d.]+) % .*→ 🟢", "H1 зелёный")
    if m and (int(m[1]), int(m[2])) != (alarms_total, 402):
        fails.append(f"H1 окно-счёт {m[1]}/{m[2]} ≠ {alarms_total}/402")
    m = want(r"H2 \(sequence-recall\): (\d+)/(\d+) = (\d+) % .*→ 🔴", "H2 красный путь")
    if m and (int(m[1]), int(m[2])) != (4, 5):
        fails.append(f"H2 {m[1]}/{m[2]} ≠ 4/5")
    m = want(r"H3 \(TTD\): медиана ([\d.]+) с ≤10 ∧ p90 ([\d.]+) с ≤20 → 🟢", "H3")
    if m and (float(m[1]), float(m[2])) != (3.0, 10.0):
        fails.append(f"H3 med/p90 {m[1]}/{m[2]} ≠ 3,0/10,0")
    want(r"ВЕРДИКТ it-87 .*: НЕ ПОДТВЕРЖДЕНО", "итог-вердикт от H2")
    # CSV: TTD pos-строк
    csv = (Path(str(TMP / "s.txt") + ".csv")).read_text(encoding="utf-8")
    ttd = {ln.split(",")[0]: ln.split(",")[6] for ln in csv.splitlines()[1:]}
    for sid, exp in {"pos1": "2.0", "pos2": "10.0", "pos3": "0.5", "pos4": "4.0", "pos5": ""}.items():
        if ttd.get(sid) != exp:
            fails.append(f"CSV {sid} TTD={ttd.get(sid)!r} ≠ {exp!r}")
    return fails


def main():
    jsonl, man, copy, slices, alarms_total = build()
    outp = TMP / "s.txt"
    r = subprocess.run(
        [sys.executable, str(copy), f"--jsonl={jsonl}", f"--manifest={man}", f"--out={outp}",
         f"--weights-dir={TMP/'weights'}", *slices],
        capture_output=True, text=True)
    out = r.stdout + r.stderr
    fails = check(out, alarms_total)
    print(out)
    if r.returncode != 0:
        fails.append(f"returncode {r.returncode}")
    if fails:
        print("\nСАМОПРОВЕРКА: ПРОВАЛ")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("\nСАМОПРОВЕРКА it87_holdout_eval на синтетике: ВСЕ ПУТИ H1/H2/H3/CSV СОВПАЛИ С РУЧНОЙ АРИФМЕТИКОЙ 🟢")


if __name__ == "__main__":
    main()
