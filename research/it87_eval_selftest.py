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
  TTD [2,10,0.5,4] → медиана 3,0; p90 (nearest-rank ceil(0,9n)-й ряд; при n=4 — максимум
       ряда) = 10,0 → H3 🟢;
  сценарии B/C (пре-открыточный аудит 24.09): B — 5 пролётов [1,2,3,4,58] → p90=58 (max-ряд,
  проверяет ceil вместо banker's round); C — перестановка потока → mono-гейт ловит НЕ-монотонность
  по порядку jsonl (sorted-ряд тривиально монотонен) и H0 блокирует вердикты.
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


def build_bc():
    """Сценарии B/C: 5 pos с TTD [1,2,3,4,58] + 2 neg; C = тот же поток с перестановкой
    (мнимая не-монотонность). Проверки пре-открыточного аудита 24.09: nearest-rank p90 при
    n=5 берёт max-ряд (58 → H3 🔴), а mono читается ПО ПОРЯДКУ ПОТОКА, а не sorted-ряда."""
    (TMP / "bc").mkdir(exist_ok=True)
    pos_alarms = [{21.0}, {22.0}, {23.0}, {24.0}, {78.0}]
    plan = [(f"bpos{i+1}", "pos", a) for i, a in enumerate(pos_alarms)]
    plan += [("bneg1", "neg", {i * 0.5 for i in range(20, 40, 2)}),
             ("bneg2", "neg", {i * 0.5 for i in range(40, 100, 2)})]
    recs, slices = [], []
    for sid, role, al in plan:
        start_line = len(recs) + 1
        recs.extend(windows(alarms=al))
        slices.append(f"{sid}={start_line}:{len(recs)}")
    man = TMP / "bc" / "manifest.tsv"
    man.write_text("\n".join(
        f"{sid}\t{role}\tsynth.mp4\tsynth.wav\t110\t20\t80"
        if role == "pos" else f"{sid}\t{role}\tsynth.mp4\tsynth.wav\t110\t0\t0"
        for sid, role, _ in plan) + "\n", encoding="utf-8")
    return recs, man, slices


def check_b(out):
    fails = []
    m = re.search(r"H2 \(sequence-recall\): (\d+)/(\d+) = .*→ 🟢", out)
    if not m or (int(m[1]), int(m[2])) != (5, 5):
        fails.append("B: H2 ≠ 5/5 🟢")
    m = re.search(r"H3 \(TTD\): медиана ([\d.]+) с ≤10 ∧ p90 ([\d.]+) с ≤20 → (🟢|🔴)", out)
    if not m or (float(m[1]), float(m[2]), m[3]) != (3.0, 58.0, "🔴"):
        fails.append(f"B: H3 {m.groups() if m else None} ≠ 3,0/58,0/🔴 (p90 обязан брать max-ряд при n=5)")
    return fails


def check_c(out):
    fails = []
    if not re.search(r"\[bpos1\].*mono=ПРОВАЛ", out):
        fails.append("C: перестановка потока не поймана mono-гейтом (bpos1)")
    if not re.search(r"H0 .*→ ПРОВАЛ", out):
        fails.append("C: H0 не заблокирован")
    if re.search(r"H1 \(FP", out):
        fails.append("C: при H0-ПРОВАЛЕ вердикты H1–H3 напечатаны (не должно быть)")
    return fails


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
    recs_b, man_b, slices_b = build_bc()
    # B: корректный поток — p90 nearest-rank при n=5 обязан быть max-рядом (58 → H3 🔴)
    jb = TMP / "bc" / "b.jsonl"
    jb.write_text("\n".join(json.dumps(x) for x in recs_b) + "\n", encoding="utf-8")
    rb = subprocess.run(
        [sys.executable, str(copy), f"--jsonl={jb}", f"--manifest={man_b}",
         f"--out={TMP/'bc'/'b.txt'}", f"--weights-dir={TMP/'weights'}", *slices_b],
        capture_output=True, text=True)
    fails += check_b(rb.stdout + rb.stderr)
    # C: тот же поток, но bpos1 перемешан (окно 2,5 с впереди 1,0 с) — mono по порядку потока
    recs_c = list(recs_b)
    recs_c[3], recs_c[8] = recs_c[8], recs_c[3]
    jc = TMP / "bc" / "c.jsonl"
    jc.write_text("\n".join(json.dumps(x) for x in recs_c) + "\n", encoding="utf-8")
    rc = subprocess.run(
        [sys.executable, str(copy), f"--jsonl={jc}", f"--manifest={man_b}",
         f"--out={TMP/'bc'/'c.txt'}", f"--weights-dir={TMP/'weights'}", *slices_b],
        capture_output=True, text=True)
    fails += check_c(rc.stdout + rc.stderr)
    if r.returncode != 0 or rb.returncode != 0 or rc.returncode != 0:
        fails.append(f"returncode {r.returncode}/{rb.returncode}/{rc.returncode}")
    if fails:
        print("\nСАМОПРОВЕРКА: ПРОВАЛ")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("\nСАМОПРОВЕРКА it87_holdout_eval на синтетике: ВСЕ ПУТИ A(H1/H2/H3/CSV) "
          "+ B(p90 n=5 max-ряд) + C(mono по потоку → H0-блок) СОВПАЛИ 🟢")


if __name__ == "__main__":
    main()
