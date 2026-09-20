#!/usr/bin/env python3
"""Симуляция политик fusion на ПОЛНОСТЬЮ смешанных окнах (YOLO×AST), it-06 → it-36.

В отличие от jsonl-прогонов (где аудио в fusion почти не доходило), здесь обе
модальности доступны в каждом из 144 окон: p_v — YOLO max_conf по секундам
(it-04, imgsz 480), p_a — AST p(drone) по окнам 1с/0.5с (it-05; с it-36 — ЧЕСТНАЯ
вероятность, без обнуления non-drone предсказаний — ревью §4).
GT: airborne (majority секунды окна).

it-36 (ревью §2/§5.1): симметричная временнáя обработка для ОБОИХ методов —
audio-only и late сравниваются при одинаковых фильтрах (k=1 контроль / каузальная
медиана / центрированная медиана, явно помеченная «офлайн, видит будущее»).
Запуск: research/.venv/bin/python research/fusion_sim_full.py
"""
import csv
import math

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
import argparse
_YA = argparse.ArgumentParser(); _YA.add_argument("--yolo-csv", default="research/yolo_sandbox_frames.csv"); _YA.add_argument("--suffix", default="", help="суффикс для вых. CSV, напр. -new (it-66)"); _ARGS = _YA.parse_known_args()[0]; _YSRC = str(__import__("pathlib").Path(ROOT) / _ARGS.yolo_csv); _SFX = _ARGS.suffix  # it-66: пересчёт новыми весами

ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(_YSRC))
        if r["imgsz"] == "480"}

WINS = sorted(ast.keys(), key=float)

def H(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if P + R else float("nan")
    return P, R, F

# окна: (t0, p_v, p_a, gt_airborne)
wins = []
for t0 in WINS:
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
    pa = float(ast[t0]["p_drone"])
    wins.append((float(t0), pv, pa, int(ast[t0]["airborne_gt"])))

def fuse(pv, pa, policy):
    if policy == "video-only":
        return pv
    if policy == "audio-only":
        return pa
    if policy.startswith("late"):
        wv = float(policy.split("w_v=")[1]) if "w_v=" in policy else 0.5
        base = wv * pv + (1 - wv) * pa
        if "+Δ" in policy:
            if pv >= 0.5 and pa >= 0.5:
                base += 0.1          # δ_conf: согласие «дрон»
            elif (pv >= 0.5) != (pa >= 0.5):
                base -= 0.1          # δ_unconf: противоречие
        return min(max(base, 0.0), 1.0)
    if policy.startswith("entropy"):
        lam = float(policy.split("λ=")[1])
        e_v, e_a = math.exp(-lam * H(pv)), math.exp(-lam * H(pa))
        return (e_v * pv + e_a * pa) / (e_v + e_a)
    if policy == "consensus":
        if (pv >= 0.5) == (pa >= 0.5):
            return 0.5 * pv + 0.5 * pa
        # противоречие: вес менее уверенного канала → 0
        return pv if pv >= pa else pa
    raise ValueError(policy)

POLICIES = ["video-only", "audio-only", "late 0.5/0.5", "late 0.5/0.5+Δ",
            "late w_v=0.7", "late w_v=0.9", "entropy λ=1", "entropy λ=2", "consensus"]

print(f"окон: {len(wins)} (все смешанные: YOLO×AST) | airborne окон: {sum(w[3] for w in wins)}\n")
print(f"{'политика':<18} {'P':>6} {'R':>6} {'F1':>6}   FP   FN")

def report_row(name, mk, pas=None):
    """Строка метрик: preds по окнам; pas — сглаженный ряд p_a (иначе сырой)."""
    seq = pas if pas is not None else [w[2] for w in wins]
    preds = [(int(mk(w[1], pa, w) >= 0.5), w[3]) for w, pa in zip(wins, seq, strict=True)]
    P, R, F = prf(preds)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    print(f"{name:<18} {P:6.3f} {R:6.3f} {F:6.3f}  {fp:4d} {fn:4d}")
    return dict(policy=name, P=round(P, 3), R=round(R, 3), F1=round(F, 3), FP=fp, FN=fn)

results_csv = [report_row(pol, lambda pv, pa, w: fuse(pv, pa, pol)) for pol in POLICIES]

# --- временнáя обработка: СИММЕТРИЧНО для audio-only и late ---
def causal_median(vals: list[float], k: int) -> list[float]:
    """Каузальная медиана последних k значений (текущее включается) — потоковый вариант."""
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out

def centered_median(vals: list[float], k: int) -> list[float]:
    """Центрированная медиана (±k//2) — ОФЛАЙН: использует будущие окна (ревью §5.1)."""
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k // 2), min(len(vals), i + k // 2 + 1)
        seg = sorted(vals[lo:hi])
        out.append(seg[len(seg) // 2])
    return out

print("\n--- временнáя обработка p_a (симметрично для обоих методов) ---")
print("(k=1 — контроль: та же вероятность без сглаживания)")
FILTERS = [("k=1", lambda v: list(v)),
           ("causal median-5 (поток)", lambda v: causal_median(v, 5)),
           ("centered median-5 (ОФЛАЙН, видит будущее)", lambda v: centered_median(v, 5)),
           ("causal median-7 (поток)", lambda v: causal_median(v, 7)),
           ("centered median-7 (ОФЛАЙН, видит будущее)", lambda v: centered_median(v, 7))]
for fname, ffn in FILTERS:
    pas = ffn([w[2] for w in wins])
    for name, mk in (("audio-only", lambda pv, pa, w: pa),
                     ("late 0.5/0.5", lambda pv, pa, w: 0.5 * pv + 0.5 * pa)):
        results_csv.append(report_row(f"{name}+{fname}", mk, pas))

with open(f"{ROOT}/research/fusion_sim_results{_SFX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["policy", "P", "R", "F1", "FP", "FN"])
    w.writeheader()
    w.writerows(results_csv)
print(f"\nCSV: research/fusion_sim_results.csv ({len(results_csv)} строк)")

# профиль одной лучшей политики: где late ошибается
print("\nlate 0.5/0.5+Δ: расхождения с GT (t0, p_v, p_a, pred, gt):")
for t0, pv, pa, gt in wins:
    pred = int(fuse(pv, pa, "late 0.5/0.5+Δ") >= 0.5)
    if pred != gt:
        print(f"  t={t0:5.1f}  p_v={pv:.2f}  p_a={pa:.2f}  pred={pred}  gt={gt}")
