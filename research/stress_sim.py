#!/usr/bin/env python3
"""Стресс-симуляция политик fusion на смешанных окнах (it-07 → it-36).

Стрессы (аналоги «канала деградации» пайплайна, офлайн):
  - WindowDrop аудио: окно теряет аудио с вероятностью p_drop → для late — решение
    по видео (перенормировка), для audio-only — НЕТ решения (не тревога);
  - NoiseDegradation ОЦЕНОК аудио: p_a' = clip(p_a + N(0, sigma), 0, 1) — повреждается
    ВЫХОД классификатора, а не акустический сигнал (ревью §7: это проверка правила
    объединения, а не устойчивости модели к ветру/компрессии/SNR — не смешивать);
  - Outage: аудио полностью пропадает на последних X% клипа (отказ модальности).
Каждая точка усреднена по N_SEEDS сидам (mean ± std). GT: airborne (majority секунды).
it-36 (ревью §5.1/§7): медиана в late+median5 — КАУЗАЛЬНАЯ (последние k валидных окон);
«audio-only» без аудио больше не отдаёт видео-решение (скрытый fallback маскировал
несопоставимость baseline — ревью §7/§12). Запуск: research/.venv/bin/python research/stress_sim.py
"""
import csv
import math
import random
from statistics import mean, stdev

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
import argparse
_YA = argparse.ArgumentParser(); _YA.add_argument("--yolo-csv", default="research/yolo_sandbox_frames.csv"); _YSRC = str(__import__("pathlib").Path(ROOT) / _YA.parse_known_args()[0].yolo_csv)  # it-66: пересчёт новыми весами
N_SEEDS = 25

ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(_YSRC))
        if r["imgsz"] == "480"}

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"])
    wins.append((float(t0), pv, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))
N = len(wins)


def H(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def causal_median_valid(pas, i, k=5):
    """Каузальная медиана последних k ВАЛИДНЫХ значений (текущее включается; пропуски ≠ ноль).

    Отличие от прежней центрированной (±k//2): не заглядывает в будущие окна (ревью §5.1).
    """
    vals = [pas[j] for j in range(max(0, i - k + 1), i + 1) if pas[j] is not None]
    return sorted(vals)[len(vals) // 2] if vals else None


def decide(pol, pv, pa):
    """Возвращает 1/0; None = решения нет (audio-only при пропавшем аудио).

    ВАЖНО (ревью §7): audio-only при pa=None НЕ имеет права на видео-fallback —
    иначе это не одномодальный baseline. Для late пропуск аудио = перенормировка на
    видео (заявленная политика fallback, после it-32 — честная маска каналов).
    """
    if pol == "video-only":
        return int(pv >= 0.5)
    if pol == "audio-only":
        return None if pa is None else int(pa >= 0.5)
    if pa is None:
        # late/entropy/consensus: аудио отсутствует — решение по видео (перенормировка)
        return int(pv >= 0.5)
    if pol.startswith("late"):
        wv = float(pol.split("w_v=")[1]) if "w_v=" in pol else 0.5
        s = wv * pv + (1 - wv) * pa
        if "+Δ" in pol:
            if pv >= 0.5 and pa >= 0.5:
                s += 0.1
            elif (pv >= 0.5) != (pa >= 0.5):
                s -= 0.1
        return int(s >= 0.5)
    if pol.startswith("entropy"):
        lam = float(pol.split("λ=")[1])
        e_v, e_a = math.exp(-lam * H(pv)), math.exp(-lam * H(pa))
        return int((e_v * pv + e_a * pa) / (e_v + e_a) >= 0.5)
    if pol == "consensus":
        if (pv >= 0.5) == (pa >= 0.5):
            return int(0.5 * pv + 0.5 * pa >= 0.5)
        return int((pv if pv >= pa else pa) >= 0.5)
    if pol == "late+median5":
        return None  # обрабатывается отдельно (нужен весь ряд)
    raise ValueError(pol)


def run_trial(pol, drop_p=0.0, sigma=0.0, outage_frac=0.0, rng=None):
    rng = rng or random.Random(0)
    pas = []
    for i, (_, pv, pa, _) in enumerate(wins):
        if rng.random() < drop_p or (outage_frac and i >= N * (1 - outage_frac)):
            pas.append(None)
        else:
            pas.append(min(max(pa + rng.gauss(0, sigma), 0.0), 1.0))
    if pol == "late+median5":
        # КАУЗАЛЬНАЯ медиана по валидным окнам (it-36); семантика рантайма (fusion после
        # it-31/32): окно БЕЗ аудио — моно-видео (состояние фильтра не читается без аудио),
        # окно с аудио — медиана последних k валидных значений включая текущее.
        preds = []
        for i, (_, pv, _, gt) in enumerate(wins):
            if pas[i] is None:
                preds.append((int(pv >= 0.5), gt))
                continue
            sm = causal_median_valid(pas, i, k=5)
            preds.append((int(0.5 * pv + 0.5 * sm >= 0.5), gt))
    else:
        raw = [decide(pol, pv, pa) for (_, pv, _, gt), pa in zip(wins, pas, strict=True)]
        # None = «нет тревоги»: считается как не-положительное (FN при положительном GT, иначе корректный отказ)
        preds = [(int(bool(p)), gt) for p, (_, _, _, gt) in zip(raw, wins, strict=True)]
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return 2 * P * R / (P + R) if P + R else 0.0


POLICIES = ["video-only", "audio-only", "late 0.5/0.5", "late 0.5/0.5+Δ",
            "late w_v=0.7", "late w_v=0.9", "entropy λ=1", "consensus", "late+median5"]

rows = []
print("=== 1. WindowDrop аудио (F1 mean±std, 25 сидов) ===")
print(f"{'p_drop':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
csv_out = []
for drop_p in (0.0, 0.2, 0.4, 0.6, 0.8):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, drop_p=drop_p, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"drop{drop_p}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{drop_p:>7} | " + " | ".join(f"{c:>9}" for c in cells))

print("\n=== 2. Шумовая деградация ОЦЕНОК p_a (сигнал не повреждается!), sigma (F1 mean±std) ===")
print(f"{'sigma':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
for sigma in (0.0, 0.2, 0.4, 0.6):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, sigma=sigma, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"noise{sigma}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{sigma:>7} | " + " | ".join(f"{c:>9}" for c in cells))

print("\n=== 3. Отказ аудио на последних X% клипа (F1 mean±std) ===")
print(f"{'outage':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
for of in (0.1, 0.25, 0.5):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, outage_frac=of, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"outage{of}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{of:>7} | " + " | ".join(f"{c:>9}" for c in cells))

with open(f"{ROOT}/research/stress_sim_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stress", "policy", "F1_mean", "F1_std"])
    w.writeheader()
    w.writerows(csv_out)
print(f"\nCSV: research/stress_sim_results.csv ({len(csv_out)} строк)")
