#!/usr/bin/env python3
"""Скоринг этапов ablation по срезам decisions.jsonl против GT (it-30 → it-45).

Использование:
  research/.venv/bin/python research/score_stages.py [--burn-in-s 90] B=START:END C=...   # медиа-время
  research/.venv/bin/python research/score_stages.py --fit-phase B=...                    # ДИАГНОСТИКА: подгонка фазы по GT

Выравнивание к GT:
  * по умолчанию — событийное время `media_ts` решения (it-34/35): clip_time = media_ts % CLIP.
    Фаза НЕ подбирается по разметке (ревью §5.2, P0). media_ts=0 == старт клипа в симуляторе.
  * `--fit-phase` — прежний диагностический режим: wall-clock `ts`, перебор 41 сдвига ±5 с
    с выбором максимума F1 на оцениваемой разметке (оптимистично; НЕ для сравнения методов).

Вес решений (it-45, ревью §10: «единица оценки должна быть фиксирована заранее»):
  * RAW  — каждое решение весит 1 (прежний подсчёт); чувствителен к темпу обработки
    (бурст YOLO даёт до 29 решений на медиа-секунду против 1-2 в стационаре).
  * NORM — вес решения = 1 / (число решений в его медиа-секунде): каждый медиа-секундный
    слот вносит суммарный вес 1 независимо от темпа конвейера; это оценка на МЕДИА-СЕКУНДАХ,
    а не на решениях. Печатается основной.
  * `--burn-in-s S` — исключить решения первых S секунд этапа по wall-clock (переходный
    режим: стартовый бурст/догоняние аудио, it-42/43); стационарный участок скорится отдельно.

`смешанных` = решений, где в окне участвовали ОБЕ модальности (p_v и p_a не None; it-32/35).
"""
import csv
import json
import sys
from collections import Counter

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
CLIP = 72.609

gt = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]

FIT_PHASE = "--fit-phase" in sys.argv
BURN_IN_S = 0.0
argv = sys.argv[1:]
if "--burn-in-s" in argv:
    i = argv.index("--burn-in-s")
    BURN_IN_S = float(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
args = [a for a in argv if a != "--fit-phase"]

def prf(preds, weights=None):
    """P/R/F1; weights=None — каждое решение весит 1 (RAW), иначе взвешенно (NORM)."""
    if weights is None:
        weights = [1.0] * len(preds)
    tp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and t)
    fp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and not t)
    fn = sum(w for w, (p, t) in zip(weights, preds, strict=True) if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def event_ts(r):
    """Событийное время решения: media_ts (it-35) либо None, если прогон его не содержит."""
    return r.get("media_ts")

def media_second(r):
    return int(event_ts(r) % CLIP)

def norm_weights(seg):
    """Вес решения = 1 / (число решений в его медиа-секунде): слот весит 1 (it-45)."""
    cnt = Counter(media_second(r) for r in seg)
    return [1.0 / cnt[media_second(r)] for r in seg]

def report(seg, label, t0=None, wall=False):
    """Печать RAW и NORM строк по модам + ИТОГ; t0 — смещение фазы (только fit-phase)."""
    def second_of(r):
        if wall:
            return gt[int((r["ts"] - t0) % CLIP)]
        return gt[media_second(r)]

    for kind, wf in (("RAW ", lambda rs: None), ("NORM", norm_weights)):
        for mode in sorted({r["mode"] for r in seg}):
            rs = [r for r in seg if r["mode"] == mode]
            preds = [(int(r["decision"]), second_of(r)[1]) for r in rs]
            P, R, F = prf(preds, wf(rs))
            joint = sum(1 for r in rs
                        if r["contributions"]["p_a"] is not None and r["contributions"]["p_v"] is not None)
            print(f"  [{kind}] {mode:<12} vs airborne: P={P:.3f} R={R:.3f} F1={F:.3f} "
                  f"(n={len(rs)}, смешанных={joint})")
        preds = [(int(r["decision"]), second_of(r)[1]) for r in seg]
        P, R, F = prf(preds, wf(seg))
        print(f"  [{kind}] {'ИТОГ':<12} vs airborne: P={P:.3f} R={R:.3f} F1={F:.3f}")

for arg in args:
    name, rng = arg.split("=", 1)  # имя может содержать '=' (напр. "B прогрев AST k=0")
    a, b = (int(x) for x in rng.split(":"))
    seg_all = rows[a - 1:b]
    if not seg_all:
        print(f"{name}: пусто")
        continue

    if not FIT_PHASE:
        usable = [r for r in seg_all if event_ts(r) is not None]
        skipped = len(seg_all) - len(usable)
        t_min = min(r["ts"] for r in usable) if usable else 0.0
        steady = [r for r in usable if r["ts"] - t_min >= BURN_IN_S]
        print(f"\n### Этап {name}: n={len(seg_all)}, usable(media_ts)={len(usable)}"
              + (f", без media_ts={skipped}" if skipped else "")
              + f", после burn-in {BURN_IN_S:.0f}с: {len(steady)}")
        if not usable:
            continue
        print("  -- весь этап (включая переходный режим) --")
        report(usable, name)
        if steady:
            print(f"  -- стационарный режим (burn-in {BURN_IN_S:.0f} с) --")
            report(steady, name)
        continue

    # --- ДИАГНОСТИКА: wall-clock + перебор фазы по GT (оптимистично, не для сравнения методов) ---
    t_first = seg_all[0]["ts"]
    best = (-1.0, None)
    for off100 in range(-20, 21):
        t0 = t_first - off100 * 0.25
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][1]) for r in seg_all]
        P, R, F = prf(preds)
        if F > best[0]:
            best = (F, t0, P, R)
    F, t0, P, R = best
    print(f"\n### Этап {name}: n={len(seg_all)}, РЕЖИМ ПОДГОНКИ ФАЗЫ ПО GT (диагностика!), "
          f"выравнивание t0-t_first={t0 - t_first:+.2f} с")
    report(seg_all, name, t0=t0, wall=True)
