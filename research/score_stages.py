#!/usr/bin/env python3
"""Скоринг этапов ablation по срезам decisions.jsonl против GT (research/it-30 → it-35).

Использование:
  research/.venv/bin/python research/score_stages.py B=START:END C=... D=...   # медиа-время (по умолчанию)
  research/.venv/bin/python research/score_stages.py --fit-phase B=...         # ДИАГНОСТИКА: перебор фазы по GT

Выравнивание к GT:
  * по умолчанию — событийное время `media_ts` решения (it-34/35): clip_time = media_ts % CLIP.
    Фаза НЕ подбирается по разметке: media_ts=0 соответствует старту клипа в симуляторе.
    Это устраняет подгонку выравнивания по тестовому F1 (ревью GPT-6-Astra §5.2/§6, P0).
  * `--fit-phase` — прежний диагностический режим: wall-clock `ts`, перебор 41 сдвига
    ±5 с с выбором максимума F1 НА ОЦЕНИВАЕМОЙ разметке (оптимистично; только для
    диагностики дрейфа задержки, НЕ для сравнения методов).

Решения из прогонов без media_ts (it-30 и ранее) в режиме по умолчанию пропускаются
с явным счётчиком — старые прогоны скорятся только в --fit-phase.

`смешанных` = решений, где в окне участвовали ОБЕ модальности (p_v и p_a не None);
раньше считалось только наличие p_a (ревью §5.3).
"""
import csv
import json
import sys

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
CLIP = 72.609

gt = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]

FIT_PHASE = "--fit-phase" in sys.argv
args = [a for a in sys.argv[1:] if a != "--fit-phase"]

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def event_ts(r):
    """Событийное время решения: media_ts (it-35) либо None, если прогон его не содержит."""
    return r.get("media_ts")

def clip_second(r, t0):
    """Секунда клипа для решения при известном смещении t0 (шкала = шкала event_ts)."""
    return gt[int((event_ts(r) - t0) % CLIP)]

for arg in args:
    name, rng = arg.split("=")
    a, b = (int(x) for x in rng.split(":"))
    seg = rows[a - 1:b]
    if not seg:
        print(f"{name}: пусто")
        continue

    if not FIT_PHASE:
        # --- режим по умолчанию: медиа-время, фиксированная фаза (без подгонки) ---
        usable = [r for r in seg if event_ts(r) is not None]
        skipped = len(seg) - len(usable)
        print(f"\n### Этап {name}: n={len(seg)}, usable(media_ts)={len(usable)}"
              + (f", ПРОПУЩЕНО без media_ts={skipped} (прогон до it-35 — только --fit-phase)" if skipped else ""))
        if not usable:
            continue
        t0 = 0.0  # media_ts=0 == старт клипа в симуляторе; фаза не подбирается
        for mode in sorted({r["mode"] for r in usable}):
            rs = [r for r in usable if r["mode"] == mode]
            preds = [(int(r["decision"]), clip_second(r, t0)[1]) for r in rs]
            P, R, F = prf(preds)
            mixed = sum(1 for r in rs
                        if r["contributions"]["p_a"] is not None and r["contributions"]["p_v"] is not None)
            print(f"  {mode:<12} vs airborne: P={P:.3f} R={R:.3f} F1={F:.3f} (n={len(rs)}, смешанных(обе мод.)={mixed})")
        for gname, gix in (("visible", 0), ("airborne", 1)):
            preds = [(int(r["decision"]), clip_second(r, t0)[gix]) for r in usable]
            P, R, F = prf(preds)
            print(f"  ИТОГ {'все':<12} vs {gname:<8}: P={P:.3f} R={R:.3f} F1={F:.3f}")
        continue

    # --- ДИАГНОСТИКА: wall-clock + перебор фазы по GT (оптимистично, не для сравнения методов) ---
    t_first = seg[0]["ts"]
    best = (-1.0, None)
    for off100 in range(-20, 21):
        t0 = t_first - off100 * 0.25
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][1]) for r in seg]
        P, R, F = prf(preds)
        if F > best[0]:
            best = (F, t0, P, R)
    F, t0, P, R = best
    print(f"\n### Этап {name}: n={len(seg)}, РЕЖИМ ПОДГОНКИ ФАЗЫ ПО GT (диагностика!), "
          f"выравнивание t0-t_first={t0 - t_first:+.2f} с")
    for mode in sorted({r["mode"] for r in seg}):
        rs = [r for r in seg if r["mode"] == mode]
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][1]) for r in rs]
        P, R, F = prf(preds)
        mixed = sum(1 for r in rs
                    if r["contributions"]["p_a"] is not None and r["contributions"]["p_v"] is not None)
        print(f"  {mode:<12} vs airborne: P={P:.3f} R={R:.3f} F1={F:.3f} (n={len(rs)}, смешанных(обе мод.)={mixed})")
    for gname, gix in (("visible", 0), ("airborne", 1)):
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][gix]) for r in seg]
        P, R, F = prf(preds)
        print(f"  ИТОГ {'все':<12} vs {gname:<8}: P={P:.3f} R={R:.3f} F1={F:.3f}")
