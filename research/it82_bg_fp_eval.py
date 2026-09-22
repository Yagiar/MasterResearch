#!/usr/bin/env python3
"""it-82 — разбор живого фонового прогона: FP (доля положительных решений на бездронном материале).

Использование:
  research/.venv/bin/python research/it82_bg_fp_eval.py [--burn-in-s 60] A=START:END B=... C=...
    (срезы — 1-based строки decisions.jsonl из «SCORE_OFF» лога it82_bg_run)

Единицы замера:
  * Кадровая (основная): отправленный кадр = уникальное media_ts решения; FP-доля этапа =
    положительных кадров / всех покрытых после burn-in (wall-clock ts от первого решения).
    Обоснование: в режиме папки mmaud_replay считает media_ts по _DEFAULT_FPS=30
    (mmaud_replay.py:121), а темп отправки — по requested_fps=2 → медиа-таймлайн сжат ×15,
    и предрегистрированный слот 0,5 с агрегирует ~15 РАЗНЫХ картинок (дыра в конструкции
    зафиксирована в PLANNED как пост-хок оговорка). Каждое решение здесь имеет уникальное
    media_ts → слотная и кадровая метрики расходятся ровно из-за сжатия таймлайна.
  * Слотная (предрегистрированная, сохраняется для честности): media_ts → слот
    round(media_ts*2)/2 (шаг 0,5 с), положительный, если есть decision=true.
ДИ Уилсона (95 %) по числу единиц; оговорка: loop ×1,2 → автокорреляция, ДИ оптимистичны.
GT тривиален: все окна отрицательны (дрона в материале нет).
Критерии: P0 FP(A)>0; P1 FP(B)<FP(A); P2 FP(C)>=FP(A).
Артефакты: research/it82_bg_fp.csv, research/it82_bg_summary.txt.
"""
import json
import math
import sys

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"

argv = sys.argv[1:]
BURN = 60.0
if "--burn-in-s" in argv:
    i = argv.index("--burn-in-s")
    BURN = float(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]

out = []
for spec in argv:
    name, rng = spec.split("=")
    a, b = (int(x) for x in rng.split(":"))
    part = rows[a - 1 : b]
    if not part:
        print(f"{name}: ПУСТОЙ СРЕЗ {rng}")
        continue
    media = [(r["media_ts"], bool(r.get("decision")), r["ts"]) for r in part if r.get("media_ts") is not None]
    t0 = min(r["ts"] for r in part)
    kept = [(m, dec) for m, dec, ts in media if ts - t0 >= BURN]
    # кадровая метрика: уникальное media_ts = отправленный кадр
    frames: dict[float, bool] = {}
    slots: dict[float, bool] = {}
    for m, d in kept:
        frames[m] = frames.get(m, False) or bool(d)
        s = round(m * 2) / 2
        slots[s] = slots.get(s, False) or bool(d)
    nf, kf = len(frames), sum(frames.values())
    lof, hif = wilson(kf, nf)
    fpf = kf / nf if nf else float("nan")
    n = len(slots)
    k = sum(slots.values())
    lo, hi = wilson(k, n)
    fp = k / n if n else float("nan")
    out.append((name, len(part), len(kept), nf, kf, fpf, lof, hif, n, k, fp, lo, hi))
    print(f"{name}: решений {len(part)} | после burn-in {len(kept)} | "
          f"КАДРОВ {nf} | положит {kf} | FP-кадр {fpf:.4f} ДИ [{lof:.4f}; {hif:.4f}] | "
          f"(слоты 0,5с: {n}, положит {k}, FP-слот {fp:.4f} ДИ [{lo:.4f}; {hi:.4f}])")

print(f"\nburn-in {BURN:.0f} с; основная единица — отправленный кадр (уникальный media_ts); "
      f"слот 0,5 с — предрегистрированная (сжатие таймлайна ×15 делает её агрегатом ~15 картинок); "
      f"GT: все окна отрицательны.")
lines = [l for l in out]
if len(out) >= 3:
    (_, _, _, nfA, kfA, fpfA, *_), (_, _, _, nfB, kfB, fpfB, *_), (_, _, _, nfC, kfC, fpfC, *_) = lines[:3]
    print(f"P0 разделимость FP(A)>0: {'🟢' if kfA > 0 else '🔴 (FP(A)=0 — мерить нечего, вердикт n/a для P1/P2)'} ({fpfA:.4f})")
    print(f"P1 эффект AND FP(B)<FP(A): {'🟢' if fpfB < fpfA else '🔴'} ({fpfB:.4f} vs {fpfA:.4f}, Δ={fpfB - fpfA:+.4f})")
    print(f"P2 контроль OR FP(C)>=FP(A): {'🟢' if fpfC >= fpfA else '🔴'} ({fpfC:.4f} vs {fpfA:.4f})")

with open(f"{ROOT}/research/it82_bg_fp.csv", "w", encoding="utf-8") as f:
    f.write("stage,n_decisions,after_burn_in,n_frames,fp_frames,fp_frame_frac,ci_lo,ci_hi,"
            "n_slots,fp_slots,fp_slot_frac,slot_ci_lo,slot_ci_hi\n")
    for name, nd, nk, nf, kf, fpf, lof, hif, n, k, fp, lo, hi in out:
        f.write(f"{name},{nd},{nk},{nf},{kf},{fpf:.6f},{lof:.6f},{hif:.6f},{n},{k},{fp:.6f},{lo:.6f},{hi:.6f}\n")

with open(f"{ROOT}/research/it82_bg_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"it-82 живой фоновый прогон (strict400 HD Open Images, пайплайн GPU, burn-in {BURN:.0f} с wall-clock)\n")
    f.write("этапы: A vote=off (old-only) / B AND@0,4 / C OR@0,4; fusion: watermark k=5 τ=0,5 Δ=0; fps отправки=2\n")
    f.write("единица — отправленный кадр (уникальный media_ts); ДИ Уилсона 95 % (оптимистичны: loop ×1,2)\n\n")
    for name, nd, nk, nf, kf, fpf, lof, hif, n, k, fp, lo, hi in out:
        f.write(f"{name}: кадров {nf}, положит {kf}, FP={fpf:.4f} ДИ [{lof:.4f}; {hif:.4f}] "
                f"(слоты 0,5с: {k}/{n}={fp:.4f})\n")
    if len(out) >= 3:
        (_, _, _, nfA, kfA, fpfA, *_), (_, _, _, nfB, kfB, fpfB, *_), (_, _, _, nfC, kfC, fpfC, *_) = lines[:3]
        f.write(f"\nP0 FP(A)>0: {'🟢' if kfA > 0 else '🔴'} ({fpfA:.4f})\n")
        f.write(f"P1 FP(B)<FP(A): {'🟢' if fpfB < fpfA else '🔴'} ({fpfB:.4f} vs {fpfA:.4f}, Δ={fpfB - fpfA:+.4f})\n")
        f.write(f"P2 FP(C)>=FP(A): {'🟢' if fpfC >= fpfA else '🔴'} ({fpfC:.4f} vs {fpfA:.4f})\n")
        f.write("\nОговорка: media_ts в режиме папки считается по _DEFAULT_FPS=30 при темпе 2 к/с\n")
        f.write("(mmaud_replay.py:121) → таймлайн сжат ×15; предрег. слот 0,5 с агрегирует ~15 картинок,\n")
        f.write("поэтому вердикты вынесены по кадровой метрике, слотовая приведена как предрег. контроль.\n")
