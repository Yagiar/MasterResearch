#!/usr/bin/env python3
"""it-83 — разбор sanity-прогона после фикса media_ts (режим папки mmaud_replay).

Использование:
  research/.venv/bin/python research/it83_sanity_eval.py [--burn-in-s 60] A=START:END B=START:END
    (срезы — 1-based строки decisions.jsonl из «SCORE_OFF» лога it83_sanity.sh)

Критерии (предрегистрированы в it-83):
  * S1 (таймбейс): для каждого этапа отношение медиа-размаха к wall-размаху
    (max−min media_ts) / (max−min ts) ∈ [0,8; 1,2]. До фикса (fps_nominal=30 при
    подаче 2 к/с) expected ratio ≈ 0,067 (сжатие ×15); после — ≈1.
  * S2 (направление FP не перевернулось): кадровая FP-доля (уникальный media_ts,
    decision=true, после burn-in) FP(B AND@0,4) < FP(A off). Абсолютные значения
    здесь не канон (этап короткий), проверяется только знак эффекта it-82.
Артефакт: research/it83_sanity_summary.txt.
"""
import json
import sys

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"

argv = sys.argv[1:]
BURN = 60.0
if "--burn-in-s" in argv:
    i = argv.index("--burn-in-s")
    BURN = float(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]

out = []
for spec in argv:
    name, rng = spec.split("=")
    a, b = (int(x) for x in rng.split(":"))
    part = rows[a - 1 : b]
    if not part:
        print(f"{name}: ПУСТОЙ СРЕЗ {rng}")
        out.append((name, 0, float("nan"), float("nan"), 0, 0, float("nan")))
        continue
    mts = [r["media_ts"] for r in part if r.get("media_ts") is not None]
    tss = [r["ts"] for r in part]
    media_span = max(mts) - min(mts)
    wall_span = max(tss) - min(tss)
    ratio = media_span / wall_span if wall_span > 0 else float("nan")
    t0 = min(tss)
    frames: dict[float, bool] = {}
    for r in part:
        if r.get("media_ts") is None or r["ts"] - t0 < BURN:
            continue
        m = r["media_ts"]
        frames[m] = frames.get(m, False) or bool(r.get("decision"))
    nf = len(frames)
    kf = sum(frames.values())
    fp = kf / nf if nf else float("nan")
    out.append((name, len(part), ratio, media_span, nf, kf, fp))
    print(f"{name}: решений {len(part)} | media_span {media_span:.1f} с | ratio media/wall {ratio:.3f} "
          f"| после burn-in кадров {nf}, положит {kf}, FP {fp:.4f}")

print(f"\nburn-in {BURN:.0f} с; единица — отправленный кадр (уникальный media_ts); GT: весь материал бездронный.")
for name, nd, ratio, ms, nf, kf, fp in out:
    ok = (ratio == ratio) and 0.8 <= ratio <= 1.2
    print(f"S1 {name}: ratio={ratio:.3f} ∈ [0,8;1,2] — {'🟢' if ok else '🔴'}")
s2 = "n/a"
if len(out) >= 2 and out[0][6] == out[0][6] and out[1][6] == out[1][6]:
    fpA, fpB = out[0][6], out[1][6]
    s2 = fpB < fpA
    print(f"S2 направление: FP(B)={fpB:.4f} < FP(A)={fpA:.4f}, Δ={fpB - fpA:+.4f} — {'🟢' if s2 else '🔴'}")

with open(f"{ROOT}/research/it83_sanity_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"it-83 sanity (strict400 HD-фоны, fps=2, фикс media_ts режима папки; burn-in {BURN:.0f} с)\n\n")
    for name, nd, ratio, ms, nf, kf, fp in out:
        ok = (ratio == ratio) and 0.8 <= ratio <= 1.2
        f.write(f"{name}: решений {nd}, media_span {ms:.1f} с, ratio media/wall {ratio:.4f} "
                f"S1 {'OK' if ok else 'FAIL'}, кадров {nf}, положит {kf}, FP {fp:.4f}\n")
    f.write(f"S2 направление FP(B AND@0,4)<FP(A off): {s2}\n")
