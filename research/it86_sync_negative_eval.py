#!/usr/bin/env python3
"""it-86 разбор: мультимодальный синхронный негатив V0/V1/V2 (GT «дрона нет» → alarm = FA).

Аргументы: имена-срезы вида "V0 off-audio=START:END" (1-based строки decisions.jsonl, из SCORE_OFF).
Протокол/гейты: iterations/it-86-multimodal-sync-negative-PLANNED.md.
M0 (блокатор телеметрии): в V1/V2 доля решений с непустым contributions.p_a > 0 И audio-окна
фактически покрывают сегмент — иначе V1/V2 не интерпретируются (урок it-30).
При T_сегмента < 60 с прогон — SANITY обвязки: FA/hour не является вердиктом (широкий ДИ,
автокорреляция), выводятся только дескриптивные числа.
Объединённый срез плеча (it-86-full: N сегментов подряд): таймлайн media_file перезапускается
с 0 на каждом файле (media_base=0), поэтому span=max−мин НЕ годится; покрытие считается как
сумма union-длин окон по сегментам, отделённым по перезапуску медиа-шкалы (drop > RESTART_GAP).
Запуск: research/.venv/bin/python research/it86_sync_negative_eval.py "V0 off-audio=1:35" ...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "MasterDiploma/data/decisions/decisions.jsonl"
OUT = ROOT / "research/it86_sync_summary.txt"
# опциональные пути для синтетической самопроверки парсера (дефолт — боевые)
for _a in [x for x in sys.argv[1:] if x.startswith("--")]:
    if "=" not in _a:
        sys.exit(f"ОТКАЗ: опция '{_a}' должна быть вида --jsonl=ПУТЬ / --out=ПУТЬ")
    k, v = _a.split("=", 1)
    if k == "--jsonl":
        JSONL = Path(v)
    elif k == "--out":
        OUT = Path(v)
sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:] if not x.startswith("--")]
SANITY_MAX_T = 60.0
RESTART_GAP = 5.0  # с: падение конца окна ниже running max на бо́льшее — перезапуск медиа-шкалы
LOG = []


def union_len(intervals):
    total, cur_e = 0.0, None
    cur_s = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def coverage(recs):
    """Покрытие медиа-шкалы (с) и число сегментов в срезе (по перезапускам таймлайна)."""
    segs, cur, run_max = [], [], float("-inf")
    for r in recs:
        if r.get("media_ts") is None:
            continue  # wall-clock обвязка до первого media_ts — в охват не входит
        t0, t1 = sorted((r["ts_window"] + [0.0, 0.0])[:2])
        if cur and t1 < run_max - RESTART_GAP:
            segs.append(cur)
            cur, run_max = [], float("-inf")  # без сброса max каждый кадр второго
                                              # сегмента ложно считался бы перезапуском
        cur.append((t0, t1))
        run_max = max(run_max, t1)
    if cur:
        segs.append(cur)
    return sum(union_len(iv) for iv in segs), len(segs)


def p(line=""):
    print(line)
    LOG.append(line)


def main(slices):
    lines = JSONL.read_text(encoding="utf-8").splitlines()
    p("it-86 — мультимодальный синхронный негатив (GT: дрона нет → каждое decision=true = FA)")
    res = {}
    for name, a, b in slices:
        recs = [json.loads(x) for x in lines[a - 1:b]]
        n = len(recs)
        fa = sum(1 for r in recs if r["decision"])
        pa = sum(1 for r in recs if (r.get("contributions") or {}).get("p_a") is not None)
        # покрытие: union по окнам внутри сегментов, сегменты разделены по перезапуску
        # медиа-шкалы (каждый файл media_file стартует с 0) — span max−мин на объединённом
        # multi-segment срезе занижал бы знаменатель FA/hour до одного сегмента
        span, nseg = coverage(recs)
        res[name] = dict(n=n, fa=fa, pa=pa, span=span, nseg=nseg,
                         fa_rate=(fa / n if n else float("nan")),
                         fa_hr=(fa * 3600.0 / span if span > 0 else float("nan")))
        p(f"\n[{name}] окон={n}, FA={fa} ({100 * fa / max(n, 1):.1f} %), "
          f"окон с p_a≠null={pa} ({100 * pa / max(n, 1):.1f} %), "
          f"медиа-покрытие={span:.1f} с (сегм. {nseg}), FA/hour≈{res[name]['fa_hr']:.0f}")

    names = [s[0] for s in slices]
    sanity = all(res[nm]["span"] < SANITY_MAX_T for nm in names)
    p("\nМ0 (блокатор телеметрии аудио):")
    m0_ok = True
    for nm in names:
        if "+audio" in nm:
            okk = res[nm]["pa"] > 0
            m0_ok = m0_ok and okk
            p(f"  {nm}: окон с p_a≠null = {res[nm]['pa']} → {'OK' if okk else 'ПРОВАЛ (аудио не текло)'}")
    p(f"  M0 → {'ПРОЙДЕН' if m0_ok else 'НАРУШЕН — V1/V2 не интерпретируются, чинить обвязку'}")
    if sanity:
        p(f"\nАТРИБУЦИЯ: медиа-покрытие < {SANITY_MAX_T:.0f} с на всех плечах → это SANITY ОБВЯЗКИ "
          "(pre-registration it-86): вердикты M1–M3 не выносятся, числа — порядок величины "
          "и проверка, что контур жив и аудио-гейт подключён.")
    elif m0_ok:
        p("\nСегменты в плечах (сверить с числом neg-сегментов манифеста; "
          "меньше — часть сегментов не попала в срез): "
          + ", ".join(f"{nm}: {res[nm]['nseg']}" for nm in names))
        v0, v1 = res.get("V0 off-audio"), res.get("V1 +audio")
        if v0 and v1:
            d = "подавление" if v1["fa"] < v0["fa"] else ("добавление" if v1["fa"] > v0["fa"] else "нейтрально")
            p(f"\nM2 (вклад аудио, V1 vs V0): FA {v0['fa']} → {v1['fa']} — направление: {d} "
              "(двусторонний ДИ по окнам — см. PLANNED; сила не заявляется без bootstrap).")
        v2 = res.get("V2 and+audio")
        if v2:
            p(f"\nM3 (боевая V2): FA-доля {100 * v2['fa_rate']:.1f} %, FA/hour≈{v2['fa_hr']:.0f}.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("it-86 sync negative — срез stdout (протокол: iterations/it-86-multimodal-sync-negative-PLANNED.md)\n\n"
                + "\n".join(LOG) + "\n")
    p(f"\nСводка: {OUT}")


if __name__ == "__main__":
    sl = []
    for arg in sys.argv[1:]:
        nm, rng = arg.split("=", 1)
        a, b = rng.split(":")
        sl.append((nm, int(a), int(b)))
    assert sl, "передай срезы вида \"V0 off-audio=1:35\""
    main(sl)
