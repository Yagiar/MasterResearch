#!/usr/bin/env python3
"""it-87 разбор: frozen final holdout — единственный confirmatory-замер боевой линейки.

Протокол/гейты: iterations/it-87-frozen-holdout-PLANNED.md. ЗАПУСК ОДИН РАЗ: повторный разбор
одних и тех же срезов допустим (детерминирован), повторный ПРОГОН того же сегмента — нарушение.

Аргументы: --manifest=ПУТЬ (tsv: id role video audio dur gt_start gt_end), затем срезы
"SEGID=START:END" (1-based строки decisions.jsonl из SCORE_OFF). Опции --jsonl/--out —
синтетическая самопроверка парсера (дефолт — боевые пути).

H0 (блокаторы, вердиктов не существует при провале): sha весов; audio-телеметрия на каждом
сегменте (доля окон с p_a>0 … строжайше: непустой audio-инференс — по jsonl-полю p_a);
media_ts-monotonic; окно-спана ratio (media-покрытие/ожидаемая длительность) ∈ [0,8;1,2].
H1 (neg): доля alarm-окон ≤ 0,15 (знаменатель — окна сегмента с media_ts≠null; хвост ~10
невыпущенных окон = зарегистрированная tail-потеря, в знаменатель не входит и не добавляется).
H2 (pos): sequence-recall ≥ 0,90: в пролёте ≥1 alarm-окно с media_ts ∈ [gt_start, gt_end].
H3 (pos): медиана TTD ≤ 10 с, p90 ≤ 20 с; TTD = первый alarm media_ts ≥ gt_start минус gt_start.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
JSONL = MD / "data/decisions/decisions.jsonl"
MANIFEST = MD / "sandboxDataForSimulator/holdout-24/manifest.tsv"
OUT = ROOT / "research/it87_holdout_summary.txt"
CSV = ROOT / "research/it87_holdout.csv"
SHA_OLD = "41f3fd55"
SHA_NEW = "7042602a"
TAIL_LOSS_S = 10.0  # ~10 окон × 0,5 с: watermark-хвост конечного прохода (it-86)

WEIGHTS = MD / "models/visual"
argv = sys.argv[1:]
for a in [x for x in argv if x.startswith("--")]:
    if "=" not in a:
        sys.exit(f"ОТКАЗ: опция '{a}' должна быть вида --jsonl=ПУТЬ / --manifest=ПУТЬ (голый флаг молча проглотился бы)")
    k, v = a.split("=", 1)
    if k == "--jsonl":
        JSONL = Path(v)
    elif k == "--manifest":
        MANIFEST = Path(v)
    elif k == "--out":
        OUT = Path(v)
        CSV = Path(v + ".csv")
    elif k == "--weights-dir":
        WEIGHTS = Path(v)
    else:
        sys.exit(f"ОТКАЗ: неизвестная опция '{k}' (опечатка? иначе аргумент молча исчез из срезов)")
sys.argv = argv = [x for x in argv if not x.startswith("--")]
assert argv, 'передай срезы "SEGID=START:END" и --manifest'

LOG = []


def p(line=""):
    print(line)
    LOG.append(line)


def sha_prefix(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_manifest(path):
    segs = {}
    # utf-8-sig: срезает BOM, если автор пересохранит манифест «UTF-8 с BOM» (Notepad/Excel)
    for ln in path.read_text(encoding="utf-8-sig").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        assert len(f) >= 7, f"манифест: мало полей в '{ln}'"
        assert f[1] in ("pos", "neg"), f"манифест: role='{f[1]}' у '{f[0]}' (только pos|neg)"
        segs[f[0]] = dict(id=f[0], role=f[1], video=f[2], audio=f[3],
                           dur=float(f[4]), gs=float(f[5]), ge=float(f[6]))
    return segs


def main(slices):
    segs = load_manifest(MANIFEST)
    lines = JSONL.read_text(encoding="utf-8").splitlines()
    p("it-87 — frozen final holdout (боевая линейка AND@0,40 + audio gate; однократное открытие)")

    # --- H0.1: sha весов ---
    h0 = []
    for fn, want in (("yolov8s-uav.pt", SHA_OLD), ("uav-yolov8s-bg-best.pt", SHA_NEW)):
        got = sha_prefix(WEIGHTS / fn)
        ok = got == want
        h0.append(ok)
        p(f"H0 sha {fn}: {got} (канон {want}…) → {'OK' if ok else 'ПРОВАЛ'}")

    per = {}
    for arg in slices:
        sid, rng = arg.split("=")
        a, b = rng.split(":")
        assert sid in segs, f"срез '{sid}' нет в манифесте"
        recs = [json.loads(x) for x in lines[int(a) - 1:int(b)]]
        # mono — по ПОРЯДКУ ПОТОКА (jsonl), не по отсортированному: sorted-ряд монотонен
        # тривиально и гейт H0 был бы пустым (найдено пре-открыточным аудитом 24.09)
        mm_stream = [(r["media_ts"], r["decision"]) for r in recs if r.get("media_ts") is not None]
        mono = all(x[0] <= y[0] + 1e-9 for x, y in zip(mm_stream, mm_stream[1:]))
        mm = sorted(mm_stream)
        pa = sum(1 for r in recs if (r.get("contributions") or {}).get("p_a") is not None)
        span = (mm[-1][0] - mm[0][0]) if mm else 0.0
        exp = segs[sid]["dur"] - TAIL_LOSS_S
        ratio = span / exp if exp > 0 else float("nan")
        per[sid] = dict(n=len(recs), mm=mm, mono=mono, pa_frac=pa / max(len(recs), 1),
                        ratio=ratio)
        p(f"[{sid}] role={segs[sid]['role']} окон={len(recs)} p_a≠null={100*per[sid]['pa_frac']:.0f} % "
          f"media-покрытие={span:.0f} с ratio={ratio:.3f} mono={'OK' if mono else 'ПРОВАЛ'}")

    # --- H0.2/3/4: блокаторы ---
    h0 += [all(per[s]["pa_frac"] > 0 for s in per),
           all(per[s]["mono"] for s in per),
           all(0.8 <= per[s]["ratio"] <= 1.2 for s in per)]
    if len(per) != len(segs):
        p(f"пройдено сегментов: {len(per)} из {len(segs)} — не все → H0 ПРОВАЛ")
    h0_ok = all(h0) and len(per) == len(segs)
    p(f"\nH0 (блокаторы: sha, аудио-телеметрия, media_ts-monotonic, ratio∈[0,8;1,2], полнота) → "
      f"{'ПРОЙДЕН' if h0_ok else 'ПРОВАЛ — вердиктов H1–H3 НЕ СУЩЕСТВУЮТ; чинить обвязку, повторный прогон — только с явной пометкой о первом пуске'}")
    if not h0_ok:
        report(slices, per, segs, None, None, None)
        return

    # --- H1: негативы ---
    neg_ids = [s for s in per if segs[s]["role"] == "neg"]
    alarms = windows = 0
    for s in neg_ids:
        alarms += sum(1 for t, d in per[s]["mm"] if d)
        windows += len(per[s]["mm"])
    h1_rate = alarms / windows if windows else float("nan")
    h1 = h1_rate <= 0.15
    p(f"\nH1 (FP на негативах): alarm-окна {alarms}/{windows} = {100*h1_rate:.1f} % ≤ 15 % → "
      f"{'🟢 ПОДТВЕРЖДЕНО' if h1 else '🔴 КРАСНО (фиксация без перекалибровки)'}")

    # --- H2/H3: позитивы ---
    pos_ids = sorted(s for s in per if segs[s]["role"] == "pos")
    caught, ttds = 0, []
    for s in pos_ids:
        gs, ge = segs[s]["gs"], segs[s]["ge"]
        gt = [t for t, d in per[s]["mm"] if d and gs <= t <= ge]
        if gt:
            caught += 1
            ttds.append(max(0.0, min(gt) - gs))
        p(f"  {s}: alarm∈[{gs:.0f},{ge:.0f}] {'ЕСТЬ (TTD ' + format(min(gt)-gs, '.1f') + ' с)' if gt else 'НЕТ'}")
    n = len(pos_ids)
    h2_val = caught / n if n else float("nan")
    h2 = n >= 5 and h2_val >= 0.90
    p(f"\nH2 (sequence-recall): {caught}/{n} = {100*h2_val:.0f} % ≥ 90 % (n≥5) → "
      f"{'🟢' if h2 else '🔴'}")
    if ttds:
        med = statistics.median(ttds)
        # nearest-rank (предрегистр.): ceil(0,9·n)-й ряд; round даёт banker's rounding и
        # при n=5 (min spec pos≥5) брал 4-й ряд вместо max-ряда → гейт слабел
        p90 = sorted(ttds)[-(-9 * len(ttds) // 10) - 1]  # ceil(0,9n)-й ряд
        h3 = med <= 10.0 and p90 <= 20.0
        p(f"H3 (TTD): медиана {med:.1f} с ≤10 ∧ p90 {p90:.1f} с ≤20 → {'🟢' if h3 else '🔴'}")
    else:
        h3, med, p90 = False, float("nan"), float("nan")
        p("H3 (TTD): нет пойманных пролётов → 🔴")
    verdict = "ПОДТВЕРЖДЕНО" if (h1 and h2 and h3) else "НЕ ПОДТВЕРЖДЕНО"
    p(f"\nВЕРДИКТ it-87 (H0∧H1∧H2∧H3): {verdict}. Красный — честная фиксация; политика/τ не меняются.")
    report(slices, per, segs, (h1_rate, alarms, windows), (caught, n), (med, p90, ttds))


def report(slices, per, segs, h1, h2, h3):
    with open(CSV, "w", encoding="utf-8") as f:
        f.write("segment,role,n_windows,n_alarm,media_span_s,ratio,ttd_s,caught\n")
        for sid in per:
            mm = per[sid]["mm"]
            gs, ge = segs[sid]["gs"], segs[sid]["ge"]
            gt = [t for t, d in mm if d and gs <= t <= ge] if segs[sid]["role"] == "pos" else []
            ttd = (f"{max(0.0, min(gt) - gs):.1f}" if gt else "") if segs[sid]["role"] == "pos" else ""
            f.write(f"{sid},{segs[sid]['role']},{len(mm)},{sum(1 for _, d in mm if d)},"
                    f"{(mm[-1][0]-mm[0][0]) if mm else 0:.1f},{per[sid]['ratio']:.3f},"
                    f"{ttd},{1 if gt else 0 if segs[sid]['role']=='pos' else ''}\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("it-87 holdout — срез stdout (протокол: it-87-frozen-holdout-PLANNED.md)\n\n"
                + "\n".join(LOG) + "\n")
    p(f"\nСводка: {OUT}; сегментная таблица: {CSV}")


if __name__ == "__main__":
    main(argv)
