"""it-79: анализ SAHI-FP на HD-фонах (Open Images) — ряды old/new/AND/OR + вердикт X0–X2.

Вход: it79_old.csv/it79_new.csv (harness it73_coco_sahi_fp.py: img,n_boxes,max_conf,...)
и manifest отбора (it79_manifest.csv — копия DATA/manifest.csv). Выход: it79_sahi_hd_fp.{csv,txt}.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research"

TAUS = [0.30, 0.35, 0.40, 0.45, 0.50]
AND_MAIN = 0.40
X1_LIMIT = 12.8          # порог E1/V1, %
V1_FP = 7.2              # канон AND@0,40 на 640-px COCO, %
X2_BAND = 8.0            # диапазон consistency it-76 (± п.п.)
N_EXPECT = 400
MIN_SIDE_EXPECT = 768


def load_rows(path: Path) -> dict[str, tuple[float, int]]:
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        out[row["img"]] = (float(row["max_conf"]), int(row["n_boxes"]))
    return out


def selfcheck(path: Path) -> None:
    """Сами-проверка ряда: fp@0.4 в CSV harness обязан точно равняться (max_conf >= 0.40)."""
    for row in csv.DictReader(open(path, encoding="utf-8")):
        assert (float(row["max_conf"]) >= 0.40) == bool(int(row["fp@0.4"])), f"ряд {path.name}: {row['img']}"


def main() -> None:
    selfcheck(OUT / "it79_old.csv")
    selfcheck(OUT / "it79_new.csv")
    old = load_rows(OUT / "it79_old.csv")
    new = load_rows(OUT / "it79_new.csv")
    man_path = OUT / "it79_manifest.csv"
    bird: dict[str, int] = {}
    dims: list[int] = []
    if man_path.exists():
        for row in csv.DictReader(open(man_path, encoding="utf-8")):
            if row["status"] == "accepted":
                bird[f"{row['ImageID']}.jpg"] = int(row["bird"])
                dims.append(min(int(row["w"]), int(row["h"])))

    keys = sorted(set(old) & set(new))
    n = len(keys)
    # X0: блокатор выборки
    x0_ok = n == N_EXPECT and (not dims or min(dims) >= MIN_SIDE_EXPECT)
    lines = []
    lines.append("it-79: SAHI-FP на HD-фонах Open Images (tile 640/0,2; max_conf на кадр)")
    lines.append(f"X0: кадров в пересечении old∩new = {n} (ожидалось {N_EXPECT}); "
                 f"факт min-сторона: {'n/a' if not dims else f'min={min(dims)} / max={max(dims)}'}; "
                 f"лицензии CC и ¬aerial — по манифесту отбора → {'OK' if x0_ok else 'НАРУШЕН'}")
    if not x0_ok:
        lines.append("ВЕРДИКТ: X0 нарушен → замеры не интерпретируются (см. PLANNED).")
        (OUT / "it79_sahi_hd_fp.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    rows = []
    for tau in TAUS:
        for name, series in [
            ("old", [old[k][0] for k in keys]),
            ("new", [new[k][0] for k in keys]),
            ("AND", [min(old[k][0], new[k][0]) for k in keys]),
            ("OR", [max(old[k][0], new[k][0]) for k in keys]),
        ]:
            hits = sum(c >= tau for c in series)
            rows.append({"tau": f"{tau:.2f}", "row": name, "fp_pct": round(100 * hits / n, 1), "n_frames_hit": hits})
    with open(OUT / "it79_sahi_hd_fp.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "row", "fp_pct", "n_frames_hit"])
        w.writeheader()
        w.writerows(rows)

    fp_and = {r["tau"]: r["fp_pct"] for r in rows if r["row"] == "AND"}
    fp_main = fp_and[f"{AND_MAIN:.2f}"]
    x1 = fp_main <= X1_LIMIT
    x2 = abs(fp_main - V1_FP) <= X2_BAND
    med_tiles = sum(old[k][1] for k in keys) / n  # n_boxes harness = срабатывания, не тайлы — справка
    bird_share = 100 * sum(bird.get(k, 0) for k in keys) / n if bird else float("nan")

    lines.append("")
    lines.append("FP %, τ × ряд (old/new/AND/OR):")
    for tau in TAUS:
        vals = {r["row"]: r["fp_pct"] for r in rows if r["tau"] == f"{tau:.2f}"}
        lines.append(f"  τ={tau:.2f}: old {vals['old']:5.1f} | new {vals['new']:5.1f} | AND {vals['AND']:5.1f} | OR {vals['OR']:5.1f}")
    lines.append("")
    lines.append(f"X1 (главный): FP(AND@{AND_MAIN:.2f}) = {fp_main} % ≤ {X1_LIMIT} → {'ЗЕЛЁНЫЙ' if x1 else 'КРАСНЫЙ'}")
    lines.append(f"X2 (consistency V1=7,2 %): |{fp_main} − 7,2| = {abs(fp_main - V1_FP):.1f} п.п. ≤ {X2_BAND} → {'ЗЕЛЁНЫЙ' if x2 else 'КРАСНЫЙ'}")
    lines.append(f"X3 справочно: средняя доля срабатываний/кадр old={sum(old[k][1] for k in keys)/n:.2f}, "
                 f"new={sum(new[k][1] for k in keys)/n:.2f}; доля кадров с птицами в аннотациях = {bird_share:.1f} %")
    if x1 and x2:
        lines.append("ВЕРДИКТ: X1∧X2 зелёные → FP-доказательство AND@0,40 робастно к режиму входа "
                     "(full-frame 640 → SAHI HD); W4-ось закрыта для ансамбля измеренным фактом.")
    else:
        lines.append("ВЕРДИКТ: X1/X2 красный → регистрируется НОВОЕ ограничение профиля ансамбля: "
                     "в HD/SAHI-режиме FP смещается (число выше); вердикт it-73 не меняется.")
    txt = "\n".join(lines) + "\n"
    (OUT / "it79_sahi_hd_fp.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
