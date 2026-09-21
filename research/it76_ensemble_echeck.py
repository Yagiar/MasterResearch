#!/usr/bin/env python3
"""it-76: покрытие E1–E8 ансамблем old⊕new — матрица + доизмеры Ф/С, ноль инференса.

Барьеры предрегистрированы в iterations/it-76-ensemble-e-coverage-PLANNED.md
(коммит 1f99eba): Ф — |FP(min-ряд≥0,4) на 90 старых фонах − 7,2 %| ≤ 8 п.п.;
С — presence min-ряда ≥0,4 на 18 с negative-session ≥ 16/18.

Запуск: research/.venv/bin/python research/it76_ensemble_echeck.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = []
TAU = 0.4


def p(line: str = "") -> None:
    print(line)
    LOG.append(line)


def load(path: str, col: str) -> dict[str, float]:
    return {r["img"]: float(r[col]) for r in csv.DictReader(open(ROOT / path, encoding="utf-8"))}


# ---- Ф: 90 старых фонов it-65 (col max_conf; сами по себе не «old V1 400» — другой сэмпл)
old_b = load("research/coco_bg_fp_old-yolov8s-uav.csv", "max_conf")
new_b = load("research/coco_bg_fp_new-yolov8s-bg.csv", "max_conf")
bk = sorted(set(old_b) & set(new_b))
fp90 = sum(1 for k in bk if min(old_b[k], new_b[k]) >= TAU) / len(bk)
p(f"Ф (90 старых фонов it-65, n={len(bk)}): FP AND@{TAU} = {fp90:.1%} "
  f"(single old@0,5 = {sum(1 for k in bk if old_b[k] >= 0.5) / len(bk):.1%} — канон-сверка 25,6 %±; "
  f"single new@0,5 = {sum(1 for k in bk if new_b[k] >= 0.5) / len(bk):.1%})")
phi_ok = abs(fp90 - 0.072) <= 0.08
p(f"  барьер |Ф − 7,2 %| ≤ 8 п.п. → {'ЗЕЛЁНЫЙ' if phi_ok else 'красный'}")

# ---- С: negative-session 18 с (presence стоящего дрона, min-ряд)
old_s = {int(r["second"]): float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/session_vis_probe_old-yolov8s-uav.csv"))}
new_s = {int(r["second"]): float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/session_vis_probe_new-yolov8s-bg.csv"))}
sk = sorted(set(old_s) & set(new_s))
hit = sum(1 for s in sk if min(old_s[s], new_s[s]) >= TAU)
p(f"\nС (negative-session, n={len(sk)} с): presence AND@{TAU} = {hit}/{len(sk)} = {hit / len(sk):.1%}")
p(f"  справочно single: old {sum(1 for s in sk if old_s[s] >= TAU)}/{len(sk)}, "
  f"new {sum(1 for s in sk if new_s[s] >= TAU)}/{len(sk)}; old@0,5 = {sum(1 for s in sk if old_s[s] >= 0.5)}/{len(sk)} (канон E4 18/18)")
chi_ok = hit >= 16
p(f"  барьер ≥ 16/18 → {'ЗЕЛЁНЫЙ' if chi_ok else 'красный'}")

# ---- Матрица покрытия (документационный результат)
p("\nМатрица E1–E8 × ансамбль (AND@0,40 image / OR-ряд контур):")
for row in [
    "  E1 фоны: ✅ it-74 V1-400 = 7,2 % + it-76 Ф (этот прогон) — consistency",
    "  E2 DUT600 mAP: ⬜ пробел — нужен merge боксов (GPU-predict, будущая итерация)",
    "  E3 HF600 mAP: ⬜ пробел — тот же merge",
    "  E4 стоящий presence: ✅ it-76 С (этот прогон)",
    "  E5 SAHI-полёт: ✅ it-74 = 94,7 % [92,2;96,9]",
    "  E6 контур airborne: ✅ it-74 (AND R 0,946 красный) + it-75 (OR R 1,000 зелёный)",
    "  E7/E8 аудио/stress: n/a по конструкции (визуальное голосование аудио-ветку не меняет)",
    "  живой A/B: ⬜ вне офлайна (правка visual-detector на 2 модели — ждёт решения автора о ×2-доставке)",
]:
    p(row)
p(f"\nВЕРДИТ доизмеров: {'Ф и С зелёные — ансамбль не хуже канона на переиспользуемых E-офлайн осях' if phi_ok and chi_ok else 'ЕСТЬ КРАСНЫЙ доизмер — профиль ансамбля помечен пятном (см. строки выше)'}")

OUT = ROOT / "research/it76_ensemble_echeck.txt"
with open(OUT, "w", encoding="utf-8") as f:
    f.write("it-76 ансамбль × E-линейка — срез stdout (протокол: iterations/it-76-ensemble-e-coverage-PLANNED.md)\n\n")
    f.write("\n".join(LOG) + "\n")
p(f"\nTXT: {OUT}")
