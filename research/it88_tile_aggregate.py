#!/usr/bin/env python3
"""it-88 анализатор: tile-aware агрегации поверх per-tile дампов (ноль инференса).

Входы (артефакты it88_sahi_tiled.py):
  research/it88_bg400_{old,new}.csv   — FP-ось (400 HD-фонов Open Images, дрон отсутствует);
  research/it88_mmaud_{old,new}.csv   — recall-guardrail (подмножество flight, stride).
Политики по КАЖДОМУ ряду (frame-level из tile_confs/full_conf640), затем ансамбль = AND двух
рядов по тому же правилу:
  MAX      := max(tile_confs) ≥ τ                       (текущий SAHI; воспроизводит it-79)
  CONS-k   := #{tiles ≥ τ} ≥ k                          (пересечение соседних тайлов реальным дроном)
  NULL     := #{tiles ≥ τ} ≥ c, c = min{m: P[Binom(n,p̂_τ) ≥ m] ≤ α}  (поправка на размер выборки)
  FULL     := (max ≥ τ) ∧ (full_conf640 ≥ τ)            (full-frame corroboration)
τ-сетка {0.30,0.35,0.40,0.45,0.50}, основная τ=0.40; α=0.05; k∈{2,3}.
Сами-проверки канонов + гейт T1 (exploratory) — см. PLANNED. Выход: it88_tile_agg.{csv,txt}.
Запуск: research/.venv/bin/python research/it88_tile_aggregate.py
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research"
TAUS = [0.30, 0.35, 0.40, 0.45, 0.50]
MAIN = 0.40
ALPHA = 0.05
KS = [2, 3]
FP_X1 = 12.8          # цель FP(AND@τ), % — порог E1/V1
RECALL_FLOOR = 85.0   # guardrail recall(AND@τ), %
MAX_IT79_FP = 48.8    # канон it-79 MAX-AND@0,40 на strict400, %
IT79_TOL = 3.0
LOG = []


def p(line=""):
    print(line)
    LOG.append(line)


def ok(name, cond):
    p(f"  [{'OK' if cond else 'ПРОВАЛ'}] {name}")
    return bool(cond)


def load(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        tc = [float(x) for x in r["tile_confs"].split(";")] if r["tile_confs"] else []
        d[r["img"]] = (tc, float(r["full_conf640"]), int(r["n_tiles"]))
    return d


def binom_tail_ge(n, pr, m):
    """P[Binom(n,pr) ≥ m]."""
    if m <= 0:
        return 1.0
    s = 0.0
    for j in range(m, n + 1):
        s += math.comb(n, j) * (pr ** j) * ((1 - pr) ** (n - j))
    return s


def null_c(n_tiles, p_hat, alpha):
    for m in range(1, n_tiles + 1):
        if binom_tail_ge(n_tiles, p_hat, m) <= alpha:
            return m
    return n_tiles + 1  # никогда не срабатывает


def decide(tc, full, policy, tau, p_hat):
    """frame-level решение по одному ряду."""
    if not tc:
        return False
    exceed = sum(1 for c in tc if c >= tau)
    n = len(tc)
    if policy == "MAX":
        return max(tc) >= tau
    if policy == "FULL":
        return (max(tc) >= tau) and (full >= tau)
    if policy.startswith("CONS"):
        k = int(policy[4:])
        return exceed >= k
    if policy == "NULL":
        return exceed >= null_c(n, p_hat, ALPHA)
    raise ValueError(policy)


def tile_null_rate(*series, tau):
    """p̂_τ = доля фоновых тайлов с conf≥τ (оценка нуль-ставки на bg400)."""
    tot = hit = 0
    for tc_list in series:
        for tc in tc_list:
            for c in tc:
                tot += 1
                if c >= tau:
                    hit += 1
    return hit / max(1, tot)


def main():
    policies = ["MAX", "FULL"] + [f"CONS{k}" for k in KS] + ["NULL"]
    bg_old = load(OUT / "it88_bg400_old.csv")
    bg_new = load(OUT / "it88_bg400_new.csv")
    md_old = load(OUT / "it88_mmaud_old.csv")
    md_new = load(OUT / "it88_mmaud_new.csv")
    bkeys = sorted(set(bg_old) & set(bg_new))
    mkeys = sorted(set(md_old) & set(md_new))
    assert len(bkeys) == 400, f"bg400 join: {len(bkeys)}"
    p(f"bg400 (FP-ось): {len(bkeys)} | mmaud subset (recall): {len(mkeys)}")

    # нуль-ставка по фоновым тайлам (объединённая old+new)
    p_hat = {tau: tile_null_rate([bg_old[k][0] for k in bkeys], [bg_new[k][0] for k in bkeys], tau=tau)
             for tau in TAUS}
    p("\nнуль-ставка фонового тайла p̂_τ (объед. old+new, bg400):")
    for tau in TAUS:
        p(f"  τ={tau:.2f}: p̂={p_hat[tau]:.3%}")

    rows = []
    for pol in policies:
        for tau in TAUS:
            ph = p_hat[tau]
            fp = sum(1 for k in bkeys
                     if decide(bg_old[k][0], bg_old[k][1], pol, tau, ph)
                     and decide(bg_new[k][0], bg_new[k][1], pol, tau, ph)) / len(bkeys)
            rec = sum(1 for k in mkeys
                      if decide(md_old[k][0], md_old[k][1], pol, tau, ph)
                      and decide(md_new[k][0], md_new[k][1], pol, tau, ph)) / len(mkeys)
            rows.append(dict(policy=pol, tau=round(tau, 2), fp_pct=round(100 * fp, 1),
                             recall_pct=round(100 * rec, 1)))

    # ---- сами-проверки (блокатор интерпретации) ----
    p("\nсами-проверки канонов:")
    max_main = next(r for r in rows if r["policy"] == "MAX" and r["tau"] == round(MAIN, 2))
    sc1 = ok(f"MAX-AND@0,40 FP={max_main['fp_pct']} % ≈ канон it-79 {MAX_IT79_FP} ±{IT79_TOL}",
             abs(max_main["fp_pct"] - MAX_IT79_FP) <= IT79_TOL)
    # recall MAX-old solo @0.5 на подмножестве (без ансамбля) — сверка порядка с каноном 94,5
    rec_old_solo = sum(1 for k in mkeys if (md_old[k][0] and max(md_old[k][0]) >= 0.5)) / len(mkeys)
    sc2 = ok(f"MAX-old solo recall@0,5={100 * rec_old_solo:.1f} % (порядок канонов 94,5 ±3)",
             abs(100 * rec_old_solo - 94.5) <= 3.0)
    gates_clean = sc1 and sc2

    p("\nFP% / recall% по политик × τ (ансамбль AND двух рядов):")
    p(f"  {'policy':7} " + " ".join(f"τ={t:.2f}" for t in TAUS))
    for pol in policies:
        fp_s = " ".join(f"{r['fp_pct']:5.1f}" for r in rows if r["policy"] == pol)
        rc_s = " ".join(f"{r['recall_pct']:5.1f}" for r in rows if r["policy"] == pol)
        p(f"  {pol:7} FP {fp_s}")
        p(f"  {'':7} RC {rc_s}")

    # ---- гейт T1 ----
    p("\nГЕЙТ T1 (exploratory): ∃ не-MAX политика с FP(AND@0,40) ≤ "
      f"{FP_X1} % ∧ recall(AND@0,40) ≥ {RECALL_FLOOR} % ?")
    winners = []
    if gates_clean:
        for r in rows:
            if r["policy"] != "MAX" and r["tau"] == round(MAIN, 2) \
               and r["fp_pct"] <= FP_X1 and r["recall_pct"] >= RECALL_FLOOR:
                winners.append(r)
    for r in winners:
        p(f"  кандидат: {r['policy']}@{r['tau']:.2f}: FP={r['fp_pct']} %, recall={r['recall_pct']} %")
    if not gates_clean:
        p("  ВЕРДИКТ НЕ ВЫНОСИТСЯ: сами-проверки не пройдены → дамп дефектен, перемерить.")
    elif winners:
        p(f"  T1 ЗЕЛЁНЫЙ: {len(winners)} политик(а) достигают угла → КАНДИДАТ; подтверждение — "
          "только на замороженном holdout (it-87-класс). На dev-наборах зелёный переисключён "
          "(adaptive reuse). it-88 → не закрывается как null; кандидат передаётся в очередь holdout.")
    else:
        p("  T1 КРАСНЫЙ: ни одна tile-aware агрегация не собирает угол FP≤12,8 ∧ recall≥85 на "
          "HD/SAHI → exploratory-негатив; граница применимости ансамбля (SYNTHESIS п.9) "
          "НЕ снимается, перекалибровка τ запрещена стоп-критерием.")

    with open(OUT / "it88_tile_agg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "tau", "fp_pct", "recall_pct"])
        w.writeheader()
        w.writerows(rows)
    (OUT / "it88_tile_agg.txt").write_text(
        "it-88 tile-aware агрегация — срез stdout (протокол: iterations/it-88-tile-aware-sahi-PLANNED.md)\n\n"
        + "\n".join(LOG) + "\n", encoding="utf-8")
    p(f"\nCSV: {OUT / 'it88_tile_agg.csv'} ({len(rows)} строк)")


if __name__ == "__main__":
    main()
