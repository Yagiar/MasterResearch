#!/usr/bin/env python3
"""Детектор здоровья аудиоканала: скользящая доля drone-решений → авто-снижение w_a (it-14).

Профили деградации: healthy / deaf_first25 / deaf_last50 / intermittent / full_deaf.
Политики: static late 0.5, static w_v=0.9, late+median5, и те же + health-gate.
GT: airborne. Детерминировано (без сидов).
Запуск: research/.venv/bin/python research/channel_health_sim.py
"""
import csv

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(f"{ROOT}/research/yolo_sandbox_frames.csv"))
        if r["imgsz"] == "480"}

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    wins.append((float(yolo[str(sec)]["max_conf"]), float(ast[t0]["p_drone"]),
                 int(ast[t0]["airborne_gt"])))
N = len(wins)
W, FLOOR, HYST = 12, 0.15, 0.25  # параметры детектора здоровья (каузальное окно, порог отключения/включения)

# --- профили деградации: True = окно глухое (аудио-канал систематически врёт: p_a=0) ---
def profile(name):
    if name == "healthy":
        return [False] * N
    if name == "deaf_first25":
        return [i < N * 0.25 for i in range(N)]
    if name == "deaf_last50":
        return [i >= N * 0.5 for i in range(N)]
    if name == "intermittent":
        return [(i // 10) % 3 == 2 for i in range(N)]  # каждый 3-й блок из 10 окон глухой
    if name == "full_deaf":
        return [True] * N
    raise ValueError(name)

def causal_med(vals, k=5):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def simulate(profile_name, policy, W=W, floor=FLOOR, hyst=HYST, w_healthy=0.5):
    """Возвращает (P, R, F1). health-gate: w_a→0 при доле drone-решений аудио < floor."""
    deaf = profile(profile_name)
    hist_drone: list[int] = []   # последние W аудио-решений (1 = label drone)
    gate_open = True             # аудио допущено к слиянию
    preds = []
    for i, (pv, pa, gt) in enumerate(wins):
        pa_eff = 0.0 if deaf[i] else pa
        if policy == "video-only":
            preds.append((int(pv >= 0.5), gt))
            continue
        if policy == "late05":
            preds.append((int(0.5 * pv + 0.5 * pa_eff >= 0.5), gt))
            continue
        if policy == "late09":
            preds.append((int(0.9 * pv + 0.1 * pa_eff >= 0.5), gt))
            continue
        if policy == "late05_med":
            pa_all = [0.0 if deaf[j] else wins[j][1] for j in range(i + 1)]
            pa_eff = causal_med_local(pa_all)
            preds.append((int(0.5 * pv + 0.5 * pa_eff >= 0.5), gt))
            continue
        # --- политики с health-gate ---
        if policy in ("late05_health", "late05_med_health"):
            hist_drone.append(1 if (not deaf[i]) and pa >= 0.5 else 0)
            recent = hist_drone[-W:]
            health = sum(recent) / len(recent) if recent else 1.0
            if gate_open and health < floor and len(recent) >= W // 2:
                gate_open = False
            elif not gate_open and health > hyst:
                gate_open = True
            if policy == "late05_health":
                w_a = w_healthy if gate_open else 0.0
                preds.append((int((1 - w_a) * pv + w_a * pa_eff >= 0.5), gt))
                continue
            # late05_med_health: медиана + гейт (на глухом участке медиана по нулям ≈ 0 → гейт важен)
            pa_all = [0.0 if deaf[j] else wins[j][1] for j in range(i + 1)]
            pa_eff = causal_med_local(pa_all) if gate_open else 0.0
            w_a = w_healthy if gate_open else 0.0
            preds.append((int((1 - w_a) * pv + w_a * pa_eff >= 0.5), gt))
            continue
        raise ValueError(policy)
    return prf(preds)

def causal_med_local(vals, k=5):
    i = len(vals) - 1
    seg = sorted(vals[max(0, i - k + 1): i + 1])
    return seg[len(seg) // 2]

POLICIES = ["video-only", "late05", "late09", "late05_med", "late05_health", "late05_med_health"]
PROFILES = ["healthy", "deaf_first25", "deaf_last50", "intermittent", "full_deaf"]

print(f"окон: {N} | W={W}, floor={FLOOR}, hyst={HYST} (детектор здоровья)")
print(f"{'профиль':<14} | " + " | ".join(f"{p:<17}" for p in POLICIES))
csv_rows = []
best_health, best_static = [], []
for prof in PROFILES:
    cells = []
    for pol in POLICIES:
        P, R, F = simulate(prof, pol)
        cells.append(f"{F:.3f}")
        csv_rows.append(dict(profile=prof, policy=pol, P=round(P, 3), R=round(R, 3), F1=round(F, 3)))
    print(f"{prof:<14} | " + " | ".join(f"{c:<17}" for c in cells))

with open(f"{ROOT}/research/channel_health_sim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["profile", "policy", "P", "R", "F1"])
    w.writeheader()
    w.writerows(csv_rows)
print(f"\nCSV: research/channel_health_sim.csv ({len(csv_rows)} строк)")
print("\nОжидаемая связка для пайплайна: audio_temporal_k=5 (медиана) + health-gate (W=12, floor=0.15)")
