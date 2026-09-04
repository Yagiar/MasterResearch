#!/usr/bin/env python3
"""Health-признак v2 (it-16): «звук есть (энергия), а p_a всегда 0 → канал глухой».

Отличает тишину (энергия низкая — каналу нечего ловить, это норма) от глухоты
(энергия высокая, канал детерминированно молчит). Профили деградации как в it-14.
Запуск: research/.venv/bin/python research/channel_health_v2.py
"""
import csv
import math

import numpy as np
import soundfile as sf
import librosa

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
WAV = f"{ROOT}/MasterDiploma/sandboxDataForSimulator/sandbox-audio-for-simulator.wav"
W, FLOOR_STD, HYST = 12, 0.05, 0.15  # скользящее окно, порог std_pa, гистерезис

ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(f"{ROOT}/research/yolo_sandbox_frames.csv"))
        if r["imgsz"] == "480"}
gt = {int(r["second"]): int(r["airborne"])
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

# --- энергия окон (RMS), из реального wav ---
pcm, sr = sf.read(WAV, dtype="int16")
if pcm.ndim > 1:
    pcm = pcm[:, 0]
x = librosa.resample(pcm.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000)

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    chunk = x[int(float(t0) * 16000): int(float(t0) * 16000) + 16000]
    rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
    wins.append(dict(t0=float(t0), pv=float(yolo[str(sec)]["max_conf"]),
                     pa=float(ast[t0]["p_drone"]),
                     airborne=int(ast[t0]["airborne_gt"]), rms=rms))
N = len(wins)
rms_med = float(np.median([w["rms"] for w in wins]))
print(f"окон: {N}; RMS медиана клипа = {rms_med:.4f}")
r_stand = [w["rms"] for w in wins if not w["airborne"]]
r_fly = [w["rms"] for w in wins if w["airborne"]]
print(f"RMS стоянка: медиана {np.median(r_stand):.4f} (min {min(r_stand):.4f}) | "
      f"полёт: медиана {np.median(r_fly):.4f} (min {min(r_fly):.4f})")

with open(f"{ROOT}/research/window_energy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(wins[0]))
    w.writeheader()
    w.writerows(wins)

# --- профили деградации (True = окно глухое: канал систематически молчит при живом звуке) ---
def profile(name):
    return {"healthy": [False] * N,
            "deaf_first25": [i < N * 0.25 for i in range(N)],
            "deaf_last50": [i >= N * 0.5 for i in range(N)],
            "intermittent": [(i // 10) % 3 == 2 for i in range(N)],
            "full_deaf": [True] * N}[name]

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

def simulate(profile_name, policy):
    """Политики: video-only, late05, late05_med, late05_med_healthV2.

    healthV2: подозрение на глухоту = (скользящая RMS > 0.8 медианы клипа) и
    (скользящий std p_a < FLOOR_STD) на протяжении >= 6 окон подряд → w_a=0;
    возврат при (std_pa >= HYST) или (RMS < 0.5 медианы).
    """
    deaf = profile(profile_name)
    gate_open = True
    suspect_run = 0
    preds = []
    for i, w in enumerate(wins):
        pa_eff_raw = 0.0 if deaf[i] else w["pa"]
        if policy == "video-only":
            preds.append((int(w["pv"] >= 0.5), w["airborne"]))
            continue
        if policy == "late05":
            preds.append((int(0.5 * w["pv"] + 0.5 * pa_eff_raw >= 0.5), w["airborne"]))
            continue
        if policy == "late05_med":
            pa_m = causal_med([0.0 if deaf[j] else wins[j]["pa"] for j in range(i + 1)])[-1]
            preds.append((int(0.5 * w["pv"] + 0.5 * pa_m >= 0.5), w["airborne"]))
            continue
        if policy == "late05_med_healthV2":
            pa_hist = [0.0 if deaf[j] else wins[j]["pa"] for j in range(i + 1)]
            last = pa_hist[-W:]
            std_pa = float(np.std(last)) if len(last) > 1 else 0.0
            rms_rel = float(np.mean([wins[j]["rms"] for j in range(max(0, i - W + 1), i + 1)])) / rms_med
            if gate_open:
                if rms_rel > 0.8 and std_pa < FLOOR_STD:
                    suspect_run += 1
                else:
                    suspect_run = 0
                if suspect_run >= 6:
                    gate_open = False
            else:
                if std_pa >= HYST or rms_rel < 0.5:
                    gate_open = True
                    suspect_run = 0
            w_a = 0.5 if gate_open else 0.0
            pa_use = causal_med(pa_hist)[-1] if gate_open else 0.0
            preds.append((int((1 - w_a) * w["pv"] + w_a * pa_use >= 0.5), w["airborne"]))
            continue
        raise ValueError(policy)
    return prf(preds)

POLICIES = ["video-only", "late05", "late05_med", "late05_med_healthV2"]
PROFILES = ["healthy", "deaf_first25", "deaf_last50", "intermittent", "full_deaf"]
print(f"\nhealthV2: W={W}, std_pa<{FLOOR_STD}, RMS>0.8·медианы, подозрений>=6 подряд\n")
print(f"{'профиль':<14} | " + " | ".join(f"{p:<20}" for p in POLICIES))
csv_rows = []
for prof in PROFILES:
    cells = []
    for pol in POLICIES:
        P, R, F = simulate(prof, pol)
        Fd = F if F == F else 0.0
        cells.append(f"{F:.3f}")
        csv_rows.append(dict(profile=prof, policy=pol, P=round(P, 3) if P == P else None,
                             R=round(R, 3), F1=round(Fd, 3)))
    print(f"{prof:<14} | " + " | ".join(f"{c:<20}" for c in cells))

with open(f"{ROOT}/research/channel_health_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["profile", "policy", "P", "R", "F1"])
    w.writeheader()
    w.writerows(csv_rows)
print(f"\nCSV: research/channel_health_v2.csv ({len(csv_rows)} строк)")
