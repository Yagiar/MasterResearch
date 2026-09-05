#!/usr/bin/env python3
"""YOLO-инференс на 73 кадрах sandbox-клипа против GT (it-04).
Оценка «видео domain gap»: детектирует ли обученная YOLOv8n дрон в целевом домене.
Запуск: research/.venv/bin/python research/yolo_frames_eval.py
"""
import csv, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
FRAMES = f"{ROOT}/research/sandbox_frames/full"
WEIGHTS = f"{ROOT}/MasterDiploma/models/visual/yolov8s-uav.pt"
GT = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

from ultralytics import YOLO
model = YOLO(WEIGHTS)

rows = []
for thr_imgsz in (480, 640):
    for sec in sorted(GT):
        f = f"{FRAMES}/f_{sec+1:03d}.jpg"
        res = model.predict(f, device="cpu", imgsz=thr_imgsz, conf=0.05, verbose=False)[0]
        confs = [float(b.conf) for b in res.boxes] if res.boxes is not None else []
        rows.append(dict(imgsz=thr_imgsz, second=sec, visible=GT[sec][0], airborne=GT[sec][1],
                         n_det=len(confs), max_conf=max(confs) if confs else 0.0))

with open(f"{ROOT}/research/yolo_sandbox_frames.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def prf(rs, thr, gix):
    tp=fp=fn=0
    for r in rs:
        pred = int(r["max_conf"] >= thr)
        t = r["visible" if gix=="visible" else "airborne"]
        tp += pred and t; fp += pred and not t; fn += (not pred) and t
    P = tp/(tp+fp) if tp+fp else float("nan"); R = tp/(tp+fn) if tp+fn else float("nan")
    F = 2*P*R/(P+R) if P+R else float("nan")
    return P,R,F

for sz in (480, 640):
    rs = [r for r in rows if r["imgsz"]==sz]
    for thr in (0.25, 0.5):
        for gix in ("visible","airborne"):
            P,R,F = prf(rs, thr, gix)
            print(f"imgsz={sz} thr={thr}: {gix:<8} P={P:.3f} R={R:.3f} F1={F:.3f}")
    air = [r["max_conf"] for r in rs if r["airborne"]]
    gnd = [r["max_conf"] for r in rs if not r["airborne"]]
    import statistics as st
    print(f"imgsz={sz}: median conf airborne={st.median(air):.3f}  ground={st.median(gnd):.3f}")
