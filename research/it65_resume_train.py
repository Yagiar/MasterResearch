#!/usr/bin/env python3
"""Продолжить оборванную перезагрузкой тренировку uav-yolov8s-bg (it-65).

Ultralytics resume=True восстанавливает аргументы, оптимайзер и эпоху из last.pt
(22/30 на момент kernel panic 2026-09-20 18:45 MSK). Запуск из MasterDiploma
(data-путь в args.yaml относительный):
  cd MasterDiploma && nohup ./venv/bin/python ../research/it65_resume_train.py \\
    > train/runs/visual/uav-yolov8s-bg/resume.log 2>&1 &
"""
from ultralytics import YOLO

model = YOLO("train/runs/visual/uav-yolov8s-bg/weights/last.pt")
model.train(resume=True)
