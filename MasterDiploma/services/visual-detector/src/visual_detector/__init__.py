"""visual-detector — визуальный детектор БПЛА.

Consumer-сервис: топик `video.raw` -> декод JPEG -> предобработка -> YOLOv8
(Ultralytics) -> (опц.) ByteTrack -> публикация `InferenceMsg` (modality=video)
в топик `inference`. На MVP может работать на предобученных весах (COCO) как
заглушка; целевые веса (`yolov8s-uav.pt`) кладёт обучающий пайплайн в `models/visual/`.
"""
