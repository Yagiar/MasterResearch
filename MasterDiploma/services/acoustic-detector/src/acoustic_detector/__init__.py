"""acoustic-detector — акустический детектор БПЛА.

Consumer-сервис: топик `audio.raw` -> декод PCM -> извлечение признаков (MFCC / мел-спектрограмма)
-> lightweight CNN («дрон / не-дрон») -> публикация `InferenceMsg` (modality=audio, с подсказкой
качества `snr_db`) в топик `inference`.

Веса (`lwcnn.pt` — `state_dict` модели `LightweightAudioCNN`) кладёт обучающий пайплайн
(`train/`, `python -m uavtrain.cli train-acoustic` / `export-acoustic`) в `models/acoustic/`.
Если весов нет — детектор работает в режиме **энергетического порога** (заглушка: «дрон», если
энергия аудио-окна выше адаптивного порога) — чтобы сквозной мультимодальный путь работал без обучения.

ВАЖНО: архитектура `LightweightAudioCNN` (модуль `cnn`) должна совпадать с архитектурой
в `train/src/uavtrain/train_acoustic.py` (веса — `state_dict`).
"""
