# train/ — обучающий пайплайн (`uavtrain`)

Отдельное приложение монорепо (не сервис): **скачка датасетов → подготовка → обучение
(YOLOv8 для видео, lightweight CNN для аудио) → eval → export лучших весов в `../models/`**.
Тяжёлые шаги (обучение) рассчитаны на сервер с GPU (RTX 3090); подготовка/eval — везде.

> Реализованы: подготовка визуальных данных (парсеры `hf-drone-detection` [YOLO-формат со сплитами],
> DUT Anti-UAV / Drone-vs-Bird — терпимы к разным раскладкам архивов: YOLO-txt / VOC XML / flat-txt;
> видео + per-frame аннотации), подготовка акустических данных (нарезка окон → MFCC/мел-спектрограммы),
> обучение и оценка YOLOv8 (Ultralytics) и lightweight CNN (PyTorch), export весов.
> Авто-загрузка: DUT Anti-UAV (Google Drive, `gdown`), DroneDetectionDataset / `hf-drone-detection`
> (Hugging Face Hub, `datasets` — материализуется в YOLO-формат), DroneAudioDataset и ESC-50
> (http-архив репозитория). Drone-vs-Bird / Multiclass Acoustic / MMAUD / AudioSet — ручная
> загрузка: `download()` печатает инструкцию (см. `list-datasets`). Всё авто-перечисленное и `gdown`,
> `datasets`, `pillow` входят в зависимости пакета.

## Установка
```bash
cd train
pip install -e .              # пакет uavtrain + зависимости (ultralytics, torch, librosa, ...)
pip install -e '.[notebook]'  # + jupyter, ipykernel
```
Зависимости закрепляют `numpy<2` (OpenCV/torch на многих образах ещё не готовы к NumPy 2.x —
иначе при `import cv2`/`import ultralytics` будет `numpy.core.multiarray failed to import`).
Если в окружении уже стоял numpy 2.x — `pip install "numpy<2"`.

Или Docker (GPU): `docker build -t uavtrain ./train && docker run --gpus all -p 8888:8888 -v $PWD/train/data:/app/data -v $PWD/train/runs:/app/runs -v $PWD/models:/app/../models uavtrain`.

## Два способа запуска
1. **Ноутбук** — `train/notebooks/pipeline.ipynb`: весь путь по секциям (0–6), вызывает функции `uavtrain.*`.
2. **CLI** — `python -m uavtrain.cli <команда>` (см. ниже).

## Этапы (runbook обучения)

### 1. Скачать датасеты
```bash
python -m uavtrain.cli list-datasets                 # реестр: [hf|gdrive|url|manual] по каждому
python -m uavtrain.cli download hf-drone-detection   # Hugging Face Hub (~54k кадров) — самый надёжный авто-источник
python -m uavtrain.cli download dut-anti-uav         # Google Drive (train/val/test) — нужен gdown (ставится с пакетом)
python -m uavtrain.cli download drone-audio-dataset  # http-архив репозитория
python -m uavtrain.cli download esc-50               # http-архив (негативы для акустики)
python -m uavtrain.cli download drone-vs-bird        # выведет инструкцию ручной загрузки
```
Датасеты кладутся в `train/data/<name>/` (в git не коммитятся). `hf-drone-detection` сразу
материализуется в YOLO-формат (`images/{train,test}/` + `labels/{train,test}/`); собственный train/test
этого датасета сохраняется как есть (`prepare-visual` его не пере-делит).

| Датасет | Модальность | Роль | Лицензия | Загрузка |
|---|---|---|---|---|
| DroneDetectionDataset (`hf-drone-detection`) — Pawełczyk & Wojtyra, IEEE Access 2020; HF-зеркало `pathikg/drone-detection-dataset` | видео (кадры) | позитивы | MIT | **авто** (HF Hub, `datasets`): `download hf-drone-detection` |
| DUT Anti-UAV Detection (IEEE-TITS) — `wangdongdut/DUT-Anti-UAV` | видео | позитивы | (см. репо) | **авто** (Google Drive, `gdown`): `download dut-anti-uav` |
| Drone-vs-Bird Detection Challenge | видео | позитивы | (по запросу) | вручную (запрос доступа на сайте challenge) → `train/data/drone-vs-bird/` |
| **DADS — Drone Audio Detection Samples** (`dads-audio`) — агрегат 6 источников звука дронов (DroneAudioDataset, DREGON, SPCup19, DroneNoise, AUDROK, fault-classification) + 4 не-дрон (UrbanSound8K, TUT Scenes, ESC-50, DNC); HF `geronimobasso/drone-audio-detection-samples` | аудио | дрон + не-дрон | MIT | **авто** (HF Hub, ~6.8 ГБ): `download dads-audio` → `train/data/dads-audio/{drone,non-drone}/*.wav`. **Рекомендованный** обучающий датасет акустики — много разных дронов и условий (вкл. уличные записи), решает domain shift |
| DroneAudioDataset (Al-Emadi) | аудио | позитивы | (см. репо) | **авто** (http-архив репо): `download drone-audio-dataset` (узкий — 2 модели в помещении; для разнообразия лучше `dads-audio`) |
| Multiclass Acoustic (Linn 2025, arXiv:2509.04715) | аудио | позитивы | (см. статью) | вручную (репозиторий авторов / Zenodo) — 32 модели дронов, для multiclass-постановки |
| ESC-50 | аудио | негативы | CC BY-NC | **авто** (http-архив): `download esc-50` |
| AudioSet (подвыборка) | аудио | негативы | CC BY 4.0 (метки) | вручную (yt-dlp по списку video_id) — «похожие» негативы (вертолёт/винт/самолёт) |
| MMAUD (NTU) | видео+аудио | fusion-пары | (форма доступа) | вручную (форма доступа) |

> Если `gdown` недоступен — DUT можно скачать руками со страницы https://github.com/wangdongdut/DUT-Anti-UAV
> (GoogleDrive или Baidu, архивы train/val/test) и распаковать **все** в `train/data/dut-anti-uav/`.
> Обучать можно на комбинации, напр. `prepare-visual --datasets hf-drone-detection,dut-anti-uav` — больше данных, выше mAP.

### 2. Подготовить данные
```bash
python -m uavtrain.cli prepare-visual --datasets hf-drone-detection            # самый надёжный; можно добавить ,dut-anti-uav
#   -> train/data/_prepared/visual/{images,labels}/{train,val,test} + data.yaml  (классы сведены к таксономии «дрон»)

# акустика — рекомендуется DADS (разнообразные дроны/условия) + лог-мел-спектрограмма:
python -m uavtrain.cli download dads-audio
python -m uavtrain.cli prepare-audio --positives dads-audio#drone --negatives dads-audio#non-drone \
    --feature melspec --n-mels 64 --win-ms 500 --hop-ms 250 --max-windows-per-file 10
#   -> train/data/_prepared/audio/{X.npy,y.npy,split.npy,index.csv,meta.json}  (X.npy — memmap, не грузится целиком в RAM)
#   ВАЖНО про DADS: дрон-клипы там по ~0.5с → нужен --win-ms 500 (при дефолтных 1000 они дадут 0 окон!);
#     не-дрон записи длинные (медиана ~4с, до 16с) → --max-windows-per-file 10 выравнивает классы
#     (иначе перекос ~1.5:1 в сторону не-дрон). --feature должен совпадать с acoustic_detector.feature в configs/pilot.yaml.
# (старый узкий вариант: --positives drone-audio-dataset#DroneAudioDataset-master/Binary_Drone_Audio/yes_drone,... --negatives esc-50 --feature mfcc)
```
Split — детерминированный (по видео/сцене для видео; по исходному wav для аудио), seed фиксирован.

### 3. Обучить
```bash
# на сервере с GPU:
python -m uavtrain.cli train-visual --data train/data/_prepared/visual/data.yaml --epochs 100 --batch 16 --device 0
#   -> train/runs/visual/uav-yolov8s/weights/best.pt

# акустика — две архитектуры (флаг --arch); по умолчанию имя прогона = uav-<arch>:
#  lwcnn    — компактная CNN ~24k параметров (edge), быстро:
python -m uavtrain.cli train-acoustic --features train/data/_prepared/audio/ --arch lwcnn --epochs 80 --device cuda
#   -> train/runs/acoustic/uav-lwcnn/lwcnn.pt
#  resnet18 — ResNet-18 ~11M параметров + АУГМЕНТАЦИИ под целевой домен (закрытие domain gap, как в MERIDIAN
#  arXiv:2506.11049): фон/gain/pitch/codec/SpecAugment; нужен пул фонового шума (ESC-50, не-дрон-записи):
python -m uavtrain.cli download esc-50
python -m uavtrain.cli train-acoustic --features train/data/_prepared/audio/ --arch resnet18 --augment \
    --bg-noise esc-50,dads-audio#non-drone --epochs 60 --batch-size 64 --device cuda --workers 8
#   -> train/runs/acoustic/uav-resnet18/lwcnn.pt  (имя файла историческое; внутри — state_dict ResNet-18)
#   ВАЖНО: --arch и --feature/n-mels/n-frames должны совпадать с acoustic_detector.{arch,feature,n_mels,n_frames} в configs/pilot.yaml.
#   --augment перечитывает сигнал из index.csv (поэтому train/data/dads-audio/ должен оставаться на месте).
```

### 4. Eval на test
```bash
python -m uavtrain.cli eval-visual --weights train/runs/visual/uav-yolov8s/weights/best.pt --data train/data/_prepared/visual/data.yaml --device 0
#   -> train/runs/eval/visual-uav-yolov8s/metrics.json (mAP@0.5, mAP@0.5:0.95, precision, recall)
python -m uavtrain.cli eval-acoustic --weights train/runs/acoustic/uav-resnet18/lwcnn.pt --features train/data/_prepared/audio/ --arch resnet18 --device cuda --name uav-resnet18
#   -> train/runs/eval/acoustic-uav-resnet18/metrics.json (accuracy, precision/recall/F1 macro) + confusion_matrix.png
#   (для lwcnn: --weights train/runs/acoustic/uav-lwcnn/lwcnn.pt --arch lwcnn --name uav-lwcnn)
```
> Eval на test даёт метрику на ДОМЕНЕ DADS. Обобщение на целевой звук (sandbox-аудио из видео) — отдельная
> проверка: `python scripts/probe_audio.py sandboxDataForSimulator/sandbox-audio-for-simulator.wav --weights models/acoustic/lwcnn.pt --arch <arch> --feature melspec --win-ms 500 --hop-ms 250 --n-mels 64`
> (печатает распределение p(drone) по окнам). Если p(drone) не поднимается — нужны более агрессивные аугментации
> или доменная адаптация (добавить кусок целевого аудио в позитивы).

### 5. Export весов
```bash
python -m uavtrain.cli export-visual   --weights train/runs/visual/uav-yolov8s/weights/best.pt --metrics train/runs/eval/visual-uav-yolov8s/metrics.json
#   -> models/visual/yolov8s-uav.pt  + строка в models/README.md
python -m uavtrain.cli export-acoustic --weights train/runs/acoustic/uav-resnet18/lwcnn.pt --metrics train/runs/eval/acoustic-uav-resnet18/metrics.json
#   -> models/acoustic/lwcnn.pt      + строка в models/README.md
#   ВАЖНО: arch в весах НЕ записан — после экспорта проверьте, что acoustic_detector.arch в configs/pilot.yaml
#   соответствует тому, чем обучали (lwcnn / resnet18), иначе load_state_dict в сервисе упадёт.
```
После этого детектор-сервисы (`visual-detector`, `acoustic-detector`) подхватят веса из `models/` (монтируются как `/models` в контейнеры). Дальше — поднять инфру и запустить пайплайн: см. **корневой `README.md`**.

## Структура `src/uavtrain/`
| Модуль | Назначение |
|---|---|
| `config.py` | пути (`data/`, `runs/`, `../models/`), таксономия классов (`VISUAL_CLASSES`=`["drone"]`, `AUDIO_CLASSES`=`["non-drone","drone"]`), сплиты, seed |
| `datasets.py` | реестр датасетов (`REGISTRY`) + `download()` (sha256-проверка) / инструкции ручной загрузки |
| `prepare_visual.py` | сборка YOLO-датасета: парсеры `hf-drone-detection` (YOLO-формат со сплитами, материализован из HF), DUT Anti-UAV / Drone-vs-Bird (YOLO-txt / VOC XML / flat-txt; видео+per-frame) → единая таксономия → split по сцене (или сохранение исходного train/test) → `data.yaml` |
| `prepare_audio.py` | сборка акустического набора: wav → окна (`win_ms`/`hop_ms`) → MFCC/мел-спектрограммы (фикс. ширина, z-score) → split по файлу → `X.npy`(memmap)+`y.npy`+`split.npy`+`index.csv`+`meta.json` |
| `train_visual.py` | дообучение YOLOv8 (Ultralytics API) |
| `audio_models.py` | архитектуры акустического классификатора: `build_audio_model("lwcnn"\|"resnet18")` — синхронизированы с `acoustic-detector/cnn.py` |
| `audio_augment.py` | аугментации train-сплита аудио (закрытие domain gap): подмешивание фона (ESC-50/не-дрон), gain/pitch/codec-sim, SpecAugment; `collect_background_wavs` |
| `train_acoustic.py` | обучение акустического классификатора (PyTorch): `--arch {lwcnn,resnet18}`, `--augment` (перечитывает сигнал из `index.csv`, аугментирует on-the-fly, затем melspec+SpecAugment); class-weights, выбор по balanced_acc, батчевый predict |
| `evaluate.py` | eval на test (visual: mAP/precision/recall; acoustic: accuracy/precision/recall/F1 + confusion matrix) + `evaluate_fusion_jsonl` (офлайн-eval fusion по jsonl из sink — заготовка под этап 6) |
| `export.py` | копирование лучших весов в `../models/{visual,acoustic}/` + строка в `models/README.md` |
| `cli.py` | CLI (`python -m uavtrain.cli ...`) — list-datasets / download / prepare-visual / prepare-audio / train-visual / train-acoustic / eval-visual / eval-acoustic / export-visual / export-acoustic |
