"""Реестр датасетов + загрузка с проверкой целостности.

Поддерживаемые способы загрузки (поля `DatasetSpec`):
  - `url`        — прямая ссылка на архив (zip/tar): потоковая загрузка + (опц.) sha256;
  - `gdrive`     — список Google Drive file-id (и/или {"folder": id}): загрузка через `gdown`
                   (нужен `pip install gdown`); все архивы распаковываются в один каталог датасета;
  - `hf_repo`    — id датасета на Hugging Face Hub: грузится через `datasets.load_dataset` и
                   материализуется в YOLO-формат прямо в каталог датасета (`download()`);
  - ничего из перечисленного — ручная загрузка: `download()` поднимает RuntimeError с `manual_instructions`.

Использование:
    from uavtrain.datasets import REGISTRY, download
    download("dut-anti-uav")           # скачать (gdown) + распаковать в train/data/dut-anti-uav/
    download("esc-50")                 # скачать (http) + проверить sha256 + распаковать
    REGISTRY["mmaud"].manual_instructions   # что сделать руками
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR


@dataclass(frozen=True)
class DatasetSpec:
    """Описание датасета."""

    name: str
    modality: str                       # "visual" | "audio"
    role: str                           # "positive" | "negative" | "fusion-pair"
    url: str | None = None              # прямая ссылка на архив (zip/tar) — если есть
    sha256: str | None = None           # ожидаемый sha256 архива (если известен; только для url)
    archive_kind: str | None = None     # "zip" | "tar" | None (если url указывает на не-архив)
    gdrive: list[str | dict] = field(default_factory=list)  # Google Drive: file-id или {"folder": folder-id}
    hf_repo: str | None = None          # id датасета на Hugging Face Hub (datasets.load_dataset)
    hf_kind: str = "visual_coco"        # как материализовать HF-датасет: "visual_coco" (bbox->YOLO) | "audio_binary" (wav по классам)
    hf_format: str = "coco"             # формат bbox в HF-датасете: "coco" ([x,y,w,h]) | "pascal_voc" ([x1,y1,x2,y2]) — только для visual
    license: str = ""                   # лицензия (для отчёта/цитирования)
    citation: str = ""                  # первоисточник (статья/репозиторий)
    description: str = ""
    manual_instructions: str = ""       # шаги ручной загрузки, если нет url/gdrive/hf_repo
    homepage: str = ""

    @property
    def has_auto_download(self) -> bool:
        return bool(self.url) or bool(self.gdrive) or bool(self.hf_repo)


# Реестр.
REGISTRY: dict[str, DatasetSpec] = {
    "dut-anti-uav": DatasetSpec(
        name="dut-anti-uav",
        modality="visual",
        role="positive",
        # DUT Anti-UAV Detection (IEEE-TITS): три архива train/val/test на Google Drive.
        # https://github.com/wangdongdut/DUT-Anti-UAV
        gdrive=[
            "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",  # train
            "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",  # val
            "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",  # test
        ],
        description="DUT Anti-UAV Detection: изображения с БПЛА + bbox-разметка (train/val/test).",
        homepage="https://github.com/wangdongdut/DUT-Anti-UAV",
        manual_instructions=(
            "Авто: pip install gdown && python -m uavtrain.cli download dut-anti-uav.\n"
            "Вручную (если gdown недоступен): скачать train/val/test архивы со страницы "
            "https://github.com/wangdongdut/DUT-Anti-UAV (GoogleDrive или Baidu) и распаковать "
            "ВСЕ в train/data/dut-anti-uav/, затем prepare_visual.convert(datasets=['dut-anti-uav'])."
        ),
    ),
    "hf-drone-detection": DatasetSpec(
        name="hf-drone-detection",
        modality="visual",
        role="positive",
        hf_repo="pathikg/drone-detection-dataset",
        hf_kind="visual_coco",
        hf_format="coco",  # objects.bbox = [x, y, w, h]
        license="MIT",
        citation=(
            "M. Pawełczyk, M. Wojtyra, «Real World Object Detection Dataset for Quadcopter Unmanned "
            "Aerial Vehicle Detection», IEEE Access, 2020. Репозиторий: https://github.com/Maciullo/DroneDetectionDataset. "
            "HF-зеркало (parquet): https://huggingface.co/datasets/pathikg/drone-detection-dataset"
        ),
        description="DroneDetectionDataset (Pawełczyk & Wojtyra, IEEE Access 2020) — ~54k кадров (видео YouTube) с БПЛА, "
        "bbox-разметка (COCO, 1 класс drone), 640×480; HF-зеркало pathikg/drone-detection-dataset, MIT.",
        homepage="https://huggingface.co/datasets/pathikg/drone-detection-dataset",
        manual_instructions=(
            "Авто: python -m uavtrain.cli download hf-drone-detection (нужен пакет `datasets`; "
            "грузится с Hugging Face Hub и материализуется в YOLO-формат в train/data/hf-drone-detection/)."
        ),
    ),
    "dads-audio": DatasetSpec(
        name="dads-audio",
        modality="audio",
        role="positive",
        hf_repo="geronimobasso/drone-audio-detection-samples",
        hf_kind="audio_binary",  # столбцы: audio, label (0=non-drone, 1=drone)
        license="MIT",
        citation=(
            "DADS — Drone Audio Detection Samples (G. Basso). Агрегирует: DroneAudioDataset (Al-Emadi 2019), "
            "DREGON / SPCup19 Egonoise (Inria 2019), DroneNoise (Ramos-Romero et al. 2024), AUDROK (2023), "
            "Sound-Based Drone Fault Classification (Yi et al. 2023) — дрон; UrbanSound8K, TUT Acoustic Scenes 2017, "
            "ESC-50, DNC — не-дрон. https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples"
        ),
        description="DADS — крупнейший публичный аудио-датасет детекции БПЛА: ~180k клипов (бинарно дрон/не-дрон; "
        "163.6k дрон + 16.7k не-дрон), 16 кГц моно WAV, агрегат из 6 источников звука дронов (вкл. уличные записи и "
        "разные модели) + 4 источников не-дрон. MIT.",
        homepage="https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples",
        manual_instructions=(
            "Авто: python -m uavtrain.cli download dads-audio (нужен пакет `datasets` + `soundfile`; грузится с HF Hub "
            "[~6.8 ГБ] и материализуется в train/data/dads-audio/{drone,non-drone}/*.wav). Использовать как positives И "
            "negatives: prepare-audio --positives dads-audio#drone --negatives dads-audio#non-drone"
        ),
    ),
    "drone-vs-bird": DatasetSpec(
        name="drone-vs-bird",
        modality="visual",
        role="positive",
        description="Drone-vs-Bird Detection Challenge: видео с дронами и птицами + per-video аннотации.",
        homepage="https://wosdetc2023.wordpress.com/drone-vs-bird-detection-challenge/",
        manual_instructions=(
            "1) Запросить доступ на сайте Drone-vs-Bird Challenge (организаторы присылают ссылку).\n"
            "2) Скачать видео + файлы аннотаций, распаковать в train/data/drone-vs-bird/ "
            "(видео + рядом .txt/.csv с разметкой по кадрам).\n"
            "3) prepare_visual.convert(datasets=['drone-vs-bird']) — извлечёт кадры и переведёт разметку в YOLO-формат."
        ),
    ),
    "multiclass-acoustic": DatasetSpec(
        name="multiclass-acoustic",
        modality="audio",
        role="positive",
        description="Multiclass Acoustic Dataset (Linn et al., 2025; arXiv:2509.04715) — классы дронов/фона.",
        homepage="https://arxiv.org/abs/2509.04715",
        manual_instructions=(
            "1) Получить датасет по ссылке из статьи arXiv:2509.04715 (репозиторий авторов / Zenodo).\n"
            "2) Распаковать в train/data/multiclass-acoustic/ (wav по классам).\n"
            "3) prepare_audio.build(positives=['multiclass-acoustic'], negatives=['esc-50'])."
        ),
    ),
    "drone-audio-dataset": DatasetSpec(
        name="drone-audio-dataset",
        modality="audio",
        role="positive",
        # архив репозитория с master-веткой (внутри — каталоги Binary_Drone_Audio/yes_drone, unknown и др.)
        url="https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip",
        archive_kind="zip",
        description="DroneAudioDataset (Al-Emadi): записи звука дронов (yes_drone) и фона (unknown).",
        homepage="https://github.com/saraalemadi/DroneAudioDataset",
        manual_instructions=(
            "Авто: python -m uavtrain.cli download drone-audio-dataset (качает архив репозитория).\n"
            "Или вручную: git clone https://github.com/saraalemadi/DroneAudioDataset в train/data/drone-audio-dataset/."
        ),
    ),
    "esc-50": DatasetSpec(
        name="esc-50",
        modality="audio",
        role="negative",
        url="https://github.com/karoldvl/ESC-50/archive/master.zip",
        archive_kind="zip",
        description="ESC-50: 2000 коротких записей окружающих звуков (50 классов) — источник негативов.",
        homepage="https://github.com/karolpiczak/ESC-50",
        manual_instructions=(
            "Авто: python -m uavtrain.cli download esc-50. "
            "Если зеркало недоступно — взять архив с GitHub-релизов ESC-50, распаковать в train/data/esc-50/."
        ),
    ),
    "audioset-neg": DatasetSpec(
        name="audioset-neg",
        modality="audio",
        role="negative",
        description="AudioSet (подвыборка фоновых классов) — дополнительные негативы для акустики.",
        homepage="https://research.google.com/audioset/",
        manual_instructions=(
            "1) Взять список video_id нужных классов из метаданных AudioSet.\n"
            "2) Выгрузить аудио-сегменты (yt-dlp + ffmpeg) в train/data/audioset-neg/.\n"
            "3) Использовать как negatives в prepare_audio.build()."
        ),
    ),
    "mmaud": DatasetSpec(
        name="mmaud",
        modality="visual",  # содержит и видео, и аудио — для fusion-пар
        role="fusion-pair",
        description="MMAUD (NTU): мультимодальный датасет анти-UAV (видео+аудио+др.) — для пилотных fusion-замеров.",
        homepage="https://github.com/ntu-aris/MMAUD",
        manual_instructions=(
            "1) Получить MMAUD по инструкции репозитория NTU-ARIS (форма доступа).\n"
            "2) Распаковать в train/data/mmaud/.\n"
            "3) Использовать как источник синхронных пар видео/аудио для оценки fusion (этап 6)."
        ),
    ),
}


# --- утилиты ---
def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_stream(url: str, dest: Path) -> None:
    import requests  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "uavtrain/0.1"}) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def _download_gdrive(items: list[str | dict], dest_dir: Path) -> list[Path]:
    """Скачать элементы Google Drive в dest_dir (нужен `gdown`). Вернуть пути к скачанному."""
    try:
        import gdown  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "для загрузки с Google Drive нужен пакет gdown: pip install gdown"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    before = {p.name for p in dest_dir.iterdir()}
    for i, item in enumerate(items):
        if isinstance(item, dict) and "folder" in item:
            gdown.download_folder(id=str(item["folder"]), output=str(dest_dir), quiet=False, use_cookies=False)
        else:
            file_id = str(item)
            # output=None -> gdown сам берёт имя файла из Drive
            path = gdown.download(id=file_id, output=str(dest_dir) + "/", quiet=False)
            if path:
                out.append(Path(path))
    # то, что появилось в каталоге после загрузки
    after = {p.name for p in dest_dir.iterdir()}
    for name in sorted(after - before):
        p = dest_dir / name
        if p.is_file() and p not in out:
            out.append(p)
    return out


def _hf_bbox_to_yolo(bbox: list[float], w: int, h: int, fmt: str) -> tuple[float, float, float, float]:
    """Конвертировать bbox HF-датасета в нормированный YOLO [xc, yc, w, h]."""
    if fmt == "pascal_voc":  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox[:4]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    else:  # "coco": [x, y, w, h]
        x1, y1, bw, bh = bbox[:4]
    return ((x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h)


_AUDIO_COL_CANDIDATES = ("audio", "file", "audio_file", "wav", "path", "filepath", "filename")
_LABEL_COL_CANDIDATES = ("label", "labels", "class", "class_id", "target", "is_drone", "drone")


def _is_drone_label(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    return str(v).strip().lower() in {"1", "drone", "yes", "yes_drone", "uav", "true", "positive"}


def _audio_value_bytes(a):
    """Достать сырые байты WAV из значения столбца (dict {'bytes'/'path'} / base64-строка / JSON-строка / путь)."""
    import base64  # noqa: PLC0415
    import json  # noqa: PLC0415

    if isinstance(a, dict):
        b = a.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
        if isinstance(b, str):
            try:
                return base64.b64decode(b)
            except Exception:  # noqa: BLE001
                return b.encode("latin-1", errors="ignore")
        if a.get("path") and Path(a["path"]).exists():
            return Path(a["path"]).read_bytes()
        return None
    if isinstance(a, (bytes, bytearray)):
        return bytes(a)
    if isinstance(a, str):
        if Path(a).exists():
            return Path(a).read_bytes()
        s = a.strip()
        if s.startswith("{"):
            try:
                return _audio_value_bytes(json.loads(s))
            except Exception:  # noqa: BLE001
                return None
    return None


def _write_wav_from_bytes(raw: bytes, dst: Path) -> bool:
    """Записать WAV из сырых байт в dst (через soundfile; крайний случай — байты как есть, это WAV)."""
    import io  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    try:
        data, sr = sf.read(io.BytesIO(raw))
        sf.write(dst, data, sr)
        return True
    except Exception:  # noqa: BLE001
        try:
            dst.write_bytes(raw)
            return True
        except Exception:  # noqa: BLE001
            return False


def _materialize_audio_binary_from_parquet(hf_repo: str, out_dir: Path) -> None:
    """Материализовать HF audio-датасет в out_dir/{drone,non-drone}/*.wav, читая parquet НАПРЯМУЮ (pyarrow).

    Минует `datasets.load_dataset` и его Audio-декодер (torchcodec/FFmpeg, которых может не быть на хосте):
    качаем parquet-файлы датасета через huggingface_hub и читаем pyarrow.parquet. Ожидаем столбцы
    ~`audio` (struct {bytes, path} либо JSON-строка с байтами WAV) и ~`label` (int/строка/bool; 1 = дрон).
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("нужны пакеты: pip install pyarrow huggingface_hub") from exc

    api = HfApi()
    files = [f for f in api.list_repo_files(hf_repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"[audio_binary] в датасете {hf_repo} не найдено .parquet файлов")
    print(f"[audio_binary] {hf_repo}: {len(files)} parquet-файлов")

    drone_dir, bg_dir = out_dir / "drone", out_dir / "non-drone"
    drone_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **kw: x  # noqa: E731

    n_d = n_b = n_skip = idx = 0
    audio_col = label_col = None
    for rel in tqdm(sorted(files), desc="DADS parquet→wav", unit="файл"):
        local = hf_hub_download(hf_repo, rel, repo_type="dataset")
        t = pq.read_table(local)
        cols = t.column_names
        if audio_col is None:
            audio_col = next((c for c in _AUDIO_COL_CANDIDATES if c in cols), None) or \
                        next((c for c in cols if "audio" in c.lower() or "wav" in c.lower()), None)
            label_col = next((c for c in _LABEL_COL_CANDIDATES if c in cols), None) or \
                        next((c for c in cols if "label" in c.lower() or "class" in c.lower()), None)
            if audio_col is None or label_col is None:
                raise RuntimeError(f"[audio_binary] не нашёл столбцы аудио/метки: cols={cols}")
            print(f"[audio_binary] audio_col={audio_col}, label_col={label_col}")
        a_arr = t.column(audio_col).to_pylist()
        l_arr = t.column(label_col).to_pylist()
        for a, lbl in zip(a_arr, l_arr):
            idx += 1
            dst = (drone_dir if _is_drone_label(lbl) else bg_dir) / f"{idx:08d}.wav"
            raw = _audio_value_bytes(a)
            if raw and _write_wav_from_bytes(raw, dst):
                n_d += int(dst.parent.name == "drone"); n_b += int(dst.parent.name == "non-drone")
            else:
                n_skip += 1
    print(f"[audio_binary] готово: drone={n_d}, non-drone={n_b}, пропущено={n_skip}")
    if n_d == 0 and n_b == 0:
        raise RuntimeError("[audio_binary] не записано ни одного файла — проверьте формат датасета (см. лог про столбцы)")


def _materialize_audio_binary(ds, out_dir: Path) -> None:
    """Материализовать уже загруженный `datasets`-объект в out_dir/{drone,non-drone}/*.wav (fallback-путь).

    Используется, если parquet-путь недоступен. Не кастует столбец в Audio() — но если фича уже Audio,
    итерация может упасть на torchcodec; в таком случае используйте parquet-путь (см. _download_hf).
    """
    import io  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    drone_dir, bg_dir = out_dir / "drone", out_dir / "non-drone"
    drone_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)
    n_d = n_b = n_skip = idx = 0
    for split_name in ds:
        split = ds[split_name]
        cols = list(split.column_names) if hasattr(split, "column_names") else list((getattr(split, "features", {}) or {}).keys())
        audio_col = next((c for c in _AUDIO_COL_CANDIDATES if c in cols), None) or next((c for c in cols if "audio" in c.lower() or "wav" in c.lower()), None)
        label_col = next((c for c in _LABEL_COL_CANDIDATES if c in cols), None) or next((c for c in cols if "label" in c.lower() or "class" in c.lower()), None)
        if audio_col is None or label_col is None:
            raise RuntimeError(f"[audio_binary] не нашёл столбцы аудио/метки в '{split_name}': cols={cols}")
        for ex in split:
            idx += 1
            a = ex[audio_col]
            dst = (drone_dir if _is_drone_label(ex[label_col]) else bg_dir) / f"{idx:08d}.wav"
            ok = False
            if isinstance(a, dict) and a.get("array") is not None:
                try:
                    sf.write(dst, np.asarray(a["array"]), int(a.get("sampling_rate") or 16000)); ok = True
                except Exception:  # noqa: BLE001
                    ok = False
            else:
                raw = _audio_value_bytes(a)
                ok = bool(raw) and _write_wav_from_bytes(raw, dst)
            if ok:
                n_d += int(dst.parent.name == "drone"); n_b += int(dst.parent.name == "non-drone")
            else:
                n_skip += 1
    print(f"[audio_binary] готово (fallback): drone={n_d}, non-drone={n_b}, пропущено={n_skip}")
    if n_d == 0 and n_b == 0:
        raise RuntimeError("[audio_binary] не записано ни одного файла")


def _download_hf(spec: DatasetSpec, out_dir: Path) -> None:
    """Скачать HF-датасет (datasets.load_dataset) и материализовать в локальную раскладку в out_dir.

    `hf_kind="visual_coco"` -> out_dir/images/<split>/*.jpg + out_dir/labels/<split>/*.txt (YOLO; класс 0=drone)
       — затем prepare_visual подхватывает как «картинки+аннотации рядом» (_parse_image_dataset).
    `hf_kind="audio_binary"` -> out_dir/{drone,non-drone}/*.wav (по столбцу label) — затем prepare_audio
       берёт как источники: positives='<name>#drone', negatives='<name>#non-drone'.
    """
    if spec.hf_kind == "audio_binary":
        # parquet НАПРЯМУЮ — минуя datasets.load_dataset и его Audio-декодер (torchcodec/FFmpeg).
        # Если parquet-путь почему-то не сработал — fallback на load_dataset (может упасть на torchcodec).
        try:
            _materialize_audio_binary_from_parquet(spec.hf_repo, out_dir)
            return
        except RuntimeError as exc:
            print(f"[audio_binary] parquet-путь не сработал ({exc}); fallback на datasets.load_dataset")
        from datasets import load_dataset  # noqa: PLC0415
        _materialize_audio_binary(load_dataset(spec.hf_repo), out_dir)
        return

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("для загрузки с Hugging Face нужен пакет datasets: pip install datasets") from exc

    ds = load_dataset(spec.hf_repo)

    # hf_kind == "visual_coco"
    for split_name in ds:
        sub = "test" if "test" in split_name.lower() else ("val" if "val" in split_name.lower() else "train")
        img_dir = out_dir / "images" / sub
        lbl_dir = out_dir / "labels" / sub
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        split = ds[split_name]
        for i, ex in enumerate(split):
            img = ex["image"]
            w = int(ex.get("width") or img.width)
            h = int(ex.get("height") or img.height)
            stem = f"{split_name}_{ex.get('image_id', i):08d}"
            img.convert("RGB").save(img_dir / f"{stem}.jpg", quality=90)
            objs = ex.get("objects") or {}
            bboxes = objs.get("bbox") or []
            with (lbl_dir / f"{stem}.txt").open("w", encoding="utf-8") as fh:
                for bb in bboxes:
                    xc, yc, bw, bh = _hf_bbox_to_yolo(list(bb), w, h, spec.hf_format)
                    fh.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def _extract_into(archive: Path, out_dir: Path) -> bool:
    """Распаковать архив в out_dir; вернуть True, если это был архив (иначе оставить файл на месте)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
        return True
    return False


def dataset_dir(name: str) -> Path:
    """Каталог датасета внутри train/data/."""
    return DATA_DIR / name


def _suffix_for(spec: DatasetSpec) -> str:
    if spec.archive_kind == "zip":
        return ".zip"
    if spec.archive_kind == "tar":
        return ".tar.gz"
    if spec.url and spec.url.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
        return "." + spec.url.rsplit(".", 1)[-1]
    return ".bin"


def download(name: str, *, force: bool = False) -> Path:
    """Скачать датасет (http / Google Drive) и распаковать в train/data/<name>/.

    - `url`: потоковая загрузка в train/data/_archives/<name>.* + (опц.) проверка sha256 + распаковка;
    - `gdrive`: загрузка через gdown прямо в каталог датасета; скачанные архивы распаковываются на месте,
      сами архивы удаляются (остаётся распакованное содержимое).
    Для датасетов без авто-загрузки — RuntimeError с инструкцией. Возвращает путь к каталогу датасета.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"датасет не зарегистрирован: {name!r}; доступны: {sorted(REGISTRY)}")

    out_dir = dataset_dir(name)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return out_dir

    if not spec.has_auto_download:
        raise RuntimeError(f"датасет {name!r} требует ручной загрузки.\n{spec.manual_instructions}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.hf_repo:
        _download_hf(spec, out_dir)
        return out_dir

    if spec.gdrive:
        downloaded = _download_gdrive(spec.gdrive, out_dir)
        for p in downloaded:
            if _extract_into(p, out_dir):
                p.unlink(missing_ok=True)  # архив больше не нужен — содержимое распаковано
        return out_dir

    # spec.url
    archives = DATA_DIR / "_archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive_path = archives / f"{name}{_suffix_for(spec)}"
    if not archive_path.exists() or force:
        _download_stream(spec.url, archive_path)
    if spec.sha256:
        actual = _sha256_of(archive_path)
        if actual.lower() != spec.sha256.lower():
            raise RuntimeError(
                f"sha256 архива {name} не совпал: ожидалось {spec.sha256}, получено {actual}. Проверьте источник."
            )
    if not _extract_into(archive_path, out_dir):
        shutil.copy2(archive_path, out_dir / archive_path.name)
    return out_dir


def list_datasets(modality: str | None = None) -> list[DatasetSpec]:
    """Список зарегистрированных датасетов (опц. фильтр по модальности)."""
    specs = list(REGISTRY.values())
    if modality:
        specs = [s for s in specs if s.modality == modality]
    return specs
