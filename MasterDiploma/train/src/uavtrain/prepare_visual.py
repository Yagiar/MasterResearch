"""Подготовка визуальных данных к обучению YOLOv8.

Из скачанных датасетов (DUT Anti-UAV, Drone-vs-Bird, опц. собственный датасет ВКР в YOLO-формате)
формирует единый набор:
    <PREPARED_DIR>/visual/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt        # YOLO: "<cls> <xc> <yc> <w> <h>" (нормированные)
        data.yaml                            # path/train/val/test/names

Шаги: парсинг исходной разметки каждого датасета -> приведение классов к единой таксономии
(`VISUAL_CLASSES` — на пилоте бинарно «дрон»; кадры без дрон-боксов = негативы). it-37 (ревью §8):
боксы НЕ-дроновых классов (птицы, самолёты, ...) ЯВНО отбрасываются по id (YOLO-txt) или имени
(VOC) — старое поведение «всё → дрон» подменяло классы; сводка отброшенного печатается в конце.
Дальше — детерминированный split по сцене/видео (кадры одной сцены не утекают между сплитами,
включая выделение val из train) -> запись data.yaml.

Парсеры терпимы к разным раскладкам архивов: ищут изображения + аннотации по распространённым
схемам; если структура не распознана — поднимают понятную ошибку с описанием того, что ожидалось.
Аннотации поддерживаются в виде: YOLO-txt (`cls xc yc w h`), Pascal VOC XML (`<bndbox>`),
«flat»-txt (`x1 y1 x2 y2` или `x y w h` на строку, абсолютные пиксели).
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from .config import DEFAULT_SPLITS, PREPARED_DIR, RANDOM_SEED, VISUAL_CLASSES
from .datasets import dataset_dir

VISUAL_PREPARED = PREPARED_DIR / "visual"
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_DRONE_CLS = 0  # единственный класс на пилоте


@dataclass(frozen=True)
class YoloSample:
    """Один кадр + его боксы (нормированные YOLO-координаты)."""

    image_path: Path
    boxes: list[tuple[int, float, float, float, float]]  # (cls, xc, yc, w, h)
    group: str                                            # ключ группировки для split (видео/сцена)
    split: str | None = None                              # если задан ("train"/"val"/"test") — используется как есть


# --- split / запись ---
def _split_groups(groups: list[str], splits: dict[str, float], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train, n_val = int(n * splits["train"]), int(n * splits["val"])
    return {"train": set(uniq[:n_train]), "val": set(uniq[n_train : n_train + n_val]), "test": set(uniq[n_train + n_val :])}


def _link_or_copy(src: Path, dst: Path) -> None:
    """Положить картинку в dst: жёсткой ссылкой (быстро, без расхода диска), иначе copy2."""
    if dst.exists():
        return
    try:
        import os  # noqa: PLC0415

        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yolo(samples: list[YoloSample], split_map: dict[str, set[str]]) -> None:
    for sub in ("train", "val", "test"):
        (VISUAL_PREPARED / "images" / sub).mkdir(parents=True, exist_ok=True)
        (VISUAL_PREPARED / "labels" / sub).mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(tqdm(samples, desc="prepare-visual: запись кадров", unit="кадр")):
        # сэмпл с явным split используется как есть; иначе — по результату split-по-группам
        sub = s.split if s.split in ("train", "val", "test") else next((k for k, g in split_map.items() if s.group in g), "train")
        # уникальное имя (датасеты могут иметь одинаковые имена кадров)
        stem = f"{s.group}_{i:07d}_{s.image_path.stem}".replace("/", "_")
        dst_img = VISUAL_PREPARED / "images" / sub / (stem + s.image_path.suffix.lower())
        _link_or_copy(s.image_path, dst_img)
        with (VISUAL_PREPARED / "labels" / sub / (stem + ".txt")).open("w", encoding="utf-8") as fh:
            for cls, xc, yc, w, h in s.boxes:
                fh.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _write_data_yaml() -> Path:
    VISUAL_PREPARED.mkdir(parents=True, exist_ok=True)
    path = VISUAL_PREPARED / "data.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"path": str(VISUAL_PREPARED), "train": "images/train", "val": "images/val", "test": "images/test",
             "names": dict(enumerate(VISUAL_CLASSES))},
            fh, allow_unicode=True, sort_keys=False,
        )
    return path


# --- утилиты парсинга аннотаций ---
def _image_size(path: Path) -> tuple[int, int]:
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"не удалось прочитать изображение: {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return ((x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h)


# it-37 (ревью §8): фильтрация классов ЯВНО, а не «всё → дрон».
# Мультиклассовый датасет, прошедший через старый парсер, превращал птиц/самолёты
# в положительные примеры дрона (тихая подмена классов).
DEFAULT_DRONE_CLASS_IDS = frozenset({0})          # id класса «дрон» в YOLO-разметке источника
DEFAULT_DRONE_VOC_NAMES = frozenset({"drone", "uav", "uas", "quadcopter", "quadcopter", "uav-drone"})
_dropped_boxes: dict[str, int] = {}               # имя класса/id → сколько боксов отброшено


def _note_dropped(key: str, n: int = 1) -> None:
    _dropped_boxes[key] = _dropped_boxes.get(key, 0) + n


def dropped_boxes_report() -> str:
    """Сводка отброшенных боксов (печатается по завершении парсинга)."""
    if not _dropped_boxes:
        return "отброшенных боксов нет"
    return "; ".join(f"{k}: {v}" for k, v in sorted(_dropped_boxes.items()))


def _parse_yolo_txt(txt: Path, drone_cls_ids: frozenset[int] = DEFAULT_DRONE_CLASS_IDS) -> list[tuple[int, float, float, float, float]]:
    """YOLO-txt -> боксы ТОЛЬКО классов-дронов (id из `drone_cls_ids`), нормированные (xc, yc, w, h).

    Боксы других классов ОТБРАСЫВАЮТСЯ (и учитываются в сводке) — кадр с птицей/самолётом
    остаётся негативом, а не превращается в позитив дрона.
    """
    boxes = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls_raw, xc, yc, w, h = parts[:5]
            cls = int(cls_raw)
            if cls not in drone_cls_ids:
                _note_dropped(f"yolo-cls-{cls}")
                continue
            boxes.append((_DRONE_CLS, float(xc), float(yc), float(w), float(h)))
    return boxes


def _parse_voc_xml(
    xml_path: Path,
    drone_names: frozenset[str] = DEFAULT_DRONE_VOC_NAMES,
) -> tuple[Path | None, list[tuple[float, float, float, float]], tuple[int, int] | None]:
    """VOC XML -> (путь к картинке если указан, список xyxy-боксов ТОЛЬКО классов-дронов, (w,h))."""
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename")
    size_el = root.find("size")
    size = None
    if size_el is not None:
        try:
            size = (int(float(size_el.findtext("width"))), int(float(size_el.findtext("height"))))
        except (TypeError, ValueError):
            size = None
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in drone_names:
            _note_dropped(f"voc-{name or 'unnamed'}")
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        boxes.append((float(bb.findtext("xmin")), float(bb.findtext("ymin")), float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    img_path = (xml_path.parent / fname) if fname else None
    return img_path, boxes, size


def _parse_flat_txt(txt: Path, box_format: str = "auto") -> list[tuple[float, float, float, float]]:
    """Текст с боксами по строкам. Формат: `auto` (эвристика), либо явно `xyxy` / `xywh`.

    Эвристика auto: 4 числа = либо x1 y1 x2 y2, либо x y w h — различаем по тому,
    монотонны ли первые две и вторые две координаты (ревью §8: неоднозначность должна
    разрешаться схемой датасета — передавайте box_format явно для известного набора).
    Возвращаем в формате xyxy (для xywh: x2=x+w, y2=y+h).
    """
    out = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        nums = [float(t) for t in line.replace(",", " ").split() if _is_float(t)]
        if len(nums) < 4:
            continue
        a, b, c, d = nums[-4:]  # последние 4 числа строки (могут быть frame_id, conf и т.п. впереди)
        if box_format == "xyxy" or (box_format == "auto" and c > a and d > b and (c - a) < 1e4):
            out.append((a, b, c, d))
        else:  # "auto" (не похоже на xyxy) или явный "xywh"
            out.append((a, b, a + c, b + d))
    return out


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _all_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT)


def _group_of(root: Path, img: Path) -> str:
    """Ключ группировки = относительный путь к родительской папке кадра (обычно — сцена/видео)."""
    rel = img.parent.relative_to(root)
    return str(rel) if str(rel) != "." else root.name


# --- парсеры датасетов ---
def _parse_image_dataset(root: Path) -> list[YoloSample]:
    """Универсальный парсер «картинки + аннотации рядом».

    Для каждого изображения ищет аннотацию: <name>.txt (YOLO или flat) или <name>.xml (VOC)
    в той же папке либо в зеркальной `labels/`/`annotations/`. Изображения без аннотации
    считаются негативами (пустой список боксов) — это валидно для YOLO (фоновые кадры).
    """
    images = _all_images(root)
    if not images:
        raise RuntimeError(f"в {root} не найдено изображений ({sorted(_IMG_EXT)})")
    samples: list[YoloSample] = []
    for img in tqdm(images, desc=f"parse {root.name}", unit="img"):
        boxes_xyxy: list[tuple[float, float, float, float]] = []
        yolo_boxes: list[tuple[int, float, float, float, float]] = []
        ann_candidates = [
            img.with_suffix(".txt"),
            img.with_suffix(".xml"),
            img.parent.parent / "labels" / (img.stem + ".txt"),                  # images/X.jpg -> labels/X.txt
            img.parent.parent / "annotations" / (img.stem + ".xml"),
            # YOLO-раскладка со сплитами: images/<split>/X.jpg -> labels/<split>/X.txt
            img.parents[2] / "labels" / img.parent.name / (img.stem + ".txt") if len(img.parents) >= 3 else img,
        ]
        for ann in ann_candidates:
            if not ann.exists():
                continue
            if ann.suffix == ".txt":
                # сначала пробуем как YOLO (нормированные < 1.0)
                yb = _parse_yolo_txt(ann)
                if yb and all(0.0 <= v <= 1.0 for _, *coords in yb for v in coords):
                    yolo_boxes = yb
                else:
                    boxes_xyxy = _parse_flat_txt(ann)
            else:  # .xml
                _, bxs, _ = _parse_voc_xml(ann)
                boxes_xyxy = bxs
            break
        if boxes_xyxy:
            w, h = _image_size(img)
            yolo_boxes = [(_DRONE_CLS, *_xyxy_to_yolo(*bb, w, h)) for bb in boxes_xyxy]
        samples.append(YoloSample(image_path=img, boxes=yolo_boxes, group=_group_of(root, img)))
    return samples


def _parse_video_dataset(root: Path, *, frame_stride: int = 5) -> list[YoloSample]:
    """Парсер «видео + аннотации по кадрам» (как Drone-vs-Bird).

    Для каждого .mp4/.avi ищет файл аннотаций с тем же стемом (.txt) формата
    `<frame_idx> <x> <y> <w> <h>` (по строке на объект; абсолютные пиксели). Извлекает
    кадры с шагом `frame_stride` в `<PREPARED_DIR>/visual/_frames/<video>/`, нормирует боксы.
    Если файлов аннотаций нет — поднимает ошибку (видео без разметки не пригодно для обучения).
    """
    import cv2  # noqa: PLC0415

    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    if not videos:
        raise RuntimeError(f"в {root} не найдено видеофайлов")
    frames_dir = VISUAL_PREPARED / "_frames"
    samples: list[YoloSample] = []
    found_any_ann = False
    for vid in tqdm(videos, desc=f"кадры из видео ({root.name})", unit="видео"):
        ann = next((vid.with_suffix(ext) for ext in (".txt", ".csv") if vid.with_suffix(ext).exists()), None)
        per_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        if ann is not None:
            found_any_ann = True
            for line in ann.read_text(encoding="utf-8", errors="ignore").splitlines():
                nums = [float(t) for t in line.replace(",", " ").split() if _is_float(t)]
                if len(nums) >= 5:
                    fi, x, y, w_, h_ = int(nums[0]), nums[1], nums[2], nums[3], nums[4]
                    per_frame.setdefault(fi, []).append((x, y, x + w_, y + h_))
        out_dir = frames_dir / vid.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vid))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0 and (idx in per_frame or not per_frame):
                h, w = frame.shape[:2]
                img_path = out_dir / f"{vid.stem}_{idx:07d}.jpg"
                cv2.imwrite(str(img_path), frame)
                yolo_boxes = [(_DRONE_CLS, *_xyxy_to_yolo(*bb, w, h)) for bb in per_frame.get(idx, [])]
                samples.append(YoloSample(image_path=img_path, boxes=yolo_boxes, group=vid.stem))
            idx += 1
        cap.release()
    if not found_any_ann:
        raise RuntimeError(
            f"в {root} есть видео, но не найдено файлов аннотаций (.txt/.csv с `frame x y w h`) — "
            "проверьте раскладку датасета Drone-vs-Bird (видео + per-video аннотации)."
        )
    return samples


def _parse_dut(root: Path) -> list[YoloSample]:
    """DUT Anti-UAV: обычно — папки изображений + аннотации (VOC XML или txt). Универсальный парсер картинок."""
    if _all_images(root):
        return _parse_image_dataset(root)
    # некоторые релизы DUT содержат видео
    return _parse_video_dataset(root)


def _parse_dvb(root: Path) -> list[YoloSample]:
    """Drone-vs-Bird: видео + per-video аннотации по кадрам."""
    return _parse_video_dataset(root)


def _parse_yolo_split_dir(root: Path) -> list[YoloSample]:
    """Готовый YOLO-датасет со сплитами: images/{train,val,test}/*.jpg + labels/{train,val,test}/*.txt.

    Используется, в частности, для HF-датасета `hf-drone-detection` (его материализует `datasets.download`
    в эту раскладку): сплит берётся из имени подпапки и сохраняется как есть (не пере-делится).
    """
    img_root = root / "images"
    if not img_root.exists():
        return _parse_image_dataset(root)  # на случай иной раскладки
    samples: list[YoloSample] = []
    for split_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        split = split_dir.name if split_dir.name in ("train", "val", "test") else "train"
        imgs = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in _IMG_EXT)
        for img in tqdm(imgs, desc=f"parse {root.name}/{split_dir.name}", unit="img"):
            lbl = root / "labels" / split_dir.name / (img.stem + ".txt")
            yolo_boxes: list[tuple[int, float, float, float, float]] = []
            if lbl.exists():
                yb = _parse_yolo_txt(lbl)
                if yb and all(0.0 <= v <= 1.0 for _, *coords in yb for v in coords):
                    yolo_boxes = yb
                else:  # вдруг абсолютные пиксели
                    w, h = _image_size(img)
                    yolo_boxes = [(_DRONE_CLS, *_xyxy_to_yolo(*bb, w, h)) for bb in _parse_flat_txt(lbl)]
            samples.append(YoloSample(image_path=img, boxes=yolo_boxes, group=f"{root.name}/{split}", split=split))
    if not samples:
        raise RuntimeError(f"в {img_root} не найдено изображений в подпапках split")
    return samples


def _parse_yolo_dir(root: Path) -> list[YoloSample]:
    """Готовый YOLO-датасет (images/ + labels/) — напр. собственный датасет ВКР."""
    return _parse_image_dataset(root)


_PARSERS = {
    "dut-anti-uav": _parse_dut,
    "drone-vs-bird": _parse_dvb,
    "hf-drone-detection": _parse_yolo_split_dir,
    # собственный YOLO-датасет можно положить в train/data/vkr-uav/ и добавить сюда:
    # "vkr-uav": _parse_yolo_dir,
}


def _carve_val_from_train(samples: list[YoloSample], frac: float, seed: int) -> list[YoloSample]:
    """Если нет ни одного сэмпла со split=='val' — выделить долю `frac` из train под val (детерм.).

    Нужно, когда исходный датасет имеет только train/test (как hf-drone-detection): YOLOv8 требует
    непустой val. it-37 (ревью §8): val выделяется ЦЕЛЫМИ ГРУППАМИ (сцена/видео), а не отдельными
    кадрами — соседние кадры одной записи не должны оказаться по разные стороны train/val.
    Группа уходит в val, когда накопленная доля образцов достигает `frac`.
    """
    has_val = any(s.split == "val" for s in samples)
    if has_val or frac <= 0:
        return samples
    train = [s for s in samples if s.split == "train"]
    n_train = len(train)
    if n_train == 0:
        return samples
    # группируем по group; сортировка + shuffle с сидом = детерминизм
    groups: dict[str, list[YoloSample]] = {}
    for s in train:
        groups.setdefault(s.group, []).append(s)
    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    val_groups: set[str] = set()
    taken = 0
    for g in group_keys:  # целые группы, пока не наберём долю frac
        if taken / n_train >= frac:
            break
        val_groups.add(g)
        taken += len(groups[g])
    return [
        YoloSample(image_path=s.image_path, boxes=s.boxes, group=s.group,
                   split="val" if s.group in val_groups else s.split)
        for s in samples
    ]


def convert(*, datasets: list[str] | None = None, splits: dict[str, float] | None = None, seed: int = RANDOM_SEED) -> Path:
    """Собрать единый YOLO-датасет из перечисленных источников; вернуть путь к data.yaml.

    Сэмплы с явным `split` (из YOLO-датасетов со сплитами) попадают в свой сплит; остальные
    делятся детерминированно по группам (сцена/видео) согласно `splits`. Если в итоге нет
    val-сплита — часть train отрезается под val (YOLOv8 требует непустой val).
    """
    splits = splits or DEFAULT_SPLITS
    names = datasets or list(_PARSERS)

    all_samples: list[YoloSample] = []
    for name in names:
        parser = _PARSERS.get(name)
        if parser is None:
            raise KeyError(f"нет парсера для визуального датасета {name!r}; есть: {sorted(_PARSERS)}")
        print(f"[prepare-visual] парсинг датасета {name}...")
        all_samples.extend(parser(dataset_dir(name)))
    if not all_samples:
        raise RuntimeError("парсеры не вернули ни одного кадра — проверьте, что датасеты скачаны и распакованы")

    # split-по-группам — только для сэмплов без явного split
    groups_to_split = [s.group for s in all_samples if s.split not in ("train", "val", "test")]
    split_map = _split_groups(groups_to_split, splits, seed) if groups_to_split else {"train": set(), "val": set(), "test": set()}
    # для сэмплов с явным split: если val-а нет — отрезать его от train (ЦЕЛЫМИ группами, it-37)
    all_samples = _carve_val_from_train(all_samples, float(splits.get("val", 0.1)), seed)
    print(f"[prepare-visual] фильтр классов (дрон-only): {dropped_boxes_report()}")
    _write_yolo(all_samples, split_map)
    return _write_data_yaml()
