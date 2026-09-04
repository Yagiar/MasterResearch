"""Подготовка акустических данных к обучению lightweight CNN.

Из скачанных датасетов формирует набор «дрон / не-дрон»:
    <PREPARED_DIR>/audio/
        features.npz     # X: float32 [N, F, T] (MFCC или мел-спектрограмма), y: int [N], split: int [N] (0=train,1=val,2=test)
        index.csv        # source_wav, label, split, offset_s, dur_s
        meta.json        # параметры извлечения признаков + классы + сплиты + seed

Алгоритм: собрать wav-файлы (positives — из дрон-датасетов; negatives — ESC-50/AudioSet/DroneAudio
«unknown»), порезать каждый на окна `win_ms` с шагом `hop_ms`, извлечь признаки (librosa), привести
к фиксированной ширине `n_frames` + z-score; split — по исходному файлу (окна одной записи не утекают
между сплитами). Признаки совпадают по формату с тем, что считает `acoustic-detector` в рантайме.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import AUDIO_CLASSES, DEFAULT_SPLITS, PREPARED_DIR, RANDOM_SEED
from .datasets import dataset_dir

AUDIO_PREPARED = PREPARED_DIR / "audio"
_EPS = 1e-8


def resolve_features_dir(features_path: Path) -> Path:
    """По пути --features (X.npy / features.npz / каталог) вернуть каталог с X.npy/y.npy/split.npy.

    Поддерживается и старый формат features.npz (X/y/split в одном архиве) — тогда возвращается его каталог.
    """
    p = Path(features_path)
    if p.is_dir():
        return p
    if p.name in ("X.npy", "features.npz") or p.suffix in (".npy", ".npz"):
        return p.parent
    return p.parent if p.parent.exists() else p


def load_features(features_path: Path, *, mmap: bool = True):
    """Загрузить (X, y, split). X — memmap (mmap=True) либо обычный массив. Понимает npy-trio и старый npz."""
    d = resolve_features_dir(features_path)
    x_npy = d / "X.npy"
    if x_npy.exists():
        X = np.load(x_npy, mmap_mode="r" if mmap else None)
        y = np.load(d / "y.npy")
        split = np.load(d / "split.npy")
        return X, y.astype(np.int64), split.astype(np.int64)
    npz = d / "features.npz"
    if npz.exists():  # старый формат
        data = np.load(npz)
        return data["X"].astype(np.float32), data["y"].astype(np.int64), data["split"].astype(np.int64)
    # на случай, если передали путь прямо на .npz/.npy
    p = Path(features_path)
    if p.suffix == ".npz" and p.exists():
        data = np.load(p)
        return data["X"].astype(np.float32), data["y"].astype(np.int64), data["split"].astype(np.int64)
    raise FileNotFoundError(f"не найдены признаки: ни {x_npy}, ни {npz} (путь: {features_path})")


def load_split(features_path: Path, split_idx: int, *, mmap: bool = True):
    """Загрузить один сплит (0=train,1=val,2=test) как (X_split, y_split). X_split — обычный массив (materialized)."""
    X, y, split = load_features(features_path, mmap=mmap)
    mask = split == split_idx
    return np.asarray(X[mask], dtype=np.float32), y[mask]

# синхронизированы с services/acoustic-detector/.../features.py
DEFAULT_FEATURE_PARAMS = {
    "feature": "mfcc",     # "mfcc" | "melspec"
    "sample_rate": 16000,
    "win_ms": 1000,
    "hop_ms": 500,
    "n_mfcc": 40,
    "n_mels": 64,
    "n_frames": 64,
    # STFT-параметры: 0 -> librosa-дефолты (n_fft=2048). Для коротких окон (напр. 500 мс @16k = 8000 сэмплов)
    # имеет смысл явно задать поменьше (n_fft=512, hop_length=256), иначе librosa паддит нулями и шумит warnings.
    "n_fft": 0,
    "hop_length": 0,
    # ограничение числа окон с одного исходного wav (0 = без лимита) — против перекоса от длинных записей.
    "max_windows_per_file": 0,
}


@dataclass(frozen=True)
class _ClipRef:
    source_wav: Path
    label_idx: int
    offset_s: float
    dur_s: float


def _resolve_source_dir(spec: str) -> Path:
    """spec — имя датасета или `имя#подпуть` (подпуть относительно каталога датасета).

    Пример: `drone-audio-dataset#DroneAudioDataset-master/Binary_Drone_Audio/yes_drone`.
    Если `#`-часть не указана — берётся весь каталог датасета.
    """
    if "#" in spec:
        name, sub = spec.split("#", 1)
        return dataset_dir(name.strip()) / sub.strip()
    return dataset_dir(spec.strip())


def _collect_wavs(dataset_specs: list[str]) -> list[Path]:
    wavs: list[Path] = []
    for spec in dataset_specs:
        root = _resolve_source_dir(spec)
        if not root.exists():
            raise FileNotFoundError(
                f"путь к аудио не найден: {root} (источник {spec!r}; скачайте датасет или проверьте подпуть после '#')"
            )
        wavs.extend(sorted(root.rglob("*.wav")))
    if not wavs:
        raise RuntimeError(f"в источниках {dataset_specs} не найдено ни одного .wav")
    return wavs


def _windows_of(
    wav: Path, label_idx: int, *, sr: int, win_s: float, hop_s: float, librosa_mod, max_windows: int = 0
) -> list[_ClipRef]:
    dur = float(librosa_mod.get_duration(path=str(wav)))
    if dur < win_s:  # короче окна — берём файл целиком как одно окно (длина < win_s, паддинг при извлечении)
        return [_ClipRef(wav, label_idx, 0.0, max(dur, 0.0))]
    out: list[_ClipRef] = []
    offset = 0.0
    while offset + win_s <= dur + _EPS:
        out.append(_ClipRef(wav, label_idx, round(offset, 3), win_s))
        offset += hop_s
        if max_windows and len(out) >= max_windows:
            break
    return out


def _split_by_file(clips: list[_ClipRef], splits: dict[str, float], seed: int) -> dict[Path, int]:
    """Назначить каждому исходному wav номер сплита (0/1/2) детерминированно."""
    rng = random.Random(seed)
    files = sorted({c.source_wav for c in clips}, key=str)
    rng.shuffle(files)
    n = len(files)
    n_train, n_val = int(n * splits["train"]), int(n * splits["val"])
    mapping: dict[Path, int] = {}
    for i, f in enumerate(files):
        mapping[f] = 0 if i < n_train else (1 if i < n_train + n_val else 2)
    return mapping


def _fix_width(feat: np.ndarray, n_frames: int) -> np.ndarray:
    f, t = feat.shape
    if t == n_frames:
        out = feat
    elif t > n_frames:
        out = feat[:, :n_frames]
    else:
        out = np.pad(feat, ((0, 0), (0, n_frames - t)), mode="constant")
    return out.astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + _EPS)


def _stft_kwargs(params: dict) -> dict:
    """n_fft/hop_length для librosa.feature.* — только если заданы (>0); иначе librosa-дефолты."""
    kw: dict = {}
    if int(params.get("n_fft") or 0) > 0:
        kw["n_fft"] = int(params["n_fft"])
    if int(params.get("hop_length") or 0) > 0:
        kw["hop_length"] = int(params["hop_length"])
    return kw


def _extract_features(signal: np.ndarray, params: dict, librosa_mod) -> np.ndarray:
    if signal.size == 0:
        dim = params["n_mfcc"] if params["feature"] == "mfcc" else params["n_mels"]
        return np.zeros((dim, params["n_frames"]), dtype=np.float32)
    stft_kw = _stft_kwargs(params)
    if params["feature"] == "mfcc":
        feat = librosa_mod.feature.mfcc(y=signal, sr=params["sample_rate"], n_mfcc=params["n_mfcc"], **stft_kw)
    else:
        mel = librosa_mod.feature.melspectrogram(y=signal, sr=params["sample_rate"], n_mels=params["n_mels"], **stft_kw)
        feat = librosa_mod.power_to_db(mel, ref=np.max)
    return _zscore(_fix_width(feat, params["n_frames"]))


def build(
    *,
    positives: list[str],
    negatives: list[str],
    feature_params: dict | None = None,
    splits: dict[str, float] | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """Собрать акустический датасет «дрон / не-дрон»; вернуть путь к features.npz."""
    import librosa  # noqa: PLC0415

    params = {**DEFAULT_FEATURE_PARAMS, **(feature_params or {})}
    splits = splits or DEFAULT_SPLITS
    AUDIO_PREPARED.mkdir(parents=True, exist_ok=True)

    sr = int(params["sample_rate"])
    win_s = params["win_ms"] / 1000.0
    hop_s = params["hop_ms"] / 1000.0
    drone_idx = AUDIO_CLASSES.index("drone")
    nondrone_idx = AUDIO_CLASSES.index("non-drone")

    max_win = int(params.get("max_windows_per_file") or 0)
    pos_wavs, neg_wavs = _collect_wavs(positives), _collect_wavs(negatives)
    clips: list[_ClipRef] = []
    for wav in tqdm(pos_wavs, desc="окна (дрон)", unit="wav"):
        clips += _windows_of(wav, drone_idx, sr=sr, win_s=win_s, hop_s=hop_s, librosa_mod=librosa, max_windows=max_win)
    for wav in tqdm(neg_wavs, desc="окна (не-дрон)", unit="wav"):
        clips += _windows_of(wav, nondrone_idx, sr=sr, win_s=win_s, hop_s=hop_s, librosa_mod=librosa, max_windows=max_win)
    print(f"[prepare-audio] окон: {len(clips)} (из {len(pos_wavs)} дрон + {len(neg_wavs)} не-дрон wav)")

    split_map = _split_by_file(clips, splits, seed)

    n = len(clips)
    feat_dim = int(params["n_mfcc"]) if params["feature"] == "mfcc" else int(params["n_mels"])
    n_frames = int(params["n_frames"])
    # X пишем сразу в memmap на диск (для 300k+ окон 64×64 это ~5 ГБ — в RAM не влезет, OOM-killer убьёт процесс).
    x_path = AUDIO_PREPARED / "X.npy"
    X_mm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, feat_dim, n_frames))
    y = np.empty((n,), dtype=np.int64)
    s = np.empty((n,), dtype=np.int64)

    index_path = AUDIO_PREPARED / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_wav", "label", "split", "offset_s", "dur_s"])
        w.writeheader()
        for i, c in enumerate(tqdm(clips, desc="признаки", unit="окно")):
            sig, _ = librosa.load(str(c.source_wav), sr=sr, offset=c.offset_s, duration=c.dur_s if c.dur_s > 0 else None, mono=True)
            X_mm[i] = _extract_features(sig.astype(np.float32), params, librosa)
            y[i] = c.label_idx
            sp = split_map[c.source_wav]
            s[i] = sp
            w.writerow({"source_wav": str(c.source_wav), "label": AUDIO_CLASSES[c.label_idx],
                        "split": ["train", "val", "test"][sp], "offset_s": c.offset_s, "dur_s": c.dur_s})
            if (i + 1) % 50000 == 0:
                X_mm.flush()
    X_mm.flush()
    del X_mm  # закрыть memmap (данные уже на диске)
    np.save(AUDIO_PREPARED / "y.npy", y)
    np.save(AUDIO_PREPARED / "split.npy", s)

    with (AUDIO_PREPARED / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"feature_params": params, "classes": AUDIO_CLASSES, "splits": splits, "seed": seed,
             "n_clips": n, "feature_dim": feat_dim, "n_frames": n_frames,
             "positives": positives, "negatives": negatives, "format": "npy-trio (X.npy/y.npy/split.npy)"},
            fh, ensure_ascii=False, indent=2,
        )
    return x_path
