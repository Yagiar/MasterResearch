"""Обучение акустического детектора — CNN-классификация спектрограмм «дрон / не-дрон» (PyTorch).

Архитектура — флагом `--arch`: `lwcnn` (LightweightAudioCNN ~24k параметров, edge) либо `resnet18`
(ResNet-18 с 1-канальным входом ~11M, качество/обобщение). Определения — в `uavtrain.audio_models`
и ДОЛЖНЫ совпадать с `services/acoustic-detector/src/acoustic_detector/cnn.py` (веса = `state_dict`).

Вход — каталог `_prepared/audio/` (X.npy memmap / y.npy / split.npy / index.csv / meta.json) из
`prepare_audio.build()`. С `--augment` train-сплит проходит аугментации (см. `uavtrain.audio_augment`):
сигнал перечитывается из исходных wav по index.csv, аугментируется (фон/gain/pitch/codec) on-the-fly,
затем melspec + SpecAugment. Val/test — из готового X.npy без аугментаций. Лучший по balanced_acc
чекпойнт пишется в `train/runs/acoustic/<name>/lwcnn.pt`; оттуда `export_acoustic()` копирует в `models/acoustic/`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_models import build_audio_model
from .config import AUDIO_CLASSES, RANDOM_SEED, RUNS_DIR


@dataclass
class AcousticTrainConfig:
    features_npz: Path                       # каталог _prepared/audio/ (или X.npy / features.npz)
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    name: str = "uav-lwcnn"
    seed: int = RANDOM_SEED
    arch: str = "lwcnn"                       # lwcnn | resnet18
    pretrained: bool = True                  # для resnet18 — ImageNet-инициализация conv1/блоков
    augment: bool = False                    # аугментации train-сплита (закрытие domain gap)
    bg_noise: list[str] = field(default_factory=list)  # источники фонового шума для миксов: 'esc-50', 'dads-audio#non-drone', ...
    workers: int = 8                         # воркеров DataLoader для train (аугментации = CPU-bound librosa)
    feature_params: dict = field(default_factory=dict)  # из meta.json (feature/sr/win_ms/n_mels/n_frames/...)


# --- архитектура (обёртка над audio_models для обратной совместимости) ---
def build_audio_cnn(n_classes: int = 2):
    """LightweightAudioCNN — оставлено для обратной совместимости; для resnet18 см. audio_models.build_audio_model."""
    return build_audio_model("lwcnn", n_classes=n_classes)


def _load_splits(features_path: Path):
    """(Xtr,ytr),(Xva,yva),(Xte,yte). X читается через memmap — материализуется только нужный сплит."""
    from .prepare_audio import load_split  # noqa: PLC0415

    return load_split(features_path, 0), load_split(features_path, 1), load_split(features_path, 2)


def _read_index_train_rows(features_path: Path):
    """Прочитать строки index.csv для train-сплита: список (source_wav, offset_s, dur_s, label_idx)."""
    from .prepare_audio import resolve_features_dir  # noqa: PLC0415

    idx_path = resolve_features_dir(features_path) / "index.csv"
    if not idx_path.exists():
        return None
    drone_idx = AUDIO_CLASSES.index("drone")
    nondrone_idx = AUDIO_CLASSES.index("non-drone")
    rows: list[tuple[str, float, float, int]] = []
    with idx_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "train":
                continue
            li = drone_idx if r.get("label") == "drone" else nondrone_idx
            rows.append((r["source_wav"], float(r["offset_s"] or 0.0), float(r["dur_s"] or 0.0), li))
    return rows


class _AugmentedAudioDataset:
    """Dataset для train с on-the-fly аугментацией: перечитывает 0.5с-окно из исходного wav по index.csv,
    аугментирует сигнал (фон/gain/pitch/codec), считает melspec, применяет SpecAugment, отдаёт ([1,F,T], label).
    Импорты librosa/torch — ленивые (в воркерах)."""

    def __init__(self, rows, feature_params: dict, aug_cfg) -> None:
        self._rows = rows
        self._p = feature_params
        self._aug = aug_cfg

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int):
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from .audio_augment import augment_signal, spec_augment  # noqa: PLC0415
        from .prepare_audio import _extract_features  # noqa: PLC0415

        src, off, dur, label = self._rows[i]
        sr = int(self._p.get("sample_rate", 16000))
        sig, _ = librosa.load(src, sr=sr, offset=off, duration=dur if dur > 0 else None, mono=True)
        sig = sig.astype(np.float32)
        if self._aug.enabled:
            sig = augment_signal(sig, self._aug, sr, librosa)
        feat = _extract_features(sig, self._p, librosa)  # [F, T], z-score
        if self._aug.enabled:
            feat = spec_augment(feat, self._aug)
        return torch.from_numpy(feat).float().unsqueeze(0), int(label)


def _predict(model, X: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    """Предсказания батчами (целый массив на GPU может не влезть — напр. на 6 ГБ при ~50k окон 64×64)."""
    import torch  # noqa: PLC0415

    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size])).float().unsqueeze(1).to(device)  # [B,1,F,T]
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.int64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """accuracy + per-class recall + balanced accuracy (среднее per-class recall).

    На несбалансированных данных ориентируемся на `balanced_acc`, а не на `accuracy`
    (иначе «всегда мажоритарный класс» выглядит хорошо при нулевой пользе).
    """
    if len(y_true) == 0:
        return {"accuracy": 0.0, "balanced_acc": 0.0}
    acc = float((y_pred == y_true).mean())
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        recalls.append(float((y_pred[mask] == c).mean()) if mask.any() else float("nan"))
    valid = [r for r in recalls if not np.isnan(r)]
    bal = float(np.mean(valid)) if valid else 0.0
    out = {"accuracy": acc, "balanced_acc": bal}
    for c, r in enumerate(recalls):
        out[f"recall_{AUDIO_CLASSES[c]}"] = (0.0 if np.isnan(r) else r)
    return out


def _class_weights(y: np.ndarray, n_classes: int):
    """Веса классов = N / (n_classes * count_c) — компенсация дисбаланса в CrossEntropy."""
    import torch  # noqa: PLC0415

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _make_train_loader(cfg: AcousticTrainConfig, Xtr: np.ndarray, ytr: np.ndarray, *, dataset_dir_fn):
    """Собрать DataLoader для train: с аугментацией — _AugmentedAudioDataset (из index.csv), иначе — TensorDataset поверх X.npy."""
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    if cfg.augment:
        rows = _read_index_train_rows(cfg.features_npz)
        if not rows:
            raise RuntimeError("--augment требует index.csv в каталоге признаков (перечитывает сигнал из исходных wav); "
                               "запустите prepare-audio заново или уберите --augment")
        from .audio_augment import AugmentConfig, collect_background_wavs  # noqa: PLC0415

        bg = collect_background_wavs(cfg.bg_noise, dataset_dir_fn=dataset_dir_fn) if cfg.bg_noise else []
        aug_cfg = AugmentConfig(enabled=True, bg_wavs=bg)
        print(f"[acoustic] аугментации ВКЛ: {len(rows)} train-окон из index.csv, фоновых wav для миксов: {len(bg)}")
        ds = _AugmentedAudioDataset(rows, cfg.feature_params or {}, aug_cfg)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                          num_workers=cfg.workers, persistent_workers=cfg.workers > 0, pin_memory=True)
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(Xtr)).float().unsqueeze(1), torch.from_numpy(ytr).long())
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)


def train(cfg: AcousticTrainConfig) -> Path:
    """Обучить акустический классификатор; вернуть путь к лучшим весам (state_dict, lwcnn.pt)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    from .datasets import dataset_dir  # noqa: PLC0415

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from .prepare_audio import load_features, load_split  # noqa: PLC0415

    # при --augment train читается из index.csv (raw wav), Xtr целиком в RAM не нужен — берём только y/split
    if cfg.augment:
        _, y_all, split_all = load_features(cfg.features_npz, mmap=True)
        ytr = y_all[split_all == 0]
        Xtr = np.empty((len(ytr), 0, 0), np.float32)  # плейсхолдер: для весов классов нужна только длина/ytr
    else:
        Xtr, ytr = load_split(cfg.features_npz, 0)
    Xva, yva = load_split(cfg.features_npz, 1)
    Xte, yte = load_split(cfg.features_npz, 2)
    if len(ytr) == 0:
        raise RuntimeError(f"в {cfg.features_npz} пустой train-сплит — проверьте prepare_audio.build()")
    n_cls = len(AUDIO_CLASSES)

    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out_dir = RUNS_DIR / "acoustic" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "lwcnn.pt"  # имя файла историческое; внутри — state_dict выбранной arch

    tr_counts = np.bincount(ytr, minlength=n_cls)
    print(f"[acoustic] arch={cfg.arch}  train: {len(ytr)} окон, классы {dict(zip(AUDIO_CLASSES, tr_counts.tolist()))}; "
          f"val: {len(Xva)}, test: {len(Xte)}; device={device}; augment={cfg.augment}")

    loader = _make_train_loader(cfg, Xtr, ytr, dataset_dir_fn=dataset_dir)

    fd = int((cfg.feature_params or {}).get("feature_dim") or (cfg.feature_params or {}).get("n_mels") or 64)
    nf = int((cfg.feature_params or {}).get("n_frames") or 64)
    model = build_audio_model(cfg.arch, feature_dim=fd, n_frames=nf, n_classes=n_cls, pretrained=cfg.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(ytr, n_cls).to(device))

    from tqdm import tqdm  # noqa: PLC0415

    X_val_eval, y_val_eval = (Xva, yva) if len(Xva) else (Xtr, ytr)
    best_score = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"acoustic[{cfg.arch}] epoch {epoch}/{cfg.epochs}", unit="batch", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running = 0.9 * running + 0.1 * float(loss.item()) if running else float(loss.item())
            pbar.set_postfix(loss=f"{running:.4f}")
        m = _metrics(y_val_eval, _predict(model, X_val_eval, device), n_cls)
        score = m["balanced_acc"]  # выбираем лучший по сбалансированной точности
        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), str(best_path))
        print(f"[acoustic] epoch {epoch}/{cfg.epochs}  val_acc={m['accuracy']:.4f}  val_balanced_acc={m['balanced_acc']:.4f}  "
              f"(recall: drone={m.get('recall_drone', 0):.3f}, non-drone={m.get('recall_non-drone', 0):.3f})  best_bal={best_score:.4f}")

    if len(Xte):
        mt = _metrics(yte, _predict(model, Xte, device), n_cls)
        print(f"[acoustic] обучение завершено ({cfg.arch}). test: acc={mt['accuracy']:.4f} balanced_acc={mt['balanced_acc']:.4f} "
              f"recall(drone)={mt.get('recall_drone', 0):.3f} recall(non-drone)={mt.get('recall_non-drone', 0):.3f}  -> {best_path}")
    else:
        print(f"[acoustic] обучение завершено ({cfg.arch}): best_val_balanced_acc={best_score:.4f}  -> {best_path}")
    if not best_path.exists():
        raise RuntimeError(f"чекпойнт {best_path} не создан — проверьте логи обучения")
    return best_path
