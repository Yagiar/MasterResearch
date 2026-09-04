"""CLI обучающего пайплайна: `python -m uavtrain.cli <команда>`.

Команды:
  list-datasets [--modality visual|audio]   — показать реестр датасетов
  download <name> [--force]                 — скачать (или показать инструкцию ручной загрузки)
  prepare-visual [--datasets a,b]           — собрать YOLO-датасет (data.yaml)
  prepare-audio --positives a,b --negatives c,d   — собрать акустический датасет
  train-visual --data <data.yaml> [--epochs N ...]   — обучить YOLOv8
  train-acoustic --features <features.npz> [...]     — обучить CNN
  eval-visual --weights best.pt --data data.yaml     — оценить YOLOv8 на test
  export-visual --weights best.pt [--metrics metrics.json]  — экспорт весов в ../models

Большинство тяжёлых шагов вынесены в модули `uavtrain.*` — этот CLI лишь парсит аргументы.
Альтернатива CLI — ноутбук `train/notebooks/pipeline.ipynb` (те же функции по секциям).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import datasets as ds
from .config import ensure_dirs


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    for spec in ds.list_datasets(args.modality):
        src = "hf" if spec.hf_repo else ("gdrive" if spec.gdrive else ("url" if spec.url else "manual"))
        print(f"{spec.name:24s} {spec.modality:7s} {spec.role:12s} [{src:6s}]  {spec.description}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        path = ds.download(args.name, force=args.force)
        print(f"готово: {path}")
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _cmd_prepare_visual(args: argparse.Namespace) -> int:
    from . import prepare_visual  # noqa: PLC0415

    path = prepare_visual.convert(datasets=_csv(args.datasets))
    print(f"data.yaml: {path}")
    return 0


def _cmd_prepare_audio(args: argparse.Namespace) -> int:
    from . import prepare_audio  # noqa: PLC0415

    fparams = {
        "feature": args.feature, "n_mfcc": args.n_mfcc, "n_mels": args.n_mels, "n_frames": args.n_frames,
        "win_ms": args.win_ms, "hop_ms": args.hop_ms, "n_fft": args.n_fft, "hop_length": args.hop_length,
        "max_windows_per_file": args.max_windows_per_file,
    }
    path = prepare_audio.build(
        positives=_csv(args.positives) or [],
        negatives=_csv(args.negatives) or [],
        feature_params=fparams,
    )
    print(f"features.npz: {path}  (feature={args.feature})")
    return 0


def _cmd_train_visual(args: argparse.Namespace) -> int:
    from .train_visual import VisualTrainConfig, train  # noqa: PLC0415

    best = train(
        VisualTrainConfig(
            data_yaml=Path(args.data),
            base_weights=args.base_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            cache=args.cache,
            fraction=args.fraction,
            name=args.name,
        )
    )
    print(f"best weights: {best}")
    return 0


def _cmd_train_acoustic(args: argparse.Namespace) -> int:
    import json  # noqa: PLC0415

    from .prepare_audio import resolve_features_dir  # noqa: PLC0415
    from .train_acoustic import AcousticTrainConfig, train  # noqa: PLC0415

    # подтянуть feature_params из meta.json подготовленного набора (feature/sr/win_ms/n_mels/n_frames/...)
    feat_params: dict = {}
    meta_path = resolve_features_dir(Path(args.features)) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feat_params = dict(meta.get("feature_params") or {})
            if "feature_dim" in meta:
                feat_params.setdefault("feature_dim", meta["feature_dim"])
            if "n_frames" in meta:
                feat_params.setdefault("n_frames", meta["n_frames"])
        except Exception:  # noqa: BLE001
            pass
    name = args.name if args.name != "uav-lwcnn" else f"uav-{args.arch}"
    best = train(
        AcousticTrainConfig(
            features_npz=Path(args.features),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            name=name,
            arch=args.arch,
            pretrained=not args.no_pretrained,
            augment=args.augment,
            bg_noise=_csv(args.bg_noise) or [],
            workers=args.workers,
            feature_params=feat_params,
        )
    )
    print(f"best weights: {best}")
    return 0


def _cmd_eval_visual(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_visual  # noqa: PLC0415

    report = evaluate_visual(Path(args.weights), Path(args.data), imgsz=args.imgsz, device=args.device, name=args.name)
    print(f"metrics: {report.metrics}\nartifacts: {report.artifacts_dir}")
    return 0


def _cmd_eval_acoustic(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_acoustic  # noqa: PLC0415

    report = evaluate_acoustic(Path(args.weights), Path(args.features), device=args.device, name=args.name, arch=args.arch)
    print(f"metrics: {report.metrics}\nartifacts: {report.artifacts_dir}")
    return 0


def _cmd_export_visual(args: argparse.Namespace) -> int:
    from .export import export_visual  # noqa: PLC0415

    dst = export_visual(Path(args.weights), metrics_json=Path(args.metrics) if args.metrics else None, out_name=args.out_name)
    print(f"экспортировано: {dst}")
    return 0


def _cmd_export_acoustic(args: argparse.Namespace) -> int:
    from .export import export_acoustic  # noqa: PLC0415

    dst = export_acoustic(Path(args.weights), metrics_json=Path(args.metrics) if args.metrics else None, out_name=args.out_name)
    print(f"экспортировано: {dst}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uavtrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list-datasets", help="показать реестр датасетов")
    s.add_argument("--modality", choices=["visual", "audio"], default=None)
    s.set_defaults(func=_cmd_list_datasets)

    s = sub.add_parser("download", help="скачать датасет (или показать инструкцию)")
    s.add_argument("name")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=_cmd_download)

    s = sub.add_parser("prepare-visual", help="собрать YOLO-датасет")
    s.add_argument("--datasets", default=None, help="имена через запятую (по умолчанию все с парсером)")
    s.set_defaults(func=_cmd_prepare_visual)

    s = sub.add_parser("prepare-audio", help="собрать акустический датасет «дрон / не-дрон»")
    s.add_argument("--positives", required=True,
                   help="источники «дрон» через запятую; элемент = имя датасета или 'имя#подпуть' "
                        "(напр. 'dads-audio#drone' или 'drone-audio-dataset#DroneAudioDataset-master/Binary_Drone_Audio/yes_drone')")
    s.add_argument("--negatives", required=True,
                   help="источники «не-дрон» через запятую (имя датасета или 'имя#подпуть'); напр. 'dads-audio#non-drone' или 'esc-50'")
    s.add_argument("--feature", default="mfcc", choices=["mfcc", "melspec"],
                   help="признак: mfcc (MFCC) | melspec (лог-мел-спектрограмма). ВАЖНО: должен совпадать с acoustic_detector.feature в конфиге")
    s.add_argument("--n-mfcc", type=int, default=40, help="число MFCC (для feature=mfcc)")
    s.add_argument("--n-mels", type=int, default=64, help="число мел-полос (для feature=melspec; и при mfcc — внутр. мел-банк)")
    s.add_argument("--n-frames", type=int, default=64, help="ширина признаков по времени (паддинг/обрезка)")
    s.add_argument("--win-ms", type=int, default=1000, help="длина окна нарезки, мс (для DADS-клипов по 0.5с ставьте 500)")
    s.add_argument("--hop-ms", type=int, default=500, help="шаг окна нарезки, мс")
    s.add_argument("--n-fft", type=int, default=0, help="n_fft для STFT (0 = librosa-дефолт 2048; для коротких окон лучше 512)")
    s.add_argument("--hop-length", type=int, default=0, help="hop_length для STFT (0 = librosa-дефолт n_fft//4; для n_fft=512 → 256)")
    s.add_argument("--max-windows-per-file", type=int, default=0, help="максимум окон с одного wav (0 = без лимита; против перекоса от длинных записей)")
    s.set_defaults(func=_cmd_prepare_audio)

    s = sub.add_parser("train-visual", help="обучить YOLOv8")
    s.add_argument("--data", required=True, help="путь к data.yaml")
    s.add_argument("--base-weights", default="yolov8s.pt")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--device", default="0")
    s.add_argument("--patience", type=int, default=20, help="early stopping (эпох без улучшения val)")
    s.add_argument("--workers", type=int, default=4, help="воркеров DataLoader (меньше — меньше RAM); на машинах с <24 ГБ RAM ставьте 2")
    s.add_argument("--cache", action="store_true", help="кешировать изображения в RAM (НЕ включать на больших датасетах / малой RAM)")
    s.add_argument("--fraction", type=float, default=1.0, help="доля датасета 0<f<=1 (для быстрых прогонов / слабых машин)")
    s.add_argument("--name", default="uav-yolov8s")
    s.set_defaults(func=_cmd_train_visual)

    s = sub.add_parser("train-acoustic", help="обучить акустический классификатор (lwcnn | resnet18)")
    s.add_argument("--features", required=True, help="каталог _prepared/audio/ (или X.npy / features.npz)")
    s.add_argument("--arch", default="lwcnn", choices=["lwcnn", "resnet18"],
                   help="архитектура: lwcnn (~24k параметров, edge) | resnet18 (~11M, качество/обобщение). "
                        "ВАЖНО: должна совпадать с acoustic_detector.arch в configs/pilot.yaml")
    s.add_argument("--no-pretrained", action="store_true", help="для resnet18 — НЕ использовать ImageNet-инициализацию (по умолчанию используется)")
    s.add_argument("--augment", action="store_true",
                   help="аугментации train-сплита (фон/gain/pitch/codec/SpecAugment) — закрытие domain gap; перечитывает wav из index.csv")
    s.add_argument("--bg-noise", default=None,
                   help="источники фонового шума для миксов через запятую (имя датасета или 'имя#подпуть'); напр. 'esc-50,dads-audio#non-drone'")
    s.add_argument("--epochs", type=int, default=50)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--device", default="cuda")
    s.add_argument("--workers", type=int, default=8, help="воркеров DataLoader для train (аугментации = CPU-bound librosa; без --augment не используется)")
    s.add_argument("--name", default="uav-lwcnn", help="имя прогона (по умолчанию uav-<arch>)")
    s.set_defaults(func=_cmd_train_acoustic)

    s = sub.add_parser("eval-visual", help="оценить YOLOv8 на test")
    s.add_argument("--weights", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--device", default="0")
    s.add_argument("--name", default="uav-yolov8s")
    s.set_defaults(func=_cmd_eval_visual)

    s = sub.add_parser("eval-acoustic", help="оценить акустический классификатор на test")
    s.add_argument("--weights", required=True, help="путь к lwcnn.pt (state_dict)")
    s.add_argument("--features", required=True, help="каталог _prepared/audio/ (или X.npy / features.npz)")
    s.add_argument("--arch", default="lwcnn", choices=["lwcnn", "resnet18"], help="архитектура весов (должна совпадать с train-acoustic); ast — отдельной командой, eval на нашем test-сплите для него не предусмотрен")
    s.add_argument("--device", default="cpu")
    s.add_argument("--name", default="uav-lwcnn")
    s.set_defaults(func=_cmd_eval_acoustic)

    s = sub.add_parser("export-visual", help="экспорт весов в ../models/visual")
    s.add_argument("--weights", required=True)
    s.add_argument("--metrics", default=None, help="путь к metrics.json (опц.)")
    s.add_argument("--out-name", default="yolov8s-uav.pt")
    s.set_defaults(func=_cmd_export_visual)

    s = sub.add_parser("export-acoustic", help="экспорт весов в ../models/acoustic")
    s.add_argument("--weights", required=True, help="путь к lwcnn.pt (state_dict)")
    s.add_argument("--metrics", default=None, help="путь к metrics.json (опц.)")
    s.add_argument("--out-name", default="lwcnn.pt")
    s.set_defaults(func=_cmd_export_acoustic)

    return p


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
