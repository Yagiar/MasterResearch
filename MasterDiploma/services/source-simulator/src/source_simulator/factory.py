"""AdapterFactory — создание адаптера источника и (опц.) канала деградации по
конфигу (паттерн Factory). Скрывает сопоставление `source.adapter` -> класс и
сборку цепочки стратегий деградации.
"""

from __future__ import annotations

from typing import Any

from .adapters.base import DataSourceAdapter
from .adapters.dataset_replay import DatasetReplayAdapter
from .adapters.media_file import MediaFileAdapter
from .adapters.mmaud_replay import MmaudReplayAdapter
from .adapters.synthetic import SyntheticAdapter
from .adapters.video_file import VideoFileAdapter
from .degradation.channel import DegradationChannel
from .degradation.strategies import (
    AudioNoise,
    ConfidenceJitter,
    DegradationStrategy,
    DtShift,
    FrameDrop,
    ModalityDropout,
    PassThrough,
    WindowDrop,
)

_ADAPTERS = {
    "dataset_replay": DatasetReplayAdapter,
    "media_file": MediaFileAdapter,
    "mmaud_replay": MmaudReplayAdapter,
    "video_file": VideoFileAdapter,
    "synthetic": SyntheticAdapter,
}


def build_adapter(source_cfg: dict[str, Any]) -> DataSourceAdapter:
    """Собрать адаптер источника по секции `source` из конфига."""
    adapter_name = str(source_cfg.get("adapter", "dataset_replay"))
    cls = _ADAPTERS.get(adapter_name)
    if cls is None:
        raise ValueError(f"неизвестный адаптер источника: {adapter_name!r}; доступны: {sorted(_ADAPTERS)}")
    source_id = str(source_cfg.get("source_id", "cam-01"))

    if adapter_name == "dataset_replay":
        p = dict(source_cfg.get("dataset_replay", {}))
        return DatasetReplayAdapter(
            source_id=source_id,
            video_path=str(p.get("video_path", "/data/sample/sample.mp4")),
            fps=float(source_cfg.get("fps", p.get("fps", 25.0))),
            loop=bool(p.get("loop", True)),
            jpeg_quality=int(source_cfg.get("jpeg_quality", p.get("jpeg_quality", 75))),
        )
    if adapter_name == "media_file":
        p = dict(source_cfg.get("media_file", {}))
        audio_cfg = dict(source_cfg.get("audio", {}))
        return MediaFileAdapter(
            source_id=source_id,
            video_path=str(p.get("video_path", "/data/sample/sample.mp4")),
            audio_path=(str(p["audio_path"]) if p.get("audio_path") else None),
            fps=float(source_cfg.get("fps", p.get("fps", 25.0))),
            loop=bool(p.get("loop", True)),
            jpeg_quality=int(source_cfg.get("jpeg_quality", p.get("jpeg_quality", 75))),
            audio_sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            audio_win_ms=int(audio_cfg.get("win_ms", 1000)),
            audio_hop_ms=int(audio_cfg.get("hop_ms", 500)),
        )
    if adapter_name == "mmaud_replay":
        p = dict(source_cfg.get("mmaud_replay", {}))
        audio_cfg = dict(source_cfg.get("audio", {}))
        return MmaudReplayAdapter(
            source_id=source_id,
            video_path=str(p.get("video_path", "/data/mmaud/V1/Mavic2/image")),
            audio_path=(str(p["audio_path"]) if p.get("audio_path") else None),
            annotations_path=(str(p["annotations_path"]) if p.get("annotations_path") else None),
            fps=float(p.get("fps", 30.0)),
            loop=bool(p.get("loop", True)),
            jpeg_quality=int(p.get("jpeg_quality", 80)),
            audio_sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            audio_win_ms=int(audio_cfg.get("win_ms", 1000)),
            audio_hop_ms=int(audio_cfg.get("hop_ms", 500)),
            audio_channel=int(p.get("audio_channel", 0)),
        )
    # заготовки (video_file / synthetic): создаём с минимальным набором
    return cls(source_id=source_id, **dict(source_cfg.get(adapter_name, {})))


def _build_strategies(degr_cfg: dict[str, Any]) -> list[DegradationStrategy]:
    """Собрать список стратегий деградации по секции `source.degradation`.

    Поддерживаемые ключи: `dt_shift_ms` (+ `dt_shift_target`), `frame_drop_prob`,
    `window_drop_prob`, `audio_noise_snr_db`, `modality_dropout` (список [start_s, end_s] +
    `modality_dropout_target`), `confidence_jitter_prob` (+ `confidence_jitter_quality`).
    `seed` — общий сид для воспроизводимости. Если ничего не задано — `[PassThrough()]`.
    """
    strategies: list[DegradationStrategy] = []
    seed = degr_cfg.get("seed")

    if float(degr_cfg.get("dt_shift_ms", 0.0)) != 0.0:
        strategies.append(DtShift(shift_ms=float(degr_cfg["dt_shift_ms"]), target=str(degr_cfg.get("dt_shift_target", "audio"))))
    if float(degr_cfg.get("frame_drop_prob", 0.0)) > 0:
        strategies.append(FrameDrop(drop_prob=float(degr_cfg["frame_drop_prob"]), seed=seed))
    if float(degr_cfg.get("window_drop_prob", 0.0)) > 0:
        strategies.append(WindowDrop(drop_prob=float(degr_cfg["window_drop_prob"]), seed=seed))
    if degr_cfg.get("audio_noise_snr_db") is not None:
        strategies.append(AudioNoise(target_snr_db=float(degr_cfg["audio_noise_snr_db"]), seed=seed))
    intervals = degr_cfg.get("modality_dropout") or []
    if intervals:
        ivs = [(float(a), float(b)) for a, b in intervals]
        strategies.append(ModalityDropout(modality=str(degr_cfg.get("modality_dropout_target", "audio")), intervals_s=ivs))
    if float(degr_cfg.get("confidence_jitter_prob", 0.0)) > 0:
        strategies.append(ConfidenceJitter(prob=float(degr_cfg["confidence_jitter_prob"]), low_quality=int(degr_cfg.get("confidence_jitter_quality", 25)), seed=seed))

    return strategies or [PassThrough()]


def maybe_wrap_degradation(adapter: DataSourceAdapter, source_cfg: dict[str, Any]) -> DataSourceAdapter:
    """Обернуть адаптер DegradationChannel'ом, если `source.degradation.enabled: true`."""
    degr_cfg = dict(source_cfg.get("degradation", {}))
    if not degr_cfg.get("enabled", False):
        return adapter
    return DegradationChannel(adapter, _build_strategies(degr_cfg))
