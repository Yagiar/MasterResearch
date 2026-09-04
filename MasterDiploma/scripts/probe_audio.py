#!/usr/bin/env python3
"""Прогнать wav через РАНТАЙМ-код акустического детектора и напечатать распределение p(drone) по окнам.

Поддерживает три бэкенда (флаг --arch):
  - lwcnn / resnet18 — наш FeatureExtractor (melspec/mfcc) → LightweightCnnDetector со state_dict (.pt);
  - ast              — AstAudioClassifier (HF transformers), --weights указывает на КАТАЛОГ модели
                       (config.json + preprocessor_config.json + model.safetensors), напр.
                       models/acoustic/samid-drone-detector. У AST свой ASTFeatureExtractor — флаги
                       --feature/--n-mels/--n-mfcc/--n-frames игнорируются (используется его config).

Запуск (lwcnn/resnet18):
    python scripts/probe_audio.py sandboxDataForSimulator/sandbox-audio-for-simulator.wav \
        --weights models/acoustic/lwcnn.pt --arch lwcnn --feature melspec --win-ms 500 --hop-ms 250 --n-mels 64

Запуск (AST samid-drone-detector):
    python scripts/probe_audio.py sandboxDataForSimulator/sandbox-audio-for-simulator.wav \
        --weights models/acoustic/samid-drone-detector --arch ast --win-ms 1000 --hop-ms 500
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
# подключаем рантайм-пакеты сервиса (без установки): acoustic_detector + uavdet_common
sys.path.insert(0, str(_REPO / "services" / "acoustic-detector" / "src"))
sys.path.insert(0, str(_REPO / "libs" / "common" / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--weights", default=str(_REPO / "models" / "acoustic" / "lwcnn.pt"),
                    help="lwcnn/resnet18: путь к .pt со state_dict; ast: КАТАЛОГ с config.json+model.safetensors")
    ap.add_argument("--arch", default="lwcnn", choices=["lwcnn", "resnet18", "ast"], help="бэкенд (= acoustic_detector.arch в configs/pilot.yaml)")
    ap.add_argument("--feature", default="melspec", choices=["mfcc", "melspec"], help="игнорируется для arch=ast")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--win-ms", type=int, default=500)
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--n-mfcc", type=int, default=40)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число окон (0 = все)")
    args = ap.parse_args()

    import librosa  # noqa: PLC0415

    if args.arch == "ast":
        from acoustic_detector.ast_classifier import AstAudioClassifier  # noqa: PLC0415
        detector = AstAudioClassifier(weights_path=args.weights, device=args.device, target_sample_rate=args.sample_rate)
        feature_dim_str, n_frames_str = "AST-internal", "AST-internal"
    else:
        from acoustic_detector.cnn import LightweightCnnDetector  # noqa: PLC0415
        from acoustic_detector.features import FeatureExtractor  # noqa: PLC0415
        extractor = FeatureExtractor(feature=args.feature, sample_rate=args.sample_rate,
                                     n_mfcc=args.n_mfcc, n_mels=args.n_mels, n_frames=args.n_frames)
        detector = LightweightCnnDetector(weights_path=args.weights, feature_dim=extractor.feature_dim,
                                          n_frames=extractor.n_frames, device=args.device, arch=args.arch)
        feature_dim_str, n_frames_str = str(extractor.feature_dim), str(extractor.n_frames)

    print(f"[probe] backend={detector.backend} arch={detector.arch} model={detector.model_name} feature_dim={feature_dim_str} n_frames={n_frames_str}")
    if detector.backend not in ("cnn", "ast"):
        print(f"[probe] ВНИМАНИЕ: веса не загрузились ({args.weights}) — режим energy-threshold, диагностика модели невозможна")

    sig, sr = librosa.load(args.wav, sr=args.sample_rate, mono=True)
    win = int(args.sample_rate * args.win_ms / 1000)
    hop = int(args.sample_rate * args.hop_ms / 1000)
    print(f"[probe] wav: {args.wav}  длит.={len(sig)/sr:.1f}с  sr={sr}  окно={win} сэмплов ({args.win_ms}мс)  шаг={hop} ({args.hop_ms}мс)")

    # тот же путь, что в рантайме: PCM int16 -> base64. Эмулируем PCM-кодирование (как делает source-simulator).
    pcm_full = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    p_drones: list[float] = []
    snrs: list[float] = []
    off, n = 0, 0
    while off + win <= len(sig):
        chunk_b64 = base64.b64encode(pcm_full[off * 2 : (off + win) * 2]).decode("ascii")  # *2 байта на int16
        if getattr(detector, "expects_pcm", False):
            snr = float(detector.features_from_pcm(chunk_b64, src_sample_rate=sr, channels=1).snr_db)
            det = detector.detect(chunk_b64, src_sample_rate=sr, channels=1)
        else:
            feats = extractor.from_audio_raw(chunk_b64, src_sample_rate=sr, channels=1)
            det = detector.detect(feats.array)
            snr = float(feats.snr_db)
        p_drone = det.confidence if det.label == "drone" else 1.0 - det.confidence
        p_drones.append(p_drone)
        snrs.append(snr)
        n += 1
        off += hop
        if args.limit and n >= args.limit:
            break

    if not p_drones:
        print("[probe] нет окон (файл короче окна?)")
        return 1
    p = np.array(p_drones)
    print(f"\n[probe] окон: {len(p)}")
    print(f"[probe] p(drone): min={p.min():.3f}  p10={np.percentile(p,10):.3f}  median={np.median(p):.3f}  mean={p.mean():.3f}  p90={np.percentile(p,90):.3f}  max={p.max():.3f}")
    for thr in (0.5, 0.7, 0.9):
        print(f"[probe]   окон с p(drone) >= {thr}: {int((p>=thr).sum())}/{len(p)} ({100*(p>=thr).mean():.0f}%)")
    print(f"[probe] решение label=drone (p>=0.5): {int((p>=0.5).sum())}/{len(p)} окон")
    print(f"[probe] snr_db: median={np.median(snrs):.1f}  mean={np.mean(snrs):.1f}")
    hist, edges = np.histogram(p, bins=10, range=(0, 1))
    print("[probe] гистограмма p(drone):")
    for i in range(10):
        bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"   [{edges[i]:.1f},{edges[i+1]:.1f})  {hist[i]:5d}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
