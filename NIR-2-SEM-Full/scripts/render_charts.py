from __future__ import annotations
from pathlib import Path
import csv
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT_PNG = BASE / "charts_png"
OUT_SVG = BASE / "charts_svg"
OUT_PNG.mkdir(exist_ok=True)
OUT_SVG.mkdir(exist_ok=True)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT_PNG / f"{name}.png", dpi=220)
    fig.savefig(OUT_SVG / f"{name}.svg")
    plt.close(fig)


def yolov8_metrics():
    rows = read_csv(DATA / "yolov8_metrics.csv")
    metrics = [r["metric"] for r in rows]
    series = [
        ("ВКР baseline", [float(r["vkr_baseline"]) for r in rows]),
        ("val, 20 эпох", [float(r["val_20_epochs"]) for r in rows]),
        ("test", [float(r["test"]) for r in rows]),
    ]
    x = list(range(len(metrics)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, (label, values) in enumerate(series):
        xs = [v + (i - 1) * width for v in x]
        bars = ax.bar(xs, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Основные метрики YOLOv8n: baseline, validation и test")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    save(fig, "05_yolov8_metrics")


def acoustic_metrics():
    rows = read_csv(DATA / "acoustic_metrics.csv")
    labels = [r["metric"] for r in rows]
    values = [float(r["value"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bars = ax.bar(range(len(labels)), values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Основные метрики вариантов акустического детектора")
    ax.set_ylabel("Значение")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(4, 0.20, "AST: доля окон на целевом\nаудио с p(drone) >= 0.5",
            ha="center", va="center", fontsize=7)
    save(fig, "06_acoustic_metrics")


if __name__ == "__main__":
    yolov8_metrics()
    acoustic_metrics()
