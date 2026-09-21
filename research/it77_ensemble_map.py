#!/usr/bin/env python3
"""it-77 — ансамбль-NMS mAP50/50-95 на DUT600 и HF600 (предрег: iterations/it-77-ensemble-nms-map-PLANNED.md).

GPU-инференс разрешён автором; обучения нет. Правила pre-defined ровно два:
merge-NMS(IoU=0,5) с детерминированным tie-break (conf↓ → old раньше → индекс бокса)
и WRF (справка: подавленный дубль остаётся с conf=min пары и сам ничего не подавляет).
AP — сама функция канона: ultralytics ap_per_class (101-точечная COCO-интерполяция),
матчинг — копия не-scipy ветки Validator.match_predictions (iouv=linspace(0.5,0.95,10)).
Блокатор валидации конвенции: одиночные прогоны харнесом воспроизводят каноны
|mAP50 − канон| ≤ 0,010 на всех 4 ячейках; при расхождении mAP50-95 > 0,020 вердикт не выносится.

Запуск: MasterDiploma/venv/bin/python research/it77_ensemble_map.py
"""
import csv
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
WEIGHTS = {"old": MD / "models/visual/yolov8s-uav.pt",
           "new": MD / "train/runs/visual/uav-yolov8s-bg/weights/best.pt"}
SUBSETS = {"dut": MD / "train/data/_prepared/visual-dut-test600",
           "hf": MD / "train/data/_prepared/visual-hf-test600"}
# каноны eval-visual (train/runs/eval/visual-{old,new}-{dut,hf}600/metrics.json): (mAP50, mAP50-95)
CANON = {("old", "dut"): (0.7201265534220053, 0.3866315075774684),
         ("new", "dut"): (0.9058894731396591, 0.5970806826016426),
         ("old", "hf"): (0.867296503123639, 0.4138648215273979),
         ("new", "hf"): (0.883016759504097, 0.4350995167548751)}
IOUV = np.linspace(0.5, 0.95, 10)
LINES = []


def log(s: str = "") -> None:
    LINES.append(s)
    print(s, flush=True)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy-наборы (M,4)×(N,4) → IoU (M,N) — та же семантика, что torch box_iou."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    sa = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    sb = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / (sa[:, None] + sb[None, :] - inter + 1e-9)


def match_tp(gt: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Копия не-scipy ветки Validator.match_predictions для одного класса: tp (Ndet,10)."""
    tp = np.zeros((len(det), len(IOUV)), dtype=bool)
    if len(gt) == 0 or len(det) == 0:
        return tp
    iou = box_iou(gt, det)  # (Mgt, Ndet)
    for i, thr in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if matches.shape[0]:
            tp[matches[:, 1].astype(int), i] = True
    return tp


def predict(model: YOLO, img_paths: list[Path], shapes: dict[str, tuple]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{stem: (xyxy N×4, conf N)}; boxes уже в пикселях оригинала; shapes — (h,w) попутно.
    Ключ — по порядку отдачи (stream сохраняет порядок): r.path в 8.4 — синтетическое 'imageN.jpg'.
    Чанками по 150 (датасет держит список в RAM — OOM-урок 1) и batch=1 (активации — CUDA OOM на 6 GiB)."""
    out = {}
    for i in range(0, len(img_paths), 150):
        chunk = img_paths[i:i + 150]
        for j, r in enumerate(model.predict(chunk, imgsz=640, conf=0.001, max_det=300, device=0,
                                            verbose=False, stream=True, cache=False, batch=1)):
            p = chunk[j]
            out[p.stem] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
            shapes[p.stem] = r.orig_shape  # (h, w)
        model.predictor.dataset = None
        model.predictor = None  # del нельзя: __getattr__ делегирует на DetectionModel
        torch.cuda.empty_cache()
    assert len(out) == len(img_paths), f"predict отдал {len(out)} из {len(img_paths)}"
    return out


def load_gt(sub_dir: Path, stems: list[str], shapes: dict[str, tuple]) -> dict[str, np.ndarray]:
    gt = {}
    for stem in stems:
        f = sub_dir / "labels/test" / f"{stem}.txt"
        if f.exists() and f.stat().st_size:
            rows = np.loadtxt(f, ndmin=2)  # columns: cls cx cy w h (normalized)
        else:
            rows = np.zeros((0, 5))
        h, w = shapes[stem]
        x1 = np.clip((rows[:, 1] - rows[:, 3] / 2) * w, 0, w)
        y1 = np.clip((rows[:, 2] - rows[:, 4] / 2) * h, 0, h)
        x2 = np.clip((rows[:, 1] + rows[:, 3] / 2) * w, 0, w)
        y2 = np.clip((rows[:, 2] + rows[:, 4] / 2) * h, 0, h)
        gt[stem] = np.stack([x1, y1, x2, y2], axis=1)
    return gt


def score(preds: dict[str, tuple[np.ndarray, np.ndarray]], gt: dict[str, np.ndarray]) -> tuple[float, float, int]:
    tps, confs, tgts = [], [], []
    for stem, (det, cf) in preds.items():
        tps.append(match_tp(gt[stem], det))
        confs.append(cf)
        tgts.append(np.zeros(len(gt[stem])))
    conf_cat = np.concatenate(confs)
    ap = ap_per_class(np.concatenate(tps), conf_cat,
                      np.zeros(len(conf_cat)), np.concatenate(tgts), prefix="")[5]
    return float(ap[0, 0]), float(ap[0].mean()), int(len(conf_cat))


def merge_nms(a: dict[str, tuple[np.ndarray, np.ndarray]], b: dict[str, tuple[np.ndarray, np.ndarray]],
              wrf: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """old⊕new покадрово: сортировка conf↓ → old раньше → индекс бокса; жадный class-agnostic NMS IoU=0,5.
    merge: дубль выбрасывается; wrf: дубль остаётся с conf=min(дубль, подавившего) и не подавляет сам."""
    out = {}
    for stem in a:
        (ba, ca), (bb, cb) = a[stem], b[stem]
        recs = [(ba[i], float(ca[i]), 0, i) for i in range(len(ca))]
        recs += [(bb[i], float(cb[i]), 1, i) for i in range(len(cb))]
        recs.sort(key=lambda x: (-x[1], x[2], x[3]))
        kbox, kconf, sup_box, sup_conf = [], [], [], []
        active = np.zeros((0, 4))  # только «живые» выжившие подавляют (wrf-дубли инертны)
        for box, conf, _m, _i in recs:
            if len(active) and box_iou(active, box.reshape(1, 4))[:, 0].max() > 0.5:
                if wrf:
                    j = int(box_iou(active, box.reshape(1, 4))[:, 0].argmax())
                    sup_box.append(box)
                    sup_conf.append(min(conf, kconf[j]))
                continue
            kbox.append(box)
            kconf.append(conf)
            active = np.stack(kbox)
        boxes = np.array(kbox + sup_box).reshape(-1, 4)
        confs = np.array(kconf + sup_conf)
        out[stem] = (boxes, confs)
    return out


log("it-77: ансамбль-NMS mAP на DUT600/HF600 — GPU-инференс old/new, merge-NMS(IoU=0,5) + WRF (справка)")
results, ndets = {}, {}
for sub, sub_dir in SUBSETS.items():
    imgs = sorted((sub_dir / "images/test").iterdir())
    stems = [p.stem for p in imgs]
    shapes: dict[str, tuple] = {}
    preds = {}
    for tag, w in WEIGHTS.items():
        log(f"[{sub}/{tag}] инференс {len(imgs)} кадров (imgsz640, conf0,001, max_det300)…")
        preds[tag] = predict(YOLO(str(w)), imgs, shapes)
    gt = load_gt(sub_dir, stems, shapes)
    log(f"[{sub}] GT-объектов: {sum(len(g) for g in gt.values())}")
    variants = {"old": preds["old"], "new": preds["new"],
                "merge": merge_nms(preds["old"], preds["new"], wrf=False),
                "wrf": merge_nms(preds["old"], preds["new"], wrf=True)}
    for vt, pv in variants.items():
        m50, m595, nd = score(pv, gt)
        results[(sub, vt)] = (m50, m595)
        ndets[(sub, vt)] = nd
        log(f"[{sub}/{vt:>5}] детекций: {nd:>6}  mAP50={m50:.4f}  mAP50-95={m595:.4f}")

log("\n== блокатор валидации конвенции (одиночные против канонов; допуск mAP50 ±0,010; mAP50-95 расхождение >0,020 → вердикт не выносится) ==")
blocker_ok, mAP5095_far = True, False
for (tag, sub), (c50, c595) in CANON.items():
    h50, h595 = results[(sub, tag)]
    d50, d595 = h50 - c50, h595 - c595
    ok = abs(d50) <= 0.010
    blocker_ok &= ok
    mAP5095_far |= abs(d595) > 0.020
    log(f"  {tag:>3}-{sub}: харнес mAP50={h50:.4f} канон={c50:.4f} Δ={d50:+.4f} {'OK' if ok else 'РАСХОЖДЕНИЕ'} | "
        f"mAP50-95 харнес={h595:.4f} канон={c595:.4f} Δ={d595:+.4f}")

log("\n== вердикт H1–H3 ==")
if not blocker_ok:
    log("БЛОКАТОР КРАСНЫЙ: харнес не воспроизводит канон по mAP50 — вердикт H1–H3 НЕ ВЫНОСИТСЯ.")
elif mAP5095_far:
    log("БЛОКАТОР КРАСНЫЙ: расхождение mAP50-95 > 0,020 — харнес расходящийся, вердикт H1–H3 НЕ ВЫНОСИТСЯ.")
else:
    log("Блокатор зелёный (одиночные воспроизведены); вердикт:")
    h1, h2, h3 = True, False, True
    for sub in SUBSETS:
        b50 = max(results[(sub, "old")][0], results[(sub, "new")][0])
        b595 = max(results[(sub, "old")][1], results[(sub, "new")][1])
        m50, m595 = results[(sub, "merge")]
        r1, r2, r3 = m50 >= b50, m50 - b50 >= 0.010, m595 >= b595 - 0.005
        h1 &= r1
        h2 |= r2
        h3 &= r3
        log(f"  {sub}: merge {m50:.4f} vs лучшая одиночка {b50:.4f} → H1 {'OK' if r1 else 'КРАСН'}, Δ={m50 - b50:+.4f}"
            f" → H2(≥+0,010) {'OK' if r2 else 'нет'}; mAP50-95 {m595:.4f} vs {b595:.4f} → H3 {'OK' if r3 else 'КРАСН'}")
    if h1 and h2:
        log("  ИТОГ: ЗЕЛЁНЫЙ (H1∧H2) — ансамбль-merge = полноценный кандидат офлайн-линейки по всем визуальным осям.")
    elif not h1:
        log("  ИТОГ: H1 КРАСНЫЙ — ансамбль остаётся точечной находкой image-линейки, mAP-ось закрыта отрицательно.")
    else:
        log("  ИТОГ: H1 OK, H2 не набран — не-регрессия без прироста; ансамбль не расширяется на mAP-ось.")

with open(ROOT / "research/it77_ensemble_map.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["subset", "variant", "mAP50", "mAP50_95", "n_dets"])
    for (sub, vt), (m50, m595) in results.items():
        w.writerow([sub, vt, f"{m50:.6f}", f"{m595:.6f}", ndets[(sub, vt)]])
(ROOT / "research/it77_ensemble_map.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("артефакты: research/it77_ensemble_map.{csv,txt}")
