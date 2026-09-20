#!/bin/bash
# it-68: живой A/B конфига fusion на АКТУАЛЬНЫХ весах — конфигурация A (поставка:
# per-message/k=0) против B (испытанное it-43: watermark/k=5), τ=0,5 (it-43: 0,55 хуже).
# Протокол = боевые этапы ablation_v5.sh (A→B→C, 240 с, скоринг score_stages --burn-in-s 90);
# v5 не правим (исторический воспроизведённый скрипт it-46). Если it-66/C3 даст новый
# оптимальный τ-video — править τ надо в КОПИИ override-логики, осознанно, не здесь.
# U-критерии — в research/iterations/it-68-pilot-config-sync-PLANNED.md.
set -eu
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
LOG=$ROOT/research/it68_ab_run.log

# --- защиты: только после полного финала цепочек it-65 и при свободной памяти ---
grep -q "цепочка it-65 завершена" "$RUN/it65_chain.log" || { echo "ОТКАЗ: цепочка 1 не завершена"; exit 1; }
grep -q "цепочка 2 завершена" "$RUN/it65_chain2.log" || { echo "ОТКАЗ: цепочка 2 не завершена"; exit 1; }
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$avail" -ge 2048 ] || { echo "ОТКАЗ: available ${avail} МиБ < 2048 МиБ"; exit 1; }

# --- provenance (U4): какие веса увидит контейнер (монтируется models/:/models:ro) ---
{
  echo "=== it-68 A/B $(date '+%F %T') ==="
  echo "веса visual: $(sha256sum "$MD/models/visual/yolov8s-uav.pt")"
  echo "реестр: $(tail -1 "$MD/models/registry.csv")"
} | tee "$LOG"

# --- A/B/C-этапы (GPU по умолчанию; GPU=0 — CPU-вариант) ---
bash "$ROOT/research/ablation_v5.sh" 2>&1 | tee -a "$LOG"

echo "готово: лог $LOG; скоринг: research/.venv/bin/python research/score_stages.py --burn-in-s 90 <этапы из строк 'SCORE_OFF' лога>"
