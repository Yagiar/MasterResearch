#!/bin/bash
# 5-мин сэмплер витальных признаков для диагностики kernel panic (it-65).
# Строчка CSV: время, GPU темп/утилизация/память, две термозоны CPU, RAM, swap, loadavg.
LOG=/home/otrix/code/GeneralFolderMasterDiploma/research/vitals.csv
[ -f "$LOG" ] || echo "ts,gpu_temp_c,gpu_util_pct,gpu_mem_mib,thermal1_c,thermal2_c,ram_used_mb,ram_total_mb,swap_used_mb,swap_total_mb,load1,load5,load15" > "$LOG"
read -r gt gu gm < <(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')
t1=$(( $(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null || echo 0) / 1000 ))
t2=$(( $(cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null || echo 0) / 1000 ))
read -r ru rt su st < <(free -m | awk 'NR==2{r=$3"/"$2} NR==3{s=$3"/"$2} END{split(r,a,"/"); split(s,b,"/"); print a[1], a[2], b[1], b[2]}')
read -r l1 l5 l15 < <(cut -d' ' -f1-3 /proc/loadavg)
echo "$(date '+%F %T'),$gt,$gu,$gm,$t1,$t2,$ru,$rt,$su,$st,$l1,$l5,$l15" >> "$LOG"
