#!/usr/bin/env bash
# Скоринг срезов ablation v3 по media_ts (it-35/it-43): БЕЗ подгонки фазы по GT.
# Вытягивает SCORE_OFF <этап> <a:b> из лога и прогоняет score_stages.py (режим media_ts),
# плюс опционально --fit-phase для сравнения с протоколом v2.
# Запуск: bash research/score_ablation_v3.sh [/tmp/ablation_v3.log] [--fit-phase]
set -u
LOG="${1:-/tmp/ablation_v3.log}"
MODE_FLAG=""
if [ "${2:-}" = "--fit-phase" ]; then MODE_FLAG="--fit-phase"; fi

ARGS=()
while read -r _ rest; do
  rng="${rest##* }"          # последний токен строки = диапазон a:b
  name="${rest% *}"          # остальное — имя этапа (может содержать пробелы)
  name="${name//=/-}"        # '=' внутри имени ломает парсер score_stages (напр. "k=0")
  ARGS+=("${name}=${rng}")
done < <(grep "^SCORE_OFF" "$LOG")

if [ ${#ARGS[@]} -eq 0 ]; then
  echo "нет срезов SCORE_OFF в $LOG"; exit 1
fi

PY="$(dirname "$0")/.venv/bin/python"
[ -x "$PY" ] || PY="$(dirname "$0")/.venv/bin/python"
"$PY" "$(dirname "$0")/score_stages.py" $MODE_FLAG "${ARGS[@]}"
