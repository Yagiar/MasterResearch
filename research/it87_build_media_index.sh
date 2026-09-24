#!/usr/bin/env bash
# it-87 H0-новизна: строит sha256-индекс ВСЕГО медиа-фундамента цикла (входы it-01…86),
# чтобы в момент появления материала проверка «хэши не пересекают» была grep -F за секунды,
# а не часовым сканом перед однократным пуском.
# Выход: research/it87_media_index.sha256 — строки `<sha>  <путь>` (2-space — формат sha256sum).
# holdout-24 ИСКЛЮЧЁН из обхода (prune): иначе пересборка индекса после появления материала
# включила бы боевые файлы, и проверка «не пересекают» матчевала бы их самих себя.
# Проверка в бою (шаг 1в): `sha256sum holdout-24/*.{mp4,wav} | cut -d' ' -f1 | grep -F -f - индекс`
# → непусто = материал НЕ новый (пуск не состоится); пусто = H0-новизна зафиксирована.
# Честная область: детектирует ПОЛНОЕ байтовое совпадение файла (в т.ч. целиком лежит внутри
# архива — архивы тоже хэшируются); перемонтированный клип-фрагмент индекс НЕ ловит —
# новизна фрагментов гарантируется протоколом записи (свежая съёмка автора, спека it-87).
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
OUT="$ROOT/research/it87_media_index.sha256"
TMP="$OUT.tmp"
: > "$TMP"
find "$ROOT/MasterDiploma/train/data" \
     "$ROOT/MasterDiploma/sandboxDataForSimulator" \
     "$ROOT/MasterDiploma/data" \
     -path "$ROOT/MasterDiploma/sandboxDataForSimulator/holdout-24" -prune -o \
     -type f \( -iname '*.mp4' -o -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' \
             -o -iname '*.m4a' -o -iname '*.ogg' -o -iname '*.avi' -o -iname '*.mov' \
             -o -iname '*.mkv' -o -iname '*.zip' \) -print0 \
| xargs -0 -r sha256sum >> "$TMP"
mv "$TMP" "$OUT"
echo "индекс готов: $(wc -l < "$OUT") записей → $OUT"
