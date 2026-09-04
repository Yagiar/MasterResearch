# INDEX — исследовательские итерации (непрерывный режим)

> Рабочий журнал цикла «ресёрч + эксперименты» по развитию диплома. Одна итерация = один .md в `iterations/`.
> Смежные документы: аудит и синтез — в вольте (`MasterDiplomaVaultObsidian/00 — Карта/03 — НИР-2 (текущий семестр)/`).

| Итерация | Дата | Тема | Статус | Ключевой результат |
|---|---|---|---|---|
| [it-01-web-research](iterations/it-01-web-research.md) | 2026-09-04 | Веб-ресёрч улучшений (YOLO small-object, акустика, fusion, MMAUD) | ✅ | P2 +6% mAP, imgsz1280 +25%, SAHI; MMAUD как датасет согласия; entropy-gating литературно подкреплён |
| [it-02-sandbox-gt](iterations/it-02-sandbox-gt.md) | 2026-09-04 | GT-разметка sandbox-видео по секундам | ✅ | Дрон виден все 73 с (0–10 и 66–72 — на полу); «пустые» сегменты только по аудио; GT двухслойный (visible/airborne) |
| [it-03-fusion-vs-gt](iterations/it-03-fusion-vs-gt.md) | 2026-09-04 | Честные P/R/F1 fusion + симуляция весов | ✅ | audio-only(AST) лучший vs airborne F1=0.920 > video 0.870 > late 0.865; глухая акустика роняет late до 0.40; w_v=0.9 лечит (0.77); entropy-gating против уверенно-ошибающегося бесполезен; смешанных окон 0.35% — fusion не активируется |
| it-04-yolo-frames | 2026-09-04 | YOLO-инференс на кадрах sandbox (видео domain gap) | 🔄 | — |
| it-05-ablation-audio | — | Разбор: куда исчезает аудио в №4 (7% vs 75% смешанных окон) | 📋 план | требует docker-infra или instrumentation fusion |

## Артефакты
- `iterations/` — журналы итераций
- `sandbox_frames/` — кадры sandbox-видео для разметки (в git не грузятся, см. .gitignore)
- `gt_sandbox_video.csv` — ground truth (сек → visible/airborne)
- `score_fusion_vs_gt.py` — скоринг решений и симуляция политик
- `.venv/` — окружение для YOLO-инференса (в git не грузится)
