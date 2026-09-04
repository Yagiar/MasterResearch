# INDEX — исследовательские итерации (непрерывный режим)

> Рабочий журнал цикла «ресёрч + эксперименты» по развитию диплома. Одна итерация = один .md в `iterations/`.
> Смежные документы: аудит и синтез — в вольте (`MasterDiplomaVaultObsidian/00 — Карта/03 — НИР-2 (текущий семестр)/`).

| Итерация | Дата | Тема | Статус | Ключевой результат |
|---|---|---|---|---|
| [it-01-web-research](iterations/it-01-web-research.md) | 2026-09-04 | Веб-ресёрч улучшений (YOLO small-object, акустика, fusion, MMAUD) | ✅ | P2 +6% mAP, imgsz1280 +25%, SAHI; MMAUD как датасет согласия; entropy-gating литературно подкреплён |
| [it-02-sandbox-gt](iterations/it-02-sandbox-gt.md) | 2026-09-04 | GT-разметка sandbox-видео по секундам | ✅ | Дрон виден все 73 с (0–10 и 66–72 — на полу); «пустые» сегменты только по аудио; GT двухслойный (visible/airborne) |
| [it-03-fusion-vs-gt](iterations/it-03-fusion-vs-gt.md) | 2026-09-04 | Честные P/R/F1 fusion + симуляция весов | ✅ | audio-only(AST) лучший vs airborne F1=0.920 > video 0.870 > late 0.865; глухая акустика роняет late до 0.40; w_v=0.9 лечит (0.77); entropy-gating против уверенно-ошибающегося бесполезен; смешанных окон 0.35% — fusion не активируется |
| [it-04-yolo-frames](iterations/it-04-yolo-frames.md) | 2026-09-04 | YOLO-инференс на кадрах sandbox | ✅ | Видео-домен-гэпа нет (recall 100%, P=1.0 vs visible); фаза it-03 валидирована; conf на земле выше, чем в воздухе |
| [it-05-ast-windows](iterations/it-05-ast-windows.md) | 2026-09-04 | AST офлайн по 144 окнам аудио | ✅ | P=0.977, R=0.766, F1=0.859 vs airborne; профиль совпал с GT (взлёт 10.5 с); FP — выбег винтов после посадки |
| [it-06-fusion-sim-full](iterations/it-06-fusion-sim-full.md) | 2026-09-04 | Fusion на полностью смешанных окнах | ✅ | Веса не помогают (видео неинформативно для airborne); **медиана-5 p_a → F1=0.922, P=1.000** — лучший результат, +5.1 п.п. к одиночным; entropy-gating опровергнут данными |
| [it-07-stress-sim](iterations/it-07-stress-sim.md) | 2026-09-04 | Стресс-симуляция политик (дроп/шум/outage) | ✅ | late+медиана-5 самая устойчивая (0.888 @ 60% дроп > video-only); Δ вредит под шумом; медиану — только по валидным окнам |
| [it-08-temporal-filters-patch](iterations/it-08-temporal-filters-patch.md) | 2026-09-04 | Медиана vs EMA vs гистерезис + эскиз патча | ✅ | Каузальная медиана-5: F1 0.913 без задержки; эскиз патча fusion + критерии приёмки |
| [it-09-web-research-2](iterations/it-09-web-research-2.md) | 2026-09-04 | Веб-ресёрч №2 (recall AST, конфьюзеры, MMAUD) | ✅ | Калибровка порога + PETL; AeroSonicDB/YPAD-0523 как конфьюзеры; MMAUD доступен на GitHub |
| [it-10-summary-vault](../../MasterDiplomaVaultObsidian/00%20—%20Карта/03%20—%20НИР-2%20(текущий%20семестр)/Цикл%20улучшения%20НИР-2%20—%20сводка%20итераций%20(2026-09-04).md) | 2026-09-04 | Сводный отчёт цикла в Vault | ✅ | Сводка 1–9 + план внедрения перенесены в базу знаний |
| [it-11-median-smoother-impl](iterations/it-11-median-smoother-impl.md) | 2026-09-04 | MedianSmoother в коде fusion (off по умолчанию) | ✅ | fusion/temporal.py + consumer/__main__/config (+15 строк), 7/7 тестов, ruff чист |
| [it-12-threshold-calibration](iterations/it-12-threshold-calibration.md) | 2026-09-04 | Калибровка порога AST | ✅ | Опровергнута: F1 плоский 0.05–0.5 (модель бимодальна); late оптимум 0.45–0.5; рекомендация отдавать вероятности обоих классов |
| [it-13-systematic-channel-shift](iterations/it-13-systematic-channel-shift.md) | 2026-09-04 | Систематический сдвиг канала | ✅ | Глухой канал на смешанных окнах: late F1=0.000 (!); защита трёхуровневая: медиана (промахи) + w_v≥0.7/детектор здоровья (глухота) |
| [it-14-channel-health](iterations/it-14-channel-health.md) | 2026-09-04 | Детектор здоровья канала | ✅ | Спасает при глухоте (0.539→0.827), вредит при тишине — «тишину≠глухоту по классовому выходу не отличить»; не внедрять в текущем виде |
| [it-15-p-drone-probability](iterations/it-15-p-drone-probability.md) | 2026-09-04 | p(drone) в AudioDetection (AST) | ✅ | Информация «насколько не дрон» больше не теряется; 3/3 теста; проброс в контракт — отдельное решение |
| [it-16-health-v2-energy](iterations/it-16-health-v2-energy.md) | 2026-09-04 | Health v2: энергия входа + std выхода | ✅ | «Тишина ≠ глухота» решена (RMS 0.005 vs 0.070); healthy без потерь 0.913, full_deaf 0.902, deaf_last50 0.840; intermittent — честный предел |
| [it-17-mmaud-plan](iterations/it-17-mmaud-plan.md) | 2026-09-04 | MMAUD: разбор репо и план адаптера | ✅ | ROSBag, V1 5×11–20 ГБ (OneDrive), CC BY-NC-SA; план: V1 Mavic3 → tools/mmaud_extract.py → ablation; AV-FDTI как референс |
| [it-19-quality-hint-health-gate](iterations/it-19-quality-hint-health-gate.md) | 2026-09-04 | audio_rms в QualityHint + ChannelHealthGate в fusion | ✅ | Контракт+акустика+fusion, 23/23 теста; 2 дефекта дизайна найдены тестами (mean_pa, rms_abs_floor); off по умолчанию |
| [it-20-etap6-corrections](iterations/it-20-etap6-corrections.md) | 2026-09-04 | Поправки в Vault «Этап 6» | ✅ | Переинтерпретация №4 (двухслойный GT), entropy-gating опровергнут, GT-вопрос закрыт, новые открытые пункты |
| [it-21-otchet-section7](iterations/it-21-otchet-section7.md) | 2026-09-04 | Черновик раздела 7 отчёта НИР-2 | ✅ | otchet.md: заглушка → полный текст (7.1–7.8) + источники [37]–[39]; числа только из артефактов |
| [it-22-pandoc-render](iterations/it-22-pandoc-render.md) | 2026-09-04 | Pandoc-рендер отчёта | ✅ | pandoc 3.9 (pypandoc-binary): docx собрался, 8 таблиц, 10/10 контент-проверок раздела 7; docx в git не входит |
| [it-23-citation-verification](iterations/it-23-citation-verification.md) | 2026-09-04 | Верификация источников [37]–[39] | ✅ | Все три реальны; библиография уточнена (DOI 10.1109/ACCESS.2020.3026192, Interspeech 2021, 180320 клипов DADS) |
| [it-24-citation-verification-2](iterations/it-24-citation-verification-2.md) | 2026-09-04 | Верификация библиографии [26]–[36] | ✅ | 3 уточнения: ByteTrack → ECCV 2022, DUT Anti-UAV → IEEE T-ITS 23(12), Al-Emadi IWCMC 2019 (не 2021); библиография отчёта вся веб-верифицирована |
| [it-25-yolo11-26-vs-v8](iterations/it-25-yolo11-26-vs-v8.md) | 2026-09-04 | YOLO11/YOLO26 vs YOLOv8 | ✅ | База остаётся v8; YOLO11n — дешёвый сравнительный эксперимент (+2.2 mAP nano, C2PSA); YOLO26 отложен — NMS-free конфликтует с ByteTrack |
| [it-26-clean-consistency](iterations/it-26-clean-consistency.md) | 2026-09-04 | Консистентность otchet.md ↔ clean ↔ docx | ✅ | clean перегенерирован (раздел 7 + [37]–[39] на месте, маркеров нет), оба docx собраны, README-статус обновлён |
| [it-27-gost-table-captions](iterations/it-27-gost-table-captions.md) | 2026-09-04 | ГОСТ-подписи таблиц раздела 7 | ✅ | Подписи 7.1–7.4 вставлены; найден дефект рендера (подпись вплотную роняет таблицу из docx) — исправлено пустой строкой, 8/8 таблиц + 4/4 подписи |
| [it-28-gost-captions-2](iterations/it-28-gost-captions-2.md) | 2026-09-04 | Подписи таблиц 1.1/4.1/4.2/5.1 | ✅ | Смещение подписи на строку роняло таблицы из docx — поймано подсчётом <w:tbl>, исправлено; итог 8/8 таблиц + 8/8 подписей |
| [it-18-p-drone-contract](iterations/it-18-p-drone-contract.md) | 2026-09-04 | p_drone в контракте InferenceMsg | ✅ | Аддитивно, schema_ver=1 сохранён; lwcnn+AST отдают softmax; fusion использует; 22/22 теста |
| [it-18-p-drone-contract](iterations/it-18-p-drone-contract.md) | 2026-09-04 | p_drone в контракте InferenceMsg | ✅ | Аддитивно, schema_ver=1; lwcnn+AST отдают softmax; 22/22 теста |
| [it-29-citations-1-25](iterations/it-29-citations-1-25.md) | 2026-09-04 | Верификация источников 1–25 | 🔄 13/25 | TRIDENT/Aydın/Dafrallah/Jamil/Dumitrescu/Linn/Liu Z/Ren/AV-DTEC подтверждены; 12 остались (поиск таймаутит) |
| it-05-ablation-audio | — | Разбор: куда исчезает аудио в №4 (7% vs 75% смешанных окон) | 📋 план | аномалия подтверждена и в журнале (161/2394), нужна instrumentation fusion (run_id, счётчики окон) |

## Главные результаты цикла (на 2026-09-04)
1. Первый честный GT-скоринг пайплайна: лучший для «БПЛА активен» — **late+медиана p_a F1=0.922**; для «объект присутствует» — video-only F1=1.000.
2. Опровергнуто данными: entropy-gating по выходу; «fusion средним создаёт информацию» (на неинформативной модальности не создаёт).
3. Дёшево внедряемо: медианная фильтрация p_a в fusion-слое (+6.3 п.п. F1), w_v>0.5 при деградации аудио (+0.35 F1 на глухой акустике).

## Артефакты
- `iterations/` — журналы итераций
- `sandbox_frames/` — кадры sandbox-видео для разметки (в git не грузятся, см. .gitignore)
- `gt_sandbox_video.csv` — ground truth (сек → visible/airborne)
- `score_fusion_vs_gt.py` — скоринг решений и симуляция политик
- `.venv/` — окружение для YOLO-инференса (в git не грузится)
