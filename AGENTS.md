# AGENTS.md — GeneralFolderMasterDiploma

Рабочая папка студента ИТМО (Терешкин Д., П4114 / 4253): магистерская диссертация по мультимодальному обнаружению БПЛА (видео YOLOv8+ByteTrack + аудио MFCC+легкая CNN, fusion) + материалы бакалаврской ВКР. Всё содержание и документы — **на русском**; отчёты — по ГОСТ 7.32-2017.

**Язык работы: продолжать на русском** — все ответы, пояснения, вопросы пользователю, комментарии в коде и новые документы вести на русском (даже если запрос пришёл на английском).

## Структура (корень НЕ git-репозиторий; внутри — 3 отдельных репо + 1 папка)

- `MasterDiploma/` — **основной код** (git, remote github.com/Yagiar/MasterDiploma): монорепо микросервисного Kafka-пайплайна + train-приложение. **Сначала читать `MasterDiploma/CLAUDE.md` и `MasterDiploma/README.md`** — там runbook и архитектура.
- `MasterDiplomaVaultObsidian/` — Obsidian-вольт, база знаний (git): заметки по главам, литкарточки, архитектура НИР-2. Конвенции заметок обязательны: YAML-frontmatter (`type`, `tags`, `source`, `status`, `aliases`, `related`), wikilinks `[[...]]`, `#теги`; номера источников `[N]` = нумерация Приложения 1 отчёта НИР.
- `NIR-2-SEM-Full/` — исходники графиков/диаграмм отчёта НИР-2 (git). **Читать `HANDOFF.md` перед работой в этой папке** (план работ + правила). `data/*.csv` — единственный источник чисел для графиков; `scripts/render_charts.py` пересобирает PNG/SVG в `charts_png/`/`charts_svg/`.
- `.agents/skills/` (в корне workspace) — 24 навыка (yolo*, literature-review*, paper-review, tufte-viz, lab-notes и др.); перенесены сюда из `NIR-2-SEM-Full/.agents/skills/` 2026-09-04 — workspace-скиллы ZCode сканирует только от корня workspace вверх, из дочерних папок они не видны.
- `Диплом-бака/` — документы бакалаврской ВКР (docx/pdf/pptx) + standalone OpenCV-скрипты (`server_bpla.py`, `02-04-2025-cams-sync.py`, `24-03-2025-diploma-video.py`). Не git-репо. Файлы `~$...` — временные lock-файлы MS Office, не трогать.

## Команды (в `MasterDiploma/`)

```bash
pip install -e 'libs/proto[dev]' -e libs/common && make proto-gen  # перед первым запуском: gRPC-стабы
make install         # pip install -e всех пакетов
make train-install   # train-приложение отдельно (тяжелые: torch/ultralytics/librosa)
make test            # pytest (libs/common/tests, services)
make lint            # ruff check (line-length 100, py310; *_pb2* исключены; E501 игнорируется)
make infra-up && make topics-create && make db-migrate   # Kafka (KRaft) + PostgreSQL + Liquibase
make run-pipeline[-mm|-gpu] / run-dashboard / pipeline-logs / pipeline-down / infra-down
```

`NIR-2-SEM-Full/`: `python scripts/render_charts.py` — пересборка графиков из `data/*.csv`.

## Архитектурные границы (MasterDiploma)

- Поток: `source-simulator` → (gRPC) → `ingest-gateway` → топики `video.raw`/`audio.raw` → `visual-detector` | `acoustic-detector` → `inference` → `fusion` (late/hybrid + adaptive gating) → `decisions` → `sink` (Prometheus + jsonl + PostgreSQL схема `uavdet`). `gateway` — dashboard-BFF (REST/WS, профиль `dashboard`).
- `libs/common` (uavdet-common) — общие pydantic-схемы сообщений, Kafka-обертка, базовый consumer; менять контракты сообщений только здесь (JSON, key=`source_id`, `schema_ver: 1`).
- `libs/proto` — gRPC-контракт `ingest.proto`; после правки всегда `make proto-gen` (стабы коммитятся, импорт в `*_pb2_grpc.py` чинится sed'ом на относительный).
- `train/` — отдельное приложение обучения (`python -m uavtrain.cli`: download → prepare → train → eval → export); веса пишутся в `models/`.

## Известные ловушки

- **Пути в документах устарели**: `MasterDiploma/CLAUDE.md`, `README.md` и `NIR-2-SEM-Full/HANDOFF.md` ссылаются на macOS-пути (`/Users/otrix/...`). Фактически вольт лежит рядом: `../MasterDiplomaVaultObsidian/`.
- В `MasterDiploma` **не коммитить**: `.docx` отчёта (генерируется `pandoc reports/НИР-2/otchet.md -o otchet.docx` из markdown), веса `models/`, медиа `sandboxDataForSimulator/`.
- На CPU `visual-detector` не держит реалтайм — лаг копится; для реальных прогонов `make run-pipeline-gpu` (нужен nvidia-container-toolkit). Реального Docker-прогона пайплайна до недавнего времени не было — проверяйте фактическое состояние.
- `models/visual/` без весов: `visual-detector` требует `yolov8n.pt`; `acoustic-detector` без весов работает в режиме энергетического порога.
- Git-коммиты — фактические, **без сторонних соавторов** (без Co-Authored-By).
- Дисциплина цитирования в текстах ВКР/НИР: каждое фактическое утверждение — с проверенным источником (DOI/arXiv), ссылки не выдумывать.
