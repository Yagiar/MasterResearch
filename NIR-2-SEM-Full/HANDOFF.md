# HANDOFF — установленные скиллы и план работ (из сессии установки скиллов)

> Прочитай этот файл полностью перед началом работы. Он передаёт контекст: какие скиллы
> установлены в проект, что с ними нужно сделать, и как перепланировать уже выполненную
> работу (аудит НИР-пакета из этой сессии) под новые скиллы.
>
> Порядок работы: **строго step by step**, перед каждым шагом проверять список доступных
> скиллов и загружать/применять релевантные. После каждого шага — короткий статус.

## 0. Контекст проекта

- Workspace: `/Users/otrix/NIR-2-SEM-Full` — НИР-2 (ИТМО, Терешкин Дмитрий, P4114).
- Тема: мультимодальное обнаружение БПЛА (YOLOv8n + ByteTrack по видео, лёгкая CNN/AST по аудио, late/hybrid fusion, Kafka-пайплайн).
- Аудит пакета уже выполнен (7 шагов, вердикт: готов к сдаче, критических проблем нет). Известные дефекты — см. раздел 4.

## 1. Что установлено: 24 скилла (быв. `.agents/skills/` этого репо)

> **Обновление 2026-09-04:** скиллы перенесены из `NIR-2-SEM-Full/.agents/skills/` в корень workspace `GeneralFolderMasterDiploma/.agents/skills/` — клиент ZCode сканирует workspace-скиллы только от корня workspace вверх, из дочерней папки они не обнаруживались. Папка `.agents/` из этого репо удалена.

Все установлены из GitHub, frontmatter валидирован (name = имени папки, description на месте).
Источник указан в скобках. Группировка по назначению:

**Литературный поиск и обзор (4):**
- `literature-review` — pinshuai/literature-review-skill, самый полный: `scripts/verify_citations.py`, `search_databases.py`, `assets/review_template.md`. В начало SKILL.md добавлена адаптационная заметка: hardcoded путь автора к `paper-search` не работает, искать через WebSearch/WebFetch и API Semantic Scholar / Crossref / OpenAlex / arXiv.
- `literature-review-evidence` — eresuntomatito/literature-review-codex-plugin: закрытый evidence-based протокол (source cards, citation ledger, матрицы: synthesis_matrix.csv, citation_ledger.csv и др.).
- `literature-review-omnitric` — omnitric/agent-skills: форк pinshuai (различия реальные, ~504 строки диффа).
- `literature-review-zh` — taoyunudt/literature-review-skill: гайд на китайском.

**Написание и рецензирование статьи (6):**
- `academic-paper` — omnitric/agent-skills: 12-агентный пайплайн, 10 режимов (full/plan/outline/revision/abstract/lit-review/citation-check и др.), 114 файлов.
- `paper-review`, `referee2`, `backmanreview`, `openaireview`, `code-review` — lcrawfurd/claude-skills. Исходно это были слэш-команды без frontmatter — сконвертированы в полноценные скиллы (добавлены name/description).

**Исследовательский цикл и эксперименты (5):**
- `ai-research` — Toadoum/ai-research-skill: полный цикл hypothesis → baseline → experiments → writeup.
- `experiment-agent` — Imbad0202/experiment-agent: запуск/мониторинг экспериментов (22 файла).
- `lab-notes` — eins78/agent-skills (skills/lab-notes): журнал экспериментов, hypothesis-first.
- `machine-learning` — itallstartedwithaidea/agent-skills (skills/scientific-research/machine-learning).
- `model-training` — h4vzz/awesome-ai-agent-skills (ai-ml-operations/model-training).

**YOLO / CV (8):**
- `yolo`, `yolo-datasets`, `yolo-export`, `yolo-inference`, `yolo-models`, `yolo-training`, `yolo-tuning` — официальные ultralytics/skills.
- `yolo-master-agent` — Tencent/YOLO-Master (папка `agent/`, sparse-clone): 87 файлов, включает runtime-код на Python и большие таксономии (LVIS 1203 классов, V3Det 13204). Привязан к своему репо.

**Визуализация (1):**
- `tufte-viz` — lcrawfurd/claude-skills: генерация/критика фигур по Тафти.

**Не установилось (2 из 14 репо):** `christophacham/agent-skills-library` — репо удалено с GitHub (404); `Epsilon617/Codex-Academic-Skills` — каталог-ссылка без самих скиллов.

**Примечание:** `.agents/` пока не закоммичен в git (untracked). Решение о коммите — за пользователем.

## 2. Задача 1 — чистка скиллов: оставить только важное

Принцип: в каждом скилле обязаны остаться `SKILL.md` (+ файлы, на которые он прямо ссылается: `references/`, рабочие `scripts/`, нужные шаблоны). Всё остальное — примеры, дубли, тяжёлые ассеты, runtime — удалить. Конкретно:

1. `literature-review` — удалить `examples/` (wildfire-streamflow с PDF; это чужой пример, не по теме). Оставить SKILL.md, `references/`, `scripts/`, `assets/`.
2. `yolo-master-agent` — привязан к репо YOLO-Master, runtime здесь не заработает. Удалить скилл целиком ИЛИ оставить только SKILL.md + `references/`, удалив `runtime/`, `scripts/`, `assets/` (таксономии весят больше всего). Рекомендация: удалить целиком — 7 официальных ultralytics-скиллов уже покрывают YOLO.
3. `academic-paper` — 114 файлов: удалить `examples/`, тяжёлые `agents/`-заготовки, если SKILL.md не требует их поимённо; проверить ссылки из SKILL.md перед удалением `shared/`, `templates/`, `references/`.
4. `experiment-agent` — удалить `docs/`, `CHANGELOG.md`, `README.zh-TW.md`; оставить SKILL.md, `references/`, `templates/`, `agents/` (если ссылается).
5. `literature-review-omnitric` — дублирует `literature-review` по назначению. Рекомендация: удалить (оставить pinshuai как основной). Окончательное решение — за пользователем, можно спросить в конце шага.
6. `literature-review-zh` — гайд на китайском, для русскоязычной ВКР малополезен. Рекомендация: удалить (или оставить — уточнить у пользователя там же).
7. `literature-review-evidence` — оставить SKILL.md + `assets/` (промпты, шаблоны, матрицы — реально полезны) + `scripts/` (на них есть ссылки). Ничего не удалять без проверки ссылок.
8. После чистки: повторная валидация всех SKILL.md (frontmatter, name = имя папки, description непустой) и `du -sh` до/после. Скиллы из 1-файла (paper-review, referee2, backmanreview, openaireview, tufte-viz, code-review, machine-learning, model-training, ultralytics-набор, lab-notes, ai-research) уже минимальны — не трогать.

## 3. Задача 2 — инициализировать AGENTS.md под скиллы

Создать workspace `AGENTS.md` (в корне репо, рядом с этим файлом) со секциями:

1. **Проект**: краткое описание НИР-2 (тема, автор, что где лежит: отчёт DOCX/PDF, `data/*.csv` + `metrics_source.json` — единственный источник чисел, `scripts/render_charts.py` — пересборка графиков, `drawio/` — исходники фигур).
2. **Скиллы проекта**: карта «задача → скилл» после чистки (литобзор → `literature-review`/`literature-review-evidence`; проверка цитирований → `literature-review` + verify_citations; рецензирование → `paper-review`/`referee2`/`backmanreview`; YOLO → `yolo*`; трекинг экспериментов → `lab-notes`/`experiment-agent`; планирование исследования → `ai-research`; фигуры → `tufte-viz`).
3. **Правила работы со скиллами**: перед каждым шагом проверять релевантные скиллы; вызов через `$skill-name` или авто-триггер; step by step, один шаг за раз.
4. **Дисциплина цитирования**: каждое фактическое утверждение в тексте ВКР — с проверенным источником (DOI/arXiv); не выдумывать ссылки.
5. **Бэклог дефектов** из аудита (раздел 4) — как текущий task list.

## 4. Задача 3 — реплан: что уже было → новые шаги под скиллы

Аудит пакета в этой сессии уже выполнен (вердикт: готов к сдаче). Известные дефекты, найденные аудитом, и продолжение работы — переупаковать в план, где каждому шагу назначен скилл:

- **Шаг A. Правка рис. 4** (в тексте 4.1.4 обещан финальный порог как decision node, в activity-диаграмме его нет): править `drawio/04_04_uml_activity.drawio` (добавить decision diamond) либо убрать фразу. Скилл: `tufte-viz` (принципы для диаграмм), ручная правка drawio.
- **Шаг B. Форматирование списка литературы** (п. 29 «29.McFee», п. 37 «37.Gong» — номер слит с текстом; п. 19 «lipbayeva L.» → «Ilipbayeva», «el al.» → «et al.»; п. 3 — уточнить по транскрипту аудита). Скилл: `literature-review` (verify_citations.py для проверки реальности источников) + `paper-review`.
- **Шаг C. Обновить/удалить `docx_media_raw/`** (устаревший снапшот от 4 июня, README больше не соответствует). Скилл: без специального, по правилам AGENTS.md.
- **Шаг D. Выравнивание DPI графиков** (charts_png 220dpi vs 180dpi в DOCX) — опционально, через `scripts/render_charts.py`.
- **Шаги E+ (продолжение к магистерской)**: citation snowballing по 38 источникам отчёта (скилл `literature-review` / `literature-review-evidence` → evidence matrix), планирование следующих экспериментов (entropy-based gating, missing modality, стресс-тесты, ΔF1 — скиллы `ai-research`, `experiment-agent`, `lab-notes`), метрики брать строго из `data/`.

Формат реплана: таблица/список шагов с колонками «шаг → скилл(ы) → артефакт на выходе → критерий готовности». Выполнение — по одному шагу за раз, с применением назначенных скиллов (задачи 4–5 пользователя).

## 5. Проверка скиллов

После чистки: перезапустить сессию/клиент, проверить обнаружение в Settings → Skills. Диагностика проблем обнаружения — скилл `zcode-guide:diagnosing-skills`, конфигурации — `zcode-guide:zcode-configuration-guide`.
