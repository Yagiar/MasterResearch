# Исходники графиков и диаграмм для отчета НИР

Содержимое архива собрано для ручной правки фигур из приложения 1 исправленного отчета.

## Что где лежит

- `drawio/all_figures.drawio` — один файл diagrams.net/draw.io со всеми 6 фигурами на отдельных страницах.
- `drawio/01_...drawio` ... `drawio/06_...drawio` — те же фигуры отдельными файлами.
- `png_from_docx/` — PNG, которые реально вставлены в DOCX, с понятными именами.
- `docx_media_raw/` — все PNG, лежащие внутри DOCX-пакета, включая старые/неиспользованные варианты.
- `data/` — CSV/JSON с исходными числами для графиков.
- `scripts/render_charts.py` — скрипт, который пересобирает графики из `data/*.csv` в PNG и SVG.
- `charts_png/`, `charts_svg/` — уже сгенерированные версии графиков.
- `mermaid/` — текстовые исходники диаграмм, удобные для быстрой правки в Markdown/diagrams.net.
- `plantuml/` — PlantUML-исходники для UML-диаграмм 2–4.

## Соответствие файлам в отчете

1. Рисунок 1 — архитектура системы: `drawio/01_01_architecture.drawio`, `png_from_docx/01_architecture_from_docx.png`, `mermaid/01_architecture.mmd`.
2. Рисунок 2 — UML-диаграмма компонентов: `drawio/02_02_uml_component.drawio`, `png_from_docx/02_uml_component_from_docx.png`, `mermaid/02_component.mmd`, `plantuml/02_component_uml.puml`.
3. Рисунок 3 — UML-диаграмма последовательности: `drawio/03_03_uml_sequence.drawio`, `png_from_docx/03_uml_sequence_from_docx.png`, `mermaid/03_sequence.mmd`, `plantuml/03_sequence_uml.puml`.
4. Рисунок 4 — UML-диаграмма деятельности: `drawio/04_04_uml_activity.drawio`, `png_from_docx/04_uml_activity_from_docx.png`, `mermaid/04_activity.mmd`, `plantuml/04_activity_uml.puml`.
5. Рисунок 5 — метрики YOLOv8n: `drawio/05_05_yolov8_metrics_chart.drawio`, `data/yolov8_metrics.csv`, `scripts/render_charts.py`.
6. Рисунок 6 — метрики акустического детектора: `drawio/06_06_acoustic_metrics_chart.drawio`, `data/acoustic_metrics.csv`, `scripts/render_charts.py`.

## Как править

Самый простой вариант: открыть `drawio/all_figures.drawio` в diagrams.net, внести правки, затем File → Export as → PNG/SVG и заменить картинку в Word.

Для графиков лучше править числа в `data/*.csv` и запускать:

```bash
python scripts/render_charts.py
```

После этого обновленные картинки появятся в `charts_png/` и `charts_svg/`.

Собрано: 2026-06-04 11:17.


## Исправление draw.io

В этой версии исправлены `.drawio`-файлы: страницы сохранены как полноценный uncompressed XML с реальными узлами `<mxGraphModel>`, а не как текст внутри `<diagram>`. Поэтому diagrams.net не должен вызывать `atob()` для строк с кириллицей.

Открывать в первую очередь: `drawio/all_figures.drawio`. Если на старой версии diagrams.net снова появится ошибка импорта, используйте резервные ASCII-версии из `drawio_compressed_ascii/` — там содержимое страниц сжато и закодировано в base64.
