# ИТЕРАЦИЯ 09 — Веб-ресёрч №2: recall AST, «жёсткие негативы», практический доступ к MMAUD

- **Дата:** 2026-09-04
- **Метод:** веб-поиск, 3 направления; ссылки проверены существованием на дату.

## 1. Как поднять recall акустики (у AST R=0.766 — it-05)

| Приём | Источник | Что даёт нам |
|---|---|---|
| Порог по PR-кривой валидации (авто-калибровка, не 0.5) | [SED weak labels, auto-threshold](https://arxiv.org/pdf/1912.04761); [threshold-independent SED eval](https://inria.hal.science/hal-03562763v1/document) | дёшево: пересчитать порог на валидации под recall — у нас P=0.977 при R=0.766, есть запас precision на trade |
| Цепочки аугментаций под OOM-домен (AuDroK, mean TPR ↑) | [Improving acoustic drone detection generalization](https://arxiv.org/html/2605.31329v1) | прямое подтверждение нашего подхода (codec round-trip и пр.); добавить в fine-tune списка кандидатов |
| **PETL-адаптеры вместо full fine-tune** (мало данных → overfit) | [Parameter-Efficient Transfer Learning of AST](https://arxiv.org/html/2312.03694v3), [PETL_AST](https://github.com/umbertocappellazzo/PETL_AST) | дообучить AST на наших «полевых» записях с адаптерами — направление для диссертации |
| Мелкие данные + CNN/трансформеры для UAV-аудио | [Small Data Training for UAV Audio](https://arxiv.org/html/2505.23782v1) | референс постановки |

**Связка с нашими результатами:** it-08 (медиана-5, F1 0.859→0.913 без переобучения) и калибровка порога — ортогональны и дёшевы; PETL + конфьюзеры — тяжёлое, на диссертацию.

## 2. «Жёсткие негативы» для акустики (закрытие P0.3 аудита)

- **AudioSet**: классы Aircraft → Helicopter/Propeller/Aircraft engine ([онтология](https://research.google.com/audioset/ontology/aircraft_1.html)); аудио удобнее брать зеркалом [agkphysics/AudioSet (HF)](https://huggingface.co/datasets/agkphysics/AudioSet) (10-сек клипы с онтологией) или тулзой [IvanBirkmaier/Audioset](https://github.com/IvanBirkmaier/Audioset).
- **AeroSonicDB** ([Kaggle](https://www.kaggle.com/datasets/gray8ed/audio-dataset-of-low-flying-aircraft-aerosonicdb)): 12.4 ч маловысотных самолётов, метаданные (тип двигателя, пропеллер) — готовый набор конфьюзеров.
- **AeroSonic: YPAD-0523** ([Zenodo](https://zenodo.org/records/8004081)): клипы с **ADS-B-верификацией** (гарантированно самолёты, не дроны) — самые чистые негативы.

План эксперимента (диссертация): retrain lwcnn/resnet18 на «дрон vs {ESC-50}» vs «дрон vs {ESC-50 + aircraft-конфьюзеры}» vs «дрон vs {aircraft}» — ожидаемая деградация и есть вклад в знание.

## 3. MMAUD — практический доступ

- Репо: [github.com/ntu-aris/MMAUD](https://github.com/ntu-aris/MMAUD); сенсоры: **2 синхронные камеры + mmWave-радар + 4 аудио-массива** (+LiDAR); >1700 с, 6 типов дронов (Mavic2/3, Avata, Phantom4...). Файлы хостятся внешне — ссылки в README репо ([paper](https://arxiv.org/html/2402.03706v1)).
- Для адаптера `mmaud_replay` нужно: скачать 1 короткую sequence, понять формат (розбагрить README), сшить аудио-канал(ы) с кадрами по времени; базовое сравнение — AV-FDTI (Machine Learning with Applications) на том же датасете.
- Риск: 4-канальный аудиомассив — наш пайплайн моно; брать downmix первого канала (как делает `_pcm_int16_to_float`).

## 4. Обновление приоритетов бэклога

1. **Калибровка порога AST на валидации** (часы, без переобучения) + **медиана-5** (it-08, патч готов) — быстрый выигрыш recall/F1.
2. Конфьюзеры из AeroSonicDB/AudioSet в train (GPU-день).
3. MMAUD: скачать 1 sequence, набросать адаптер (день).
4. PETL-дообучение AST — на диссертацию.
