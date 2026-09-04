# fusion — движок мультимодального слияния и принятия решений

Consumer-сервис: топик `inference` → **временное выравнивание** детекций видео/аудио
в окне ±ε (per `source_id`) → **стратегия слияния** → публикация `DecisionMsg` в топик `decisions`.

## Место в пайплайне
```
inference  -->  [fusion]  -->  decisions  -->  sink / dashboard
```

## Режимы слияния (`fusion.mode`)
| Режим | Что |
|---|---|
| `video-only` | решение по видеоканалу: `p_fused = p_v` (baseline; MVP-режим пайплайна) |
| `audio-only` | решение по акустическому каналу: `p_fused = p_a` (baseline; задействуется после акустической ветки) |
| `late` | позднее слияние: `p_fused = clip(w̃_v·p_v + w̃_a·p_a + Δ, 0, 1)`, где `w̃` — веса w_v/w_a, **перенормированные по активным каналам** (если в окне нет аудио → `w̃_v=1, w̃_a=0`, т.е. `p_fused=p_v` — вес не «теряется в пустоту»); Δ — правило компенсации (раздел 5.2 отчёта): `+δ_conf` при взаимном подтверждении обоих каналов, `−δ_unconf` при противоречии одного, `Δ=0` если активен один канал |
| `hybrid` | гибрид на уровне решений: адаптивные веса + при согласии каналов — нелинейный буст (вероятностное «ИЛИ» `1−(1−p_v)(1−p_a)` + `δ_conf`), при противоречии — `−δ_unconf`, при одном канале — решение по нему |

Веса каналов `w_v`/`w_a` даёт `GatingPolicy`:
- `FixedGating` — фиксированные веса из `fusion.weights` (нормируются к сумме 1); для `video-only`/`audio-only` стратегия сама выставляет 1/0;
- `AdaptiveGating` (`fusion.gating.enabled: true`) — веса как функция качества каналов в окне: резкость/яркость кадра (`InferenceMsg.quality.img_sharpness`/`img_brightness` — visual-detector считает дисперсию лапласиана и среднюю яркость) и SNR аудио (`quality.snr_db` — заполнит acoustic-detector); при отсутствии подсказок откатывается к базовым весам. Сводные `snr_audio`/`img_quality` пишутся в `DecisionMsg.gating`.

Эти режимы — основа ablation на этапе 6: `video-only` / `audio-only` (baseline) vs `late` vs `hybrid` (+adaptive gating).

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
fusion:
  group_id: fusion
  mode: video-only            # video-only | audio-only | late | hybrid
  window_epsilon_ms: 80
  decision_threshold: 0.5
  weights:                    # базовые веса (для late/hybrid и FixedGating)
    w_v: 0.5
    w_a: 0.5
  late:
    delta_conf: 0.1
    delta_unconf: 0.1
    # label_threshold: 0.5    # порог «канал сказал drone» (по умолчанию = decision_threshold)
  hybrid:
    delta_conf: 0.1
    delta_unconf: 0.15
  gating:
    enabled: false            # true -> AdaptiveGating (веса от качества кадра/SNR)
    q_floor: 0.2
    sharpness_ref: 150.0
    snr_ref_db: 20.0
    snr_floor_db: 0.0
  auto_offset_reset: latest
  metrics_port: 0             # >0 -> /metrics
```

## Запуск
```bash
make run-pipeline     # в составе MVP-пайплайна (Docker)
make run-fusion       # отдельно (Docker)

# локально (нужен поднятый Kafka):
pip install -e libs/common -e services/fusion
python -m fusion --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `consumer.py` | `InferenceConsumer(KafkaConsumerService)` — оркестрация |
| `window_buffer.py` | `TimeWindowBuffer` — выравнивание детекций в окне ±ε (per `source_id`) |
| `strategies/video_only.py`, `audio_only.py` | `VideoOnly` (MVP-режим) / `AudioOnly` — baseline'ы |
| `strategies/late.py` | `LateFusion` — позднее слияние с правилом компенсации Δ |
| `strategies/hybrid.py` | `HybridFusion` — гибрид на уровне решений (нелинейный + адаптивные веса) |
| `gating.py` | `FixedGating` / `AdaptiveGating` (веса от качества кадра/SNR) → `GatingResult(w_v, w_a, snr_audio, img_quality)` |
| `factory.py` | `FusionStrategyFactory` + сборка `GatingPolicy` по конфигу |
| `__main__.py` | точка входа (`python -m fusion`) |

## Проверка
```bash
docker compose -f infra/docker-compose.yml exec kafka \
  /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic decisions --from-beginning --max-messages 5
```
