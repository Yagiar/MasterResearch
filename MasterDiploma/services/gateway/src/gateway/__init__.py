"""gateway — dashboard backend-for-frontend.

FastAPI-сервис: фоновый Kafka-потребитель топика `decisions` -> кольцевой буфер
последних N решений + агрегаты (число решений, доля положительных, средние p_fused
и e2e-latency, разбивка по режимам fusion). Отдаёт REST (`/decisions`, `/status`,
`/stats`) и WebSocket `/ws/decisions` (стрим новых решений в реальном времени),
плюс минимальный статический дашборд (`/` -> `static/index.html`).

На пилоте может быть слит с `sink`; вынесен в отдельный сервис для будущего фронтенда.
В docker-compose — под профилем `dashboard`.
"""
