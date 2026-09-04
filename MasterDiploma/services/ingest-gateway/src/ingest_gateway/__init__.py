"""ingest-gateway — тонкий приёмник потоков.

gRPC-сервер: принимает стрим `Frame`/`AudioWindow` от source-simulator (или, в
будущем, от sensor-driver — взаимозаменяемых клиентов), нормализует
(`msg_id`/`ingest_ts`/`schema_ver`, валидация) и публикует `VideoRawMsg`/`AudioRawMsg`
в Kafka-топики `video.raw` / `audio.raw` (ключ — `source_id`).
"""
