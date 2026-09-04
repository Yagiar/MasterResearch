# uavdet-proto — gRPC-контракт `source-simulator` ↔ `ingest-gateway`

Определяет канал передачи сырых кадров/аудио-окон от источника данных (имитатор `source-simulator` или, в будущем, `sensor-driver` с реальными камерой/микрофоном) к приёмнику `ingest-gateway`, который нормализует поток и публикует его в Kafka (`video.raw` / `audio.raw`).

## Файлы
- `src/uavdet_proto/ingest.proto` — определение: сервис `SourceStream` (`StreamVideo(stream Frame) -> Ack`, `StreamAudio(stream AudioWindow) -> Ack`), сообщения `Frame`, `AudioWindow`, `Ack`.
- `src/uavdet_proto/ingest_pb2.py`, `ingest_pb2_grpc.py` — **генерируются** из `.proto` (в git не коммитятся).

## Генерация стабов
```bash
pip install -e 'libs/proto[dev]'   # тянет grpcio-tools
make proto-gen                     # из корня репо
```
(внутри: `python -m grpc_tools.protoc -I<src> --python_out=<src> --grpc_python_out=<src> <src>/ingest.proto` + правка импорта `pb2` на относительный).

## Использование
```python
from uavdet_proto import ingest_pb2, ingest_pb2_grpc
# клиент (source-simulator): stub = ingest_pb2_grpc.SourceStreamStub(channel); stub.StreamVideo(frame_iter)
# сервер (ingest-gateway): добавить SourceStreamServicer в grpc.server(...)
```
