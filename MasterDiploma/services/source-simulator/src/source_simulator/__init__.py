"""source-simulator — имитатор источников данных.

gRPC-клиент: читает датасет через DataSourceAdapter, проигрывает поток
покадрово/по аудио-окнам с реальным FPS, (опц.) пропускает через
DegradationChannel и стримит в ingest-gateway по контракту uavdet_proto.ingest.

Взаимозаменяем с будущим sensor-driver (реальные сенсоры) — оба реализуют
один gRPC-клиентский контракт.
"""
