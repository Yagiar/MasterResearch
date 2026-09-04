"""uavdet-proto — gRPC-контракт source-simulator ↔ ingest-gateway.

Стабы ingest_pb2.py / ingest_pb2_grpc.py генерируются из ingest.proto
командой `make proto-gen` (в git не коммитятся, см. .gitignore).
"""

__all__ = ["ingest_pb2", "ingest_pb2_grpc"]
