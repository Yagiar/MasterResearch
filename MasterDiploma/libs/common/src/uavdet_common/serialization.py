"""JSON-сериализатор сообщений (pydantic <-> bytes) + хелперы base64 для бинарных payload.

На пилоте используется JSON (`schema_ver: 1`); в дизайн заложена возможность Avro/Protobuf + Schema Registry.
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel


class JsonSerializer:
    """Кодирует pydantic-модель в JSON-bytes и обратно."""

    def encode(self, obj: Any) -> bytes:
        if isinstance(obj, BaseModel):
            return obj.model_dump_json().encode("utf-8")
        if isinstance(obj, (bytes, bytearray)):
            return bytes(obj)
        # dict / список и т.п.
        import json

        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def decode(self, raw: bytes, model: type | None = None) -> Any:
        if model is not None and issubclass(model, BaseModel):
            return model.model_validate_json(raw)
        import json

        return json.loads(raw.decode("utf-8"))


def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode_str(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


__all__ = ["JsonSerializer", "b64encode_bytes", "b64decode_str"]
