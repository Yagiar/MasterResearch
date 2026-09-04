"""FastAPI-приложение dashboard-bff.

Жизненный цикл (lifespan): на старте — поднимает `DecisionStore`, фоновый Kafka-поток
(`DecisionsFeedConsumer` на топике `decisions`) и WS-broadcaster; на остановке — гасит их.
Эндпоинты:
  GET  /healthz                 — health-проба;
  GET  /status                  — состояние сервиса (подключён ли consumer, размер буфера);
  GET  /stats                   — агрегаты по решениям (total, drone_ratio, avg p_fused/e2e, by_mode);
  GET  /decisions?limit=N       — последние N решений (новые сверху);
  WS   /ws/decisions            — стрим новых решений в реальном времени;
  GET  /                        — статический дашборд (static/index.html).

Конфиг — через uavdet_common.config (секции `kafka`, `gateway`).
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger

from .consumer import DecisionsFeedConsumer
from .state import DecisionStore

API_VERSION = "0.1.0"
_STATIC_DIR = Path(__file__).parent / "static"


class _WsHub:
    """Простой набор активных WebSocket-подключений + рассылка из любого потока."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)

    def broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        """Вызывается из потока consumer'а — планирует рассылку в event loop приложения."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def create_app() -> FastAPI:
    cfg = load_config(None)
    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("gateway")

    kafka_cfg = dict(cfg.get("kafka", {}))
    gw_cfg = dict(cfg.get("gateway", {}))

    store = DecisionStore(capacity=int(gw_cfg.get("buffer_size", 500)))
    hub = _WsHub()
    state: dict[str, Any] = {"consumer": None, "thread": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.bind_loop(asyncio.get_running_loop())
        bus = KafkaMessageBus(
            bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "kafka:9092")),
            client_id="gateway",
            auto_offset_reset=str(gw_cfg.get("auto_offset_reset", "latest")),
        )
        consumer = DecisionsFeedConsumer(
            bus,
            group_id=str(gw_cfg.get("group_id", "gateway-dashboard")),
            store=store,
            on_decision=hub.broadcast_threadsafe,
        )
        t = threading.Thread(target=consumer.run, name="gateway-decisions", daemon=True)
        t.start()
        state["consumer"], state["thread"] = consumer, t
        log.info("gateway: dashboard-bff запущен", buffer=store.capacity)
        try:
            yield
        finally:
            if state["consumer"] is not None:
                state["consumer"]._stop = True  # noqa: SLF001 - кооперативная остановка
            if state["thread"] is not None:
                state["thread"].join(timeout=10.0)
            log.info("gateway: остановлен")

    app = FastAPI(title="uavdet dashboard-bff", version=API_VERSION, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": API_VERSION}

    @app.get("/status")
    def status() -> dict[str, Any]:
        t = state.get("thread")
        return {
            "version": API_VERSION,
            "consumer_alive": bool(t.is_alive()) if t is not None else False,
            "buffer_size": store.capacity,
            "topic": "decisions",
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return store.stats()

    @app.get("/decisions")
    def recent_decisions(limit: int = 20) -> dict[str, Any]:
        return {"items": store.recent(limit=limit), "limit": limit}

    @app.websocket("/ws/decisions")
    async def ws_decisions(ws: WebSocket) -> None:
        await hub.connect(ws)
        # отправим последние решения сразу при подключении
        for item in reversed(store.recent(limit=20)):
            await ws.send_json(item)
        try:
            while True:
                await ws.receive_text()  # клиент ничего не шлёт; держим соединение
        except WebSocketDisconnect:
            hub.disconnect(ws)
        except Exception:  # noqa: BLE001
            hub.disconnect(ws)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

    return app


app = create_app()
