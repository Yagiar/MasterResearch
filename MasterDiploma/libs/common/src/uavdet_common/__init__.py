"""uavdet-common — общая библиотека для сервисов системы обнаружения БПЛА (НИР-2).

Содержит:
- abstractions  — интерфейсы (DataSource, Detector, FusionStrategy, GatingPolicy, DegradationStrategy, MessageBus, Serializer, MetricsSink);
- messages      — pydantic-схемы сообщений Kafka-топиков (VideoRawMsg, AudioRawMsg, InferenceMsg, DecisionMsg, ConfigCommandMsg) + имена топиков;
- serialization — JSON-сериализатор (pydantic <-> bytes);
- bus           — KafkaMessageBus (confluent-kafka) и InMemoryBus (для тестов);
- consumer_service — KafkaConsumerService (Template Method) с graceful shutdown и идемпотентностью;
- config        — базовые настройки (pydantic-settings) + загрузка YAML-профиля;
- metrics       — Prometheus-метрики + HTTP /metrics;
- logging       — настройка structlog.
"""

__version__ = "0.1.0"
