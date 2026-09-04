"""FrameDecoder — base64 JPEG (payload VideoRawMsg) -> numpy-изображение BGR + оценка качества кадра."""

from __future__ import annotations

import numpy as np

from uavdet_common.serialization import b64decode_str


class FrameDecoder:
    """Декодер кадров из payload сообщения video.raw."""

    def __init__(self) -> None:
        # cv2 импортируем здесь, чтобы пакет можно было импортировать без OpenCV в тестах схем
        import cv2  # noqa: PLC0415

        self._cv2 = cv2

    def decode_jpeg_b64(self, payload_b64: str) -> np.ndarray:
        """Декодировать base64 JPEG в массив HxWx3 (BGR)."""
        raw = b64decode_str(payload_b64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("не удалось декодировать JPEG из payload")
        return img

    def quality_hint(self, img: np.ndarray) -> tuple[float, float]:
        """Дешёвая оценка качества кадра: (резкость = дисперсия лапласиана, средняя яркость 0..1)."""
        cv2 = self._cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean()) / 255.0
        return sharpness, brightness
