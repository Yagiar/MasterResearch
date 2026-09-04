"""GatingPolicy — определение весов модальностей w_v / w_a (паттерн Strategy).

- `FixedGating` — фиксированные веса из конфига (нормируются к сумме 1). На MVP-режиме
  `video-only` фактически не влияет (стратегия сама выставляет w_v=1, w_a=0).
- `AdaptiveGating` — веса как функция качества каналов в окне: резкость/яркость кадра
  для видео, SNR аудио для звука (подсказки `InferenceMsg.quality`). Если подсказок нет —
  откатывается к базовым весам. Используется для пилотных замеров hybrid-режима.

`weights(window)` возвращает `GatingResult(w_v, w_a, snr_audio, img_quality)`: веса +
сводные показатели качества (для записи в `DecisionMsg.gating`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .window_buffer import AlignedWindow


@dataclass(frozen=True)
class GatingResult:
    w_v: float
    w_a: float
    snr_audio: float | None = None
    img_quality: float | None = None


def _normalize(w_v: float, w_a: float) -> tuple[float, float]:
    s = w_v + w_a
    if s <= 0:
        return 0.5, 0.5
    return w_v / s, w_a / s


@dataclass(frozen=True)
class FixedGating:
    """Фиксированные веса w_v / w_a (нормируются к сумме 1)."""

    w_v: float = 0.5
    w_a: float = 0.5

    def weights(self, _window: AlignedWindow) -> GatingResult:
        wv, wa = _normalize(self.w_v, self.w_a)
        return GatingResult(w_v=wv, w_a=wa)


@dataclass(frozen=True)
class AdaptiveGating:
    """Адаптивные веса по качеству каналов.

    Идея: для каждой модальности считаем «фактор качества» q∈[q_min, 1] по подсказкам в окне,
    итоговый вес = base_weight · q (затем нормируем). Это снижает вклад деградированного
    канала (мутный/тёмный кадр, шумное аудио) без полного его отключения.

    Параметры:
      - base_w_v / base_w_a — базовые веса (при идеальном качестве обоих);
      - q_floor — нижняя граница фактора качества (чтобы канал не обнулялся полностью);
      - sharpness_ref — «эталонная» резкость кадра (дисперсия лапласиана), выше которой q_img=1;
      - snr_ref_db — «эталонный» SNR аудио (дБ), выше которого q_snr=1; snr_floor_db — ниже которого q_snr=q_floor.
    """

    base_w_v: float = 0.5
    base_w_a: float = 0.5
    q_floor: float = 0.2
    sharpness_ref: float = 150.0
    snr_ref_db: float = 20.0
    snr_floor_db: float = 0.0

    def _img_quality(self, window: AlignedWindow) -> float | None:
        best_v = window.best_video()
        if best_v is None:
            return None
        sharp = best_v.quality.img_sharpness
        bright = best_v.quality.img_brightness
        if sharp is None and bright is None:
            return None
        # резкость нормируем к эталону; яркость штрафует за «слишком темно/светло» (отклонение от 0.5)
        q_sharp = 1.0 if sharp is None else max(0.0, min(1.0, sharp / self.sharpness_ref))
        q_bright = 1.0 if bright is None else max(0.0, 1.0 - 2.0 * abs(bright - 0.5))
        q = q_sharp * q_bright if (sharp is not None and bright is not None) else (q_sharp if sharp is not None else q_bright)
        return max(self.q_floor, q)

    def _audio_quality(self, window: AlignedWindow) -> tuple[float | None, float | None]:
        best_a = window.best_audio()
        if best_a is None or best_a.quality.snr_db is None:
            return None, None
        snr = best_a.quality.snr_db
        span = max(1e-6, self.snr_ref_db - self.snr_floor_db)
        q = (snr - self.snr_floor_db) / span
        return max(self.q_floor, min(1.0, q)), snr

    def weights(self, window: AlignedWindow) -> GatingResult:
        q_img = self._img_quality(window)
        q_aud, snr = self._audio_quality(window)
        eff_v = self.base_w_v * (q_img if q_img is not None else 1.0)
        eff_a = self.base_w_a * (q_aud if q_aud is not None else 1.0)
        wv, wa = _normalize(eff_v, eff_a)
        return GatingResult(w_v=wv, w_a=wa, snr_audio=snr, img_quality=q_img)
