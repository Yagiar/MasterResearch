"""HybridFusion — гибридное слияние (для пилотных замеров этапа 6).

Замысел диссертации: сочетание раннего/промежуточного слияния признаков и позднего
слияния решений с adaptive gating. Полноценное feature-level слияние требует общей
точки в архитектурах детекторов (вне объёма пилота). Здесь реализован **прагматичный
гибрид на уровне решений с нелинейной комбинацией и адаптивными весами**:

  1) веса w_v / w_a приходят от AdaptiveGating (зависят от качества кадра / SNR аудио);
  2) базовая оценка — взвешенная сумма: s = w_v·p_v + w_a·p_a;
  3) если ОБА канала активны и оба «уверенно дрон» — нелинейный буст
     (вероятностное «ИЛИ»: p = 1 − (1−p_v)(1−p_a), берём max(s, p) + δ_conf);
  4) если каналы активны, но противоречат (один drone, другой нет) — штраф −δ_unconf к s;
  5) если активен только один канал — решение по нему (без штрафа за отсутствие второго);
  6) clip(·, 0, 1), сравнение с порогом.

Отличие от LateFusion: п. 3 (вероятностное ИЛИ при согласии) и опора на адаптивные веса.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..window_buffer import AlignedWindow
from .base import FusionOutcome, _clip01

_LABEL_DRONE = "drone"


@dataclass
class HybridFusion:
    """Гибридное слияние (decision-level, нелинейное, с адаптивными весами)."""

    delta_conf: float = 0.1
    delta_unconf: float = 0.15
    label_threshold: float = 0.5
    mode: str = "hybrid"

    def _says_drone(self, msg) -> bool:
        return msg is not None and msg.label == _LABEL_DRONE and msg.confidence >= self.label_threshold

    def fuse(self, window: AlignedWindow, *, w_v: float, w_a: float, threshold: float) -> FusionOutcome | None:
        best_v = window.best_video()
        best_a = window.best_audio()
        if best_v is None and best_a is None:
            return None

        v_active, a_active = best_v is not None, best_a is not None
        p_v = (best_v.confidence if best_v and best_v.label == _LABEL_DRONE else 0.0) if best_v else 0.0
        p_a = (best_a.confidence if best_a and best_a.label == _LABEL_DRONE else 0.0) if best_a else 0.0

        if v_active and not a_active:
            p_fused, delta, eff_w_v, eff_w_a = p_v, 0.0, 1.0, 0.0
        elif a_active and not v_active:
            p_fused, delta, eff_w_v, eff_w_a = p_a, 0.0, 0.0, 1.0
        else:
            eff_w_v, eff_w_a = w_v, w_a
            s = eff_w_v * p_v + eff_w_a * p_a
            v_drone, a_drone = self._says_drone(best_v), self._says_drone(best_a)
            if v_drone and a_drone:
                prob_or = 1.0 - (1.0 - p_v) * (1.0 - p_a)
                delta = self.delta_conf
                p_fused = max(s, prob_or) + delta
            elif v_drone ^ a_drone:
                delta = -self.delta_unconf
                p_fused = s + delta
            else:
                delta = 0.0
                p_fused = s

        p_fused = _clip01(p_fused)
        ids = tuple(m.msg_id for m in (best_v, best_a) if m is not None)
        return FusionOutcome(
            p_fused=p_fused,
            decision=p_fused >= threshold,
            p_v=p_v if v_active else None,
            p_a=p_a if a_active else None,
            w_v=eff_w_v,
            w_a=eff_w_a,
            delta=delta,
            source_msg_ids=ids,
        )
