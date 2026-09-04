"""LateFusion — позднее слияние на уровне решений.

Взвешенная сумма уверенностей по **активным** модальностям + правило компенсации Δ:

    p_fused = clip( w̃_v·p_v + w̃_a·p_a + Δ , 0, 1 )

где `w̃` — веса w_v/w_a, **перенормированные по активным каналам** (канал активен, если по нему
есть детекция в окне). То есть если аудио в окне нет — `w̃_v = 1`, `w̃_a = 0` (вес не «теряется
в пустоту» — иначе p_fused был бы искусственно занижен; это и есть базовый принцип gating).

Правило компенсации Δ (раздел 5.2 отчёта по НИР):
  - +δ_conf, если ОБА активных канала дают «drone» (взаимное подтверждение);
  - −δ_unconf, если только ОДИН канал даёт «drone», а второй активен и даёт «non-drone»
    (один канал противоречит — снижаем уверенность);
  - Δ = 0, если активен только один канал (второй молчит — не штрафуем за отсутствие данных).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..window_buffer import AlignedWindow
from .base import FusionOutcome, _clip01

_LABEL_DRONE = "drone"


def _effective_weights(w_v: float, w_a: float, v_active: bool, a_active: bool) -> tuple[float, float]:
    """Перенормировать веса по активным каналам (сумма по активным = 1)."""
    wv = w_v if v_active else 0.0
    wa = w_a if a_active else 0.0
    s = wv + wa
    if s <= 0:
        # оба «активны» формально, но веса нулевые — равномерно по активным
        n = (1 if v_active else 0) + (1 if a_active else 0)
        return (1.0 / n if v_active and n else 0.0, 1.0 / n if a_active and n else 0.0)
    return wv / s, wa / s


@dataclass
class LateFusion:
    """Позднее слияние с перенормировкой весов по активным каналам и правилом компенсации Δ."""

    delta_conf: float = 0.1       # бонус за взаимное подтверждение обоих каналов
    delta_unconf: float = 0.1     # штраф за противоречие одного канала
    label_threshold: float = 0.5  # порог, выше которого считаем, что канал «сказал drone»
    mode: str = "late"

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
        v_drone, a_drone = self._says_drone(best_v), self._says_drone(best_a)

        delta = 0.0
        if v_active and a_active:
            if v_drone and a_drone:
                delta = self.delta_conf
            elif v_drone ^ a_drone:
                delta = -self.delta_unconf

        eff_w_v, eff_w_a = _effective_weights(w_v, w_a, v_active, a_active)
        p_fused = _clip01(eff_w_v * p_v + eff_w_a * p_a + delta)

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
