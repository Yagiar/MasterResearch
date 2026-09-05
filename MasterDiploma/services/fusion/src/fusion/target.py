"""Целевая переменная fusion (it-52, ревью §3): presence | airborne.

Ревью §3 и исследование it-51: «дрон присутствует», «виден», «летит» — разные события.
Конфиг `fusion.target` фиксирует целевую переменную:
  - `presence`  (по умолчанию, прежнее поведение): «дрон присутствует в кадре/зоне» —
    видео-канал подтверждает цель самим фактом детекции;
  - `airborne`  : «БПЛА активен (в полёте)» — детекция видео подтверждает цель только
    при наличии признака состояния (движение трека ≥ `motion_floor`); статичная цель
    (стоит на земле) переводится видео-каналом в «non-drone».
Акустический канал в обоих режимах работает одинаково (p_a — признак работающих винтов).
Если motion_score неизвестен (трекер выключен), фильтр не применяется — режим airborne
требует включённого трекера для осмысленной дискриминации.
"""

from __future__ import annotations

from .window_buffer import AlignedWindow

_TARGETS = ("presence", "airborne")


def validate_target(target: str) -> str:
    if target not in _TARGETS:
        raise ValueError(f"неизвестная целевая переменная: {target!r}; допустимо: {_TARGETS}")
    return target


def apply_target_filter(window: AlignedWindow, *, target: str, motion_floor: float) -> AlignedWindow:
    """Для target=airborne: статичное видео-подтверждение (motion < floor) → видео «не дрон».

    Окно без видео, видео не-«дрон», либо без замера движения возвращается без изменений
    (нельзя утверждать «не летит» без признака состояния).
    """
    if target != "airborne":
        return window
    bv = window.best_video()
    if bv is None or bv.label != "drone":
        return window
    ms = bv.quality.motion_score
    if ms is None or ms >= motion_floor:
        return window
    new_v = bv.model_copy(update={"label": "non-drone", "confidence": 0.0})
    video = [new_v if m.msg_id == bv.msg_id else m for m in window.video]
    return AlignedWindow(source_id=window.source_id, t0=window.t0, t1=window.t1,
                         video=video, audio=list(window.audio), late=window.late,
                         joint=window.joint, media_ts=window.media_ts)
