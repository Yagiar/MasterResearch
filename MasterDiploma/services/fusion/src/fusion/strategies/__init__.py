"""Стратегии слияния (паттерн Strategy): video-only (MVP), audio-only, late, hybrid.
Каждая получает AlignedWindow (+ веса от GatingPolicy) и возвращает FusionOutcome
(вероятность p_fused, бинарное решение, разбивка вкладов) либо None (окна недостаточно)."""
