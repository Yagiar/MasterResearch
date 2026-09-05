#!/usr/bin/env python3
"""Isolated behavioral checks for Yagiar/MasterResearch, inspected 2026-09-05.

IMPORTANT: This is a reviewer-authored, minimal transcription of selected logic
from the public main branch, with lightweight message stubs. It does not import
or run the repository, its test suite, Kafka, model weights, or reported ML runs.
It demonstrates counterexamples to properties of the inspected algorithms.
Run with Python 3.10+: python reviewer_checks.py

Source paths are listed next to the relevant checks. Revisions were not pinned;
compare with the repository before using these as regression tests after fixes.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class Message:
    msg_id: str
    ts: float
    modality: str
    confidence: float = 0.9
    label: str = "drone"
    source_id: str = "test-source"
    p_drone: float | None = None


@dataclass
class Window:
    video: list[Message] = field(default_factory=list)
    audio: list[Message] = field(default_factory=list)


class Buffer:
    """Relevant logic from services/fusion/src/fusion/window_buffer.py."""
    def __init__(self, epsilon_ms: float = 600.0) -> None:
        self.eps = max(0.0, epsilon_ms / 1000.0)
        self.video: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=256))
        self.audio: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=256))

    def add(self, msg: Message) -> Window:
        sid = msg.source_id
        target = self.video if msg.modality == "video" else self.audio
        target[sid].append(msg)
        t0, t1 = msg.ts - self.eps, msg.ts + self.eps
        for q in (self.video[sid], self.audio[sid]):
            while q and q[0].ts < t0:
                q.popleft()
        return Window(
            [m for m in self.video[sid] if t0 <= m.ts <= t1]
            or ([msg] if msg.modality == "video" else []),
            [m for m in self.audio[sid] if t0 <= m.ts <= t1]
            or ([msg] if msg.modality == "audio" else []),
        )


class Median:
    """Relevant logic from services/fusion/src/fusion/temporal.py."""
    def __init__(self, k: int = 5) -> None:
        self.history: deque[float] = deque(maxlen=k)

    def update(self, p: float) -> float:
        self.history.append(float(p))
        values = sorted(self.history)
        return values[len(values) // 2]


def late_present(pv: float, pa: float, wv: float, wa: float) -> float:
    """Both channels present, respective drone flags given by >= 0.5.
    Relevant logic from services/fusion/src/fusion/strategies/late.py.
    """
    total = wv + wa
    ev, ea = (wv / total, wa / total) if total > 0 else (0.5, 0.5)
    vd, ad = pv >= 0.5, pa >= 0.5
    delta = 0.1 if vd and ad else (-0.1 if vd != ad else 0.0)
    return min(1.0, max(0.0, ev * pv + ea * pa + delta))


def hybrid_present(pv: float, pa: float, wv: float, wa: float) -> float:
    """Both channels present; strategies/hybrid.py relevant arithmetic."""
    s = wv * pv + wa * pa
    vd, ad = pv >= 0.5, pa >= 0.5
    if vd and ad:
        p = max(s, 1.0 - (1.0 - pv) * (1.0 - pa)) + 0.1
    elif vd != ad:
        p = s - 0.15
    else:
        p = s
    return min(1.0, max(0.0, p))


def stress_decide(policy: str, pv: float, pa: float | None) -> int:
    """First branches of research/stress_sim.py:decide."""
    if policy == "video-only":
        return int(pv >= 0.5)
    if pa is None:
        return int(pv >= 0.5)
    if policy == "audio-only":
        return int(pa >= 0.5)
    raise ValueError("This minimal check implements only unimodal policies")


def main() -> None:
    result: dict[str, object] = {
        "scope": "Reviewer-authored isolated transcriptions; not repository execution",
        "inspected_branch": "main (unpinned)",
        "inspection_date": "2026-09-05",
    }
    v10 = Message("v10", 10.0, "video")
    a10 = Message("a10", 10.0, "audio")
    v12 = Message("v12", 12.0, "video")
    counts = []
    for stream in ([v10, a10, v12], [v10, v12, a10]):
        buf = Buffer()
        windows = [buf.add(m) for m in stream]
        counts.append(sum(bool(w.video and w.audio) for w in windows))
    assert counts == [1, 0]
    result["arrival_order_mixed_windows"] = {
        "v10_a10_v12": counts[0], "v10_v12_a10": counts[1],
        "note": "Same event timestamps; late a10 cannot recover evicted v10."
    }

    # Consumer calls apply_audio_smoothing on every received inference message.
    # One new audio message remains best_audio for two subsequent video triggers.
    med = Median(5)
    for _ in range(5):
        med.update(0.9)
    buf = Buffer()
    stream = [Message("same-audio", 10.0, "audio", 0.9, "non-drone", p_drone=0.1),
              Message("video-1", 10.04, "video"),
              Message("video-2", 10.08, "video")]
    repeated = []
    audio_ids = []
    for msg in stream:
        window = buf.add(msg)
        best_audio = max(window.audio, key=lambda m: m.confidence)
        score = best_audio.p_drone if best_audio.p_drone is not None else (
            best_audio.confidence if best_audio.label == "drone" else 0.0
        )
        repeated.append(med.update(score))
        audio_ids.append(best_audio.msg_id)
    assert repeated == [0.9, 0.9, 0.1]
    assert len(set(audio_ids)) == 1
    result["one_audio_message_counted_three_times"] = {
        "audio_ids": audio_ids, "median_outputs": repeated,
        "expected_if_update_once_per_unique_audio": [0.9, 0.9, 0.9],
    }

    late = late_present(0.55, 0.0, 0.5, 0.0)
    hybrid = hybrid_present(0.9, 0.0, 0.5, 0.0)
    assert abs(late - 0.45) < 1e-12
    assert abs(hybrid - 0.30) < 1e-12
    result["zero_audio_weight_does_not_remove_audio_effect"] = {
        "late_video_only": 0.55, "late_audio_present_weight_zero": late,
        "hybrid_video_only": 0.9, "hybrid_audio_present_weight_zero": hybrid,
        "decision_threshold": 0.5,
    }

    med = Median(5)
    for _ in range(5):
        med.update(0.1)
    step = [med.update(0.9) for _ in range(5)]
    delay_hops = next(i for i, p in enumerate(step) if p >= 0.5)
    assert delay_hops == 2
    result["causal_median_step_response"] = {
        "outputs_after_step": step, "delay_vs_raw_hops": delay_hops,
        "hop_seconds": 0.5, "delay_vs_raw_seconds": delay_hops * 0.5,
    }

    fallback_outputs = [stress_decide("audio-only", pv, None) for pv in (0.1, 0.9)]
    assert fallback_outputs == [0, 1]
    result["stress_audio_only_depends_on_video_when_audio_missing"] = {
        "video_scores": [0.1, 0.9], "audio_scores": [None, None],
        "audio_only_outputs": fallback_outputs,
    }

    output = Path(__file__).with_name("reviewer_checks_results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("All isolated counterexample assertions passed.")


if __name__ == "__main__":
    main()
