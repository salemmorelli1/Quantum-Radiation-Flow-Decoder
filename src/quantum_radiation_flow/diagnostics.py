"""Curve diagnostics and uncertainty summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CurveDiagnostic:
    peak_time: float
    peak_value: float
    pre_peak_violations: int
    post_peak_violations: int


def curve_diagnostic(times: Sequence[float], values: Sequence[float]) -> CurveDiagnostic:
    if len(times) != len(values) or len(times) < 3:
        raise ValueError("times and values must have the same length of at least three")
    peak_index = max(range(len(values)), key=lambda index: values[index])
    pre = sum(values[i + 1] < values[i] for i in range(peak_index))
    post = sum(values[i + 1] > values[i] for i in range(peak_index, len(values) - 1))
    return CurveDiagnostic(
        peak_time=float(times[peak_index]),
        peak_value=float(values[peak_index]),
        pre_peak_violations=pre,
        post_peak_violations=post,
    )


def monte_carlo_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))
