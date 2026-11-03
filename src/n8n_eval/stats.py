"""Statistics for LLM evaluations with repeats.

Repeats of one case are correlated: the same question fails for the same reason. Treating every
run as independent makes confidence intervals too narrow, so intervals here resample cases
(cluster bootstrap), and a comparison of two versions is paired by case.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Matrix = Mapping[str, Sequence[bool]]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def case_means(matrix: Matrix) -> dict[str, float]:
    return {case: float(np.mean(runs)) for case, runs in matrix.items() if len(runs)}


def cluster_bootstrap(values: Sequence[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, len(arr), size=(n, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def pass_rate(matrix: Matrix) -> dict[str, float | int]:
    means = list(case_means(matrix).values())
    runs = [r for rs in matrix.values() for r in rs]
    lo, hi = cluster_bootstrap(means)
    naive_lo, naive_hi = wilson(int(sum(runs)), len(runs))
    return {
        "rate": float(np.mean(means)) if means else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "naive_ci_low": naive_lo,
        "naive_ci_high": naive_hi,
        "cases": len(means),
        "runs": len(runs),
    }


def flaky_cases(matrix: Matrix) -> list[str]:
    """Cases that both passed and failed across repeats."""
    return [case for case, runs in matrix.items() if 0 < sum(runs) < len(runs)]


def single_run_spread(matrix: Matrix) -> dict[str, float]:
    """How much a pass rate from one run per case could vary: the range over repeat indices."""
    repeats = min((len(r) for r in matrix.values()), default=0)
    if repeats < 2:
        return {}
    rates = [float(np.mean([runs[i] for runs in matrix.values()])) for i in range(repeats)]
    return {"min": min(rates), "max": max(rates), "repeats": repeats}


def compare(base: Matrix, cand: Matrix, n: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison by case: mean of (candidate - baseline) per-case pass rates."""
    common = sorted(set(base) & set(cand))
    b, c = case_means({k: base[k] for k in common}), case_means({k: cand[k] for k in common})
    diffs = np.array([c[k] - b[k] for k in common])
    if len(diffs) == 0:
        return {"cases": 0}
    idx = np.random.default_rng(seed).integers(0, len(diffs), size=(n, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "cases": len(common),
        "baseline": float(np.mean(list(b.values()))),
        "candidate": float(np.mean(list(c.values()))),
        "diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_not_worse": float((boot >= 0).mean()),
        "broken": [k for k in common if b[k] - c[k] >= 0.5],
        "fixed": [k for k in common if c[k] - b[k] >= 0.5],
        "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
    }


def cases_needed(sd_diff: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Cases needed to detect a drop of `delta` in a paired comparison with the observed spread."""
    if sd_diff <= 0 or delta <= 0:
        return None
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil((z * sd_diff / delta) ** 2)
