"""
Shared fit-quality metrics.

Why this module exists
-----------------------
The naive MAPE formula

    mape = mean(|actual - pred| / (|actual| + 1e-12))

is what both the Results tabs and the optimizer's loss function used to
compute directly, inline, in three different places (pipeline.py's
`_postprocess_equation`, and optimizer.py's single- and joint-mode
composite losses). It works fine for Dependent 1 in a typical MMM (Sales/
Revenue), which is always comfortably far from zero. But Dependent 2 is
very often a KPI that legitimately *is* zero or very small in some
periods (trial signups, leads, search-interest index, a raw count metric,
etc.) — unlike Sales, "Consideration %" style KPIs aren't the only valid
choice for Dependent 2.

When even a single period has actual == 0, the "+ 1e-12" floor means that
period's percentage error becomes |residual| / 1e-12 — i.e. billions or
trillions of percent — and completely swamps the mean, producing a
reported "MAPE" that is meaningless (e.g. "4,896,334,011,426.49%")
even though every other period fits well. This is what was showing up as
an obviously-wrong MAPE for Dependent 2 on Tabs 5 and 6.

`safe_mape` fixes this by flooring the denominator relative to the
series' own scale (1% of the mean |actual|) instead of an absolute
constant. A handful of exactly-zero (or near-zero) periods can no longer
dominate the average; they're still penalized (their error is measured
against 1% of the series' typical size, so it's still a large relative
error), but no longer produce a nonsensical astronomical number that
makes the metric useless. For a series with no near-zero values this is
numerically identical to the original formula, so Dependent 1's MAPE is
unaffected.
"""

import numpy as np


def safe_mape(residuals, target_vals) -> float:
    """
    Zero-safe Mean Absolute Percentage Error.

    Denominator for each period is max(|actual_t|, floor), where
    floor = max(1% of mean(|actual|), 1e-8). This keeps the metric
    well-behaved (no divide-by-near-zero blowups) while leaving it
    numerically unchanged from the naive formula whenever the series
    doesn't have near-zero values.
    """
    residuals   = np.asarray(residuals, dtype=float)
    target_vals = np.asarray(target_vals, dtype=float)

    abs_actual = np.abs(target_vals)
    scale = abs_actual.mean() if abs_actual.size else 0.0
    floor = max(0.01 * scale, 1e-8)
    denom = np.maximum(abs_actual, floor)

    return float(np.mean(np.abs(residuals) / denom))
