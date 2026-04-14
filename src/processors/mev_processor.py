"""
Aggregates raw ZeroMEV block data into per-window MEV intensity I(b).

MEV intensity = total_mev_usd / (block_count * avg_block_value_usd)
Since block_value is hard to get, we normalize differently:
  I(window) = mean(extractor_profit_usd per block) / NORMALIZATION_CONSTANT

NORMALIZATION_CONSTANT is computed as the 95th-percentile of all window means
across the full dataset, so I ∈ [0, ~1] with most values < 0.2.

Smoothing: for figures, use rolling mean over W windows (implemented in pipeline).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C
import numpy as np
import logging

log = logging.getLogger(__name__)


def _extract_profit(block: dict) -> float:
    """
    Extract the extractor profit value from a ZeroMEV block dict.

    Tries keys in priority order:
      1. 'extractor_profit_usd'
      2. 'total_profit_usd'
      3. sum of 'arb_profit_usd', 'sandwich_profit_usd', 'liquidation_profit_usd'
    Returns 0.0 if none are present or all are None/NaN.
    """
    # Priority 1
    val = block.get("extractor_profit_usd")
    if val is not None:
        try:
            v = float(val)
            if not np.isnan(v):
                return max(0.0, v)
        except (ValueError, TypeError):
            pass

    # Priority 2
    val = block.get("total_profit_usd")
    if val is not None:
        try:
            v = float(val)
            if not np.isnan(v):
                return max(0.0, v)
        except (ValueError, TypeError):
            pass

    # Priority 3: sum of component fields
    total = 0.0
    found_any = False
    for key in ("arb_profit_usd", "sandwich_profit_usd", "liquidation_profit_usd"):
        val = block.get(key)
        if val is not None:
            try:
                v = float(val)
                if not np.isnan(v):
                    total += max(0.0, v)
                    found_any = True
            except (ValueError, TypeError):
                pass

    if found_any:
        return total

    return 0.0


def compute_window_mev_raw(mev_blocks: list[dict]) -> float:
    """
    Given list of ZeroMEV block dicts for a window,
    return mean extractor_profit_usd across sampled blocks.
    Returns 0.0 if list is empty.
    Field name: try 'extractor_profit_usd', fallback to 'total_profit_usd', then sum of arb+sandwich+liq.
    """
    if not mev_blocks:
        return 0.0

    profits = []
    for block in mev_blocks:
        if not isinstance(block, dict):
            continue
        profits.append(_extract_profit(block))

    if not profits:
        return 0.0

    mean_val = float(np.mean(profits))
    if np.isnan(mean_val):
        return 0.0
    return mean_val


def normalize_intensities(raw_values: list[float]) -> list[float]:
    """
    Normalize a list of raw MEV values to produce I ∈ [0, ~0.2].
    Strategy: divide by 95th percentile of non-zero values.
    If all zeros, return list of zeros.
    """
    if not raw_values:
        return []

    arr = np.array(raw_values, dtype=float)

    # Replace NaNs with 0
    arr = np.where(np.isnan(arr), 0.0, arr)

    non_zero = arr[arr > 0.0]
    if len(non_zero) == 0:
        return [0.0] * len(raw_values)

    p95 = float(np.percentile(non_zero, 95))

    if p95 == 0.0:
        return [0.0] * len(raw_values)

    normalized = arr / p95
    # Clamp to [0, inf) — no upper clamp; values above 1.0 are outliers above p95
    normalized = np.maximum(normalized, 0.0)

    return normalized.tolist()


def smooth_intensities(intensities: list[float], W: int = 10) -> list[float]:
    """
    Apply rolling mean with window W over the intensity list.
    Pad the first W-1 values with the cumulative mean.
    Returns list of same length as input.
    """
    if not intensities:
        return []

    if W <= 0:
        W = 1

    arr = np.array(intensities, dtype=float)
    # Replace NaNs with 0 for smoothing
    arr = np.where(np.isnan(arr), 0.0, arr)

    n = len(arr)
    result = np.empty(n, dtype=float)

    for i in range(n):
        if i < W - 1:
            # Cumulative mean for the first W-1 positions
            result[i] = float(np.mean(arr[: i + 1]))
        else:
            # Full rolling mean
            result[i] = float(np.mean(arr[i - W + 1 : i + 1]))

    return result.tolist()


def classify_regime(intensity: float, low_thresh: float, high_thresh: float) -> str:
    """Return 'low', 'medium', or 'high' based on thresholds."""
    try:
        v = float(intensity)
    except (ValueError, TypeError):
        return "low"

    if np.isnan(v) or v < low_thresh:
        return "low"
    if v >= high_thresh:
        return "high"
    return "medium"
