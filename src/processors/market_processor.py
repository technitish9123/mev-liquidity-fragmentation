"""
Computes volatility and gas price statistics from real market data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# Fallback gas mean from config (Gwei)
_GAS_MEAN_FALLBACK = getattr(C, "GAS_MEAN", 34.2)


def compute_volatility(prices: pd.Series) -> float:
    """
    Realized volatility = std of log-returns.
    prices: Series of daily ETH/USD prices within the window period.
    Returns 0.0 if fewer than 2 prices.
    Formula: std(log(p[t]/p[t-1])) over the window.
    """
    if prices is None:
        return 0.0

    # Drop NaN and non-positive values before log
    clean = prices.dropna()
    clean = clean[clean > 0.0]

    if len(clean) < 2:
        return 0.0

    try:
        log_returns = np.log(clean.values[1:] / clean.values[:-1])
        # Replace any resulting NaN/Inf (e.g., from zero prices that slipped through)
        log_returns = log_returns[np.isfinite(log_returns)]
        if len(log_returns) == 0:
            return 0.0
        vol = float(np.std(log_returns, ddof=1))
        return 0.0 if np.isnan(vol) else vol
    except Exception as exc:
        log.warning("compute_volatility failed: %s", exc)
        return 0.0


def compute_avg_gas(gas_values: list[float]) -> float:
    """
    Mean gas price in Gwei from sampled blocks.
    Returns C.GAS_MEAN fallback (34.2) if list is empty.
    """
    if not gas_values:
        return _GAS_MEAN_FALLBACK

    arr = np.array(gas_values, dtype=float)
    # Filter out NaN and non-finite values
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return _GAS_MEAN_FALLBACK

    mean_gas = float(np.mean(arr))
    if np.isnan(mean_gas):
        return _GAS_MEAN_FALLBACK

    return mean_gas


def compute_volume(pools: list[dict]) -> float:
    """
    Total DEX volume in $M for the window.
    Sums volumeUSD from all pools in the snapshot.
    Note: volumeUSD in subgraph is cumulative; use as proxy for activity level.
    Divides by 1_000_000 to convert to $M.
    Returns 0.0 if pools is empty.
    """
    if not pools:
        return 0.0

    total = 0.0
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        val = pool.get("volumeUSD")
        if val is None:
            continue
        try:
            v = float(val)
            if np.isfinite(v) and v >= 0.0:
                total += v
        except (ValueError, TypeError):
            continue

    return total / 1_000_000.0
