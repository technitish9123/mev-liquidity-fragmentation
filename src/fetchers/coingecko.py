"""
CryptoCompare ETH/USD daily price fetcher (replaces CoinGecko which now requires paid key).
API: https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&limit={days}&toTs={unix_ts}
Returns up to 2000 days of daily OHLCV data.
Free, no auth required.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.utils import rate_limited_get, load_cache, save_cache, block_to_timestamp
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_eth_prices(
    block_start: int = C.BLOCK_START,
    block_end: int = C.BLOCK_END,
) -> pd.DataFrame:
    """
    Fetch ETH/USD daily prices covering block_start → block_end via CryptoCompare.
    Returns DataFrame with columns: [timestamp_ms, price_usd, date].
    Cached as 'eth_prices_2025' in C.CACHE_COINGECKO.
    Cache hit: load and return. Cache miss: fetch and save.
    """
    cache_key = "eth_prices_2025"
    cached = load_cache(C.CACHE_COINGECKO, cache_key)
    if cached is not None:
        log.debug("CryptoCompare cache hit: eth_prices_2025")
        df = pd.DataFrame(cached, columns=["timestamp_ms", "price_usd"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()
        return df

    ts_end = block_to_timestamp(
        block_end,
        ref_block=C.REFERENCE_BLOCK,
        ref_ts=C.REFERENCE_TIMESTAMP,
        block_time=C.BLOCK_TIME_SEC,
    )

    params = {
        "fsym": "ETH",
        "tsym": "USD",
        "limit": 400,
        "toTs": ts_end,
    }

    response = rate_limited_get(
        CRYPTOCOMPARE_URL,
        params=params,
        delay=1.0,
        max_retries=C.MAX_RETRIES,
    )

    if response is None:
        log.error("CryptoCompare fetch returned None")
        return pd.DataFrame(columns=["timestamp_ms", "price_usd", "date"])

    # CryptoCompare nests data under Data.Data
    outer = response.get("Data")
    if outer is None:
        log.error("CryptoCompare response missing 'Data' key: %s", list(response.keys()))
        return pd.DataFrame(columns=["timestamp_ms", "price_usd", "date"])

    entries = outer.get("Data")
    if not entries or not isinstance(entries, list):
        log.error("CryptoCompare response missing 'Data.Data' list")
        return pd.DataFrame(columns=["timestamp_ms", "price_usd", "date"])

    # Build raw list [[timestamp_ms, price_usd], ...] for cache storage
    raw = []
    for entry in entries:
        ts = entry.get("time")
        close = entry.get("close")
        if ts is None or close is None:
            continue
        # Skip zero-price entries (API sometimes pads boundaries with zeroes)
        if float(close) == 0.0:
            continue
        raw.append([int(ts) * 1000, float(close)])

    if not raw:
        log.error("CryptoCompare returned no usable price entries")
        return pd.DataFrame(columns=["timestamp_ms", "price_usd", "date"])

    save_cache(C.CACHE_COINGECKO, cache_key, raw)

    df = pd.DataFrame(raw, columns=["timestamp_ms", "price_usd"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()

    log.info(
        "CryptoCompare: fetched %d price points from %s to %s",
        len(df),
        df["date"].iloc[0].date() if len(df) else "N/A",
        df["date"].iloc[-1].date() if len(df) else "N/A",
    )
    return df


def prices_for_window(
    prices_df: pd.DataFrame,
    block_start: int,
    block_end: int,
    expand_days: int = 7,
) -> pd.Series:
    """
    Return price_usd values for volatility computation around the window.

    Since each window is ~1.4 days and daily prices give only 1-2 points,
    we expand the lookup by ±expand_days (default 7) to get enough points
    for a meaningful std(log-returns) calculation (~14 points).
    """
    if prices_df.empty:
        return pd.Series(dtype=float)

    ts_mid = block_to_timestamp(
        (block_start + block_end) // 2,
        ref_block=C.REFERENCE_BLOCK,
        ref_ts=C.REFERENCE_TIMESTAMP,
        block_time=C.BLOCK_TIME_SEC,
    )
    expand_sec = expand_days * 86400
    ts_start_ms = (ts_mid - expand_sec) * 1000
    ts_end_ms   = (ts_mid + expand_sec) * 1000

    mask = (prices_df["timestamp_ms"] >= ts_start_ms) & (
        prices_df["timestamp_ms"] <= ts_end_ms
    )
    return prices_df.loc[mask, "price_usd"].reset_index(drop=True)
