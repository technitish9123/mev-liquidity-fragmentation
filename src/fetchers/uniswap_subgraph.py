"""
DeFi Llama pool data fetcher (replaces Uniswap V3 hosted subgraph which was shut down).

Two-phase approach:
  Phase 1 (once): Fetch all Uniswap V3 Ethereum pool IDs + metadata from /pools
  Phase 2 (per window): Fetch historical TVL snapshot for each pool at window's date

APIs used:
  List all pools:   GET https://yields.llama.fi/pools
  Pool history:     GET https://yields.llama.fi/chart/{pool_id}

DeFi Llama has 609 Uniswap V3 Ethereum pools with daily TVL since 2022.
This gives us the full liquidity graph per day, which we map to block windows.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.utils import rate_limited_get, load_cache, save_cache, block_to_timestamp, window_midpoint
import pandas as pd
import logging
import datetime

log = logging.getLogger(__name__)

DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"
DEFILLAMA_CHART_URL = "https://yields.llama.fi/chart/{pool_id}"


def fetch_pool_list() -> list[dict]:
    """
    Fetch and cache all Uniswap V3 Ethereum pools from DeFi Llama.

    Filters:
      - project == "uniswap-v3"
      - chain == "Ethereum"
      - tvlUsd > C.UNISWAP_MIN_TVL_USD

    Sorts by tvlUsd descending, returns top C.MAX_POOLS_PER_WINDOW pools.
    Cache key: "uniswap_v3_pool_list" in C.CACHE_UNISWAP.
    """
    cache_key = "uniswap_v3_pool_list"
    cached = load_cache(C.CACHE_UNISWAP, cache_key)
    if cached is not None:
        log.debug("DeFi Llama pool list cache hit: %d pools", len(cached))
        return cached

    response = rate_limited_get(
        DEFILLAMA_POOLS_URL,
        delay=C.REQUEST_DELAY_SEC,
        max_retries=C.MAX_RETRIES,
    )

    if response is None:
        log.error("DeFi Llama /pools returned None")
        return []

    all_pools = response.get("data")
    if not all_pools or not isinstance(all_pools, list):
        log.error("DeFi Llama /pools response missing 'data' list: %s", list(response.keys()))
        return []

    # Filter to Uniswap V3 on Ethereum above TVL threshold
    filtered = []
    for pool in all_pools:
        if pool.get("project") != "uniswap-v3":
            continue
        if pool.get("chain") != "Ethereum":
            continue
        tvl = pool.get("tvlUsd") or 0.0
        try:
            tvl = float(tvl)
        except (ValueError, TypeError):
            tvl = 0.0
        if tvl < C.UNISWAP_MIN_TVL_USD:
            continue
        filtered.append(pool)

    # Sort by TVL descending and keep top MAX_POOLS_PER_WINDOW
    filtered.sort(key=lambda p: float(p.get("tvlUsd") or 0.0), reverse=True)
    filtered = filtered[: C.MAX_POOLS_PER_WINDOW]

    # Retain only fields needed downstream
    slim = []
    for pool in filtered:
        underlying = pool.get("underlyingTokens") or []
        slim.append(
            {
                "pool": pool.get("pool", ""),
                "symbol": pool.get("symbol", ""),
                "tvlUsd": float(pool.get("tvlUsd") or 0.0),
                "volumeUsd1d": float(pool.get("volumeUsd1d") or 0.0),
                "underlyingTokens": underlying,
            }
        )

    save_cache(C.CACHE_UNISWAP, cache_key, slim)
    log.info("DeFi Llama pool list: %d Uniswap V3 Ethereum pools after filter", len(slim))
    return slim


def fetch_pool_history(pool_id: str) -> list[dict]:
    """
    Fetch and cache TVL history for a single pool from DeFi Llama.

    Returns list of dicts with keys: timestamp (ISO string), tvlUsd, apy.
    Cache key: f"pool_history_{pool_id}" in C.CACHE_UNISWAP.
    Returns [] on failure.
    """
    cache_key = f"pool_history_{pool_id}"
    cached = load_cache(C.CACHE_UNISWAP, cache_key)
    if cached is not None:
        log.debug("DeFi Llama pool history cache hit: %s (%d entries)", pool_id, len(cached))
        return cached

    url = DEFILLAMA_CHART_URL.format(pool_id=pool_id)
    response = rate_limited_get(
        url,
        delay=C.REQUEST_DELAY_SEC,
        max_retries=C.MAX_RETRIES,
    )

    if response is None:
        log.warning("DeFi Llama chart returned None for pool %s", pool_id)
        return []

    history = response.get("data")
    if not history or not isinstance(history, list):
        log.warning(
            "DeFi Llama chart missing 'data' for pool %s: %s",
            pool_id,
            list(response.keys()) if isinstance(response, dict) else type(response),
        )
        return []

    # Normalise entries
    normalised = []
    for entry in history:
        ts = entry.get("timestamp")
        tvl = entry.get("tvlUsd")
        apy = entry.get("apy")
        if ts is None or tvl is None:
            continue
        normalised.append(
            {
                "timestamp": str(ts),
                "tvlUsd": float(tvl) if tvl is not None else 0.0,
                "apy": float(apy) if apy is not None else 0.0,
            }
        )

    save_cache(C.CACHE_UNISWAP, cache_key, normalised)
    log.debug("DeFi Llama pool history %s: %d entries", pool_id, len(normalised))
    return normalised


def get_pool_tvl_at_date(history: list[dict], target_date: str) -> float:
    """
    Find TVL for target_date (YYYY-MM-DD) in pool history using nearest-neighbor match.

    history entries have 'timestamp' as ISO-format strings (e.g. "2025-01-15T00:00:00").
    Returns 0.0 if history is empty or no entry is within a reasonable range.
    """
    if not history:
        return 0.0

    try:
        target_dt = datetime.date.fromisoformat(target_date)
    except ValueError:
        log.warning("Invalid target_date format: %s", target_date)
        return 0.0

    best_entry = None
    best_delta = None

    for entry in history:
        ts_str = entry.get("timestamp", "")
        try:
            # Handle both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS" formats
            entry_dt = datetime.date.fromisoformat(ts_str[:10])
        except (ValueError, TypeError):
            continue
        delta = abs((entry_dt - target_dt).days)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_entry = entry

    if best_entry is None:
        return 0.0

    # Only accept matches within 7 days to avoid stale data for large gaps
    if best_delta > 7:
        log.debug(
            "Nearest TVL entry for date %s is %d days away — returning 0.0",
            target_date,
            best_delta,
        )
        return 0.0

    return float(best_entry.get("tvlUsd") or 0.0)


def fetch_pools_at_block(block_number: int) -> list[dict]:
    """
    Return list of pool dicts for block_number in the format expected by graph_builder.

    Steps:
      1. Get the pool list (cached after first call).
      2. Convert block to a date string.
      3. For each pool, fetch its TVL history and find TVL at that date.
      4. Build and return pool dicts.

    Pool dict format:
      {
          "id": pool_id,
          "token0": {"id": token0_addr, "symbol": sym0},
          "token1": {"id": token1_addr, "symbol": sym1},
          "totalValueLockedUSD": tvl_for_date,
          "volumeUSD": volume_for_date,
          "txCount": 0,
          "feeTier": "3000"
      }
    """
    pool_list = fetch_pool_list()
    if not pool_list:
        log.warning("fetch_pools_at_block: empty pool list for block %d", block_number)
        return []

    # Convert block number to date string
    ts = block_to_timestamp(
        block_number,
        ref_block=C.REFERENCE_BLOCK,
        ref_ts=C.REFERENCE_TIMESTAMP,
        block_time=C.BLOCK_TIME_SEC,
    )
    target_date = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    log.debug(
        "fetch_pools_at_block: block %d → date %s, querying %d pools",
        block_number,
        target_date,
        len(pool_list),
    )

    result_pools = []
    for pool_meta in pool_list:
        pool_id = pool_meta.get("pool", "")
        if not pool_id:
            continue

        symbol = pool_meta.get("symbol", "")
        underlying = pool_meta.get("underlyingTokens") or []
        volume_1d = pool_meta.get("volumeUsd1d", 0.0)

        # Parse token addresses and symbols from the pool metadata
        token0_addr = underlying[0].lower() if len(underlying) > 0 else ""
        token1_addr = underlying[1].lower() if len(underlying) > 1 else ""

        # Derive per-token symbols from the combined symbol string (e.g. "USDC-WETH")
        parts = symbol.split("-")
        sym0 = parts[0] if len(parts) > 0 else ""
        sym1 = parts[1] if len(parts) > 1 else ""

        # Fetch historical TVL
        history = fetch_pool_history(pool_id)
        tvl_at_date = get_pool_tvl_at_date(history, target_date)

        # If history fetch failed entirely, fall back to current pool TVL
        if tvl_at_date == 0.0 and not history:
            tvl_at_date = pool_meta.get("tvlUsd", 0.0)

        result_pools.append(
            {
                "id": pool_id,
                "token0": {"id": token0_addr, "symbol": sym0},
                "token1": {"id": token1_addr, "symbol": sym1},
                "totalValueLockedUSD": tvl_at_date,
                "volumeUSD": float(volume_1d),
                "txCount": 0,
                "feeTier": "3000",
            }
        )

    log.info(
        "fetch_pools_at_block: block %d (%s) → %d pools assembled",
        block_number,
        target_date,
        len(result_pools),
    )
    return result_pools


def fetch_pools_for_window(block_start: int, block_end: int) -> list[dict]:
    """
    Fetch pools at the midpoint block of [block_start, block_end).
    Wraps fetch_pools_at_block with window midpoint calculation.
    """
    mid = window_midpoint(block_start, block_end)
    log.debug(
        "DeFi Llama window [%d, %d): querying midpoint block %d",
        block_start,
        block_end,
        mid,
    )
    return fetch_pools_at_block(mid)


def prefetch_pool_histories(delay: float = 0.35) -> int:
    """
    Pre-fetch and cache ALL pool histories sequentially before the parallel pipeline runs.
    This avoids the 403 rate-limit errors that occur when 5 threads hit DeFi Llama simultaneously.

    Should be called ONCE at the start of run_pipeline(). Subsequent calls are near-instant
    because all histories are already cached.

    Parameters
    ----------
    delay : seconds between each HTTP request (default 0.35 — safe for DeFi Llama free tier)

    Returns
    -------
    Number of pool histories successfully fetched/cached.
    """
    import time as _time
    pools = fetch_pool_list()
    total = len(pools)
    ok = 0
    already_cached = 0

    log.info("Prefetching %d pool histories (delay=%.2fs each)…", total, delay)
    log.info("  First run: ~%.0fs | Subsequent runs: instant (cache hit)", total * delay)

    for i, pool in enumerate(pools):
        pool_id = pool["pool"]
        cache_key = f"pool_history_{pool_id}"

        # Skip if already cached
        cached = load_cache(C.CACHE_UNISWAP, cache_key)
        if cached is not None:
            already_cached += 1
            ok += 1
            continue

        history = fetch_pool_history(pool_id)
        if history:
            ok += 1

        # Progress every 50 pools
        if (i + 1) % 50 == 0:
            log.info(
                "  Prefetch progress: %d/%d  ok=%d  already_cached=%d",
                i + 1, total, ok, already_cached,
            )

        _time.sleep(delay)

    log.info(
        "Prefetch complete: %d/%d histories cached (was_cached=%d, newly_fetched=%d)",
        ok, total, already_cached, ok - already_cached,
    )
    return ok
