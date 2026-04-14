"""
MEV-Boost relay block_value fetcher (replaces ZeroMEV which is down).

Uses the ultrasound.money MEV-Boost relay API which has 2025+ data:
  GET https://relay.ultrasound.money/relay/v1/data/bidtraces/proposer_payload_delivered?block_number={N}

Returns block_value in wei (total ETH paid to proposer = base fee rebate + priority fees + MEV).
This is a well-established MEV intensity proxy: higher block_value = more MEV competition.

Falls back to agnostic-relay.net and bloxroute if ultrasound is empty.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.utils import rate_limited_get, load_cache, save_cache, sample_blocks
import logging

log = logging.getLogger(__name__)

RELAY_URLS = [
    "https://relay.ultrasound.money/relay/v1/data/bidtraces/proposer_payload_delivered",
    "https://agnostic-relay.net/relay/v1/data/bidtraces/proposer_payload_delivered",
    "https://bloxroute.max-profit.blxrbdn.com/relay/v1/data/bidtraces/proposer_payload_delivered",
]

# Rough ETH/USD price used for USD conversion until pipeline calibrates it.
_ROUGH_ETH_USD = 3200.0


def fetch_block_mev_eth(block_number: int) -> float:
    """
    Fetch block_value in ETH for block_number from MEV-Boost relays.

    Tries RELAY_URLS in order; takes max value found across all relays.
    Returns 0.0 if block is not found in any relay (vanilla/non-MEV block).
    Cache key: f"relay_block_{block_number}" in C.CACHE_ZEROMEV — saved as float (ETH).
    """
    cache_key = f"relay_block_{block_number}"
    cached = load_cache(C.CACHE_ZEROMEV, cache_key)
    if cached is not None:
        log.debug("MEV relay cache hit: block %d", block_number)
        return float(cached)

    params = {"block_number": block_number}
    max_value_eth = 0.0

    for relay_url in RELAY_URLS:
        try:
            response = rate_limited_get(
                relay_url,
                params=params,
                delay=C.REQUEST_DELAY_SEC,
                max_retries=C.MAX_RETRIES,
            )
        except Exception as exc:
            log.warning("Relay request exception for %s block %d: %s", relay_url, block_number, exc)
            continue

        if response is None:
            log.debug("Relay %s returned None for block %d", relay_url, block_number)
            continue

        # Each relay returns a list of payload objects
        if not isinstance(response, list):
            log.debug("Relay %s returned non-list for block %d: %s", relay_url, block_number, type(response))
            continue

        for payload in response:
            val_str = payload.get("value")
            if val_str is None:
                continue
            try:
                val_eth = int(val_str) / 1e18
                if val_eth > max_value_eth:
                    max_value_eth = val_eth
            except (ValueError, TypeError) as exc:
                log.warning(
                    "Could not parse block_value '%s' from relay %s block %d: %s",
                    val_str,
                    relay_url,
                    block_number,
                    exc,
                )

    save_cache(C.CACHE_ZEROMEV, cache_key, max_value_eth)
    log.debug("MEV relay block %d: block_value=%.6f ETH", block_number, max_value_eth)
    return max_value_eth


def fetch_mev_window(
    block_start: int,
    block_end: int,
    samples: int = None,
) -> list[dict]:
    """
    Sample `samples` blocks from [block_start, block_end), fetch MEV block_value for each.
    samples defaults to C.ZEROMEV_SAMPLES_PER_WINDOW.

    Returns list of dicts:
        {"block_number": N, "extractor_profit_usd": block_value_eth * eth_usd_price}

    Non-MEV-Boost blocks return 0.0 — these are legitimate observations, not failures.
    ETH price uses rough constant _ROUGH_ETH_USD; pipeline calibrates with real prices.
    """
    if samples is None:
        samples = C.ZEROMEV_SAMPLES_PER_WINDOW

    blocks = sample_blocks(block_start, block_end, samples, seed=C.SEED)
    results = []
    for blk in blocks:
        try:
            val_eth = fetch_block_mev_eth(blk)
            results.append(
                {
                    "block_number": blk,
                    "extractor_profit_usd": val_eth * _ROUGH_ETH_USD,
                }
            )
        except Exception as exc:
            log.warning("fetch_block_mev_eth failed for block %d: %s", blk, exc)

    log.info(
        "MEV relay window [%d, %d): fetched %d/%d blocks",
        block_start,
        block_end,
        len(results),
        len(blocks),
    )
    return results
