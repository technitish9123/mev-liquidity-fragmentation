"""
Etherscan V2 gas price fetcher.
API: https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_getBlockByNumber&tag={hex}&boolean=false&apikey={KEY}
Extracts baseFeePerGas (hex wei) → Gwei. All blocks >15M are post-EIP-1559.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.utils import rate_limited_get, load_cache, save_cache, sample_blocks
import logging

log = logging.getLogger(__name__)

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"


def fetch_block_gas_gwei(block_number: int) -> float | None:
    """
    Fetch baseFeePerGas for block_number via Etherscan V2 API.
    Returns value in Gwei, or None on failure.
    Cache key: f"block_{block_number}" in C.CACHE_ETHERSCAN — saved as float.
    """
    cache_key = f"block_{block_number}"
    cached = load_cache(C.CACHE_ETHERSCAN, cache_key)
    if cached is not None:
        log.debug("Etherscan cache hit: block %d", block_number)
        return float(cached)

    hex_block = hex(block_number)
    params = {
        "chainid": "1",
        "module": "proxy",
        "action": "eth_getBlockByNumber",
        "tag": hex_block,
        "boolean": "false",
        "apikey": C.ETHERSCAN_API_KEY,
    }

    response = rate_limited_get(
        ETHERSCAN_V2_URL,
        params=params,
        delay=C.REQUEST_DELAY_SEC,
        max_retries=C.MAX_RETRIES,
    )

    if response is None:
        log.warning("Etherscan V2 returned None for block %d", block_number)
        return None

    result = response.get("result")
    if not result or not isinstance(result, dict):
        log.warning(
            "Etherscan V2 unexpected result shape for block %d: %s",
            block_number,
            response,
        )
        return None

    base_fee_hex = result.get("baseFeePerGas")
    if base_fee_hex is None:
        log.warning(
            "baseFeePerGas missing for block %d (pre-EIP-1559?)", block_number
        )
        return None

    try:
        gwei = int(base_fee_hex, 16) / 1e9
    except (ValueError, TypeError) as exc:
        log.warning(
            "Could not parse baseFeePerGas '%s' for block %d: %s",
            base_fee_hex,
            block_number,
            exc,
        )
        return None

    save_cache(C.CACHE_ETHERSCAN, cache_key, gwei)
    log.debug("Etherscan V2 block %d: baseFee=%.3f Gwei", block_number, gwei)
    return gwei


def fetch_gas_window(
    block_start: int,
    block_end: int,
    samples: int = None,
) -> list[float]:
    """
    Sample `samples` blocks from [block_start, block_end), fetch gas for each.
    Returns list of valid (non-None) Gwei values.
    samples defaults to C.ETHERSCAN_SAMPLES_PER_WINDOW.
    """
    if samples is None:
        samples = C.ETHERSCAN_SAMPLES_PER_WINDOW

    blocks = sample_blocks(block_start, block_end, samples, seed=C.SEED)
    results = []
    for blk in blocks:
        try:
            gwei = fetch_block_gas_gwei(blk)
            if gwei is not None:
                results.append(gwei)
        except Exception as exc:
            log.warning("fetch_block_gas_gwei failed for block %d: %s", blk, exc)

    log.info(
        "Etherscan V2 window [%d, %d): fetched %d/%d gas samples",
        block_start,
        block_end,
        len(results),
        len(blocks),
    )
    return results
