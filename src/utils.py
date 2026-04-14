"""Shared utilities: caching, rate-limited HTTP, block↔timestamp conversion."""
import os
import json
import time
import logging
import math
import random
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ── Cache ────────────────────────────────────────────────────────────────────

def cache_path(cache_dir: str, key: str) -> Path:
    """Return Path for cache file. Creates parent dirs."""
    p = Path(cache_dir) / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_cache(cache_dir: str, key: str):
    """Return parsed JSON if cache file exists, else None."""
    p = cache_path(cache_dir, key)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cache read failed for %s/%s: %s", cache_dir, key, exc)
            return None
    return None


def save_cache(cache_dir: str, key: str, data) -> None:
    """Write data as JSON to cache file."""
    p = cache_path(cache_dir, key)
    try:
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except OSError as exc:
        log.warning("Cache write failed for %s/%s: %s", cache_dir, key, exc)


# ── HTTP ─────────────────────────────────────────────────────────────────────

def rate_limited_get(
    url: str,
    params: dict = None,
    delay: float = 0.3,
    max_retries: int = 3,
    headers: dict = None,
) -> dict | None:
    """GET with delay + exponential-backoff retry. Returns parsed JSON or None."""
    time.sleep(delay)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning(
                    "HTTP %d for %s (attempt %d/%d), retrying in %ds",
                    resp.status_code, url, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            # 4xx that is not 429 — no point retrying
            log.warning("HTTP %d for %s — giving up", resp.status_code, url)
            return None
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning(
                "Request error for %s (attempt %d/%d): %s, retrying in %ds",
                url, attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    log.error("Permanent failure after %d attempts: %s", max_retries, url)
    return None


def graphql_post(
    url: str,
    query: str,
    variables: dict = None,
    delay: float = 0.5,
    max_retries: int = 3,
) -> dict | None:
    """POST GraphQL query. Returns response['data'] or None."""
    time.sleep(delay)
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code == 200:
                body = resp.json()
                if "errors" in body:
                    log.warning("GraphQL errors from %s: %s", url, body["errors"])
                    # Some errors are retriable (rate-limit / indexing), others not.
                    # Be conservative and retry.
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                return body.get("data")
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning(
                    "HTTP %d from GraphQL %s (attempt %d/%d), retrying in %ds",
                    resp.status_code, url, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            log.warning("GraphQL HTTP %d from %s — giving up", resp.status_code, url)
            return None
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning(
                "GraphQL request error %s (attempt %d/%d): %s, retrying in %ds",
                url, attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    log.error("Permanent GraphQL failure after %d attempts: %s", max_retries, url)
    return None


# ── Block / Timestamp ────────────────────────────────────────────────────────

def block_to_timestamp(
    block: int,
    ref_block: int = 21_700_000,
    ref_ts: int = 1_735_689_600,
    block_time: int = 12,
) -> int:
    """Approximate Unix timestamp for a block number."""
    return ref_ts + (block - ref_block) * block_time


def timestamp_to_block(
    ts: int,
    ref_block: int = 21_700_000,
    ref_ts: int = 1_735_689_600,
    block_time: int = 12,
) -> int:
    """Approximate block number for a Unix timestamp."""
    return ref_block + (ts - ref_ts) // block_time


# ── Window helpers ───────────────────────────────────────────────────────────

def window_midpoint(block_start: int, block_end: int) -> int:
    """Return midpoint block of a window."""
    return (block_start + block_end) // 2


def sample_blocks(
    block_start: int, block_end: int, n: int, seed: int = 42
) -> list[int]:
    """Return n uniformly sampled block numbers from [block_start, block_end)."""
    rng = random.Random(seed)
    span = block_end - block_start
    if span <= 0:
        return []
    if n >= span:
        return list(range(block_start, block_end))
    step = span / n
    blocks = []
    for i in range(n):
        low = block_start + int(i * step)
        high = block_start + int((i + 1) * step)
        high = min(high, block_end - 1)
        blocks.append(rng.randint(low, high))
    return blocks


def setup_cache_dirs(config) -> None:
    """Create all cache directories from config."""
    dirs = [
        getattr(config, attr)
        for attr in dir(config)
        if attr.startswith("CACHE_") and isinstance(getattr(config, attr), str)
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Cache dir ready: %s", d)
