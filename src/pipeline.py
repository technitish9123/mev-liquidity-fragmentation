"""
Main pipeline: fetch real on-chain data → process → assemble 263-row DataFrame.

Modes:
  "real"      — fetch everything from APIs, use synthetic only as per-window fallback
  "hybrid"    — real data where available, synthetic fill for gaps
  "synthetic" — delegate entirely to legacy data_generator.py

Usage:
  from src.pipeline import run_pipeline
  df = run_pipeline()                      # full 263 windows, ~35 min first run
  df = run_pipeline(n_windows=5)           # quick smoke test
  df = run_pipeline(mode="synthetic")      # legacy synthetic path
"""

import os, sys, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.utils import setup_cache_dirs, block_to_timestamp

# Fetchers
from src.fetchers.zeromev          import fetch_mev_window
from src.fetchers.uniswap_subgraph import fetch_pools_for_window, prefetch_pool_histories
from src.fetchers.coingecko        import fetch_eth_prices, prices_for_window
from src.fetchers.etherscan        import fetch_gas_window

# Processors
from src.processors.graph_builder    import compute_topology_metrics
from src.processors.mev_processor    import (compute_window_mev_raw,
                                              normalize_intensities,
                                              smooth_intensities,
                                              classify_regime)
from src.processors.market_processor import (compute_volatility,
                                              compute_avg_gas,
                                              compute_volume)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_row(window_id: int, block_start: int) -> dict:
    """Return a row of synthetic values matching the DataFrame schema."""
    rng = np.random.default_rng(C.SEED + window_id)
    mev = float(np.clip(rng.normal(C.MEV_MEAN if hasattr(C, "MEV_MEAN") else 0.034,
                                    0.028), 0.001, 0.20))
    mod = float(np.clip(rng.normal(0.45, 0.20), 0.0, 1.0))
    return {
        "modularity":          mod,
        "spectral_gap":        float(np.clip(rng.normal(0.15, 0.07), 0.0, 0.5)),
        "eff_connectivity":    float(np.clip(rng.normal(0.62, 0.20), 0.0, 1.0)),
        "avg_path_length":     float(np.clip(rng.normal(2.8, 0.6), 1.0, 8.0)),
        "giant_component_pct": float(np.clip(rng.normal(78, 15), 10.0, 100.0)),
        "avg_eff_resistance":  float(np.clip(rng.normal(3.2, 1.5), 0.0, 20.0)),
        "nodes":               int(np.clip(rng.integers(140, 240), 10, 500)),
        "edges":               int(np.clip(rng.integers(350, 800), 10, 2000)),
        "active_pools":        int(np.clip(rng.integers(200, 500), 10, 1000)),
        "volatility":          float(np.clip(rng.normal(0.031, 0.019), 0.005, 0.12)),
        "gas_price_gwei":      float(np.clip(rng.normal(34.2, 22.7), 1.0, 200.0)),
        "volume_usd_m":        float(np.clip(rng.normal(842, 431), 50.0, 5000.0)),
        "mev_raw":             mev,
        "data_source":         "synthetic",
    }


# ── Per-window worker ─────────────────────────────────────────────────────────

def _process_window(window_id: int, block_start: int, block_end: int,
                    prices_df: pd.DataFrame, mode: str) -> dict:
    """
    Fetch + process one window. Returns a partial dict (without mev_intensity —
    that requires normalization across all windows done in post-processing).
    """
    row = {"window_id": window_id, "block_start": block_start, "block_end": block_end}
    sources = []

    # ── Topology (Uniswap subgraph) ──
    pools = []
    try:
        pools = fetch_pools_for_window(block_start, block_end)
        if pools:
            topo = compute_topology_metrics(pools)
            row.update(topo)
            row["volume_usd_m"] = compute_volume(pools)
            sources.append("real")
        else:
            raise ValueError("empty pools")
    except Exception as e:
        log.warning(f"[W{window_id:03d}] topology fallback: {e}")
        syn = _synthetic_row(window_id, block_start)
        for k in ("modularity","spectral_gap","eff_connectivity","avg_path_length",
                  "giant_component_pct","avg_eff_resistance","nodes","edges",
                  "active_pools","volume_usd_m"):
            row[k] = syn[k]
        sources.append("synthetic")

    # ── MEV (ZeroMEV) ──
    try:
        mev_blocks = fetch_mev_window(block_start, block_end)
        row["mev_raw"] = compute_window_mev_raw(mev_blocks)
        sources.append("real")
    except Exception as e:
        log.warning(f"[W{window_id:03d}] MEV fallback: {e}")
        row["mev_raw"] = _synthetic_row(window_id, block_start)["mev_raw"]
        sources.append("synthetic")

    # ── Gas (Etherscan) ──
    try:
        gas_vals = fetch_gas_window(block_start, block_end)
        row["gas_price_gwei"] = compute_avg_gas(gas_vals)
        sources.append("real")
    except Exception as e:
        log.warning(f"[W{window_id:03d}] gas fallback: {e}")
        row["gas_price_gwei"] = C.GAS_MEAN
        sources.append("synthetic")

    # ── Volatility (CoinGecko) ──
    try:
        window_prices = prices_for_window(prices_df, block_start, block_end)
        row["volatility"] = compute_volatility(window_prices)
        sources.append("real")
    except Exception as e:
        log.warning(f"[W{window_id:03d}] volatility fallback: {e}")
        row["volatility"] = _synthetic_row(window_id, block_start)["volatility"]
        sources.append("synthetic")

    # ── Data source tag ──
    real_count = sources.count("real")
    if real_count == len(sources):
        row["data_source"] = "real"
    elif real_count == 0:
        row["data_source"] = "synthetic"
    else:
        row["data_source"] = "partial"

    return row


# ── Post-processing ───────────────────────────────────────────────────────────

def _derive_user_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive modeled user-facing columns from real topology metrics."""
    df = df.copy()
    df["avg_slippage_pct"]     = (0.30
                                   + 0.85 * df["modularity"]
                                   + 0.15 * (1 - df["eff_connectivity"])
                                  ).clip(0.1, 5.0)
    df["avg_route_hops"]       = (1.2 + 2.8 * df["avg_path_length"] / 5.0).clip(1.0, 8.0)
    df["failed_routes_pct"]    = (25 * (1 - df["eff_connectivity"]) - 3).clip(0.0, 80.0)
    df["median_exec_cost_usd"] = (3.5 + 12 * df["avg_slippage_pct"] / 100
                                      * df["volume_usd_m"].clip(lower=1)).clip(1.0, 50.0)
    # Builder HHI proxy: positively correlated with MEV intensity + noise
    rng = np.random.default_rng(C.SEED)
    df["hhi_builders"] = (0.15 + 0.4 * df["mev_intensity"]
                          + rng.normal(0, 0.03, len(df))).clip(0.05, 0.9)
    return df


def _assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Classify mev_intensity into low/medium/high using percentile thresholds."""
    low_thresh  = df["mev_intensity"].quantile(C.MEV_REGIME_LOW_PERCENTILE  / 100)
    high_thresh = df["mev_intensity"].quantile(C.MEV_REGIME_HIGH_PERCENTILE / 100)
    df["mev_regime"] = df["mev_intensity"].apply(
        lambda x: classify_regime(x, low_thresh, high_thresh)
    )
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    mode:       str  = None,
    block_start: int = None,
    block_end:   int = None,
    n_windows:   int = None,
    use_cache:   bool = True,
    save:        bool = True,
) -> pd.DataFrame:
    """
    Run the full data pipeline.

    Parameters
    ----------
    mode        : "real" | "hybrid" | "synthetic" (default: C.DATA_MODE)
    block_start : first block (default: C.BLOCK_START)
    block_end   : last block  (default: C.BLOCK_END)
    n_windows   : number of windows to process (default: C.N_WINDOWS)
                  Set to a small number (e.g. 5) for smoke testing.
    use_cache   : use cached API responses (default: True)
    save        : save output CSV to data/processed/ (default: True)

    Returns
    -------
    pd.DataFrame with 23+ columns matching the schema in
    .claude/skills/defi-data-pipeline/references/dataframe-schema.md
    """
    mode        = mode        or C.DATA_MODE
    block_start = block_start or C.BLOCK_START
    block_end   = block_end   or C.BLOCK_END
    n_windows   = n_windows   or C.N_WINDOWS

    # ── Synthetic shortcut ──
    if mode == "synthetic":
        log.info("Mode=synthetic — delegating to data_generator")
        from src.data_generator import generate_dataset
        return generate_dataset()

    setup_cache_dirs(C)
    t0 = time.time()
    log.info(f"Pipeline start | mode={mode} | windows={n_windows} "
             f"| blocks {block_start:,}→{block_end:,}")

    # ── Define windows ──
    windows = [
        (i, block_start + i * C.BLOCKS_PER_WINDOW,
             block_start + (i + 1) * C.BLOCKS_PER_WINDOW)
        for i in range(n_windows)
    ]

    # ── Pre-fetch all pool histories sequentially (avoids DeFi Llama rate limits) ──
    if mode != "synthetic":
        log.info("Pre-fetching pool histories (first run: ~105s, subsequent: instant)…")
        prefetch_pool_histories(delay=0.35)

    # ── Fetch ETH prices once (single API call) ──
    log.info("Fetching ETH/USD prices from CryptoCompare…")
    try:
        prices_df = fetch_eth_prices(block_start, block_end)
        log.info(f"  → {len(prices_df)} daily price points")
    except Exception as e:
        log.warning(f"CoinGecko failed: {e} — using empty price series")
        prices_df = pd.DataFrame(columns=["timestamp_ms", "price_usd", "date"])

    # ── Process windows in parallel ──
    log.info(f"Processing {n_windows} windows with {C.N_FETCH_THREADS} threads…")
    rows = [None] * n_windows

    with ThreadPoolExecutor(max_workers=C.N_FETCH_THREADS) as pool:
        futures = {
            pool.submit(_process_window, i, bs, be, prices_df, mode): i
            for i, bs, be in windows
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                log.error(f"[W{idx:03d}] unhandled error: {e}")
                i, bs, _ = windows[idx]
                rows[idx] = _synthetic_row(i, bs)
                rows[idx].update({"window_id": i,
                                  "block_start": bs,
                                  "block_end": bs + C.BLOCKS_PER_WINDOW,
                                  "data_source": "synthetic"})
            done += 1
            if done % 25 == 0 or done == n_windows:
                pct = done / n_windows * 100
                elapsed = time.time() - t0
                eta = (elapsed / done) * (n_windows - done)
                log.info(f"  Progress: {done}/{n_windows} ({pct:.0f}%) "
                         f"| elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    df = pd.DataFrame([r for r in rows if r is not None])

    # ── Normalize MEV intensity ──
    raw_vals = df["mev_raw"].tolist()
    normed   = normalize_intensities(raw_vals)
    smoothed = smooth_intensities(normed, W=10)
    df["mev_intensity"] = smoothed
    df.drop(columns=["mev_raw"], inplace=True)

    # ── Regime classification ──
    df = _assign_regimes(df)

    # ── Derived user-facing metrics ──
    df = _derive_user_metrics(df)

    # ── Column ordering ──
    ordered_cols = [
        "window_id", "block_start", "block_end",
        "mev_intensity", "mev_regime",
        "modularity", "spectral_gap", "eff_connectivity",
        "avg_path_length", "giant_component_pct", "avg_eff_resistance",
        "volatility", "gas_price_gwei", "volume_usd_m",
        "active_pools", "nodes", "edges",
        "avg_slippage_pct", "avg_route_hops", "failed_routes_pct",
        "median_exec_cost_usd", "hhi_builders", "data_source",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[ordered_cols]

    # ── Summary ──
    elapsed = time.time() - t0
    real_pct = (df["data_source"] == "real").mean() * 100
    part_pct = (df["data_source"] == "partial").mean() * 100
    syn_pct  = (df["data_source"] == "synthetic").mean() * 100
    log.info(f"Pipeline complete in {elapsed:.0f}s")
    log.info(f"  Data provenance — real: {real_pct:.0f}%  "
             f"partial: {part_pct:.0f}%  synthetic: {syn_pct:.0f}%")
    log.info(f"  MEV intensity: mean={df['mev_intensity'].mean():.4f}  "
             f"std={df['mev_intensity'].std():.4f}  "
             f"max={df['mev_intensity'].max():.4f}")
    log.info(f"  Modularity: mean={df['modularity'].mean():.3f}  "
             f"std={df['modularity'].std():.3f}")

    # ── Save ──
    if save:
        os.makedirs(C.PROCESSED_DIR, exist_ok=True)
        out = os.path.join(C.PROCESSED_DIR, "windows_500_real.csv")
        df.to_csv(out, index=False)
        log.info(f"  Saved → {out}")

    return df
