"""
Configuration constants for the MEV Liquidity Fragmentation project.

Paper: "Dynamic Graph Rewiring in Decentralized Exchange Networks:
        A Study of MEV-Induced Liquidity Fragmentation"
Conference: NBC '26, August 21-22, 2026, NTU Singapore
"""

import os

# ============================================================
# PIPELINE MODE
# ============================================================
DATA_MODE = "real"        # "real" | "hybrid" | "synthetic"
PROTOCOL  = "uniswap_v3"  # Only Uniswap V3 mainnet

# ============================================================
# BLOCK RANGE — Jan 1 2025 → Dec 31 2025
# ============================================================
BLOCK_START        = 21_700_000   # ≈ Jan 1, 2025
BLOCK_END          = 24_328_000   # ≈ Dec 31, 2025
BLOCKS_PER_WINDOW  = 10_000       # ≈ 1.4 days per window
N_WINDOWS          = 263          # (BLOCK_END - BLOCK_START) // BLOCKS_PER_WINDOW
BLOCK_TIME_SEC     = 12           # seconds per block

# Reference for block→timestamp conversion
REFERENCE_BLOCK     = 21_700_000
REFERENCE_TIMESTAMP = 1735689600  # 2025-01-01 00:00:00 UTC

# ============================================================
# API ENDPOINTS
# ============================================================
ZEROMEV_BASE_URL    = "https://data.zeromev.org/v1"
UNISWAP_V3_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
COINGECKO_BASE_URL  = "https://api.coingecko.com/api/v3"
ETHERSCAN_BASE_URL  = "https://api.etherscan.io/api"

# Set via environment variable or paste key here
ETHERSCAN_API_KEY   = os.environ.get("ETHERSCAN_API_KEY", "")

# ============================================================
# FETCHER SETTINGS (tuned for speed)
# ============================================================
ZEROMEV_SAMPLES_PER_WINDOW    = 10   # blocks sampled per window for MEV
ETHERSCAN_SAMPLES_PER_WINDOW  = 5    # blocks sampled per window for gas
MAX_POOLS_PER_WINDOW          = 300  # top pools by TVL per snapshot
UNISWAP_MIN_TVL_USD           = 10_000  # filter dust pools

N_FETCH_THREADS   = 3     # parallel threads (3 avoids Etherscan 3/sec rate limit)
REQUEST_DELAY_SEC = 0.5   # polite delay between calls (Etherscan free = 3/sec)
MAX_RETRIES       = 3     # retries on 429/5xx

# ============================================================
# CACHE DIRECTORIES
# ============================================================
CACHE_DIR     = "data/cache"
PROCESSED_DIR = "data/processed"

CACHE_ZEROMEV   = os.path.join(CACHE_DIR, "zeromev")
CACHE_UNISWAP   = os.path.join(CACHE_DIR, "uniswap")
CACHE_COINGECKO = os.path.join(CACHE_DIR, "coingecko")
CACHE_ETHERSCAN = os.path.join(CACHE_DIR, "etherscan")

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
W_MIN             = 50_000   # min edge weight (TVL $USD) for C_eff path check
SEED              = 42
LOUVAIN_RESOLUTION = 1.0

# ============================================================
# MEV REGIME THRESHOLDS
# ============================================================
MEV_REGIME_LOW_PERCENTILE  = 33
MEV_REGIME_HIGH_PERCENTILE = 67

# ============================================================
# PHASE TRANSITION — will be updated after real fit
# ============================================================
# Phase transition fit failed on real data (R²=0.29, no sigmoid)
# These are initial guesses for curve_fit; actual fit is degenerate
I_STAR = 0.30   # approximate median MEV intensity
KAPPA  = 5.0
Q_MIN  = 0.20
Q_MAX  = 0.10

# ============================================================
# SPECTRAL THRESHOLD
# ============================================================
LAMBDA2_THRESHOLD = 0.15

# ============================================================
# THEORETICAL PARAMETERS (Section 5, Remark 1)
# ============================================================
SIGMA_P           = 0.02
RHO_INTER         = 0.3
RHO_INTRA         = 0.8
MU_REPLENISHMENT  = 0.001
W0_WMIN_RATIO     = 10

# ============================================================
# REGRESSION PARAMETERS — will be updated after real fit
# ============================================================
OLS_BETA_MEV      = -0.015
OLS_BETA_VOL      = -0.027
IV_BETA_MEV       = -0.015
IV_FIRST_STAGE_F  = 0.0  # IV not primary spec

# ============================================================
# FIGURE SETTINGS
# ============================================================
FIGURE_DPI          = 300
FIGURE_FORMAT       = "png"
ACM_SINGLE_COL_WIDTH = 3.3
ACM_DOUBLE_COL_WIDTH = 7.0
ACM_COL_HEIGHT       = 2.5

# MEV axis limits for figures (will auto-adjust after real data)
MEV_MAX = 3.20  # real data max ~3.1 (normalized by 95th percentile)
