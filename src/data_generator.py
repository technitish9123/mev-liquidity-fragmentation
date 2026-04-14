"""
Synthetic data generator calibrated to the paper's reported statistics.

Generates 500 observation windows with correlated MEV intensity,
volatility, gas prices, volume, and topology metrics that reproduce
the paper's Tables 1-3 and phase transition behavior (Eq. 31).
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def _truncated_normal(mean, std, low, high, size, rng):
    """Sample from truncated normal distribution."""
    a = (low - mean) / std
    b = (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)


def logistic(I, I_star=C.I_STAR, kappa=C.KAPPA, Q_min=C.Q_MIN, Q_max=C.Q_MAX):
    """Phase transition logistic function (Eq. 31)."""
    return Q_max / (1.0 + np.exp(-kappa * (I - I_star))) + Q_min


def generate_dataset(seed=C.SEED):
    """
    Generate the full synthetic dataset of 500 windows.

    Returns:
        pd.DataFrame with columns for all variables in Tables 1-3
    """
    rng = np.random.default_rng(seed)
    N = C.N_WINDOWS

    # --- Step 1: Generate correlated base variables ---
    # MEV intensity and volatility are correlated (paper shows volatility
    # Granger-causes MEV, Table 5: F=12.87)
    correlation_mev_vol = 0.55

    # Generate correlated normals
    cov = np.array([[1.0, correlation_mev_vol],
                    [correlation_mev_vol, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=N)

    # Transform to truncated distributions via CDF mapping
    mev_raw = _truncated_normal(C.MEV_MEAN, C.MEV_STD, C.MEV_MIN, C.MEV_MAX, N, rng)
    vol_raw = _truncated_normal(C.VOL_MEAN, C.VOL_STD, C.VOL_MIN, C.VOL_MAX, N, rng)

    # Sort both by the correlated normal to induce correlation structure
    mev_order = np.argsort(z[:, 0])
    vol_order = np.argsort(z[:, 1])

    mev_sorted = np.sort(mev_raw)
    vol_sorted = np.sort(vol_raw)

    mev_intensity = np.empty(N)
    volatility = np.empty(N)
    mev_intensity[mev_order] = mev_sorted
    volatility[vol_order] = vol_sorted

    # Add temporal autocorrelation with regime-like bursts
    # Use mild smoothing to keep variance high enough
    for i in range(1, N):
        mev_intensity[i] = 0.85 * mev_intensity[i] + 0.15 * mev_intensity[i - 1]
        volatility[i] = 0.80 * volatility[i] + 0.20 * volatility[i - 1]

    # Inject high-MEV burst episodes (simulates real market stress events)
    n_bursts = 8
    burst_indices = rng.choice(np.arange(20, N - 20), size=n_bursts, replace=False)
    for idx in burst_indices:
        burst_len = rng.integers(5, 15)
        burst_level = rng.uniform(0.10, C.MEV_MAX)
        for j in range(max(0, idx), min(N, idx + burst_len)):
            mev_intensity[j] = max(mev_intensity[j], burst_level * rng.uniform(0.7, 1.0))
            volatility[j] = max(volatility[j], C.VOL_MEAN + C.VOL_STD * rng.uniform(1.0, 2.5))

    # Re-scale to match target statistics
    mev_intensity = (mev_intensity - mev_intensity.mean()) / mev_intensity.std() * C.MEV_STD + C.MEV_MEAN
    mev_intensity = np.clip(mev_intensity, C.MEV_MIN, C.MEV_MAX)

    volatility = (volatility - volatility.mean()) / volatility.std() * C.VOL_STD + C.VOL_MEAN
    volatility = np.clip(volatility, C.VOL_MIN, C.VOL_MAX)

    # --- Step 2: Generate other control variables ---
    gas_price = _truncated_normal(C.GAS_MEAN, C.GAS_STD, C.GAS_MIN, C.GAS_MAX, N, rng)
    # Gas is somewhat correlated with volatility
    gas_price = 0.7 * gas_price + 0.3 * (
        (volatility - C.VOL_MEAN) / C.VOL_STD * C.GAS_STD + C.GAS_MEAN
    )
    gas_price = np.clip(gas_price, C.GAS_MIN, C.GAS_MAX)

    volume = _truncated_normal(C.VOLUME_MEAN, C.VOLUME_STD, C.VOLUME_MIN, C.VOLUME_MAX, N, rng)
    active_pools = _truncated_normal(C.POOLS_MEAN, C.POOLS_STD, C.POOLS_MIN, C.POOLS_MAX, N, rng).astype(int)
    nodes = _truncated_normal(C.NODES_MEAN, C.NODES_STD, C.NODES_MIN, C.NODES_MAX, N, rng).astype(int)
    edges = _truncated_normal(C.EDGES_MEAN, C.EDGES_STD, C.EDGES_MIN, C.EDGES_MAX, N, rng).astype(int)

    # --- Step 3: Compute topology metrics via phase transition model ---
    # Modularity follows logistic function (Eq. 31) + noise
    modularity_base = logistic(mev_intensity)
    # Add volatility contribution (β₂=2.17 from Table 4)
    vol_contribution = (C.OLS_BETA_VOL / C.OLS_BETA_MEV) * (volatility - C.VOL_MEAN) * 0.5
    modularity = modularity_base + vol_contribution + rng.normal(0, 0.05, N)
    # Ensure we hit the paper's reported range: low~0.31, high~0.97
    modularity = np.clip(modularity, 0.15, 0.99)

    # Spectral gap inversely related to modularity
    # Low MEV: 0.23, High MEV: 0.09 (Table 2)
    # Linear mapping: modularity 0.28->0.26, modularity 0.96->0.06
    mod_norm = (modularity - 0.20) / (0.99 - 0.20)  # 0-1 normalized
    spectral_gap = 0.28 - 0.22 * mod_norm
    spectral_gap += rng.normal(0, 0.02, N)
    spectral_gap = np.clip(spectral_gap, 0.02, 0.35)

    # Effective connectivity
    eff_connectivity = 0.95 - 0.55 * (modularity - C.Q_MIN) / (C.Q_MAX)
    eff_connectivity += rng.normal(0, 0.03, N)
    eff_connectivity = np.clip(eff_connectivity, 0.2, 0.95)

    # Path length
    path_length = 1.8 + 1.5 * (modularity - C.Q_MIN) / (C.Q_MAX)
    path_length += rng.normal(0, 0.15, N)
    path_length = np.clip(path_length, 1.2, 5.0)

    # Giant component fraction
    giant_component = 0.97 - 0.38 * (modularity - C.Q_MIN) / (C.Q_MAX)
    giant_component += rng.normal(0, 0.03, N)
    giant_component = np.clip(giant_component, 0.4, 0.99)

    # Effective resistance
    eff_resistance = 1.2 + 4.0 * (modularity - C.Q_MIN) / (C.Q_MAX)
    eff_resistance += rng.normal(0, 0.4, N)
    eff_resistance = np.clip(eff_resistance, 0.5, 8.0)

    # --- Step 4: User-facing metrics ---
    slippage = 0.35 + 0.25 * (modularity - C.Q_MIN) / (C.Q_MAX)
    slippage += rng.normal(0, 0.05, N)
    slippage = np.clip(slippage, 0.1, 1.0)

    route_hops = 1.5 + 0.9 * (modularity - C.Q_MIN) / (C.Q_MAX)
    route_hops += rng.normal(0, 0.12, N)
    route_hops = np.clip(route_hops, 1.0, 4.0)

    failed_routes = 1.0 + 14.0 * (modularity - C.Q_MIN) / (C.Q_MAX)
    failed_routes += rng.normal(0, 1.5, N)
    failed_routes = np.clip(failed_routes, 0.1, 25.0)

    execution_cost = 3.0 + 7.0 * (modularity - C.Q_MIN) / (C.Q_MAX)
    execution_cost += rng.normal(0, 1.0, N)
    execution_cost = np.clip(execution_cost, 1.0, 15.0)

    # --- Step 5: Block numbers ---
    block_numbers = np.linspace(C.BLOCK_START, C.BLOCK_END, N, dtype=int)

    # --- Step 6: MEV regime classification ---
    p33 = np.percentile(mev_intensity, 33)
    p67 = np.percentile(mev_intensity, 67)
    mev_regime = np.where(mev_intensity < p33, 'low',
                          np.where(mev_intensity < p67, 'medium', 'high'))

    # --- Step 7: Instrument variable (HHI) ---
    # Builder concentration correlated with MEV (first-stage F=28.4)
    hhi = 0.15 + 0.5 * (mev_intensity / C.MEV_MAX) + rng.normal(0, 0.05, N)
    hhi = np.clip(hhi, 0.05, 0.8)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        'window_id': np.arange(N),
        'block_start': block_numbers,
        'block_end': block_numbers + C.BLOCKS_PER_WINDOW,
        'mev_intensity': mev_intensity,
        'volatility': volatility,
        'gas_price': gas_price,
        'volume_usd_m': volume,
        'active_pools': active_pools,
        'nodes': nodes,
        'edges': edges,
        'modularity': modularity,
        'spectral_gap': spectral_gap,
        'eff_connectivity': eff_connectivity,
        'avg_path_length': path_length,
        'giant_component_frac': giant_component,
        'avg_eff_resistance': eff_resistance,
        'avg_slippage_pct': slippage,
        'avg_route_hops': route_hops,
        'failed_routes_pct': failed_routes,
        'median_exec_cost_usd': execution_cost,
        'mev_regime': mev_regime,
        'builder_hhi': hhi,
    })

    return df


if __name__ == '__main__':
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'synthetic', 'windows_500.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} windows -> {out_path}")
    print(f"\nMEV Intensity: mean={df['mev_intensity'].mean():.3f}, "
          f"std={df['mev_intensity'].std():.3f}")
    print(f"Modularity: mean={df['modularity'].mean():.3f}, "
          f"std={df['modularity'].std():.3f}")
    print(f"Spectral Gap: mean={df['spectral_gap'].mean():.3f}, "
          f"std={df['spectral_gap'].std():.3f}")

    # Regime stats
    for regime in ['low', 'high']:
        sub = df[df['mev_regime'] == regime]
        print(f"\n{regime.upper()} MEV ({len(sub)} windows):")
        print(f"  Modularity: {sub['modularity'].mean():.2f} +/- {sub['modularity'].std():.2f}")
        print(f"  Spectral Gap: {sub['spectral_gap'].mean():.2f} +/- {sub['spectral_gap'].std():.2f}")
        print(f"  Eff. Connectivity: {sub['eff_connectivity'].mean():.2f} +/- {sub['eff_connectivity'].std():.2f}")
        print(f"  Slippage: {sub['avg_slippage_pct'].mean():.2f} +/- {sub['avg_slippage_pct'].std():.2f}")
