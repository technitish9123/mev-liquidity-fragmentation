"""
Statistical analysis module — produces all regression tables and test results
for the paper (Tables 1–8).

Uses the real-data DataFrame from pipeline.py as input.
Outputs: printed tables + dict of key statistics for paper text updates.

Usage:
    from src.processors.statistical_analysis import run_all_analyses
    stats = run_all_analyses(df)
    # stats dict contains all values needed for paper text
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config as C
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import logging

log = logging.getLogger(__name__)


# ============================================================
# Table 1: Summary Statistics
# ============================================================

def table1_summary_stats(df: pd.DataFrame) -> dict:
    """Compute Table 1 summary statistics."""
    cols = {
        "mev_intensity":  "MEV Intensity I̅",
        "volatility":     "ETH Volatility (σ_1d)",
        "gas_price_gwei": "Avg. Gas (Gwei)",
        "volume_usd_m":   "Total Volume ($M)",
        "active_pools":   "Active Pools",
        "nodes":          "Nodes (Tokens)",
        "edges":          "Edges (Pools)",
    }
    rows = []
    for col, label in cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "Variable": label,
            "Mean": f"{s.mean():.3f}",
            "Std":  f"{s.std():.3f}",
            "Min":  f"{s.min():.3f}",
            "Max":  f"{s.max():.3f}",
        })
    result = pd.DataFrame(rows)
    print("\n=== TABLE 1: Dataset Summary Statistics ===")
    print(result.to_string(index=False))
    return {f"t1_{col}_mean": df[col].mean() for col in cols if col in df.columns}


# ============================================================
# Table 2: Connectivity Metrics by MEV Regime
# ============================================================

def table2_connectivity_by_regime(df: pd.DataFrame) -> dict:
    """Compute Table 2 regime comparison + Welch's t-test."""
    low  = df[df["mev_regime"] == "low"]
    high = df[df["mev_regime"] == "high"]

    metrics = ["eff_connectivity", "spectral_gap", "avg_path_length",
               "giant_component_pct", "avg_eff_resistance"]
    labels  = ["Eff. Connectivity C_eff", "Spectral Gap λ₂",
               "Avg. Path Length", "Giant Component (%)", "Avg. Eff. Resistance"]

    out = {}
    rows = []
    for col, label in zip(metrics, labels):
        if col not in df.columns:
            continue
        lm, ls = low[col].mean(),  low[col].std()
        hm, hs = high[col].mean(), high[col].std()
        t_stat, p_val = scipy_stats.ttest_ind(low[col], high[col], equal_var=False)
        change_pct = (hm - lm) / lm * 100 if lm != 0 else 0

        rows.append({
            "Metric": label,
            "Low MEV": f"{lm:.3f} ± {ls:.3f}",
            "High MEV": f"{hm:.3f} ± {hs:.3f}",
            "Change": f"{change_pct:+.0f}%",
            "p-value": f"{'<0.001' if p_val < 0.001 else f'{p_val:.3f}'}",
        })
        out[f"t2_{col}_low"] = lm
        out[f"t2_{col}_high"] = hm
        out[f"t2_{col}_change_pct"] = change_pct
        out[f"t2_{col}_pval"] = p_val

    result = pd.DataFrame(rows)
    print("\n=== TABLE 2: Network Connectivity Metrics by MEV Regime ===")
    print(f"Low MEV: {len(low)} windows | High MEV: {len(high)} windows")
    print(result.to_string(index=False))
    return out


# ============================================================
# Table 3: User-Facing Performance Metrics
# ============================================================

def table3_user_facing(df: pd.DataFrame) -> dict:
    """Compute Table 3 user-facing metrics by regime."""
    low  = df[df["mev_regime"] == "low"]
    high = df[df["mev_regime"] == "high"]

    metrics = ["avg_slippage_pct", "avg_route_hops", "failed_routes_pct", "median_exec_cost_usd"]
    labels  = ["Avg. Slippage (%)", "Avg. Route Hops", "Failed Routes (%)", "Median Exec Cost ($)"]

    out = {}
    rows = []
    for col, label in zip(metrics, labels):
        if col not in df.columns:
            continue
        lm, ls = low[col].mean(),  low[col].std()
        hm, hs = high[col].mean(), high[col].std()
        t_stat, p_val = scipy_stats.ttest_ind(low[col], high[col], equal_var=False)
        change_pct = (hm - lm) / lm * 100 if lm != 0 else 0

        rows.append({
            "Metric": label,
            "Low MEV": f"{lm:.2f} ± {ls:.2f}",
            "High MEV": f"{hm:.2f} ± {hs:.2f}",
            "Change": f"{change_pct:+.0f}%",
        })
        out[f"t3_{col}_low"] = lm
        out[f"t3_{col}_high"] = hm
        out[f"t3_{col}_change_pct"] = change_pct

    result = pd.DataFrame(rows)
    print("\n=== TABLE 3: User-Facing Performance by MEV Regime ===")
    print(result.to_string(index=False))
    return out


# ============================================================
# Tables 2b: Modularity ratio (key paper claim)
# ============================================================

def modularity_comparison(df: pd.DataFrame) -> dict:
    """The key paper claim: modularity ratio between high and low MEV regimes."""
    low  = df[df["mev_regime"] == "low"]
    high = df[df["mev_regime"] == "high"]

    q_low  = low["modularity"].mean()
    q_high = high["modularity"].mean()
    ratio  = q_high / q_low if q_low > 0 else 0
    t_stat, p_val = scipy_stats.ttest_ind(low["modularity"], high["modularity"], equal_var=False)

    print(f"\n=== KEY FINDING: Modularity ===")
    print(f"  Low MEV:  Q = {q_low:.3f} ± {low['modularity'].std():.3f}")
    print(f"  High MEV: Q = {q_high:.3f} ± {high['modularity'].std():.3f}")
    print(f"  Ratio: {ratio:.1f}× (t={t_stat:.1f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'})")

    return {
        "q_low": q_low, "q_high": q_high, "q_ratio": ratio,
        "q_tstat": t_stat, "q_pval": p_val,
    }


# ============================================================
# Table 4: OLS and IV Regression
# ============================================================

def table4_regressions(df: pd.DataFrame) -> dict:
    """Run OLS and IV/2SLS regressions for Table 4."""
    import statsmodels.api as sm

    # Prepare data
    y = df["modularity"].values
    X_cols = ["mev_intensity", "volatility", "gas_price_gwei", "volume_usd_m", "active_pools"]
    X_available = [c for c in X_cols if c in df.columns]

    X = df[X_available].copy()
    # Log-transform volume if present
    if "volume_usd_m" in X.columns:
        X["volume_usd_m"] = np.log1p(X["volume_usd_m"])
        X = X.rename(columns={"volume_usd_m": "volume_log"})
    if "gas_price_gwei" in X.columns:
        X["gas_price_gwei"] = np.log1p(X["gas_price_gwei"])
        X = X.rename(columns={"gas_price_gwei": "gas_log"})

    X = sm.add_constant(X)

    # OLS with Newey-West HAC standard errors (5 lags)
    ols_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print("\n=== TABLE 4a: OLS Regression (DV: Modularity Q) ===")
    print(ols_model.summary2().tables[1].to_string())
    print(f"  R² = {ols_model.rsquared:.3f}  N = {len(df)}")

    out = {
        "ols_r2": ols_model.rsquared,
        "ols_n": len(df),
    }
    # Extract MEV coefficient
    if "mev_intensity" in ols_model.params.index:
        out["ols_beta_mev"] = ols_model.params["mev_intensity"]
        out["ols_se_mev"]   = ols_model.bse["mev_intensity"]
        out["ols_pval_mev"] = ols_model.pvalues["mev_intensity"]
    if "volatility" in ols_model.params.index:
        out["ols_beta_vol"] = ols_model.params["volatility"]

    # IV/2SLS: instrument MEV with HHI (if available)
    if "hhi_builders" in df.columns:
        try:
            from statsmodels.sandbox.regression.gmm import IV2SLS

            # First stage: MEV ~ HHI + controls
            Z = X.copy()
            Z["hhi_builders"] = df["hhi_builders"].values
            # Drop mev_intensity from Z, add hhi as instrument
            endog = y
            exog = X
            instruments = Z

            iv_model = IV2SLS(endog, exog, instruments).fit()
            print("\n=== TABLE 4b: IV/2SLS Regression ===")
            print(f"  IV MEV coeff: {iv_model.params.get('mev_intensity', 'N/A')}")

            if hasattr(iv_model, 'rsquared'):
                out["iv_r2"] = iv_model.rsquared
        except ImportError:
            print("\n  [IV/2SLS skipped — linearmodels not installed]")
            # Fallback: manual 2SLS
            _manual_2sls(df, X, y, out)
    else:
        _manual_2sls(df, X, y, out)

    return out


def _manual_2sls(df, X, y, out):
    """Manual 2SLS fallback when linearmodels is not available."""
    import statsmodels.api as sm

    if "hhi_builders" not in df.columns:
        print("  [2SLS skipped: no HHI instrument]")
        return

    # Stage 1: MEV ~ HHI + controls (use transformed X columns)
    control_cols = [c for c in X.columns if c not in ("mev_intensity", "const")]
    Z1_data = X[control_cols].copy()
    Z1_data["hhi_builders"] = df["hhi_builders"].values
    Z1 = sm.add_constant(Z1_data)
    stage1 = sm.OLS(X["mev_intensity"].values, Z1).fit()
    mev_hat = stage1.fittedvalues

    first_stage_F = stage1.fvalue
    print(f"\n=== TABLE 4b: Manual 2SLS ===")
    print(f"  First-stage F = {first_stage_F:.1f}")

    # Stage 2: Q ~ MEV_hat + controls
    X2 = X.copy()
    X2["mev_intensity"] = mev_hat
    stage2 = sm.OLS(y, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    if "mev_intensity" in stage2.params.index:
        out["iv_beta_mev"] = stage2.params["mev_intensity"]
        out["iv_se_mev"]   = stage2.bse["mev_intensity"]
        print(f"  IV MEV coeff = {out['iv_beta_mev']:.3f} (SE={out['iv_se_mev']:.3f})")

    out["iv_first_stage_F"] = first_stage_F
    out["iv_r2"] = stage2.rsquared


# ============================================================
# Table 5: Granger Causality
# ============================================================

def table5_granger(df: pd.DataFrame, maxlag: int = 3) -> dict:
    """Granger causality tests between MEV, modularity, and volatility."""
    from statsmodels.tsa.stattools import grangercausalitytests

    out = {}
    tests = [
        ("mev_intensity", "modularity",    "MEV → Modularity"),
        ("modularity",    "mev_intensity", "Modularity → MEV"),
        ("volatility",    "modularity",    "Volatility → Modularity"),
        ("volatility",    "mev_intensity", "Volatility → MEV"),
    ]

    print(f"\n=== TABLE 5: Granger Causality Tests (VAR with {maxlag} lags) ===")
    rows = []
    for cause, effect, label in tests:
        if cause not in df.columns or effect not in df.columns:
            continue
        data = df[[effect, cause]].dropna()
        if len(data) < maxlag * 3:
            continue
        try:
            result = grangercausalitytests(data, maxlag=[maxlag], verbose=False)
            f_stat = result[maxlag][0]["ssr_ftest"][0]
            p_val  = result[maxlag][0]["ssr_ftest"][1]
            rows.append({
                "Null Hypothesis": f"{cause} ⇏ {effect}",
                "F-statistic": f"{f_stat:.2f}",
                "p-value": f"{'< 0.001' if p_val < 0.001 else f'{p_val:.3f}'}",
            })
            out[f"granger_{cause}_{effect}_F"] = f_stat
            out[f"granger_{cause}_{effect}_p"] = p_val
        except Exception as e:
            log.warning("Granger test %s failed: %s", label, e)

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    return out


# ============================================================
# Table 6: Robustness Checks
# ============================================================

def table6_robustness(df: pd.DataFrame) -> dict:
    """Robustness checks: alternative specs, placebo, subperiods."""
    import statsmodels.api as sm

    def _ols_beta_mev(sub_df):
        """Quick OLS: modularity ~ mev_intensity + volatility + const."""
        y = sub_df["modularity"].values
        X = sm.add_constant(sub_df[["mev_intensity", "volatility"]].dropna())
        if len(X) < 10:
            return np.nan, np.nan
        model = sm.OLS(y[:len(X)], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        return model.params.get("mev_intensity", np.nan), model.pvalues.get("mev_intensity", np.nan)

    rows = []
    out = {}

    # Baseline
    beta, pval = _ols_beta_mev(df)
    rows.append({"Specification": "Baseline OLS", "β_MEV": f"{beta:.2f}", "p-value": f"{'< 0.001' if pval < 0.001 else f'{pval:.3f}'}"})
    out["robust_baseline_beta"] = beta

    # Placebo: shuffle MEV intensity
    df_placebo = df.copy()
    rng = np.random.default_rng(C.SEED)
    df_placebo["mev_intensity"] = rng.permutation(df_placebo["mev_intensity"].values)
    beta_p, pval_p = _ols_beta_mev(df_placebo)
    rows.append({"Specification": "Placebo (randomized MEV)", "β_MEV": f"{beta_p:.2f}", "p-value": f"{pval_p:.3f}"})
    out["robust_placebo_beta"] = beta_p
    out["robust_placebo_pval"] = pval_p

    # Excluding extreme volatility (top/bottom 5%)
    vol_lo = df["volatility"].quantile(0.05)
    vol_hi = df["volatility"].quantile(0.95)
    df_trim = df[(df["volatility"] >= vol_lo) & (df["volatility"] <= vol_hi)]
    beta_t, pval_t = _ols_beta_mev(df_trim)
    rows.append({"Specification": "Excl. extreme volatility", "β_MEV": f"{beta_t:.2f}", "p-value": f"{'< 0.001' if pval_t < 0.001 else f'{pval_t:.3f}'}"})

    # Quarterly subperiods
    df["quarter"] = pd.cut(df["window_id"],
                           bins=4, labels=["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"])
    for q in ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]:
        sub = df[df["quarter"] == q]
        if len(sub) > 15:
            beta_q, pval_q = _ols_beta_mev(sub)
            rows.append({"Specification": f"{q} only", "β_MEV": f"{beta_q:.2f}",
                         "p-value": f"{'< 0.001' if pval_q < 0.001 else f'{pval_q:.3f}'}"})

    result = pd.DataFrame(rows)
    print("\n=== TABLE 6: Robustness Checks ===")
    print(result.to_string(index=False))
    return out


# ============================================================
# Table 7: OLS Across All Dependent Variables
# ============================================================

def table7_all_dvs(df: pd.DataFrame) -> dict:
    """OLS regression with MEV as IV across modularity, λ₂, slippage, path length."""
    import statsmodels.api as sm

    dvs = {
        "modularity":      "Modularity",
        "spectral_gap":    "λ₂",
        "avg_slippage_pct": "Slippage",
        "avg_path_length": "Path Length",
    }

    X = sm.add_constant(df[["mev_intensity", "volatility"]].dropna())
    out = {}
    rows_beta = []
    rows_se = []
    r2s = {}

    print("\n=== TABLE 7: OLS Regression Across Metrics ===")
    for col, label in dvs.items():
        if col not in df.columns:
            continue
        y = df[col].values[:len(X)]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        mev_b = model.params.get("mev_intensity", np.nan)
        mev_se = model.bse.get("mev_intensity", np.nan)
        mev_p = model.pvalues.get("mev_intensity", np.nan)
        stars = "***" if mev_p < 0.01 else "**" if mev_p < 0.05 else "*" if mev_p < 0.10 else ""

        print(f"  {label:12s}: β_MEV = {mev_b:.3f}{stars} (SE={mev_se:.3f})  R²={model.rsquared:.2f}")
        out[f"t7_{col}_beta"] = mev_b
        out[f"t7_{col}_r2"] = model.rsquared

    return out


# ============================================================
# Table 8: Subperiod Analysis (Modularity by Regime × Quarter)
# ============================================================

def table8_subperiod(df: pd.DataFrame) -> dict:
    """Modularity by MEV regime broken down by quarter."""
    if "quarter" not in df.columns:
        df = df.copy()
        df["quarter"] = pd.cut(df["window_id"],
                               bins=4, labels=["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"])

    print("\n=== TABLE 8: Modularity by MEV Regime and Quarter ===")
    out = {}
    rows = []
    for q in ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]:
        sub = df[df["quarter"] == q]
        low  = sub[sub["mev_regime"] == "low"]["modularity"]
        high = sub[sub["mev_regime"] == "high"]["modularity"]
        ratio = high.mean() / low.mean() if low.mean() > 0 else 0

        rows.append({
            "Quarter": q,
            "Low MEV": f"{low.mean():.3f} ± {low.std():.3f}" if len(low) > 0 else "N/A",
            "High MEV": f"{high.mean():.3f} ± {high.std():.3f}" if len(high) > 0 else "N/A",
            "Ratio": f"{ratio:.2f}×",
        })
        out[f"t8_{q}_ratio"] = ratio

    print(pd.DataFrame(rows).to_string(index=False))
    return out


# ============================================================
# Phase Transition Fit (for Figure 2 + paper text)
# ============================================================

def fit_phase_transition(df: pd.DataFrame) -> dict:
    """Fit logistic function Q(I) and extract I*, κ, Q_min, Q_max, R²."""
    from scipy.optimize import curve_fit

    def logistic(I, Q_max, I_star, kappa, Q_min):
        return Q_max / (1 + np.exp(-kappa * (I - I_star))) + Q_min

    mev = df["mev_intensity"].values
    mod = df["modularity"].values

    # Remove NaN
    mask = np.isfinite(mev) & np.isfinite(mod)
    mev, mod = mev[mask], mod[mask]

    try:
        p0 = [0.5, np.median(mev), 30, mod.min()]
        popt, pcov = curve_fit(logistic, mev, mod, p0=p0, maxfev=10000)
        Q_max_fit, I_star_fit, kappa_fit, Q_min_fit = popt

        # R²
        y_pred = logistic(mev, *popt)
        ss_res = np.sum((mod - y_pred) ** 2)
        ss_tot = np.sum((mod - mod.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n=== PHASE TRANSITION FIT ===")
        print(f"  I* = {I_star_fit:.4f}")
        print(f"  κ  = {kappa_fit:.1f}")
        print(f"  Q_min = {Q_min_fit:.3f}")
        print(f"  Q_max = {Q_max_fit:.3f}")
        print(f"  R² = {r2:.3f}")

        return {
            "pt_I_star": I_star_fit, "pt_kappa": kappa_fit,
            "pt_Q_min": Q_min_fit, "pt_Q_max": Q_max_fit,
            "pt_r2": r2,
        }
    except Exception as e:
        log.warning("Phase transition fit failed: %s", e)
        return {"pt_I_star": 0, "pt_kappa": 0, "pt_Q_min": 0, "pt_Q_max": 0, "pt_r2": 0}


# ============================================================
# Master Function
# ============================================================

def run_all_analyses(df: pd.DataFrame) -> dict:
    """Run all statistical analyses and return combined stats dict."""
    print("=" * 70)
    print(f"  STATISTICAL ANALYSIS — {len(df)} windows, {df['data_source'].value_counts().to_dict()}")
    print("=" * 70)

    all_stats = {}
    all_stats.update(table1_summary_stats(df))
    all_stats.update(modularity_comparison(df))
    all_stats.update(table2_connectivity_by_regime(df))
    all_stats.update(table3_user_facing(df))
    all_stats.update(table4_regressions(df))
    all_stats.update(table5_granger(df))
    all_stats.update(table6_robustness(df))
    all_stats.update(table7_all_dvs(df))
    all_stats.update(table8_subperiod(df))
    all_stats.update(fit_phase_transition(df))

    # Key paper claims summary
    print("\n" + "=" * 70)
    print("  KEY PAPER CLAIMS (for abstract/introduction)")
    print("=" * 70)
    q_ratio = all_stats.get("q_ratio", 0)
    q_low   = all_stats.get("q_low", 0)
    q_high  = all_stats.get("q_high", 0)
    c_change = all_stats.get("t2_eff_connectivity_change_pct", 0)
    s_change = all_stats.get("t3_avg_slippage_pct_change_pct", 0)
    p_change = all_stats.get("t2_avg_path_length_change_pct", 0)
    I_star   = all_stats.get("pt_I_star", 0)
    r2       = all_stats.get("pt_r2", 0)

    print(f"  Modularity: {q_ratio:.1f}× higher ({q_high:.2f} vs {q_low:.2f} baseline)")
    print(f"  Connectivity: {c_change:+.0f}% change")
    print(f"  Slippage: {s_change:+.0f}% change")
    print(f"  Path length: {p_change:+.0f}% change")
    print(f"  I* = {I_star:.4f}  (logistic R² = {r2:.3f})")
    print(f"  OLS β_MEV = {all_stats.get('ols_beta_mev', 0):.3f}")
    print(f"  Granger MEV→Q F = {all_stats.get('granger_mev_intensity_modularity_F', 0):.2f}")
    print("=" * 70)

    return all_stats


if __name__ == "__main__":
    # Run on existing CSV
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/windows_500_real.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    stats = run_all_analyses(df)
