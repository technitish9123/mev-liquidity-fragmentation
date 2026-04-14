"""
Master script: fetch real on-chain data + generate all 3 publication figures.

Usage:
  python3 figures/generate_figures.py                  # full real pipeline
  python3 figures/generate_figures.py --mode synthetic # legacy synthetic
  python3 figures/generate_figures.py --test           # 5-window smoke test
"""

import sys, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from src.pipeline import run_pipeline
from figures.fig1_modularity      import plot_modularity_evolution
from figures.fig2_phase_transition import plot_phase_transition
from figures.fig3_spectral_gap    import plot_spectral_gap


def print_summary(df):
    """Print Table 1 + Table 2 summary stats to terminal."""
    print("\n--- Table 1: Summary Statistics ---")
    col_map = [
        ("mev_intensity",   "MEV Intensity"),
        ("volatility",      "ETH Volatility"),
        ("gas_price_gwei",  "Avg. Gas (Gwei)"),
        ("volume_usd_m",    "Total Volume ($M)"),
        ("active_pools",    "Active Pools"),
        ("nodes",           "Nodes"),
        ("edges",           "Edges"),
    ]
    for col, label in col_map:
        if col in df.columns:
            s = df[col]
            print(f"  {label:22s}: mean={s.mean():.3f}  std={s.std():.3f}"
                  f"  min={s.min():.3f}  max={s.max():.3f}")

    print("\n--- Table 2: Connectivity by MEV Regime ---")
    for regime in ["low", "high"]:
        sub = df[df["mev_regime"] == regime]
        if len(sub) == 0:
            continue
        print(f"  {regime.upper():6s} MEV ({len(sub)} windows):")
        for col, label in [("modularity",       "Modularity"),
                            ("spectral_gap",     "Spectral Gap"),
                            ("eff_connectivity", "Eff. Connectivity"),
                            ("avg_path_length",  "Path Length")]:
            if col in sub.columns:
                print(f"    {label:20s}: {sub[col].mean():.3f} ± {sub[col].std():.3f}")

    # Data provenance summary
    if "data_source" in df.columns:
        print("\n--- Data Provenance ---")
        counts = df["data_source"].value_counts()
        total  = len(df)
        for src, cnt in counts.items():
            print(f"  {src:12s}: {cnt:4d} windows ({cnt/total*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate MEV fragmentation figures")
    parser.add_argument("--mode",  default=C.DATA_MODE,
                        choices=["real", "hybrid", "synthetic"],
                        help="Data pipeline mode")
    parser.add_argument("--test",  action="store_true",
                        help="Smoke test: run only 5 windows")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save CSV to data/processed/")
    args = parser.parse_args()

    n_windows = 5 if args.test else None

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 1: Loading data  [mode={args.mode}"
          + (" | SMOKE TEST 5 windows" if args.test else "") + "]")
    print("=" * 60)

    df = run_pipeline(
        mode      = args.mode,
        n_windows = n_windows,
        save      = not args.no_save and not args.test,
    )

    print(f"  DataFrame: {len(df)} rows × {len(df.columns)} columns")
    print_summary(df)

    if args.test:
        print("\nSmoke test passed — exiting before figure generation.")
        return df

    # ── Step 2: Figures ───────────────────────────────────────────────────────
    fig_dir = os.path.join(ROOT, "output", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Step 2: Figure 1 — Modularity Evolution")
    print("=" * 60)
    plot_modularity_evolution(df, os.path.join(fig_dir, "fig1_modularity.png"))

    print("\n" + "=" * 60)
    print("Step 3: Figure 2 — Phase Transition")
    print("=" * 60)
    plot_phase_transition(df, os.path.join(fig_dir, "fig2_phase_transition.png"))

    print("\n" + "=" * 60)
    print("Step 4: Figure 3 — Spectral Gap")
    print("=" * 60)
    plot_spectral_gap(df, os.path.join(fig_dir, "fig3_spectral_gap.png"))

    # Copy PDFs to paper/figures/
    paper_fig_dir = os.path.join(ROOT, "paper", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    import shutil
    for fig in ["fig1_modularity", "fig2_phase_transition", "fig3_spectral_gap"]:
        for ext in ["pdf", "png"]:
            src = os.path.join(fig_dir, f"{fig}.{ext}")
            dst = os.path.join(paper_fig_dir, f"{fig}.{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    print("\n" + "=" * 60)
    print("DONE — figures saved to output/figures/ and paper/figures/")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
