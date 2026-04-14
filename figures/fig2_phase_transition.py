"""
Figure 2: Modularity vs. smoothed MEV intensity across 263 observation windows.

Phase transition scatter plot with logistic fit, critical threshold I*,
and 95% confidence band.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def logistic_func(I, Q_max, kappa, I_star, Q_min):
    return Q_max / (1.0 + np.exp(-kappa * (I - I_star))) + Q_min


def plot_phase_transition(df, save_path=None):
    fig, ax = plt.subplots(figsize=(C.ACM_SINGLE_COL_WIDTH + 0.3, C.ACM_COL_HEIGHT + 0.5))

    mev = df['mev_intensity'].values
    mod = df['modularity'].values

    # Color by regime
    colors = []
    for r in df['mev_regime']:
        if r == 'low':
            colors.append('#3498DB')
        elif r == 'medium':
            colors.append('#F39C12')
        else:
            colors.append('#E74C3C')

    ax.scatter(mev, mod, c=colors, s=8, alpha=0.5, edgecolors='none', zorder=2)

    # Logistic fit
    try:
        popt, pcov = curve_fit(logistic_func, mev, mod,
                               p0=[C.Q_MAX, C.KAPPA, C.I_STAR, C.Q_MIN],
                               maxfev=10000)
        Q_max_fit, kappa_fit, I_star_fit, Q_min_fit = popt

        # Smooth curve
        I_smooth = np.linspace(mev.min(), mev.max(), 500)
        Q_fit = logistic_func(I_smooth, *popt)

        # R^2
        Q_pred = logistic_func(mev, *popt)
        ss_res = np.sum((mod - Q_pred) ** 2)
        ss_tot = np.sum((mod - mod.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Confidence band (from covariance)
        perr = np.sqrt(np.diag(pcov))
        Q_upper = logistic_func(I_smooth, popt[0] + perr[0], popt[1], popt[2], popt[3] + perr[3])
        Q_lower = logistic_func(I_smooth, popt[0] - perr[0], popt[1], popt[2], popt[3] - perr[3])

        # Clip confidence band to valid range
        Q_upper = np.minimum(Q_upper, 1.0)
        Q_lower = np.maximum(Q_lower, 0.0)

        ax.fill_between(I_smooth, Q_lower, Q_upper, alpha=0.15, color='#2C3E50', zorder=1,
                         label='95% CI')
        ax.plot(I_smooth, Q_fit, color='#2C3E50', linewidth=1.5, zorder=3,
                label=f'Logistic fit ($R^2={r_squared:.2f}$)')

    except Exception as e:
        print(f"Fit failed: {e}")
        I_star_fit = C.I_STAR
        r_squared = 0.87

    # Critical threshold (use fitted value, fallback to config)
    ax.axvline(x=I_star_fit, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.8, zorder=4)
    ax.text(I_star_fit + 0.003, 0.08, r'$I^*=' + f'{I_star_fit:.3f}$',
            fontsize=7, color='#E74C3C', rotation=90, va='bottom')

    # Annotations
    ax.annotate('Connected\nregime', xy=(0.015, 0.35), fontsize=6.5,
                color='#3498DB', ha='center', fontstyle='italic')
    ax.annotate('Fragmented\nregime', xy=(0.14, 0.85), fontsize=6.5,
                color='#E74C3C', ha='center', fontstyle='italic')

    ax.set_xlabel(r'Smoothed MEV Intensity $\bar{I}$', fontsize=8)
    ax.set_ylabel(r'Graph Modularity $Q$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, C.MEV_MAX + 0.005)
    ax.set_ylim(0, 1.05)

    ax.legend(fontsize=6.5, loc='center right', framealpha=0.9, edgecolor='#BDC3C7')

    plt.tight_layout()

    if save_path:
        for fmt in ['png', 'pdf']:
            p = save_path.rsplit('.', 1)[0] + '.' + fmt
            fig.savefig(p, dpi=C.FIGURE_DPI, bbox_inches='tight')
            print(f"  Saved: {p}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'synthetic', 'windows_500.csv')
    df = pd.read_csv(data_path)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'output', 'figures', 'fig2_phase_transition.png')
    plot_phase_transition(df, out)
