"""
Figure 3: Spectral gap (lambda_2) evolution.

Shows spectral gap tracking MEV intensity inversely, collapsing from
0.23 to 0.09 during high-MEV periods. Includes lambda_2=0.15 threshold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def plot_spectral_gap(df, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(C.ACM_DOUBLE_COL_WIDTH, 4.5),
                                     sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

    blocks = df['block_start'].values / 1e6

    # --- Top panel: MEV intensity ---
    ax1.fill_between(blocks, df['mev_intensity'], alpha=0.4, color='#E74C3C', linewidth=0)
    ax1.plot(blocks, df['mev_intensity'], color='#C0392B', linewidth=0.6, alpha=0.8)
    ax1.axhline(y=C.I_STAR, color='#2C3E50', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.text(blocks[-1] + 0.02, C.I_STAR, r'$I^*$', fontsize=8, va='center', color='#2C3E50')
    ax1.set_ylabel(r'MEV Intensity $\bar{I}$', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.set_ylim(0, C.MEV_MAX + 0.01)

    # --- Bottom panel: Spectral gap ---
    # Shade high-MEV regions
    high_mev = df['mev_regime'] == 'high'
    in_region = False
    region_start = None
    for i in range(len(df)):
        if high_mev.iloc[i] and not in_region:
            region_start = blocks[i]
            in_region = True
        elif not high_mev.iloc[i] and in_region:
            ax2.axvspan(region_start, blocks[i], alpha=0.12, color='#E74C3C', linewidth=0)
            ax1.axvspan(region_start, blocks[i], alpha=0.12, color='#E74C3C', linewidth=0)
            in_region = False
    if in_region:
        ax2.axvspan(region_start, blocks[-1], alpha=0.12, color='#E74C3C', linewidth=0)
        ax1.axvspan(region_start, blocks[-1], alpha=0.12, color='#E74C3C', linewidth=0)

    ax2.plot(blocks, df['spectral_gap'], color='#27AE60', linewidth=0.8, alpha=0.9)

    # Threshold line
    ax2.axhline(y=C.LAMBDA2_THRESHOLD, color='#8E44AD', linestyle='--',
                linewidth=1, alpha=0.7)
    ax2.text(blocks[-1] + 0.02, C.LAMBDA2_THRESHOLD,
             r'$\lambda_2=0.15$', fontsize=7, va='center', color='#8E44AD')

    # Data-driven reference levels
    low_sg = df.loc[df['mev_regime'] == 'low', 'spectral_gap'].mean()
    high_sg = df.loc[df['mev_regime'] == 'high', 'spectral_gap'].mean()

    ax2.axhline(y=low_sg, color='#BDC3C7', linestyle=':', linewidth=0.6, alpha=0.5)
    ax2.text(blocks[0] - 0.08, low_sg, f'{low_sg:.2f}', fontsize=6, va='center',
             ha='right', color='#7F8C8D')

    ax2.axhline(y=high_sg, color='#BDC3C7', linestyle=':', linewidth=0.6, alpha=0.5)
    ax2.text(blocks[0] - 0.08, high_sg, f'{high_sg:.2f}', fontsize=6, va='center',
             ha='right', color='#7F8C8D')

    ax2.set_ylabel(r'Spectral Gap $\lambda_2$', fontsize=8)
    ax2.set_xlabel('Block Number (millions)', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_ylim(0, 0.35)
    ax2.set_xlim(blocks[0], blocks[-1])

    # Annotation
    ax2.annotate(r'$\lambda_2$ collapses below threshold',
                xy=(blocks[len(blocks)//3], 0.06), fontsize=6.5,
                color='#C0392B', fontstyle='italic')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.15, label='High-MEV periods'),
        Line2D([0], [0], color='#27AE60', linewidth=1, label=r'Spectral Gap $\lambda_2$'),
        Line2D([0], [0], color='#8E44AD', linestyle='--', linewidth=1,
               label=r'Routing threshold ($\lambda_2=0.15$)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=6.5,
               framealpha=0.9, edgecolor='#BDC3C7')

    plt.tight_layout(h_pad=0.3)

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
                       'output', 'figures', 'fig3_spectral_gap.png')
    plot_spectral_gap(df, out)
