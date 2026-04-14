"""
Figure 1: Network modularity evolution across 5 million Ethereum blocks.

Shows modularity Q over time with high-MEV periods shaded and
theoretical threshold I* marked.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def plot_modularity_evolution(df, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(C.ACM_DOUBLE_COL_WIDTH, 4.5),
                                     sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

    blocks = df['block_start'].values / 1e6  # in millions

    # --- Top panel: MEV intensity ---
    ax1.fill_between(blocks, df['mev_intensity'], alpha=0.4, color='#E74C3C', linewidth=0)
    ax1.plot(blocks, df['mev_intensity'], color='#C0392B', linewidth=0.6, alpha=0.8)
    ax1.axhline(y=C.I_STAR, color='#2C3E50', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.text(blocks[-1] + 0.02, C.I_STAR, r'$I^*$', fontsize=8, va='center', color='#2C3E50')
    ax1.set_ylabel(r'MEV Intensity $\bar{I}$', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.set_ylim(0, C.MEV_MAX + 0.01)

    # --- Bottom panel: Modularity ---
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

    ax2.plot(blocks, df['modularity'], color='#2980B9', linewidth=0.8, alpha=0.9)

    # Data-driven reference lines
    low_q = df.loc[df['mev_regime'] == 'low', 'modularity'].mean()
    high_q = df.loc[df['mev_regime'] == 'high', 'modularity'].mean()
    ratio = high_q / low_q if low_q > 0 else 1.0

    ax2.axhline(y=low_q, color='#7F8C8D', linestyle=':', linewidth=0.7, alpha=0.6)
    ax2.text(blocks[0] - 0.08, low_q, f'$Q_{{base}}={low_q:.2f}$', fontsize=6.5,
             va='center', ha='right', color='#7F8C8D')

    ax2.axhline(y=high_q, color='#7F8C8D', linestyle=':', linewidth=0.7, alpha=0.6)
    ax2.text(blocks[0] - 0.08, high_q, f'$Q_{{high}}={high_q:.2f}$', fontsize=6.5,
             va='center', ha='right', color='#7F8C8D')

    ax2.set_ylabel(r'Graph Modularity $Q$', fontsize=8)
    ax2.set_xlabel('Block Number (millions)', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(blocks[0], blocks[-1])

    # Annotation for modularity increase
    ax2.annotate(f'{ratio:.1f}' + r'$\times$ increase', xy=(blocks[len(blocks)//3], 0.92),
                fontsize=7, color='#C0392B', fontstyle='italic')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.15, label='High-MEV periods'),
        Line2D([0], [0], color='#2980B9', linewidth=1, label='Modularity $Q$'),
        Line2D([0], [0], color='#2C3E50', linestyle='--', linewidth=0.8, label=r'Threshold $I^*$'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=6.5,
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
                       'output', 'figures', 'fig1_modularity.png')
    plot_modularity_evolution(df, out)
