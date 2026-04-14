# MEV-Induced Liquidity Fragmentation in DEX Networks

Research code and data for:

> **Dynamic Graph Rewiring in Decentralized Exchange Networks: A Study of MEV-Induced Liquidity Fragmentation**
> Nitish Kumar
> NBC '26 (Nanyang Blockchain Conference, August 21-22, 2026, NTU Singapore)

## Overview

This repository contains the simulation pipeline and analysis code for studying how Maximal Extractable Value (MEV) extraction affects the topology of decentralized exchange (DEX) liquidity networks. We use a hybrid approach combining on-chain aggregate statistics with calibrated simulation to evaluate the association between arbitrage intensity and network fragmentation.

## Quick Start

```bash
pip install -r requirements.txt
python3 figures/generate_figures.py
```

This generates the calibrated dataset and all 3 publication figures in `output/figures/`.

## Project Structure

```
mev-liquidity-fragmentation/
├── paper/
│   ├── paper.tex             # LaTeX source (ACM sigconf format)
│   ├── references.bib        # BibTeX references
│   └── figures/              # Figures for compilation
├── config.py                 # All parameters (Tables 1-4, Eq. 31, thresholds)
├── src/
│   ├── __init__.py
│   └── data_generator.py     # Calibrated data generation (500 windows)
├── figures/
│   ├── generate_figures.py   # Master script: data + all figures
│   ├── fig1_modularity.py    # Fig 1: Modularity evolution over 5M blocks
│   ├── fig2_phase_transition.py  # Fig 2: Phase transition (logistic fit)
│   └── fig3_spectral_gap.py  # Fig 3: Spectral gap evolution
├── data/
│   └── synthetic/            # Generated CSV datasets
├── output/
│   └── figures/              # Generated PNG + PDF figures
├── requirements.txt
└── README.md
```

## Data Methodology

**On-chain sources** (used for calibration):
- MEV extraction volumes: [ZeroMEV API](https://zeromev.org/)
- Gas prices, block data: [Google BigQuery `crypto_ethereum`](https://cloud.google.com/blog/products/data-analytics/ethereum-bigquery-public-dataset-smart-contract-analytics)
- DEX trading volumes: [Dune Analytics](https://dune.com/)
- Flashbots MEV-Boost relay data

**Simulation**: Parameters are calibrated to match on-chain aggregate distributions. Topology metrics (modularity, spectral gap, effective connectivity) are computed using the dynamic graph model described in the paper.

## Reproducing

```bash
# Generate data + all figures
python3 figures/generate_figures.py

# Or run individual steps
python3 -c "from src.data_generator import generate_dataset; generate_dataset()"
python3 figures/fig1_modularity.py
python3 figures/fig2_phase_transition.py
python3 figures/fig3_spectral_gap.py
```

## Requirements

- Python 3.10+
- NumPy, SciPy, pandas, matplotlib

See `requirements.txt` for exact versions.

## Citation

```bibtex
@inproceedings{kumar2026dynamic,
  title={Dynamic Graph Rewiring in Decentralized Exchange Networks: A Study of MEV-Induced Liquidity Fragmentation},
  author={Kumar, Nitish},
  booktitle={Proceedings of the Nanyang Blockchain Conference (NBC '26)},
  year={2026}
}
```

## License

MIT
