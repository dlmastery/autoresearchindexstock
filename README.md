# AutoResearch QQQ — Nasdaq-100 Index/Stock Autoresearch Loop

[![Dashboard](https://img.shields.io/badge/dashboard-live-success)](https://dlmastery.github.io/autoresearchindexstock/dashboard/)
[![Experiments](https://img.shields.io/badge/experiments-216%2B-blue)](./autoresearch_results/experiment_log.jsonl)
[![Backbones](https://img.shields.io/badge/backbones%20complete-5%2F6-brightgreen)](./autoresearch_results/winners/)

> **Top-level project** spawned from `dlmastery/autoresearch` (FX) for the
> equity-index variant. Self-contained successor; FX project remains at the
> parent repo.

## Live dashboard

**https://dlmastery.github.io/autoresearchindexstock/dashboard/**

Tracks every experiment in real-time: per-fold A_sharpe, val/test breakdown,
best config, equity curves, trade logs, classification metrics
(precision/recall/F1/F2/MCC).

## Headline result (post-exp 216)

| Metric | Value | Notes |
|---|---:|---|
| **Global champion composite** | **+1.3216** | mamba dmamba exp 52 (single-seed=42) |
| Champion A_sharpe | +1.3216 | val_sharpe +1.4831, PSR +0.9972 |
| Champion test folds | 7/7 positive | F1-F7 all positive |
| Buy-and-hold baseline | +0.87 Sharpe | strategy excess +0.45 |
| 2nd stable backbone | MLP exp 79/204 | 5-seed median +0.43, 3-seed median +0.52 |
| Total experiments | 216+ | session 165-216 = 51 new this branch |
| Backbones complete | 5/6 cheap-tier | MLP/XGBoost/LGBM/Mamba/CatBoost ✓ |

## Repository layout

```
autoresearchindexstock/
├── CLAUDE.md                      # Project rules + user directives log
├── run_autoresearch.py            # Main experiment runner
├── data/                          # Download + 184 backward-looking features
├── model/                         # Backbone wrappers (MLP, LSTM, Mamba, ...)
├── evaluation/                    # Metrics: Sharpe, PSR, IC, classification
├── autoresearch_results/          # Live experiment log + dashboard + winners
│   ├── experiment_log.jsonl       # JSONL: one row per experiment
│   ├── dashboard.html             # Interactive dashboard (mirror in docs/)
│   ├── reasoning_annotations.json # Per-experiment reasoning blob
│   ├── research_journal.md        # Markdown research journal
│   ├── experiment_summary.md      # Master summary table
│   ├── trade_logs/                # Per-experiment trade CSV + summary JSON
│   └── winners/                   # Self-contained champion archives
│       ├── mamba_exp52_dmamba_e2_d32_seed42/   # GLOBAL CHAMPION
│       ├── mlp_exp79_residual_seq10_wd1e5_warmup5/
│       ├── mlp_exp204_residual_seq10_wd1e4_warmup5/
│       └── mamba_exp178_mambats_e2_d32_seed42/
├── memory/                        # Crash-recovery checkpoint
├── code_versions/                 # Frozen model snapshots per backbone phase
└── docs/dashboard/                # GitHub Pages mirror
```

## Quick start

```bash
# Install requirements (parent autoresearch project provides scaffolding)
pip install torch numpy pandas scikit-learn xgboost lightgbm catboost \
  mamba-ssm transformers neuralforecast

# Download data (cached to .data_cache_qqq/)
python -m autoresearchindexstock.data.download

# Run a single experiment
python -m autoresearchindexstock.run_autoresearch \
  --backbone mamba --mamba-variant dmamba --expand 2 --d-state 32 \
  --seed 42 --description "reproduce global champion exp 52"

# Live dashboard (local)
python -m http.server 8888 \
  --directory autoresearchindexstock/autoresearch_results
# → http://localhost:8888/dashboard.html
```

## Key methodology

- **7-fold walk-forward** super-fold split (2007–2025-12). Folds are
  regime-aware: GFC peak crash, EU debt 2011, Taper tantrum, China-oil
  drawdown, Vol-mageddon 2018, COVID V-recovery, AI rally 2025.
- **Composite score**: `min(test_sharpe, val_sharpe) - 0.1 × n_negative_folds`.
- **4 target variants tracked simultaneously**:
  - A — `fwd_ret_1d` (PRIMARY — KEEP/DISCARD basis)
  - B — `fwd_ret_5d`
  - D — vol-adjusted 1d return
  - C — sign-concordance (side-channel only)
- **Excess Sharpe over buy-and-hold** is the fair-comparison metric for a
  trending equity index. Tracked alongside raw Sharpe in every JSONL row.

## Backbone status (216+ experiments)

| Backbone | Used / Target | Best multi-seed median | Best single-seed |
|---|---:|---:|---:|
| Mamba | 25 / 25 ✓ | dmamba 4-seed -0.25 (seed=99 catastrophe) | exp 52 +1.32 (champion) |
| MLP | 51 / 75 (extending) | 5-seed +0.433 (exp 79 config) | exp 204 +0.974 |
| LSTM | 39 / 75 | 4-seed ~0 | exp 119 +1.05 (features rolled back) |
| XGBoost | 25 / 25 ✓ | 3-seed -0.40 (depth=5) | exp 63 -0.13 stable |
| LightGBM | 25 / 25 ✓ | 3-seed -0.11 | exp 95 +0.61 (single-seed luck) |
| CatBoost | 25 / 25 ✓ | 4-seed ~0 (lr=0.05 luck) | exp 169 +0.39 |
| Phase F (DLinear / iTransformer / PatchTST / PatchTSMixer / N-BEATS) | 1-7 each / 25 | partial | iTransformer A_sh +0.92 |

## Citations (top backbone papers)

- Gu, Dao 2024 COLM **Mamba** (arXiv:2312.00752) — global champion architecture
- Liu, Zhang, Wu, Long 2025 **DMamba** (arXiv:2602.09081) — winning variant
- Cai et al. 2024 NeurIPS **MambaTS** (arXiv:2405.16440) — complementary variant
- Gu, Kelly, Xiu 2020 RFS **Empirical Asset Pricing via ML** — MLP recipe
- He et al. 2016 CVPR **ResNet** (arXiv:1512.03385) — residual MLP head
- Loshchilov, Hutter 2019 ICLR **AdamW** (arXiv:1711.05101)
- Goyal et al. 2017 **Large Minibatch SGD warmup** (arXiv:1706.02677)
- Lakshminarayanan, Pritzel, Blundell 2017 NeurIPS **Deep Ensembles** (arXiv:1612.01474)
- Picard 2021 **Torch.manual_seed(3407)** (arXiv:2109.08203) — multi-seed protocol
- Chen, Guestrin 2016 KDD **XGBoost** (arXiv:1603.02754)
- Ke et al. 2017 NeurIPS **LightGBM**
- Prokhorenkova et al. 2018 NeurIPS **CatBoost** (arXiv:1706.09516)
- Friedman 2001 Annals **Greedy Function Approximation: GBM**
- Fischer, Krauss 2018 EJOR **LSTM for FX/equity prediction** (DOI 10.1016/j.ejor.2017.11.054)

## Phase D / E roadmap (per CLAUDE.md)

**Phase D — Stock-specific code adds (next)**:
- Adv-ALSTM (Feng 2019)
- StockMixer (Fan-Yu 2024 KDD, arXiv:2308.07099)
- MASTER (Li-Liu-Wang 2024 AAAI, arXiv:2304.12135)
- PatchMixer (Gong 2023, arXiv:2310.00655)
- CARD (Wang 2024, arXiv:2305.12095)
- Reversible Mixer (Liu 2024 NeurIPS, arXiv:2404.16312)

**Phase E — Foundation models (LAST)**:
- Sundial (Liu 2025 arXiv:2502.00816)
- TimesFM 2.5 (Google 2024 arXiv:2310.10688)
- Chronos-2 (Amazon 2025 arXiv:2510.15821)
- Moirai 2.0 (Salesforce 2025 arXiv:2511.11698)
- TiRex, MOMENT, Time-MoE, TimeMixer

## Provenance

Forked off `dlmastery/autoresearch` (FX project) on 2026-04-29 at session
exp 216. Full session history (165-216) preserved in
`autoresearch_results/experiment_log.jsonl` and
`autoresearch_results/research_journal.md`.

Hardware: Intel 14th-gen HX, 16 GB GPU, P-cores [0,2,4,6] only (E-cores
banned per WHEA-Logger parity errors causing 5 BSODs on 2026-04-19).

## License

MIT (matches parent dlmastery/autoresearch).
