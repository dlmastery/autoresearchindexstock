# dMamba @ FX-Mamba-winner config — QQQ exp 17

> **🏆 Global champion** of the QQQ AutoResearch project as of session
> 2026-04-26 (commit `3145496`). Strongest single-model composite +0.8625,
> 7/7 positive test folds across all named equity regimes from GFC peak
> through 2025 AI rally.

## Headline metrics

| Metric | Value |
|---|---:|
| **Composite score** | **+0.8625** |
| **A_Sharpe (1d return, primary)** | **+0.8625** |
| Excess-Sharpe vs buy-and-hold | -0.3576 (BH +1.2201) |
| Test_pos_folds | **7 / 7** |
| Val_pos_folds | 4 / 7 |
| B_Sharpe (5d return, auxiliary) | +0.1753 |
| D_Sharpe (vol-adjusted 1d) | +0.8625 |
| Hit-rate (test) | 55.5% |
| Total trades (test) | 1480 |
| Runtime | 2473 s (~41 min on RTX 4090 Laptop) |
| Convergence | epoch 62 (early-stopped, pat=20 from best at ep=42) |

## Architecture

**dMamba** = Liu 2025 (arXiv:2602.09081) trend+seasonal decomposition
variant of Mamba (Gu & Dao 2024 COLM, arXiv:2312.00752 — Selective
State Space Model).

- `variant=dmamba` — trend + seasonal decomposition before SSM, then
  recombine. Captures regime-aware dynamics better than vanilla Mamba's
  pure SSM on equity index data.
- `expand=4` — d_inner = d_model × 4. Richer per-token state than
  vanilla Mamba's expand=2.
- `d_state=16` — SSM hidden state size.
- `seq_len=60` — 60-day look-back per prediction.
- `hidden_size=128`, `num_layers=2`, bidirectional via parent's
  CurrencyMamba wrapper.
- Dual head: ret_1d + ret_5d, residual-connected MLP head with
  head_dropout=0.1.

## Training recipe

```python
{
    "backbone":      "mamba",
    "mamba_variant": "dmamba",
    "mamba_expand":  4,
    "mamba_d_state": 16,
    "seq_len":       60,
    "lr":            5e-4,
    "batch_size":    32,
    "epochs":        100,
    "patience":      20,
    "weight_decay":  0.1,
    "warmup_epochs": 10,
    "head_dropout":  0.1,
    "huber_delta":   1.0,
    "grad_clip":     1.0,
    "seed":          42,
    "optimizer":     "AdamW (decoupled wd, Loshchilov-Hutter 2019)",
    "loss":          "Heteroscedastic Huber (Kendall & Gal 2017)",
    "scheduler":     "Linear warmup 10 ep + cosine annealing",
    "device":        "cuda (RTX 4090 Laptop, 16 GB)",
}
```

## Per-fold breakdown (test set, target A — 1d return)

| Fold | Regime | A_Sharpe | BH_Sharpe | Excess | Hit% | n |
|---|---|---:|---:|---:|---:|---:|
| 1 | GFC peak crash (Lehman + Mar-2009 bottom) | **+0.66** | +0.43 | **+0.23** | 50.8 | 66 |
| 2 | 2011 US-downgrade + EU debt | **+5.09** | +4.87 | **+0.23** | 62.4 | 87 |
| 3 | Taper tantrum + 2014 H1 | +1.02 | +2.06 | -1.04 | 55.1 | 188 |
| 4 | China devaluation + oil crash | **+0.21** | -0.80 | **+1.02** | 56.6 | 167 |
| 5 | 2018 Vol-mageddon + Q4 sell-off | +0.84 | +1.18 | -0.35 | 55.1 | 187 |
| 6 | COVID crash + V-recovery | +0.33 | +2.43 | -2.10 | 52.9 | 232 |
| 7 | Inflation bear + AI rally + 2025 | +0.76 | +0.79 | -0.03 | 56.0 | 435 |
| **All** | **Super-Fold** | **+0.86** | **+1.22** | **-0.36** | 55.5 | 1480 |

**Pattern:** strategy alpha lights up in chaotic regimes (folds 1, 2, 4 —
GFC, US-downgrade, China crash) and underperforms passive in trending
recoveries (folds 3, 6 — Taper, COVID-recovery). This matches the
FX-paper finding and the equity-prediction literature: directional
strategies beat buy-and-hold during volatility, lose during persistent
trends. Excess-Sharpe of -0.36 over the full super-fold is the average
of these regime-specific outcomes.

## Why dMamba (not vanilla Mamba)

Both vanilla Mamba (exp 15) and dMamba (exp 17) at the same SOTA recipe:

| Variant | A_Sharpe | Composite |
|---|---:|---:|
| vanilla Mamba (Gu-Dao 2024 default) | +0.78 | -0.19 |
| **dMamba (Liu 2025 decomp)** | **+0.86** | **+0.86** |

The trend+seasonal decomposition adds ~+0.08 A_sharpe and lifts the
composite from -0.19 to +0.86 (i.e., negative-folds-penalty disappears
because dMamba is positive on all 7 test folds). Confirms the FX
project's Mamba-phase finding (FX champion was dmamba e=4 composite
+5.5996) transfers to QQQ.

## Files

- `code/` — frozen snapshot of the package at the time of the win
  (`download.py`, `features.py`, `splits.py`, `metrics.py`,
  `run_autoresearch.py`, `CLAUDE.md`).
- `model_checkpoint.pt` — weights + scaler_mean + scaler_scale +
  feature_columns + config + provenance, fully self-contained.
- `README.md` — this file.
- (planned) `audit_report.md` — 14-section audit per CLAUDE.md spec.
- (planned) `inference/predict.py` — standalone inference script.
- (planned) `colab_train_and_infer.ipynb` — self-contained Colab notebook.

## Reproduce

```bash
cd C:/Users/evija/autoresearch
"C:/Users/evija/anaconda3/python.exe" -u -m autoresearchindexstock.run_autoresearch \
  --backbone mamba --seq-len 60 --lr 5e-4 --bs 32 --epochs 100 --patience 20 \
  --weight-decay 0.1 --warmup-epochs 10 --head-dropout 0.1 --seed 42 \
  --mamba-variant dmamba --expand 4 \
  --description "dMamba reproduction"
```

Expected runtime ~40 min on RTX 4090 Laptop.

## Caveats

- **Single-seed result.** Per CLAUDE.md "single-seed champions are often
  luck"; multi-seed verification (seeds 0, 99) is the next step. We expect
  Mamba to be more stable than MLP based on FX's mamba-phase 7-seed
  variance study (mean +4.45, std +0.89 — moderate but not extreme).
- **Excess-Sharpe is negative** (-0.36) — strategy underperforms passive
  QQQ over the full super-fold. The dMamba win is on COMPOSITE (which
  rewards consistent positive folds) not on excess vs buy-and-hold.
- **Late-arriving CatBoost full + XGBoost full are still pending** —
  this could change the per-backbone champion table.
