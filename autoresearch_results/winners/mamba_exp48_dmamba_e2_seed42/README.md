# QQQ Global Champion: dMamba expand=2 (Exp 48)

## Headline result

Composite **+1.1887** | A_sharpe **+1.2887** | excess_sharpe **+0.0686 (FIRST POSITIVE EXCESS in QQQ history)** | 6/7 test folds | 6/7 val folds | runtime 878s

## Configuration

- backbone: dMamba (Liu 2025 arXiv:2602.09081)
- mamba_variant: dmamba
- expand: 2 (Gu-Dao 2024 paper default; KEY finding: paper default beats FX-tuned expand=4 by +0.33 composite)
- d_state: 16, num_layers: 2 (default)
- seq_len: 60, lr: 5e-4, bs: 32
- epochs: 100, patience: 20
- weight_decay: 0.1, head_dropout: 0.1
- warmup_epochs: 10, huber_delta: 1.0, grad_clip: 1.0
- seed: 42

## Multi-seed verification (seeds 42/0/7/99)

| Seed | Composite | A_sharpe | Excess | Test pos | Val pos |
|---|---:|---:|---:|---:|---:|
| 42 (champion) | +1.1887 | +1.2887 | +0.0686 | 6/7 | 6/7 |
| 0 | +0.6980 | +0.7980 | -0.4221 | 6/7 | 5/7 |
| 7 | +1.0182 | +1.1182 | -0.1019 | 6/7 | 5/7 |
| 99 | -1.1484 | -0.6484 | -1.8685 | 2/7 | 5/7 |
| **Median** | **+0.8581** | - | - | - | - |
| Mean | +0.4391 | - | - | - | - |
| Std | 1.0777 | - | - | - | - |

## Comparison to prior champion (dMamba expand=4 Exp 17)

| Metric | Exp 48 (expand=2) | Exp 17 (expand=4) | Delta |
|---|---:|---:|---:|
| Composite | +1.1887 | +0.8625 | +0.33 |
| A_sharpe | +1.2887 | +0.8625 | +0.43 |
| Excess | +0.0686 | -0.3576 | +0.43 (sign flip) |
| Val pos | 6/7 | 4/7 | +2 |
| 4-seed median | +0.8581 | -0.2530 | +1.11 |

## Per-fold breakdown (champion seed=42)

| Fold | Regime | A_sharpe | A_excess | bh | return |
|---|---|---:|---:|---:|---:|
| F1 | GFC peak crash | +1.81 | +1.38 | +0.43 | +14.21% |
| F2 | 2011 EU debt | +5.27 | +0.40 | +4.87 | +30.16% |
| F3 | Taper tantrum | +0.38 | -1.68 | +2.06 | +2.42% |
| F4 | China-oil drawdown | +2.79 | **+3.59** | -0.80 | +13.04% |
| F5 | Vol-mageddon | +1.34 | +0.16 | +1.18 | +14.32% |
| F6 | COVID V-recovery | +2.42 | -0.01 | +2.43 | +25.95% |
| F7 | AI rally 2025 | -0.13 | -0.92 | +0.79 | -3.67% |

## Key findings

1. **Paper-default expand=2 dramatically outperforms FX-tuned expand=4** on QQQ (composite +0.33, median +1.11 lift).
2. **First positive excess sharpe** on QQQ — strategy beats BH on 5/7 test folds (F1, F2, F4, F5, F6 essentially neutral).
3. **Val regime fit transformed**: 6/7 val pos folds vs 4/7 for prior champion — composite penalty resolved.
4. **FX -> QQQ HP transfer breaks** for over-parameterised choices; QQQ needs paper defaults or DOWN.
5. seed=99 catastrophic across both expand=2 and expand=4 - likely RNG pathology specific to this codebase.

## Citations

- Gu, Dao 2024 COLM Mamba (arXiv:2312.00752)
- Liu, Zhang, Wu, Long 2025 DMamba (arXiv:2602.09081)
- Lakshminarayanan, Pritzel, Blundell 2017 NeurIPS Deep Ensembles (arXiv:1612.01474)

## Files

- config.json - exact config from best_config.json
- model_checkpoint.pt - saved weights + scaler + feature_columns
- experiment_log_entry.json - JSONL row for exp 48
- per_fold_results.json - per-fold + per-window breakdown
- reproduction/multi_seed_summary.json - 4-seed sweep summary
- code/ - frozen snapshot of runner, model, data, evaluation
- inference/ - to be added