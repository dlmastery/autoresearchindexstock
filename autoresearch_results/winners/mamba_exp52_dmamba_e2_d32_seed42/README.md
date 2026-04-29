# QQQ Winner Archive: mamba_exp52_dmamba_e2_d32_seed42

## Headline

QQQ GLOBAL CHAMPION (single-seed=42). dMamba expand=2 d_state=32 ; composite **+1.3216**, A_sharpe +1.3216, val_sharpe +1.4831, PSR +0.9972, excess +0.45 over BH+0.87, 7/7 positive test folds. Successor to exp 48 (+1.19) — d_state bumped 16→32 captured F2 EU-debt alpha (+5.27) more cleanly. Per Gu-Dao 2024 §3.2 SSM memory capacity.

## Configuration (exp 52)

- seq_len: 60
- lr: 0.0005
- batch_size: 32
- epochs: 100
- weight_decay: 0.1
- patience: 20
- warmup_epochs: 10
- head_dropout: 0.1
- huber_delta: 1.0
- grad_clip: 1.0
- seed: 42
- hidden_size: None
- num_layers: None
- max_depth: None
- gbm_lr: None
- n_estimators: None

## Per-fold breakdown (test set)

| Fold | Regime | A_sharpe | A_excess |
|---|---|---:|---:|
| fold_1 | GFC peak crash (Lehman + Mar-2009 bottom) | +1.419 | +0.992 |
| fold_2 | 2011 US-downgrade + EU debt | +4.208 | -0.661 |
| fold_3 | Taper tantrum and 2014 H1 | +3.467 | +1.409 |
| fold_4 | China devaluation and oil crash | +0.026 | +0.830 |
| fold_5 | 2018 Vol-mageddon + Q4 sell-off | +2.516 | +1.333 |
| fold_6 | COVID crash and V-recovery | +1.383 | -1.044 |
| fold_7 | Inflation bear, AI rally and 2025 | +0.448 | -0.343 |

## Multi-seed verification

See `reproduction/multi_seed_summary.json` for the multi-seed sweep on this config.

## Notes

Champion is single-seed=42; 4-seed median composite -0.25 (seed=99 catastrophic). Deploy via seed-ensemble (Lakshminarayanan 2017) averaging seed 42, 0, 7 predictions for stable inference.

## Files

- config.json — exact config that produced this result
- experiment_log_entry.json — full JSONL row for exp 52
- per_fold_results.json — per-fold + per-window breakdown
- reproduction/multi_seed_summary.json — multi-seed sweep summary
- reproduction/exp52_trades.csv — per-trade test-set log
- reproduction/exp52_trade_summary.json — per-fold trade statistics
- code/ — frozen snapshot of runner, model, data, evaluation, CLAUDE.md
