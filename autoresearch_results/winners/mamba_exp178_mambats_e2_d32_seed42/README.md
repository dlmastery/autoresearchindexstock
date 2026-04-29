# QQQ Winner Archive: mamba_exp178_mambats_e2_d32_seed42

## Headline

QQQ Mamba MAMBATS variant single-seed=42. mambats + expand=2 + d_state=32 ; composite **+0.4193**, A_sharpe +0.62, **excess +0.015 (POSITIVE)**, 5/7 positive folds. Per Cai et al. 2024 NeurIPS MambaTS (arXiv:2405.16440) — season-trend decomposition variant.

## Configuration (exp 178)

- seq_len: 10
- lr: 0.0003
- batch_size: 32
- epochs: 50
- weight_decay: 1e-05
- patience: 10
- warmup_epochs: 0
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
| fold_1 | GFC peak crash (Lehman + Mar-2009 bottom) | -0.746 | -0.302 |
| fold_2 | 2011 US-downgrade + EU debt | +1.417 | -0.315 |
| fold_3 | Taper tantrum and 2014 H1 | +2.099 | +0.755 |
| fold_4 | China devaluation and oil crash | +0.411 | +0.401 |
| fold_5 | 2018 Vol-mageddon + Q4 sell-off | -0.066 | -0.375 |
| fold_6 | COVID crash and V-recovery | +0.584 | -0.304 |
| fold_7 | Inflation bear, AI rally and 2025 | +1.442 | +0.470 |

## Multi-seed verification

See `reproduction/multi_seed_summary.json` for the multi-seed sweep on this config.

## Notes

Complementary variant to dmamba: mambats wins F3 Taper (+2.10) and F7 AI-rally (+1.44) via trend-component but loses F1 GFC (-0.75). 2-seed mean +0.38 (reproducible). Useful for ensemble alongside dmamba which has opposite per-fold signature.

## Files

- config.json — exact config that produced this result
- experiment_log_entry.json — full JSONL row for exp 178
- per_fold_results.json — per-fold + per-window breakdown
- reproduction/multi_seed_summary.json — multi-seed sweep summary
- reproduction/exp178_trades.csv — per-trade test-set log
- reproduction/exp178_trade_summary.json — per-fold trade statistics
- code/ — frozen snapshot of runner, model, data, evaluation, CLAUDE.md
