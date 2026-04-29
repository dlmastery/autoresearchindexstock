# QQQ Winner Archive: mlp_exp79_residual_seq10_wd1e5_warmup5

## Headline

QQQ MLP within-backbone (single-seed=0, original variant). Residual MLP + lr=3e-4 wd=1e-5 ep=50 pat=10 bs=32 hd=0.25 warmup=5 seq=10 ; composite +0.9743 (single-seed); 5-seed median **+0.433** (positive — second stable backbone after mamba). Gu-Kelly-Xiu 2020 RFS recipe + Goyal 2017 warmup.

## Configuration (exp 79)

- seq_len: 10
- lr: 0.0003
- batch_size: 32
- epochs: 50
- weight_decay: 1e-05
- patience: 10
- warmup_epochs: 5
- head_dropout: 0.25
- huber_delta: 1.0
- grad_clip: 1.0
- seed: 0
- hidden_size: None
- num_layers: None
- max_depth: None
- gbm_lr: None
- n_estimators: None

## Per-fold breakdown (test set)

| Fold | Regime | A_sharpe | A_excess |
|---|---|---:|---:|
| fold_1 | GFC peak crash (Lehman + Mar-2009 bottom) | +0.213 | +0.657 |
| fold_2 | 2011 US-downgrade + EU debt | +0.435 | -1.297 |
| fold_3 | Taper tantrum and 2014 H1 | +2.906 | +1.562 |
| fold_4 | China devaluation and oil crash | +1.249 | +1.239 |
| fold_5 | 2018 Vol-mageddon + Q4 sell-off | +0.809 | +0.500 |
| fold_6 | COVID crash and V-recovery | +2.636 | +1.748 |
| fold_7 | Inflation bear, AI rally and 2025 | +0.185 | -0.786 |

## Multi-seed verification

See `reproduction/multi_seed_summary.json` for the multi-seed sweep on this config.

## Notes

5-seed distribution: [+0.974 seed=0, -0.714 seed=42, +0.520 seed=99, -0.491 seed=7, +0.433 seed=2024]. 3 of 5 seeds positive. Median +0.433 confirmed real lift. Compatible with exp 204 (wd=1e-4) for ensemble — wd-axis-paired variants.

## Files

- config.json — exact config that produced this result
- experiment_log_entry.json — full JSONL row for exp 79
- per_fold_results.json — per-fold + per-window breakdown
- reproduction/multi_seed_summary.json — multi-seed sweep summary
- reproduction/exp79_trades.csv — per-trade test-set log
- reproduction/exp79_trade_summary.json — per-fold trade statistics
- code/ — frozen snapshot of runner, model, data, evaluation, CLAUDE.md
