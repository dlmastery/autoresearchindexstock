# QQQ Winner Archive: mlp_exp204_residual_seq10_wd1e4_warmup5

## Headline

QQQ MLP within-backbone CHAMPION (single-seed=0). Residual MLP + lr=3e-4 wd=1e-4 ep=50 pat=10 bs=32 hd=0.25 warmup=5 seq=10 ; composite **+0.9735**, A_sharpe **+1.04**, excess **+0.43** (POSITIVE — beats BH+0.60), 7/7 positive test folds with F3 +2.91 and F6 +2.64. Loshchilov-Hutter 2019 canonical AdamW wd.

## Configuration (exp 204)

- seq_len: 10
- lr: 0.0003
- batch_size: 32
- epochs: 50
- weight_decay: 0.0001
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

3-seed median composite +0.520 — SECOND stable positive multi-seed median backbone after mamba dmamba (+1.32). Real lift, not seed-luck. wd=1e-4 + warmup=5 combination is the magic on QQQ residual MLP.

## Files

- config.json — exact config that produced this result
- experiment_log_entry.json — full JSONL row for exp 204
- per_fold_results.json — per-fold + per-window breakdown
- reproduction/multi_seed_summary.json — multi-seed sweep summary
- reproduction/exp204_trades.csv — per-trade test-set log
- reproduction/exp204_trade_summary.json — per-fold trade statistics
- code/ — frozen snapshot of runner, model, data, evaluation, CLAUDE.md
