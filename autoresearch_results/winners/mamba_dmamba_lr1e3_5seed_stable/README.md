# QQQ Stable Champion: dMamba lr=1e-3 (5-seed median +0.38)

## Headline (exp 252, 256, 257, 258, 259)

5-seed dMamba lr=1e-3 distribution:
- seed=42: comp +0.164 (F2 RECORD +6.61)
- seed=0:  comp +0.262
- seed=99: comp +0.605 (champion seed=99 was -1.15 catastrophic!)
- seed=7:  comp +0.591
- seed=2024: comp +0.382

**Median +0.38, Mean +0.40, ALL 5 POSITIVE.**

## Why this is the new stable champion

Compared to single-seed champion (exp 52, lr=5e-4 +1.32):
- Exp 52 4-seed median: -0.25 (seed=99 catastrophic at -1.15)
- This config 5-seed median: +0.38 (no seed below +0.16)

Trades peak Sharpe for cross-seed robustness — deployment-grade stability.

## Configuration

```bash
python -m autoresearchindexstock.run_autoresearch   --backbone mamba --mamba-variant dmamba   --hidden-size 128 --num-layers 2 --d-state 32 --expand 2   --lr 0.001 --bs 32 --epochs 100 --patience 20   --weight-decay 0.1 --head-dropout 0.1 --warmup-epochs 10   --seq-len 60 --seed <42|0|99|7|2024>
```

## Citations

- Gu, Dao 2024 COLM Mamba (arXiv:2312.00752) §3.2 — selective state mechanism
- Liu, Zhang, Wu, Long 2025 DMamba (arXiv:2602.09081) — decomposition variant
- Smith 2017 (arXiv:1803.09820) §3 — lr leverage axis (highest-impact HP)
- Lakshminarayanan, Pritzel, Blundell 2017 NeurIPS (arXiv:1612.01474) — multi-seed ensembles
