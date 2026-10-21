# Results

## Model quality (test split: 56,962 rows, ~98 frauds; single seed)
| Model | PR-AUC | R@P>=0.5 | R@P>=0.8 | R@P>=0.9 |
|---|---|---|---|---|
| LogReg (balanced) | 0.728 | 0.878 | 0.816 | 0.000 |
| MLP 29-64-32-1 (untuned) | 0.860 | 0.878 | 0.816 | 0.765 |
| XGBoost (spw=full, 841 trees) | 0.885 | 0.898 | 0.878 | 0.837 |

Notes
- Test set has ~98 frauds: PR-AUC gaps under ~0.03 are within noise. XGBoost vs MLP is not a significant difference.
- XGBoost scale_pos_weight settings (1 / sqrt / full) were statistically tied (val PR-AUC 0.842-0.843); selection used a validation slice of the train split, never the test set.
- Time column dropped (seconds since first transaction; does not generalise to new traffic).

## ONNX parity (max |native - onnx| over the test split)
| Model | Max score diff |
|---|---|
| LogReg | 3.4e-06 |
| XGBoost | 3.4e-07 |
| MLP | 1.4e-07 |

LogReg PR-AUC differs slightly (0.7280 native vs 0.7264 ONNX) despite ~3e-6 score differences; cause not investigated.

## Gateway smoke test (single request, not a benchmark)
Cache MISS 22.68 ms vs HIT 0.69 ms (includes cold-start effects). Proper latency/throughput numbers: Phase 7.

## Latency / throughput
TODO (Phase 7): sklearn / XGBoost native vs ONNX Runtime vs Snap ML, amd64 only.
