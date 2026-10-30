# CRF settings and validation results

Regularisation was selected by fitting on the TurEngMix train split and choosing the highest macro-F1 on the validation split. The test split was not used for tuning, evaluation, or model selection.

- **Search space for both `c1` and `c2`:** `0.01, 0.05, 0.1, 0.5, 1`
- **Selected `c1`:** `0.5`
- **Selected `c2`:** `0.01`
- **Best tuning validation macro-F1:** `0.4870` (fixed six-label macro-F1)
- **CRF configuration:** L-BFGS, `max_iterations=100`, `all_possible_transitions=True`

## Model runs

| Run | Generated sentences | Validation macro-F1 |
|---|---:|---:|
| `crf_gen0` | 0 | 0.4870 |
| `crf_gen2500` | 2,500 | 0.4382 |
| `crf_gen5000` | 5,000 | 0.4704 |
| `crf_gen10000` | 10,000 | 0.5149 |

Test predictions were saved but test labels were not evaluated.
