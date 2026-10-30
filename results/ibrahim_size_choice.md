# Choice of generated-data size for run 6 (XLM-R)

Rule (from `TASKS.md`): for each size, average the best **validation** macro-F1 of BERTurk over
the 4 seeds (`best_validation_macro_f1` in `results/runs/berturk_gen<N>_seed<S>.json`). The highest
average wins. Test scores were not used.

All runs: `dbmdz/bert-base-turkish-cased`, `src/train_encoder.py` with its default (paper) settings,
`--fp16`, seeds 1–4, on the final `data/generated/generated.jsonl` (10,004 sentences, so gen2500 was
re-run after the label fixes described in `data/generated/README.md`).

| Generated sentences | Seed 1 | Seed 2 | Seed 3 | Seed 4 | **Mean** | SD |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.4917 | 0.5011 | 0.5229 | 0.5182 | 0.5085 | 0.0146 |
| 2,500 | 0.6484 | 0.5594 | 0.5711 | 0.6339 | 0.6032 | 0.0445 |
| 5,000 | 0.7371 | 0.5730 | 0.5883 | 0.7468 | **0.6613** | 0.0934 |
| 10,000 | 0.5972 | 0.6000 | 0.6146 | 0.7068 | 0.6297 | 0.0520 |

**Choice: 5,000.** Run 6 is `xlmr_gen5000`.

Notes:
- Every size with generated data beats real data only (0.5085) on validation.
- 5,000 has the highest mean but also the largest spread between seeds (0.57–0.75), and the
  validation split is small (26 sentences), so the gap to 10,000 (0.6297) is not clear-cut.
  The rule above was applied as written.
