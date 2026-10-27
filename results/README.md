# results/

| Folder / file | What goes here | Who |
|---|---|---|
| `predictions/` | One CSV per model and split: `<run>_validation.csv`, `<run>_test.csv` | everyone with a model |
| `runs/` | Training logs from `src/train_encoder.py` (`<run>.json`) | Ibrahim |
| `figures/` | Figures for the README and presentation | Seleni |
| `results_table.csv` / `.md` | Final scores, made by `python src/score_all.py` | Shahem |
| `*.md` notes | e.g. `ibrahim_size_choice.md`, `crf_settings.md`, `zemberek_coverage.md`, `error_analysis.md`, `human_agreement.md` | as in `TASKS.md` |

Don't edit prediction files by hand.
