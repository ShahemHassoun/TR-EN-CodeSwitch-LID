# results/predictions/

Model predictions, saved with `save_predictions()` from `src/lid_io.py`.

Name: `<run>_<split>.csv`, e.g. `berturk_gen2500_seed1_test.csv`, `crf_gen0_validation.csv`,
`rules_zemberek_test.csv`, `gpt4o_0shot_test.csv`.

Each file = the TurEngMix rows of that split plus a `pred_lid` column.
`python src/score_all.py` reads every `*_test.csv` here.
