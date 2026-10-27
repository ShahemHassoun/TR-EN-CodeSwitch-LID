# Data

| Path | What it is | Licence |
|---|---|---|
| `benchmark.csv` | TurEngMix benchmark: 250 Ekşi Sözlük posts, 15,012 tokens, one row per token. Copied unchanged. | CC BY-NC-SA 4.0 (`TURENGMIX_DATA_LICENSE`) |
| `splits/{train,validation,test}.txt` | TurEngMix's official document-level split (200 / 25 / 25 posts). Copied unchanged. | CC BY-NC-SA 4.0 |
| `generated/` | Our generated dataset (seeded by the TurEngMix **train** split only). | CC BY-NC-SA 4.0 |

Source: https://github.com/kemnguyenle/TurEngMix (commit `34f389d`), paper arXiv:2609.06963
(Dogan, Nguyen-Le and Mendelsohn, University of Maryland, 2026).

Read `benchmark.csv` with `keep_default_na=False`: tokens such as `nan` and `null` occur in the text.
The loaders in `src/lid_io.py` and `turengmix/data.py` already do this.

**Use of the splits in this project:** `train` = seed data and training data · `validation` =
choosing settings · `test` = final scores only, used once.
