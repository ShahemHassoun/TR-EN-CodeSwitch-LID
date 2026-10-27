# Project 2 — Task Split

**Project:** token-level language identification for Turkish–English code-mixed text.
Every word gets one of six labels: `TR`, `EN`, `MIXED`, `NE`, `AMBIGUOUS`, `OTHER`.
`MIXED` = an English word with a Turkish ending (`linki`, `meeting'e`). This is the hard class,
and the one our project is about.

**Data:** TurEngMix (Ekşi Sözlük posts, 2026) is our seed data and our real test set. We build our
own generated dataset (10,000 sentences) and test whether adding it helps models on real text.

**Presentation: Monday 2 November.** All work uploaded to GitHub by **Saturday 31 October**.

| Member | Task | Due |
|---|---|---|
| **Shahem** | Dataset (10,000 sentences), shared code, final results, README, presentation | Thu–Sat |
| **Ibrahim** | Fine-tune BERTurk and XLM-R, with and without our data + annotation | Fri |
| **Mahmut** | Zemberek / dictionary baseline, CRF baseline, error analysis + annotation | Fri–Sat |
| **Tuğra** | GPT-4o baseline + human-agreement study + annotation | Fri |
| **Seleni** | Figures, licence and citation files, contributions page | Sat |

---

## Before you start (everyone)

1. Read `report/annotation_guide.md` (10 minutes) so you know exactly what the six labels mean.
2. **Upload your work to GitHub yourself** when it's done. Put each file in the folder named in
   your section, and only upload or edit **your own** files. Message Shahem when it's up.
3. Don't change the shared files: `src/lid_io.py`, `src/train_encoder.py`, `src/score_all.py`,
   `turengmix/` and `data/`. If one of them gives you an error, send the error to Shahem.
4. **Never use the test split to make a decision.** Choose settings with `train` and
   `validation` only. The test split is only for the final scores.
5. **Never upload API keys, passwords or tokens.** Keys go only in a local `.env` file.
6. Use the file names exactly as written. The scoring script finds results by their names.

**How results work:** every model saves its predictions with `save_predictions()` from
`src/lid_io.py` into `results/predictions/`. Then one script (`src/score_all.py`) scores all
models the same way: the same 25 test posts, TurEngMix's own scoring code, and macro-F1 with a
95% confidence interval. You don't need to calculate final scores yourself.

---

## Shahem — dataset, shared code, results, README, presentation

- [x] Repo, TurEngMix seed data, licence notices
- [x] Label rules checked against TurEngMix's real data (`report/annotation_guide.md`)
- [x] Shared code: data loading, training script, scoring script, data checker (all tested)
- [ ] Annotation files for Ibrahim, Mahmut and Tuğra (Wednesday)
- [ ] First 2,500 generated sentences (Wednesday), all 10,000 (Thursday), passing the checker
- [ ] Dataset description `data/generated/README.md`; prompt with our examples for Tuğra
- [ ] Final scoring of all models, `README.md`, presentation

---

## Ibrahim — BERTurk and XLM-R fine-tuning

**Question you answer:** does adding our generated data make a fine-tuned model better on
*real* TurEngMix text, and how much data is best?

**How:** use the shared training script `src/train_encoder.py` on Google Colab (T4 GPU). It
already uses the **same training settings as the TurEngMix paper** (learning rate 2e-5, up to 10
epochs, early stopping, best epoch picked on the validation split, 4 seeds). Keep those settings
unchanged so our results are comparable with the paper. The script's top comment explains its
options. For each run it saves validation and test predictions and deletes the large model files.

| # | Model | Generated sentences added | When |
|---|---|---|---|
| 1 | BERTurk (`dbmdz/bert-base-turkish-cased`) | 0 (real data only) | Wednesday |
| 2 | XLM-R (`FacebookAI/xlm-roberta-base`) | 0 | Wednesday |
| 3 | BERTurk | 2,500 | after the first 2,500 are on GitHub |
| 4 | BERTurk | 5,000 | after all 10,000 are on GitHub |
| 5 | BERTurk | 10,000 | same |
| 6 | XLM-R | the best size from runs 3–5 | Friday |

**Sanity check after run 1:** BERTurk with real data only should score roughly 0.43–0.50 test
macro-F1. The paper reports 0.4734 ± 0.0355. If you get something far off, tell Shahem before
continuing.

**Choosing the best size (run 6):** for each size, average the best **validation** score over the
4 seeds (it's in `results/runs/*.json`). The highest average wins. Don't use test scores for this.
Write the four averages and your choice in `results/ibrahim_size_choice.md`.

**Upload:** all `results/predictions/berturk_*` and `xlmr_*` files, all `results/runs/*.json`
files, and `results/ibrahim_size_choice.md`. **By Friday.**

---

## Mahmut — classical baselines and error analysis

**Question you answer:** how far do simple methods get without deep learning? This includes
Zemberek, which the professor asked us to check.

**Setup notes:** use **Python 3.12**. Zemberek's Python version (`zemberek-python`) doesn't work
on 3.13 or newer, and it needs `setuptools<81`. Load data and save predictions with
`src/lid_io.py` (`load_split_sentences`, `load_generated`, `save_predictions`).

### 1. Rule-based baselines: `src/baseline_rules.py` (Wednesday–Thursday)

Two fixed rule systems with no learning. Write the rules first, then run them. Don't tune them by
looking at results.
- **`rules_wordfreq`** (dictionary only): use the `wordfreq` package to compare how common a word
  is in English and in Turkish. Label it NE if it's in a list of names taken from the **train**
  split, MIXED if it has an apostrophe and the part before it is an English word, EN if it is more
  common in English, and TR otherwise.
- **`rules_zemberek`**: the same, plus Zemberek. If Zemberek can analyse the word as Turkish
  (not as a proper noun), label it TR unless it is clearly more English.
- Save predictions for `validation` and `test`.
- Also write `results/zemberek_coverage.md`: on the train split, the share of Turkish words that
  Zemberek recognises, **with** vs **without** Turkish letters (ç ğ ı ö ş ü). We expect it to fail
  on Turkish typed without Turkish letters (`henuz`). That's a good finding for the presentation.

### 2. CRF baseline: `src/baseline_crf.py` (Thursday–Friday)

A CRF (`sklearn-crfsuite`), like the feature-based taggers in the lecture.
- Features come from the word itself and its neighbours: prefixes and suffixes, apostrophe and the
  ending after it, Turkish letters, length, `wordfreq` English and Turkish frequency, and the same
  for the previous and next word.
- Pick the regularisation settings (`c1`, `c2`) with a small search on **validation**, then keep
  them fixed.
- Four runs, trained on TurEngMix train plus 0 / 2,500 / 5,000 / 10,000 generated sentences:
  `crf_gen0`, `crf_gen2500`, `crf_gen5000`, `crf_gen10000`. Save validation and test predictions.
  Write the chosen settings in `results/crf_settings.md`.

### 3. Error analysis: `notebooks/error_analysis.ipynb` (Saturday, after Ibrahim's runs)

This part needs a Turkish speaker. Compare BERTurk without our data (`berturk_gen0_seed1`) and with
it (the best size, seed 1) on the test split. List every token where the gold label or a
prediction is MIXED, and give each error a short category (apostrophe / no apostrophe / name vs
MIXED / no Turkish letters / gold label looks wrong / other). Save it as `results/error_analysis.md`
with a short paragraph: what our data fixed, and what it didn't.

**Upload:** the two scripts, the notebook, all `rules_*` and `crf_*` prediction files, and the
three `.md` files.

---

## Tuğra — GPT-4o baseline and human-agreement study

### 1. GPT-4o (Thursday–Friday)

Use TurEngMix's own LLM script, `src/run_llm.py`, and their prompts, so it's a fair replication.
Important difference from the paper: the paper scored GPT-4o on all 250 posts, but we score it
**on the 25 test posts only**, like every other model.
- Three runs on the **test split**: zero-shot (`gpt4o_0shot_test`), three-shot with the paper's
  examples (`gpt4o_3shot_test`), and three-shot with examples from **our** dataset
  (`gpt4o_3shot-ours_test`, prompt file `prompts/lid_few_shot_ours.txt`, which Shahem uploads on
  Thursday). The script has a `--dry-run` option to check everything before spending money.
- Needs an OpenAI **API** key in a `.env` file. All three runs together cost well under $1.
- Each run reports `sentences aligned X/41` at the end. If X is below 39, tell Shahem.

### 2. Human agreement: `src/agreement.py` (Thursday–Friday, after annotations come back)

Shahem gives you the three filled annotation files and an answer key (`annotation/key.csv`). The
key says which items are real TurEngMix text and which are ours. Compute:
1. **Agreement between the three annotators** (Cohen's kappa for each pair and the average),
   overall and **per label**, separately for real and generated items.
2. **Annotators vs TurEngMix gold** on real items: this is our measured human ceiling.
3. **Annotators vs our labels** on generated items, plus the majority vote: this measures how
   correct our generated labels are.
4. How many items each label has, so small groups are visible.

Save as `results/human_agreement.md` and `results/human_agreement.json`.

**Upload:** all `results/predictions/gpt4o_*` files, `src/agreement.py`, and both agreement files.
**By Friday.**

---

## Annotation: Ibrahim, Mahmut, Tuğra (about 1 hour each)

On Wednesday Shahem sends each of you an Excel file of about 400 rows. Each row shows a sentence
with one word marked `[[like this]]`, and you pick its label from a drop-down.
- Work **alone**. Don't discuss items or compare answers. We're measuring how much native
  speakers naturally disagree, which is the ceiling no model can be expected to beat.
- Answer every row. If unsure, give your best guess.
- Use your own judgement, not AI. That's the point of this part.
- Send it back within a day.

---

## Seleni — figures, licences, contributions

### 1. Licence and citation files (Wednesday)
- `LICENSE` (repo root): standard MIT License, "Copyright (c) 2026 TR-EN-CodeSwitch-LID team",
  plus one line saying the code in `turengmix/`, `src/run_llm.py` and `prompts/` comes from
  TurEngMix (MIT, see `turengmix/LICENSE_TURENGMIX_MIT.txt`).
- `data/generated/LICENSE.md`: our generated dataset is released under **CC BY-NC-SA 4.0**
  (https://creativecommons.org/licenses/by-nc-sa/4.0/), because it was built using TurEngMix
  (also CC BY-NC-SA 4.0) as seed data.
- `CITATION.md`: the TurEngMix paper (*TurEngMix: A Text Corpus and Benchmark for
  Turkish-English Code-Mixed Language Identification and Named Entity Recognition*, Ilayda Dogan,
  Phuong-Anh Nguyen-Le and Julia Mendelsohn, University of Maryland, 2026, arXiv:2609.06963,
  https://github.com/kemnguyenle/TurEngMix), Zemberek (https://github.com/ahmetaa/zemberek-nlp),
  BERTurk (`dbmdz/bert-base-turkish-cased`) and XLM-R (`FacebookAI/xlm-roberta-base`).

### 2. Figures: `src/plot_figures.py` (Friday–Saturday)
A small matplotlib script that reads `results/results_table.csv` (made by `src/score_all.py`) and
saves three images in `results/figures/`:
1. `fig1_macro_f1.png`: one bar per model, test macro-F1 with its confidence interval, best at top.
2. `fig2_data_size.png`: line chart of BERTurk (and CRF, if available) scores against the number of
   generated sentences added (0, 2,500, 5,000, 10,000): overall macro-F1 and MIXED F1.
3. `fig3_confusion.png`: two confusion matrices (BERTurk without and with our data), in percent
   per row.

Start building it as soon as a few results exist. It should skip any results that are missing
instead of crashing. Shahem tells you which run is "best" for figure 3.

### 3. Contributions: `contributions/CONTRIBUTIONS.md` (Saturday)
Ask each member for 2–3 sentences on what they did, and put them under each name. Add your own.

**Upload:** `LICENSE`, `data/generated/LICENSE.md`, `CITATION.md`, `src/plot_figures.py`,
`results/figures/*.png`, `contributions/CONTRIBUTIONS.md`.

---

## Timeline

| When | What |
|---|---|
| **Wed** | Annotation files out · first 2,500 sentences · Ibrahim runs 1–3 · Mahmut rules · Seleni licences |
| **Thu** | Annotations back · all 10,000 sentences · Ibrahim runs 4–5 · Tuğra GPT-4o |
| **Fri** | Ibrahim run 6 · Mahmut CRF · Tuğra agreement · all results on GitHub |
| **Sat** | Final scoring · error analysis · figures · README · everything uploaded |
| **Mon 2 Nov** | **Presentation** |
