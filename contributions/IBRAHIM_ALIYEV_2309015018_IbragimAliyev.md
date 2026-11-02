# Contribution: Ibrahim Aliyev

- **Student ID:** 2309015018
- **GitHub:** [IbragimAliyev](https://github.com/IbragimAliyev)
- **Role:** Fine-tuning of BERTurk and XLM-R with and without generated data, data-size selection on validation, blind manual annotation

---

## Summary

I fine-tuned the two transformer encoders of the project, BERTurk and XLM-R, on the TurEngMix training
split with 0, 2,500, 5,000 and 10,000 of our generated sentences added: 6 configurations × 4 seeds =
**24 fine-tuning runs**, all with the TurEngMix paper's training recipe. My real-data-only BERTurk run
reproduces the paper (0.471 vs 0.4734 test macro-F1), and adding 5,000 generated sentences raises
test macro-F1 to 0.550 (BERTurk) and 0.564 (XLM-R). I chose the data size on the validation split only,
and I was one of the three native-speaker annotators of the agreement study (402 items).

---

## 1. Fine-Tuning Setup (`src/train_encoder.py`)

- **Models:** BERTurk (`dbmdz/bert-base-turkish-cased`, Turkish BERT) and XLM-R
  (`FacebookAI/xlm-roberta-base`, multilingual RoBERTa), both as token classifiers over the six labels.
- **Hardware:** Google Colab, T4 GPU, mixed precision (`--fp16`).
- **Training recipe (unchanged from the paper, so results are comparable):** learning rate 2e-5, weight
  decay 0.01, warm-up 6%, batch size 16, at most 10 epochs, early stopping after 3 epochs without
  improvement, best epoch chosen by macro-F1 on the **real** TurEngMix validation split, seeds 1–4.
- **Sub-word alignment:** the label goes on the first sub-word of each word; the other sub-words get
  `-100` and are ignored in the loss.
- **Generated data:** `--extra-n N` adds the first N sentences of `data/generated/generated.jsonl`
  (nested sets: 2,500 ⊂ 5,000 ⊂ 10,000), so only the amount of data changes between runs.
- For every run the script saves validation and test predictions (`results/predictions/`) and a log
  with settings and training history (`results/runs/*.json`), then deletes the model files to keep the
  Colab disk free.

---

## 2. Runs and Results

Test macro-F1 on the 25 TurEngMix test posts (1,373 words), mean of 4 seeds, scored by
`src/score_all.py`:

| # | Run | Generated sentences | Validation macro-F1 | Test macro-F1 [95% CI] | `MIXED` F1 |
|--:|---|--:|--:|--:|--:|
| 1 | `berturk_gen0` | 0 | 0.5085 | **0.471** [0.42, 0.51] | 0.22 |
| 2 | `xlmr_gen0` | 0 | 0.4711 | 0.430 [0.39, 0.45] | 0.04 |
| 3 | `berturk_gen2500` | 2,500 | 0.6032 | 0.521 [0.45, 0.58] | 0.47 |
| 4 | `berturk_gen5000` | 5,000 | **0.6613** | 0.550 [0.49, 0.60] | 0.55 |
| 5 | `berturk_gen10000` | 10,000 | 0.6297 | 0.556 [0.49, 0.62] | 0.58 |
| 6 | `xlmr_gen5000` | 5,000 | 0.6323 | **0.564** [0.50, 0.62] | 0.59 |

- **Reproduction:** `berturk_gen0` scores 0.471; the paper reports 0.4734 ± 0.0355.
- **Generated data helps both models:** BERTurk +0.079 (paired 95% CI +0.04 to +0.13), XLM-R +0.134.
  The largest gain is on `MIXED`: 0.22 → 0.55 (BERTurk) and 0.04 → 0.59 (XLM-R).
- **Every size beats real data only**, on validation and on test.
- Run 3 was trained twice: once on the first 2,500 sentences, and again on the final dataset after the
  label fixes described in `data/generated/README.md`. The table shows the final run.
- Without generated data, all 8 runs used the full 10 epochs; with generated data, most runs stopped
  early (4–9 epochs), since the larger training set reaches its best validation score sooner.

---

## 3. Data-Size Selection (`results/ibrahim_size_choice.md`)

- **Rule:** for each size, average BERTurk's best validation macro-F1 over the 4 seeds; the highest
  mean wins. Test scores were not used.
- **Result:** 0 → 0.5085, 2,500 → 0.6032, **5,000 → 0.6613**, 10,000 → 0.6297. Run 6 (XLM-R) therefore
  used 5,000 sentences.
- **Caveat:** 5,000 also had the largest spread between seeds (0.57–0.75) and the validation split is
  small, so the gap to 10,000 is not clear-cut; on test the two are almost equal (0.550 vs 0.556).

---

## 4. Blind Manual Annotation (`annotation/annotation_ibrahim.xlsx`)

- Labelled all 402 items (202 real TurEngMix words and 200 generated words) alone, without AI and
  without knowing which items were real, following `report/annotation_guide.md`.
- Against our generated labels: accuracy 0.770, `MIXED` F1 0.896. Against TurEngMix gold on real
  items: accuracy 0.668, `MIXED` F1 0.667. Full results in `results/human_agreement.md`.

---

## Deliverables

- **Predictions (`results/predictions/`):** `berturk_gen{0,2500,5000,10000}_seed{1-4}_{validation,test}.csv`,
  `xlmr_gen{0,5000}_seed{1-4}_{validation,test}.csv`
- **Training logs (`results/runs/`):** 24 `*.json` files with settings and validation history
- **Report:** `results/ibrahim_size_choice.md`
- **Annotation:** `annotation/annotation_ibrahim.xlsx`
