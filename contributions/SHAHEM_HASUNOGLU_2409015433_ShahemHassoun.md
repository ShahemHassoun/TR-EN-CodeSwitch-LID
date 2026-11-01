# Contribution: Shahem Hasunoglu

- **Student ID:** 2409015433
- **GitHub:** [ShahemHassoun](https://github.com/ShahemHassoun)
- **Role:** Project lead and repository owner; generated dataset and its quality checks; shared code (data loading, fine-tuning script, unified scoring); human-annotation sample design; final integration, README and presentation

---

## Summary

I set the research question (does generated training data improve token-level Turkish–English language identification on **real** text?), designed the size experiment (0 / 2,500 / 5,000 / 10,000 generated sentences, chosen on validation only, tested once on the real TurEngMix test split), split the work across the team and owned the repository. I built the 10,004-sentence generated dataset (98,875 tokens) together with the automatic checks that keep it consistent and guard against test leakage, wrote the shared code every model uses for loading, training and scoring, designed the blind 402-item annotation sample, and assembled the final results, README and presentation.

---

## 1. Project Design and Repository (`TASKS.md`, `report/annotation_guide.md`, `data/`)

- **Research question and protocol:** train on TurEngMix train (254 sentences, 12,466 tokens) plus 0 / 2,500 / 5,000 / 10,000 generated sentences; choose settings and data size on the validation split only; report every model once on the 25 real test posts (1,373 tokens) with TurEngMix's own scorer.
- **Benchmark setup:** vendored TurEngMix's data, official document-level splits (200 / 25 / 25 posts) and scoring code unchanged (commit `34f389d`), with their licences (`data/TURENGMIX_DATA_LICENSE`, `turengmix/LICENSE_TURENGMIX_MIT.txt`).
- **Label guide (`report/annotation_guide.md`):** the six labels (TR, EN, MIXED, NE, AMBIGUOUS, OTHER) with edge cases checked against TurEngMix's own labelling, e.g. established loanwords (`internet`, `film`, `telefon`) are TR, a name with a Turkish suffix is NE (`netflix'te`), nationality words are NE, an AMBIGUOUS stem with a Turkish suffix is TR (`platformu`).
- **Task split and timeline (`TASKS.md`):** one owner per part (encoders, rules/CRF/error analysis, GPT-4o/agreement, figures/licences), with the exact files each person delivers.

---

## 2. Generated Dataset (`data/generated/`, `src/build_generated.py`)

- **Writing:** 70 batches of Ekşi Sözlük-style code-mixed sentences on varied topics (work, tech, gaming, series, sport, food, travel, university, money, shopping), labelled inline while writing. Every MIXED word records its English stem and Turkish suffix (`linki|M=link+i`), so its label can be checked automatically. Only the TurEngMix **train** split was used as a style and labelling reference.
- **Size and label mix (gen10000):** 98,875 tokens, about 8× the real train split. TR 79.7%, EN 9.8%, NE 5.7%, MIXED 4.5% (4,409 words), AMBIGUOUS 0.2%, OTHER 0.1%. MIXED is deliberately oversampled compared with real data (1.3%).
- **Nested, frozen subsets:** each batch is shuffled with a fixed seed and appended in order, so gen2500 ⊂ gen5000 ⊂ gen10000. Batches 01–12 were frozen once the first 2,500 sentences were shared, and 01–26 once the first 5,000 were shared, so teammates' runs never changed underneath them.
- **Realism steps in the builder:**
  - 25% of sentences are written without Turkish letters (ç→c, ğ→g, ı→i, ö→o, ş→s, ü→u); 32% of our Turkish words contain Turkish letters (TurEngMix: about 30%).
  - From batch 13 on, 70% of MIXED words (stems of 4+ letters, only when the result still passes the MIXED check) and 30% of suffixed names are written without an apostrophe (`deadline'dan` → `deadlinedan`). Apostrophe-free MIXED words: 56% in gen10000 vs 63% in TurEngMix train.
- **Dataset card (`data/generated/README.md`):** how the data was made, statistics per size, label rules, every post-release change (34 label fixes and 3 re-spellings in the first 2,500, about 0.1% of its tokens) and limitations.

---

## 3. Quality Checks and Leakage Prevention (`src/check_generated.py`)

The checker rejects the whole file if any hard check fails; the final file passes with 0 errors.

- **Format:** malformed lines, duplicate ids or sentences, capital letters, punctuation-only tokens.
- **MIXED validity:** the recorded split must spell the word, the stem must be an English word (wordfreq), and the ending must be a chain of real Turkish suffixes. Measured on TurEngMix's own train data, the check accepts 142 of 149 real MIXED words, and its "possible missed MIXED" warning fires on 38 of 5,157 Turkish words (0.7%), which were reviewed by hand.
- **Copying:** no run of 8 words may be shared with any TurEngMix text.
- **Test leakage:** blocks any word that TurEngMix labels MIXED, NE, OTHER or AMBIGUOUS in validation or test but never in train, ignoring attached punctuation and ’ vs '. One documented exception (`startup'larının`, 1 sentence in the frozen first 2,500) was found by the stricter version of the check after training runs had started; it is listed in the dataset card.
- **Extra checks run on every batch while writing** (helper scripts; results summarised in the dataset card): loanwords tagged EN or MIXED, partially tagged multi-word names, and near-duplicates (sentences sharing at least half their words were rewritten or removed).

---

## 4. Shared Code Used by Every Model

- **`src/lid_io.py`:** loading TurEngMix and the generated data (with `keep_default_na=False`, since tokens such as `nan` and `null` occur in the text) and writing prediction files in one format for all models.
- **`src/train_encoder.py`:** BERTurk / XLM-R fine-tuning adapted from the paper's script with its settings unchanged (learning rate 2e-5, weight decay 0.01, 6% warm-up, batch 16, up to 10 epochs, early stopping with patience 3 on validation macro-F1, first-subword labels). It adds `--extra-n N` to append the first N generated sentences, and records per-run logs in `results/runs/`.
- **`src/score_all.py`:** scores every prediction file with TurEngMix's scorer; groups seeds into mean and SD; computes 95% confidence intervals by resampling the 25 test posts 2,000 times (document-level bootstrap), and the paired difference of every run against real-only BERTurk on the same resamples.
- **`src/run_llm.py`** and the paper's prompts are vendored from TurEngMix; I added `prompts/lid_few_shot_ours.txt` (same instructions, three examples from our data) for the third GPT-4o run.

---

## 5. Human Annotation Sample (`src/make_annotation.py`, `annotation/`)

- Designed the blind sample: 202 words from real TurEngMix text (train and validation only; the test split is never shown) and 200 words from our first 2,500 sentences, stratified by label so rare labels have enough items, shuffled together without telling annotators the source.
- Generated identical workbooks for the three annotators (drop-down labels) and the hidden answer key `annotation/key.csv`, which stayed private until all three sheets were back.

---

## 6. Integration, Verification and Presentation

- **Results:** re-checked every number in the final table against the prediction files (all 12 model rows, intervals, paired differences, scores without OTHER) and wrote the main `README.md` (question, labels, results, findings, caveats, reproduction steps).
- **Review of teammates' deliverables:** checked the figure script and licence/citation files before release; fixed the confusion-matrix figure to compare against the validation-selected run (gen5000), made figure labels round the same way as the table, and corrected the third-party code note in `LICENSE`.
- **Presentation:** the slides (`report/TR-EN-CodeSwitch-LID_presentation.pptx`), the talk, and the team's Q&A preparation.

---

## Deliverables & Artifacts

- **Data:** `data/generated/generated.jsonl`, `data/generated/raw/batch_01..70.txt`, `data/generated/README.md`
- **Code:** `src/build_generated.py`, `src/check_generated.py`, `src/lid_io.py`, `src/train_encoder.py`, `src/score_all.py`, `src/make_annotation.py`, `prompts/lid_few_shot_ours.txt`
- **Docs:** `README.md`, `TASKS.md`, `report/annotation_guide.md`, `report/TR-EN-CodeSwitch-LID_presentation.pptx`
- **Annotation:** `annotation/annotation_*.xlsx` (blank workbooks), `annotation/key.csv`

---

## Tools

I used an AI assistant (Claude, Anthropic) to write the generated sentences and to help with code and documents. I defined the research question, the experiment, the label rules and the quality checks, reviewed and corrected the output batch by batch, and verified every reported number against the result files.
