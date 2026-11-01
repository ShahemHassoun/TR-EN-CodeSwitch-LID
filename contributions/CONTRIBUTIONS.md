# Contributions

Team: The Avengers (repository TR-EN-CodeSwitch-LID), NLP course project 2, 2026.

## Shahem
- Project lead: research question, experiment design (0 / 2,500 / 5,000 / 10,000 generated
  sentences, size chosen on validation only) and task split.
- Generated dataset: 10,004 labelled sentences written with an AI model, the label guide
  (`report/annotation_guide.md`), and the quality checks (`src/check_generated.py`: format,
  test-word leakage, copying and near-duplicates), with nested, frozen subsets.
- Shared code (`src/lid_io.py`, `src/train_encoder.py`, `src/score_all.py`,
  `src/build_generated.py`, `src/make_annotation.py`), final results, README and presentation.

I built the generated dataset and the checks that keep it clean and separate from the test set,
and wrote the shared code that every model uses for loading, training and scoring. I also put
the results together, wrote the README and gave the presentation.

## Ibrahim
- Fine-tuned BERTurk with 0 / 2,500 / 5,000 / 10,000 generated sentences and XLM-R with 0 and
  5,000, 4 seeds each (`results/runs/`, `results/predictions/`).
- Chose the data size on validation (`results/ibrahim_size_choice.md`).
- Native-speaker annotation of 402 words for the agreement study.

I ran all 24 fine-tuning runs with the paper's training settings and picked the data size for
XLM-R from the validation scores only. I also labelled the 402 words of the agreement study.

## Mahmut
- Rule baselines with wordfreq and Zemberek (`src/baseline_rules.py`,
  `results/zemberek_coverage.md`).
- CRF baseline tuned on validation, run with all four data sizes (`src/baseline_crf.py`,
  `results/crf_settings.md`).
- Error analysis of BERTurk with 0 and 5,000 generated sentences (`results/error_analysis.md`,
  `notebooks/error_analysis.ipynb`); native-speaker annotation.

I built the rule and CRF baselines and tuned the CRF on the validation split only. I also
analysed which `MIXED` words the best model still gets wrong, and labelled the 402 words of the
agreement study.

## Tuğra
- GPT-4o baselines: zero-shot, three examples from the paper, and three examples from our data
  (`results/predictions/gpt4o_*`).
- Human agreement study (`src/agreement.py`, `results/human_agreement.md`): pairwise Cohen's
  kappa and comparison of the majority vote with the reference labels.
- Final scoring of all runs (`results/results_table.csv`); native-speaker annotation.

I ran the three GPT-4o baselines and wrote the agreement study that measures how well the three
annotators agree and how close our generated labels are to their majority vote. I also labelled
the 402 words and produced the final results table.

## Seleni
- Figures (`src/plot_figures.py`, `results/figures/`).
- Licence and citation files (`LICENSE`, `data/generated/LICENSE.md`, `CITATION.md`) and this
  contributions page.

I wrote the script that makes the three result figures, prepared the licence and citation files
for our code, our dataset and the work we build on, and collected this contributions page.
