# Contribution: Mahmut Örnek

- **Student ID:** 220901545
- **GitHub:** [Mahmut-Ornek](https://github.com/Mahmut-Ornek)
- **Role:** Rule-based & Zemberek baselines, CRF sequence tagger & hyperparameter tuning, linguistic error analysis, blind manual annotation

---

## Summary

I designed and implemented the classical rule-based baselines (`wordfreq` and `Zemberek` morphological analysis), quantified Zemberek's coverage drop on diacritic-free Turkish text, developed and tuned the feature-based Linear-Chain CRF sequence tagger across four synthetic data scales (0 to 10,000 sentences), conducted the token-level linguistic error analysis on the `MIXED` class comparing BERTurk before and after synthetic data augmentation, and completed the blind human annotation study (402 instances) to establish the human performance ceiling and verify synthetic label quality.

---

## 1. Rule-Based Baselines & Zemberek Coverage (`src/baseline_rules.py`, `results/zemberek_coverage.md`)

- **Fixed Rule Architecture (Zero Test Leakage):** Built two non-parametric baselines without tuning thresholds on validation or test splits. Extracted the named-entity (`NE`) vocabulary strictly from the `train` split using Unicode punctuation stripping and case-folding.
- **`rules_wordfreq` Baseline:** Applied a deterministic decision hierarchy:
  1. Match against the train-derived `NE` vocabulary $\rightarrow$ `NE`.
  2. Check for an apostrophe (`'`) where the stem prefix has positive English frequency in `wordfreq` (`word_frequency(prefix, 'en') > 0`) $\rightarrow$ `MIXED`.
  3. Compare English vs. Turkish word frequencies (`word_frequency(token, 'en') > word_frequency(token, 'tr')`) $\rightarrow$ `EN`, otherwise `TR`.
- **`rules_zemberek` Baseline:** Integrated `zemberek-python` (`TurkishMorphology.create_with_defaults()`). Preserved `NE` and `MIXED` decisions, and labeled tokens as `TR` if Zemberek returned at least one valid, non-unknown, non-runtime morphological analysis whose secondary POS was not `ProperNoun` (unless English word frequency was at least $10\times$ the Turkish frequency).
- **Diacritic Sensitivity & Coverage Analysis (`results/zemberek_coverage.md`):** Evaluated Zemberek recognition across all gold `TR` token occurrences in the `train` split:
  - **With Turkish characters (`ç, ğ, ı, ö, ş, ü`):** **97.4%** recognition ($2,895 / 2,972$).
  - **Without Turkish characters (de-ASCIIfied spelling, e.g., `henuz`):** **83.1%** recognition ($5,806 / 6,990$).
  - **Key Finding:** Demonstrated a **14.3 percentage-point drop** on diacritic-free social media text, explaining why pure morphological analyzers struggle on informal Ekşi Sözlük posts (scoring ~0.37 test Macro-F1).

---

## 2. Linear-Chain CRF Baseline (`src/baseline_crf.py`, `results/crf_settings.md`)

- **Contextual Feature Engineering:** Extracted token-level and $\pm 1$ neighbor context features (`BOS`/`EOS` flags and adjacent token dictionaries) for `sklearn-crfsuite`:
  - **Orthographic & Shape Features:** Raw token, lowercased form, character length, `isupper()`, `istitle()`, `isdigit()`, and presence of Turkish-specific letters (`ç, ğ, ı, ö, ş, ü`).
  - **Subword Affixes:** Character prefixes and suffixes of lengths $1, 2, 3$ to capture Turkish agglutinative endings.
  - **Code-Switching Boundary Features:** Explicit boolean flag for apostrophe presence, stem prefix before the apostrophe, and Turkish case/tense suffix after the apostrophe (targeting tokens like `meeting'e`).
  - **Lexical Frequency Signals:** Cached English and Turkish `wordfreq` unigram frequencies.
- **Hyperparameter Tuning (Validation Split Only):** Conducted a $5 \times 5$ grid search over L1 and L2 regularization ($c_1, c_2 \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$) using L-BFGS (`max_iterations=100`, `all_possible_transitions=True`). Selected **$c_1 = 0.5, c_2 = 0.01$**, which achieved the highest validation six-label Macro-F1 of **0.4870**.
- **Scaling Across Synthetic Data Sizes:** Trained four models with fixed hyperparameters on the TurEngMix train split plus $0$, $2,500$, $5,000$, and $10,000$ generated sentences:
  - **Validation Macro-F1 Progression:** `crf_gen0`: **0.4870** $\rightarrow$ `crf_gen2500`: **0.4382** $\rightarrow$ `crf_gen5000`: **0.4704** $\rightarrow$ `crf_gen10000`: **0.5149** (+2.79 pp over `gen0`).
  - **Test Set Generalization:** On the real TurEngMix test set, adding our generated data boosted CRF Macro-F1 from **0.475** (matching real-only BERTurk at 0.471) to **0.525** (+5.0 pp).

---

## 3. Linguistic Error Analysis (`notebooks/error_analysis.ipynb`, `results/error_analysis.md`)

- **Target Comparison:** Aligned gold test split tokens ($1,373$ tokens across $41$ sentences) with `berturk_gen0_seed1` (real data only) and `berturk_gen5000_seed1` (validation-selected optimal synthetic size) by `(doc_id, sent_id, tok_id)`.
- **MIXED Performance Gains (Seed 1):**
  - **`berturk_gen0_seed1`:** Precision = **0.333**, Recall = **0.273**, F1 = **0.300** ($\text{TP}=3, \text{FP}=6, \text{FN}=8$).
  - **`berturk_gen5000_seed1`:** Precision = **0.667**, Recall = **0.364**, F1 = **0.471** ($\text{TP}=4, \text{FP}=2, \text{FN}=7$).
- **Taxonomy & Transition Breakdown:** Inspected all $18$ `MIXED`-involved token instances ($16$ error rows) across six canonical linguistic categories (`apostrophe`, `no apostrophe`, `name vs MIXED`, `no Turkish letters`, `gold label looks wrong`, `other`):
  - **Transitions:** **6 fixed**, **1 degraded** (`views'teki`), **9 persistently wrong**, and **2 correct in both** (`subredditlerde`, `database'i`).
  - **What Synthetic Data Fixed:** Eliminated false-positive `MIXED` predictions on uninflected English nouns (`demo` $\times 2 \rightarrow \text{EN}$), platform names (`telegram` $\rightarrow \text{NE}$), and suffixed product acronyms (`jslerle` $\rightarrow \text{NE}$), while beginning to resolve un-apostrophed Turkish suffixes on English stems (`downloadla`, `mottom` $\rightarrow \text{MIXED}$).
  - **Persistent Bottlenecks:** Direct suffixation on bare English stems without an apostrophe remains the primary failure mode ($5$ of $7$ `no apostrophe` errors persist: `linki`, `appletti`, `webin`, `folku`, `postların`), alongside rare `NE` vs. `MIXED` apostrophe boundary cases (`pseudocide'ı`, `culotte'lar`) and one gold-label inconsistency (`textin` annotated as `TR`).

---

## 4. Blind Manual Annotation Study (`annotation/annotation_mahmut.xlsx`)

- **Independent Labelling:** Annotated all $402$ blind items ($202$ real TurEngMix tokens and $200$ generated tokens) without AI assistance, strictly following `report/annotation_guide.md`.
- **Agreement & Ceiling Results:** Achieved **75.3% accuracy** and **$\kappa = 0.6727$** (0.6414 Macro-F1) against TurEngMix gold labels on real items, contributing to our three-annotator study ($\text{mean } \kappa = 0.7091$ overall; **0.9670 `MIXED` F1** on generated items).

---

## Deliverables & Artifacts

- **Code & Notebooks:** `src/baseline_rules.py`, `src/baseline_crf.py`, `notebooks/error_analysis.ipynb`
- **Reports:** `results/zemberek_coverage.md`, `results/crf_settings.md`, `results/error_analysis.md`
- **Predictions (`results/predictions/`):** `rules_wordfreq_{validation,test}.csv`, `rules_zemberek_{validation,test}.csv`, `crf_gen{0,2500,5000,10000}_{validation,test}.csv`
- **Annotation:** `annotation/annotation_mahmut.xlsx`