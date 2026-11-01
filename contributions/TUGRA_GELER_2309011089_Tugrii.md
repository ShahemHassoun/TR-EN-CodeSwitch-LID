# Individual Technical Contribution Report
## Token-Level Language Identification on Turkish–English Code-Switched Text (`TR-EN-CodeSwitch-LID`)

- **Student Name:** Tuğra Geler
- **Student ID:** 2309011089
- **GitHub Handle:** `@Tugrii`
- **Course:** Natural Language Processing — İstinye University, Computer Engineering
- **Assigned Modules:** 
  1. **Task 3:** Inter-Annotator Agreement Engine, Human Ceiling Estimation & Synthetic Data Quality Assurance (`src/agreement.py`, `results/human_agreement.md`, `results/human_agreement.json`)
  2. **Task 2:** Autoregressive LLM Baselines (GPT-4o In-Context Learning) & Final Unified Benchmark Evaluation (`src/run_llm.py`, `src/score_all.py`, `results/predictions/gpt4o_*`, `results/results_table.md`)

---

## 1. Executive Summary of Project & Role

The **`TR-EN-CodeSwitch-LID`** project addresses **Token-Level Language Identification (LID)** in informal Turkish–English code-switched social media discourse. Unlike sentence-level classification, token-level sequence tagging requires resolving word-internal morphological code-switching (`MIXED`, where an English stem takes agglutinative Turkish suffixes, e.g., `counselor'ı`, `standup'a`, `linke`), distinguishing cross-lingual homographs and loanwords (`AMBIGUOUS`, e.g., `model`, `protein`, `anime`), and isolating named entities (`NE`) from standard monolingual tokens (`TR`, `EN`, `OTHER`).

Because `MIXED` tokens represent a severe minority class in natural corpora, standard supervised models suffer from gradient starvation on code-switched boundaries. Our project tests whether targeted synthetic data augmentation (2.5k, 5k, and 10k generated sentences) bridges this gap across four modeling paradigms:
1. **Heuristic & Morphological Rules** (`rules_wordfreq`, `rules_zemberek`)
2. **Linear-Chain Conditional Random Fields** (`crf_gen0`, `crf_gen2500`, `crf_gen5000`)
3. **Fine-Tuned Pre-trained Transformer Encoders** (`BERTurk` and `XLM-RoBERTa`)
4. **Prompted Autoregressive LLM Baselines** (`GPT-4o` under `0-shot`, `3-shot`, and `3-shot-ours` regimes)

Within this architecture, my work focused on establishing the **empirical evaluation bounds** of the study:
- **Establishing the Upper Bound (Human Ceiling) & Synthetic Data Verification:** Designing and implementing `src/agreement.py` to process blind triple-annotated data ($n = 402$), computing pairwise and per-class Cohen's Kappa ($\kappa$), proving the quality of our synthetic data pipeline, and establishing the human performance ceiling against ground-truth labels.
- **Establishing the Generative LLM Baseline & Final Benchmark Integration:** Executing zero-shot and few-shot `GPT-4o` evaluations across the test split, diagnosing autoregressive sequence alignment failures and distribution shift under synthetic in-context prompts, and compiling the master comparison tables (`results/results_table.md` and `.csv`).

---

## 2. Task 3: Human Agreement Study, Human Ceiling & Synthetic QA (`src/agreement.py`)

### 2.1 Methodology & Implementation Architecture
To evaluate how consistently humans apply the 6-class taxonomy (`TR`, `EN`, `MIXED`, `NE`, `AMBIGUOUS`, `OTHER`) and to validate whether our generated data matches natural code-switching distributions, three native Turkish speakers with bilingual English proficiency (**Tuğra**, **İbrahim**, and **Mahmut**) independently annotated 402 target tokens in context (`[[target]]`) under blind conditions.

I engineered `src/agreement.py` as an end-to-end evaluation pipeline that:
1. Dynamically ingests multi-format annotator spreadsheets (`.xlsx` / `.csv`), normalizes label schemas, and extracts target token windows.
2. Computes pairwise and mean **Cohen's Kappa ($\kappa$)** both globally and via one-vs-rest binary reductions for every individual label class.
3. Resolves majority-vote consensus (`3/3 Unanimous`, `2/3 Majority`, and `1/1/1 Total Disagreement`).
4. Cross-references annotations against the master answer key (`annotation/key.csv`), partitioning the dataset into **Real TurEngMix items ($n = 202$)** and **Generated Synthetic items ($n = 200$)** to separately measure the **Human Performance Ceiling** and **Synthetic Data Quality Assurance (QA)**.

### 2.2 Inter-Annotator Agreement Results (All, Real, and Generated Splits)

| Dataset Split | Sample Size ($n$) | Tuğra–İbrahim $\kappa$ | Tuğra–Mahmut $\kappa$ | İbrahim–Mahmut $\kappa$ | **Mean Pairwise $\kappa$** | 3/3 Unanimous | $\ge$ 2/3 Consensus | 1/1/1 Disagreement |
|---|---|---|---|---|---|---|---|---|
| **All Items** | 402 | `0.7052` | `0.7740` | `0.6481` | **`0.7091`** | 285 (`70.9%`) | 377 (`93.8%`) | 25 (`6.2%`) |
| **Real TurEngMix Only** | 202 | `0.6909` | `0.7263` | `0.5770` | **`0.6647`** | 135 (`66.8%`) | 188 (`93.1%`) | 14 (`6.9%`) |
| **Generated (Synthetic) Only** | 200 | `0.7160` | `0.8186` | `0.7168` | **`0.7505`** | 150 (`75.0%`) | 189 (`94.5%`) | 11 (`5.5%`) |

#### Per-Label Agreement Breakdown (All Items, $n = 402$)

| Label | Tuğra–İbrahim $\kappa$ | Tuğra–Mahmut $\kappa$ | İbrahim–Mahmut $\kappa$ | **Mean $\kappa$** | Label Counts (Tuğra / İbrahim / Mahmut / Majority / Gold Ref) |
|---|---|---|---|---|---|
| **`TR`** | `0.8097` | `0.8131` | `0.7553` | **`0.7927`** | 162 / 140 / 158 / **160** / `121` |
| **`EN`** | `0.6829` | `0.8656` | `0.6631` | **`0.7372`** | 93 / 138 / 90 / **99** / `89` |
| **`MIXED`** | `0.7973` | `0.8816` | `0.7896` | **`0.8228`** | 76 / 84 / 68 / **75** / `75` |
| **`NE`** | `0.4865` | `0.6481` | `0.3142` | **`0.4829`** | 41 / 17 / 77 / **43** / `80` |
| **`AMBIGUOUS`** | `0.2100` | `-0.0128` | `-0.0127` | **`0.0615`** | 17 / 16 / 3 / **13** / `30` |
| **`OTHER`** | `0.6931` | `0.5164` | `0.7655` | **`0.6583`** | 13 / 7 / 6 / **12** / `7` |

**Key Engineering & Linguistic Findings:**
- **High Stability on Core Project Target (`MIXED`):** Across all 402 tokens, `MIXED` achieved the highest inter-annotator agreement of any class (**Mean $\kappa = 0.8228$**, reaching **`0.8631`** on synthetic items). Furthermore, the majority-voted human count for `MIXED` (**75**) matched the ground-truth reference count (**75**) exactly.
- **Boundary Friction in `NE` and `AMBIGUOUS`:** While core language tags (`TR`, `EN`, `MIXED`) exhibit strong agreement ($\kappa \in [0.73, 0.82]$), overall kappa is pulled down by subjective boundaries between established loanwords (`AMBIGUOUS`, $\kappa = 0.0615$) and lowercase/inflected named entities (`NE`, $\kappa = 0.4829$).

### 2.3 Measured Human Ceiling (Annotators vs. TurEngMix Gold on Real Items)
Evaluating human annotators directly against the 202 real TurEngMix gold labels establishes the theoretical upper bound for any machine learning model trained on this taxonomy:

| Annotator | Cohen's Kappa ($\kappa$) | **Macro-F1 (Human Ceiling)** | Accuracy | `MIXED` F1 | `NE` F1 |
|---|---|---|---|---|---|
| **Tuğra** | `0.6629` | **`0.6558`** | `0.7426` | `0.7931` | `0.5172` |
| **Majority Vote** | `0.6630` | **`0.6497`** | `0.7426` | `0.8136` | `0.5085` |
| **Mahmut** | `0.6727` | **`0.6414`** | `0.7525` | `0.8519` | `0.6988` |
| **İbrahim** | `0.5672` | **`0.6136`** | `0.6683` | `0.6667` | `0.1702` |

**Why This Result Is Critical for Our Project:**
Without a human ceiling study, a test Macro-F1 of **`0.5638`** (achieved by our best fine-tuned model, `xlmr_gen5000`) might appear modest. By proving that native human speakers plateau at **`0.6497` (Majority Vote) – `0.6558` (Tuğra)** Macro-F1 due to inherent annotation noise in `AMBIGUOUS` and `NE` conventions, we demonstrate that our fine-tuned XLM-R model actually captures **~86.8% of the human ceiling**.

### 2.4 Synthetic Data Quality Assurance (Annotators vs. Generated Labels)
Comparing human annotations against the programmatic labels on the 200 generated items validates our synthetic data pipeline (`src/build_generated.py` & `src/check_generated.py`):

| Annotator | Cohen's Kappa ($\kappa$) | **Macro-F1** | Accuracy | **`MIXED` F1** | `NE` F1 |
|---|---|---|---|---|---|
| **Tuğra** | `0.7389` | `0.7038` | `0.8000` | `0.9247` | `0.6349` |
| **İbrahim** | `0.6990` | `0.6859` | `0.7700` | `0.8958` | `0.4000` |
| **Mahmut** | `0.8163` | `0.7743` | `0.8600` | `0.9663` | `0.7297` |
| **Majority Vote** | **`0.7840`** | **`0.7474`** | **`0.8350`** | **`0.9670`** | **`0.6562`** |

The **`0.9670` F1 score** between human majority vote and our synthetic `MIXED` labels empirically confirms that our generated code-switched tokens are virtually indistinguishable from gold-standard human judgments.

### 2.5 Qualitative Analysis of 1/1/1 Disagreements (Edge Cases)
All 25 three-way ties (`1/1/1`) were programmatically extracted and inspected. They cluster into three linguistic edge cases:
1. **Unapostrophized Orthographic Code-Switching:** Words such as `[[turntablea]]` (Row 288, Gold: `MIXED`) and `[[linke]]` (Row 371, Gold: `MIXED`), where English stems receive Turkish dative suffixes (`-a`, `-e`) without an apostrophe. Annotators split between `MIXED`, `EN`/`TR`, and `AMBIGUOUS`.
2. **Integrated Loanwords vs. Synchronically Ambiguous Forms:** Tokens like `[[model]]` (Rows 176, 258, 266), `[[protein]]` (Row 108), `[[anime]]` (Rows 83, 239), and `[[pardon]]` (Row 7), where dictionary integration conflicts with etymological origin.
3. **Inflected Proper Nouns:** Tokens such as `[[meksika'dan]]` (Row 126) and `[[ibrahimoviç'le]]` (Row 344), where proper nouns (`NE`) carry Turkish case suffixes, creating tension between entity tagging (`NE`), monolingual tagging (`TR`), and morphological mixing (`MIXED`).

---

## 3. Task 2: Autoregressive LLM Baselines (`GPT-4o`) & Benchmark Evaluation

### 3.1 Experimental Setup (`src/run_llm.py`)
While encoder models (`BERTurk`, `XLM-R`) project subword hidden states through a fixed-dimension linear classification head ($W \in \mathbb{R}^{768 \times 6}$), autoregressive LLMs frame sequence tagging as a text-to-text generation task via **In-Context Learning (ICL)** without gradient updates.

I configured and executed three evaluation regimes on the held-out test split (41 social media posts) using `gpt-4o`:
1. **`gpt4o_0shot_test`:** Zero-shot execution using only system instructions and taxonomy definitions.
2. **`gpt4o_3shot_test`:** Few-shot execution using 3 standard in-domain demonstration examples.
3. **`gpt4o_3shot-ours_test`:** Few-shot execution using 3 synthetic demonstration examples drawn from our generated corpus (`prompts/lid_few_shot_ours.txt`).

### 3.2 Sequence Alignment & Performance Analysis

| LLM Run | Prompt Regime | Sentence Alignment | Unparseable Tokens | Test Macro-F1 | 95% Bootstrap CI |
|---|---|---|---|---|---|
| **`gpt4o_0shot`** | Zero-shot instructions only | **41 / 41** (`100.0%`) | `0` (`0.00%`) | **`0.4799`** | `[0.4033, 0.5693]` |
| **`gpt4o_3shot`** | 3-shot standard demonstrations | **40 / 41** (`97.6%`) | `0` (`0.00%`) | **`0.6207`** | `[0.4912, 0.6697]` |
| **`gpt4o_3shot-ours`** | 3-shot synthetic demonstrations | **39 / 41** (`95.1%`) | `0` (`0.00%`) | **`0.4959`** | `[0.4368, 0.5326]` |

#### Key Technical Insights from LLM Runs:
1. **The Structural Alignment Bottleneck (`41/41` $\to$ `40/41` $\to$ `39/41`):**
   Unlike encoder architectures where output tensor length is strictly bound to input token length ($N_{\text{out}} = N_{\text{in}}$), an autoregressive decoder generates token-label pairs sequentially. As prompt complexity increased with few-shot examples, `GPT-4o` occasionally merged punctuation or dropped a token on long social media posts, causing 1 misaligned sentence in `3-shot` and 2 misaligned sentences in `3-shot-ours`.
2. **In-Context Learning Power (`0.4799` $\to$ `0.6207`):**
   Providing just 3 real in-domain examples boosted `GPT-4o` by **+14.08 Macro-F1 points** (`0.6207`), allowing it to outperform all locally trained encoder models and reach **95.5% of the human majority-vote ceiling (`0.6497`)**.
3. **Distribution Shift in `gpt4o_3shot-ours` (`0.6207` vs. `0.4959`):**
   When prompted with 3 synthetic examples (`3-shot-ours`), performance dropped to `0.4959`. While synthetic sentences are highly effective for *gradient-based fine-tuning* (where thousands of examples regularize subword representations), using just 3 synthetic templates in an ICL prompt induces a narrow template bias that mismatches the noisy punctuation and orthographic variance of real test posts.

---

## 4. Unified Project Benchmark (`src/score_all.py`)

By integrating all team predictions with my `GPT-4o` runs and human ceiling metrics via `src/score_all.py`, we obtained the complete performance hierarchy across all modeling paradigms (`results/results_table.md`):

| Rank | System / Model | Paradigm | Synthetic Data Added | Test Macro-F1 | 95% Confidence Interval | Seeds |
|---|---|---|---|---|---|---|
| — | **Human Ceiling (Tuğra)** | Human Annotator vs. Gold | — | **`0.6558`** | — | — |
| — | **Human Consensus (Majority)** | 3-Annotator Majority Vote | — | **`0.6497`** | — | — |
| **1** | **`gpt4o_3shot`** | Autoregressive LLM (ICL) | 0 | **`0.6207`** | `[0.4912, 0.6697]` | 1 |
| **2** | **`xlmr_gen5000`** | Multilingual Transformer Encoder | +5,000 | **`0.5638`** | `[0.4953, 0.6240]` | 4 |
| **3** | **`berturk_gen10000`** | Monolingual Transformer Encoder | +10,000 | **`0.5561`** | `[0.4874, 0.6240]` | 4 |
| **4** | **`berturk_gen5000`** | Monolingual Transformer Encoder | +5,000 | **`0.5501`** | `[0.4880, 0.6018]` | 4 |
| **5** | **`berturk_gen2500`** | Monolingual Transformer Encoder | +2,500 | **`0.5214`** | `[0.4491, 0.5824]` | 4 |
| **6** | **`crf_gen5000`** | Linear-Chain CRF | +5,000 | **`0.5122`** | `[0.4226, 0.6048]` | 1 |
| **7** | **`crf_gen2500`** | Linear-Chain CRF | +2,500 | **`0.5041`** | `[0.4140, 0.5930]` | 1 |
| **8** | **`gpt4o_3shot-ours`** | Autoregressive LLM (Synthetic ICL) | 0 | **`0.4959`** | `[0.4368, 0.5326]` | 1 |
| **9** | **`gpt4o_0shot`** | Autoregressive LLM (Zero-Shot) | 0 | **`0.4799`** | `[0.4033, 0.5693]` | 1 |
| **10** | **`crf_gen0`** | Linear-Chain CRF | 0 | **`0.4749`** | `[0.3618, 0.5836]` | 1 |
| **11** | **`berturk_gen0`** | Monolingual Transformer Encoder | 0 | **`0.4710`** | `[0.4200, 0.5133]` | 4 |
| **12** | **`xlmr_gen0`** | Multilingual Transformer Encoder | 0 | **`0.4295`** | `[0.3920, 0.4474]` | 4 |
| **13** | **`rules_wordfreq`** | Lexical Frequency Heuristic | 0 | **`0.3695`** | `[0.3204, 0.4165]` | 1 |
| **14** | **`rules_zemberek`** | Morphological Analyzer Heuristic | 0 | **`0.3678`** | `[0.3164, 0.4164]` | 1 |
| **15** | **`majority`** | Dummy Majority Baseline | 0 | **`0.1414`** | `[0.1291, 0.1498]` | 1 |

---

## 5. Summary of Delivered Artifacts & Reproducibility

All code, raw API logs, prediction matrices, and statistical reports produced under my responsibility are committed to the repository and fully reproducible via the following commands:

```bash
# 1. Reproduce Inter-Annotator Agreement, Human Ceiling, and Synthetic QA Report
python3 src/agreement.py
# -> Outputs: results/human_agreement.md, results/human_agreement.json

# 2. Reproduce GPT-4o LLM Baseline Predictions (0-shot, 3-shot, 3-shot-ours)
python3 src/run_llm.py --task lid --model gpt-4o --shots 0 --split test --run-name gpt4o_0shot_test
python3 src/run_llm.py --task lid --model gpt-4o --shots 3 --split test --run-name gpt4o_3shot_test
python3 src/run_llm.py --task lid --model gpt-4o --shots 3 --split test --prompt prompts/lid_few_shot_ours.txt --run-name gpt4o_3shot-ours_test
# -> Outputs: results/predictions/gpt4o_*_test.{csv,meta.json,calls.jsonl}

# 3. Reproduce Master Benchmark Evaluation Table
python3 src/score_all.py
# -> Outputs: results/results_table.md, results/results_table.csv