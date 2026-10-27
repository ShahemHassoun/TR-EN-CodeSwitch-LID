# Generated Turkish–English code-mixed LID dataset

**Status:** first 2,504 sentences finished and checked (target: 10,000). The numbers below are for
the current file and will be updated when it is complete.

## What it is

Sentences in the style of Ekşi Sözlük posts, mixing Turkish and English, with one language label
per word (`TR`, `EN`, `MIXED`, `NE`, `AMBIGUOUS`, `OTHER`). It follows the TurEngMix label scheme
and our `report/annotation_guide.md`.

**Purpose:** TurEngMix's training split has only **159 MIXED tokens**, and fine-tuned models score
0.00–0.22 F1 on that class in the paper. This dataset adds thousands of labelled MIXED examples.
The experiments test whether that helps on **real** TurEngMix text.

## How it was made

1. **Seed.** Only the TurEngMix **training split** was used as the reference for style and labelling:
   topics, tone, how words are mixed, and how borderline words are labelled. Its validation and
   test splits were never used for writing.
2. **Writing.** Sentences were written by an AI model (Claude, Anthropic) in batches of about 250
   on varied topics: work, tech, gaming, series, sport, food, travel, family, university, money,
   shopping and more. Labels were assigned while writing, in a compact inline format
   (`raw/batch_NN.txt`, e.g. `linki|M=link+i`, `deadline|E`, `netflix'te|N`). Every MIXED word
   records its English stem and Turkish ending.
3. **Building.** `src/build_generated.py` turns the raw batches into `generated.jsonl`.
   - Each batch is shuffled with a fixed seed and added in order, so the first 2,500 sentences
     never change when more are added (gen2500 ⊂ gen5000 ⊂ gen10000). **Batches 01–12 are frozen.**
   - **Typing without Turkish letters:** 25% of sentences are converted (ç→c, ğ→g, ı→i, ö→o, ş→s,
     ü→u), matching how TurEngMix users write. Result: 32% of our Turkish words contain Turkish
     letters (TurEngMix: about 30%).
4. **Checking.** `src/check_generated.py` rejects the whole file if any of these is found:
   - malformed lines, duplicate ids or sentences, capital letters, tokens that are only punctuation
   - a MIXED word whose recorded split doesn't spell the word, whose stem isn't an English word
     (wordfreq), or whose ending isn't a chain of real Turkish suffixes
   - an EN-labelled word that is clearly Turkish
   - copying: any run of 8 words shared with TurEngMix
   - **test leakage:** any word that TurEngMix labels MIXED / NE / OTHER / AMBIGUOUS in its
     validation or test splits but never in train
   Measured on TurEngMix's own training data, the MIXED check accepts 142 of 149 distinct real MIXED
   words, and its "possible missed MIXED" warning fires on 38 of 5,157 real Turkish words (0.7%),
   which are then reviewed by hand.
5. **Human check.** Three native speakers label a blind sample of 200 generated words mixed with
   202 real TurEngMix words (`src/make_annotation.py`). How often they agree with our labels
   measures how correct the generated labels are (results: `results/human_agreement.md`).

## Labelling decisions (applied the same way across the whole file)

- **Names beat MIXED:** `netflix'te`, `google'ladı`, `tiktok'çular` → NE. Every word of a multi-word
  name or product is NE (`amazon prime`, `xbox game pass'e`, `the office`), as in TurEngMix.
- **Established Turkish loanwords are TR**, even with English roots: `internet`, `site`, `film`,
  `video`, `telefon`, `bar`, `test`, `modem`, `market`, `park`, `form`, `tablet`, `logo`, `format`,
  `transfer`.
- **English-spelled words are EN** inside Turkish sentences (`trend`, `online`, `deadline`), as in
  TurEngMix.
- **A number plus a Turkish ending is TR** (`2026'da`, `%50'sini`). A number inside a name is NE
  (`windows 10`).
- **AMBIGUOUS** is used only for words both languages use (`platform`, `web`, `tv`, `model`, `anime`,
  `manga`, `protein`) and for interjections (`hey`, `ooh`, `pardon`).
- **OTHER** = a third language (`arigato`, `merci`, `schadenfreude`, `hola`).
- Keyboard mashes and cheat codes are left out, because TurEngMix labels them inconsistently.

## Statistics (current file)

| | Ours | TurEngMix train |
|---|--:|--:|
| Sentences | 2,504 | 254 |
| Tokens | 28,280 | 12,466 |
| TR | 79.6% | 79.9% |
| EN | 9.6% | 11.9% |
| MIXED | 4.3% (about 1,230 tokens) | 1.3% (159 tokens) |
| NE | 6.1% | 6.5% |
| AMBIGUOUS | 0.2% | 0.4% |
| OTHER | 0.2% | 0.02% |
| Sentences with at least one MIXED | 40.9% | 35% |

## Limitations (stated openly)

- **Written by an AI model, not by real users.** The sentences are cleaner and more uniform than
  real posts: shorter (11 words on average, against 49 in TurEngMix's "sentences", which are often
  whole posts), with fewer typos and less slang variety. That's why every result is measured on
  **real** TurEngMix test text.
- **MIXED is deliberately over-represented** (4.3% vs 1.3%), so models may predict MIXED more
  often than real text needs. The 0 / 2,500 / 5,000 / 10,000 size experiment measures this.
- **Labels follow our guide.** Where TurEngMix is itself inconsistent (e.g. `linki` is MIXED but
  `linklerden` is TR), we chose one rule and applied it everywhere, so our data and their test set
  can disagree on such words.
- The test-leakage filter blocks exact words only. It can't rule out similar topics.

## Licence

CC BY-NC-SA 4.0, because it was created using TurEngMix (CC BY-NC-SA 4.0) as seed data.
Cite TurEngMix (see `CITATION.md`).
