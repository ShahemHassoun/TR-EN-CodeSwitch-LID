# Generated Turkish–English code-mixed LID dataset

**Status:** complete. 10,004 sentences, all passing the checker. The size experiment uses the
first 2,500 / 5,000 / 10,000 lines (each set is contained in the next).

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
2. **Writing.** Sentences were written by an AI model (Claude, Anthropic) in 70 batches of 100–260 sentences
   on varied topics: work, tech, gaming, series, sport, food, travel, family, university, money,
   shopping and more. Labels were assigned while writing, in a compact inline format
   (`raw/batch_NN.txt`, e.g. `linki|M=link+i`, `deadline|E`, `netflix'te|N`). Every MIXED word
   records its English stem and Turkish ending.
3. **Building.** `src/build_generated.py` turns the raw batches into `generated.jsonl`.
   - Each batch is shuffled with a fixed seed and added in order, so the first 2,500 sentences
     never change when more are added (gen2500 ⊂ gen5000 ⊂ gen10000). Batches 01–12 were frozen once
     the first 2,500 were shared, and batches 01–26 once the first 5,000 were shared.
   - **Typing without Turkish letters:** 25% of sentences are converted (ç→c, ğ→g, ı→i, ö→o, ş→s,
     ü→u), matching how TurEngMix users write. Result: 32% of our Turkish words contain Turkish
     letters (TurEngMix: about 30%).
   - **Dropped apostrophes (batch 13 onward).** In TurEngMix's training data 63% of MIXED words have
     no apostrophe (`linki`, `influencerlar`, `editleyip`); our first batches almost always had one
     (`link'i`). From batch 13 on, the builder writes 70% of MIXED words without it (stems of 4+
     letters, only when the result still passes the MIXED check, e.g. `deadline'dan` →
     `deadlinedan`), and 30% of names with an ending (`netflix'te` → `netflixte`). Batches 01–12
     are left as they were.
4. **Checking.** `src/check_generated.py` rejects the whole file if any of these is found:
   - malformed lines, duplicate ids or sentences, capital letters, tokens that are only punctuation
   - a MIXED word whose recorded split doesn't spell the word, whose stem isn't an English word
     (wordfreq), or whose ending isn't a chain of real Turkish suffixes
   - an EN-labelled word that is clearly Turkish
   - copying: any run of 8 words shared with TurEngMix
   - **test leakage:** any word that TurEngMix labels MIXED / NE / OTHER / AMBIGUOUS in its
     validation or test splits and that never appears in train (attached punctuation and ’ vs '
     ignored). One exception: `startup'larının` (sentence 2,011, in the frozen first 2,500) matches
     the test word `startup’larının` once the apostrophe style is ignored. It was found after the
     first 2,500 had been used in training runs, so it was kept.
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
- More established loanwords that are TR: `final`, `pilot`, `sponsor`, `poster`, `optimize`, `avm`.
- **Nationality words are NE** (`türk`, `japon`, `alman`, `fransız`), like language names, which is
  TurEngMix's majority choice.
- **AMBIGUOUS** is used only for words both languages use (`platform`, `web`, `tv`, `model`, `anime`,
  `manga`, `protein`) and for interjections (`hey`, `ooh`, `pardon`). With a Turkish ending they are
  TR (`platformu`, `tv'de`, `mangası`), as in TurEngMix.
- English phrases keep loanwords EN: `open bar`, `open plan`, `black friday` (TurEngMix:
  `black` EN, `friday'de` MIXED).
- **OTHER** = a third language (`arigato`, `merci`, `schadenfreude`, `hola`).
- Keyboard mashes and cheat codes are left out, because TurEngMix labels them inconsistently.

## Statistics

| | Ours | TurEngMix train |
|---|--:|--:|
| | gen2500 | gen5000 | gen10000 | TurEngMix train |
|---|--:|--:|--:|--:|
| Sentences | 2,500 | 5,000 | 10,000 | 254 |
| Tokens | 28,235 | 51,875 | 98,875 | 12,466 |
| TR | 79.7% | 79.1% | 79.7% | 79.9% |
| EN | 9.5% | 9.9% | 9.8% | 11.9% |
| MIXED | 4.3% (1,211) | 4.6% (2,411) | 4.5% (4,409) | 1.3% (159) |
| NE | 6.2% | 6.0% | 5.7% | 6.5% |
| AMBIGUOUS | 0.2% | 0.2% | 0.2% | 0.4% |
| OTHER | 0.2% | 0.1% | 0.1% | 0.02% |
| Sentences with at least one MIXED | 40.4% | 41.3% | 38.5% | 35% |
| MIXED words without an apostrophe | 1% | 28% | 56% | 63% |
| Turkish words with ç ğ ı ö ş ü | 32% | 32% | 32% | about 30% |

## Limitations (stated openly)

- **Written by an AI model, not by real users.** The sentences are cleaner and more uniform than
  real posts: shorter (about 10 words on average, against 49 in TurEngMix's "sentences", which are often
  whole posts), with fewer typos and less slang variety. That's why every result is measured on
  **real** TurEngMix test text.
- **MIXED is deliberately over-represented** (4.3% vs 1.3%), so models may predict MIXED more
  often than real text needs. The 0 / 2,500 / 5,000 / 10,000 size experiment measures this.
- **Labels follow our guide.** Where TurEngMix is itself inconsistent (e.g. `linki` is MIXED but
  `linklerden` is TR), we chose one rule and applied it everywhere, so our data and their test set
  can disagree on such words.
- The test-leakage filter blocks exact words only. It can't rule out similar topics.
- **Style changes across the sizes.** Batches 13 onward drop more apostrophes (see above), so the
  share of apostrophe-free MIXED words grows from 1% (gen2500) to 56% (gen10000). Part of any
  difference between sizes may come from this style change, not only from the amount of data.
- **Near-duplicates.** Every sentence after the first 2,500 was compared with all others; pairs
  sharing at least half of their words were rewritten or removed. Similar sentence frames
  ("annem yeni bir … aldı") still repeat more often than in real posts.

## Changes after the first push (first 2,500 sentences)

After the first 2,500 sentences were pushed, 34 labels in them were corrected (rules above:
`final`/`pilot`/`sponsor`/`poster`/`optimize` → TR, nationality words → NE, AMBIGUOUS stem + ending
→ TR, `stack overflow` / `black mirror` / `internet explorer` → NE), and 3 words were re-spelled
(`noter'de` → `noterde`, `plan'ları` → `planları`, `market'ten` → `marketten`). That is about 0.1% of
its tokens. Runs on gen2500 should use the final file so all sizes share the same labels.

## Licence

CC BY-NC-SA 4.0, because it was created using TurEngMix (CC BY-NC-SA 4.0) as seed data.
Cite TurEngMix (see `CITATION.md`).
