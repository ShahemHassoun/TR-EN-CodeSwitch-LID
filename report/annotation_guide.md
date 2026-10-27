# Annotation Guide — Turkish–English Code-Switching LID

Token-level language identification for Turkish–English code-mixed text.

This guide follows the conventions of the **TurEngMix** benchmark (Ekşi Sözlük posts,
15,012 tokens, CC BY-NC-SA 4.0). Every rule below was checked against how TurEngMix
actually labels its data, so that models trained on our data are scored fairly on
theirs. **Do not add, rename or merge labels.**

Every example in this guide is either taken from the TurEngMix **training** portion or
made up. None comes from their validation or test portions, and we never look at those
when writing data, so the final test stays fair.

---

## 1. The six labels

| Label | Meaning | Share of TurEngMix |
|---|---|--:|
| `TR` | Turkish word | 78.1% |
| `EN` | English word with no Turkish suffix | 13.3% |
| `NE` | Any token that is part of a name (person, brand, place, title, organisation) | 6.9% |
| `MIXED` | One token whose stem and suffix come from different languages | 1.3% |
| `AMBIGUOUS` | Could belong to either language, and the sentence does not decide | 0.4% |
| `OTHER` | A third language (e.g. Japanese `seinen`) | <0.1% |

`MIXED` is the rare, hard class and the reason this project exists. In the published
results, fine-tuned BERTurk reaches only **0.13 F1** on it.

---

## 2. Text conventions

Our data must look like TurEngMix text, or models learn signals that do not exist in
the test set.

- **Everything lowercase:** `netflix'te`, not `Netflix'te`. Capital letters are **not** a
  clue for names here.
- **Punctuation is never its own token.** Sometimes a comma, bracket or quote is still
  stuck to a word (`refund,` `bilmiyorum)` — about 4% of tokens). Label the word as if the
  punctuation were not there.
- **Turkish letters are often missing:** `henuz`, `mantigi`, `icinde`. Only about 30% of
  Turkish words in TurEngMix contain ç ğ ı ö ş ü. These are still `TR`.
- **Apostrophes stay:** `download'ladım` keeps its apostrophe.

---

## 3. Rules, in order

Apply these **in order**. The first rule that matches wins.

### Rule 1 — Is it part of a name? → `NE`
Every token inside a named entity is `NE`, **including English function words inside a
title**, and **including names that carry a Turkish suffix**.

- `netflix` → NE
- `netflix'e`, `twitter'da`, `microsoft'un`, `messi'yle` → NE (not MIXED)
- `the` inside a title → NE (TurEngMix labels `the` as NE 21 times for this reason)
- numbers inside a name → NE: `xbox 360`, `windows 10`, `gpt-4`
- `google`, `youtube`, `instagram` → NE

### Rule 2 — Stem and suffix from different languages? → `MIXED`
Test: strip the Turkish ending. Is what is left an English word that is *not* a name?

- `entry'nin` → entry + nin → MIXED
- `linki` → link + i → MIXED
- `frameworkler` → framework + ler → MIXED
- `reply'ı`, `feed'lere`, `subreddit'den` → MIXED
- `tweetlememesi` → tweet + lememesi → MIXED
- `configurasyonlari` → MIXED (Turkish-spelled English stem + suffix still counts)

With or without an apostrophe makes no difference: `linki` and `link'i` are both MIXED.

### Rule 3 — A third language? → `OTHER`
- `seinen` (Japanese, a manga genre) → OTHER

### Rule 4 — A word both languages share, with no deciding context? → `AMBIGUOUS`
Two kinds only:
1. Words spelled identically and used in both languages — `platform`, `web`, `tv`,
   `model`, `anime`, `manga`, `protein` — **when the sentence around them does not show
   which language is being spoken**.
2. Interjections that are not clearly one language: `hey`, `ooh`, `pardon`.
   (`ha` is `TR` in TurEngMix.)

If the word sits inside a Turkish sentence, it is `TR`. If it sits inside an English
stretch, it is `EN`. `AMBIGUOUS` is the last resort. TurEngMix uses it for 0.4% of
tokens; if you are above roughly 1%, you are overusing it.

### Rule 5 — Otherwise: English → `EN`, Turkish → `TR`
Decide by **meaning in context**, not spelling alone.

- `her` in *her gün* ("every day") → TR; in *her phone* → EN
- `an` in *o an* ("that moment") → TR; in *an idea* → EN
- `on` in *on dakika* ("ten minutes") → TR; in *on the table* → EN

Nativised loanwords are `TR`: `telefon`, `bilgisayar`, `internet`, `site`, `film`,
`oyun`. An English word with English spelling and no suffix is `EN`: `online`, `post`,
`download`, `deadline`.

---

## 4. Edge cases, decided

These are taken from how TurEngMix labels them. The answer is fixed, so there is
nothing to argue about later.

| Token | Label | Why |
|---|---|---|
| `netflix'e` | NE | names beat MIXED |
| `linki` | MIXED | English stem + Turkish suffix |
| `link` | EN | no suffix |
| `lol`, `no` | EN | English |
| `abi`, `yani`, `ya`, `de`, `da`, `ha` | TR | Turkish |
| `bkz`, `vs`, `vb` | TR | Turkish abbreviations |
| `türkçe`, `ingilizce` | NE | language names (TurEngMix majority) |
| `xbox 360`, `windows 10` | NE | number is part of a name |
| `15dk`, `%50'sini`, `2026'da` | TR | number outside a name, in Turkish |
| `tv'nin` | TR | follows TurEngMix (seen once) |
| `site`, `internet` in a Turkish sentence | TR | nativised |
| keyboard mash (`kasdjkajh`) | — | TurEngMix is inconsistent (OTHER once, AMBIGUOUS once). Do not put these in generated data. |

---

## 5. Tokenization

**Split on spaces. Nothing else.** One token = one space-separated word.

- `download'ladım` is **one** token. The apostrophe never splits.
- Punctuation is never split off. `refund,` stays one token.

*Why not TurEngMix's own tokenizer script?* We tested it: it splits punctuation into
separate tokens (`refund,` → `refund` `,`), but their released benchmark contains no
such tokens. Splitting on spaces is what reproduces the benchmark exactly.

---

## 6. File format

One JSON object per line (`.jsonl`), UTF-8. One line = one sentence.

```json
{"id":"gen_000001","text":"yarın meeting'e geç kalacağım","tokens":["yarın","meeting'e","geç","kalacağım"],"labels":["TR","MIXED","TR","TR"],"source":"generated"}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | unique; `gen_` = generated, `seed_` = from TurEngMix |
| `text` | string | the sentence, lowercase |
| `tokens` | list of strings | output of the tokenizer |
| `labels` | list of strings | same length as `tokens`; each one of the six labels |
| `source` | string | `generated` or `seed` |

**Hard requirement:** `len(tokens) == len(labels)` on every line. The validator
rejects the file otherwise.

---

## 7. For the human-ceiling study

Three of us (Ibrahim, Mahmut, Tuğra) will each label the **same** sample,
**independently**. Do not talk about it while labelling and do not look at anyone
else's answers. We are measuring how often native speakers naturally disagree. That
number is the ceiling no model can be expected to beat.

For reference, TurEngMix's annotators reached Cohen's κ = **0.96** overall but only
**0.75 on MIXED**. Even experts disagree on the hard class.

If you are stuck on a token, write your best guess and move on. Do not leave blanks.
Expected time: about one hour.
