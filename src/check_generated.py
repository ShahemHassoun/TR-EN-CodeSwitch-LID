#!/usr/bin/env python3
"""Quality checks for our generated dataset.

    python src/check_generated.py data/generated/generated.jsonl

Every line must pass every HARD check, or the file is rejected (exit code 1).
SOFT checks list tokens worth a human look; they do not reject the file.

HARD checks
  1. valid JSON with id, text, tokens, labels, source == "generated"; ids unique
  2. tokens are exactly text.split(" ")     (our tokenization rule)
  3. one label per token, each one of the six TurEngMix labels
  4. no capital letters                      (TurEngMix text is lowercase)
  5. no token made only of punctuation
  6. no exact duplicate sentences
  7. no copying: no run of 8 tokens shared with any TurEngMix sentence
  8. no test leakage: no token that TurEngMix labels MIXED / NE / OTHER /
     AMBIGUOUS in its validation or test portion unless it also appears in
     the training portion; plus a short explicit block-list
  10. no token labelled EN that is clearly Turkish (Turkish frequency >= 3,
     English frequency < 1.5), e.g. a Turkish quote put inside << >> by mistake
  9. every MIXED token carries the split we recorded when writing it
     ("mixed": {"<token index>": ["link", "i"]}), and that split is checked:
     the two parts spell the token, the stem is an English word (wordfreq),
     and the ending is a real chain of Turkish suffixes (SUFFIXES list)

SOFT checks (printed for review)
  * TR tokens that look like English stem + Turkish ending (missed MIXED?)
  * TR tokens that look English; EN tokens that look Turkish or are not
    known English words (maybe OTHER or NE)
  * sentence openings repeated so often they look like templates

Finally prints the label distribution next to TurEngMix's training portion.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid_io import load_split_sentences  # noqa: E402
from turengmix.labels import LID_LABELS  # noqa: E402

from wordfreq import zipf_frequency  # noqa: E402

PUNCT_ONLY = re.compile(r"^[^\w]+$")
APOS = "'’"
TR_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyz")
# Words that look like test-set items we happened to see while checking the
# label rules; banned outright so they cannot leak into training data.
BLOCKLIST = {"sans", "culotte", "culotte'lar"}
NGRAM = 8


def en(w: str) -> float:
    return zipf_frequency(w, "en")


def tr(w: str) -> float:
    return zipf_frequency(w, "tr")


def english_stem(stem: str, threshold: float) -> bool:
    """A known English word, an abbreviation-like stem, or two English words
    glued together (websocket, moodboard, techbro)."""
    stem = stem.replace("-", "")
    if en(stem) >= threshold:
        return True
    return any(en(stem[:k]) >= 3.0 and en(stem[k:]) >= 3.0
               for k in range(3, len(stem) - 2))


V = "[aeıioöuü]"          # any Turkish vowel (vowel harmony is not enforced)
A = "[ae]"                 # two-way harmony vowel
I = "[ıiuü]"               # four-way harmony vowel
D = "[dt]"
# Inflectional and common derivational Turkish suffixes. ğ/ş also accept g/s,
# because some sentences are typed without Turkish letters (see build_generated.py). An ending counts as
# Turkish only if it can be cut, start to end, into pieces from this list.
SUFFIXES = [
    f"l{A}r", f"{I}?m{I}z", f"{I}?n{I}z", f"{I}?m", f"{I}?n", f"s?{I}", f"n{I}",
    f"y?{I}", f"y?{A}", f"n?{D}{A}", f"n?{D}{A}n", f"n?{I}n", f"y?l{A}",
    "ki", f"{D}{I}r", f"y?{D}{I}", f"y?m{I}[şs]", f"y?s{A}", f"{D}{I}k", f"{D}{I}[ğg]",
    f"l{A}", f"l{A}[şs]", f"m{A}k", f"m{A}", f"m{A}z", f"m{A}s{I}", f"{I}?yor",
    f"y?{A}c{A}k", f"y?{A}c{A}[ğg]", f"{I}?l", f"{I}?[şs]", f"{D}", f"y?{I}p",
    f"y?{A}n", f"y?{A}r", f"{I}r", f"y?{A}bil", f"l{I}k", f"l{I}[ğg]", f"c{I}", f"ç{I}",
    f"s{I}z", f"l{I}", "k", f"l{A}rd{A}", f"y?k{A}n", f"c{A}", f"ç{A}",
]
SUFFIX_CHAIN = re.compile("^(?:" + "|".join(SUFFIXES) + ")+$")


def mixed_ok(token: str, stem: str, ending: str) -> tuple[bool, str]:
    """Check one MIXED token against the split recorded when it was written.

    token  : the word as it appears, e.g. "link'i" or "linki"
    stem   : the English part, e.g. "link"
    ending : the Turkish part, e.g. "i"
    """
    t = token.strip(".,!?;:\"()[]")
    joined_ok = t in (stem + ending, *(stem + a + ending for a in APOS))
    if not joined_ok:
        return False, f"'{stem}' + '{ending}' does not spell '{t}'"
    if not SUFFIX_CHAIN.match(ending):
        return False, f"ending '{ending}' is not a Turkish suffix chain"
    if not any(ch.isalpha() for ch in stem):
        return False, f"stem '{stem}' is a number (number + Turkish ending is TR)"
    has_apostrophe = any(a in t for a in APOS)
    short_abbrev = has_apostrophe and stem.isalpha() and len(stem) <= 4   # pc'den, cv'si
    if not (english_stem(stem, 2.5) or short_abbrev):
        return False, f"stem '{stem}' is not a known English word"
    if tr(stem) >= 4.0 and tr(stem) > en(stem) + 0.5:
        return False, f"stem '{stem}' is mainly a Turkish word"
    if tr(t) >= 5.0:      # e.g. 'ama', 'bana': ordinary Turkish words (real MIXED max is ~4.4)
        return False, f"'{t}' is a common Turkish word"
    return True, f"{stem} + {ending}"


def guess_split(token: str) -> tuple[str, str] | None:
    """Best guess at English stem + Turkish ending, used only for SOFT warnings
    on tokens labelled TR (a possible missed MIXED)."""
    t = token.strip(".,!?;:\"()[]")
    for k in range(len(t) - 1, 3, -1):           # stems of 4+ letters only
        stem, ending = t[:k], t[k:]
        if (SUFFIX_CHAIN.match(ending) and en(stem) >= 3.5
                and en(stem) > tr(stem) + 0.5 and tr(t) < 2.0):
            return stem, ending
    return None


def main(path: str) -> int:
    hard: list[str] = []
    soft: list[str] = []

    train = load_split_sentences("train")
    held = load_split_sentences("validation") + load_split_sentences("test")
    train_tokens = {t for s in train for t in s.tokens}
    leak = {t for s in held for t, l in zip(s.tokens, s.labels)
            if l in {"MIXED", "NE", "OTHER", "AMBIGUOUS"} and t not in train_tokens}
    leak |= BLOCKLIST
    seen_ngrams = {tuple(s.tokens[i:i + NGRAM]) for s in train + held
                   for i in range(len(s.tokens) - NGRAM + 1)}

    rows, ids, texts = [], set(), Counter()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"line {n}"
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                hard.append(f"{where}: not valid JSON ({e})")
                continue
            missing = {"id", "text", "tokens", "labels", "source", "mixed"} - set(r)
            if missing:
                hard.append(f"{where}: missing fields {sorted(missing)}")
                continue
            where = f"line {n} ({r['id']})"
            if r["source"] != "generated":
                hard.append(f"{where}: source must be 'generated'")
            if r["id"] in ids:
                hard.append(f"{where}: duplicate id")
            ids.add(r["id"])
            toks, labs, text = r["tokens"], r["labels"], r["text"]
            if toks != text.split(" "):
                hard.append(f"{where}: tokens are not text.split(' ')")
            if len(toks) != len(labs):
                hard.append(f"{where}: {len(toks)} tokens but {len(labs)} labels")
                continue
            bad = set(labs) - set(LID_LABELS)
            if bad:
                hard.append(f"{where}: unknown labels {sorted(bad)}")
            if any(c.isupper() for c in text):
                hard.append(f"{where}: contains capital letters")
            texts[text] += 1
            for i in range(len(toks) - NGRAM + 1):
                if tuple(toks[i:i + NGRAM]) in seen_ngrams:
                    hard.append(f"{where}: copies 8 tokens from TurEngMix: {' '.join(toks[i:i+NGRAM])}")
                    break
            for t in toks:
                if PUNCT_ONLY.match(t):
                    hard.append(f"{where}: punctuation-only token '{t}'")
                if t in leak:
                    hard.append(f"{where}: token '{t}' is blocked (test leakage)")
            splits = r.get("mixed", {})
            for i, (t, l) in enumerate(zip(toks, labs)):
                if l == "MIXED":
                    split = splits.get(str(i))
                    if not split or len(split) != 2:
                        hard.append(f"{where}: MIXED '{t}' has no recorded split")
                        continue
                    ok, why = mixed_ok(t, split[0], split[1])
                    if not ok:
                        hard.append(f"{where}: MIXED '{t}': {why}")
                elif l == "TR" and guess_split(t):
                    soft.append(f"{where}: TR '{t}' could be MIXED {guess_split(t)}")
                elif l == "TR" and en(t) >= 4.0 and tr(t) < 1.0:
                    soft.append(f"{where}: TR '{t}' looks English")
                elif l == "EN":
                    w = t.strip(".,!?;:\"()[]")
                    if tr(w) >= 3.0 and en(w) < 1.5:
                        hard.append(f"{where}: EN '{t}' is a Turkish word")
                    elif tr(w) >= 3.0 and tr(w) >= en(w) + 1.5:
                        soft.append(f"{where}: EN '{t}' looks Turkish")
                    elif en(w) < 1.0 and not any(c.isdigit() for c in w):
                        soft.append(f"{where}: EN '{t}' is not a known English word (OTHER? NE?)")
            rows.append(r)

    for text, c in texts.items():
        if c > 1:
            hard.append(f"duplicate sentence x{c}: {text}")
    openings = Counter(" ".join(r["tokens"][:4]) for r in rows)
    for o, c in openings.most_common(10):
        if c > max(5, len(rows) * 0.005):
            soft.append(f"opening '{o}' used {c} times (template-like?)")

    # ----- report -----
    def dist(pairs):
        c = Counter(pairs)
        total = sum(c.values()) or 1
        return {lab: 100 * c[lab] / total for lab in LID_LABELS}

    ours = dist(l for r in rows for l in r["labels"])
    theirs = dist(l for s in train for l in s.labels)
    n_tok = sum(len(r["tokens"]) for r in rows)
    print(f"\n{path}: {len(rows):,} sentences, {n_tok:,} tokens, "
          f"{n_tok / max(1, len(rows)):.1f} tokens per sentence")
    print(f"{'label':<10}{'ours %':>8}{'TurEngMix train %':>20}")
    for lab in LID_LABELS:
        print(f"{lab:<10}{ours[lab]:>8.2f}{theirs[lab]:>20.2f}")
    tr_toks = [t for r in rows for t, l in zip(r["tokens"], r["labels"]) if l == "TR"]
    if tr_toks:
        share = sum(any(ch in "çğıöşü" for ch in t) for t in tr_toks) / len(tr_toks)
        print(f"TR tokens with Turkish letters: {100 * share:.1f}%  (TurEngMix: ~30%)")
    with_mixed = sum("MIXED" in r["labels"] for r in rows)
    print(f"sentences with at least one MIXED token: {100 * with_mixed / max(1, len(rows)):.1f}%")

    print(f"\nSOFT warnings (review, not errors): {len(soft)}")
    for s in soft[:40]:
        print("  " + s)
    print(f"\nHARD errors: {len(hard)}")
    for h in hard[:60]:
        print("  " + h)
    if hard:
        print("\nREJECTED: fix the hard errors above.")
        return 1
    print("\nPASSED all hard checks.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python src/check_generated.py <file.jsonl>")
    raise SystemExit(main(sys.argv[1]))
