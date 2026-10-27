#!/usr/bin/env python3
"""Build data/generated/generated.jsonl from the raw batch files we wrote.

    python src/build_generated.py

Raw format (data/generated/raw/batch_NN.txt), one sentence per line:
  * words are separated by single spaces; a word with no tag is Turkish (TR)
  * word|E  English      word|N  name (NE)      word|A  AMBIGUOUS     word|O  OTHER
  * word|M=stem+ending   MIXED, with the split recorded, e.g. linki|M=link+i
  * << ... >>   every word inside is EN unless it has its own tag
  * {{ ... }}   every word inside is NE (multi-word names and titles)
  * lines starting with # are comments

Two realism steps, both documented in data/generated/README.md:
  1. ASCII typing: TurEngMix users often type Turkish without Turkish letters
     (only ~30% of its Turkish words contain ç ğ ı ö ş ü). A fixed 25% of our
     sentences are converted the same way (ç→c ğ→g ı→i ö→o ş→s ü→u).
  2. Fixed order: batches are shuffled internally and appended in order, so
     the first 2,500 sentences never change when more batches are added
     (gen2500 ⊂ gen5000 ⊂ gen10000).

After building, run:  python src/check_generated.py data/generated/generated.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "generated" / "raw"
OUT = REPO_ROOT / "data" / "generated" / "generated.jsonl"
TAGS = {"E": "EN", "N": "NE", "A": "AMBIGUOUS", "O": "OTHER", "T": "TR"}
ASCII = str.maketrans("çğıöşü", "cgiosu")
ASCII_SHARE = 0.25
SEED = 13


def parse_line(line: str, where: str) -> dict:
    tokens, labels, mixed = [], [], {}
    mode = "TR"
    for raw in line.split(" "):
        if raw == "":
            raise ValueError(f"{where}: double space")
        if any(m in raw for m in ("<<", ">>", "{{", "}}")) and raw not in ("<<", ">>", "{{", "}}"):
            raise ValueError(f"{where}: marker glued to text in '{raw}' (put spaces around << >> {{{{ }}}})")
        if raw == "<<":
            mode = "EN"; continue
        if raw == "{{":
            mode = "NE"; continue
        if raw in (">>", "}}"):
            mode = "TR"; continue
        if "|" in raw:
            word, tag = raw.rsplit("|", 1)
            trail = len(tag) - len(tag.rstrip(",.;:!?"))     # "x|E," → word "x,"
            if trail:
                word, tag = word + tag[-trail:], tag[:-trail]
            if tag.startswith("M="):
                stem, _, ending = tag[2:].partition("+")
                if not stem or not ending:
                    raise ValueError(f"{where}: bad MIXED tag '{raw}'")
                mixed[str(len(tokens))] = [stem, ending]
                label = "MIXED"
            elif tag in TAGS:
                label = TAGS[tag]
            else:
                raise ValueError(f"{where}: unknown tag '{raw}'")
        else:
            word, label = raw, mode
        if not word:
            raise ValueError(f"{where}: empty word in '{raw}'")
        if "|" in word:
            raise ValueError(f"{where}: stray '|' inside word '{raw}'")
        tokens.append(word)
        labels.append(label)
    if mode != "TR":
        raise ValueError(f"{where}: unclosed << or {{{{")
    return {"tokens": tokens, "labels": labels, "mixed": mixed}


def asciify(row: dict) -> dict:
    row["tokens"] = [t.translate(ASCII) for t in row["tokens"]]
    row["mixed"] = {k: [s.translate(ASCII), e.translate(ASCII)] for k, (s, e) in row["mixed"].items()}
    return row


def main() -> int:
    batches = sorted(RAW_DIR.glob("batch_*.txt"))
    if not batches:
        raise SystemExit(f"no batch files in {RAW_DIR}")
    rows = []
    for b in batches:
        sentences = []
        for n, line in enumerate(b.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = parse_line(line, f"{b.name}:{n}")
            row["batch"] = b.stem
            sentences.append(row)
        rng = random.Random(f"{SEED}-{b.stem}")       # each batch: own fixed shuffle
        rng.shuffle(sentences)
        for row in sentences:
            if rng.random() < ASCII_SHARE:
                asciify(row)
        rows.extend(sentences)
        print(f"{b.name}: {len(sentences)} sentences")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(rows, start=1):
            rec = {"id": f"gen_{i:06d}", "text": " ".join(row["tokens"]),
                   "tokens": row["tokens"], "labels": row["labels"],
                   "source": "generated", "mixed": row["mixed"], "batch": row["batch"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(rows)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
