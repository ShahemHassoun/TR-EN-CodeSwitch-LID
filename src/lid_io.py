"""Shared data loading and saving for every model in this project.

Every model (rules, CRF, BERTurk, XLM-R, LLM) reads the same sentences and
writes predictions in the same file format, so one script (score_all.py)
scores all of them identically.

    from lid_io import load_split_sentences, load_generated, save_predictions

Real data   : TurEngMix benchmark, data/benchmark.csv + data/splits/*.txt
Our data    : data/generated/generated.jsonl  (one sentence per line)
Predictions : results/predictions/<run>_<split>.csv
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from turengmix.data import load_benchmark, load_split, write_predictions  # noqa: E402
from turengmix.labels import LID_LABELS, UNK  # noqa: E402

GENERATED_PATH = REPO_ROOT / "data" / "generated" / "generated.jsonl"
PRED_DIR = REPO_ROOT / "results" / "predictions"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Sentence:
    doc_id: str
    sent_id: int
    tokens: list[str]
    labels: list[str]


def _split_frame(split: str):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    df = load_benchmark()
    keep = set(load_split()[split])
    return df[df["doc_id"].isin(keep)].reset_index(drop=True)


def load_split_sentences(split: str) -> list[Sentence]:
    """Real TurEngMix sentences for one split, in file order."""
    df = _split_frame(split)
    out = []
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        out.append(Sentence(str(doc_id), int(sent_id),
                            g["token"].tolist(), g["lid"].tolist()))
    return out


def load_generated(n: int | None = None, path: str | Path = GENERATED_PATH) -> list[Sentence]:
    """The first `n` sentences of our generated dataset (all if n is None).

    The file is in a fixed shuffled order, so the first 2,500 are always
    contained in the first 5,000, which are contained in the first 10,000.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            toks, labs = row["tokens"], row["labels"]
            if len(toks) != len(labs):
                raise ValueError(f"{path}:{line_no}: {len(toks)} tokens but {len(labs)} labels")
            bad = set(labs) - set(LID_LABELS)
            if bad:
                raise ValueError(f"{path}:{line_no}: unknown labels {sorted(bad)}")
            out.append(Sentence(row["id"], 1, toks, labs))
            if n is not None and len(out) == n:
                break
    if n is not None and len(out) < n:
        raise ValueError(f"asked for {n} generated sentences but {path} has only {len(out)}")
    return out


def save_predictions(run: str, split: str,
                     preds: dict[tuple[str, int], list[str]]) -> Path:
    """Save one model's predictions for one split.

    run   : short name, e.g. "crf_train" or "berturk_gen5000_seed1"
    preds : {(doc_id, sent_id): [label for each token]}  for EVERY sentence
            returned by load_split_sentences(split)

    Writes results/predictions/<run>_<split>.csv : the benchmark rows of that
    split plus a `pred_lid` column. Same format as the TurEngMix scripts.
    """
    df = _split_frame(split)
    flat: dict[tuple[str, int, int], str] = {}
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        key = (str(doc_id), int(sent_id))
        if key not in preds:
            raise ValueError(f"no prediction for sentence {key}")
        labs = preds[key]
        if len(labs) != len(g):
            raise ValueError(f"sentence {key}: {len(g)} tokens but {len(labs)} predictions")
        for tok_id, lab in zip(g["tok_id"], labs):
            flat[(key[0], key[1], int(tok_id))] = lab if lab in LID_LABELS else UNK
    path = PRED_DIR / f"{run}_{split}.csv"
    write_predictions(df, flat, "pred_lid", path)
    return path
