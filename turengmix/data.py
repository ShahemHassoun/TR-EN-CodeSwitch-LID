"""Loading the TurEngMix benchmark and its published split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .labels import UNK, gold_column, labels_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark.csv"
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits"

BENCHMARK_COLUMNS = [
    "doc_id", "sent_id", "tok_id", "token", "lid", "borrowed_suffix", "ner",
]


@dataclass(frozen=True)
class Sentence:
    """One annotation unit: the tokens sharing a (doc_id, sent_id)."""

    doc_id: str
    sent_id: int
    tokens: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.tokens)


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """Load the canonical benchmark and assert its invariants.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark not found at {path}\n"
        )
    df = pd.read_csv(
        path,
        dtype={"doc_id": str, "token": str, "lid": str, "ner": str,
               "borrowed_suffix": str},
        keep_default_na=False,
        na_values=[],
    )
    df["sent_id"] = df["sent_id"].astype(int)
    df["tok_id"] = df["tok_id"].astype(int)

    missing = [c for c in BENCHMARK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"benchmark is missing columns: {missing}")

    key = ["doc_id", "sent_id", "tok_id"]
    dupes = df[df.duplicated(key, keep=False)]
    if len(dupes):
        raise ValueError(
            f"{len(dupes)} rows share a (doc_id, sent_id, tok_id) key, "
            f"e.g. {dupes.iloc[0][key].to_dict()}"
        )
    for task in ("lid", "ner"):
        allowed = set(labels_for(task))
        bad = set(df[task].unique()) - allowed
        if bad:
            raise ValueError(f"unexpected {task} labels in benchmark: {sorted(bad)}")
    return df


def sentences(df: pd.DataFrame, task: str) -> list[Sentence]:
    """Group a benchmark frame into annotation units, in file order.

    Grouping is on (doc_id, sent_id) with `sort=False`, so tokens keep the
    order they appear in the file. 
    """
    col = gold_column(task)
    out: list[Sentence] = []
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        out.append(Sentence(
            doc_id=str(doc_id),
            sent_id=int(sent_id),
            tokens=[str(t) for t in g["token"]],
            labels=[str(v) for v in g[col]],
        ))
    return out


def load_split(split_dir: str | Path = DEFAULT_SPLIT_DIR) -> dict[str, list[str]]:
    """Read the committed document-level split as {split_name: [doc_id, ...]}."""
    split_dir = Path(split_dir)
    split = {}
    for name in ("train", "validation", "test"):
        f = split_dir / f"{name}.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"split file not found at {f}\n"
                "Run:  python scripts/make_splits.py"
            )
        split[name] = [
            line.strip() for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    overlap = (set(split["train"]) & set(split["validation"])) \
        | (set(split["train"]) & set(split["test"])) \
        | (set(split["validation"]) & set(split["test"]))
    if overlap:
        raise ValueError(f"documents appear in more than one split: {sorted(overlap)}")
    return split


def split_sentences(
    df: pd.DataFrame, task: str, split_dir: str | Path = DEFAULT_SPLIT_DIR
) -> dict[str, list[Sentence]]:
    """Sentences grouped by split. Splitting is by document, never by sentence,
    so no post contributes tokens to two splits."""
    split = load_split(split_dir)
    known = set(df["doc_id"].unique())
    for name, docs in split.items():
        unknown = set(docs) - known
        if unknown:
            raise ValueError(
                f"split '{name}' names {len(unknown)} documents absent from the "
                f"benchmark, e.g. {sorted(unknown)[:3]}"
            )
    by_doc: dict[str, list[Sentence]] = {}
    for s in sentences(df, task):
        by_doc.setdefault(s.doc_id, []).append(s)
    return {name: [s for d in docs for s in by_doc.get(d, [])]
            for name, docs in split.items()}


def write_predictions(
    df: pd.DataFrame, predictions: dict[tuple[str, int, int], str],
    column: str, path: str | Path,
) -> pd.DataFrame:
    """Attach a prediction column keyed by (doc_id, sent_id, tok_id) and save.
    """
    out = df.copy()
    keys = list(zip(out["doc_id"], out["sent_id"], out["tok_id"]))
    out[column] = [predictions.get(k, UNK) for k in keys]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def dump_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
