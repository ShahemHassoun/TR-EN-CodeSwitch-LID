#!/usr/bin/env python3
"""Score every prediction file the same way and build the results table.

    python src/score_all.py

Reads   results/predictions/*_test.csv   (and *_validation.csv when present)
Writes  results/results_table.csv  and  results/results_table.md

Rules (identical for every model, so the comparison is fair):
  * Every model is scored on the same 25 TurEngMix TEST documents (1,373 tokens).
  * Metric = macro-F1 over the six LID labels, using TurEngMix's own scoring
    code (turengmix/scoring.py). Per-label F1 is reported too.
  * Runs named <base>_seed<N> are grouped: mean and SD over seeds.
  * 95% confidence interval: resample the 25 test documents with replacement
    2,000 times (bootstrap). For multi-seed rows, each resample scores the
    seed-averaged macro-F1.
  * If --reference is given, every row also gets the difference to that row,
    with its own 95% interval from the same resamples (paired bootstrap).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid_io import PRED_DIR, REPO_ROOT, _split_frame  # noqa: E402
from turengmix import scoring  # noqa: E402
from turengmix.labels import LID_LABELS  # noqa: E402

SEED_RE = re.compile(r"^(?P<base>.+?)_seed(?P<seed>\d+)$")
N_BOOT = 2000


def read_pred(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    if "pred_lid" not in df.columns:
        raise ValueError(f"{path.name}: no pred_lid column")
    gold = _split_frame(split)
    keys = ["doc_id", "sent_id", "tok_id"]
    gold[keys[1:]] = gold[keys[1:]].astype(str)
    merged = gold[keys + ["lid"]].merge(df[keys + ["pred_lid"]], on=keys, how="left")
    missing = merged["pred_lid"].isna().sum()
    if missing:
        raise ValueError(f"{path.name}: {missing} {split} tokens have no prediction")
    return merged


def macro_f1(gold, pred) -> float:
    return scoring.token_metrics(list(gold), list(pred), "lid")["macro_f1"]


CLASSES = list(LID_LABELS) + ["UNK"]   # UNK = no usable prediction; always wrong
C_INDEX = {c: i for i, c in enumerate(CLASSES)}


def doc_confusions(df: pd.DataFrame) -> np.ndarray:
    """Array [document, gold, pred] of token counts, documents in sorted order."""
    docs = sorted(df["doc_id"].unique())
    d_index = {d: i for i, d in enumerate(docs)}
    cm = np.zeros((len(docs), len(CLASSES), len(CLASSES)), dtype=np.int64)
    pred = [p if p in C_INDEX else "UNK" for p in df["pred_lid"]]
    np.add.at(cm, ([d_index[d] for d in df["doc_id"]],
                   [C_INDEX[g] for g in df["lid"]],
                   [C_INDEX[p] for p in pred]), 1)
    return cm


def f1_from_confusion(cm: np.ndarray) -> float:
    """Macro-F1 over the six LID labels; a label with no gold and no
    prediction scores 0, exactly like scikit-learn with zero_division=0."""
    f1s = []
    for i in range(len(LID_LABELS)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def collect(split: str) -> dict[str, dict[int | None, pd.DataFrame]]:
    groups: dict[str, dict[int | None, pd.DataFrame]] = {}
    for path in sorted(PRED_DIR.glob(f"*_{split}.csv")):
        run = path.name[: -len(f"_{split}.csv")]
        m = SEED_RE.match(run)
        base, seed = (m.group("base"), int(m.group("seed"))) if m else (run, None)
        groups.setdefault(base, {})[seed] = read_pred(path, split)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="berturk_gen0",
                    help="row to compare every other row against")
    args = ap.parse_args()

    test = collect("test")
    val = collect("validation")
    gold_test = _split_frame("test")

    # Majority-class baseline, computed directly from the gold labels.
    majority = Counter(gold_test["lid"]).most_common(1)[0][0]
    test = {"majority": {None: gold_test.assign(pred_lid=majority)[
        ["doc_id", "sent_id", "tok_id", "lid", "pred_lid"]]}} | test

    docs = sorted(gold_test["doc_id"].unique())
    rng = np.random.default_rng(0)
    samples = [rng.choice(len(docs), size=len(docs), replace=True) for _ in range(N_BOOT)]

    def boot_scores(runs: dict) -> np.ndarray:
        """Seed-averaged macro-F1 on each bootstrap resample of documents.

        Fast version: one confusion matrix per document, summed per resample.
        Gives exactly the same macro-F1 as scoring.token_metrics (checked in
        on every run in main()).
        """
        out = np.zeros(N_BOOT)
        for df in runs.values():
            per_doc = doc_confusions(df)
            for b, idx in enumerate(samples):
                out[b] += f1_from_confusion(per_doc[idx].sum(axis=0))
        return out / len(runs)

    rows, boots = [], {}
    for base, runs in test.items():
        scores = [macro_f1(df["lid"], df["pred_lid"]) for df in runs.values()]
        per_class = {lab: np.mean([
            scoring.token_metrics(df["lid"].tolist(), df["pred_lid"].tolist(), "lid")
            ["per_class"][lab]["f1"] for df in runs.values()]) for lab in LID_LABELS}
        for df in runs.values():   # safety check: fast formula == official scorer
            fast = f1_from_confusion(doc_confusions(df).sum(axis=0))
            official = macro_f1(df["lid"], df["pred_lid"])
            if abs(fast - official) > 1e-9:
                raise AssertionError(f"{base}: fast F1 {fast} != official {official}")
        boots[base] = boot_scores(runs)
        lo, hi = np.percentile(boots[base], [2.5, 97.5])
        vscores = [macro_f1(df["lid"], df["pred_lid"]) for df in val.get(base, {}).values()]
        rows.append({
            "run": base,
            "seeds": len(runs) if None not in runs else 1,
            "val_macro_f1": round(float(np.mean(vscores)), 4) if vscores else None,
            "test_macro_f1": round(float(np.mean(scores)), 4),
            "test_sd": round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else None,
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            **{f"f1_{lab}": round(float(v), 4) for lab, v in per_class.items()},
        })
        print(f"scored {base:<28} test macro-F1 {np.mean(scores):.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  ({rows[-1]['seeds']} seed(s))")

    if args.reference in boots:
        ref = boots[args.reference]
        for r in rows:
            diff = boots[r["run"]] - ref
            r[f"diff_vs_{args.reference}"] = round(float(
                r["test_macro_f1"] - next(x["test_macro_f1"] for x in rows
                                          if x["run"] == args.reference)), 4)
            r["diff_ci95_low"], r["diff_ci95_high"] = (
                round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5]))
    else:
        print(f"note: reference run '{args.reference}' not found; no differences computed")

    table = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results" / "results_table.csv"
    table.to_csv(out_csv, index=False)
    (REPO_ROOT / "results" / "results_table.md").write_text(
        table.to_markdown(index=False), encoding="utf-8")
    print(f"\nwrote {out_csv.relative_to(REPO_ROOT)} and results_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
