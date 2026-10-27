"""Metrics for the TurEngMix baselines.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .labels import NER_ENTITY_TYPES, UNK, entity_type, labels_for

# `borrowed_suffix == "MIXED"` marks a token whose English stem carries a
# Turkish suffix — the "INTEGRATED" category of §4.1.1, 386 tokens.
INTEGRATED = "MIXED"


def _clean(pred: list[str], allowed: set[str]) -> list[str]:
    """Map anything outside the label set to UNK, so it is counted as an
    error."""
    return [p if p in allowed else UNK for p in pred]


# ---------------------------------------------------------------------------
# Token-level metrics — Tables 8, 9, 10
# ---------------------------------------------------------------------------

def token_metrics(gold: list[str], pred: list[str], task: str) -> dict:
    """Flat token-level metrics over the closed label set.

    LID  -> Table 9 (macro F1) and Table 8 (per-class F1), six classes.
    NER  -> Table 10 (BIO-level macro F1), nineteen BIO labels.
    """
    labels = labels_for(task)
    pred = _clean(pred, set(labels))
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, average="macro", zero_division=0)
    micro_f1 = precision_recall_fscore_support(
        gold, pred, labels=labels, average="micro", zero_division=0)[2]
    per_p, per_r, per_f1, per_n = precision_recall_fscore_support(
        gold, pred, labels=labels, average=None, zero_division=0)

    n_unk = sum(p == UNK for p in pred)
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])),
        "per_class": {
            lab: {"precision": float(per_p[i]), "recall": float(per_r[i]),
                  "f1": float(per_f1[i]), "support": int(per_n[i])}
            for i, lab in enumerate(labels)
        },
        "n_tokens": len(gold),
        "n_malformed": n_unk,
        "malformed_rate": n_unk / max(1, len(gold)),
    }


def ner_type_metrics(gold: list[str], pred: list[str]) -> dict:
    """Per-entity-type F1 with B- and I- collapsed — Tables A3 and A4.

    `B-PER` and `I-PER` both become `PER`; `O` stays `O`. Ten classes    
    """
    allowed = set(labels_for("ner"))
    pred = _clean(pred, allowed)
    classes = ["O"] + NER_ENTITY_TYPES

    def collapse(tag: str) -> str:
        if tag == UNK:
            return UNK
        return entity_type(tag) or "O"

    g = [collapse(t) for t in gold]
    p = [collapse(t) for t in pred]
    per_p, per_r, per_f1, per_n = precision_recall_fscore_support(
        g, p, labels=classes, average=None, zero_division=0)
    macro_f1 = precision_recall_fscore_support(
        g, p, labels=classes, average="macro", zero_division=0)[2]
    return {
        "macro_f1": float(macro_f1),
        "per_type": {
            c: {"f1": float(per_f1[i]), "precision": float(per_p[i]),
                "recall": float(per_r[i]), "support": int(per_n[i])}
            for i, c in enumerate(classes)
        },
    }


def ner_binary_metrics(gold: list[str], pred: list[str]) -> dict:
    """Binary entity detection: NE vs Non-NE — Table 11.

    Collapses every entity category into one class. Separates "failed to
    notice an entity" from "found it but mislabelled the type or the span".
    """
    allowed = set(labels_for("ner"))
    pred = _clean(pred, allowed)
    binar = lambda t: "Non-NE" if t in ("O", UNK) else "NE"  # noqa: E731
    g = [binar(t) for t in gold]
    p = [binar(t) for t in pred]
    classes = ["Non-NE", "NE"]
    _, _, f1, n = precision_recall_fscore_support(
        g, p, labels=classes, average=None, zero_division=0)
    return {
        "non_ne_f1": float(f1[0]), "ne_f1": float(f1[1]),
        "non_ne_support": int(n[0]), "ne_support": int(n[1]),
    }


def majority_baseline(gold: list[str], task: str) -> dict:
    """Majority-class baseline — the `Baseline` row of Tables 9, 10 and A4.

    Predicts the most frequent gold class for every token. 
    """
    majority = Counter(gold).most_common(1)[0][0]
    return token_metrics(gold, [majority] * len(gold), task) | {
        "majority_class": majority}


# ---------------------------------------------------------------------------
# Morphological integration — Table 12, §6.3.3
# ---------------------------------------------------------------------------

def integration_error_analysis(
    gold: list[str], pred: list[str], borrowed_suffix: list[str], task: str,
) -> dict:
    """Error rates on morphologically integrated tokens vs all others.

    Reproduces Table 12 and the chi-square test in §6.3.3, which compares
    "tokens containing English-origin stems bearing Turkish suffixes
    (including morphologically adapted named entities)" against "all other
    evaluated tokens" — so the denominator is every non-integrated token.
    """
    allowed = set(labels_for(task))
    pred = _clean(pred, allowed)
    integrated = [s.strip() == INTEGRATED for s in borrowed_suffix]
    if not (len(gold) == len(pred) == len(integrated)):
        raise ValueError("gold, pred and borrowed_suffix must be the same length")

    a = sum(1 for i, f in enumerate(integrated) if f and gold[i] != pred[i])
    b = sum(1 for i, f in enumerate(integrated) if f and gold[i] == pred[i])
    c = sum(1 for i, f in enumerate(integrated) if not f and gold[i] != pred[i])
    d = sum(1 for i, f in enumerate(integrated) if not f and gold[i] == pred[i])

    mixed_rate = a / (a + b) if (a + b) else float("nan")
    other_rate = c / (c + d) if (c + d) else float("nan")

    if 0 in (a, b, c, d):
        odds = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
        corrected = True
    else:
        odds = (a * d) / (b * c)
        corrected = False

    out = {
        "mixed_error_rate": mixed_rate,
        "other_error_rate": other_rate,
        "ratio": (mixed_rate / other_rate) if other_rate else float("nan"),
        "odds_ratio": float(odds),
        "odds_ratio_haldane_corrected": corrected,
        "contingency": {"mixed_wrong": a, "mixed_right": b,
                        "other_wrong": c, "other_right": d},
        "n_integrated": a + b,
        "n_other": c + d,
    }
 
    out["chi2"] = out["p_value"] = None
    if min(a + b, c + d, a + c, b + d) > 0:
        try:
            from scipy.stats import chi2_contingency
            chi2, p, dof, _ = chi2_contingency([[a, b], [c, d]], correction=False)
            out |= {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}
        except ImportError:  
            pass
    return out


def error_rate_by_lid(
    gold: list[str], pred: list[str], lid_gold: list[str], task: str,
) -> dict:
    """Error rate broken down by the token's gold LID class, with Wilson 95%
    intervals — supporting the per-class discussion in §6.1 and §6.3.
    """
    allowed = set(labels_for(task))
    pred = _clean(pred, allowed)
    buckets: dict[str, list[bool]] = {}
    for g, p, lid in zip(gold, pred, lid_gold):
        buckets.setdefault(lid, []).append(g != p)
    out = {}
    for name, errors in sorted(buckets.items()):
        n, k = len(errors), sum(errors)
        lo, hi = wilson(k, n)
        out[name] = {"n_tokens": n, "n_errors": k,
                     "error_rate": k / n if n else 0.0, "ci95": [lo, hi]}
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_text(gold: list[str], pred: list[str], task: str) -> str:
    labels = labels_for(task)
    pred = _clean(pred, set(labels))
    return classification_report(gold, pred, labels=labels, target_names=labels,
                                 zero_division=0, digits=4)


def confusion(gold: list[str], pred: list[str], task: str):
    """Confusion matrix with UNK as a real column, so rows still sum to the
    class support when a model returns malformed output."""
    labels = labels_for(task)
    pred = _clean(pred, set(labels))
    axis = labels + [UNK]
    return axis, confusion_matrix(gold, pred, labels=axis)


def aggregate_seeds(values: list[float]) -> dict:
    """Mean and sample standard deviation across random seeds.
    """
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def label_distribution(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())
