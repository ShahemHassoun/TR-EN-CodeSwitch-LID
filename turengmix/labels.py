"""Label inventories for the TurEngMix benchmark.

Both tasks use a *fixed, closed* label set defined here.

`UNK` is not a label. It is the sentinel recorded when a model fails to
return a parseable label for a token; it is counted and reported separately
(see `turengmix.scoring`) and never folded into a class.
"""

from __future__ import annotations

# --- Language identification -------------------------------------------------
# Flat, non-BIO classes. NE is an LID class in its own right: a token that is
# part of a named entity takes LID label NE regardless of the BIO tag it
# carries in the `ner` column.
LID_LABELS: list[str] = ["TR", "EN", "MIXED", "OTHER", "NE", "AMBIGUOUS"]

# --- Named entity recognition ------------------------------------------------
NER_ENTITY_TYPES: list[str] = [
    "PER", "ORG", "LOC", "GROUP", "PROD", "TITLE", "EVENT", "TIME", "OTHER",
]
NER_LABELS: list[str] = ["O"] + [
    f"{prefix}-{t}" for t in NER_ENTITY_TYPES for prefix in ("B", "I")
]

# Sentinel for "the model returned nothing parseable for this token".
UNK = "UNK"

TASKS = ("lid", "ner")


def labels_for(task: str) -> list[str]:
    """Return the closed label list for `task` ("lid" or "ner")."""
    if task == "lid":
        return list(LID_LABELS)
    if task == "ner":
        return list(NER_LABELS)
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def gold_column(task: str) -> str:
    """Name of the gold-annotation column in the benchmark for `task`."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return task


def label2id(task: str) -> dict[str, int]:
    return {lab: i for i, lab in enumerate(labels_for(task))}


def id2label(task: str) -> dict[int, str]:
    return dict(enumerate(labels_for(task)))


def entity_type(tag: str) -> str | None:
    """Entity type of a BIO tag, or None for `O` / UNK."""
    if tag in ("O", UNK) or "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def is_wellformed_bio(tags: list[str]) -> list[int]:
    """Indices where an `I-X` tag has no `B-X` or `I-X` immediately before it.

    Returns an empty list for a well-formed sequence. Used by the
    normalisation step to surface annotation errors.
    """
    bad: list[int] = []
    previous: str | None = None
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            want = tag[2:]
            if previous is None or entity_type(previous) != want:
                bad.append(i)
        previous = tag
    return bad
