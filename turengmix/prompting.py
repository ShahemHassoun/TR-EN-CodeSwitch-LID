"""Shared input formatting and output parsing for prompted models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .labels import UNK, labels_for


def format_tokens(tokens: list[str]) -> str:
    return "\n".join(f"{i}\t{tok}" for i, tok in enumerate(tokens, start=1))


def _pattern(task: str) -> re.Pattern:
    labels = sorted(labels_for(task), key=len, reverse=True)
    alt = "|".join(re.escape(lab) for lab in labels)
    # Accept `|`, tab or colon as the separator. The prompts ask for `|`;
    return re.compile(rf"^\s*(\d+)\s*(?:\||\t|:)\s*({alt})\s*$")


@dataclass
class ParseResult:
    """Outcome of parsing one model response."""

    labels: dict[int, str]
    n_expected: int
    duplicate_indices: list[int] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)

    @property
    def missing_indices(self) -> list[int]:
        return sorted(set(range(1, self.n_expected + 1)) - set(self.labels))

    @property
    def extra_indices(self) -> list[int]:
        return sorted(set(self.labels) - set(range(1, self.n_expected + 1)))

    @property
    def aligned(self) -> bool:
        return not self.missing_indices and not self.extra_indices

    def ordered(self) -> list[str]:
        """Labels in token order, `UNK` wherever the model gave us nothing.
        """
        return [self.labels.get(i, UNK) for i in range(1, self.n_expected + 1)]


def parse_response(text: str, task: str, n_expected: int) -> ParseResult:
    """Parse `INDEX|LABEL` lines out of a model response."""
    pattern = _pattern(task)
    labels: dict[int, str] = {}
    duplicates: list[int] = []
    unparsed: list[str] = []

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            unparsed.append(line)
            continue
        idx, tag = int(m.group(1)), m.group(2)
        if idx in labels:
            duplicates.append(idx)
            continue
        labels[idx] = tag

    return ParseResult(
        labels=labels,
        n_expected=n_expected,
        duplicate_indices=sorted(set(duplicates)),
        unparsed_lines=unparsed,
    )


def strip_reasoning(text: str) -> str:
    """Remove a `<think>...</think>` block from a reasoning model's output.

    Qwen3 emits reasoning by default. The runner disables it where the
    provider supports doing so; this is the belt-and-braces path for when it
    comes back anyway, so reasoning prose is never parsed as labels.
    """
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
