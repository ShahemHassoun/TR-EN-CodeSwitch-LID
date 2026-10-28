#!/usr/bin/env python3
"""Run word-frequency and Zemberek rule baselines for Turkish-English LID.

Run from the repository root with Python 3.12:

    python src/baseline_rules.py

The rules are fixed (not learned or tuned on validation or test). Named
entities are the only information taken from the train split. Predictions are
saved through ``lid_io.save_predictions`` for validation and test.
"""

from __future__ import annotations

import sys
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from lid_io import Sentence, load_split_sentences, save_predictions  # noqa: E402
from wordfreq import word_frequency  # noqa: E402
from zemberek import TurkishMorphology  # noqa: E402
from zemberek.core.turkish import SecondaryPos  # noqa: E402

TURKISH_LETTERS = frozenset("çğıöşü")
ENGLISH_SIGNIFICANCE_RATIO = 10.0
SPLITS_TO_PREDICT = ("validation", "test")


def normalize_token(token: str) -> str:
    """Case-fold a token and remove punctuation attached to either edge."""
    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end].casefold()


@lru_cache(maxsize=100_000)
def frequencies(token: str) -> tuple[float, float]:
    """Return cached English and Turkish word frequencies for a normalized token."""
    return word_frequency(token, "en"), word_frequency(token, "tr")


def build_named_entity_vocabulary(train_sentences: Sequence[Sentence]) -> set[str]:
    """Collect normalized NE tokens from the train split only."""
    vocabulary: set[str] = set()
    for sentence in train_sentences:
        for token, label in zip(sentence.tokens, sentence.labels, strict=True):
            if label == "NE":
                normalized = normalize_token(token)
                if normalized:
                    vocabulary.add(normalized)
    return vocabulary


def predict_wordfreq(token: str, named_entities: set[str]) -> str:
    """Apply the fixed NE, English-stem apostrophe, and frequency rules."""
    normalized = normalize_token(token)
    if normalized in named_entities:
        return "NE"
    if not normalized:
        return "TR"

    if "'" in normalized:
        prefix = normalized.split("'", maxsplit=1)[0]
        if prefix and frequencies(prefix)[0] > 0:
            return "MIXED"

    english_frequency, turkish_frequency = frequencies(normalized)
    return "EN" if english_frequency > turkish_frequency else "TR"


class TurkishAnalyzer:
    """Wrap Zemberek analysis and cache recognition checks by normalized token."""

    def __init__(self) -> None:
        self._morphology = TurkishMorphology.create_with_defaults()
        self._cache: dict[str, bool] = {}

    def recognizes_turkish(self, token: str) -> bool:
        normalized = normalize_token(token)
        if not normalized:
            return False
        if normalized not in self._cache:
            analyses = self._morphology.analyze(normalized)
            self._cache[normalized] = any(
                not analysis.is_unknown()
                and not analysis.is_runtime()
                and analysis.item.secondary_pos != SecondaryPos.ProperNoun
                for analysis in analyses
            )
        return self._cache[normalized]


def is_significantly_more_english(token: str) -> bool:
    """Use a fixed 10x frequency ratio as the English override for Zemberek."""
    normalized = normalize_token(token)
    if not normalized:
        return False
    english_frequency, turkish_frequency = frequencies(normalized)
    return (
        english_frequency > 0
        and english_frequency >= ENGLISH_SIGNIFICANCE_RATIO * turkish_frequency
    )


def predict_zemberek(
    token: str, named_entities: set[str], analyzer: TurkishAnalyzer
) -> str:
    """Apply the word-frequency rules, then prefer a non-name Turkish analysis."""
    wordfreq_label = predict_wordfreq(token, named_entities)
    if wordfreq_label in {"NE", "MIXED"}:
        return wordfreq_label
    if analyzer.recognizes_turkish(token) and not is_significantly_more_english(token):
        return "TR"
    return wordfreq_label


def predict_split(
    sentences: Sequence[Sentence],
    predictor: Callable[[str], str],
) -> dict[tuple[str, int], list[str]]:
    predictions: dict[tuple[str, int], list[str]] = {}
    for sentence in sentences:
        predictions[(sentence.doc_id, sentence.sent_id)] = [
            predictor(token) for token in sentence.tokens
        ]
    return predictions


def write_coverage_report(
    train_sentences: Sequence[Sentence],
    analyzer: TurkishAnalyzer,
    output_path: Path,
) -> None:
    """Write train-token Zemberek recognition rates by Turkish-letter usage."""
    counts = {
        "with": {"recognized": 0, "total": 0},
        "without": {"recognized": 0, "total": 0},
    }
    for sentence in train_sentences:
        for token, label in zip(sentence.tokens, sentence.labels, strict=True):
            if label != "TR":
                continue
            group = (
                "with"
                if any(letter in normalize_token(token) for letter in TURKISH_LETTERS)
                else "without"
            )
            counts[group]["total"] += 1
            if analyzer.recognizes_turkish(token):
                counts[group]["recognized"] += 1

    def rate(group: str) -> str:
        total = counts[group]["total"]
        if total == 0:
            return "n/a"
        recognized = counts[group]["recognized"]
        return f"{recognized / total:.1%} ({recognized}/{total})"

    report = (
        "# Zemberek coverage on the train split\n\n"
        "Recognition is measured over token occurrences labeled `TR` in the "
        "train split. A token is recognized if Zemberek returns at least one "
        "non-runtime, non-unknown analysis that is not solely a proper-noun "
        "analysis. No validation or test labels are used.\n\n"
        "| Turkish characters in token | Recognition rate |\n"
        "|---|---:|\n"
        f"| With ç, ğ, ı, ö, ş, or ü | {rate('with')} |\n"
        f"| Without those characters | {rate('without')} |\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> int:
    train_sentences = load_split_sentences("train")
    named_entities = build_named_entity_vocabulary(train_sentences)
    analyzer = TurkishAnalyzer()

    write_coverage_report(
        train_sentences,
        analyzer,
        REPO_ROOT / "results" / "zemberek_coverage.md",
    )

    for split in SPLITS_TO_PREDICT:
        sentences = load_split_sentences(split)
        wordfreq_predictions = predict_split(
            sentences,
            lambda token: predict_wordfreq(token, named_entities),
        )
        zemberek_predictions = predict_split(
            sentences,
            lambda token: predict_zemberek(token, named_entities, analyzer),
        )
        wordfreq_path = save_predictions("rules_wordfreq", split, wordfreq_predictions)
        zemberek_path = save_predictions("rules_zemberek", split, zemberek_predictions)
        print(f"wrote {wordfreq_path.relative_to(REPO_ROOT)}")
        print(f"wrote {zemberek_path.relative_to(REPO_ROOT)}")

    print("wrote results/zemberek_coverage.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
