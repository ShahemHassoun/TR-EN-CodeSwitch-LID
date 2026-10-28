#!/usr/bin/env python3
"""Tune and run CRF baselines for Turkish-English token-level LID.

Run from the repository root with Python 3.12:

    python src/baseline_crf.py

Regularisation is selected using only the train and validation splits. The
final models are scored on validation and write predictions for validation
and test; test labels are never used for scoring or model decisions.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import sklearn_crfsuite

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from lid_io import (  # noqa: E402
    Sentence,
    load_generated,
    load_split_sentences,
    save_predictions,
)
from turengmix.labels import LID_LABELS  # noqa: E402
from turengmix.scoring import token_metrics  # noqa: E402
from wordfreq import word_frequency  # noqa: E402

TURKISH_LETTERS = frozenset("çğıöşü")
REGULARIZATION_VALUES = (0.01, 0.05, 0.1, 0.5, 1.0)
GENERATED_SIZES = (2500, 5000, 10000)
SPLITS_TO_PREDICT = ("validation", "test")
MAX_ITERATIONS = 100

FeatureValue: TypeAlias = str | int | float | bool
TokenFeatures: TypeAlias = dict[str, FeatureValue]
SequenceFeatures: TypeAlias = list[TokenFeatures]


@lru_cache(maxsize=100_000)
def _frequencies(token: str) -> tuple[float, float]:
    """Return cached English and Turkish frequencies for a token."""
    return word_frequency(token, "en"), word_frequency(token, "tr")


def token_features(token: str) -> TokenFeatures:
    """Extract lexical, shape, suffix, and language-frequency features."""
    lower = token.lower()
    before_apostrophe, apostrophe, after_apostrophe = lower.partition("'")
    english_frequency, turkish_frequency = _frequencies(token)
    features: TokenFeatures = {
        "word": token,
        "word.lower": lower,
        "word.length": len(token),
        "word.isupper": token.isupper(),
        "word.istitle": token.istitle(),
        "word.isdigit": token.isdigit(),
        "word.has_apostrophe": bool(apostrophe),
        "word.apostrophe_prefix": before_apostrophe if apostrophe else "",
        "word.apostrophe_suffix": after_apostrophe if apostrophe else "",
        "word.has_turkish_char": any(char in lower for char in TURKISH_LETTERS),
        "word.frequency.en": english_frequency,
        "word.frequency.tr": turkish_frequency,
    }
    for size in range(1, 4):
        features[f"word.prefix_{size}"] = lower[:size]
        features[f"word.suffix_{size}"] = lower[-size:]
    return features


def sentence_features(tokens: list[str]) -> SequenceFeatures:
    """Create one feature dictionary per token, including adjacent tokens."""
    features: SequenceFeatures = []
    for index, token in enumerate(tokens):
        current = token_features(token)
        if index == 0:
            current["BOS"] = True
        else:
            current.update(
                {f"-1:{key}": value for key, value in token_features(tokens[index - 1]).items()}
            )
        if index == len(tokens) - 1:
            current["EOS"] = True
        else:
            current.update(
                {f"+1:{key}": value for key, value in token_features(tokens[index + 1]).items()}
            )
        features.append(current)
    return features


def sentences_features(sentences: list[Sentence]) -> list[SequenceFeatures]:
    """Extract token and neighboring-token features for a dataset."""
    return [sentence_features(sentence.tokens) for sentence in sentences]


def sentences_labels(sentences: list[Sentence]) -> list[list[str]]:
    """Return labels in sentence order, checking token-label alignment."""
    for sentence in sentences:
        if len(sentence.tokens) != len(sentence.labels):
            raise ValueError(
                f"sentence {(sentence.doc_id, sentence.sent_id)} has "
                f"{len(sentence.tokens)} tokens but {len(sentence.labels)} labels"
            )
    return [sentence.labels for sentence in sentences]


def new_crf(c1: float, c2: float) -> sklearn_crfsuite.CRF:
    """Construct a consistently configured L-BFGS CRF."""
    return sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True,
    )


def validation_macro_f1(
    model: sklearn_crfsuite.CRF,
    validation: list[Sentence],
    validation_features: list[SequenceFeatures],
) -> float:
    """Compute fixed-six-label macro-F1 on validation only."""
    predicted = model.predict(validation_features)
    gold_flat = [label for sentence in validation for label in sentence.labels]
    predicted_flat = [label for sequence in predicted for label in sequence]
    return token_metrics(gold_flat, predicted_flat, "lid")["macro_f1"]


def tune_regularization(
    train: list[Sentence],
    validation: list[Sentence],
    train_features: list[SequenceFeatures],
    validation_features: list[SequenceFeatures],
) -> tuple[float, float, float, sklearn_crfsuite.CRF]:
    """Search c1/c2 using train data for fitting and validation for selection."""
    train_labels = sentences_labels(train)
    best_c1: float | None = None
    best_c2: float | None = None
    best_score = float("-inf")
    best_model: sklearn_crfsuite.CRF | None = None

    for c1 in REGULARIZATION_VALUES:
        for c2 in REGULARIZATION_VALUES:
            model = new_crf(c1, c2)
            model.fit(train_features, train_labels)
            score = validation_macro_f1(model, validation, validation_features)
            print(f"tuning c1={c1:g} c2={c2:g}: validation macro-F1={score:.4f}")
            if score > best_score:
                best_c1, best_c2, best_score, best_model = c1, c2, score, model

    if best_c1 is None or best_c2 is None or best_model is None:
        raise RuntimeError("CRF regularization search did not produce a model")
    return best_c1, best_c2, best_score, best_model


def predict_split(
    model: sklearn_crfsuite.CRF, sentences: list[Sentence]
) -> dict[tuple[str, int], list[str]]:
    """Predict token labels, keyed as required by ``save_predictions``."""
    predictions: dict[tuple[str, int], list[str]] = {}
    for sentence, labels in zip(
        sentences, model.predict(sentences_features(sentences)), strict=True
    ):
        if len(labels) != len(sentence.tokens):
            raise ValueError(
                f"model returned {len(labels)} labels for "
                f"{len(sentence.tokens)} tokens in {(sentence.doc_id, sentence.sent_id)}"
            )
        predictions[(sentence.doc_id, sentence.sent_id)] = labels
    return predictions


def write_settings_report(
    c1: float,
    c2: float,
    tuning_score: float,
    run_scores: dict[str, float],
    unavailable_sizes: list[int],
    generated_file_missing: bool,
) -> Path:
    """Document tuning choices, validation results, and unavailable sizes."""
    report_path = REPO_ROOT / "results" / "crf_settings.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CRF settings and validation results",
        "",
        "Regularisation was selected by fitting on the TurEngMix train split and "
        "choosing the highest macro-F1 on the validation split. The test split "
        "was not used for tuning, evaluation, or model selection.",
        "",
        f"- **Search space for both `c1` and `c2`:** "
        f"`{', '.join(f'{value:g}' for value in REGULARIZATION_VALUES)}`",
        f"- **Selected `c1`:** `{c1:g}`",
        f"- **Selected `c2`:** `{c2:g}`",
        f"- **Best tuning validation macro-F1:** `{tuning_score:.4f}` "
        "(fixed six-label macro-F1)",
        f"- **CRF configuration:** L-BFGS, `max_iterations={MAX_ITERATIONS}`, "
        "`all_possible_transitions=True`",
        "",
        "## Model runs",
        "",
        "| Run | Generated sentences | Validation macro-F1 |",
        "|---|---:|---:|",
    ]
    for run_name, score in run_scores.items():
        generated_count = 0 if run_name == "crf_gen0" else int(run_name.removeprefix("crf_gen"))
        lines.append(f"| `{run_name}` | {generated_count:,} | {score:.4f} |")

    if generated_file_missing:
        lines.extend(
            [
                "",
                "The generated-sentence file was not available; only `crf_gen0` was run.",
            ]
        )
    elif unavailable_sizes:
        sizes_text = ", ".join(f"{size:,}" for size in unavailable_sizes)
        lines.extend(
            [
                "",
                f"The generated data currently contains too few sentences for the "
                f"requested sizes: {sizes_text}. Those runs were skipped.",
            ]
        )
    lines.extend(["", "Test predictions were saved but test labels were not evaluated.", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    train = load_split_sentences("train")
    validation = load_split_sentences("validation")
    train_features = sentences_features(train)
    validation_features = sentences_features(validation)

    c1, c2, tuning_score, tuned_model = tune_regularization(
        train, validation, train_features, validation_features
    )
    print(
        f"selected c1={c1:g} c2={c2:g} "
        f"(validation macro-F1={tuning_score:.4f})"
    )

    generated_file_missing = False
    try:
        available_generated = load_generated()
    except FileNotFoundError as error:
        print(f"notice: generated data unavailable ({error}); running crf_gen0 only")
        generated_file_missing = True
        available_generated = []

    unavailable_sizes = (
        []
        if generated_file_missing
        else [size for size in GENERATED_SIZES if len(available_generated) < size]
    )
    for size in unavailable_sizes:
        print(
            f"notice: {size:,} generated sentences requested, but only "
            f"{len(available_generated):,} are available; skipping crf_gen{size}"
        )

    test = load_split_sentences("test")
    run_scores: dict[str, float] = {}
    for size in (0, *GENERATED_SIZES):
        if size and (generated_file_missing or len(available_generated) < size):
            continue

        run_name = f"crf_gen{size}"
        if size == 0:
            model = tuned_model
            training_sentences = train
            training_features = train_features
        else:
            generated = load_generated(size)
            training_sentences = train + generated
            training_features = train_features + sentences_features(generated)
            model = new_crf(c1, c2)
            model.fit(training_features, sentences_labels(training_sentences))

        score = validation_macro_f1(model, validation, validation_features)
        run_scores[run_name] = score
        print(f"{run_name}: validation macro-F1={score:.4f}")
        for split, sentences in (("validation", validation), ("test", test)):
            path = save_predictions(run_name, split, predict_split(model, sentences))
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    report_path = write_settings_report(
        c1,
        c2,
        tuning_score,
        run_scores,
        unavailable_sizes,
        generated_file_missing,
    )
    print(f"wrote {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
