#!/usr/bin/env python3
"""Fine-tune an encoder for token-level LID, optionally adding our generated data.

Adapted from TurEngMix's scripts/train_encoder.py (MIT licence). The training
recipe is unchanged, so our "real data only" runs reproduce the paper:
learning rate 2e-5, weight decay 0.01, warm-up 6%, batch 16, at most 10 epochs,
early stopping after 3 epochs without improvement, best epoch chosen by
macro-F1 on the REAL TurEngMix validation split. Labels go on the first
sub-word of each word; other sub-words are ignored in the loss.

What we added:
  --extra-n N   add the first N sentences of data/generated/generated.jsonl
                to the TurEngMix training split (0 = real data only)
  After training, the script writes predictions for the validation and test
  splits and deletes the model files, so a Colab disk never fills up.

Examples (run from the repo root):
  python src/train_encoder.py --model dbmdz/bert-base-turkish-cased --name berturk --extra-n 0 --fp16
  python src/train_encoder.py --model dbmdz/bert-base-turkish-cased --name berturk --extra-n 2500 --fp16

Output, for every seed:
  results/predictions/<name>_gen<N>_seed<S>_validation.csv
  results/predictions/<name>_gen<N>_seed<S>_test.csv
  results/runs/<name>_gen<N>_seed<S>.json      (settings + training history)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid_io import (  # noqa: E402
    REPO_ROOT, load_generated, load_split_sentences, save_predictions,
)
from turengmix import scoring  # noqa: E402
from turengmix.labels import LID_LABELS  # noqa: E402

L2I = {lab: i for i, lab in enumerate(LID_LABELS)}
I2L = {i: lab for lab, i in L2I.items()}


def tokenize_and_align(batch, tokenizer, max_length):
    """Identical to TurEngMix: label the first sub-word, -100 elsewhere."""
    enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True,
                    max_length=max_length)
    all_labels = []
    for i, words in enumerate(batch["labels"]):
        previous, ids = None, []
        for w in enc.word_ids(batch_index=i):
            if w is None:
                ids.append(-100)
            elif w != previous:
                ids.append(L2I[words[w]])
            else:
                ids.append(-100)
            previous = w
        all_labels.append(ids)
    enc["labels"] = all_labels
    return enc


def compute_metrics(eval_pred):
    logits, gold = eval_pred
    preds = np.argmax(logits, axis=2)
    t, p = [], []
    for p_row, g_row in zip(preds, gold):
        for pi, gi in zip(p_row, g_row):
            if gi != -100:
                t.append(I2L[int(gi)])
                p.append(I2L[int(pi)])
    m = scoring.token_metrics(t, p, "lid")
    return {"f1": m["macro_f1"], "accuracy": m["accuracy"]}


def predict(model, tokenizer, sentences, max_length, batch_size=32):
    """Predict one label per word, read off its first sub-word."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            enc = tokenizer([s.tokens for s in batch], is_split_into_words=True,
                            truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            best = torch.argmax(logits, dim=-1).cpu().numpy()
            for i, s in enumerate(batch):
                first = {}
                for pos, w in enumerate(enc.word_ids(batch_index=i)):
                    if w is not None and w not in first:
                        first[w] = pos
                # A word cut off by max_length gets "UNK" and counts as an error.
                out[(s.doc_id, s.sent_id)] = [
                    I2L[int(best[i, first[w]])] if w in first else "UNK"
                    for w in range(len(s.tokens))
                ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="dbmdz/bert-base-turkish-cased | FacebookAI/xlm-roberta-base")
    ap.add_argument("--name", required=True, help="short model name: berturk | xlmr")
    ap.add_argument("--extra-n", type=int, required=True, dest="extra_n",
                    help="generated sentences to add: 0, 2500, 5000 or 10000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--learning-rate", type=float, default=2e-5, dest="learning_rate")
    ap.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.06, dest="warmup_ratio")
    ap.add_argument("--max-length", type=int, default=512, dest="max_length")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--keep-model", action="store_true", dest="keep_model",
                    help="do not delete the trained model afterwards")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, EarlyStoppingCallback,
        Trainer, TrainingArguments, set_seed,
    )

    real_train = load_split_sentences("train")
    extra = load_generated(args.extra_n) if args.extra_n > 0 else []
    validation = load_split_sentences("validation")
    test = load_split_sentences("test")
    train = real_train + extra
    print(f"train: {len(real_train)} real + {len(extra)} generated sentences | "
          f"validation: {len(validation)} | test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    def to_ds(sents):
        ds = Dataset.from_list([{"tokens": s.tokens, "labels": s.labels} for s in sents])
        return ds.map(lambda b: tokenize_and_align(b, tokenizer, args.max_length),
                      batched=True, remove_columns=["tokens", "labels"])

    train_ds, val_ds = to_ds(train), to_ds(validation)
    run_dir = REPO_ROOT / "results" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run = f"{args.name}_gen{args.extra_n}_seed{seed}"
        out_dir = REPO_ROOT / "models" / run
        print(f"\n=== {run} ===")
        started = time.time()
        set_seed(seed)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(LID_LABELS), id2label=I2L, label2id=L2I)
        targs = TrainingArguments(
            output_dir=str(out_dir),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            load_best_model_at_end=True, metric_for_best_model="f1",
            greater_is_better=True, save_total_limit=2,
            seed=seed, data_seed=seed, fp16=args.fp16, report_to="none",
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
            if args.patience else [],
        )
        trainer.train()

        for split, sents in (("validation", validation), ("test", test)):
            path = save_predictions(run, split, predict(trainer.model, tokenizer,
                                                        sents, args.max_length))
            print(f"wrote {path.relative_to(REPO_ROOT)}")

        record = {
            "run": run, "model": args.model, "seed": seed,
            "n_real_train_sentences": len(real_train),
            "n_generated_sentences": len(extra),
            "hyperparameters": {k: v for k, v in vars(args).items() if k != "seeds"},
            "best_validation_macro_f1": trainer.state.best_metric,
            "epochs_run": trainer.state.epoch,
            "validation_history": [h for h in trainer.state.log_history if "eval_f1" in h],
            "minutes": round((time.time() - started) / 60, 1),
        }
        (run_dir / f"{run}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"best validation macro-F1 {trainer.state.best_metric:.4f}  "
              f"({record['minutes']} min)")

        del trainer, model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_model:
            shutil.rmtree(out_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
