#!/usr/bin/env python3
"""Run a prompted LLM baseline over the benchmark.

    python scripts/run_llm.py --task lid --model gpt-4o          --shots 0
    python scripts/run_llm.py --task ner --model qwen/qwen3-8b   --shots 3 \
        --provider openrouter

Outputs, under --out-dir (default results/predictions/):
    <run>.csv       benchmark rows plus a prediction column
    <run>.meta.json run configuration, token usage, parse statistics
    <run>.calls.jsonl  one record per API call: prompt, raw response,
                    finish_reason, latency, retry count
    <run>.ckpt.json checkpoint, removed on success
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix.data import (  # noqa: E402
    DEFAULT_BENCHMARK, load_benchmark, sentences, write_predictions,
)
from turengmix.prompting import format_tokens, parse_response, strip_reasoning  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_OUT = REPO_ROOT / "results" / "predictions"

PROVIDERS = {
    "openai": {"base_url": None, "env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "env": "OPENROUTER_API_KEY"},
}


def build_client(provider: str):
    from openai import OpenAI
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["env"])
    if not key:
        raise SystemExit(
            f"{cfg['env']} is not set.\n"
            f"Put it in a .env file or export it before running."
        )
    kwargs = {"api_key": key, "timeout": 120.0, "max_retries": 0}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def call_model(client, args, system_prompt: str, user_prompt: str) -> dict:
    from openai import (
        APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
    )
    transient = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

    extra: dict = {}
    if args.disable_reasoning:
        extra["reasoning"] = {"enabled": False}

    last: Exception | None = None
    for attempt in range(args.retries):
        started = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **({"extra_body": extra} if extra else {}),
            )
            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            return {
                "text": strip_reasoning(choice.message.content or ""),
                "finish_reason": choice.finish_reason,
                "latency_s": round(time.time() - started, 3),
                "attempts": attempt + 1,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except transient as e:
            last = e
            wait = random.uniform(0, min(2 ** attempt, 60))
            print(f"  {type(e).__name__}; retrying in {wait:.1f}s "
                  f"({attempt + 1}/{args.retries})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"exhausted {args.retries} retries: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("lid", "ner"))
    ap.add_argument("--model", required=True, help="e.g. gpt-4o, qwen/qwen3-8b")
    ap.add_argument("--shots", type=int, default=0, choices=(0, 3),
                    help="0 = zero-shot prompt, 3 = three-shot prompt")
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDERS))
    ap.add_argument("--prompt", type=Path,
                    help="override the prompt file chosen from --task/--shots")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-name", help="defaults to <model>_<task>_<shots>shot")
    ap.add_argument("--split", choices=("train", "validation", "test", "all"),
                    default="all", help="restrict to one split")
    ap.add_argument("--limit", type=int, help="stop after N sentences (smoke test)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls, for rate-limited endpoints")
    ap.add_argument("--disable-reasoning", action="store_true",
                    help="ask the provider to turn off reasoning tokens "
                         "(recommended for Qwen3 and other hybrid models)")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first prompt and exit without calling the API")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.prompt is None:
        suffix = "zero_shot" if args.shots == 0 else "few_shot"
        args.prompt = PROMPT_DIR / f"{args.task}_{suffix}.txt"
    if not args.prompt.exists():
        raise SystemExit(f"prompt file not found: {args.prompt}")
    system_prompt = args.prompt.read_text(encoding="utf-8")

    run = args.run_name or (
        f"{args.model.replace('/', '-')}_{args.task}_{args.shots}shot"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / f"{run}.csv"
    meta_path = args.out_dir / f"{run}.meta.json"
    calls_path = args.out_dir / f"{run}.calls.jsonl"
    ckpt_path = args.out_dir / f"{run}.ckpt.json"

    df = load_benchmark(args.benchmark)
    if args.split != "all":
        from turengmix.data import load_split
        keep = set(load_split()[args.split])
        df = df[df["doc_id"].isin(keep)].reset_index(drop=True)

    units = sentences(df, args.task)
    if args.limit:
        units = units[:args.limit]

    # Precomputed once: the tok_ids of each sentence, in file order.
    tok_ids: dict[tuple[str, int], list[int]] = {}
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        tok_ids[(str(doc_id), int(sent_id))] = g["tok_id"].tolist()

    if args.dry_run:
        print(f"prompt file : {args.prompt}")
        print(f"sentences   : {len(units)}")
        print(f"run name    : {run}\n")
        print("--- system ---")
        print(system_prompt[:600] + ("..." if len(system_prompt) > 600 else ""))
        print("\n--- first user message ---")
        print(format_tokens(units[0].tokens))
        return 0

    predictions: dict[tuple[str, int, int], str] = {}
    done: set[tuple[str, int]] = set()
    if ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        predictions = {tuple(json.loads(k)): v for k, v in saved["predictions"].items()}
        done = {tuple(x) for x in saved["completed_units"]}
        print(f"resuming from checkpoint: {len(done)}/{len(units)} sentences done")

    client = build_client(args.provider)
    stats = {"aligned": 0, "misaligned": 0, "truncated": 0, "unparsed_lines": 0,
             "duplicate_indices": 0, "failed_units": 0,
             "prompt_tokens": 0, "completion_tokens": 0}

    calls = open(calls_path, "a", encoding="utf-8")
    started_at = time.time()
    try:
        for n, unit in enumerate(units, start=1):
            uid = (unit.doc_id, unit.sent_id)
            if uid in done:
                continue

            user_prompt = format_tokens(unit.tokens)
            try:
                result = call_model(client, args, system_prompt, user_prompt)
            except Exception as e:
                # Leave this sentence's tokens absent from `predictions`; they
                # are written as UNK and counted.
                print(f"  FAILED {unit.doc_id} s{unit.sent_id}: {e}", file=sys.stderr)
                stats["failed_units"] += 1
                done.add(uid)
                continue

            parsed = parse_response(result["text"], args.task, len(unit))
            labels = parsed.ordered()
            for tok_id, label in zip(tok_ids[(unit.doc_id, unit.sent_id)], labels):
                predictions[(unit.doc_id, unit.sent_id, tok_id)] = label

            stats["aligned" if parsed.aligned else "misaligned"] += 1
            stats["unparsed_lines"] += len(parsed.unparsed_lines)
            stats["duplicate_indices"] += len(parsed.duplicate_indices)
            if result["finish_reason"] == "length":
                stats["truncated"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                if result.get(k):
                    stats[k] += result[k]

            calls.write(json.dumps({
                "doc_id": unit.doc_id, "sent_id": unit.sent_id,
                "n_tokens": len(unit), "input": user_prompt,
                "raw_output": result["text"],
                "finish_reason": result["finish_reason"],
                "latency_s": result["latency_s"], "attempts": result["attempts"],
                "aligned": parsed.aligned,
                "missing_indices": parsed.missing_indices,
                "extra_indices": parsed.extra_indices,
                "duplicate_indices": parsed.duplicate_indices,
                "n_unparsed_lines": len(parsed.unparsed_lines),
            }, ensure_ascii=False) + "\n")
            calls.flush()

            done.add(uid)
            if n % args.checkpoint_every == 0:
                _save_ckpt(ckpt_path, predictions, done)
                print(f"  [{n}/{len(units)}] checkpoint saved "
                      f"({stats['misaligned']} misaligned, {stats['truncated']} truncated)")
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        calls.close()

    column = f"pred_{args.task}"
    out = write_predictions(df, predictions, column, pred_path)
    n_unk = int((out[column] == "UNK").sum())

    meta = {
        "run_name": run,
        "task": args.task, "model": args.model, "provider": args.provider,
        "shots": args.shots, "prompt_file": str(args.prompt),
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "seed": args.seed, "reasoning_disabled": args.disable_reasoning,
        "split": args.split,
        "n_sentences": len(units), "n_tokens": len(out),
        "n_unparseable_tokens": n_unk,
        "unparseable_rate": n_unk / max(1, len(out)),
        "wall_clock_s": round(time.time() - started_at, 1),
        "parse_stats": stats,
        "benchmark": str(args.benchmark),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)

    print(f"\nwrote {pred_path}")
    print(f"      {meta_path}")
    print(f"      {calls_path}")
    print(f"\nsentences aligned {stats['aligned']}/{len(units)}   "
          f"unparseable tokens {n_unk} ({100 * n_unk / max(1, len(out)):.2f}%)")
    if stats["truncated"]:
        print(f"WARNING: {stats['truncated']} responses hit the token limit. "
              f"Raise --max-tokens or pass --disable-reasoning; these are "
              f"budget failures, not model errors.")
    print("\nScore it with:  python scripts/report.py")
    return 0


def _save_ckpt(path: Path, predictions: dict, done: set) -> None:
    path.write_text(json.dumps({
        "predictions": {json.dumps(list(k)): v for k, v in predictions.items()},
        "completed_units": [list(x) for x in done],
    }), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
