from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs336_alignment.utils import DEFAULT_MODEL, REPO_ROOT, generate_batch, load_generation_model, load_eval_template


def load_sst(data_path: str | Path) -> list[dict[str, str]]:
    with Path(data_path).open(encoding="utf-8-sig", newline="") as file:
        examples = list(csv.DictReader(file))
    if not examples or any(not example.get("prompts_final", "").strip() for example in examples):
        raise ValueError("SimpleSafetyTests data must contain non-empty prompts_final entries")
    return examples


def build_sst_prompts(examples: list[dict[str, str]], prompt_mode: str = "base") -> list[str]:
    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / "simple_safety_tests_zero_shot.prompt").read_text(encoding="utf-8")
    system_template = load_eval_template(prompt_mode)
    return [
        system_template.format(instruction=task_template.format(instruction=example["prompts_final"]), response="")
        for example in examples
    ]


def evaluate(
    output_dir: str | Path,
    model_name_or_path: str = DEFAULT_MODEL,
    data_path: str | Path = REPO_ROOT / "data/simple_safety_tests/simple_safety_tests.csv",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda:0",
    limit: int | None = None,
    seed: int = 0,
    dry_run: bool = False,
    prompt_mode: str = "base",
) -> dict[str, Any]:
    """Generate judge inputs; scripts/evaluate_safety.py computes the safety score."""
    if batch_size < 1 or max_new_tokens < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size, max_new_tokens and limit must be positive")
    examples = load_sst(data_path)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_sst_prompts(examples, prompt_mode)
    if dry_run:
        print(f"Loaded {len(examples)} examples. First full prompt:\n{prompts[0]}")
        return {"num_examples": len(examples), "dry_run": True, "prompt_mode": prompt_mode}

    model, tokenizer, model_device = load_generation_model(model_name_or_path, device, seed)
    run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": str(model_name_or_path), "prompt_mode": prompt_mode, "data_path": str(Path(data_path).resolve()),
        "num_examples": len(examples), "batch_size": batch_size,
        "max_new_tokens": max_new_tokens, "device": device, "seed": seed,
        "decoding": "greedy", "temperature": 0.0, "top_p": 1.0, "stop": "# Query:",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generation_seconds = 0.0
    num_generated = 0
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for start in range(0, len(examples), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            texts, elapsed = generate_batch(model, tokenizer, batch_prompts, max_new_tokens, model_device)
            generation_seconds += elapsed
            for example, prompt, text in zip(examples[start:start + batch_size], batch_prompts, texts):
                # Preserve prompts_final, which the supplied safety judge reads.
                record = {**example, "prompt": prompt, "output": text}
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_generated += 1
            file.flush()
            print(f"Generated {num_generated}/{len(examples)}", flush=True)

    summary = {
        "num_examples": num_generated, "generation_seconds": generation_seconds,
        "examples_per_second": num_generated / generation_seconds if generation_seconds > 0 else None,
        "predictions_path": str(predictions_path), "output_dir": str(run_dir),
        "judging_required": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Use predictions.jsonl as --input-path for scripts/evaluate_safety.py.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot SimpleSafetyTests generation for supplement sections 3.4 and 5.4")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/simple_safety_tests/simple_safety_tests.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-mode", choices=("base", "sft"), default="base")
    args = parser.parse_args()
    evaluate(
        model_name_or_path=args.model, data_path=args.data_path, output_dir=args.output_dir,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device,
        limit=args.limit, seed=args.seed, dry_run=args.dry_run, prompt_mode=args.prompt_mode,
    )
