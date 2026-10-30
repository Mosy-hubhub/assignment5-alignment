from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs336_alignment.utils import DEFAULT_MODEL, REPO_ROOT, generate_batch, load_generation_model


def load_alpaca(data_path: str | Path) -> list[dict[str, str]]:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("AlpacaEval data must be a non-empty JSON array")
    examples = []
    for row in rows:
        if not isinstance(row.get("instruction"), str) or not isinstance(row.get("dataset"), str):
            raise ValueError("Each AlpacaEval example needs instruction and dataset strings")
        # Deliberately exclude the reference output and reference generator.
        examples.append({"instruction": row["instruction"], "dataset": row["dataset"]})
    return examples


def build_alpaca_prompts(examples: list[dict[str, str]]) -> list[str]:
    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / "alpaca_eval_zero_shot.prompt").read_text(encoding="utf-8")
    system_template = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding="utf-8")
    return [
        system_template.format(instruction=task_template.format(instruction=example["instruction"]))
        for example in examples
    ]


def evaluate(
    model_name_or_path: str = DEFAULT_MODEL,
    data_path: str | Path = REPO_ROOT / "data/alpaca_eval/alpaca_eval_gpt4_turbo.json",
    output_dir: str | Path = REPO_ROOT / "results/alpaca_baseline",
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    device: str = "cuda:0",
    limit: int | None = None,
    seed: int = 0,
    dry_run: bool = False,
    generator: str = "llama-3.1-8b-base",
) -> dict[str, Any]:
    """Generate the JSON array expected by AlpacaEval; judging is a separate step."""
    if batch_size < 1 or max_new_tokens < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size, max_new_tokens and limit must be positive")
    examples = load_alpaca(data_path)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_alpaca_prompts(examples)
    if dry_run:
        print(f"Loaded {len(examples)} examples. First full prompt:\n{prompts[0]}")
        return {"num_examples": len(examples), "dry_run": True}

    model, tokenizer, model_device = load_generation_model(model_name_or_path, device, seed)
    run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": model_name_or_path, "generator": generator, "data_path": str(Path(data_path).resolve()),
        "num_examples": len(examples), "batch_size": batch_size,
        "max_new_tokens": max_new_tokens, "device": device, "seed": seed,
        "decoding": "greedy", "temperature": 0.0, "top_p": 1.0, "stop": "# Query:",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    outputs = []
    generation_seconds = 0.0
    # Keep per-batch records too, so an interrupted run retains generated answers.
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as file:
        for start in range(0, len(examples), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            texts, elapsed = generate_batch(model, tokenizer, batch_prompts, max_new_tokens, model_device)
            generation_seconds += elapsed
            for example, prompt, text in zip(examples[start:start + batch_size], batch_prompts, texts):
                record = {**example, "output": text, "generator": generator}
                outputs.append(record)
                file.write(json.dumps({**record, "prompt": prompt}, ensure_ascii=False) + "\n")
            file.flush()
            print(f"Generated {len(outputs)}/{len(examples)}", flush=True)

    # The judge needs a JSON array, not JSONL. Preserve the original instruction/dataset.
    predictions_path = run_dir / "predictions.json"
    predictions_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "num_examples": len(outputs), "generation_seconds": generation_seconds,
        "examples_per_second": len(outputs) / generation_seconds if generation_seconds > 0 else None,
        "predictions_path": str(predictions_path), "output_dir": str(run_dir),
        "judging_required": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Use predictions.json as --model_outputs for the AlpacaEval judge.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot AlpacaEval generation for supplement section 3.3")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/alpaca_baseline")
    parser.add_argument("--generator", default="llama-3.1-8b-base")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate(
        model_name_or_path=args.model, data_path=args.data_path, output_dir=args.output_dir,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device,
        limit=args.limit, seed=args.seed, dry_run=args.dry_run, generator=args.generator,
    )
