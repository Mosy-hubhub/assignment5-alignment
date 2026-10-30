from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from cs336_alignment.utils import DEFAULT_MODEL, REPO_ROOT, generate_batch, load_generation_model


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    numbers = re.findall(
        r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?",
        model_output,
    )
    return numbers[-1].replace(",", "") if numbers else None


def load_gsm8k(data_path: str | Path) -> list[dict[str, str]]:
    examples = []
    with Path(data_path).open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example.get("question"), str) or not isinstance(example.get("answer"), str):
                raise ValueError(f"Invalid GSM8K example at line {line_number}")
            if "####" not in example["answer"]:
                raise ValueError(f"Missing gold answer separator at line {line_number}")
            gold = example["answer"].rsplit("####", 1)[1].strip().replace(",", "")
            if not Decimal(gold).is_finite():
                raise ValueError(f"Invalid numeric gold answer at line {line_number}")
            examples.append({**example, "gold_answer": gold})
    if not examples:
        raise ValueError(f"No GSM8K examples found in {data_path}")
    return examples


def build_gsm8k_prompts(examples: list[dict[str, str]]) -> list[str]:
    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / "gsm8k_zero_shot.prompt").read_text(encoding="utf-8")
    system_template = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding="utf-8")
    return [
        system_template.format(instruction=task_template.format(question=example["question"]))
        for example in examples
    ]


def evaluate(
    model_name_or_path: str = DEFAULT_MODEL,
    data_path: str | Path = REPO_ROOT / "data/gsm8k/test.jsonl",
    output_dir: str | Path = REPO_ROOT / "results/gsm8k_baseline",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda:0",
    limit: int | None = None,
    seed: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    if batch_size < 1 or max_new_tokens < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size, max_new_tokens and limit must be positive")
    examples = load_gsm8k(data_path)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_gsm8k_prompts(examples)
    if dry_run:
        print(f"Loaded {len(examples)} examples. First full prompt:\n{prompts[0]}")
        return {"num_examples": len(examples), "dry_run": True}

    model, tokenizer, model_device = load_generation_model(model_name_or_path, device, seed)
    run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": model_name_or_path, "data_path": str(Path(data_path).resolve()),
        "num_examples": len(examples), "batch_size": batch_size,
        "max_new_tokens": max_new_tokens, "device": device, "seed": seed,
        "decoding": "greedy", "temperature": 0.0, "top_p": 1.0, "stop": "# Query:",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    generation_seconds = 0.0
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for start in range(0, len(examples), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            texts, elapsed = generate_batch(model, tokenizer, batch_prompts, max_new_tokens, model_device)
            generation_seconds += elapsed
            for example, prompt, text in zip(examples[start:start + batch_size], batch_prompts, texts):
                prediction = parse_gsm8k_response(text)
                correct = prediction is not None and Decimal(prediction) == Decimal(example["gold_answer"])
                record = {
                    **example, "prompt": prompt, "model_output": text,
                    "prediction": prediction, "parse_failed": prediction is None, "correct": correct,
                }
                records.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"Evaluated {len(records)}/{len(examples)}", flush=True)

    summary = {
        "num_examples": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "num_parse_failures": sum(record["parse_failed"] for record in records),
        "generation_seconds": generation_seconds,
        "examples_per_second": len(records) / generation_seconds if generation_seconds > 0 else None,
        "output_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    incorrect = [record for record in records if not record["correct"]]
    samples = random.Random(seed).sample(incorrect, min(10, len(incorrect)))
    (run_dir / "incorrect_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot GSM8K evaluation for supplement section 3.2")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/gsm8k_baseline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate(
        model_name_or_path=args.model, data_path=args.data_path, output_dir=args.output_dir,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device,
        limit=args.limit, seed=args.seed, dry_run=args.dry_run,
    )
