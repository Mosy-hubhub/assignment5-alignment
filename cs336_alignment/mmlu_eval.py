from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from cs336_alignment.utils import DEFAULT_MODEL, REPO_ROOT, generate_batch, load_generation_model

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.

    Note:
        mmlu_example is retained for compatibility with the handout and test interface.
        This parser uses only the output format, not the question details or gold answer.
    """
    match = re.search(
        r"\bThe\s+correct\s+answer\s+is\s+([A-D])\b",
        model_output,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).upper()


def load_mmlu(data_dir: str | Path) -> list[dict[str, Any]]:
    """Read headerless MMLU CSV files: question, A, B, C, D, gold letter."""
    examples = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        subject = path.stem.rsplit("_", 1)[0].replace("_", " ")
        with path.open(encoding="utf-8-sig", newline="") as file:
            for line_number, row in enumerate(csv.reader(file), start=1):
                if not row:
                    continue
                if len(row) != 6 or row[5].strip() not in ("A", "B", "C", "D"):
                    raise ValueError(f"Invalid MMLU row: {path}:{line_number}")
                examples.append({
                    "subject": subject,
                    "question": row[0],
                    "options": row[1:5],
                    "answer": row[5].strip(),
                })
    if not examples:
        raise ValueError(f"No MMLU examples found in {data_dir}")
    return examples


def build_mmlu_prompts(examples: list[dict[str, Any]]) -> list[str]:
    """Fill the task template, then insert it into the shared system template."""
    prompt_dir = REPO_ROOT / "cs336_alignment" / "prompts_safety"
    task_template = (prompt_dir / "mmlu_zero_shot.prompt").read_text(encoding="utf-8")
    system_template = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding="utf-8")
    return [
        system_template.format(instruction=task_template.format(
            subject=example["subject"], question=example["question"], options=example["options"]
        ))
        for example in examples
    ]


def evaluate(
    model_name_or_path: str = DEFAULT_MODEL,
    data_dir: str | Path = REPO_ROOT / "data" / "mmlu" / "test",
    output_dir: str | Path = REPO_ROOT / "results" / "mmlu_baseline",
    batch_size: int = 1,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    limit: int | None = None,
    seed: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate a causal LM on MMLU with greedy decoding and save predictions."""
    if batch_size < 1 or max_new_tokens < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size, max_new_tokens and limit must be positive")

    examples = load_mmlu(data_dir)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_mmlu_prompts(examples)
    if dry_run:
        print(f"Loaded {len(examples)} examples. First full prompt:\n{prompts[0]}")
        return {"num_examples": len(examples), "dry_run": True}

    model, tokenizer, model_device = load_generation_model(model_name_or_path, device, seed)

    run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": model_name_or_path, "data_dir": str(Path(data_dir).resolve()),
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
                model_output = text
                prediction = parse_mmlu_response(example, model_output)
                record = {
                    **example, "prompt": prompt, "model_output": model_output,
                    "prediction": prediction, "parse_failed": prediction is None,
                    "correct": prediction == example["answer"],
                }
                records.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"Evaluated {len(records)}/{len(examples)}", flush=True)

    # Failed parses count as incorrect; do not exclude them from the denominator.
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
    parser = argparse.ArgumentParser(description="Zero-shot MMLU evaluation for supplement section 3.1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/mmlu/test")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/mmlu_baseline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate(
        model_name_or_path=args.model, data_dir=args.data_dir, output_dir=args.output_dir,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device,
        limit=args.limit, seed=args.seed, dry_run=args.dry_run,
    )
