"""DPO training on the single-turn subset of Anthropic HH (supplement 6.4)."""

import argparse
import gzip
import json
import math
import re
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cs336_alignment.dpo import (
    compute_batch_dpo_loss, compute_per_instance_dpo_loss,
    get_batch_response_log_probs, tokenize_preference_batch,
)
from cs336_alignment.rl_utils import save_checkpoint, save_train_config


REPO_ROOT = Path(__file__).resolve().parents[1]
HH_FILES = (
    "harmless-base.jsonl.gz", "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz", "helpful-rejection-sampled.jsonl.gz",
)


def load_hh(data_dir):
    """Extract single-turn preference pairs and retain their source filenames."""
    examples = []
    stats = {"raw_examples": 0, "skipped_multiturn": 0, "skipped_invalid_pairs": 0}
    for filename in HH_FILES:
        with gzip.open(Path(data_dir) / filename, "rt", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["raw_examples"] += 1
                conversations = [row.get("chosen"), row.get("rejected")]
                if not all(isinstance(text, str) for text in conversations):
                    raise ValueError(f"Expected chosen/rejected strings: {filename}:{line_number}")

                # chosen/rejected 都是完整对话；任意一边有多个 Human 回合就跳过。
                if any(len(re.findall(r"(?:\A|\n\n)Human:", text.lstrip())) != 1
                       for text in conversations):
                    stats["skipped_multiturn"] += 1
                    continue
                matches = [re.fullmatch(r"\s*Human:(.*?)\n\nAssistant:(.*)", text, re.DOTALL)
                           for text in conversations]
                if not all(matches):
                    stats["skipped_invalid_pairs"] += 1
                    continue
                chosen_prompt, chosen = (part.strip() for part in matches[0].groups())
                rejected_prompt, rejected = (part.strip() for part in matches[1].groups())
                if chosen_prompt != rejected_prompt or not all((chosen_prompt, chosen, rejected)):
                    stats["skipped_invalid_pairs"] += 1
                    continue
                examples.append({
                    "prompt": chosen_prompt, "response_chosen": chosen,
                    "response_rejected": rejected, "source": filename,
                })
    if not examples:
        raise ValueError("HH contains no usable single-turn preference pairs")
    stats["single_turn_examples"] = len(examples)
    return examples, stats


def filter_long_examples(examples, tokenizer, max_seq_length):
    """Drop oversized pairs without truncating either preference response."""
    template = (Path(__file__).parent / "prompts_safety/alpaca_sft.prompt").read_text(
        encoding="utf-8"
    ).rstrip()
    kept = []
    for example in examples:
        # 与核心 loss 使用相同的分词边界，并计入回答末尾的一个 EOS。
        prefix = template.format(instruction=example["prompt"], response="")
        prompt_length = len(tokenizer.encode(prefix, add_special_tokens=True))
        response_length = max(
            len(tokenizer.encode(example[key], add_special_tokens=False))
            for key in ("response_chosen", "response_rejected")
        )
        if prompt_length + response_length + 1 <= max_seq_length:
            kept.append(example)
    return kept


def evaluate_accuracy(model, tokenizer, examples, batch_size=1):
    """Measure chosen > rejected conditional log-probability, as in handout 6.4."""
    was_training = model.training
    if batch_size <= 0 or not examples:
        raise ValueError("Validation batch size and dataset size must be positive")
    correct = 0
    margin_sum = 0.0
    model.eval()
    try:
        # 验证只比较当前模型的两个回答，不需要参考模型，也不建立计算图。
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                rows = examples[start:start + batch_size]
                batch = tokenize_preference_batch(
                    tokenizer, [row["prompt"] for row in rows],
                    [row["response_chosen"] for row in rows],
                    [row["response_rejected"] for row in rows],
                )
                # chosen/rejected 合并前向，再按实际偏好对数量计算准确率。
                scores = get_batch_response_log_probs(model, batch)
                margins = scores[:len(rows)] - scores[len(rows):]
                correct += (margins > 0).sum().item()
                margin_sum += margins.sum().item()
    finally:
        model.train(was_training)
    return {"val_accuracy": correct / len(examples), "val_log_prob_margin": margin_sum / len(examples)}


def train(args):
    """Train preference microbatches with RMSprop and sample-weighted accumulation."""
    # 检查参数及输出目录，避免覆盖之前的实验。
    if min(args.batch_size, args.epochs, args.gradient_accumulation_steps, args.eval_every,
           args.val_size, args.max_seq_length) <= 0:
        raise ValueError("Epochs, accumulation, evaluation interval and dataset sizes must be positive")
    if args.learning_rate <= 0 or args.beta <= 0 or args.max_grad_norm <= 0:
        raise ValueError("Learning rate, beta and max_grad_norm must be positive")
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    set_seed(args.seed)

    # 读取四份 HH 数据并过滤过长样本，固定取前 val_size 条作为验证集。
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError("DPO needs an EOS token")
    examples, data_stats = load_hh(args.data_dir)
    filtered = filter_long_examples(examples, tokenizer, args.max_seq_length)
    data_stats["skipped_too_long"] = len(examples) - len(filtered)
    if len(filtered) <= args.val_size:
        raise ValueError("Not enough preference pairs after filtering to leave a training set")
    val_examples = filtered[:args.val_size]
    train_examples = filtered[args.val_size:]
    # DataLoader 按字段收集字符串；batch_size 表示偏好对数，实际有 2B 条序列。
    data_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)

    # 从同一个 SFT checkpoint 加载两个模型；参考模型固定，不由优化器更新。
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_implementation
    ).to(torch.device(args.device))
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_implementation
    ).to(torch.device(args.ref_device))
    model.config.use_cache = False
    model_ref.config.use_cache = False
    model_ref.requires_grad_(False)
    model_ref.eval()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    # DPO 使用讲义建议的固定学习率；step 表示一次参数更新，而非一条偏好对。
    total_steps = args.epochs * math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    config = save_train_config(
        args, total_steps=total_steps, train_examples=len(train_examples),
        val_examples=len(val_examples), effective_batch_size=args.batch_size * args.gradient_accumulation_steps,
        resolved_eval_batch_size=eval_batch_size,
        optimizer="RMSprop", **data_stats,
    )
    # 保存本次实际使用的验证样本，便于以后复核最佳 checkpoint 的指标。
    (output_dir / "validation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in val_examples), encoding="utf-8"
    )
    global_step = 0
    best_step = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    # 训练与验证指标全部交给 W&B，不另外打印到终端。
    with wandb.init(project=args.wandb_project, name=args.wandb_name, config=config,
                    dir=str(output_dir.resolve()), mode=args.wandb_mode) as run:
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        metrics = evaluate_accuracy(model, tokenizer, val_examples, eval_batch_size)
        best_accuracy = metrics["val_accuracy"]
        run.log({"step": 0, **metrics})
        # 初始模型也参与最佳模型比较，避免训练退步时丢掉更好的起点。
        save_checkpoint(model, tokenizer, output_dir / "best", 0, metrics)

        for epoch in range(args.epochs):
            for batch_index, batch in enumerate(data_loader):
                # 每组按实际偏好对数平均；最后不足 accumulation_steps 的组也会更新。
                group_start = batch_index // args.gradient_accumulation_steps * args.gradient_accumulation_steps
                group_end = min(group_start + args.gradient_accumulation_steps, len(data_loader))
                if batch_index == group_start:
                    group_loss = 0.0
                    group_examples = min(group_end * args.batch_size, len(train_examples)) - group_start * args.batch_size
                microbatch_size = len(batch["prompt"])
                if args.batch_size == 1:
                    # 保留单样本调用接口；其内部也复用批量分词和 loss 计算。
                    loss = compute_per_instance_dpo_loss(
                        model, model_ref, tokenizer, args.beta, batch["prompt"][0],
                        batch["response_chosen"][0], batch["response_rejected"][0],
                    )
                else:
                    loss = compute_batch_dpo_loss(
                        model, model_ref, tokenizer, args.beta, batch["prompt"],
                        batch["response_chosen"], batch["response_rejected"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("DPO loss is not finite")
                # microbatch loss 是均值；按实际偏好对数加权，避免尾批被过度放大。
                weighted_loss = loss * (microbatch_size / group_examples)
                weighted_loss.backward()
                group_loss += weighted_loss.detach().item()
                del loss, weighted_loss
                if batch_index + 1 != group_end:
                    continue

                # 累积结束后统一裁剪、更新参数，并清空梯度。
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                run.log({"step": global_step, "epoch": epoch + 1, "train_loss": group_loss,
                         "grad_norm": grad_norm.item(), "learning_rate": args.learning_rate,
                         "num_preference_pairs": group_examples})

                # 按验证准确率保存最佳模型，同时保证最后一次更新后也进行验证。
                if global_step % args.eval_every == 0 or global_step == total_steps:
                    metrics = evaluate_accuracy(model, tokenizer, val_examples, eval_batch_size)
                    if metrics["val_accuracy"] > best_accuracy:
                        best_accuracy = metrics["val_accuracy"]
                        best_step = global_step
                        save_checkpoint(model, tokenizer, output_dir / "best", global_step, metrics)
                    run.log({"step": global_step, "best_val_accuracy": best_accuracy, **metrics})
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        # last 与 best 分开保存；这些 checkpoint 可评估，但不包含断点续训状态。
        save_checkpoint(model, tokenizer, output_dir / "last", global_step, metrics)
        summary = {
            "optimizer_steps": global_step, "best_step": best_step,
            "best_val_accuracy": best_accuracy, **metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "best_checkpoint": str((output_dir / "best").resolve()),
            "last_checkpoint": str((output_dir / "last").resolve()),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run.summary.update(summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-turn HH DPO training for supplement section 6.4")
    parser.add_argument("--model", required=True, help="SFT checkpoint, used for both initial models")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/hh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ref-device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Preference pairs per training microbatch")
    parser.add_argument("--eval-batch-size", type=int, help="Validation preference pairs per batch; defaults to batch-size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="cs336-dpo")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    train(args)
