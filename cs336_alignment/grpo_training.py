"""GSM8K rollout generation, policy updates, validation and checkpoint saving."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cs336_alignment.rl_utils import save_checkpoint, save_train_config
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import (
    compute_rollout_rewards, get_response_log_probs, grpo_train_step, tokenize_prompt_and_output,
)
from cs336_alignment.gsm8k_eval import load_gsm8k
from cs336_alignment.vllm_utils import VLLMServer


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_reward_fn(args, tokenizer):
    # 保留 EOS 供策略概率计算，但评分前移除结尾 EOS，避免影响数学答案解析。
    base_reward_fn = question_only_reward_fn if args.prompt == "question_only" else r1_zero_reward_fn

    def reward_fn(response, ground_truth):
        if tokenizer.eos_token and response.endswith(tokenizer.eos_token):
            response = response[:-len(tokenizer.eos_token)]
        return base_reward_fn(response, ground_truth)

    return reward_fn


def generate_rollouts(model, tokenizer, prompts, args, server=None, group_size=1, seed=0):
    """Return prompt-major responses: all G responses to prompt 0, then prompt 1."""
    stop = None if args.prompt == "question_only" else "</answer>"
    started = time.perf_counter()
    responses = []
    truncated = 0

    if server is not None:
        # 直接使用 n=G；不要对相同 prompt 分别发送 n=1、相同 seed 的请求，
        # 否则同组回答可能完全相同，失去组内奖励差异。
        sampling_params = {
            "temperature": args.temperature, "max_tokens": args.max_new_tokens,
            "n": group_size, "seed": seed,
        }
        if stop is not None:
            sampling_params.update(stop=[stop], include_stop_str_in_output=True)
        completions = server.generate_completions(
            prompts, sampling_params, batch_size=args.generation_batch_size
        )
        for completion in completions:
            text = completion.text
            # vLLM 的 text 一般不包含结束 EOS；只有实际因 EOS 结束时才补回。
            if completion.finish_reason == "stop" and (stop is None or stop not in text):
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
            responses.append(text or tokenizer.eos_token)
            truncated += completion.finish_reason == "length"
    else:
        # HF 后端使用同一个策略实例，不需要同步权重；适合 CPU 和小模型检查。
        repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        try:
            # 隔离生成随机数，避免验证次数改变后影响后续训练时的随机状态。
            with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
                torch.manual_seed(seed)
                for start in range(0, len(repeated_prompts), args.generation_batch_size):
                    inputs = tokenizer(
                        repeated_prompts[start:start + args.generation_batch_size],
                        return_tensors="pt", padding=True, add_special_tokens=False,
                        return_token_type_ids=False,
                    ).to(device)
                    kwargs = {
                        "do_sample": args.temperature > 0, "max_new_tokens": args.max_new_tokens,
                        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    if args.temperature > 0:
                        kwargs.update(temperature=args.temperature, top_p=1.0, top_k=0)
                    if stop is not None:
                        kwargs.update(stop_strings=[stop], tokenizer=tokenizer)
                    generated = model.generate(**inputs, **kwargs)
                    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                    for row in new_tokens.tolist():
                        # batch 内较早结束的回答后面会补 PAD，不把这些补位算入回答。
                        if tokenizer.eos_token_id in row:
                            row = row[:row.index(tokenizer.eos_token_id) + 1]
                        text = tokenizer.decode(row, skip_special_tokens=False)
                        if stop is not None and stop in text:
                            text = text[:text.index(stop) + len(stop)]
                        responses.append(text or tokenizer.eos_token)
                        truncated += len(row) == args.max_new_tokens and row[-1] != tokenizer.eos_token_id and (stop is None or stop not in text)
        finally:
            model.train(was_training)

    if len(responses) != len(prompts) * group_size:
        raise ValueError("Generation backend returned an unexpected number of responses")
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]
    return responses, {
        "generation_seconds": time.perf_counter() - started,
        "mean_response_length": sum(lengths) / len(lengths),
        "truncation_fraction": truncated / len(responses),
    }


def compute_old_log_probs(model, tokenizer, prompts, responses, batch_size):
    """Freeze rollout-policy probabilities before any update reuses this rollout."""
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(model.parameters()).device
    old_log_probs = []
    # 分批评分并立即存回 CPU，避免为整批 rollout 保留 logits 或计算图。
    with torch.no_grad():
        for start in range(0, len(responses), batch_size):
            scores = get_response_log_probs(
                model, tokenized["input_ids"][start:start + batch_size].to(device),
                tokenized["labels"][start:start + batch_size].to(device),
            )
            old_log_probs.append(scores["log_probs"].detach().cpu())
    return torch.cat(old_log_probs, dim=0)


def log_rollouts(run, output_dir, tag, step, prompts, responses, ground_truths, reward_fn, limit):
    # 只记录少量文本样本，便于观察训练前后格式与推理过程是否发生变化。
    records = []
    for prompt, response, gold in list(zip(prompts, responses, ground_truths))[:limit]:
        records.append({"prompt": prompt, "response": response, "ground_truth": gold,
                        **reward_fn(response, gold)})
    sample_dir = Path(output_dir) / "rollouts"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / f"{tag}_{step:06d}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    columns = ["prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
    run.log({"step": step, f"{tag}_rollouts": wandb.Table(
        columns=columns, data=[[row[column] for column in columns] for row in records]
    )})


def evaluate_policy(model, tokenizer, examples, template, reward_fn, args, server=None):
    # 验证时每题采样一个回答，沿用训练时的温度、长度上限和评分函数。
    prompts = [template.format(question=example["question"]) for example in examples]
    golds = [example["gold_answer"] for example in examples]
    responses, generation_metrics = generate_rollouts(
        model, tokenizer, prompts, args, server, seed=args.seed + 1_000_000
    )
    _, reward_metrics = compute_rollout_rewards(reward_fn, responses, golds)
    metrics = {f"val_{key}": value for key, value in {**reward_metrics, **generation_metrics}.items()}
    return metrics, prompts, responses, golds


def train(args):
    # 检查批次的单位：rollout/train batch size 都按回答数计，必须保留完整组。
    if min(args.num_rollout_steps, args.rollout_batch_size, args.train_batch_size, args.group_size,
           args.gradient_accumulation_steps, args.generation_batch_size, args.log_prob_batch_size,
           args.max_new_tokens, args.eval_every, args.epochs_per_rollout, args.log_rollouts_every,
           args.num_rollouts_to_log, args.n_train_examples, args.n_val_examples) <= 0:
        raise ValueError("Step counts, batch sizes, group size and intervals must be positive")
    if args.rollout_batch_size % args.train_batch_size or args.train_batch_size % args.group_size:
        raise ValueError("rollout_batch_size must divide into train batches containing whole groups")
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("gradient_accumulation_steps must not exceed train_batch_size")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.advantage_eps <= 0 or args.temperature < 0:
        raise ValueError("Invalid learning rate, clipping, advantage epsilon or sampling temperature")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.importance_reweighting_method == "gspo" and args.loss_normalization == "constant":
        raise ValueError("The existing GSPO core supports sequence normalization; choose --loss-normalization sequence")
    cliprange = args.cliprange if args.cliprange is not None else (3e-4 if args.importance_reweighting_method == "gspo" else 0.2)
    if cliprange <= 0:
        raise ValueError("cliprange must be positive")
    normalization_constant = args.normalization_constant
    if args.loss_normalization == "constant":
        if normalization_constant is None:
            normalization_constant = args.train_batch_size * args.max_new_tokens
        if normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive")
    device = torch.device(args.device)
    if args.backend == "vllm" and (device.type != "cuda" or args.dtype != "bfloat16"):
        raise ValueError("The existing vLLM helper requires CUDA training and a bfloat16 policy")
    if args.backend == "vllm" and (device.index or 0) == args.vllm_gpu:
        raise ValueError("vLLM and the training model must use different GPUs")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    set_seed(args.seed)
    rng = random.Random(args.seed)

    # 复用 GSM8K 读取函数提取 #### 后的答案，训练与验证使用不同的文件。
    if Path(args.train_path).resolve() == Path(args.val_path).resolve():
        raise ValueError("Use separate train and validation files")
    train_examples = load_gsm8k(args.train_path)
    val_examples = load_gsm8k(args.val_path)
    train_examples = rng.sample(train_examples, min(args.n_train_examples, len(train_examples)))
    val_examples = val_examples[:args.n_val_examples]
    template = (Path(__file__).parent / "prompts" / f"{args.prompt}.prompt").read_text(encoding="utf-8")

    # 加载策略模型；HF 生成与核心分词均不自动添加 BOS，保持相同的 prompt。
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError("GRPO generation needs an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attn_implementation
    ).to(device)
    model.config.use_cache = False
    # 训练仍保留 autograd；关闭 Dropout，使 rollout 策略与概率评分保持一致。
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    reward_fn = make_reward_fn(args, tokenizer)
    updates_per_rollout = args.rollout_batch_size // args.train_batch_size * args.epochs_per_rollout
    total_steps = args.num_rollout_steps * updates_per_rollout
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    config = save_train_config(
        args, total_steps=total_steps, updates_per_rollout=updates_per_rollout,
        train_examples=len(train_examples), val_examples=len(val_examples), optimizer="AdamW",
        resolved_cliprange=cliprange, resolved_normalization_constant=normalization_constant,
        top_p=1.0,
    )

    # vLLM 使用现有服务与 NCCL 同步接口；HF 后端直接使用训练模型。
    server = None
    if args.backend == "vllm":
        server = VLLMServer(
            model_id=str(args.model), gpu=args.vllm_gpu, port=args.vllm_port, seed=args.seed,
            gpu_memory_utilization=args.vllm_memory_utilization,
        )
    global_step = 0
    best_step = 0
    started = time.perf_counter()
    try:
        if server is not None:
            server.start()
            server.init_weight_sync(args.device)
            server.sync_policy_weights(model)
        with wandb.init(project=args.wandb_project, name=args.wandb_name, config=config,
                        dir=str(output_dir.resolve()), mode=args.wandb_mode) as run:
            run.define_metric("step")
            run.define_metric("*", step_metric="step")
            metrics, prompts, responses, golds = evaluate_policy(
                model, tokenizer, val_examples, template, reward_fn, args, server
            )
            best_reward = metrics["val_mean_reward"]
            run.log({"step": 0, **metrics})
            log_rollouts(run, output_dir, "val", 0, prompts, responses, golds, reward_fn, args.num_rollouts_to_log)
            save_checkpoint(model, tokenizer, output_dir / "best", 0, metrics)

            for rollout_step in range(1, args.num_rollout_steps + 1):
                # 每次生成前同步最新权重；同一 rollout 内更新期间保持生成策略不变。
                if server is not None:
                    server.sync_policy_weights(model)
                examples = rng.choices(train_examples, k=args.rollout_batch_size // args.group_size)
                prompts = [template.format(question=example["question"]) for example in examples]
                repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
                golds = [example["gold_answer"] for example in examples for _ in range(args.group_size)]
                responses, generation_metrics = generate_rollouts(
                    model, tokenizer, prompts, args, server, args.group_size, args.seed + rollout_step
                )

                # 重加权时只在更新前计算一次 old 概率，后续训练始终复用这份快照。
                old_log_probs = None
                if args.importance_reweighting_method != "none":
                    old_log_probs = compute_old_log_probs(
                        model, tokenizer, repeated_prompts, responses, args.log_prob_batch_size
                    )

                for _ in range(args.epochs_per_rollout):
                    for start in range(0, args.rollout_batch_size, args.train_batch_size):
                        end = start + args.train_batch_size
                        batch_prompts = repeated_prompts[start:end]
                        batch_responses = responses[start:end]
                        batch_old_log_probs = None
                        if old_log_probs is not None:
                            # 小训练批会重新 padding，只保留它实际使用的序列宽度。
                            width = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)["input_ids"].shape[1]
                            batch_old_log_probs = old_log_probs[start:end, :width]

                        # 核心函数已经包含奖励、优势、microbatch backward、裁剪和一次 step，
                        # 外层不能再 backward 或 optimizer.step，否则会重复更新参数。
                        loss, metadata = grpo_train_step(
                            model=model, tokenizer=tokenizer, optimizer=optimizer,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            max_grad_norm=args.max_grad_norm, reward_fn=reward_fn,
                            repeated_prompts=batch_prompts, rollout_responses=batch_responses,
                            repeated_ground_truths=golds[start:end], group_size=args.group_size,
                            baseline=args.baseline, advantage_eps=args.advantage_eps,
                            advantage_normalizer=args.advantage_normalizer,
                            importance_reweighting_method=args.importance_reweighting_method,
                            old_log_probs=batch_old_log_probs, cliprange=cliprange,
                            loss_normalization=args.loss_normalization,
                            normalization_constant=normalization_constant,
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("GRPO loss is not finite")
                        global_step += 1
                        record = {"step": global_step, "rollout_step": rollout_step,
                                  "train_loss": loss.item(), "learning_rate": args.learning_rate}
                        record.update({f"train_{key}": value.item() if isinstance(value, torch.Tensor) else value
                                       for key, value in metadata.items()})
                        run.log(record)
                        if global_step >= total_steps:
                            break
                    if global_step >= total_steps:
                        break
                run.log({"step": global_step, **{f"train_{key}": value for key, value in generation_metrics.items()}})

                # 周期性记录训练回答；最后一批也保留，便于比较训练前后的行为。
                if rollout_step == 1 or rollout_step % args.log_rollouts_every == 0 or global_step == total_steps:
                    log_rollouts(run, output_dir, "train", global_step, repeated_prompts, responses,
                                 golds, reward_fn, args.num_rollouts_to_log)
                if rollout_step % args.eval_every == 0 or global_step == total_steps:
                    # 验证必须先同步刚更新的模型，避免实际评估的是旧策略。
                    if server is not None:
                        server.sync_policy_weights(model)
                    metrics, val_prompts, val_responses, val_golds = evaluate_policy(
                        model, tokenizer, val_examples, template, reward_fn, args, server
                    )
                    if metrics["val_mean_reward"] > best_reward:
                        best_reward = metrics["val_mean_reward"]
                        best_step = global_step
                        save_checkpoint(model, tokenizer, output_dir / "best", global_step, metrics)
                    run.log({"step": global_step, "best_val_reward": best_reward, **metrics})
                    log_rollouts(run, output_dir, "val", global_step, val_prompts, val_responses,
                                 val_golds, reward_fn, args.num_rollouts_to_log)
                if global_step >= total_steps:
                    break

            # 复用保存工具，分别保留最佳验证奖励模型和最后一次更新后的模型。
            save_checkpoint(model, tokenizer, output_dir / "last", global_step, metrics)
            summary = {
                "optimizer_steps": global_step, "rollout_steps": rollout_step,
                "best_step": best_step, "best_val_reward": best_reward, **metrics,
                "elapsed_seconds": time.perf_counter() - started,
                "best_checkpoint": str((output_dir / "best").resolve()),
                "last_checkpoint": str((output_dir / "last").resolve()),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            run.summary.update(summary)
            return summary
    finally:
        # 无论正常结束还是报错，都停止本次启动的生成服务。
        if server is not None:
            server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO GSM8K training with reusable core policy losses")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prompt", choices=("r1_zero", "r1_zero_three_shot", "question_only"), default="r1_zero")
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--epochs-per-rollout", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--log-prob-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline", choices=("mean", "none"), default="mean")
    parser.add_argument("--advantage-normalizer", choices=("std", "none", "mean"), default="std")
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--importance-reweighting-method", choices=("none", "noclip", "grpo", "gspo"), default="none")
    parser.add_argument("--cliprange", type=float)
    parser.add_argument("--loss-normalization", choices=("sequence", "constant"), default="sequence")
    parser.add_argument("--normalization-constant", type=float)
    parser.add_argument("--eval-every", type=int, default=10, help="Validation interval in rollout batches")
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--num-rollouts-to-log", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="cs336-grpo")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    train(args)
