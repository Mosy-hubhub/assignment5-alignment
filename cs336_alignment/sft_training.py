import argparse
import gzip
import json
import math
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]

class PackedSFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle): 
        """
        Constructs the dataset. 
        tokenizer is a transformers tokenizer for tokenizing and encoding the instruction tuning data. 
        dataset_path is a path to instruction tuning data. 
        seq_length is the desired length of sequences to generate from the dataset. 
        shuffle controls whether documents are shuffled before concatenation.
        """
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must provide an EOS token to separate documents")
        self.seq_length = seq_length

        template_path = Path(__file__).resolve().parent / "prompts_safety/alpaca_sft.prompt"
        # Ignore the template file's trailing newline before adding the EOS delimiter.
        template = template_path.read_text(encoding="utf-8").rstrip()
        dataset_path = Path(dataset_path)
        open_file = gzip.open if dataset_path.suffix == ".gz" else open
        # documents：[str]
        documents = []
        with open_file(dataset_path, "rt", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                example = json.loads(line)
                if not isinstance(example.get("prompt"), str) or not isinstance(example.get("response"), str):
                    raise ValueError(f"Expected prompt/response strings at line {line_number}")
                documents.append(template.format(
                    instruction=example["prompt"], response=example["response"]
                ))

        # Shuffle whole documents before packing, not individual tokens.
        if shuffle:
            random.shuffle(documents)

        all_token_ids = []
        for document in documents:
            # Keep the tokenizer's default special tokens (e.g. Llama's BOS),
            # then explicitly append EOS as the document delimiter.
            all_token_ids.extend(tokenizer.encode(document, add_special_tokens=True))
            all_token_ids.append(tokenizer.eos_token_id)
        self.token_ids = torch.tensor(all_token_ids, dtype=torch.long)

        # Each input block needs one additional token for its shifted labels.
        # Only complete blocks are exposed; no padding is added to the tail.
        self.num_sequences = max(0, (len(self.token_ids) - 1) // self.seq_length)

    def __len__(self):
        """
        Returns an integer, the number of sequences in this Dataset.
        """
        return self.num_sequences
    
    def __getitem__(self, i):
        """
        Returns the i-th element of the Dataset. This function should
        return a dictionary with input_ids and labels, both PyTorch tensors of shape
        (seq_length,)
        """
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError("PackedSFTDataset index out of range")
        start = i * self.seq_length
        return {
            "input_ids": self.token_ids[start:start + self.seq_length],
            "labels": self.token_ids[start + 1:start + self.seq_length + 1],
        }


def compute_sft_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


def evaluate_loss(model, data_loader, device, max_batches=None):
    """Compute token-averaged validation CE without changing model parameters."""
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(data_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                loss = compute_sft_loss(model, batch, device)
                num_tokens = batch["labels"].numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Validation data contains no complete sequences")
    return total_loss / total_tokens


def save_train_config(args, total_steps, warmup_steps, train_sequences, val_sequences):
    """Save the training settings locally and return them for W&B."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": str(args.model), "train_path": str(Path(args.train_path).resolve()),
        "val_path": str(Path(args.val_path).resolve()), "device": str(args.device),
        "seq_length": args.seq_length, "microbatch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs, "total_steps": total_steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps, "max_grad_norm": args.max_grad_norm,
        "log_every": args.log_every, "eval_every": args.eval_every, "max_eval_batches": args.max_eval_batches,
        "seed": args.seed, "dtype": args.dtype, "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_sequences": train_sequences, "val_sequences": val_sequences,
        "loss_scope": "full_text",
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def train(args):
    """Train on full packed texts, validate periodically, and save a HF checkpoint."""
    if min(args.seq_length, args.batch_size, args.epochs, args.gradient_accumulation_steps, args.log_every, args.eval_every) <= 0:
        raise ValueError("Lengths, batch sizes, epochs and logging intervals must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Invalid optimizer or gradient-clipping settings")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("max_eval_batches must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory to avoid overwriting a run: {output_dir}")

    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = PackedSFTDataset(tokenizer, args.train_path, args.seq_length, shuffle=True)
    val_dataset = PackedSFTDataset(tokenizer, args.val_path, args.seq_length, shuffle=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation data must contain complete packed sequences")
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 将命令行中的精度名称转换为对应的 torch.dtype。
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    
    
    # 按梯度累积组计算实际参数更新次数，设置先 warmup、再余弦衰减的学习率。
    # 一个 step 表示一次 optimizer.step()，可能包含多个 microbatch。
    updates_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    total_steps = args.epochs * updates_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    config = save_train_config(
        args, total_steps, warmup_steps, len(train_dataset), len(val_dataset)
    )

    # training loop
    global_step = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with wandb.init(
        project=args.wandb_project, name=args.wandb_name, config=config,
        dir=str(output_dir.resolve()), mode=args.wandb_mode,
    ) as run:
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        initial_val_loss = evaluate_loss(model, val_loader, device, args.max_eval_batches)
        record = {"event": "validation", "step": 0, "val_loss": initial_val_loss}
        run.log(record)

        for epoch in range(args.epochs):
            for batch_index, batch in enumerate(data_loader):
                # 确定当前梯度累积组的范围，并统计组内实际样本数和损失。
                group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                group_end = min(group_start + args.gradient_accumulation_steps, len(data_loader))
                if batch_index == group_start:
                    group_examples = min(group_end * args.batch_size, len(train_dataset)) - group_start * args.batch_size
                    group_loss = torch.zeros((), device=device)

                # 当前 microbatch 前向计算交叉熵，再按其样本占比缩放并反向传播。
                # 这样最后不足一个完整 batch 或累积组时，梯度仍按实际样本数平均。
                micro_loss = compute_sft_loss(model, batch, device)
                weighted_loss = micro_loss * (batch["input_ids"].shape[0] / group_examples)
                weighted_loss.backward()
                # 日志只保留损失数值；及时释放计算图，减少梯度累积期间的显存占用。
                group_loss += weighted_loss.detach()
                del micro_loss, weighted_loss
                # 尚未处理完当前累积组时，继续累积梯度，暂不更新参数。
                if batch_index + 1 != group_end:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "event": "train", "step": global_step, "epoch": epoch + 1,
                    "train_loss": group_loss.item(), "grad_norm": grad_norm.item(),
                    "learning_rate": lr_used, "num_sequences": group_examples,
                }
                run.log(record)

                # 定期计算验证损失，并确保最后一次参数更新后也进行验证。
                if global_step % args.eval_every == 0 or global_step == total_steps:
                    val_loss = evaluate_loss(model, val_loader, device, args.max_eval_batches)
                    record = {"event": "validation", "step": global_step, "val_loss": val_loss}
                    run.log(record)
                # 达到计划的更新次数后，退出当前批次循环和外层 epoch 循环。
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        model.config.use_cache = original_use_cache
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        summary = {
            "optimizer_steps": global_step, "initial_val_loss": initial_val_loss,
            "final_val_loss": val_loss, "elapsed_seconds": time.perf_counter() - started,
            "output_dir": str(output_dir.resolve()),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run.summary.update(summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Packed full-text SFT for supplement section 4")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data/sft-instruct/train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data/sft-instruct/test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-sft")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    train(args)
