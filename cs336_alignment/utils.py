"""Shared model loading and greedy generation for the safety baselines."""

from pathlib import Path
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"


def load_generation_model(model_name_or_path: str, device: str, seed: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    model_device = torch.device(device)
    dtype = torch.float32
    if model_device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(model_device)[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must have a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype
    ).to(model_device)
    model.eval()
    return model, tokenizer, model_device


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, device):
    """Return answer texts and generation-only seconds, excluding tokenization."""
    import torch

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=["# Query:"], tokenizer=tokenizer,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    texts = tokenizer.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return [text.split("# Query:", 1)[0].strip() for text in texts], elapsed
