import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, PreTrainedTokenizerBase

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # 把单条偏好对包装成长度为 1 的列表，共用批量分词、mask 和 loss 计算。
    return compute_batch_dpo_loss(
        lm, lm_ref, tokenizer, beta, [prompt], [response_chosen], [response_rejected]
    )


def tokenize_preference_batch(tokenizer, prompts, responses_chosen, responses_rejected):
    """Build right-padded (2B, L) inputs: B chosen sequences, then B rejected ones."""
    batch_size = len(prompts)
    if batch_size == 0 or len(responses_chosen) != batch_size or len(responses_rejected) != batch_size:
        raise ValueError("Expected equally sized, non-empty prompt/chosen/rejected lists")
    if tokenizer.eos_token_id is None:
        raise ValueError("DPO needs an EOS token")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    template = (Path(__file__).parent / "prompts_safety/alpaca_sft.prompt").read_text(
        encoding="utf-8"
    ).rstrip()
    prompt_ids = tokenizer(
        [template.format(instruction=prompt, response="") for prompt in prompts],
        add_special_tokens=True,
    )["input_ids"]
    chosen_ids = tokenizer(responses_chosen, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(responses_rejected, add_special_tokens=False)["input_ids"]
    sequences = [
        prefix + response + [tokenizer.eos_token_id]
        for responses in (chosen_ids, rejected_ids)
        for prefix, response in zip(prompt_ids, responses)
    ]

    # 每条序列各自记录 prompt 长度；PAD 与 EOS 即使是同一个 ID 也不会混淆。
    width = max(map(len, sequences))
    input_ids = torch.full((2 * batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        prompt_length = len(prompt_ids[index % batch_size])
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1
        response_mask[index, prompt_length:length] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "response_mask": response_mask}


def get_batch_response_log_probs(model, batch):
    """Return one summed response log probability per sequence, supporting padding."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"][:, 1:].to(device)
    logits = model(input_ids, attention_mask=attention_mask).logits

    # 第 t 个位置预测第 t+1 个 token；负交叉熵就是实际目标 token 的 log 概率。
    # reduction="none" 保留每个位置，之后只累加回答及 EOS，排除 prompt 和 PAD。
    token_log_probs = -F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1), reduction="none",
    ).reshape(input_ids.shape[0], -1)
    return token_log_probs.masked_fill(~response_mask, 0).sum(dim=-1)


def compute_batch_dpo_loss(lm, lm_ref, tokenizer, beta, prompts, responses_chosen, responses_rejected):
    """Compute mean DPO loss over B preference pairs using two model forwards."""
    batch = tokenize_preference_batch(tokenizer, prompts, responses_chosen, responses_rejected)
    batch_size = len(prompts)
    # chosen/rejected 合为 2B 条序列，每个模型只前向一次；训练模型保留梯度。
    policy_scores = get_batch_response_log_probs(lm, batch)
    with torch.no_grad():
        ref_scores = get_batch_response_log_probs(lm_ref, batch).to(policy_scores.device)
    policy_margin = policy_scores[:batch_size] - policy_scores[batch_size:]
    ref_margin = ref_scores[:batch_size] - ref_scores[batch_size:]
    return -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
