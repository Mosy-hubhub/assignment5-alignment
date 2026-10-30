import torch
import torch.nn.functional as F
from pathlib import Path
import os
from typing import Any, Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from einops import rearrange

def convert_logits_to_logprob(logits, index, prompt_length):
    """Return the scalar response log probability for ONE unpadded sequence.

    This helper only supports batch_size == 1; it is not a batched interface.
    logits: (1, sequence_length, vocab_size).
    index: (1, sequence_length), containing the full input token IDs.
    prompt_length: number of prompt tokens, including any response prefix.
    
    Returns:
        a scalar tensor that retains gradients from logits.
    """
    if logits.ndim != 3 or index.shape != logits.shape[:2]:
        raise ValueError("Expected logits (1, L, V) and token IDs (1, L)")
    if logits.shape[0] != 1:
        raise ValueError("This helper only supports one sequence (batch_size == 1)")
    if not 1 <= prompt_length <= index.shape[1]:
        raise ValueError("prompt_length must be between 1 and sequence_length")

    # 最后一个 prompt 位置预测第一个回答 token；
    # EOS 由前一个位置预测；最后一个 logits 预测 EOS 后的 token，因此排除。
    response_logits = logits[:, prompt_length - 1:-1, :]
    response_ids = index[:, prompt_length:].to(logits.device)
    all_log_probs = torch.log_softmax(response_logits.float(), dim=-1)

    token_log_probs = all_log_probs.gather(
        dim=-1, index=response_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs.sum()


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
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must provide an EOS token")

    template_path = Path(__file__).resolve().parent / "prompts_safety/alpaca_sft.prompt"
    template = template_path.read_text(encoding="utf-8").rstrip()
    formatted_prompt = template.format(instruction=prompt, response="")
    prompt_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=True,
    )["input_ids"]
    
    response_chosen_ids = tokenizer(response_chosen,
                                    add_special_tokens=False,
                                    )["input_ids"]
    
    response_rejected_ids = tokenizer(response_rejected, 
                                      add_special_tokens=False
                                      )["input_ids"]

    # EOS 也是回答的一部分，需要计算模型结束回答的概率。
    response_chosen_ids.append(tokenizer.eos_token_id)
    response_rejected_ids.append(tokenizer.eos_token_id)

    chosen_ids = prompt_ids + response_chosen_ids
    rejected_ids = prompt_ids + response_rejected_ids
    device = next(lm.parameters()).device
    
    # chosen_input_ids.shape: (1, sequence_length)
    chosen_input_ids = torch.tensor([chosen_ids], dtype=torch.long, device=device)
    # rejected_input_ids.shape: (1, sequence_length)
    rejected_input_ids = torch.tensor([rejected_ids], dtype=torch.long, device=device)
    
    # online model
    chosen_logits = lm(chosen_input_ids).logits
    rejected_logits = lm(rejected_input_ids).logits

    # reference model
    ref_device = next(lm_ref.parameters()).device
    with torch.no_grad():
        ref_chosen_logits = lm_ref(chosen_input_ids.to(ref_device)).logits
        ref_rejected_logits = lm_ref(rejected_input_ids.to(ref_device)).logits

    prompt_length = len(prompt_ids)
    chosen_log_prob = convert_logits_to_logprob(chosen_logits, chosen_input_ids, prompt_length)
    rejected_log_prob = convert_logits_to_logprob(rejected_logits, rejected_input_ids, prompt_length)
    ref_chosen_log_prob = convert_logits_to_logprob(ref_chosen_logits, chosen_input_ids, prompt_length)
    ref_rejected_log_prob = convert_logits_to_logprob(ref_rejected_logits, rejected_input_ids, prompt_length)

    # beta 缩放的是 sigmoid 的输入；logsigmoid 避免先求 sigmoid 再取 log 的数值下溢。
    policy_log_ratio = chosen_log_prob - rejected_log_prob
    ref_log_ratio = (ref_chosen_log_prob - ref_rejected_log_prob).to(device)
    loss = -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio))
    return loss
