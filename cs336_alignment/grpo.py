import torch
import os
from typing import Any, Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from einops import rearrange

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask aligned with
        labels that is 1 for response tokens and 0 for other tokens (prompt or padding).
    
        Args:
            prompt_strs: list[str]
                List of prompt strings.
            output_strs: list[str]
                List of output strings.
            tokenizer: PreTrainedTokenizer
                Tokenizer to use for tokenization.
    
        Returns:
            dict[str, torch.Tensor].
                Let prompt_and_output_lens be a list containing the lengths of the
                concatenated tokenized prompt and output strings. Then the returned
                dictionary should have the following keys:
    
                input_ids
                    torch.Tensor of shape
                    (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                    prompt and output strings, with the final token sliced off.
                labels
                    torch.Tensor of shape
                    (batch_size, max(prompt_and_output_lens) - 1): shifted input
                    ids, i.e., the input ids without the first token.
                response_mask
                    torch.Tensor of shape
                    (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                    with labels, with value 1 where the corresponding label token
                    is part of the response and 0 otherwise.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt 和 output 的数量必须相同。")
    if len(prompt_strs) == 0:
        raise ValueError("输入 batch 不能为空。")

    # 1. 分别 tokenize，不自动添加 BOS、EOS 等特殊 token。
    # 返回的是 list[list[int]]，每个内部列表对应一个样本。
    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
    )["input_ids"]

    # 2. 将每条 prompt 的 token 与对应 response 的 token 拼接。
    full_ids = [
        p_ids + r_ids
        for p_ids, r_ids in zip(prompt_ids, response_ids)
    ]

    batch_size = len(full_ids)
    max_len = max(len(ids) for ids in full_ids)

    if max_len == 0:
        raise ValueError("所有样本的 token 序列均为空。")

    # 3. 确定 padding 使用的 token ID。
    # 如果没有专用 PAD，使用 EOS 作为填充值；不把它计入回答 mask。
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer 必须提供 pad_token_id 或 eos_token_id。")

    # 4. 创建补齐后的完整序列，以及尚未移位的回答 mask。
    padded_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )

    full_response_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
    )

    for i, ids in enumerate(full_ids):
        prompt_len = len(prompt_ids[i])
        total_len = len(ids)

        # 将真实 token 填入前部，剩余位置保留 padding。
        padded_ids[i, :total_len] = torch.tensor(ids, dtype=torch.long)

        # 原始完整序列中，只有 response 对应的位置标为 True。
        full_response_mask[i, prompt_len:total_len] = True

    # 5. 构造 next-token prediction 的输入和标签。
    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:]

    # mask 必须与 labels 一起移位。
    response_mask = full_response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
        from a causal language model, and optionally the entropy of the model's
        next-token distribution.
    
        Args:
            model: PreTrainedModel
                HuggingFace model used for scoring (placed on the correct device
                and in inference mode if gradients should not be computed).
            input_ids: torch.Tensor
                shape (batch_size, sequence_length), concatenated prompt + response
                tokens as produced by your tokenization method.
            labels: torch.Tensor
                shape (batch_size, sequence_length), labels as produced by your
                tokenization method.
            return_token_entropy: bool
                If True, also return per-token entropy.
    
        Returns:
            dict[str, torch.Tensor].
                "log_probs"
                    shape (batch_size, sequence_length), conditional
                    log-probabilities log p_(theta)(x_t | x_(<t)).
                "token_entropy"
                    optional, shape (batch_size, sequence_length), per-token
                    entropy for each position (present only if
                    return_token_entropy=True).
        """
    logits = model(input_ids).logits

    all_log_probs = torch.log_softmax(logits.float(), dim=-1)
    # 表示dim=-1维按照index取出来
    log_probs = all_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = -(all_log_probs.exp() * all_log_probs).sum(dim=-1)
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    return {"log_probs": log_probs}
        
        
        
def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("num of responses must equal to num of ground_truths")
    if len(rollout_responses) == 0:
        raise ValueError("responses is empty")
    batch_size = len(rollout_responses)
    raw_rewards = torch.empty((batch_size,), dtype=float)
    total_reward = 0
    total_format_reward = 0
    for i in range(batch_size):
        output = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        raw_rewards[i] = output["reward"]
        total_reward += output["reward"]
        total_format_reward +=  output["format_reward"]
    metadata = {"mean_reward": total_reward / batch_size,
                "mean_format_reward": total_format_reward / batch_size
                }
    
    return (raw_rewards, metadata)



def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    if raw_rewards.ndim != 1 or raw_rewards.numel() == 0:
        raise ValueError("raw_rewards must be a non-empty one-dimensional tensor")
    if raw_rewards.shape[0] % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    
    rewards = rearrange(
        raw_rewards, "(prompt_bs group_size) -> prompt_bs group_size", group_size=group_size
    )
    mean_rewards = torch.mean(rewards, dim = -1, keepdim=True)
    if baseline == "mean":
        rewards = rewards - mean_rewards
        
    if advantage_normalizer == "std":
        std_rewards = torch.std(
            rewards, dim=-1, keepdim=True, correction=1 if group_size > 1 else 0
        )
        rewards = rewards / (std_rewards + advantage_eps)
    elif advantage_normalizer == "mean":
        rewards = rewards / (mean_rewards + advantage_eps)
        
    rewards = rearrange(
        rewards, "prompt_bs group_size -> (prompt_bs group_size)"
    )
    
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std(correction=0).item(),
    }
    
    return rewards, metadata




        
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if old_log_probs is not None and old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("the shape of old_log_probs and policy_log_probs must equal to (batch_size, sequence_length)")  
        
    raw_rewards_or_advantages = raw_rewards_or_advantages.reshape(-1, 1)
        
    with torch.no_grad():
        if importance_reweighting_method == "gspo":
            if old_log_probs is None:
                raise ValueError("gspo algorithm needs old log probs")  
            omega = torch.exp(policy_log_probs - old_log_probs) 
            if cliprange is not None:
                omega.clip_(1 - cliprange, 1 + cliprange)
            if response_mask is not None:
                omega = omega * response_mask / response_mask.mean(dim=-1, keepdim=True)
        elif importance_reweighting_method == "grpo":
            if old_log_probs is None:
                raise ValueError("gspo algorithm needs old log probs") 
            omega = torch.exp(policy_log_probs - old_log_probs)
            if cliprange is not None:
                omega.clip_(1 - cliprange, 1 + cliprange) 
            if response_mask is not None:
                omega = omega * response_mask
        elif importance_reweighting_method == "noclip":
            if old_log_probs is None:
                omega = torch.ones_like(policy_log_probs)  
            else:
                omega = torch.exp(policy_log_probs - old_log_probs)
            if response_mask is not None:
                omega = omega * response_mask
        elif importance_reweighting_method == "none":
            omega = torch.ones_like(policy_log_probs)
            if response_mask is not None:
                omega = omega * response_mask
        
    per_token_policy_gradient_loss = omega * policy_log_probs * raw_rewards_or_advantages

    metadata = {}
    
    return (per_token_policy_gradient_loss, metadata)


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    raise NotImplementedError







def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    raise NotImplementedError