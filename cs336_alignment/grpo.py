import torch
import os
from typing import Any, Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, PreTrainedTokenizerBase
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
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")
    batch_size = policy_log_probs.shape[0]
    if raw_rewards_or_advantages.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("rewards or advantages must have shape (batch_size,) or (batch_size, 1)")
    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise ValueError("unknown importance_reweighting_method")
    if old_log_probs is not None and old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs and policy_log_probs must have the same shape")
    if response_mask is not None and response_mask.shape != policy_log_probs.shape:
        raise ValueError("response_mask and policy_log_probs must have the same shape")
    if importance_reweighting_method != "none":
        if old_log_probs is None:
            raise ValueError("importance reweighting requires old_log_probs")
        if importance_reweighting_method in ("grpo", "gspo"):
            if cliprange is None or cliprange < 0:
                raise ValueError("clipped methods require a non-negative cliprange")


    advantages = raw_rewards_or_advantages.detach().reshape(batch_size, 1)
    metadata = {}
    if importance_reweighting_method == "none":
        # no importance reweighting
        return -advantages * policy_log_probs, metadata

    elif importance_reweighting_method == "noclip":
        # apply importance reweighting without clipping
        # importance reweighting below
        log_ratio = policy_log_probs - old_log_probs.detach()
        ratio = log_ratio.exp()
        return -advantages * ratio, metadata
    
    elif importance_reweighting_method == "grpo":
        # do PPO/GRPO-style token-level reweighting and clipping
        # importance reweighting below
        log_ratio = policy_log_probs - old_log_probs.detach()
        ratio = log_ratio.exp()
        # clipping
        objective = ratio * advantages
        clipped_objective = ratio.clamp(1 - cliprange, 1 + cliprange) * advantages
        metadata["clip_fraction"] = (clipped_objective < objective).to(ratio.dtype).detach().mean()
        objective = torch.minimum(objective, clipped_objective)
        per_token_policy_gradient_loss = -objective
        return per_token_policy_gradient_loss, metadata

    elif importance_reweighting_method == "gspo":
        # do GSPO-style sequence-level reweighting and clipping.
        # GSPO-style sequence-level reweighting below
        log_ratio = policy_log_probs - old_log_probs.detach()
        if response_mask is None:
            raise ValueError("GSPO requires response_mask to identify response tokens")
        mask = response_mask.to(device=log_ratio.device, dtype=log_ratio.dtype)
        response_lengths = mask.sum(dim=-1, keepdim=True)
        if (response_lengths <= 0).any():
            raise ValueError("each GSPO response must contain at least one unmasked token")       
        # log_ratio_gspo = (pi_theta(y_1) + pi_theta(y_2) + ... - pi_0(y_1) - pi_0(y_2) - ...) / len(y)
        log_ratio_gspo = log_ratio.masked_fill(mask == 0, 0).sum(dim=-1, keepdim=True) / response_lengths
        ratio_gspo = log_ratio_gspo.exp()
        # clipping
        objective = ratio_gspo * advantages
        clipped_objective = ratio_gspo.clamp(1 - cliprange, 1 + cliprange) * advantages
        metadata["clip_fraction"] = (clipped_objective < objective).to(ratio_gspo.dtype).detach().mean()
        objective = torch.minimum(objective, clipped_objective)
        per_token_policy_gradient_loss = -objective.expand_as(policy_log_probs)
        return per_token_policy_gradient_loss, metadata



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
    if (mask.sum(dim=-1) == 0).any():
        raise ValueError("each sequence must contain at least one unmasked response token")
    per_token_policy_gradient_loss = mask * per_token_policy_gradient_loss
    batch_size = per_token_policy_gradient_loss.shape[0] 
    if loss_normalization == "sequence":
        per_token_policy_gradient_loss = per_token_policy_gradient_loss / mask.sum(dim=-1, keepdim=True)
        per_token_policy_gradient_loss.mul_(1 / batch_size)
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("constant normalization needs normalization_constant")
        if normalization_constant <= 0:
            raise ValueError("normalization_constant should bigger than 0")
        per_token_policy_gradient_loss.mul_(1 / normalization_constant)
    return per_token_policy_gradient_loss.sum()




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
    batch_size = len(rollout_responses)
    
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("gradient_accumulation_steps must be between 1 and batch_size")

  
    tokenized_batch = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )
    device = next(model.parameters()).device
    # 保持全批次的 padding 宽度一致，每次只将一个 microbatch 移到模型设备。
    input_ids = tokenized_batch["input_ids"]
    labels = tokenized_batch["labels"]
    response_mask = tokenized_batch["response_mask"]
    if old_log_probs is not None:
        old_log_probs = old_log_probs.detach()

    raw_rewards, metadata1 = compute_rollout_rewards(reward_fn,
                                                    rollout_responses,
                                                    repeated_ground_truths
                                                    )

    rewards, metadata2 = compute_group_normalized_rewards(raw_rewards,
                                                         group_size,
                                                         baseline,
                                                         advantage_eps,
                                                         advantage_normalizer,
                                                         )
    metadata = {**metadata1, **metadata2}
    loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    token_count = 0
    optimizer.zero_grad(set_to_none=True)


    for microbatch_index in range(gradient_accumulation_steps):
        # 用整数边界拆出恰好指定数量的非空 microbatch，也支持不等大的尾批。
        start = microbatch_index * batch_size // gradient_accumulation_steps
        end = (microbatch_index + 1) * batch_size // gradient_accumulation_steps
        micro_mask = response_mask[start:end].to(device)
        response_scores = get_response_log_probs(
            model,
            input_ids[start:end].to(device),
            labels[start:end].to(device),
            return_token_entropy=True,
        )
        per_token_loss, micro_metadata = compute_policy_gradient_loss(
            rewards[start:end].to(device),
            response_scores["log_probs"],
            importance_reweighting_method,
            old_log_probs[start:end].to(device) if old_log_probs is not None else None,
            cliprange,
            micro_mask,
        )
        micro_loss = aggregate_loss_across_microbatch(
            per_token_loss, micro_mask, loss_normalization, normalization_constant
        )
        microbatch_weight = (end - start) / batch_size
        if loss_normalization == "sequence":
            # 聚合函数返回小批的序列均值，按样本占比还原为全批均值。
            micro_loss = micro_loss * microbatch_weight
        # constant 返回小批损失总和 / 同一个常数，直接相加，不再除累积步数
        # 如果不每个micro_loss反向传播，而是直接不detach累加到loss里面，那么将失去accumulation的意义
        # 中间的激活态和计算图将会越来越多
        micro_loss.backward()
        loss = loss + micro_loss.detach()

        entropy_sum = entropy_sum + response_scores["token_entropy"].detach().masked_select(
            micro_mask.bool()
        ).sum()
        token_count += int(micro_mask.sum().item())
        for key, value in micro_metadata.items():
            metadata[key] = metadata.get(key, 0.0) + value.detach() * microbatch_weight

        # 不保留已完成 microbatch 的输出或计算图。
        del response_scores, per_token_loss, micro_loss


    metadata["token_entropy"] = entropy_sum / token_count
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf")
    )
    metadata["grad_norm"] = grad_norm.detach()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return loss, metadata