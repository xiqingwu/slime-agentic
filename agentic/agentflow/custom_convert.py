"""
AgentFlow multi-turn custom convert.

Splits each multi-turn trajectory into independent training samples (one per turn),
with all turns from the same trajectory sharing the same normalized advantage.

Usage:
     --custom-convert-samples-to-train-data-path custom_convert.custom_convert
"""

import torch


def custom_convert(args, samples):
    # ── 1. Trajectory-level GRPO reward normalization ──
    raw_rewards = [s.get_reward_value(args) for s in samples]
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        n = getattr(args, "n_samples_per_prompt", 1)
        if rewards_tensor.shape[-1] == n * getattr(args, "rollout_batch_size", 1):
            rewards_tensor = rewards_tensor.reshape(-1, n)
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        rewards_tensor = rewards_tensor - mean

        if (
            getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
            and getattr(args, "grpo_std_normalization", False)
        ):
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)

    normalized_rewards = rewards_tensor.flatten().tolist()

    # ── 2. Split turns into independent training samples ──
    tokens_list = []
    response_lengths = []
    loss_masks = []
    rewards = []
    raw_reward_list = []
    truncated_list = []
    sample_indices = []
    rollout_log_probs_list = []
    has_rollout_log_probs = False
    # Per-sequence multimodal tensors (pixel_values, image_grid_thw, ...). Must stay
    # aligned 1:1 with tokens_list through turn-expansion and trimming (plan §7.4).
    # None for text-only sequences; the training backend skips None entries.
    multimodal_list = []
    has_multimodal = False

    for i, sample in enumerate(samples):
        meta = sample.train_metadata
        if meta is None or "turns" not in meta:
            tokens_list.append(sample.tokens)
            response_lengths.append(sample.response_length)
            lm = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if sample.remove_sample:
                lm = [0] * sample.response_length
            loss_masks.append(lm)
            rewards.append(normalized_rewards[i])
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(1 if sample.status == sample.Status.TRUNCATED else 0)
            sample_indices.append(sample.index)
            if sample.rollout_log_probs is not None:
                rollout_log_probs_list.append(sample.rollout_log_probs)
                has_rollout_log_probs = True
            mm = getattr(sample, "multimodal_train_inputs", None)
            multimodal_list.append(mm)
            if mm is not None:
                has_multimodal = True
            continue

        turns = meta["turns"]
        # Divide by T_i so that summing over turns gives (1/T_i) * sum_t,
        # matching the J_Flow-GRPO objective which averages across turns.
        norm_reward = normalized_rewards[i] / len(turns)
        is_truncated = 1 if sample.status == sample.Status.TRUNCATED else 0

        for turn in turns:
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            lm = list(turn["loss_mask"])
            if sample.remove_sample:
                lm = [0] * turn["response_length"]
            loss_masks.append(lm)
            rewards.append(norm_reward)
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(is_truncated)
            sample_indices.append(sample.index)
            if turn.get("rollout_log_probs"):
                rollout_log_probs_list.append(turn["rollout_log_probs"])
                has_rollout_log_probs = True
            # Each turn carries the image(s) it actually used (e.g. the processed
            # image fed back into that turn). None when the turn had no image.
            mm = turn.get("multimodal_train_inputs")
            multimodal_list.append(mm)
            if mm is not None:
                has_multimodal = True

    # ── 3. Assemble train_data (no sample-count trimming) ──
    # slime v0.3.0's build_dp_schedule groups training samples by rollout_id and
    # packs `global_batch_size` *rollouts* (not samples) per step, dropping only
    # whole trailing rollouts that don't fit. The legacy trim-to-multiple-of-
    # global_batch_size (a v0.2.2 data-iterator workaround) operated on the expanded
    # *sample* count and could slice a trajectory's trailing turns off mid-rollout,
    # corrupting that rollout's reward amortization. Letting the scheduler do the
    # packing keeps every trajectory's turns intact.
    train_data = {
        "tokens": tokens_list,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "raw_reward": raw_reward_list,
        "truncated": truncated_list,
        "sample_indices": sample_indices,
        # v0.3.0 groups all turns of one trajectory into a single rollout via this
        # id; every turn carries the source trajectory's sample.index, and turns of
        # the same trajectory are contiguous (see build_dp_schedule's grouping).
        "rollout_ids": sample_indices,
    }
    if has_rollout_log_probs:
        train_data["rollout_log_probs"] = rollout_log_probs_list
    # Mirror slime's default convert: only emit when at least one sequence has images.
    if has_multimodal:
        train_data["multimodal_train_inputs"] = multimodal_list

    return train_data
