"""Synthetic rollout function for measuring SGLang weight update latency.

This module intentionally returns tiny trainable samples without calling the
terminal environment or SGLang generation APIs. The normal RolloutManager still
starts real SGLang rollout engines, so the train actor's update_weights path is
kept intact while rollout work is minimized.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@lru_cache(maxsize=1)
def _tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _encode_with_fallback(tokenizer, text: str, fallback: list[int]) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids or fallback


def _make_sample(
    *,
    args,
    rollout_id: int,
    group_index: int,
    sample_index: int,
    tokenizer,
    prompt_len: int,
    response_len: int,
) -> Sample:
    prompt_ids = _encode_with_fallback(
        tokenizer,
        f"Update weight benchmark rollout={rollout_id} group={group_index}. Answer:",
        [1],
    )[:prompt_len]
    response_ids = _encode_with_fallback(tokenizer, " ok" * max(response_len, 1), [2])[:response_len]
    if not response_ids:
        response_ids = [int(getattr(tokenizer, "eos_token_id", None) or 2)]

    # Keep GRPO non-constant when n_samples_per_prompt > 1; with n=1 the
    # framework disables GRPO std normalization and keeps the sample.
    reward_value = 1.0 if sample_index % 2 == 0 else -1.0
    reward_key = getattr(args, "reward_key", None) or "score"
    reward = {
        reward_key: reward_value,
        "score": reward_value,
        "accuracy": 1.0 if reward_value > 0 else 0.0,
    }

    tokens = prompt_ids + response_ids
    return Sample(
        group_index=group_index,
        index=sample_index,
        prompt="update weight benchmark",
        tokens=tokens,
        response="ok",
        response_length=len(response_ids),
        reward=reward,
        loss_mask=[1] * len(response_ids),
        rollout_log_probs=[0.0] * len(response_ids),
        status=Sample.Status.COMPLETED,
        metadata={
            "benchmark": "update_weight",
            "rollout_id": rollout_id,
            "_perf": {
                "total": 0.0,
                "env_client_create": 0.0,
                "env_allocate": 0.0,
                "env_reset": 0.0,
                "env_evaluate": 0.0,
                "env_heartbeat": 0.0,
                "runner_run_episode": 0.0,
                "sglang_generate": 0.0,
                "tool_exec": 0.0,
                "runner_other": 0.0,
                "sglang_turn_count": 0.0,
                "tool_call_count": 0.0,
            },
        },
    )


def generate_rollout(args, rollout_id: int, data_source: Any = None, evaluation: bool = False):
    if evaluation:
        return RolloutFnEvalOutput(data={}, metrics={"update_weight_bench/eval_skipped": 1})

    tokenizer = _tokenizer(args.hf_checkpoint)
    prompt_len = _env_int("UPDATE_WEIGHT_BENCH_PROMPT_LEN", 16)
    response_len = _env_int("UPDATE_WEIGHT_BENCH_RESPONSE_LEN", 8)

    samples: list[list[Sample]] = []
    sample_index = 0
    for group_index in range(args.rollout_batch_size):
        group = []
        for _ in range(args.n_samples_per_prompt):
            group.append(
                _make_sample(
                    args=args,
                    rollout_id=rollout_id,
                    group_index=group_index,
                    sample_index=sample_index,
                    tokenizer=tokenizer,
                    prompt_len=prompt_len,
                    response_len=response_len,
                )
            )
            sample_index += 1
        samples.append(group)

    return RolloutFnTrainOutput(
        samples=samples,
        metrics={
            "update_weight_bench/rollout_id": rollout_id,
            "update_weight_bench/groups": len(samples),
            "update_weight_bench/samples": sample_index,
            "update_weight_bench/prompt_len": prompt_len,
            "update_weight_bench/response_len": response_len,
        },
    )
