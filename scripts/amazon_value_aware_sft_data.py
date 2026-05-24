#!/usr/bin/env python3
"""Value-aware Amazon SFT data utilities following GR4AD VSL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset

from amazon_sft_data import (
    FUSION_INSTRUCTIONS,
    METADATA_INSTRUCTIONS,
    SEQUENCE_INSTRUCTIONS,
    apply_sample,
    build_fusion_prompt,
    build_history_text,
    build_item_metadata_prompt,
    build_metadata_rows,
    clean_text,
    parse_list,
    select_instruction,
    sid_to_text,
)


VALUE_TOKENS = ["[VAL_0]", "[VAL_1]", "[VAL_2]", "[VAL_3]"]


def normalize_value_token(value: Any) -> str:
    token = str(value or "").strip()
    if token in VALUE_TOKENS:
        return token
    try:
        bucket = int(value)
    except (TypeError, ValueError):
        bucket = 0
    bucket = min(max(bucket, 0), len(VALUE_TOKENS) - 1)
    return VALUE_TOKENS[bucket]


def value_bucket(value_token: str) -> int:
    token = normalize_value_token(value_token)
    return VALUE_TOKENS.index(token)


def is_value_token_text(text: str) -> bool:
    return str(text).strip() in VALUE_TOKENS


def value_sample_weight(
    value_token: str,
    *,
    scheme: str,
    min_weight: float,
    max_weight: float,
) -> float:
    if scheme == "uniform":
        return 1.0
    if scheme != "bucket_linear":
        raise ValueError(f"Unsupported value weight scheme: {scheme}")
    if len(VALUE_TOKENS) == 1:
        return float(max_weight)
    ratio = value_bucket(value_token) / (len(VALUE_TOKENS) - 1)
    return float(min_weight + (max_weight - min_weight) * ratio)


def build_value_sequence_prompt(
    history_text: str,
    target_sid: str,
    value_token: str,
) -> tuple[str, list[tuple[str, float]]]:
    prompt = (
        "The user has interacted with the following products in chronological order: "
        f"{history_text}. Predict the next product semantic ID and its commercial value bucket."
    )
    return prompt, [(target_sid, 1.0), (normalize_value_token(value_token), 1.0), ("\n", 1.0)]


def build_value_item_metadata_prompt(
    task: str,
    item_sid: str,
    item: dict[str, Any],
) -> tuple[str, list[tuple[str, float]]]:
    value_token = normalize_value_token(item.get("value_token", item.get("value_bucket", 0)))
    if task == "title2sid":
        title = clean_text(item.get("title", "product"))
        description = clean_text(item.get("description", ""))
        prompt = f"Which product has the title: {title}?"
        if description:
            prompt += f" Description: {description}"
        prompt += " Also predict its commercial value bucket."
        return prompt, [(item_sid, 1.0), (value_token, 1.0), ("\n", 1.0)]
    return build_item_metadata_prompt(task, item_sid, item)[0], [
        (build_item_metadata_prompt(task, item_sid, item)[1], 1.0)
    ]


def encode_prompt_target_with_loss_weights(
    tokenizer,
    instruction_text: str,
    user_input: str,
    target_parts: list[tuple[str, float]],
    cutoff_len: int,
    *,
    sample_weight: float,
    value_lambda: float,
) -> dict[str, list[int] | list[float]]:
    instruction = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction_text}\n\n"
    )
    prompt = f"{instruction}### User Input: \n{user_input}\n\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    target_ids: list[int] = []
    target_weights: list[float] = []
    for text, part_weight in target_parts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        target_ids.extend(ids)
        target_weights.extend([float(sample_weight * part_weight)] * len(ids))
    if tokenizer.eos_token_id is not None:
        target_ids.append(tokenizer.eos_token_id)
        target_weights.append(float(sample_weight))

    token_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    loss_weights = [0.0] * len(prompt_ids) + target_weights

    if len(token_ids) > cutoff_len:
        token_ids = token_ids[-cutoff_len:]
        labels = labels[-cutoff_len:]
        loss_weights = loss_weights[-cutoff_len:]

    return {
        "input_ids": token_ids,
        "attention_mask": [1] * len(token_ids),
        "labels": labels,
        "loss_weights": loss_weights,
    }


def tokenize_value_sequence_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    seed: int,
    cutoff_len: int,
    value_lambda: float,
    value_weight_scheme: str,
    value_weight_min: float,
    value_weight_max: float,
) -> dict[str, list[list[int]] | list[list[float]]]:
    rows: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
    for row_idx in range(len(batch["item_sid"])):
        history_item_ids = parse_list(batch["history_item_id"][row_idx])
        history_sids = parse_list(batch["history_item_sid"][row_idx])
        target_sid = str(batch["item_sid"][row_idx])
        token = normalize_value_token(batch["value_token"][row_idx])
        sample_weight = value_sample_weight(
            token,
            scheme=value_weight_scheme,
            min_weight=value_weight_min,
            max_weight=value_weight_max,
        )

        history_text = build_history_text(history_item_ids, history_sids)
        user_input, target_parts = build_value_sequence_prompt(history_text, target_sid, token)
        target_parts = [
            (text, value_lambda if is_value_token_text(text) else weight)
            for text, weight in target_parts
        ]
        encoded = encode_prompt_target_with_loss_weights(
            tokenizer,
            select_instruction(SEQUENCE_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            target_parts,
            cutoff_len,
            sample_weight=sample_weight,
            value_lambda=value_lambda,
        )
        for key in rows:
            rows[key].append(encoded[key])
    return rows


def tokenize_value_metadata_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    seed: int,
    cutoff_len: int,
    value_lambda: float,
    value_weight_scheme: str,
    value_weight_min: float,
    value_weight_max: float,
) -> dict[str, list[list[int]] | list[list[float]]]:
    rows: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
    for row_idx in range(len(batch["item_sid"])):
        item = {
            "title": batch["title"][row_idx],
            "description": batch["description"][row_idx],
            "value_token": batch["value_token"][row_idx],
        }
        token = normalize_value_token(item["value_token"])
        sample_weight = value_sample_weight(
            token,
            scheme=value_weight_scheme,
            min_weight=value_weight_min,
            max_weight=value_weight_max,
        )
        user_input, target_parts = build_value_item_metadata_prompt(
            str(batch["task"][row_idx]),
            str(batch["item_sid"][row_idx]),
            item,
        )
        target_parts = [
            (text, value_lambda if is_value_token_text(text) else weight)
            for text, weight in target_parts
        ]
        encoded = encode_prompt_target_with_loss_weights(
            tokenizer,
            select_instruction(METADATA_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            target_parts,
            cutoff_len,
            sample_weight=sample_weight,
            value_lambda=value_lambda,
        )
        for key in rows:
            rows[key].append(encoded[key])
    return rows


def tokenize_value_fusion_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    item_meta: dict[str, dict[str, Any]],
    seed: int,
    cutoff_len: int,
) -> dict[str, list[list[int]] | list[list[float]]]:
    rows: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
    for row_idx in range(len(batch["item_sid"])):
        history_item_ids = parse_list(batch["history_item_id"][row_idx])
        history_sids = parse_list(batch["history_item_sid"][row_idx])
        target_item_id = str(batch["item_id"][row_idx])
        history_text = build_history_text(history_item_ids, history_sids)
        user_input, target = build_fusion_prompt(history_text, item_meta[target_item_id])
        encoded = encode_prompt_target_with_loss_weights(
            tokenizer,
            select_instruction(FUSION_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            [(target, 1.0)],
            cutoff_len,
            sample_weight=1.0,
            value_lambda=1.0,
        )
        for key in rows:
            rows[key].append(encoded[key])
    return rows


def build_value_metadata_rows(
    item_meta: dict[str, dict[str, Any]],
    sid_index: dict[str, list[str]],
):
    dataset = build_metadata_rows(item_meta, sid_index)
    def add_value_token(row):
        item = item_meta[str(row["item_id"])]
        row["value_token"] = normalize_value_token(item.get("value_token", item.get("value_bucket", 0)))
        return row
    return dataset.map(add_value_token)


def load_and_tokenize_value_sequence_csv(
    path: str,
    *,
    tokenizer,
    seed: int,
    cutoff_len: int,
    sample: int,
    cache_dir: str,
    value_lambda: float,
    value_weight_scheme: str,
    value_weight_min: float,
    value_weight_max: float,
):
    dataset = load_dataset("csv", data_files=path, split="train", cache_dir=cache_dir)
    dataset = apply_sample(dataset, sample, seed)
    if dataset is None:
        return None
    columns = dataset.column_names
    return dataset.map(
        tokenize_value_sequence_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc=f"Tokenizing value-aware {Path(path).name}",
        fn_kwargs={
            "tokenizer": tokenizer,
            "seed": seed,
            "cutoff_len": cutoff_len,
            "value_lambda": value_lambda,
            "value_weight_scheme": value_weight_scheme,
            "value_weight_min": value_weight_min,
            "value_weight_max": value_weight_max,
        },
    )


def load_and_tokenize_value_metadata(
    *,
    item_meta: dict[str, dict[str, Any]],
    sid_index: dict[str, list[str]],
    tokenizer,
    seed: int,
    cutoff_len: int,
    sample: int,
    value_lambda: float,
    value_weight_scheme: str,
    value_weight_min: float,
    value_weight_max: float,
):
    dataset = build_value_metadata_rows(item_meta, sid_index)
    dataset = apply_sample(dataset, sample, seed)
    if dataset is None:
        return None
    columns = dataset.column_names
    return dataset.map(
        tokenize_value_metadata_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc="Tokenizing value-aware item metadata tasks",
        fn_kwargs={
            "tokenizer": tokenizer,
            "seed": seed,
            "cutoff_len": cutoff_len,
            "value_lambda": value_lambda,
            "value_weight_scheme": value_weight_scheme,
            "value_weight_min": value_weight_min,
            "value_weight_max": value_weight_max,
        },
    )


def load_and_tokenize_value_fusion_csv(
    path: str,
    *,
    tokenizer,
    item_meta: dict[str, dict[str, Any]],
    seed: int,
    cutoff_len: int,
    sample: int,
    cache_dir: str,
):
    dataset = load_dataset("csv", data_files=path, split="train", cache_dir=cache_dir)
    dataset = apply_sample(dataset, sample, seed)
    if dataset is None:
        return None
    columns = dataset.column_names
    return dataset.map(
        tokenize_value_fusion_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc=f"Tokenizing value-aware fusion {Path(path).name}",
        fn_kwargs={
            "tokenizer": tokenizer,
            "item_meta": item_meta,
            "seed": seed,
            "cutoff_len": cutoff_len,
        },
    )


def build_value_aware_train_dataset(
    train_file: str,
    *,
    tokenizer,
    item_meta: dict[str, dict[str, Any]],
    sid_index: dict[str, list[str]],
    seed: int,
    cutoff_len: int,
    sequence_sample: int,
    metadata_sample: int,
    fusion_sample: int,
    cache_dir: str,
    value_lambda: float,
    value_weight_scheme: str,
    value_weight_min: float,
    value_weight_max: float,
):
    datasets = [
        load_and_tokenize_value_sequence_csv(
            train_file,
            tokenizer=tokenizer,
            seed=seed,
            cutoff_len=cutoff_len,
            sample=sequence_sample,
            cache_dir=cache_dir,
            value_lambda=value_lambda,
            value_weight_scheme=value_weight_scheme,
            value_weight_min=value_weight_min,
            value_weight_max=value_weight_max,
        ),
        load_and_tokenize_value_metadata(
            item_meta=item_meta,
            sid_index=sid_index,
            tokenizer=tokenizer,
            seed=seed,
            cutoff_len=cutoff_len,
            sample=metadata_sample,
            value_lambda=value_lambda,
            value_weight_scheme=value_weight_scheme,
            value_weight_min=value_weight_min,
            value_weight_max=value_weight_max,
        ),
        load_and_tokenize_value_fusion_csv(
            train_file,
            tokenizer=tokenizer,
            item_meta=item_meta,
            seed=seed,
            cutoff_len=cutoff_len,
            sample=fusion_sample,
            cache_dir=cache_dir,
        ),
    ]
    datasets = [dataset for dataset in datasets if dataset is not None]
    if not datasets:
        raise ValueError("At least one value-aware SFT task must be enabled; do not set all samples to 0.")
    return concatenate_datasets(datasets).shuffle(seed=seed)
