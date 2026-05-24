#!/usr/bin/env python3
"""Amazon SFT data utilities for MiniOneRec-style training."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import Dataset, concatenate_datasets, load_dataset


def apply_sample(dataset: Dataset, sample: int, seed: int) -> Dataset | None:
    """Apply the shared sampling convention: -1 full, 0 skip, >0 sampled rows."""
    if sample == 0:
        return None
    if sample > 0:
        return dataset.shuffle(seed=seed).select(range(min(sample, len(dataset))))
    return dataset


SEQUENCE_INSTRUCTIONS = [
    "Predict the next product semantic ID from the user's chronological product history.",
    "Given the user's recent product interactions, infer the next product semantic ID.",
    "Use the product history to recommend the next product semantic ID for this user.",
    "Based on the user's ordered product sequence, generate the semantic ID of the next likely product.",
    "Analyze the user's product interaction history and predict the next item semantic ID.",
    "Considering the user's recent product preferences, identify a suitable next product semantic ID.",
    "Review the user's product sequence and estimate which semantic ID should appear next.",
    "Given the ordered history of products the user interacted with, produce the next product semantic ID.",
    "Infer the next product token sequence that best matches the user's recent behavior.",
    "Use the chronological interaction trail to predict the semantic identifier of the user's next product.",
]

METADATA_INSTRUCTIONS = [
    "Identify the semantic ID of a product from its metadata.",
    "Given product title and description, predict its semantic ID.",
    "Map this product metadata to the corresponding semantic ID.",
    "Infer the item semantic ID represented by the following product metadata.",
    "Use the product textual metadata to recover its semantic identifier.",
    "Determine which semantic ID matches this product title and description.",
    "Given the content metadata of a product, generate the semantic ID assigned to that item.",
    "Connect this product metadata with the correct semantic ID token sequence.",
    "Predict the item SID using the provided product metadata.",
    "Read the product metadata fields and output the semantic ID that represents it.",
]

FUSION_INSTRUCTIONS = [
    "Recommend the likely next product from the user's interaction history.",
    "Given the user's historical semantic IDs, infer the content of the next product.",
    "Use the chronological product history to predict the next product's content.",
    "Analyze the user's product sequence and describe the next likely item.",
    "Based on the user's semantic-ID history, predict what title the next product may have.",
    "Review the user's product sequence and infer the content of a suitable next product.",
    "Given previous product semantic IDs, generate the likely next product title.",
    "Use the user's recent product trajectory to estimate the title of the next product.",
    "Translate the user's historical semantic-ID sequence into the content of the next recommended product.",
    "Considering the user's past product interactions, produce the next product's likely content.",
]


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like value, got: {value!r}")
    return [str(item) for item in parsed]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_history_text(
    history_item_ids: Iterable[str],
    history_sids: Iterable[str],
) -> str:
    del history_item_ids
    return ", ".join(str(sid) for sid in history_sids)


def build_sequence_prompt(history_text: str, target_sid: str) -> tuple[str, str]:
    prompt = (
        "The user has interacted with the following products in chronological order: "
        f"{history_text}. Predict the next product semantic ID."
    )
    return prompt, f"{target_sid}\n"


def build_item_metadata_prompt(task: str, item_sid: str, item: dict[str, Any]) -> tuple[str, str]:
    title = clean_text(item.get("title", "product"))
    description = clean_text(item.get("description", ""))
    if task == "title2sid":
        prompt = f"Which product has the title: {title}?"
        if description:
            prompt += f" Description: {description}"
        return prompt, f"{item_sid}\n"
    if task == "sid2title":
        return f'What is the title of product "{item_sid}"?', f"{title}\n"
    raise ValueError(f"Unsupported metadata task: {task}")


def build_fusion_prompt(history_text: str, target_item: dict[str, Any]) -> tuple[str, str]:
    title = clean_text(target_item.get("title", "product"))
    prompt = (
        "The user has interacted with the following products in chronological order: "
        f"{history_text}. Predict the title of the next product."
    )
    return prompt, f"{title}\n"


def select_instruction(pool: list[str], seed: int, idx: int) -> str:
    rng = random.Random(seed + idx * 1000003)
    return pool[rng.randrange(len(pool))]


def encode_prompt_target(
    tokenizer,
    instruction_text: str,
    user_input: str,
    target: str,
    cutoff_len: int,
) -> dict[str, list[int]]:
    instruction = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction_text}\n\n"
    )
    prompt = f"{instruction}### User Input: \n{user_input}\n\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    token_ids = (prompt_ids + target_ids)[-cutoff_len:]
    prompt_label_count = max(len(token_ids) - len(target_ids), 0)
    labels = [-100] * prompt_label_count + token_ids[prompt_label_count:]
    return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids), "labels": labels}


def tokenize_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    item_meta: dict[str, dict[str, Any]],
    seed: int,
    cutoff_len: int,
) -> dict[str, list[list[int]]]:
    input_ids = []
    attention_masks = []
    labels = []

    for row_idx in range(len(batch["item_sid"])):
        history_item_ids = parse_list(batch["history_item_id"][row_idx])
        history_sids = parse_list(batch["history_item_sid"][row_idx])
        target_sid = str(batch["item_sid"][row_idx])
        if len(history_item_ids) != len(history_sids):
            raise ValueError("history_item_id and history_item_sid length mismatch")

        history_text = build_history_text(history_item_ids, history_sids)
        user_input, target = build_sequence_prompt(history_text, target_sid)
        encoded = encode_prompt_target(
            tokenizer,
            select_instruction(SEQUENCE_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            target,
            cutoff_len,
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append(encoded["labels"])

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


def tokenize_metadata_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    seed: int,
    cutoff_len: int,
) -> dict[str, list[list[int]]]:
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for row_idx in range(len(batch["item_sid"])):
        user_input, target = build_item_metadata_prompt(
            str(batch["task"][row_idx]),
            str(batch["item_sid"][row_idx]),
            {
                "title": batch["title"][row_idx],
                "description": batch["description"][row_idx],
            },
        )
        encoded = encode_prompt_target(
            tokenizer,
            select_instruction(METADATA_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            target,
            cutoff_len,
        )
        for key in rows:
            rows[key].append(encoded[key])
    return rows


def tokenize_fusion_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    tokenizer,
    item_meta: dict[str, dict[str, Any]],
    seed: int,
    cutoff_len: int,
) -> dict[str, list[list[int]]]:
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for row_idx in range(len(batch["item_sid"])):
        history_item_ids = parse_list(batch["history_item_id"][row_idx])
        history_sids = parse_list(batch["history_item_sid"][row_idx])
        target_item_id = str(batch["item_id"][row_idx])
        history_text = build_history_text(history_item_ids, history_sids)
        user_input, target = build_fusion_prompt(history_text, item_meta[target_item_id])
        encoded = encode_prompt_target(
            tokenizer,
            select_instruction(FUSION_INSTRUCTIONS, seed, indices[row_idx]),
            user_input,
            target,
            cutoff_len,
        )
        for key in rows:
            rows[key].append(encoded[key])
    return rows


def load_and_tokenize_sequence_csv(
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
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc=f"Tokenizing {Path(path).name}",
        fn_kwargs={
            "tokenizer": tokenizer,
            "item_meta": item_meta,
            "seed": seed,
            "cutoff_len": cutoff_len,
        },
    )
    return dataset


def sid_to_text(tokens: list[str]) -> str:
    return "".join(tokens)


def build_metadata_rows(item_meta: dict[str, dict[str, Any]], sid_index: dict[str, list[str]]) -> Dataset:
    rows = []
    for item_id in sorted(sid_index, key=lambda x: int(x)):
        item = item_meta[str(item_id)]
        sid = sid_to_text(sid_index[item_id])
        for task in ["title2sid", "sid2title"]:
            rows.append(
                {
                    "task": task,
                    "item_id": item_id,
                    "item_sid": sid,
                    "title": clean_text(item.get("title", "product")),
                    "description": clean_text(item.get("description", "")),
                }
            )
    return Dataset.from_list(rows)


def load_and_tokenize_metadata(
    *,
    item_meta: dict[str, dict[str, Any]],
    sid_index: dict[str, list[str]],
    tokenizer,
    seed: int,
    cutoff_len: int,
    sample: int,
):
    dataset = build_metadata_rows(item_meta, sid_index)
    dataset = apply_sample(dataset, sample, seed)
    if dataset is None:
        return None
    columns = dataset.column_names
    return dataset.map(
        tokenize_metadata_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc="Tokenizing item metadata tasks",
        fn_kwargs={"tokenizer": tokenizer, "seed": seed, "cutoff_len": cutoff_len},
    )


def load_and_tokenize_fusion_csv(
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
        tokenize_fusion_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns,
        desc=f"Tokenizing fusion {Path(path).name}",
        fn_kwargs={
            "tokenizer": tokenizer,
            "item_meta": item_meta,
            "seed": seed,
            "cutoff_len": cutoff_len,
        },
    )


def build_baseline_train_dataset(
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
):
    datasets = [
        load_and_tokenize_sequence_csv(
            train_file,
            tokenizer=tokenizer,
            item_meta=item_meta,
            seed=seed,
            cutoff_len=cutoff_len,
            sample=sequence_sample,
            cache_dir=cache_dir,
        ),
        load_and_tokenize_metadata(
            item_meta=item_meta,
            sid_index=sid_index,
            tokenizer=tokenizer,
            seed=seed,
            cutoff_len=cutoff_len,
            sample=metadata_sample,
        ),
        load_and_tokenize_fusion_csv(
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
        raise ValueError("At least one SFT task must be enabled; do not set all samples to 0.")
    return concatenate_datasets(datasets).shuffle(seed=seed)
