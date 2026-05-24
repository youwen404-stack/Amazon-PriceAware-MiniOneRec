#!/usr/bin/env python3
"""Evaluate Amazon SFT checkpoints with MiniOneRec-style constrained SID decoding."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hash(token_ids: Iterable[int]) -> str:
    return "-".join(str(token_id) for token_id in token_ids)


class ConstrainedSidLogitsProcessor(LogitsProcessor):
    """MiniOneRec-style trie constraint over legal semantic ID responses."""

    def __init__(self, prefix_allowed_tokens_fn, num_beams: int, prefix_index: int, eos_token_id: int):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = num_beams
        self.prefix_index = prefix_index
        self.eos_token_id = eos_token_id
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        for batch_id, beam_group in enumerate(input_ids.view(-1, self.num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_group):
                if self.step == 0:
                    hash_key = sent[-self.prefix_index :].tolist()
                else:
                    hash_key = sent[-self.step :].tolist()
                allowed = self.prefix_allowed_tokens_fn(batch_id, hash_key)
                if not allowed:
                    warnings.warn(f"No valid SID tokens for hash_key={hash_key}; forcing EOS.")
                    mask[batch_id * self.num_beams + beam_id, self.eos_token_id] = 0
                    continue
                mask[batch_id * self.num_beams + beam_id, allowed] = 0
        self.step += 1
        return scores + mask


def parse_sid_list(value: str) -> list[str]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected history_item_sid list, got: {value!r}")
    return [str(item) for item in parsed]


def encode_text(tokenizer, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        token_ids = [tokenizer.bos_token_id] + token_ids
    if eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    return token_ids


def build_minionerec_eval_prompt(history: str) -> tuple[str, str]:
    instruction_text = "Can you predict the next possible item that the user may expect?"
    user_input = (
        "Can you predict the next possible item the user may expect, "
        f"given the following chronological interaction history: {history}"
    )
    return instruction_text, user_input


def build_amazon_train_eval_prompt(history: str) -> tuple[str, str]:
    instruction_text = "Predict the next product semantic ID from the user's chronological product history."
    user_input = (
        "The user has interacted with the following products in chronological order: "
        f"{history}. Predict the next product semantic ID."
    )
    return instruction_text, user_input


def build_eval_prompt(
    row: dict[str, str],
    tokenizer,
    max_len: int,
    prompt_mode: str,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    history = ", ".join(parse_sid_list(row["history_item_sid"]))
    target = str(row["item_sid"]).strip()
    if prompt_mode == "minionerec_eval":
        instruction_text, user_input = build_minionerec_eval_prompt(history)
    elif prompt_mode == "amazon_train":
        instruction_text, user_input = build_amazon_train_eval_prompt(history)
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction_text}

"""
    prompt = f"""### User Input: 
{user_input}

### Response:
"""
    token_ids = encode_text(tokenizer, instruction, bos=True, eos=False)
    token_ids += encode_text(tokenizer, prompt, bos=False, eos=False)
    token_ids = token_ids[-max_len:]
    record = {
        "input": user_input,
        "output": target,
        "raw_user_id": row.get("raw_user_id", ""),
        "item_id": row.get("item_id", ""),
        "raw_parent_asin": row.get("raw_parent_asin", ""),
        "price_value": row.get("price_value", ""),
        "value_bucket": row.get("value_bucket", ""),
        "value_token": row.get("value_token", ""),
    }
    return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}, record


def load_eval_examples(test_data_path: str, tokenizer, max_len: int, sample: int, seed: int, prompt_mode: str):
    with open(test_data_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sample > 0:
        rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    examples = []
    records = []
    for row in rows:
        example, record = build_eval_prompt(row, tokenizer, max_len, prompt_mode)
        examples.append(example)
        records.append(record)
    return examples, records


def batch_examples(examples: list[dict[str, list[int]]], tokenizer, batch_size: int):
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        max_len = max(len(example["input_ids"]) for example in chunk)
        input_ids = []
        attention_mask = []
        for example in chunk:
            pad_len = max_len - len(example["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad_len + example["input_ids"])
            attention_mask.append([0] * pad_len + [1] * len(example["input_ids"]))
        yield torch.tensor(input_ids), torch.tensor(attention_mask), max_len


def build_sid_trie(tokenizer, info_file: str, base_model: str):
    with open(info_file, encoding="utf-8") as f:
        sid_strings = [line.split("\t")[0].strip() + "\n" for line in f if line.strip()]
    sid_responses = [f"### Response:\n{sid}" for sid in sid_strings]
    if "llama" in base_model.lower():
        prefix_ids = [tokenizer(text).input_ids[1:] for text in sid_responses]
    else:
        prefix_ids = [tokenizer(text).input_ids for text in sid_responses]
    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    hash_dict: dict[str, set[int]] = {}
    for token_ids in prefix_ids:
        token_ids = list(token_ids) + [tokenizer.eos_token_id]
        for i in range(prefix_index, len(token_ids)):
            if i == prefix_index:
                hash_key = get_hash(token_ids[:i])
            else:
                hash_key = get_hash(token_ids[prefix_index:i])
            hash_dict.setdefault(hash_key, set()).add(token_ids[i])
    return {key: sorted(values) for key, values in hash_dict.items()}, prefix_index


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_sid_value_maps(info_file: str, item_meta_path: str):
    item_meta = json.load(open(item_meta_path, encoding="utf-8"))
    sid_to_item_id = {}
    with open(info_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sid_to_item_id[parts[0].strip()] = str(parts[2]).strip()
    sid_to_price = {}
    sid_to_bucket = {}
    for sid, item_id in sid_to_item_id.items():
        meta = item_meta.get(str(item_id), {})
        sid_to_price[sid] = safe_float(meta.get("price_value", meta.get("price")))
        bucket = meta.get("value_bucket", meta.get("item_value_bucket"))
        try:
            sid_to_bucket[sid] = int(bucket)
        except (TypeError, ValueError):
            sid_to_bucket[sid] = None
    return sid_to_price, sid_to_bucket


def compute_metrics(records: list[dict[str, Any]], sid_to_price: dict[str, float | None], sid_to_bucket: dict[str, int | None]):
    topk_list = [1, 3, 5, 10, 20]
    metrics: dict[str, dict[str, float]] = {
        "NDCG": {},
        "HR": {},
        "AvgPrice": {},
        "AvgValueBucket": {},
        "HitAvgPrice": {},
        "HitAvgValueBucket": {},
    }
    total = len(records)
    for topk in topk_list:
        ndcg = 0.0
        hr = 0.0
        top_prices = []
        top_buckets = []
        hit_prices = []
        hit_buckets = []
        for record in records:
            target = str(record["output"]).strip()
            preds = [str(item).strip() for item in record["predict"][:topk]]
            if target in preds:
                rank = preds.index(target)
                ndcg += 1 / math.log(rank + 2)
                hr += 1
                price = sid_to_price.get(target)
                bucket = sid_to_bucket.get(target)
                if price is not None:
                    hit_prices.append(price)
                if bucket is not None:
                    hit_buckets.append(bucket)
            for pred in preds:
                price = sid_to_price.get(pred)
                bucket = sid_to_bucket.get(pred)
                if price is not None:
                    top_prices.append(price)
                if bucket is not None:
                    top_buckets.append(bucket)
        metrics["NDCG"][str(topk)] = ndcg / total / (1.0 / math.log(2)) if total else 0.0
        metrics["HR"][str(topk)] = hr / total if total else 0.0
        metrics["AvgPrice"][str(topk)] = float(np.mean(top_prices)) if top_prices else 0.0
        metrics["AvgValueBucket"][str(topk)] = float(np.mean(top_buckets)) if top_buckets else 0.0
        metrics["HitAvgPrice"][str(topk)] = float(np.mean(hit_prices)) if hit_prices else 0.0
        metrics["HitAvgValueBucket"][str(topk)] = float(np.mean(hit_buckets)) if hit_buckets else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Amazon SFT checkpoint with MiniOneRec-style SID decoding.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--item-meta-path", required=True)
    parser.add_argument("--result-json-data", required=True)
    parser.add_argument("--metrics-json-data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--cutoff-len", type=int, default=2560)
    parser.add_argument("--prompt-mode", choices=["minionerec_eval", "amazon_train"], default="minionerec_eval")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"checkpoint directory does not exist: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    examples, records = load_eval_examples(
        args.test_data_path,
        tokenizer,
        args.cutoff_len,
        args.sample,
        args.seed,
        args.prompt_mode,
    )
    sid_trie, prefix_index = build_sid_trie(tokenizer, args.info_file, args.base_model)
    sid_to_price, sid_to_bucket = load_sid_value_maps(args.info_file, args.item_meta_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def prefix_allowed_tokens_fn(_batch_id, input_ids):
        return sid_trie.get(get_hash(input_ids), [])

    outputs = []
    for input_ids, attention_mask, prompt_len in tqdm(list(batch_examples(examples, tokenizer, args.batch_size))):
        processor = ConstrainedSidLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=args.num_beams,
            prefix_index=prefix_index,
            eos_token_id=tokenizer.eos_token_id,
        )
        generation_config = GenerationConfig(
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            length_penalty=args.length_penalty,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=None,
            top_p=None,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList([processor]),
            )
        completions = generation_output.sequences[:, prompt_len:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        decoded = [text.split("Response:\n")[-1].strip() for text in decoded]
        outputs.extend(decoded[i * args.num_beams : (i + 1) * args.num_beams] for i in range(len(decoded) // args.num_beams))

    for record, prediction in zip(records, outputs):
        record["predict"] = prediction

    Path(args.result_json_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json_data).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json_data, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    metrics = compute_metrics(records, sid_to_price, sid_to_bucket)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    with open(args.metrics_json_data, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
