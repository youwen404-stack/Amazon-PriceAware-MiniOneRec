#!/usr/bin/env python3
"""Inspect one value-aware SFT batch and compare HF loss with weighted loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amazon_value_aware_sft_data import VALUE_TOKENS, build_value_aware_train_dataset
from train_amazon_value_aware_sft import ValueAwareDataCollator, load_sid_tokens


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def summarize(name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        print(f"{name}: count=0")
        return
    qs = torch.quantile(values.float().cpu(), torch.tensor([0.5, 0.9, 0.99]))
    print(
        f"{name}: count={values.numel()} "
        f"mean={values.float().mean().item():.4f} "
        f"median={qs[0].item():.4f} p90={qs[1].item():.4f} "
        f"p99={qs[2].item():.4f} max={values.float().max().item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--item-meta-path", required=True)
    parser.add_argument("--sid-index-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-lambda", type=float, default=1.0)
    parser.add_argument("--value-weight-scheme", choices=["uniform", "bucket_linear"], default="bucket_linear")
    parser.add_argument("--value-weight-min", type=float, default=1.0)
    parser.add_argument("--value-weight-max", type=float, default=2.0)
    args = parser.parse_args()

    item_meta = load_json(args.item_meta_path)
    sid_index = load_json(args.sid_index_path)
    sid_tokens = load_sid_tokens(args.sid_index_path)
    new_tokens = sorted(set(sid_tokens) | set(VALUE_TOKENS))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(new_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    dataset = build_value_aware_train_dataset(
        args.train_file,
        tokenizer=tokenizer,
        item_meta=item_meta,
        sid_index=sid_index,
        seed=args.seed,
        cutoff_len=args.cutoff_len,
        sequence_sample=args.sample,
        metadata_sample=args.sample,
        fusion_sample=args.sample,
        cache_dir=args.cache_dir,
        value_lambda=args.value_lambda,
        value_weight_scheme=args.value_weight_scheme,
        value_weight_min=args.value_weight_min,
        value_weight_max=args.value_weight_max,
    )
    features = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    batch = ValueAwareDataCollator(tokenizer)(features)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    labels = batch["labels"]
    loss_weights = batch["loss_weights"]
    model_inputs = {k: v for k, v in batch.items() if k != "loss_weights"}

    with torch.no_grad():
        hf_outputs = model(**model_inputs)
        hf_loss = hf_outputs.loss
        logits = hf_outputs.logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

    valid = shift_labels != -100
    positive_weight = valid & (shift_weights > 0)
    weighted_loss = (token_loss * shift_weights * valid.float()).sum() / (
        (shift_weights * valid.float()).sum().clamp_min(1.0)
    )
    unweighted_custom_loss = token_loss[valid].mean()

    sid_ids = set(tokenizer.convert_tokens_to_ids(sid_tokens))
    value_ids = set(tokenizer.convert_tokens_to_ids(VALUE_TOKENS))
    sid_mask = valid & torch.tensor(
        [[int(token_id) in sid_ids for token_id in row.tolist()] for row in shift_labels],
        device=shift_labels.device,
        dtype=torch.bool,
    )
    value_mask = valid & torch.tensor(
        [[int(token_id) in value_ids for token_id in row.tolist()] for row in shift_labels],
        device=shift_labels.device,
        dtype=torch.bool,
    )
    other_mask = valid & ~sid_mask & ~value_mask

    print(f"hf_loss={hf_loss.item():.6f}")
    print(f"custom_unweighted_loss={unweighted_custom_loss.item():.6f}")
    print(f"custom_weighted_loss={weighted_loss.item():.6f}")
    print(f"valid_tokens={valid.sum().item()} positive_weight_tokens={positive_weight.sum().item()}")
    print(f"weight_sum={(shift_weights * valid.float()).sum().item():.4f}")
    summarize("all_valid", token_loss[valid])
    summarize("positive_weight", token_loss[positive_weight])
    summarize("sid_tokens", token_loss[sid_mask])
    summarize("value_tokens", token_loss[value_mask])
    summarize("other_tokens", token_loss[other_mask])


if __name__ == "__main__":
    main()
