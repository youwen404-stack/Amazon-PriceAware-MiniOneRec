#!/usr/bin/env python3
"""Generate Qwen embeddings for MiniOneRec-style Amazon item metadata."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = "/home/youwen/data/minionerec/models/Qwen2.5-1.5B"


def load_json(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_items(item_json: str) -> List[Tuple[str, str]]:
    items = load_json(item_json)
    ordered_ids = sorted(items, key=lambda x: int(x))
    expected_ids = [str(i) for i in range(len(ordered_ids))]
    if ordered_ids != expected_ids:
        raise ValueError(
            "item_json keys must be contiguous internal item IDs from 0 to num_items-1; "
            f"first mismatch: {next((x for x in expected_ids if x not in items), 'unknown')}"
        )

    item_texts: List[Tuple[str, str]] = []
    for item_id in ordered_ids:
        item = items[item_id]
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        text = " ".join(part for part in [title, description] if part).strip()
        item_texts.append((item_id, text or "unknown item"))
    return item_texts


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).to(last_hidden.dtype)
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def generate_embeddings(args: argparse.Namespace) -> None:
    item_texts = load_items(args.item_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"items: {len(item_texts)}")
    print(f"model_path: {args.model_path}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(item_texts), args.batch_size):
            batch = item_texts[start : start + args.batch_size]
            texts = [text for _, text in batch]
            encoded = tokenizer(
                texts,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
            pooled = mean_pool(outputs.last_hidden_state, encoded.attention_mask)
            embeddings.append(pooled.float().cpu().numpy())
            print(f"embedded {min(start + len(batch), len(item_texts))}/{len(item_texts)}")

    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
    np.save(args.output_npy, matrix)
    write_json(args.output_ids_json, {str(row_idx): item_id for row_idx, (item_id, _) in enumerate(item_texts)})
    print(f"embedding shape: {matrix.shape}")
    print(f"saved embeddings: {args.output_npy}")
    print(f"saved row ids: {args.output_ids_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Amazon item embeddings with Qwen.")
    parser.add_argument("--item_json", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--output_ids_json", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    generate_embeddings(parse_args())


if __name__ == "__main__":
    main()
