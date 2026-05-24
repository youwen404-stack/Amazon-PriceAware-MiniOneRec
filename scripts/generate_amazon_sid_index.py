#!/usr/bin/env python3
"""Generate MiniOneRec-compatible SID index for Amazon embeddings."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


PREFIXES = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def import_minionerec_rq(minionerec_rq_dir: str):
    rq_dir = os.path.abspath(os.path.expanduser(minionerec_rq_dir))
    if not os.path.isdir(rq_dir):
        raise FileNotFoundError(f"MiniOneRec rq directory not found: {rq_dir}")
    sys.path.insert(0, rq_dir)
    from datasets import EmbDataset  # type: ignore
    from models.rqvae import RQVAE  # type: ignore
    from trainer import Trainer  # type: ignore

    return EmbDataset, RQVAE, Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_torch_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def find_best_collision_checkpoint(ckpt_dir: str) -> str:
    candidates: List[str] = []
    for root, _, files in os.walk(ckpt_dir):
        for file_name in files:
            if file_name == "best_collision_model.pth":
                candidates.append(os.path.join(root, file_name))
    if not candidates:
        raise FileNotFoundError(f"No best_collision_model.pth found under {ckpt_dir}")
    return max(candidates, key=os.path.getmtime)


def build_train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_step=args.eval_step,
        learner="AdamW",
        lr_scheduler_type="constant",
        warmup_epochs=args.warmup_epochs,
        data_path=args.embedding_npy,
        weight_decay=0.0,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        kmeans_init=True,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        device=args.device,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        quant_loss_weight=1.0,
        beta=0.25,
        layers=args.layers,
        save_limit=5,
        ckpt_dir=args.ckpt_dir,
    )


def sid_tokens(indices: np.ndarray) -> List[str]:
    return [PREFIXES[level].format(int(code)) for level, code in enumerate(indices)]


def check_collision(all_indices_str: List[str]) -> bool:
    return len(all_indices_str) == len(set(all_indices_str))


def get_collision_groups(all_indices_str: List[str]) -> List[List[int]]:
    index_to_items: Dict[str, List[int]] = collections.defaultdict(list)
    for item_idx, index in enumerate(all_indices_str):
        index_to_items[index].append(item_idx)
    return [items for items in index_to_items.values() if len(items) > 1]


def get_indices_count(all_indices_str: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = collections.defaultdict(int)
    for index in all_indices_str:
        counts[index] += 1
    return counts


def train_rqvae(args: argparse.Namespace, EmbDataset, RQVAE, Trainer) -> str:
    train_args = build_train_args(args)
    os.makedirs(train_args.ckpt_dir, exist_ok=True)
    data = EmbDataset(train_args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        beta=train_args.beta,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_epsilons=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
    )
    loader = DataLoader(
        data,
        num_workers=train_args.num_workers,
        batch_size=train_args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(train_args, model, len(loader))
    best_loss, best_collision_rate = trainer.fit(loader)
    print(f"best_loss: {best_loss}")
    print(f"best_collision_rate: {best_collision_rate}")
    ckpt_path = find_best_collision_checkpoint(train_args.ckpt_dir)
    print(f"using checkpoint: {ckpt_path}")
    return ckpt_path


def generate_sid_index(args: argparse.Namespace, EmbDataset, RQVAE) -> None:
    row_to_item_id = {int(row): str(item_id) for row, item_id in load_json(args.ids_json).items()}
    data = EmbDataset(args.embedding_npy)
    if set(row_to_item_id) != set(range(len(data))):
        raise ValueError("ids_json rows do not match embedding row count")

    ckpt_path = args.ckpt_path or find_best_collision_checkpoint(args.ckpt_dir)
    ckpt = load_torch_checkpoint(ckpt_path)
    ckpt_args = ckpt["args"]
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)
    all_indices: List[List[str]] = []
    all_indices_str: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            indices = model.get_indices(batch, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                tokens = sid_tokens(index)
                all_indices.append(tokens)
                all_indices_str.append(str(tokens))

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    rounds = 0
    while rounds < args.collision_retry and not check_collision(all_indices_str):
        collision_groups = get_collision_groups(all_indices_str)
        print(f"collision retry {rounds + 1}: groups={len(collision_groups)}")
        for collision_items in collision_groups:
            batch = data[collision_items].to(device)
            indices = model.get_indices(batch, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item_idx, index in zip(collision_items, indices):
                tokens = sid_tokens(index)
                all_indices[item_idx] = tokens
                all_indices_str[item_idx] = str(tokens)
        rounds += 1

    max_conflict = max(get_indices_count(all_indices_str).values()) if all_indices_str else 0
    collision_rate = (len(all_indices_str) - len(set(all_indices_str))) / max(len(all_indices_str), 1)
    print(f"all indices number: {len(all_indices)}")
    print(f"max number of conflicts: {max_conflict}")
    print(f"collision rate: {collision_rate}")

    output = {row_to_item_id[row_idx]: tokens for row_idx, tokens in enumerate(all_indices)}
    expected_item_ids = {str(i) for i in range(len(data))}
    if set(output) != expected_item_ids:
        raise ValueError("output SID keys are not contiguous internal item IDs")
    write_json(args.output_index_json, {item_id: output[item_id] for item_id in sorted(output, key=lambda x: int(x))})
    print(f"saved SID index: {args.output_index_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Amazon MiniOneRec-compatible SID index.")
    parser.add_argument("--embedding_npy", required=True)
    parser.add_argument("--ids_json", required=True)
    parser.add_argument("--output_index_json", required=True)
    parser.add_argument("--minionerec_rq_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--ckpt_path", default="", help="Optional existing RQ-VAE checkpoint. If omitted, train first.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--collision_retry", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    EmbDataset, RQVAE, Trainer = import_minionerec_rq(args.minionerec_rq_dir)
    if not args.ckpt_path:
        train_rqvae(args, EmbDataset, RQVAE, Trainer)
    generate_sid_index(args, EmbDataset, RQVAE)


if __name__ == "__main__":
    main()
