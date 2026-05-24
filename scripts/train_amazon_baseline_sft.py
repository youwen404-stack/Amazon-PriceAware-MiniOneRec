#!/usr/bin/env python3
"""Train Amazon MiniOneRec baseline SFT without price/value features."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from amazon_sft_data import build_baseline_train_dataset, load_and_tokenize_sequence_csv


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_sid_tokens(sid_index_path: str) -> list[str]:
    index = load_json(sid_index_path)
    return sorted({token for sid in index.values() for token in sid})


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_llm_except_new_tokens(model, original_vocab_size: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    embedding_layer = model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    def mask_grad(grad):
        grad[:original_vocab_size].zero_()
        return grad

    embedding_layer.weight.register_hook(mask_grad)


def enable_new_token_embedding_training(model, original_vocab_size: int) -> None:
    embedding_layer = model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    def mask_grad(grad):
        grad[:original_vocab_size].zero_()
        return grad

    embedding_layer.weight.register_hook(mask_grad)


def maybe_apply_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install it with: pip install peft") from exc

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    sample_tag = f"sample{args.train_sample}" if args.train_sample > 0 else "full"
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(
        Path(args.output_root)
        / "baseline_sft"
        / args.dataset_name
        / f"minionerec_baseline_{sample_tag}_seed{args.seed}_{run_tag}"
    )


def train(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

    set_seed(args.seed)
    output_dir = build_output_dir(args)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(output_dir) / "run_config.json", vars(args) | {"resolved_output_dir": output_dir})

    if not args.train_file or not args.eval_file or not args.item_meta_path or not args.sid_index_path:
        raise ValueError("train_file, eval_file, item_meta_path, and sid_index_path are required")

    item_meta = load_json(args.item_meta_path)
    sid_index = load_json(args.sid_index_path)
    sid_tokens = load_sid_tokens(args.sid_index_path)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map: Any = "auto"
    gradient_accumulation_steps = max(args.batch_size // args.micro_batch_size, 1)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    original_vocab_size = len(tokenizer)
    added = tokenizer.add_tokens(sid_tokens)
    print(f"SID tokens in index: {len(sid_tokens)}")
    print(f"Added trainable SID tokens: {added}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = maybe_apply_lora(model, args)
    if args.use_lora:
        enable_new_token_embedding_training(model, original_vocab_size)
    elif args.freeze_LLM:
        freeze_llm_except_new_tokens(model, original_vocab_size)

    cache_dir = args.cache_dir or str(Path(args.output_root) / "hf_cache" / args.dataset_name / "baseline_sft")
    metadata_sample = args.metadata_sample
    if metadata_sample == -1 and args.train_sample > 0:
        metadata_sample = args.train_sample
    fusion_sample = args.fusion_sample
    if fusion_sample == -1:
        fusion_sample = args.train_sample

    train_data = build_baseline_train_dataset(
        args.train_file,
        tokenizer=tokenizer,
        item_meta=item_meta,
        sid_index=sid_index,
        seed=args.seed,
        cutoff_len=args.cutoff_len,
        sequence_sample=args.train_sample,
        metadata_sample=metadata_sample,
        fusion_sample=fusion_sample,
        cache_dir=cache_dir,
    )
    eval_data = load_and_tokenize_sequence_csv(
        args.eval_file,
        tokenizer=tokenizer,
        item_meta=item_meta,
        seed=args.seed,
        cutoff_len=args.cutoff_len,
        sample=args.eval_sample,
        cache_dir=cache_dir,
    )
    print(train_data)
    print(eval_data)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            run_name=Path(output_dir).name,
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=output_dir,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            gradient_checkpointing=args.gradient_checkpointing,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=[],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(output_dir)
    final_dir = Path(output_dir) / "final_checkpoint"
    final_model = trainer.model
    if args.use_lora and hasattr(final_model, "merge_and_unload"):
        final_model = final_model.merge_and_unload()
    final_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final checkpoint to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Amazon MiniOneRec-aligned baseline SFT.")
    parser.add_argument("--base-model", default="/home/youwen/data/minionerec/models/Qwen2.5-1.5B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--item-meta-path", required=True)
    parser.add_argument("--sid-index-path", required=True)
    parser.add_argument("--dataset-name", default="Amazon_Automotive_priced_5core")
    parser.add_argument("--output-root", default="/home/youwen/data/rec/amazon_price_aware/outputs")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--train-sample", type=int, default=-1, help="Sequence-task rows: -1 full, 0 skip, >0 sample count.")
    parser.add_argument("--metadata-sample", type=int, default=-1, help="Metadata-task rows: -1 full/default, 0 skip, >0 sample count.")
    parser.add_argument("--fusion-sample", type=int, default=-1, help="Fusion-task rows: -1 full/default, 0 skip, >0 sample count.")
    parser.add_argument("--eval-sample", type=int, default=20000, help="Evaluation sequence rows: -1 full, >0 sample count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=float, default=0.05)
    parser.add_argument("--save-steps", type=float, default=0.05)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--freeze-LLM", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
