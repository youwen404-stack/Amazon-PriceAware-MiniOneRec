#!/usr/bin/env python3
"""Train Amazon value-aware SFT with GR4AD-style VSL weighting."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from amazon_value_aware_sft_data import (
    VALUE_TOKENS,
    build_value_aware_train_dataset,
    load_and_tokenize_value_sequence_csv,
)


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


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


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
    if is_main_process():
        model.print_trainable_parameters()
    return model


def build_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    sample_tag = f"sample{args.train_sample}" if args.train_sample > 0 else "full"
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(
        Path(args.output_root)
        / "value_aware_sft"
        / args.dataset_name
        / f"gr4ad_vsl_{sample_tag}_seed{args.seed}_{run_tag}"
    )


@dataclass
class ValueAwareDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __post_init__(self) -> None:
        import transformers

        self.base_collator = transformers.DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            padding=True,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        loss_weights = [feature["loss_weights"] for feature in features]
        model_features = [{k: v for k, v in feature.items() if k != "loss_weights"} for feature in features]
        batch = self.base_collator(model_features)
        max_len = batch["input_ids"].shape[1]
        padded_weights = []
        for weights in loss_weights:
            pad_len = max_len - len(weights)
            if self.tokenizer.padding_side == "left":
                padded = [0.0] * pad_len + list(weights)
            else:
                padded = list(weights) + [0.0] * pad_len
            padded_weights.append(padded)
        batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch


class ValueAwareTrainerMixin:
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch

        loss_weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous().to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)
        valid = (shift_labels != -100).float()
        weighted = token_loss * shift_weights * valid
        denom = (shift_weights * valid).sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        return (loss, outputs) if return_outputs else loss


def train(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

    class ValueAwareTrainer(ValueAwareTrainerMixin, transformers.Trainer):
        pass

    set_seed(args.seed)
    output_dir = build_output_dir(args)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        write_json(Path(output_dir) / "run_config.json", vars(args) | {"resolved_output_dir": output_dir})

    if not args.train_file or not args.eval_file or not args.item_meta_path or not args.sid_index_path:
        raise ValueError("train_file, eval_file, item_meta_path, and sid_index_path are required")

    item_meta = load_json(args.item_meta_path)
    sid_index = load_json(args.sid_index_path)
    sid_tokens = load_sid_tokens(args.sid_index_path)
    new_tokens = sorted(set(sid_tokens) | set(VALUE_TOKENS))

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
    added = tokenizer.add_tokens(new_tokens)
    main_print(f"SID tokens in index: {len(sid_tokens)}")
    main_print(f"Value tokens: {VALUE_TOKENS}")
    main_print(f"Added trainable SID/value tokens: {added}")

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

    cache_dir = args.cache_dir or str(Path(args.output_root) / "hf_cache" / args.dataset_name / "value_aware_sft")
    if ddp:
        cache_dir = str(Path(cache_dir) / f"rank_{os.environ.get('LOCAL_RANK') or 0}")
    metadata_sample = args.metadata_sample
    if metadata_sample == -1 and args.train_sample > 0:
        metadata_sample = args.train_sample
    fusion_sample = args.fusion_sample
    if fusion_sample == -1:
        fusion_sample = args.train_sample

    train_data = build_value_aware_train_dataset(
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
        value_lambda=args.value_lambda,
        value_weight_scheme=args.value_weight_scheme,
        value_weight_min=args.value_weight_min,
        value_weight_max=args.value_weight_max,
    )
    eval_data = load_and_tokenize_value_sequence_csv(
        args.eval_file,
        tokenizer=tokenizer,
        seed=args.seed,
        cutoff_len=args.cutoff_len,
        sample=args.eval_sample,
        cache_dir=cache_dir,
        value_lambda=args.value_lambda,
        value_weight_scheme=args.value_weight_scheme,
        value_weight_min=args.value_weight_min,
        value_weight_max=args.value_weight_max,
    )
    main_print(train_data)
    main_print(eval_data)

    trainer = ValueAwareTrainer(
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
            remove_unused_columns=False,
            gradient_checkpointing=args.gradient_checkpointing,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=[],
        ),
        data_collator=ValueAwareDataCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.model_accepts_loss_kwargs = False
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(output_dir)
    final_dir = Path(output_dir) / "final_checkpoint"
    final_model = trainer.model
    if args.use_lora and hasattr(final_model, "merge_and_unload"):
        final_model = final_model.merge_and_unload()
    final_model.save_pretrained(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
    main_print(f"Saved final checkpoint to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Amazon GR4AD-style value-aware SFT.")
    parser.add_argument("--base-model", default="models/Qwen2.5-1.5B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--item-meta-path", required=True)
    parser.add_argument("--sid-index-path", required=True)
    parser.add_argument("--dataset-name", default="Amazon_Automotive_priced_5core")
    parser.add_argument("--output-root", default="outputs")
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
    parser.add_argument("--value-lambda", type=float, default=1.0)
    parser.add_argument("--value-weight-scheme", choices=["uniform", "bucket_linear"], default="bucket_linear")
    parser.add_argument("--value-weight-min", type=float, default=1.0)
    parser.add_argument("--value-weight-max", type=float, default=2.0)
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
