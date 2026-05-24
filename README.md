# Amazon Price-Aware MiniOneRec

> Price-aware generative recommendation with Semantic IDs, MiniOneRec-style SFT, and value-weighted LoRA fine-tuning.

This repository extends the **MiniOneRec** generative recommendation pipeline to a price-aware Amazon product recommendation setting. The core idea is simple: MiniOneRec turns recommendation into language-model generation over item Semantic IDs, but its original public data does not contain explicit item value signals. We therefore build an Amazon Reviews 2023 experiment line where product `price` is used as a public-data value proxy, then study whether a generative recommender can learn both user preference and item value.

The project includes:

- Amazon Reviews 2023 raw-data auditing and priced interaction filtering.
- Full and MiniOneRec-scale dataset construction with priced 5-core filtering.
- Qwen embedding + RQ-VAE Semantic ID generation.
- MiniOneRec-aligned baseline SFT with sequence, metadata, and fusion tasks.
- Value-aware SFT with `SID + value token` targets and price-bucket loss weighting.
- Freeze, full-SFT, LoRA, and value-aware LoRA training strategy comparisons.
- Constrained SID decoding evaluation with HR/NDCG and value-oriented metrics.

Raw data, model checkpoints, embeddings, and generated large artifacts are intentionally not included in this repository.

## Highlights

### Best Mini5000 Result

On the MiniOneRec-scale `Amazon_Industrial_and_Scientific_priced_5core_mini5000` setting, LoRA is the strongest training strategy. Adding value-aware supervision further improves both recommendation accuracy and hit-item value.

| Method | NDCG@10 | HR@10 | NDCG@20 | HR@20 | HitAvgPrice@20 |
|---|---:|---:|---:|---:|---:|
| 1.5B freeze baseline | 0.00662 | 0.01374 | 0.01199 | 0.03462 | 23.80 |
| 1.5B full-SFT aux-light | 0.01017 | 0.02196 | 0.01298 | 0.03327 | 27.54 |
| 1.5B LoRA baseline | 0.03622 | 0.06008 | 0.04325 | 0.08796 | 31.28 |
| **1.5B value-aware LoRA** | **0.03792** | **0.06587** | **0.04454** | **0.09213** | **32.53** |

Compared with the LoRA baseline, value-aware LoRA improves:

- `HR@10` by **9.6%**
- `HR@20` by **4.7%**
- `NDCG@10` by **4.7%**
- `HitAvgPrice@20` by **4.0%**

### Full Industrial Diagnostic Result

On the larger full Industrial setting, simply increasing data scale or training epochs does not automatically improve top-K recommendation. Full-parameter SFT lowers token-level loss but can degrade HR/NDCG, showing a mismatch between language-model CE loss and recommendation ranking quality.

| Full Industrial baseline | NDCG@10 | HR@10 | NDCG@20 | HR@20 | Eval loss |
|---|---:|---:|---:|---:|---:|
| 1.5B full-SFT aux-light, 1 epoch | 0.00096 | 0.00212 | 0.00147 | 0.00415 | 3.706 |
| 1.5B full-SFT aux-light, 3 epoch | 0.00079 | 0.00157 | 0.00138 | 0.00386 | 3.663 |

This motivates the value-aware and LoRA experiments: SID-based generative recommendation needs stable adaptation and training signals closer to the final ranking objective.

## Method Overview

```text
Amazon Reviews 2023
        |
        v
Raw review + metadata audit
        |
        v
Priced interaction filtering + 5-core
        |
        v
Chronological train / valid / test sequences
        |
        v
Qwen item embeddings -> RQ-VAE -> Semantic IDs
        |
        v
MiniOneRec-style SFT data
        |
        +--> Baseline target:      next item SID
        |
        +--> Value-aware target:   next item SID + [VAL_i]
        |
        v
Constrained SID decoding evaluation
```

## Value-Aware SFT

Each item has a price-derived value bucket:

```text
[VAL_0], [VAL_1], [VAL_2], [VAL_3]
```

For sequence recommendation, the ordinary target:

```text
history SID sequence -> target SID
```

is extended to:

```text
history SID sequence -> target SID + value token
```

The value-aware objective has two components:

- **Value token prediction**: the model learns the value bucket associated with the target item.
- **Sample-level value weighting**: higher-value bucket samples receive larger loss weights.

This is not post-hoc reranking. The value signal is injected during supervised fine-tuning so that the model generation distribution itself becomes value-aware.

## Dataset Settings

The main category is `Industrial_and_Scientific` from Amazon Reviews 2023.

| Dataset | Items | Train | Valid | Test | Purpose |
|---|---:|---:|---:|---:|---|
| Full Industrial | 17,348 | 106,380 | 36,113 | 23,598 | Larger candidate-space diagnosis |
| Mini5000 | 3,593 | 26,418 | 12,875 | 7,424 | MiniOneRec-scale main experiment |

Mini5000 is not random sampling. It is built by selecting top popular priced items, reconstructing the interaction subgraph, re-running 5-core, and regenerating user/item IDs and splits. This avoids broken user histories or missing target items.

## Repository Structure

```text
.
├── scripts/
│   ├── audit_amazon_raw.py                  # Raw review/metadata audit
│   ├── audit_priced_kcore.py                # Category-scale priced k-core audit
│   ├── convert_amazon_to_minionerec.py      # Full dataset conversion
│   ├── convert_amazon_to_minionerec_mini.py # Mini5000 conversion
│   ├── generate_amazon_embeddings.py        # Item text embedding generation
│   ├── generate_amazon_sid_index.py         # RQ-VAE SID index generation
│   ├── export_amazon_minionerec_with_sid.py # Write SID-augmented CSVs
│   ├── amazon_sft_data.py                   # Baseline SFT task construction
│   ├── amazon_value_aware_sft_data.py       # Value-aware SFT task construction
│   ├── train_amazon_baseline_sft.py         # Baseline freeze/full/LoRA SFT
│   ├── train_amazon_value_aware_sft.py      # Value-aware freeze/full/LoRA SFT
│   └── evaluate_amazon_sft.py               # Constrained SID decoding evaluation
├── tests/                                   # Unit tests for data and training utilities
└── docs/
    └── AMAZON_PRICE_AWARE_MINIONEREC_FINAL_REPORT_ZH.md
```

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run unit tests:

```bash
pytest tests
```

Audit raw Amazon files:

```bash
python scripts/audit_amazon_raw.py \
  --reviews /path/to/reviews.jsonl \
  --meta /path/to/meta.jsonl
```

Convert a priced 5-core dataset:

```bash
python scripts/convert_amazon_to_minionerec.py \
  --category Industrial_and_Scientific \
  --reviews /path/to/reviews.jsonl \
  --meta /path/to/meta.jsonl \
  --output-root /path/to/processed_price_aware
```

Train a MiniOneRec-style baseline:

```bash
python scripts/train_amazon_baseline_sft.py \
  --base-model /path/to/Qwen2.5-1.5B \
  --train-file /path/to/train.csv \
  --eval-file /path/to/valid.csv \
  --item-meta-path /path/to/item.json \
  --sid-index-path /path/to/index.json \
  --dataset-name Amazon_Industrial_and_Scientific_priced_5core_mini5000 \
  --output-root /path/to/outputs \
  --train-sample -1 \
  --metadata-sample 3000 \
  --fusion-sample 3000 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj
```

Evaluate with constrained SID decoding:

```bash
python scripts/evaluate_amazon_sft.py \
  --base-model /path/to/final_checkpoint \
  --info-file /path/to/info.txt \
  --test-data-path /path/to/test.csv \
  --item-meta-path /path/to/item.json \
  --result-json-data /path/to/predictions.json \
  --metrics-json-data /path/to/metrics.json \
  --num-beams 20
```

## Notes on Reproducibility

- This repository does not redistribute Amazon Reviews 2023 data, model checkpoints, generated embeddings, or RQ-VAE weights.
- Server-specific paths in the original experiments should be replaced with your local paths.
- For large 7B evaluations, use small evaluation batch sizes, for example `--batch-size 1 --cutoff-len 512`, to avoid generation-time KV-cache OOM.
- The scripts are designed to preserve MiniOneRec-style task families while adding explicit value-aware targets and weights.

## Acknowledgements

This work is built on and inspired by **MiniOneRec**, which introduced a compact generative recommendation pipeline based on Semantic IDs and LLM-style item generation. We sincerely thank the MiniOneRec authors and contributors for releasing their project and making this line of experimentation possible.

This project also uses ideas inspired by value-aware supervised fine-tuning in industrial generative recommendation, especially the motivation of optimizing recommendation outputs beyond pure next-item accuracy.

## Citation

If you use this repository, please cite or acknowledge the original MiniOneRec project and the Amazon Reviews 2023 dataset source according to their respective licenses and citation guidelines.
