# Amazon Price-Aware MiniOneRec 实验总结

## 1. 项目背景

这个实验从 MiniOneRec 的生成式推荐框架出发，尝试引入快手广告推荐论文 *Generative Recommendation for Large-Scale Advertising* 中的 value-aware SFT 思路。原始 MiniOneRec 使用商品语义 ID（Semantic ID, SID）把推荐任务转化为语言模型生成任务，但项目自带数据缺少商品价格或商业价值字段，因此无法直接验证 value-aware SFT 是否能提升推荐结果的商业价值。

为了解决这个问题，我们将数据源切换到 Amazon Reviews 2023。Amazon 商品 metadata 中包含 price 字段，可以把商品价格作为 value proxy，并构造 value bucket 和 value token。实验目标不是重新设计 MiniOneRec，而是在尽量保留 MiniOneRec 原始训练方式的前提下，比较普通 MiniOneRec SFT baseline 和 value-aware SFT 的效果差异。

## 2. 数据选择

我们首先审计了多个 Amazon 类目，包括 Office Products、All Beauty、Industrial and Scientific、Automotive、Toys and Games、Sports and Outdoors、Tools and Home Improvement 等。审计指标包括 review 数量、metadata 覆盖率、price 覆盖率、有效价格比例、标题和描述覆盖率，以及 priced interaction 经过 k-core 后的剩余规模。

最终选择 `Industrial_and_Scientific` 作为主要实验类目，原因是：

- price 覆盖率较高，priced review row ratio 约为 74.9%；
- 5-core 后仍保留足够交互，适合训练生成式推荐模型；
- 相比 Automotive，item 数量和训练规模更可控；
- 商品价格分布合理，可以构造稳定的 value bucket。

在 full setting 中，`Amazon_Industrial_and_Scientific_priced_5core` 的核心规模为：

| 指标 | 数值 |
|---|---:|
| Priced interactions | 3,882,902 |
| 5-core interactions | 285,699 |
| 5-core users | 36,113 |
| 5-core items | 17,348 |
| Train examples | 106,380 |
| Valid examples | 36,113 |
| Test examples | 23,598 |

由于 full setting 的 item 候选空间仍明显大于 MiniOneRec 原始 Amazon 数据，我们进一步构建了一个 MiniOneRec-scale mini 数据集，用于更接近原始 MiniOneRec 的实验环境。

## 3. 数据处理流程

### 3.1 Full 数据处理

full 数据处理流程如下：

1. 读取 Amazon Reviews 2023 的 review jsonl 和 metadata jsonl。
2. 只保留 metadata 中存在有效正价格的商品。
3. 只保留 target item 有价格的 review interaction。
4. 在 priced interactions 上执行 5-core 过滤。
5. 对保留下来的 user/item 重新编号，生成连续内部 ID。
6. 按用户时间序列构造 train/valid/test examples。
7. 使用商品价格的 25%、50%、75% 分位数构造 4 个 value bucket。
8. 为每个 item 写入 `price_value`、`value_bucket`、`value_token`。
9. 输出 MiniOneRec 兼容的 `inter`、`value_splits`、`item.json`、user/item map。

这里最重要的设计点是：所有 split 都在过滤和 k-core 之后重新生成，避免出现历史序列中 item 缺失、metadata 不一致或 target item 不存在的问题。

### 3.2 Mini 数据处理

在 full Industrial 数据上，baseline 的 train/test HR 都较低，说明 17k item 的候选空间对 freeze LLM 的 MiniOneRec baseline 偏难。因此我们构造了 `Amazon_Industrial_and_Scientific_priced_5core_mini5000`。

mini 数据不是简单随机抽样，而是采用 popular-subgraph 策略：

1. 从 priced interactions 中统计 item 交互频次。
2. 选取 top 5000 popular priced items。
3. 过滤出这些 item 相关的交互。
4. 对过滤后的交互重新执行 5-core。
5. 重新生成 user/item ID、price bucket、train/valid/test split。
6. 后续仍然复用相同的 SID 生成、训练和评估 pipeline。

mini5000 最终规模为：

| 指标 | 数值 |
|---|---:|
| Pre-kcore interactions | 2,126,061 |
| Pre-kcore items | 5,000 |
| 5-core interactions | 86,655 |
| 5-core users | 12,875 |
| 5-core items | 3,593 |
| Train examples | 26,418 |
| Valid examples | 12,875 |
| Test examples | 7,424 |

这个规模与 MiniOneRec 原始 Amazon 数据更加接近。第一次 SID 生成后 collision rate 约 7.1%，偏高。我们重新训练 RQ-VAE/SID index，将 collision rate 降到约 0.1%，避免多个商品共享同一 SID 对评估上限造成明显影响。

## 4. SID 生成与 MiniOneRec 对齐

我们沿用 MiniOneRec 的 semantic ID 思路：

1. 使用 Qwen2.5-1.5B 对 item title/description 编码，生成 item embedding。
2. 使用 MiniOneRec RQ-VAE 模块对 embedding 进行 residual quantization。
3. 将每个 item 映射成多层 code token，例如 `<a_34><b_87><c_2>`。
4. 将 SID 写回 train/valid/test CSV，生成 `history_item_sid` 和 `item_sid`。
5. 评估时使用合法 SID trie 做 constrained generation，保证生成输出落在 item SID 空间内。

这一部分尽量保持和 MiniOneRec 原项目一致，避免把数据侧创新和模型侧创新混在一起。

## 5. Baseline 设计

baseline 的原则是忠于 MiniOneRec，而不是为了我们的 value-aware 方法人为削弱 baseline。baseline 使用 MiniOneRec 风格的三类 SFT 数据：

- sequence task：给定用户历史 SID，预测下一个 item SID；
- metadata task：title/description 到 SID，以及 SID 到 title；
- fusion task：给定历史 SID，预测下一个商品标题。

训练设置：

| 参数 | 设置 |
|---|---|
| Base model | Qwen2.5-1.5B |
| Fine-tuning | Freeze LLM |
| Trainable params | 新增 SID token embedding |
| Epoch | 1 |
| Batch size | 128 |
| Micro batch size | 4 |
| Learning rate | 5e-5 |
| Eval/save interval | 0.25 epoch |

freeze LLM 的设置保持 MiniOneRec baseline 风格：冻结主干模型，仅训练新增 SID token embedding。

## 6. Value-Aware SFT 设计

value-aware SFT 参考 *Generative Recommendation for Large-Scale Advertising* 的 VSL 思路，在 MiniOneRec baseline 的任务集合上加入 value signal。

核心设计包括：

- 使用商品价格作为 value proxy；
- 按价格分位数划分 4 个 bucket；
- 为每个 bucket 分配 value token：`[VAL_0]`、`[VAL_1]`、`[VAL_2]`、`[VAL_3]`；
- sequence target 从 `SID` 扩展为 `SID + value_token`；
- title2sid metadata task 也预测 `SID + value_token`；
- sid2title 和 fusion task 保持 baseline 形式；
- 使用 sample-level value weight，让高 value bucket 的样本获得更高训练权重。

我们保留 baseline 的三类任务，是为了让对比只体现 value-aware SFT 的增量，而不是改变 MiniOneRec 原有训练任务。

### 6.1 Loss Debug 与修正

训练过程中曾出现 value-aware loss 异常放大到 100+ 的问题。通过 batch-level 诊断，我们发现真实 batch loss 约为 5.45，但 Trainer 日志约为 180，正好接近梯度累积步数 32 倍。根因是自定义 `compute_loss` 与新版 HuggingFace Trainer 的 gradient accumulation/loss kwargs 缩放机制不对齐。

修正方式：

- 在自定义 loss 中不把 labels 传入 model 内部 loss；
- 使用 float32 logits 计算 token-level CE；
- 设置 `trainer.model_accepts_loss_kwargs = False`，避免 Trainer 将自定义 loss 按 `num_items_in_batch` 逻辑重复缩放。

修正后 value-aware SFT 的 loss 回到 5-7 左右，与 baseline 同量级，训练过程恢复正常。

### 6.2 参数调整

在 mini5000 上，我们比较了两组 value-aware 强度：

| 版本 | VALUE_LAMBDA | VALUE_WEIGHT_MAX | 现象 |
|---|---:|---:|---|
| v1 | 1.0 | 2.0 | value 指标提升明显，但 top-10 accuracy 有 tradeoff |
| v2 | 0.3 | 1.5 | accuracy 和 value 更平衡，作为 mini 主结果 |

最终 mini5000 主结果采用 v2：`VALUE_LAMBDA=0.3, VALUE_WEIGHT_MAX=1.5`。

## 7. 评估方法

评估保持 MiniOneRec-style constrained generation：

- 输入用户历史 SID；
- 生成 top-K SID candidates；
- 使用合法 SID trie 约束输出；
- 计算 HR@K、NDCG@K；
- 同时计算 value-aware 指标：
  - AvgPrice@K：推荐列表平均价格；
  - AvgValueBucket@K：推荐列表平均 value bucket；
  - HitAvgPrice@K：命中样本中的目标商品平均价格；
  - HitAvgValueBucket@K：命中样本中的目标商品平均 value bucket。

HR/NDCG 衡量推荐准确率，price/value bucket 指标衡量推荐结果的商业价值倾向。

## 8. 实验结果

### 8.1 Full Industrial 结果

在 `Amazon_Industrial_and_Scientific_priced_5core` 上，value-aware SFT 同时提升了准确率指标和价值指标。

| Metric | Baseline | Value-aware | 相对提升 |
|---|---:|---:|---:|
| NDCG@20 | 0.00207 | 0.00791 | +282.8% |
| HR@20 | 0.00585 | 0.01381 | +136.2% |
| AvgPrice@20 | 23.38 | 27.76 | +18.7% |
| AvgValueBucket@20 | 1.632 | 1.979 | +21.3% |
| HitAvgPrice@20 | 24.61 | 35.73 | +45.2% |
| HitAvgValueBucket@20 | 1.841 | 2.120 | +15.2% |

这个结果说明，在较大候选空间下，value-aware SFT 不只是提高了高价商品曝光倾向，也改善了 SID 生成推荐的命中效果。

### 8.2 Mini5000 结果

在更接近 MiniOneRec 原始规模的 mini5000 setting 中，baseline 效果明显强于 full setting。Mini5000 baseline 的 HR@20 达到 0.03462，约为 full baseline 的 5.9 倍。

mini5000 主结果采用温和 value-aware 配置：

| Metric | Baseline | Value-aware v2 | 相对变化 |
|---|---:|---:|---:|
| NDCG@10 | 0.00662 | 0.00866 | +30.9% |
| NDCG@20 | 0.01199 | 0.01238 | +3.2% |
| HR@10 | 0.01374 | 0.02047 | +49.0% |
| HR@20 | 0.03462 | 0.03475 | +0.4% |
| AvgPrice@10 | 21.42 | 43.36 | +102.4% |
| AvgPrice@20 | 22.26 | 34.15 | +53.4% |
| AvgValueBucket@20 | 1.796 | 1.881 | +4.7% |
| HitAvgPrice@20 | 23.80 | 26.49 | +11.3% |
| HitAvgValueBucket@20 | 1.949 | 2.047 | +5.0% |

mini5000 的结果更适合作为主实验展示：在更合理的 MiniOneRec-scale 数据规模下，value-aware SFT 保持 HR@20 基本不变，同时显著提升 top-10 准确率和推荐列表价格水平。相比 v1，v2 证明 value signal 强度需要调节，过强时会带来 top-K tradeoff，温和权重能获得更平衡的结果。

### 8.3 Qwen2.5-7B 扩展实验与 LoRA 诊断

为了验证模型规模是否能进一步提升生成式推荐效果，我们将 backbone 从 `Qwen2.5-1.5B` 扩展到 `Qwen2.5-7B`。在保持 freeze LLM、仅训练新增 SID token embedding 的设置下，7B 两版参数实验都明显低于 1.5B baseline。即使第二版将 `cutoff_len` 提高到 512、learning rate 提高到 `5e-5`，训练 loss 和 eval loss 更低，test HR/NDCG 仍然没有改善。

这说明在 SID-based generative recommendation 中，大模型规模优势并不会自动转化为推荐排序能力。SID 是新引入的离散 token，7B 预训练得到的自然语言能力无法直接理解 SID 序列；freeze-only 只训练新增 token embedding，无法调整 attention/hidden representation 来适配用户历史 SID 到目标 SID 的映射。

进一步地，我们加入 LoRA，在 `q_proj,k_proj,v_proj,o_proj` 上做低秩适配，同时继续训练新增 SID token embedding。7B LoRA baseline 相比 7B freeze-only 有提升，例如 HR@20 相对 freeze v1 提升约 40%，说明 freeze-only 的确存在适配不足。但当前 LoRA 设置仍低于 1.5B freeze baseline，表明在 mini5000 小数据规模下，轻量 LoRA 还不足以充分释放 7B 能力。

| 实验 | NDCG@10 | NDCG@20 | HR@10 | HR@20 | AvgPrice@20 |
|---|---:|---:|---:|---:|---:|
| 1.5B freeze baseline | 0.00662 | 0.01199 | 0.01374 | 0.03462 | 22.26 |
| 7B freeze v1 | 0.00123 | 0.00167 | 0.00256 | 0.00431 | 38.91 |
| 7B freeze v2 | 0.00082 | 0.00126 | 0.00162 | 0.00337 | 31.19 |
| 7B LoRA baseline | 0.00137 | 0.00215 | 0.00296 | 0.00606 | 35.27 |

这组实验的结论是：推荐任务中的模型规模收益依赖训练策略、数据规模和 SID 表示方式。后续可以从 LoRA rank、target modules、训练轮数和数据规模四个方向继续验证 7B 是否能在更充分适配后超过 1.5B baseline。

### 8.4 1.5B Full-SFT 与辅助任务比例诊断

在 mini5000 上，我们进一步尝试了 Qwen2.5-1.5B 的全参数 SFT。最初直接保留 MiniOneRec 三类任务的完整比例时，效果并不理想。分析后发现，mini5000 的主 sequence task 只有 26,418 条，而 fusion task 也有 26,418 条，metadata task 约 7,186 条。辅助任务占比过高时，token-level SFT loss 容易被 metadata/fusion 任务主导，模型更擅长商品语义互译，但不一定更擅长用户历史到下一商品 SID 的推荐目标。

因此我们将 SFT 数据构造脚本改为支持独立指定三类任务样本数：

```text
--train-sample      控制 sequence task
--metadata-sample   控制 metadata task
--fusion-sample     控制 fusion task

-1 = 全量
 0 = 跳过
>0 = 指定采样条数
```

在 mini5000 上使用 `sequence full + metadata 3000 + fusion 3000` 后，全参 SFT baseline 明显恢复：

| 实验 | NDCG@10 | HR@10 | NDCG@20 | HR@20 |
|---|---:|---:|---:|---:|
| 1.5B freeze baseline | 0.00662 | 0.01374 | 0.01199 | 0.03462 |
| 1.5B full-SFT 原辅助比例 | 0.00635 | 0.01280 | 0.01070 | 0.03004 |
| 1.5B full-SFT aux3000/3000 | 0.01017 | 0.02196 | 0.01298 | 0.03327 |

相对原 full-SFT baseline，aux-light 设置将 HR@10 从 0.01280 提升到 0.02196，约 +71.6%。这说明 full-SFT 本身并非不可行，关键是推荐主任务与辅助任务的比例需要控制。

### 8.5 Full 数据全参 SFT 的负结果与解释

在 full Industrial 数据上，我们也复用了 `sequence full + metadata 3000 + fusion 3000` 的 1.5B 全参 SFT 设置，用于验证数据量增大是否能改善 baseline。结果并不理想：

| Full Industrial baseline | NDCG@10 | HR@10 | NDCG@20 | HR@20 | Eval loss |
|---|---:|---:|---:|---:|---:|
| 1.5B freeze baseline | 0.00124 | 0.00250 | 0.00207 | 0.00585 | - |
| 1.5B full-SFT aux3000/3000, 1 epoch | 0.00096 | 0.00212 | 0.00147 | 0.00415 | 3.706 |
| 1.5B full-SFT aux3000/3000, 3 epoch | 0.00079 | 0.00157 | 0.00138 | 0.00386 | 3.663 |

3 epoch 继续训练时，train loss 从 3.89 降到 2.34，eval loss 也小幅下降，但 HR/NDCG 反而下降。这说明问题不是简单“没训练够”，而是 token-level SFT loss 和 top-K 推荐目标出现错配。

我们的解释是：

- full 数据 item 数从 mini5000 的 3,593 扩大到 17,348，SID 候选空间扩大约 4.8 倍；
- beam=20 下，真实 item 进入 top-K 的难度显著上升；
- CE loss 下降可能代表模型更会生成合法/高频 SID，不等价于真实目标 item 排名更靠前；
- full 数据长尾更强，更多训练样本并不必然意味着每个 item/SID 的有效监督更密；
- 全参更新会改变底座分布，在复杂 SID 空间中可能强化热门模式而非个性化 next-item ranking。

这个负结果反而帮助明确了后续方向：full 数据上不能只靠更多 epoch 或更大数据，需要更贴近排序目标的训练信号，例如 value-aware weighting、LoRA 稳定适配、hard negative 或 popularity/long-tail reweighting。

### 8.6 1.5B LoRA 主结果

在确认 full-SFT 存在分布扰动和目标错配后，我们尝试 1.5B LoRA。LoRA 在 `q_proj,k_proj,v_proj,o_proj` 上训练低秩适配参数，同时训练新增 SID token embedding。它介于 freeze 和 full-SFT 之间：比 freeze 有更强任务适配能力，又比 full-SFT 更不容易破坏底座语言模型分布。

mini5000 上，1.5B LoRA baseline 已经显著超过 freeze 和 full-SFT：

| Mini5000 实验 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | HitAvgPrice@20 |
|---|---:|---:|---:|---:|---:|
| 1.5B freeze baseline | 0.00662 | 0.01374 | 0.01199 | 0.03462 | 23.80 |
| 1.5B full-SFT aux3000/3000 | 0.01017 | 0.02196 | 0.01298 | 0.03327 | 27.54 |
| 1.5B LoRA baseline | 0.03622 | 0.06008 | 0.04325 | 0.08796 | 31.28 |
| 1.5B LoRA value-aware | 0.03792 | 0.06587 | 0.04454 | 0.09213 | 32.53 |

与 baseline LoRA 相比，value-aware LoRA 进一步带来：

| Metric | Baseline LoRA | Value-aware LoRA | 相对变化 |
|---|---:|---:|---:|
| NDCG@10 | 0.03622 | 0.03792 | +4.7% |
| HR@10 | 0.06008 | 0.06587 | +9.6% |
| NDCG@20 | 0.04325 | 0.04454 | +3.0% |
| HR@20 | 0.08796 | 0.09213 | +4.7% |
| HitAvgPrice@20 | 31.28 | 32.53 | +4.0% |

这个结果非常关键：value-aware LoRA 没有牺牲推荐相关性，反而同时提升 HR/NDCG 和命中商品价格。它说明 value signal 在参数高效微调设置下可以作为有效的推荐监督，而不是简单把模型推向高价但不相关商品。

### 8.7 当前进行中的 7B Full LoRA 实验

基于 1.5B LoRA 在 mini5000 上的强结果，我们进一步启动了 7B LoRA full Industrial 实验，用于验证更大模型容量是否能缓解 full 数据中 17k item/SID 候选空间带来的困难。当前并行运行两组：

- `Qwen2.5-7B + baseline LoRA + full Industrial + aux3000/3000 + lr=5e-5`；
- `Qwen2.5-7B + value-aware LoRA + full Industrial + aux3000/3000 + lr=5e-5 + value_lambda=0.3`。

这两组实验尚未完成，预计用于回答：更大模型容量是否能在 full 数据规模下释放收益，以及 value-aware 目标是否在更大 backbone 和更大 SID 空间下仍然有效。

## 9. 项目摘要

这个项目可以概括为：

我基于 MiniOneRec 做了一个 price-aware generative recommendation 实验。原始 MiniOneRec 数据没有 price 信息，所以无法验证快手广告生成式推荐论文里的 value-aware SFT 思路。我改用 Amazon Reviews 2023 数据，因为它有商品价格字段，可以把价格作为商业价值 proxy。

我先做了完整的数据审计，包括 price 覆盖率、metadata 匹配率、priced interaction 数量和 k-core 后规模，最后选择 Industrial and Scientific 类目。数据处理上，我只保留有有效价格的商品和交互，重新做 5-core，按用户时间序列生成 train/valid/test，并用价格分位数构造 4 个 value bucket。

为了对齐 MiniOneRec，我保留了原来的 SID 生成流程：用 Qwen 对商品文本生成 embedding，再用 RQ-VAE 生成 semantic ID，然后用 constrained generation 做评估。baseline 也尽量忠于 MiniOneRec，包括 sequence、metadata 和 fusion 三类 SFT 任务。

创新部分是 value-aware SFT。我把 target 从单纯预测 SID 扩展为预测 `SID + value token`，并对不同 value bucket 的样本设置不同 loss weight，让模型在生成推荐时感知商品价值。训练中我还定位并修复了一个自定义 loss 与 HuggingFace Trainer 梯度累积缩放不兼容的问题，避免 loss 被放大 32 倍。

实验上，我做了 full Industrial 和 MiniOneRec-scale mini5000 两套设置。早期 freeze baseline 下，full setting 上 value-aware SFT 相比 baseline 将 HR@20 提升 136.2%，NDCG@20 提升 282.8%。随后我进一步做训练策略诊断，发现全参 SFT 在 mini5000 上需要控制辅助任务比例，`sequence full + aux3000/3000` 可将 HR@10 提升到 0.02196；而在 full 数据上，继续训练到 3 epoch 虽然降低了 eval loss，但 HR/NDCG 反而下降，说明 token-level loss 和推荐 top-K 目标存在错配。最终在 mini5000 上使用 1.5B LoRA 后，baseline HR@10 达到 0.06008；加入 value-aware LoRA 后 HR@10 进一步达到 0.06587，HR@20 达到 0.09213，HitAvgPrice@20 提升约 4.0%。这说明 LoRA 是当前最稳定的适配策略，而 value-aware 目标在 LoRA 设置下仍能进一步提升推荐准确率和命中商品价值。

## 10. 简洁项目描述

较完整的项目描述：

> 基于 MiniOneRec 构建 Amazon price-aware 生成式推荐实验，将 Amazon Reviews 2023 商品价格引入 SID-based LLM 推荐框架，设计 `SID + value token` 的 value-aware SFT 目标与分桶加权 loss；完成 raw data audit、priced 5-core filtering、RQ-VAE semantic ID 生成、constrained decoding 评估与训练策略消融。系统比较 freeze、full-SFT、LoRA 与 value-aware LoRA，在 mini5000 数据上 LoRA baseline 达到 HR@10 0.06008，加入 value-aware 后 HR@10/HR@20 提升至 0.06587/0.09213，较 LoRA baseline 分别提升 9.6%/4.7%，HitAvgPrice@20 提升 4.0%；在 Industrial full setting 上，freeze value-aware 相比 baseline HR@20/NDCG@20 分别提升 136.2%/282.8%。

更短的项目描述：

> 构建 Amazon price-aware MiniOneRec 实验，引入 `SID + value token` 与 value-weighted SFT，并比较 freeze/full-SFT/LoRA 训练策略；mini5000 上 1.5B value-aware LoRA 相比 LoRA baseline 将 HR@10/HR@20 提升 9.6%/4.7%，HitAvgPrice@20 提升 4.0%；full Industrial 上 freeze value-aware 相比 baseline 将 HR@20/NDCG@20 提升 136.2%/282.8%。
