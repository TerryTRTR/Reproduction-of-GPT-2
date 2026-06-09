# CS182 课程项目方案 —— 复现并调优 nanoGPT (Track C)

> 目标：复现 Karpathy 的 nanoGPT（GPT-2 架构），在单张 8–12G 消费级 GPU 上做**系统化调优 + 严格量化对比**，而非追求 SOTA。
> 评分对齐：方法应用 25% · 实验设计与分析 30% · 报告 20% · 答辩 15% · 代码与可复现性 10%。

---

## 0. 当前仓库状态（2026-06-09）

本仓库目前已经合并三条 baseline / 主方法代码：

| 模块 | 路径 | 当前状态 |
|---|---|---|
| N-gram baseline | `ngram/` | 已实现 3-gram 评测，已有 val/test PPL 结果 |
| LSTM baseline | `LSTM_baseline/` | 已实现约 14M 参数 LSTM，已复跑得到可比结果 |
| nanoGPT | `nanogpt/` | 已实现 GPT-style Transformer，已有 14M 与 44M WikiText-2 checkpoint 结果 |

当前重要文档：

| 文档 | 用途 |
|---|---|
| `BASELINE_COMPARISON.md` | 三方 baseline 指标、同 prompt 输出样例、当前预处理差异提醒 |
| `OPTIMIZATION_TASKS.md` | 后续三人共同优化 nanoGPT 的详细任务分配，包括 LoRA 路线 |
| `nanogpt/EXPERIMENT_RESULTS.md` | nanoGPT 训练结果、过拟合分析、与 LSTM 对比解释 |
| `nanogpt/PROJECT_HANDOFF.md` | nanoGPT 环境、训练配置、迁移与运行记录 |

当前开发重点已经从“各自完成 baseline”转为：

```text
统一数据和评测 -> 固定 baseline 结果 -> 三人共同优化 nanoGPT
```

特别注意：当前三块代码的 WikiText-2 token 数略有差异，最终报告前必须统一预处理和评测脚本后重新跑最终结果。

---

## 1. 项目定位与核心论点

- **不复现** GPT-2 124M 完整训练（需 8×A100 跑 4 天，不现实）。
- **复现**的是 nanoGPT 的**训练流程与 GPT 架构本身**，在可控规模上训练"小号 GPT-2"。
- **真正的贡献 = 对比 + 调参 + 分析**：用 baseline 衬托 Transformer，用消融实验解释"为什么某个改动更好"。
- 一句话论点示例：*"在固定算力预算下，RoPE + SwiGLU + 调优学习率调度，相比 nanoGPT 默认配置，在 WikiText-2 上把验证困惑度从 X 降到 Y。"*

---

## 2. 数据集与模型规模（针对 8–12G 显存）

### ⚠️ 统一数据集约定（三人对比的前提，开工前必须先固定）
> **主数据集 = WikiText-2（`wikitext-2-raw-v1`）**。N-gram、LSTM、nanoGPT **三方必须使用完全相同的**：
> - 同一份 train / val / test 划分（用 HuggingFace 官方划分，不要各自重切）；
> - 同一套词表与分词（统一用 **GPT-2 BPE**，便于和 nanoGPT 直接比较）；
> - 同一套评测脚本（PPL/BPC，见第 4 节）。
>
> 选 WikiText-2 的理由：它是**语言模型 baseline 对比的标准基准**，N-gram / LSTM / Transformer 都有公开发表的困惑度可对照；规模约 200 万 token，单卡 8–12G 毫无压力，N-gram / LSTM 分钟级即可训完。算力有余可升级到 **WikiText-103**（~1 亿 token）。

获取方式（一行）：

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")   # train/validation/test 已划分好
```

### 双层数据策略
| 层级 | 数据集 | 分词 | 用途 | 单次耗时 |
|---|---|---|---|---|
| **主线对比层** | **WikiText-2**（升级可选 WikiText-103） | GPT-2 BPE | 头条结果、PPL/BPC、三方对比、消融 | N-gram/LSTM 分钟级，GPT 数小时 |
| 快速调试层 | 字符级 Shakespeare (1MB, repo 自带) | char | 跑通流程、debug，**不进主对比表** | 3–5 分钟 |

### 推荐模型配置（单卡 8–12G，bf16 + Flash Attention）
| 角色 | n_layer | n_head | n_embd | block_size | 约参数量 | 备注 |
|---|---|---|---|---|---|---|
| Shakespeare 消融 | 6 | 6 | 384 | 256 | ~10M | repo 默认 char 配置 |
| 主线 "GPT-2 mini" | 6–8 | 8 | 512 | 256–512 | ~25–45M | 8–12G 可稳定训练 |
| 上限尝试（可选） | 12 | 12 | 768 | 512 | ~124M | 需 batch 很小 + 梯度累积，慢 |

> 显存不够时的旋钮：减小 `batch_size`、增大 `gradient_accumulation_steps`（模拟大 batch）、减小 `block_size`、开 `dtype=bfloat16`。

---

## 3. Baseline 设计（课程硬性要求）

至少 2 个 baseline，用来衬托 Transformer：
1. **N-gram 语言模型**（bigram / trigram + 加一平滑或 Kneser-Ney）：最简单的统计基线，计算几乎零成本。报告 PPL/BPC。
2. **小型 LSTM / RNN 语言模型**：同等参数量级，体现"序列建模"但无注意力，作为强基线。
3.（可选）**随机/多数基线**：用于理论下界（如词表均匀分布的 PPL = 词表大小）。

主方法：
- **nanoGPT 默认复现**（作为"复现基线"）。
- **nanoGPT 调优版**（你们的改进，见第 5 节）。

---

## 4. 量化对比方法（项目命脉）

### 4.1 主指标
- **验证集交叉熵 Loss → 困惑度**：`PPL = exp(val_loss)`。
- **字符级同时报 BPC**：`BPC = val_loss / ln(2)`。BPC/BPB 可跨 tokenizer 公平比较（char vs BPE）。
- **铁律：相同验证集、相同 tokenizer、相同 eval 流程**才能比。

### 4.2 公平对比协议（必须在报告中声明）
两个模型比 loss 时，明确在哪条轴对齐：
- **等参数 (param-matched)**：参数量相同 → 比"结构/优化谁更优"。
- **等算力 (compute-matched)**：固定总训练 token 数 或 总 FLOPs → 更公平。
- 用 repo 的 `transformer_sizing.ipynb` / `scaling_laws.ipynb` 估算 FLOPs。

### 4.3 推荐图表（每张 y 轴都是 val loss）
1. `loss vs 训练 token 数`（收敛速度）
2. `loss vs wall-clock 时间`（实际效率）
3. `loss vs FLOPs`（最严谨的能力对比）

### 4.4 多维度指标表
| 维度 | 指标 | 工具 |
|---|---|---|
| 语言建模 | Val Loss / PPL / BPC | train.py 日志 |
| 零样本下游 | LAMBADA 末词准确率（可选 HellaSwag） | 自写 eval 脚本 |
| 收敛效率 | 达到目标 loss 所需 token / 时间 | 日志 |
| 推理效率 | tokens/sec、显存、参数量 | bench.py |
| 生成质量（定性） | 同一 prompt 的样本 | sample.py |

### 4.5 统计严谨性
- 每个关键配置跑 **≥3 个随机种子**，报告 **mean ± std**。
- 小模型很便宜，务必做到，结论才可信。

### 4.6 结果总表模板
| Model | Params | Tokens | Val Loss↓ | PPL↓ | BPC↓ | LAMBADA↑ | tok/s↑ |
|---|---|---|---|---|---|---|---|
| N-gram | – | – | – | | | – | – |
| LSTM | | | | | | | |
| nanoGPT (默认) | | | | | | | |
| nanoGPT (调优) | | | | | | | |

---

## 5. 调优 / 消融实验矩阵（深度 > 广度，选 2–4 条主线深入做）

> 原则：**每次只改一个变量**，对照默认配置。

### 第 1 层 · 优化超参（最便宜，先做）
- 学习率 + warmup + cosine 衰减：`learning_rate` / `warmup_iters` / `lr_decay_iters` / `min_lr`
- batch size 与梯度累积：`batch_size` / `gradient_accumulation_steps`
- 正则：`weight_decay` / `grad_clip` / `dropout`
- AdamW：`beta1` / `beta2`

### 第 2 层 · 架构改动（最适合作为"复现 + 改进"亮点）
- **位置编码**：学习式绝对位置 → **RoPE** 或 **ALiBi**（repo todo 明确建议；可附带"长度外推"实验）
- **归一化**：LayerNorm → **RMSNorm**；可加 QK-Norm
- **FFN/激活**：GeLU → **SwiGLU**
- 权重绑定 / 初始化缩放
- 固定参数量下 **深 vs 宽**（n_layer / n_embd / n_head 配比）

### 第 3 层 · 训练效率（体现工程能力，对应代码 10%）
- 混合精度 `bfloat16`
- Flash Attention（PyTorch 2.0 `scaled_dot_product_attention`，repo 已内置）
- `torch.compile`（Windows 不支持时用 `--compile=False`）
- batch size 渐增调度（repo todo）

### 第 4 层 · 数据/分词
- `block_size`（上下文长度）影响
- char-level vs BPE（用 BPC 公平比较）

**推荐主线（示例）**：① 学习率调度搜索 ② RoPE 替换 + 外推实验 ③ SwiGLU ④ LSTM/N-gram baseline 对比。

---

## 6. 分工与时间线（3 人：A / B / C）

### 当前分工原则

Baseline 已经基本完成，后续三个人共同转向 nanoGPT 优化：

| 成员 | 后续重点 | 主要职责 |
|---|---|---|
| **成员 A** | nanoGPT 数据与评测 | 统一 WikiText-2 预处理、统一 eval、结果表、loss 曲线、定性样例汇总 |
| **成员 B** | nanoGPT 超参与训练实验 | 学习率/dropout/block_size sweep，多随机种子，整理训练日志 |
| **成员 C** | nanoGPT 架构与 LoRA | RoPE/RMSNorm/SwiGLU/LoRA 实现与消融实验 |

N-gram 与 LSTM 后续只做必要维护：保证能复现指标、能生成同 prompt 样例、能接入统一评测脚本。详细任务见 `OPTIMIZATION_TASKS.md`。

### 时间线
| 阶段 | 内容 | 负责 |
|---|---|---|
| W1 | Baseline 代码合并：N-gram / LSTM / nanoGPT | 已完成 |
| W2 | 统一数据预处理与评测脚本，重新确认三方 baseline 指标 | 全员 |
| W3 | nanoGPT 超参搜索：LR / dropout / block_size / 多种子 | A / B / C |
| W4 | nanoGPT 架构与 LoRA 消融：RoPE / RMSNorm / SwiGLU / LoRA | A / B / C |
| W5 | 汇总结果、写报告、整理答辩材料 | 全员 |

---

## 7. 报告结构（NeurIPS 模板，≤6 页正文）

1. Introduction：问题与动机、为什么复现 nanoGPT
2. Background：GPT/Transformer、语言建模与困惑度
3. Method：复现细节 + 你们的调优/改进
4. Experimental Setup：数据、模型规模、指标、对比协议（等参数/等算力）
5. Results & Analysis：结果表 + scaling 曲线 + 消融 + 错误/样本分析
6. Conclusion & Limitations
- 附录/参考文献不计入页数

---

## 8. 环境与运行（Windows 注意事项）

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
# 复现快速跑通：
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py        # 有 GPU
# Windows 若 torch.compile 报错：加 --compile=False
python sample.py --out_dir=out-shakespeare-char
```

- Windows 上 `torch.compile` 可能不支持 → `--compile=False`（变慢但能跑）。
- `bfloat16` 取决于显卡（图灵及以前可能要用 `float16` + GradScaler）。

---

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| 显存不足 (OOM) | 减小 batch_size / block_size，增大梯度累积，开 bf16 |
| Windows compile 报错 | `--compile=False` |
| 主线训练太慢 | 缩小模型，或主线改用 enwik8 子集 / 减少 max_iters |
| 结果不显著 | 多种子 + 报告 std；对比应在等算力轴上呈现 |
| 跨 tokenizer 不可比 | 统一用 BPC/BPB |

---

## 10. 加分方向（可选，冲 bonus）

- RoPE 的**长度外推**实验：训练短上下文、测试更长上下文的 PPL 退化曲线。
- 在固定算力下做小型 **scaling law** 拟合（loss vs 参数量/FLOPs）。
- 揭示一个反直觉的消融结论（如某正则在小数据反而有害）。
