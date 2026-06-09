# nanoGPT 复现（成员 C 核心复现）

本文件夹是对 Karpathy [nanoGPT](https://github.com/karpathy/nanoGPT) 的**从零干净复现**（结构对齐、代码自写），用于 CS182 课程项目的 GPT-2 复现部分。范围为**核心复现**：GPT 模型、训练循环、文本采样、数据预处理与默认配置。架构改进（RoPE / SwiGLU / RMSNorm）与 PPL/BPC 评测脚本留作后续单独添加。

> **迁移到新机器？** 见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)（含当前训练记录、环境、配置、GPU 搭建步骤）。

## 目录结构

```
nanogpt/
├── model.py                        # GPT 模型（CausalSelfAttention / MLP / Block / GPT）
├── train.py                        # 训练循环（AdamW + cosine LR + warmup + 梯度累积 + amp + ckpt）
├── sample.py                       # 从 checkpoint 采样生成文本（随机采样）
├── generate_fixed.py               # 固定 prompt + 确定性解码生成（供三方定性对比）
├── prompts_shakespeare.txt         # 字符级模型的固定 prompt 列表
├── prompts_wikitext.txt            # BPE 模型的固定 prompt 列表
├── configurator.py                 # 命令行 / 配置文件覆盖工具
├── config/
│   ├── train_shakespeare_char.py   # ~10M 字符级配置（debug，几分钟跑通）
│   └── train_wikitext2.py          # 主线 "GPT-2 mini"（BPE）
└── data/
    ├── shakespeare_char/prepare.py # tiny shakespeare → 字符词表 → train/val.bin + meta.pkl
    └── wikitext2/prepare.py        # wikitext-2-raw-v1 → GPT-2 BPE → train/val/test.bin
```

## 环境

```bash
pip install -r requirements.txt
```

- Windows 上 `torch.compile` 可能不支持 → 训练时加 `--compile=False`。
- `bfloat16` 取决于显卡（图灵及以前可用 `--dtype=float16` + GradScaler，代码已自动处理）。
- 无 GPU 时加 `--device=cpu`（仅建议用 shakespeare_char 小配置 debug）。

## 快速跑通（字符级 Shakespeare，debug）

```bash
# 在 nanogpt/ 目录下执行
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False
python sample.py --out_dir=out-shakespeare-char
```

## 主线（WikiText-2 + GPT-2 BPE）

```bash
python data/wikitext2/prepare.py
python train.py config/train_wikitext2.py --compile=False
python sample.py --out_dir=out-wikitext2
```

## 固定输入的输出（三方定性对比）

为了和 N-gram / LSTM 公平对比"生成质量（定性）"，用**同一组固定 prompt + 确定性（贪心）解码 + 固定种子**生成可复现的续写：

```bash
# WikiText-2 BPE 模型
python generate_fixed.py --out_dir=out-wikitext2 --prompts_file=prompts_wikitext.txt
# 字符级 Shakespeare 模型
python generate_fixed.py --out_dir=out-shakespeare-char --prompts_file=prompts_shakespeare.txt
```

结果会同时写入 `<out_dir>/fixed_samples.txt`（人读，直接贴报告）和 `<out_dir>/fixed_samples.json`（结构化，便于脚本汇总三方）。三方约定用**同一份 prompt 文件**即可对齐输入。默认贪心解码（确定、可复现）；加 `--greedy=False --temperature=0.8 --top_k=200` 可改随机采样。

## 指标换算

训练日志会定期打印 `val loss`，可直接换算：

- 困惑度：`PPL = exp(val_loss)`
- 字符级：`BPC = val_loss / ln(2)`

## 说明

- 所有路径默认相对 `nanogpt/` 运行；`out_dir`（checkpoint 输出目录）也相对运行目录。
- 数据脚本会在各自的 `data/<name>/` 下生成 `.bin` 与 `meta.pkl`；`train.py` 通过 `dataset` 配置项定位它们。
