# LSTM Baseline — WikiText-2 语言模型基线

GPT-2 复现项目（CS182 Track C）的 LSTM 语言模型基线。默认使用仓库根目录 `data/wikitext2` 中的 WikiText-2 + GPT-2 BPE 统一数据，与 N-gram 和 nanoGPT 进行公平对比。

**模型架构**：多层 LSTM + LayerNorm + 权重绑定，约 14M 参数，单 GPU **5 分钟内**完成训练。

## 快速开始

```bash
# 1. 安装 GPU PyTorch（必须，不支持 CPU 训练）
#    大多数 GPU（GTX 10xx / RTX 20xx/30xx/40xx/50xx）：
pip install torch --index-url https://download.pytorch.org/whl/cu124

#    验证 GPU 可见：
python -c "import torch; assert torch.cuda.is_available(), 'GPU not found'"

# 2. 安装其余依赖
pip install -r LSTM_baseline/requirements.txt

# 3. 准备统一数据（在仓库根目录）
python data/wikitext2/prepare.py

# 4. 训练（从 LSTM_baseline 目录运行，RTX 3060 约 8 分钟）
cd LSTM_baseline
python src/lstm.py --config=src/config_lstm.py

# 5. 查看结果
python src/visualize.py
```

## 目录结构

```
LSTM_baseline/
├── src/
│   ├── prepare_data.py      # 下载并分词 WikiText-2
│   ├── lstm.py              # 模型定义 + 训练循环
│   ├── eval_lm.py           # 独立 PPL/BPC 评测
│   ├── config_lstm.py       # 训练超参数
│   └── visualize.py         # 结果报告 + 损失曲线
├── data/wikitext2/          # 旧 LSTM 本地预处理数据，仅作开发记录
├── out/                     # 模型 checkpoint、结果、图表
├── requirements.txt
└── README.md
```

## 模型结构

```
Embedding(词表=50257 → 维度=256)
  → LSTM(256→256, 2层, dropout=0.2)
  → LayerNorm(256)
  → Linear(256→50257, 与 embedding 共享权重)
```

| 超参数 | 取值 |
|:---|---:|
| Embedding 维度 | 256 |
| LSTM 层数 | 2 |
| 上下文长度 | 128 |
| 批次大小 | 64 |
| Dropout | 0.2 |
| 参数量 | 13.9M |

**训练配置**：AdamW + 余弦学习率调度（3e-3 → 1e-4）+ 线性预热，梯度裁剪 1.0。

## 仅评测

```bash
cd LSTM_baseline
python src/eval_lm.py --checkpoint=out/ckpt_best.pt
```

以 JSON 格式输出 `val_loss`、`val_ppl`、`val_bpc`、`test_loss`、`test_ppl`、`test_bpc`。

## 实验结果（WikiText-2）

| 划分 | Loss | PPL | BPC |
|:---|---:|---:|---:|
| Best Val | 5.5046 | 245.8 | 7.9414 |
| Val (full) | 5.5037 | 245.6 | 7.9401 |
| **Test** | **5.5611** | **260.1** | **8.0230** |

> 均匀分布基线（随机）：PPL = 50,257。模型实现了约 200 倍的降低。

## 配置说明

编辑 `LSTM_baseline/src/config_lstm.py` 或通过 `--config=your_config.py` 指定：

```python
# 模型
n_embd = 256           # embedding / LSTM 隐层维度
n_layer = 2            # LSTM 层数
dropout = 0.2

# 训练
learning_rate = 3e-3
max_iters = 2000
batch_size = 64
block_size = 128

# 系统
device = "cuda"
dtype = "auto"         # auto = bfloat16 > float16（自动适配 GPU）
```

如需更强的基线，增大 `n_embd` 和 `n_layer`（如 `n_embd=512, n_layer=3` → 32M 参数，需更长训练时间）。
