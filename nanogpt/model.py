"""
GPT-2 模型定义（nanoGPT 复现，代码自写、结构对齐）。

包含：
  - GPTConfig：超参数据类
  - LayerNorm：支持可选 bias 的 LayerNorm
  - CausalSelfAttention：因果自注意力（优先用 PyTorch 2.0 Flash Attention）
  - MLP：GeLU 前馈网络（4x 隐藏维）
  - Block：pre-LN Transformer block
  - GPT：完整模型（权重绑定、特殊初始化、generate、configure_optimizers 等）
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 词表 50257，向上取整到 64 的倍数以提升效率
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True 时 Linear/LayerNorm 带 bias（与 GPT-2 一致）；False 略快略好
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    swiglu_hidden_mult: float = 8 / 3
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: str = "attn"
    lora_freeze_base: bool = True


class LayerNorm(nn.Module):
    """带可选 bias 的 LayerNorm。PyTorch 的 F.layer_norm 不支持直接关掉 bias，故自封装。"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm：只按均方根归一化，现代 GPT 变体常用。"""

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class LoRALinear(nn.Module):
    """在线性层旁路添加低秩更新；base linear 可冻结，只训练 adapter。"""

    def __init__(self, in_features, out_features, bias, rank, alpha, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        base = self.linear(x)
        update = self.lora_b(self.lora_a(self.lora_dropout(x))) * self.scaling
        return base + update


def _make_norm(config):
    return RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)


def _lora_target_enabled(config, target):
    """Return whether a named projection should be wrapped with LoRA adapters."""
    if not config.use_lora:
        return False
    targets = {t.strip() for t in config.lora_targets.split(",") if t.strip()}
    if "all" in targets:
        return True
    if target.startswith("attn.") and "attn" in targets:
        return True
    if target.startswith("mlp.") and "mlp" in targets:
        return True
    return target in targets


def _make_linear(config, in_features, out_features, target):
    if _lora_target_enabled(config, target):
        return LoRALinear(
            in_features, out_features, config.bias,
            config.lora_rank, config.lora_alpha, config.lora_dropout,
        )
    return nn.Linear(in_features, out_features, bias=config.bias)


def _rotate_half(x):
    # RoPE treats adjacent halves as 2D rotation pairs: (x1, x2) -> (-x2, x1).
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, inv_freq):
    """Apply rotary position embedding to q/k without changing tensor shapes."""
    T = q.size(-2)
    pos = torch.arange(T, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :].to(dtype=q.dtype)
    sin = emb.sin()[None, None, :, :].to(dtype=q.dtype)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 一次性算出 q, k, v 的投影
        self.c_attn = _make_linear(config, config.n_embd, 3 * config.n_embd, "attn.c_attn")
        # 输出投影
        self.c_proj = _make_linear(config, config.n_embd, config.n_embd, "attn.c_proj")
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        head_dim = config.n_embd // config.n_head
        if self.use_rope:
            assert head_dim % 2 == 0, "RoPE 需要偶数 head_dim"
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        # PyTorch >= 2.0 才有 scaled_dot_product_attention（Flash / 内存高效注意力）
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: 未检测到 PyTorch 2.0 的 Flash Attention，回退到手写注意力")
            # 下三角因果 mask
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch, 序列长度, 嵌入维度

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.use_rope:
            q, k = _apply_rope(q, k, self.rope_inv_freq)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        hidden_dim = int(config.swiglu_hidden_mult * config.n_embd) if self.use_swiglu else 4 * config.n_embd
        self.c_fc = _make_linear(
            config,
            config.n_embd,
            2 * hidden_dim if self.use_swiglu else hidden_dim,
            "mlp.c_fc",
        )
        self.gelu = nn.GELU()
        self.c_proj = _make_linear(config, hidden_dim, config.n_embd, "mlp.c_proj")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        if self.use_swiglu:
            # SwiGLU splits the projection into value and gate branches.
            x, gate = x.chunk(2, dim=-1)
            x = F.silu(gate) * x
        else:
            x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """pre-LN Transformer block：x = x + attn(ln1(x)); x = x + mlp(ln2(x))"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = _make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),   # token 嵌入
                wpe=None if config.use_rope else nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=_make_norm(config),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：输入 token 嵌入与输出投影共享权重
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.reset_lora_parameters()
        # 对残差投影做特殊缩放初始化（GPT-2 论文做法）
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("c_proj.linear.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """统计参数量。默认不计入位置嵌入（习惯上不算）。token 嵌入因权重绑定也算在 lm_head 里。"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def mark_only_lora_as_trainable(self):
        """冻结 base 权重，仅训练 LoRA adapter。"""
        for name, param in self.named_parameters():
            param.requires_grad = ".lora_" in name

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"序列长度 {t} 超过 block_size {self.config.block_size}"
        )
        tok_emb = self.transformer.wte(idx)   # (b, t, n_embd)
        if self.transformer.wpe is None:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)    # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # 推理时只需最后一个位置的 logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """把模型的 block_size 裁短（例如加载较大的预训练模型后用更短的上下文）。"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """从 HuggingFace 加载 GPT-2 预训练权重（便于与官方模型对照）。"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]
        # HF 的部分权重是 Conv1D，需要转置
        transposed = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"键数量不一致：{len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """配置 AdamW：>=2 维参数（matmul/embedding）做 weight decay，1 维参数（bias/LN）不做。"""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        # 若可用则启用 fused AdamW（CUDA 上更快）
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估算模型 FLOPs 利用率（MFU），相对 A100 bfloat16 峰值 312 TFLOPS。"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """给定上下文 idx (b, t)，自回归生成 max_new_tokens 个 token。"""
        for _ in range(max_new_tokens):
            # 上下文超过 block_size 时裁剪
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
