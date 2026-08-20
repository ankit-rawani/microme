"""MicroMe transformer — single readable file (plan §4).

Modern-Llama skeleton with the cheap speedrun upgrades:
RMSNorm pre-norm, RoPE, QK-norm, ReLU^2 MLP (no gate), untied embeddings,
logit softcap, zero-init residual projections. SDPA -> FlashAttention kernel.

Kept deliberately hackable: no framework, no config files yet. Presets live
in GPTConfig. Everything else in the plan (yaml runs, Muon, data shards) is a
later milestone and is NOT scaffolded here on purpose.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    ctx: int = 64
    softcap: float = 15.0
    rope_base: float = 10000.0
    # --- ablation-lab knobs (defaults reproduce the shipped micro_* models) ---
    mlp: str = "relu2"            # "relu2" | "swiglu"
    tie_embeddings: bool = False  # share input/output embedding weights
    qk_norm: bool = True          # RMSNorm on q,k before attention
    layer_reuse: int = 1          # MobileLLM-style: apply each block N times (more depth, same params)
    grad_ckpt: bool = False       # gradient/activation checkpointing (fit bigger models on 8GB)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        hd = self.d_model // self.n_head
        assert hd % 2 == 0, "head_dim must be even for RoPE"
        return hd


# Presets from the plan (§4). micro_1m is the M0 smoke model (~1M params).
PRESETS = {
    "micro_1m":   GPTConfig(vocab_size=512,   n_layer=4,  n_head=4,  d_model=128, ctx=64),
    "micro_30m":  GPTConfig(vocab_size=24576, n_layer=8,  n_head=7,  d_model=448, ctx=512),
    "micro_125m": GPTConfig(vocab_size=24576, n_layer=20, n_head=10, d_model=640, ctx=1024),
    # capstone: deep-thin + tied embeddings (the 3-seed-confirmed lab winner) + gradient checkpointing
    "mini_220m":  GPTConfig(vocab_size=24576, n_layer=28, n_head=12, d_model=768, ctx=1024,
                            tie_embeddings=True, grad_ckpt=True),
}


def rmsnorm(x, weight=None, eps=1e-6):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x if weight is None else x * weight


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rmsnorm(x, self.weight)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd). Interleaved rotary.
    T = x.size(-2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:T], sin[:T]  # (T, hd/2)
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack((xr1, xr2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.nh, self.hd, self.qk_norm = cfg.n_head, cfg.head_dim, cfg.qk_norm
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = rmsnorm(q), rmsnorm(k)           # QK-norm (weightless)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.kind = cfg.mlp
        if self.kind == "swiglu":                    # gated; hidden sized to match ReLU²'s 8*d² params
            h = max(32, int(round(8 * d / 3 / 32)) * 32)
            self.gate = nn.Linear(d, h, bias=False)
            self.up = nn.Linear(d, h, bias=False)
            self.proj = nn.Linear(h, d, bias=False)
        else:                                         # relu2: simpler, no gate
            self.fc = nn.Linear(d, 4 * d, bias=False)
            self.proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.proj(F.silu(self.gate(x)) * self.up(x))
        return self.proj(F.relu(self.fc(x)) ** 2)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # untied

        # RoPE tables (buffers, not params).
        hd = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(cfg.ctx).float()
        freqs = torch.outer(t, inv_freq)            # (ctx, hd/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

        self.apply(self._init)
        # zero-init residual output projections (stable start)
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.proj.weight)
            nn.init.zeros_(blk.mlp.proj.weight)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight    # share input/output embeddings

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        T = idx.size(1)
        assert T <= self.cfg.ctx, f"seq len {T} > ctx {self.cfg.ctx}"
        x = self.wte(idx)
        ckpt = self.cfg.grad_ckpt and self.training
        for blk in self.blocks:
            for _ in range(self.cfg.layer_reuse):    # MobileLLM immediate block reuse
                if ckpt:                             # recompute activations in backward (saves VRAM)
                    x = torch.utils.checkpoint.checkpoint(blk, x, self.cos, self.sin, use_reentrant=False)
                else:
                    x = blk(x, self.cos, self.sin)
        logits = self.lm_head(self.norm_f(x))
        cap = self.cfg.softcap
        logits = cap * torch.tanh(logits / cap)     # logit softcap
        loss = None
        if targets is not None:
            if loss_mask is None:                    # pretraining: mean over all tokens
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:                                    # SFT: mean over assistant tokens only
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                m = loss_mask.reshape(-1).to(ce.dtype)
                loss = (ce * m).sum() / m.sum().clamp(min=1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, rep_penalty=1.0):
        # no KV cache yet (that's M7); fine for short eval samples.
        start = idx.size(1)                      # penalize only tokens WE generate, not the prompt/context
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.ctx:])
            logits = logits[:, -1, :]
            if rep_penalty != 1.0 and idx.size(1) > start:
                for b in range(idx.size(0)):     # divide logits of already-generated tokens (anti-loop)
                    logits[b, idx[b, start:]] /= rep_penalty
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


if __name__ == "__main__":
    m = GPT(PRESETS["micro_1m"])
    print(f"micro_1m params: {m.num_params()/1e6:.2f}M")
    x = torch.randint(0, 512, (2, 64))
    logits, loss = m(x, x)
    print("logits", tuple(logits.shape), "loss", float(loss))
