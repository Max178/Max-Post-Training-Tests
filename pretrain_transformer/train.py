"""
pretrain_transformer/train.py — Skeleton for pretraining a ~500M parameter GPT-style transformer.

Architecture (≈505M params):
    d_model   = 1024
    n_layers  = 36
    n_heads   = 16
    d_ff      = 4096   (4x d_model)
    vocab     = 50257  (GPT-2 BPE tokenizer)
    max_seq   = 1024

Usage:
    python -m pretrain_transformer.train --data_dir /path/to/token/shards --out_dir checkpoints/
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int   = 50257
    max_seq_len: int  = 1024
    d_model: int      = 1024
    n_layers: int     = 36
    n_heads: int      = 16
    d_ff: int         = 4096
    dropout: float    = 0.0
    bias: bool        = False   # no bias → cleaner scaling


@dataclass
class TrainConfig:
    # Data
    data_dir: str           = "data/tokens"
    out_dir: str            = "checkpoints"

    # Optimizer
    max_steps: int          = 100_000
    batch_size: int         = 8          # per device
    grad_accum_steps: int   = 8          # effective batch = batch_size * grad_accum * world_size
    max_lr: float           = 3e-4
    min_lr: float           = 3e-5       # 10% of max, cosine decay floor
    warmup_steps: int       = 2_000
    weight_decay: float     = 0.1
    grad_clip: float        = 1.0

    # Logging / checkpointing
    log_every: int          = 10
    eval_every: int         = 500
    save_every: int         = 1_000
    resume_from: Optional[str] = None

    # Precision
    dtype: str              = "bfloat16"  # "float32" | "float16" | "bfloat16"

    # Model
    model: ModelConfig      = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Flash attention (PyTorch 2.0+) — falls back gracefully otherwise
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff,    bias=cfg.bias)
        self.fc2  = nn.Linear(cfg.d_ff,    cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                  # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64
    ):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dataset  (expects pre-tokenized .bin shards of uint16 token IDs)
# ---------------------------------------------------------------------------

class TokenShardDataset(IterableDataset):
    """
    Streams token IDs from pre-tokenized binary shards.
    Each shard is a flat array of uint16 token IDs saved with numpy:
        np.array(tokens, dtype=np.uint16).tofile("shard_0000.bin")
    """

    def __init__(self, data_dir: str, seq_len: int, split: str = "train"):
        self.seq_len = seq_len
        self.shards  = sorted(Path(data_dir).glob(f"{split}_*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No {split}_*.bin shards found in {data_dir}")

    def __iter__(self):
        import numpy as np

        for shard in self.shards:
            tokens = np.frombuffer(shard.read_bytes(), dtype=np.uint16).astype(np.int64)
            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                x = torch.from_numpy(tokens[i     : i + self.seq_len])
                y = torch.from_numpy(tokens[i + 1 : i + self.seq_len + 1])
                yield x, y


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.max_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    ctx = torch.autocast(device_type=device, dtype=ptdtype) if device == "cuda" else torch.autocast("cpu", dtype=ptdtype)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    model = GPT(cfg.model).to(device)
    print(f"Parameters: {model.num_params() / 1e6:.1f}M")

    # --- Optimizer ---
    # Separate weight-decayed and non-decayed params (embeddings, LN, biases → no decay)
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device == "cuda",
    )

    # --- Data ---
    train_ds = TokenShardDataset(cfg.data_dir, cfg.model.max_seq_len, split="train")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    train_iter = iter(train_dl)

    # --- Resume ---
    start_step = 0
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # --- Compile (PyTorch 2.0+) ---
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    t0 = time.perf_counter()
    for step in range(start_step, cfg.max_steps):

        # LR
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradient clip
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t0
            tokens_per_sec = (
                cfg.log_every * cfg.grad_accum_steps * cfg.batch_size * cfg.model.max_seq_len / dt
            )
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm:.3f} | {tokens_per_sec:,.0f} tok/s"
            )
            t0 = time.perf_counter()

        # Checkpoint
        if step > 0 and step % cfg.save_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            torch.save(
                {
                    "step":      step,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config":    cfg,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint → {ckpt_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default="data/tokens")
    parser.add_argument("--out_dir",         default="checkpoints")
    parser.add_argument("--max_steps",       type=int,   default=100_000)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--grad_accum",      type=int,   default=8)
    parser.add_argument("--max_lr",          type=float, default=3e-4)
    parser.add_argument("--warmup_steps",    type=int,   default=2_000)
    parser.add_argument("--dtype",           default="bfloat16")
    parser.add_argument("--resume_from",     default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir        = args.data_dir,
        out_dir         = args.out_dir,
        max_steps       = args.max_steps,
        batch_size      = args.batch_size,
        grad_accum_steps= args.grad_accum,
        max_lr          = args.max_lr,
        warmup_steps    = args.warmup_steps,
        dtype           = args.dtype,
        resume_from     = args.resume_from,
    )

    train(cfg)
