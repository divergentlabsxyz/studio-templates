"""Trains a small GPT-2 from scratch to chat, then asks it three things.

    uv run train.py --budget 120

Two phases share the budget, as nanochat's base and chat training do in
miniature: first stories, to learn English, then chats, to learn the
turns. CHAT_FRACTION is the chat phase's share of the time.

The model is GPT-2 as published: learned positions, LayerNorm, GELU, and
the input and output embeddings tied. It is the baseline, not the ceiling:
nanochat's GPT is the same idea with a decade of small fixes.

prepare.evaluate() is the yardstick: bits per byte on the test chats, which
no phase trains on. A node may change anything else.
"""

import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import prepare
from studio import Budget, point, result, run_dir

# The agent sets SIZE from `python3 machine.py`'s tier before the baseline.
SIZE = "medium"
SIZES = {
    "small": {"layers": 4, "heads": 4, "width": 128, "batch": 16},
    "medium": {"layers": 6, "heads": 6, "width": 288, "batch": 32},
    "large": {"layers": 8, "heads": 8, "width": 512, "batch": 32},
}
CONTEXT = 256
VOCAB = prepare.VOCAB
CHAT_FRACTION = 0.3
LEARNING_RATE = 3e-3
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.1
EVALS = 6  # Times the yardstick is read during a run, the last at the end.
SEED = 0
PROMPTS = [
    "Hi! How are you today?",
    "What should I cook for dinner tonight?",
    "Can you help me plan a picnic?",
]


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, context, layers, heads, width):
        super().__init__()
        self.context = context
        self.wte = nn.Embedding(vocab, width)
        self.wpe = nn.Embedding(context, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.wte(x) + self.wpe(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def batches(stream, batch, context, device, rng):
    """Random windows of the stream, forever."""
    data = torch.from_numpy(stream.astype(np.int64))
    while True:
        starts = [rng.randrange(len(data) - context - 1) for _ in range(batch)]
        x = torch.stack([data[s : s + context] for s in starts])
        y = torch.stack([data[s + 1 : s + context + 1] for s in starts])
        yield x.to(device), y.to(device)


@torch.no_grad()
def reply(model, tok, prompt, device, tokens=80, temperature=0.8, top_k=40, seed=SEED):
    """What the model answers to one user turn."""
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = tok.encode(f"<|bos|><|user|>{prompt}<|end|><|assistant|>").ids
    end = tok.token_to_id("<|end|>")
    out = []
    for _ in range(tokens):
        x = torch.tensor([ids[-model.context :]], device=device)
        logits = model(x)[0, -1].float().cpu() / temperature
        cutoff = torch.topk(logits, top_k).values[-1]
        logits[logits < cutoff] = -float("inf")
        token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        if token == end:
            break
        ids.append(token)
        out.append(token)
    model.train()
    return tok.decode(out).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=120, help="Training seconds.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    size = SIZES[SIZE]
    tok = prepare.tokenizer(VOCAB)
    config = {
        "vocab": VOCAB, "context": CONTEXT,
        "layers": size["layers"], "heads": size["heads"], "width": size["width"],
    }  # fmt: skip
    model = GPT(**config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters / 1e6:.1f}M parameters on {device}, {args.budget:.0f} seconds")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": other, "weight_decay": 0}],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )
    phases = {
        "stories": batches(prepare.tokens("stories", VOCAB), size["batch"], CONTEXT, device, rng),
        "chats": batches(prepare.tokens("chats", VOCAB), size["batch"], CONTEXT, device, rng),
    }

    def loss_sum(x, y):
        return F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten(), reduction="sum")

    def bpb():
        model.eval()
        value = prepare.evaluate(loss_sum, VOCAB, CONTEXT, device)
        model.train()
        return value

    budget = Budget(args.budget)
    step = tokens = 0
    next_eval = 1
    while not budget.spent:
        step += 1
        phase = "chats" if budget.fraction >= 1 - CHAT_FRACTION else "stories"
        x, y = next(phases[phase])
        warmup = min(step / WARMUP_STEPS, 1)
        lr = LEARNING_RATE * warmup * (1 - 0.9 * budget.fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += x.numel()
        if step % 25 == 0:
            point(
                step=step,
                seconds=round(budget.elapsed, 1),
                **{f"loss_{phase}": loss.item()},
                lr=lr,
                tokens_per_second=round(tokens / budget.elapsed),
            )
        if budget.fraction >= next_eval / EVALS and next_eval < EVALS:
            point(step=step, val_bpb=bpb())
            next_eval += 1

    trained = budget.elapsed
    val_bpb = bpb()
    point(step=step, val_bpb=val_bpb)
    checkpoint = run_dir() / "model.pt"
    torch.save({"config": config, "state": model.state_dict()}, checkpoint)
    result(
        {
            "val_bpb": round(val_bpb, 4),
            "tokens_trained": tokens,
            "tokens_per_second": round(tokens / trained),
            "steps": step,
            "parameters": parameters,
            "size": SIZE,
            "samples": [{"user": p, "model": reply(model, tok, p, device)} for p in PROMPTS],
            "checkpoint": str(checkpoint.resolve()),
        }
    )


if __name__ == "__main__":
    main()
