"""Trains a character-level PII redactor from scratch, and measures what it leaks.

    uv run train.py --budget 60

Every character of a message gets a kind (prepare.KINDS): nothing, a name,
an email, a phone number or an account number. A run of one kind is masked
as [NAME], [EMAIL], [PHONE] or [ACCOUNT]. Training draws fresh messages
from prepare.train_message for the whole budget; it never runs out of data.

The baseline is a stack of 1-D convolutions over bytes. Its window is a few
characters wide, which is too narrow to tell "Mila" in "this is Mila" from
"Milan" in "sent from Milan".

evaluate(), regex_redact() and prepare.load() are the yardstick: a node may
change anything else.
"""

import argparse
import random
import re

import torch
import torch.nn.functional as F
from torch import nn

import prepare
from studio import Budget, point, result, run_dir

# The agent sets SIZE from `python3 machine.py`'s tier before the baseline.
SIZE = "medium"
SIZES = {
    "small": {"width": 64, "layers": 3, "batch": 64},
    "medium": {"width": 128, "layers": 4, "batch": 128},
    "large": {"width": 192, "layers": 6, "batch": 128},
}
KERNEL = 5
LEARNING_RATE = 2e-3
LOWERCASE = 0.0  # The share of training messages lowercased.
THRESHOLD = 0.5  # Mask a character when P(not O) is at least this.
EVAL_EVERY = 100
SEED = 0


class Redactor(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.embed = nn.Embedding(256, width)
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, KERNEL, padding=KERNEL // 2) for _ in range(layers)
        )
        self.out = nn.Linear(width, len(prepare.KINDS))

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return self.out(h.transpose(1, 2))


def encode(texts, device):
    data = [list(text.encode("utf-8", "replace"))[:256] for text in texts]
    width = max(len(row) for row in data)
    x = torch.zeros(len(data), width, dtype=torch.long)
    for i, row in enumerate(data):
        x[i, : len(row)] = torch.tensor(row)
    return x.to(device)


def byte_labels(text, labels):
    """Character labels, spread over each character's UTF-8 bytes."""
    out = []
    for char, kind in zip(text, labels):
        out += [kind] * len(char.encode("utf-8", "replace"))
    return out[:256]


def predict(model, texts, device, threshold=THRESHOLD):
    """A kind per character of each text."""
    model.eval()
    with torch.no_grad():
        probs = model(encode(texts, device)).softmax(-1).cpu()
    model.train()
    out = []
    for text, p in zip(texts, probs):
        kinds, i = [], 0
        for char in text:
            row = p[min(i, 255)]
            pii = 1 - row[0].item()
            kinds.append(int(row[1:].argmax()) + 1 if pii >= threshold else 0)
            i += len(char.encode("utf-8", "replace"))
        out.append(kinds)
    return out


def regex_redact(text):
    """What a page of regular expressions catches, for reference."""
    kinds = [0] * len(text)
    rules = [
        (r"[\w.+-]+@[\w-]+\.[\w.]+", 2),
        (r"\b[A-Za-z]{2}\d{2}(?: ?[A-Za-z0-9]{4}){3,4}(?: ?\d{1,4})?\b", 4),
        (r"(?:\+\d{2} ?|\(\d{3}\) ?|0)\d[\d -]{5,}\d", 3),
        (r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{8,10}\b", 4),
    ]
    for pattern, kind in rules:
        for match in re.finditer(pattern, text):
            for i in range(*match.span()):
                kinds[i] = kinds[i] or kind
    return kinds


def evaluate(rows, predicted):
    """What leaked and what was masked for nothing. An entity leaks when any
    of its characters is left as it was; its kind need not be right."""
    leaked = {kind: 0 for kind in prepare.KINDS[1:]}
    total = {kind: 0 for kind in prepare.KINDS[1:]}
    pii_chars = leaked_chars = plain_chars = overmasked_chars = 0
    for (_, labels), kinds in zip(rows, predicted):
        for start, end, kind in prepare.entities(labels):
            name = prepare.KINDS[kind]
            total[name] += 1
            leaked[name] += any(k == 0 for k in kinds[start:end])
        for want, got in zip(labels, kinds):
            if want:
                pii_chars += 1
                leaked_chars += got == 0
            else:
                plain_chars += 1
                overmasked_chars += got != 0
    entities = sum(total.values())
    return {
        "leaked_entities": sum(leaked.values()),
        "entities": entities,
        "leak_rate": round(sum(leaked.values()) / entities, 4),
        "overmask_rate": round(overmasked_chars / plain_chars, 4),
        "char_leak_rate": round(leaked_chars / pii_chars, 4),
        "leak_rate_by_kind": {k.lower(): round(leaked[k] / total[k], 4) for k in total if total[k]},
    }


def mask(text, kinds):
    out, i = "", 0
    while i < len(text):
        if kinds[i]:
            j = i
            while j < len(text) and kinds[j] == kinds[i]:
                j += 1
            out += f"[{prepare.KINDS[kinds[i]]}]"
            i = j
        else:
            out += text[i]
            i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=60, help="Training seconds.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    size = SIZES[SIZE]
    validation, test = prepare.load()
    config = {"width": size["width"], "layers": size["layers"]}
    model = Redactor(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters on {device}, {args.budget:.0f} seconds")

    budget = Budget(args.budget)
    step = 0
    while not budget.spent:
        step += 1
        rows = [prepare.train_message(rng, LOWERCASE) for _ in range(size["batch"])]
        x = encode([text for text, _ in rows], device)
        y = torch.full(x.shape, -100, dtype=torch.long)
        for i, (text, labels) in enumerate(rows):
            target = byte_labels(text, labels)
            y[i, : len(target)] = torch.tensor(target)
        lr = LEARNING_RATE * (1 - 0.9 * budget.fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = F.cross_entropy(model(x).flatten(0, 1), y.to(device).flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0:
            scores = evaluate(validation, predict(model, [t for t, _ in validation], device))
            point(
                step=step,
                seconds=round(budget.elapsed, 1),
                loss=loss.item(),
                val_leak_rate=scores["leak_rate"],
                val_overmask_rate=scores["overmask_rate"],
            )

    texts = [text for text, _ in test]
    predicted = [kinds for i in range(0, len(texts), 256) for kinds in predict(model, texts[i : i + 256], device)]
    scores = evaluate(test, predicted)
    regex = evaluate(test, [regex_redact(text) for text in texts])
    leaks = [
        {"message": text, "redacted": mask(text, kinds)}
        for (text, labels), kinds in zip(test, predicted)
        if any(want and not got for want, got in zip(labels, kinds))
    ]
    checkpoint = run_dir() / "redactor.pt"
    torch.save(
        {"config": config, "threshold": THRESHOLD, "state": model.state_dict()}, checkpoint
    )
    result(
        {
            **scores,
            "regex_leaked_entities": regex["leaked_entities"],
            "steps": step,
            "parameters": parameters,
            "size": SIZE,
            "leak_examples": rng.sample(leaks, min(4, len(leaks))),
            "checkpoint": str(checkpoint.resolve()),
        }
    )


if __name__ == "__main__":
    main()
