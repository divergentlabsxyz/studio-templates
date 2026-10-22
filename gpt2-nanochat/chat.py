"""Talk to a trained model.

    uv run chat.py --checkpoint <path from a run's result>

The model's code is read from the run's own snapshot, beside its checkpoint,
so a node that changed the architecture still loads. It remembers the
conversation until its context is full. Ctrl-D ends it.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="model.pt, from a run's result.")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    snapshot = Path(args.checkpoint).parent / "src"
    if (snapshot / "train.py").exists():
        sys.path.insert(0, str(snapshot))
    import torch

    import prepare
    import train

    saved = torch.load(args.checkpoint, map_location="cpu")
    model = train.GPT(**saved["config"])
    model.load_state_dict(saved["state"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    tok = prepare.tokenizer(saved["config"]["vocab"])
    end = tok.token_to_id("<|end|>")
    history = "<|bos|>"
    turn = 0
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        history += f"<|user|>{text}<|end|><|assistant|>"
        ids = tok.encode(history).ids
        out = []
        with torch.no_grad():
            generator = torch.Generator().manual_seed(turn)
            for _ in range(120):
                logits = model(torch.tensor([ids[-model.context :]], device=device))[0, -1]
                logits = logits.float().cpu() / args.temperature
                logits[logits < torch.topk(logits, 40).values[-1]] = -float("inf")
                token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
                if token == end:
                    break
                ids.append(token)
                out.append(token)
        answer = tok.decode(out).strip()
        print(f"bot> {answer}")
        history += f"{answer}<|end|>"
        turn += 1


if __name__ == "__main__":
    main()
