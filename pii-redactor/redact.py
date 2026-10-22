"""Masks PII in text you type, with a trained redactor.

    uv run redact.py --checkpoint <path from a run's result> "Hi, I'm Ada, call me on 06-12345678"
    uv run redact.py --checkpoint <path>        # one message a line, until Ctrl-D

The model's code is read from the run's own snapshot, beside its checkpoint,
so a node that changed the architecture still loads.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="redactor.pt, from a run's result.")
    parser.add_argument("messages", nargs="*")
    args = parser.parse_args()

    snapshot = Path(args.checkpoint).parent / "src"
    if (snapshot / "train.py").exists():
        sys.path.insert(0, str(snapshot))
    import torch

    import train

    saved = torch.load(args.checkpoint, map_location="cpu")
    model = train.Redactor(**saved["config"])
    model.load_state_dict(saved["state"])
    messages = args.messages or (line.rstrip("\n") for line in sys.stdin)
    for text in messages:
        kinds = train.predict(model, [text], "cpu", saved["threshold"])[0]
        print(train.mask(text, kinds))


if __name__ == "__main__":
    main()
