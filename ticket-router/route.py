"""Routes tickets you type with a trained router.

    uv run route.py --checkpoint <path from a run's result> "my card got eaten by the atm"
    uv run route.py --checkpoint <path>        # one ticket a line, until Ctrl-D
"""

import argparse
import pickle
import sys
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="router.pkl, from a run's result.")
    parser.add_argument("tickets", nargs="*")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as file:
        saved = pickle.load(file)
    with open("queues.toml", "rb") as file:
        titles = {queue: spec["title"] for queue, spec in tomllib.load(file).items()}

    if saved["method"] == "tfidf":
        predict = saved["model"].predict
    else:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        model = saved["model"]
        encoder = SentenceTransformer(snapshot_download(model["encoder"]))

        def predict(batch):
            vectors = encoder.encode(batch, normalize_embeddings=True)
            return [model["labels"][i] for i in (vectors @ model["weight"] + model["bias"]).argmax(1)]

    tickets = args.tickets or (line.strip() for line in sys.stdin)
    for ticket in tickets:
        if not ticket:
            continue
        intent = predict([ticket])[0]
        queue = saved["route"][intent]
        print(f"{titles.get(queue, queue):<26} {intent:<40} {ticket}")


if __name__ == "__main__":
    main()
