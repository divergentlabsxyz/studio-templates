"""Routes Banking77's tickets to queues, and measures how well.

    uv run train.py --budget 60

The router predicts a ticket's intent and sends it to that intent's queue
(queues.toml). METHOD says how a ticket becomes features:

- "tfidf": word TF-IDF and a logistic regression. Seconds, on the CPU, and
  the baseline to beat.
- "embeddings": a frozen sentence encoder (prepare.ENCODER) and a linear
  head trained for the budget on top, with a point per epoch.

evaluate() and prepare.load() are the yardstick: a node may change anything
else.
"""

import argparse
import pickle
import time

import numpy as np

import prepare
from studio import Budget, cache_dir, point, result, run_dir

METHOD = "tfidf"
# The agent sets SIZE from `python3 machine.py`'s tier before the baseline.
SIZE = "medium"
SIZES = {
    "small": {"batch": 32},
    "medium": {"batch": 64},
    "large": {"batch": 128},
}
SEED = 0


def evaluate(route, predicted, expected):
    """How many tickets land in the wrong queue, how many fraud reports miss
    the fraud queue, and the rates behind them, from predicted and expected
    intents."""
    predicted_queues = [route[intent] for intent in predicted]
    expected_queues = [route[intent] for intent in expected]
    pairs = list(zip(predicted_queues, expected_queues))
    f1s = []
    for queue in sorted(set(route.values())):
        hits = sum(p == e == queue for p, e in pairs)
        claimed = sum(p == queue for p, _ in pairs)
        actual = sum(e == queue for _, e in pairs)
        precision = hits / claimed if claimed else 0.0
        recall = hits / actual if actual else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if hits else 0.0)
    fraud = [p for p, e in pairs if e == "fraud"]
    return {
        "misrouted": sum(p != e for p, e in pairs),
        "fraud_missed": sum(p != "fraud" for p in fraud),
        "queue_accuracy": round(sum(p == e for p, e in pairs) / len(pairs), 4),
        "fraud_recall": round(sum(p == "fraud" for p in fraud) / max(len(fraud), 1), 4),
        "intent_accuracy": round(sum(p == e for p, e in zip(predicted, expected)) / len(pairs), 4),
        "queue_macro_f1": round(sum(f1s) / len(f1s), 4),
    }


def train_tfidf(train, budget):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(sublinear_tf=True), LogisticRegression(C=10, max_iter=2000)
    )
    texts, intents = zip(*train)
    model.fit(texts, intents)
    return lambda batch: list(model.predict(batch)), model


def encode(model, texts, name, batch):
    """Sentence embeddings, kept in the cache by split, since a frozen encoder
    says the same of the same text every time."""
    path = cache_dir("ticket-router") / f"embeddings-{prepare.ENCODER.replace('/', '_')}-{name}.npy"
    if name and path.exists():
        return np.load(path)
    vectors = model.encode(
        list(texts), batch_size=batch, normalize_embeddings=True, convert_to_numpy=True
    )
    if name:
        np.save(path, vectors)
    return vectors


def train_embeddings(train, validation, route, budget, size):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer(prepare.encoder(), device=device)
    labels = sorted({intent for _, intent in train})
    index = {label: i for i, label in enumerate(labels)}
    x = torch.tensor(encode(encoder, [t for t, _ in train], "train", size["batch"]))
    y = torch.tensor([index[i] for _, i in train])
    vx = torch.tensor(encode(encoder, [t for t, _ in validation], "validation", size["batch"]))
    vy = torch.tensor([index[i] for _, i in validation])

    head = torch.nn.Linear(x.shape[1], len(labels))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-2, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch = 0
    while not budget.spent and epoch < 300:
        epoch += 1
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), 256):
            batch = order[start : start + 256]
            loss = loss_fn(head(x[batch] * 20), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        with torch.no_grad():
            logits = head(vx * 20)
            val_loss = loss_fn(logits, vy).item()
            predicted = [labels[i] for i in logits.argmax(1).tolist()]
        scores = evaluate(route, predicted, [i for _, i in validation])
        point(
            epoch=epoch,
            train_loss=total / len(x),
            val_loss=val_loss,
            val_queue_accuracy=scores["queue_accuracy"],
            val_intent_accuracy=scores["intent_accuracy"],
            val_fraud_recall=scores["fraud_recall"],
        )

    weight = head.weight.detach().numpy().T * 20
    bias = head.bias.detach().numpy()

    def predict(batch):
        vectors = encode(encoder, batch, None, size["batch"])
        return [labels[i] for i in (vectors @ weight + bias).argmax(1)]

    return predict, {"encoder": prepare.ENCODER, "labels": labels, "weight": weight, "bias": bias}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=60, help="Training seconds.")
    args = parser.parse_args()

    train, validation, test = prepare.load()
    route, _ = prepare.queues()
    budget = Budget(args.budget)
    if METHOD == "tfidf":
        predict, model = train_tfidf(train, budget)
    elif METHOD == "embeddings":
        predict, model = train_embeddings(train, validation, route, budget, SIZES[SIZE])
    else:
        raise SystemExit(f"METHOD is {METHOD!r}; it is 'tfidf' or 'embeddings'.")
    trained = budget.elapsed

    texts, expected = zip(*test)
    predict(list(texts[:32]))
    started = time.perf_counter()
    predicted = predict(list(texts))
    ms_per_ticket = (time.perf_counter() - started) * 1000 / len(texts)

    scores = evaluate(route, predicted, expected)
    wrong = [
        {"ticket": text, "queue": route[want], "routed_to": route[got]}
        for text, want, got in zip(texts, expected, predicted)
        if route[want] != route[got]
    ]
    checkpoint = run_dir() / "router.pkl"
    with open(checkpoint, "wb") as file:
        pickle.dump({"method": METHOD, "route": route, "model": model}, file)
    result(
        {
            **scores,
            "ms_per_ticket": round(ms_per_ticket, 3),
            "train_seconds": round(trained, 1),
            "method": METHOD,
            "size": SIZE,
            "misrouted_examples": wrong[:5],
            "checkpoint": str(checkpoint.resolve()),
        }
    )


if __name__ == "__main__":
    main()
