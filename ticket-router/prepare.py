"""Setup, once: Banking77's tickets and the sentence encoder, into the cache.

    uv run prepare.py

Banking77 (PolyAI, CC BY 4.0) is 13,083 customer messages to a bank, each
labelled with one of 77 intents. queues.toml groups the intents into the
queues a support team works from. Running this again fetches nothing.
"""

import csv
import random
import sys
import tomllib
import urllib.request

from studio import cache_dir

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
ENCODER = "BAAI/bge-small-en-v1.5"
CACHE = cache_dir("ticket-router")
VALIDATION_FRACTION = 0.1


def fetch():
    for split in ("train", "test"):
        target = CACHE / f"{split}.csv"
        if not target.exists():
            print(f"Downloading Banking77 {split}")
            urllib.request.urlretrieve(f"{SOURCE}/{split}.csv", target)


def read(split):
    with open(CACHE / f"{split}.csv", newline="") as file:
        return [(row["text"], row["category"]) for row in csv.DictReader(file)]


def queues(path="queues.toml"):
    """intent -> queue, and the queues in file order, each intent in one."""
    with open(path, "rb") as file:
        table = tomllib.load(file)
    route = {}
    for queue, spec in table.items():
        for intent in spec["intents"]:
            if intent in route:
                sys.exit(f"queues.toml: {intent} is in both {route[intent]} and {queue}.")
            route[intent] = queue
    return route, list(table)


def load():
    """train, validation and test as (text, intent) lists. Validation is a
    fixed tenth of Banking77's train split, so it is the same in every run;
    test is Banking77's own and is never trained on."""
    fetch()
    train = read("train")
    random.Random(0).shuffle(train)
    cut = int(len(train) * VALIDATION_FRACTION)
    validation, train = train[:cut], train[cut:]
    test = read("test")
    route, _ = queues()
    missing = {intent for _, intent in train + test} - route.keys()
    if missing:
        sys.exit(f"queues.toml: these intents are in no queue: {', '.join(sorted(missing))}.")
    return train, validation, test


def encoder():
    """The encoder's directory, fetched once and read from the cache after."""
    from huggingface_hub import snapshot_download

    files = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]
    try:
        return snapshot_download(ENCODER, allow_patterns=files, local_files_only=True)
    except Exception:
        return snapshot_download(ENCODER, allow_patterns=files)


if __name__ == "__main__":
    train, validation, test = load()
    print(f"Banking77: {len(train)} train, {len(validation)} validation, {len(test)} test")
    print(f"Encoder {ENCODER} at {encoder()}")
    _, names = queues()
    print(f"Queues: {', '.join(names)}")
    print(f"Cache: {CACHE}")
