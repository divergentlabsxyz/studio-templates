"""Setup, once: the text, a tokenizer trained on it, and the yardstick.

    uv run prepare.py

Two downloads, about 25 MB together, both into the cache:

- TinyStories (Eldan and Li, CDLA-Sharing-1.0): short stories in the words
  a small child knows, written by GPT-4. Its validation file, 22 MB, is the
  pretraining text: plenty for a few minutes on a Mac.
- everyday-conversations (Hugging Face, Apache-2.0): 2,379 short chats
  between a user and an assistant. Its train split teaches the model to
  chat; its 119 test chats are the yardstick, never trained on.

A chat is written with four special tokens:
<|bos|><|user|>Hi<|end|><|assistant|>Hello! How can I help?<|end|>...

The tokenizer is a byte-level BPE trained here, cached by vocabulary size,
so a node can change the size and compare fairly: the yardstick is bits per
byte of the test chats, which does not depend on the tokenizer.
"""

import math
import urllib.request

import numpy as np

from studio import cache_dir

CACHE = cache_dir("gpt2-nanochat")
STORIES = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
CHATS = "https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/main/data/{split}_sft-00000-of-00001.parquet"
SPECIAL = ["<|bos|>", "<|user|>", "<|assistant|>", "<|end|>"]
VOCAB = 4096


def fetch(url, name):
    target = CACHE / name
    if not target.exists():
        print(f"Downloading {name}")
        partial = target.with_suffix(".partial")
        urllib.request.urlretrieve(url, partial)
        partial.rename(target)
    return target


def stories():
    text = fetch(STORIES, "stories.txt").read_text(encoding="utf-8")
    return [story.strip() for story in text.split("<|endoftext|>") if story.strip()]


def chats(split):
    """Each chat as a list of (role, text), role "user" or "assistant"."""
    import pyarrow.parquet as pq

    path = fetch(CHATS.format(split=split), f"chats-{split}.parquet")
    return [
        [(m["role"], m["content"].strip()) for m in row]
        for row in pq.read_table(path, columns=["messages"]).column("messages").to_pylist()
    ]


def render(chat):
    """A chat as the model reads it, and the text it is made of."""
    return "<|bos|>" + "".join(f"<|{role}|>{text}<|end|>" for role, text in chat)


def tokenizer(vocab=VOCAB):
    """A byte-level BPE of `vocab` tokens, the four special ones included,
    trained on the stories and the training chats once and then read."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = CACHE / f"tokenizer-{vocab}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    print(f"Training a {vocab}-token tokenizer")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(stories() + [render(c) for c in chats("train")], trainer)
    tok.save(str(path))
    return tok


def tokens(split, vocab=VOCAB):
    """The token stream of "stories", "chats" (train) or "test" (the test
    chats), as uint16, tokenized once and then read from the cache."""
    path = CACHE / f"{split}-{vocab}.npy"
    if path.exists():
        return np.load(path)
    tok = tokenizer(vocab)
    if split == "stories":
        texts = ["<|bos|>" + story for story in stories()]
    else:
        texts = [render(c) for c in chats("train" if split == "chats" else "test")]
    ids = [i for encoding in tok.encode_batch(texts) for i in encoding.ids]
    array = np.array(ids, dtype=np.uint16)
    np.save(path, array)
    return array


def test_bytes():
    """UTF-8 bytes in the test chats, special tokens not counted."""
    return sum(len(text.encode("utf-8")) for chat in chats("test") for _, text in chat)


def evaluate(loss_sum, vocab, context, device, batch=16):
    """Bits per byte on the test chats: lower is better. `loss_sum(x, y)`
    returns the summed cross-entropy, in nats, of predicting y from x. The
    test tokens are read in windows of `context`, each scored once."""
    import torch

    ids = torch.from_numpy(tokens("test", vocab).astype(np.int64))
    windows = (len(ids) - 1) // context
    x = ids[: windows * context].view(windows, context)
    y = ids[1 : windows * context + 1].view(windows, context)
    total = 0.0
    with torch.no_grad():
        for i in range(0, windows, batch):
            total += float(loss_sum(x[i : i + batch].to(device), y[i : i + batch].to(device)))
    scored = windows * context / (len(ids) - 1)
    return total / math.log(2) / (test_bytes() * scored)


if __name__ == "__main__":
    for split in ("stories", "chats", "test"):
        print(f"{split}: {len(tokens(split)):,} tokens")
    print(f"Test chats: {test_bytes():,} bytes")
    print(f"Cache: {CACHE}")
