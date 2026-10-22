# GPT-2 nanochat

Train a small GPT-2 from scratch on this Mac until it holds a conversation.
Karpathy's [nanochat](https://github.com/karpathy/nanochat) does it on eight
H100s for $100; this is the same idea at a size a laptop trains in two
minutes. The first model is 7 million parameters and answers like this:

```text
you> Can you help me plan a picnic?
bot> My dog is a good idea to wash your dog's teeth when your dog gets
     around and wash your mouth.
```

The agent's job is to make it learn faster: from modern GPT details, the
optimizer, the model's shape, or how it splits the time between stories
and chats.

## Start

1. Open this folder in Divergent Studio as a new project. Its first
   conversation starts by itself.
2. Answer the agent's two questions: how long you have, and what to go
   after: the lowest test loss, the best answers, or the fastest training.

The agent checks what your Mac can take, plans the runs to fit your time,
reads the papers behind the most promising direction, and ends with the
winner and the command to talk to it:

```sh
uv run chat.py --checkpoint <path>
```

## Needs

A Mac with Apple silicon and [uv](https://docs.astral.sh/uv/). About 1 GB
of disk for PyTorch, and 25 MB of data.

## Data

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  (Eldan and Li), its GPT-4 validation file, CDLA-Sharing-1.0.
- [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k)
  (Hugging Face), Apache-2.0.
