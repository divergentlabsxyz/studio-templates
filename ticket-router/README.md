# Ticket router

Sort a bank's support tickets into the queues its team works from: cards,
fraud, transfers, top-ups, cash, fees, and accounts. The first router
misroutes 109 of 3,080 test tickets and misses 14 fraud reports, in under a
second of training.

```text
Cash and ATMs        card_swallowed      my card got eaten by the atm
Fraud and security   compromised_card    someone used my card in Lisbon, I've never been there
```

The agent's job is to get the misroutes down: better features, a sentence
encoder, fine-tuning it, or a floor that never lets fraud slip.

## Start

1. To route to your own queues, edit `queues.toml` first.
2. Open this folder in Divergent Studio as a new project. Its first
   conversation starts by itself.
3. Answer the agent's two questions: how long you have, and what matters
   most: fewest misroutes, never missing fraud, or a router that is fast
   and small.

The agent checks what your Mac can take, plans the runs to fit your time,
reads the papers behind the most promising direction, and ends with the
winner and the command to try it:

```sh
uv run route.py --checkpoint <path> "I was charged twice for one coffee"
```

## Needs

A Mac with Apple silicon and [uv](https://docs.astral.sh/uv/). About 1.5 GB
of disk for PyTorch and the encoder.

## Data

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (Casanueva
et al., PolyAI), CC BY 4.0. The encoder is
[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (BAAI), MIT.
