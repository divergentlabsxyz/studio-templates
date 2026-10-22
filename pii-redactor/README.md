# PII redactor

Train a model that masks names, emails, phone numbers and account numbers in
customer messages. It trains from scratch on this Mac in under a minute a
run, with nothing to download but PyTorch.

```text
Hi, this is Ada Lovelace, call me on 06-12345678 or mail ada@hey.com
Hi, this is [NAME], call me on [PHONE] or mail [EMAIL]
```

A page of regular expressions catches the emails and the numbers, and
leaks every name. The first model leaks about 28% of what it should mask,
on names and phrasings it has never seen. The agent's job is to get that
down without masking everything else.

## Start

1. Open this folder in Divergent Studio as a new project. Its first
   conversation starts by itself.
2. Answer the agent's two questions: how long you have, and what matters
   more: fewer leaks, or masking only what is PII.

The agent checks what your Mac can take, plans the runs to fit your time,
reads the papers behind the most promising direction, and ends with the
winner and the command to try it:

```sh
uv run redact.py --checkpoint <path> "Refund €40 to NL91 ABNA 0417 1643 00, thanks, Bram"
```

## Needs

A Mac with Apple silicon and [uv](https://docs.astral.sh/uv/). About 1 GB
of disk for PyTorch.

## Data

Written by `prepare.py` from templates and name lists in this folder. No
real person's data is in it.
