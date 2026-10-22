# Divergent Studio templates

Small experiments that train on a Mac in minutes, for
[Divergent Studio](https://divergentlabs.xyz). Open one as a project, say
how long you have, and an agent plans the runs to fit, reads the papers
behind the best direction, grows an experiment tree, and hands you the
winner to try.

| Template | What it trains | The number to beat | Try the winner |
| --- | --- | --- | --- |
| [`pii-redactor`](pii-redactor) | A model that masks names, emails, phone numbers and account numbers | 1,170 of 4,128 entities leaked | `uv run redact.py` |
| [`gpt2-nanochat`](gpt2-nanochat) | A 7M-parameter GPT-2 that chats, from scratch | 1.30 bits per byte | `uv run chat.py` |
| [`ticket-router`](ticket-router) | A router for a bank's support tickets | 109 of 3,080 tickets misrouted | `uv run route.py` |

Each run trains for a fixed number of seconds, so a 5-minute session and an
hour-long one use the same code with more or fewer runs. The agent checks
what the Mac can take first and sizes the model to it.

## Use one

Copy a template into a folder of its own:

```sh
curl -L https://github.com/divergentlabsxyz/studio-templates/archive/refs/heads/main.tar.gz \
  | tar xz --strip-components=1 studio-templates-main/pii-redactor
```

Then, in Divergent Studio, create a project on that folder. Its first
conversation starts by itself, from the template's `.studio-start.md`.

Needs a Mac with Apple silicon and [uv](https://docs.astral.sh/uv/).

## How a template is made

| File | What it is |
| --- | --- |
| `.studio-start.md` | The brief Studio sends with the project's first prompt: the task, the yardstick, budgets by time, directions, and the session's steps. Only that first conversation reads it, so forks and later conversations start with a short context. |
| `prepare.py` | Setup: packages and data, into `~/.cache/divergent-studio/<template>/`. |
| `train.py` | One run: trains for `--budget` seconds, writes its result to `DIVERGENT_RESULT_PATH` and its points to `DIVERGENT_TELEMETRY_PATH`. |
| `telemetry.toml` | The run's charts in Studio. |
| `machine.py` | What this Mac can give a run now, and the size tier that fits. |
| `studio.py` | The result, the points, and the clock. |

`machine.py`, `studio.py` and the steps at the end of each
`.studio-start.md` are shared. Edit them in `shared/` and run
`python3 scripts/sync.py`; CI fails on a template that drifted.

## Licences

The code is MIT. The data keeps its own licence, named in each template's
README.
