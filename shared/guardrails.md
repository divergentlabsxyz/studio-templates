## How to run this session

These steps come on top of Studio's own instructions. Where the brief above
and these disagree, these win.

### 1. Ask how long

Before anything else, call `ask_user` once with two questions:

1. "How long should this take?", detail "Setup, every run and the write-up
   fit in it.", options "5 minutes", "15 minutes", "30 minutes", "An hour".
2. The brief's question about what to go after, with its options.

Read the minutes out of an answer in words; with no time given, plan for 15
minutes and say so. Then start the clock: `date +%s > .session-start`. Ask
nothing else until the end unless a step below says to.

### 2. Check the Mac

Run `python3 machine.py`. Set `SIZE` in `train.py` to the tier it prints,
in the baseline's worktree. Under 5 GB of free disk: say what is needed and
stop. On battery or in Low Power Mode, the tier has already stepped down;
say once that runs are slower unplugged.

### 3. Set up once

Make the folder a git repository with a commit if it is not one, then run
`uv run prepare.py`. It installs the packages and fetches the data into
`~/.cache/divergent-studio/<template>/`. It is setup: no node. If `uv` is
missing, ask whether to install it (`curl -LsSf
https://astral.sh/uv/install.sh | sh`), and install it only on a yes.

### 4. Plan, then run the baseline

A run trains for a fixed number of seconds, set in the run command:
`uv run train.py --budget <seconds>`, from the brief's table for the
person's time. Every node inherits the command, so every node trains for
the same seconds and wins by learning faster, not longer. Never work around
it in `train.py`. Say the plan in one line ("15 minutes: a baseline and
about 5 nodes of 90 seconds, one at a time"), then create and run the
baseline.

### 5. Research while the baseline trains

Every session does this, whatever its length, before the first round of
nodes:

1. Search with `search_papers` for the brief's paper and for the direction
   that fits the person's goal: one or two specific queries.
2. Read the abstracts. With 30 minutes or more, `download_paper` the most
   useful one and read its method.
3. `save_paper` the one or two the next round builds on.
4. Pick the first round's nodes from what they say, and `cite_paper` each
   from the node that uses its idea.
5. Tell the person in two or three lines what you found and what you will
   try because of it.

### 6. One run at a time

Runs share the GPU, the memory and the CPU; two at once each train slower,
and the budget measures the crowding, not the idea. Launch one,
`wait_for_runs`, read it with `get_run`, launch the next. Make the next
node's change while a run trains.

### 7. Watch the clock

Before every launch, check the minutes gone:
`echo $(( ($(date +%s) - $(cat .session-start)) / 60 ))`. Launch only when
the budget, the brief's overhead per run and two minutes for the write-up
fit in what is left. At the deadline, wait for a run in flight if it ends
within a minute, and `cancel_run` it otherwise.

### 8. Stay in bounds

- The brief names the yardstick. Never change it in a node, never train on
  what it scores. If it looks wrong, tell the person.
- Download nothing during a run: fetch a new package or model in the
  worktree first, so the budget measures training.
- A run killed for memory (exit code 137, "MPS backend out of memory")
  answered nothing: step `SIZE` down one tier in that node and run again.
- Write nothing outside this folder and the cache.
- Keep the demo working with every node's checkpoint: a node that changes
  how the model is built saves what rebuilds it in the checkpoint's
  `config`.

### 9. End on time

Stop at the deadline, when the goal is met, or after three runs in a row
that do not beat the best. Write the winner's notes with
`set_experiment_notes`. End with a table of the nodes (title, change,
headline, difference from the baseline), the winner, the demo command with
the winner's checkpoint, and the one experiment to run next.
