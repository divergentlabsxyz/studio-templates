# Divergent Studio templates

Each top-level folder but `shared/`, `scripts/` and `.github/` is a template:
a project of its own once copied out, with no reference to this repository.

- `machine.py`, `studio.py` and the steps between the guardrails markers in
  each `.studio-start.md` are copies. Edit them in `shared/`, then run
  `python3 scripts/sync.py`.
- `.studio-start.md` is sent once, with a project's first prompt. Keep it
  short: everything in it sits in that conversation's context for good.
  Never add an `AGENTS.md` or `CLAUDE.md` to a template, since every
  conversation and fork would load it.
- A template's folder name is its id in Studio (`Starter` in
  `apps/studio/App/Onboarding.swift`).
- Every number a brief or a README quotes is measured: change the code and
  run it again before changing the number.
- Before committing, run a template's `uv run prepare.py` and
  `uv run train.py --budget 10`.
