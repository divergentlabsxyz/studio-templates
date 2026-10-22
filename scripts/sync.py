"""Copies what every template shares from shared/ into each template.

    python3 scripts/sync.py          # write
    python3 scripts/sync.py --check  # exit 1 if a template has drifted

A template is copied out of this repository on its own, so each carries its
own machine.py, studio.py and guardrails. They are edited in shared/ alone.
The guardrails replace what sits between the two markers in
.studio-start.md, the brief Studio sends with a project's first prompt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "shared"
TEMPLATES = ["pii-redactor", "gpt2-nanochat", "ticket-router"]
FILES = ["machine.py", "studio.py"]
START = ".studio-start.md"
BEGIN = "<!-- guardrails: begin. Edited in shared/guardrails.md; run scripts/sync.py. -->"
END = "<!-- guardrails: end -->"


def wanted(template):
    out = {name: (SHARED / name).read_text() for name in FILES}
    brief = (ROOT / template / START).read_text()
    if BEGIN not in brief or END not in brief:
        sys.exit(f"{template}/{START} has no guardrails markers.")
    head, rest = brief.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    guardrails = (SHARED / "guardrails.md").read_text().replace("<template>", template)
    out[START] = f"{head}{BEGIN}\n{guardrails.strip()}\n{END}{tail}"
    return out


def main():
    check = "--check" in sys.argv
    drifted = []
    for template in TEMPLATES:
        for name, text in wanted(template).items():
            path = ROOT / template / name
            if path.exists() and path.read_text() == text:
                continue
            if check:
                drifted.append(f"{template}/{name}")
            else:
                path.write_text(text)
                print(f"wrote {template}/{name}")
    if drifted:
        sys.exit("Out of date, run scripts/sync.py: " + ", ".join(drifted))


if __name__ == "__main__":
    main()
