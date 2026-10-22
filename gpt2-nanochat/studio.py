"""What a run tells Divergent Studio, and the clock it trains against.

Studio tells a run DIVERGENT_RESULT_PATH and DIVERGENT_TELEMETRY_PATH. The
result is one JSON object, written once at the end; telemetry is one JSON
object per line, appended as the run goes. Outside Studio neither is set:
the result is printed, and points are dropped.
"""

import json
import os
import time
from pathlib import Path


def cache_dir(template):
    """Where data and downloads live, outside the code, so every run's
    snapshot finds them without fetching again."""
    root = os.environ.get("DIVERGENT_CACHE") or Path.home() / ".cache" / "divergent-studio"
    path = Path(root) / template
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir():
    """Where a run keeps what it makes, such as its checkpoint: beside its
    result in Studio, and `out/` otherwise."""
    target = os.environ.get("DIVERGENT_RESULT_PATH")
    path = Path(target).parent if target else Path("out")
    path.mkdir(parents=True, exist_ok=True)
    return path


def point(**values):
    """One point for the charts: point(step=120, loss=0.41)."""
    target = os.environ.get("DIVERGENT_TELEMETRY_PATH")
    if not target:
        return
    with open(target, "a") as file:
        file.write(json.dumps(values) + "\n")


def result(values):
    """The run's answer. Call it once, last."""
    text = json.dumps(values, indent=2)
    print(text)
    target = os.environ.get("DIVERGENT_RESULT_PATH")
    if target:
        Path(target).write_text(text)


class Budget:
    """Wall-clock seconds for training. Every node in a tree trains for the
    same number, so a node that learns faster wins rather than one that
    trained longer."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = time.monotonic()

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def spent(self):
        return self.elapsed >= self.seconds

    @property
    def fraction(self):
        return min(self.elapsed / self.seconds, 1.0)
