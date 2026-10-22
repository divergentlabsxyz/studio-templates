"""What this Mac can give a run right now, and the size tier that fits it.

Standard library only, so it runs before anything is installed:

    python3 machine.py          # for a person
    python3 machine.py --json   # for an agent

The tier is a starting point, not a verdict. Each template's AGENTS.md
says what `small`, `medium` and `large` mean for its model.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys

GIB = 1 << 30


def sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def command(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def free_memory():
    """Bytes a new process can take without swapping: free, inactive,
    speculative and purgeable pages, as vm_stat counts them."""
    text = command("vm_stat")
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    size = int(page.group(1))
    pages = 0
    for key in ("free", "inactive", "speculative", "purgeable"):
        match = re.search(rf"Pages {key}:\s+(\d+)", text)
        if match:
            pages += int(match.group(1))
    return pages * size


def gpu_cores():
    text = command("ioreg", "-rc", "IOAccelerator", "-d", "1")
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', text)
    return int(match.group(1)) if match else None


def on_battery():
    return "Battery Power" in command("pmset", "-g", "batt")


def low_power_mode():
    match = re.search(r"lowpowermode\s+(\d)", command("pmset", "-g"))
    return bool(match and match.group(1) == "1")


def tier(memory_gib, free_gib, battery, low_power):
    """small under 16 GB, medium to 32 GB, large above; one step down when
    less than 4 GB is free, and one more on battery or in Low Power Mode."""
    steps = ["small", "medium", "large"]
    index = 0 if memory_gib < 15 else 1 if memory_gib < 31 else 2
    if free_gib is not None and free_gib < 4:
        index -= 1
    if battery or low_power:
        index -= 1
    return steps[max(index, 0)]


def main():
    memory = int(sysctl("hw.memsize") or 0)
    free = free_memory()
    battery = on_battery()
    low_power = low_power_mode()
    disk = shutil.disk_usage(os.path.expanduser("~")).free
    memory_gib = memory / GIB
    free_gib = free / GIB if free is not None else None
    facts = {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "performance_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0),
        "efficiency_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0),
        "gpu_cores": gpu_cores(),
        "memory_gb": round(memory_gib, 1),
        "free_memory_gb": round(free_gib, 1) if free_gib is not None else None,
        "free_disk_gb": round(disk / GIB, 1),
        "on_battery": battery,
        "low_power_mode": low_power,
        "apple_silicon": platform.machine() == "arm64",
        "tier": tier(memory_gib, free_gib, battery, low_power),
    }
    if "--json" in sys.argv:
        print(json.dumps(facts, indent=2))
        return
    gpu = f"{facts['gpu_cores']}-core GPU" if facts["gpu_cores"] else "GPU"
    print(f"{facts['chip']}, {facts['performance_cores']}+{facts['efficiency_cores']} cores, {gpu}")
    print(f"Memory: {facts['memory_gb']} GB, {facts['free_memory_gb']} GB free")
    print(f"Disk: {facts['free_disk_gb']} GB free")
    power = "battery" if battery else "power adapter"
    print(f"Power: {power}{', Low Power Mode' if low_power else ''}")
    print(f"Tier: {facts['tier']}")
    if not facts["apple_silicon"]:
        print("This is not Apple silicon: there is no mps device, and runs train on the CPU.")
    if facts["free_disk_gb"] < 5:
        print("Less than 5 GB of disk is free: free some before setting up.")


if __name__ == "__main__":
    main()
