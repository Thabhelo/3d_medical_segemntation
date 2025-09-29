#!/usr/bin/env python3
from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DATASETS = ["brats", "msd_liver", "totalsegmentator"]
ARCHITECTURES = ["unet", "unetr", "segresnet"]

# Per-dataset IO settings
IO = {
    "brats": {"in_channels": 4, "out_channels": 4},
    "msd_liver": {"in_channels": 1, "out_channels": 3},
    "totalsegmentator": {"in_channels": 1, "out_channels": 2},
}

MAX_EPOCHS = 2
BATCH_SIZE = 2
NUM_WORKERS = 2
OUTPUT_BASE = Path("results/colab_runs")


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for ds, arch in itertools.product(DATASETS, ARCHITECTURES):
        out_dir = OUTPUT_BASE / f"{ds}_{arch}"
        best = out_dir / "best.pth"
        if best.exists():
            print(f"SKIP  {ds:16} {arch:11}  ({best})")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        io = IO[ds]
        log = out_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [
            sys.executable,
            "-u",
            "scripts/train_model.py",
            "--dataset",
            ds,
            "--architecture",
            arch,
            "--in_channels",
            str(io["in_channels"]),
            "--out_channels",
            str(io["out_channels"]),
            "--max_epochs",
            str(MAX_EPOCHS),
            "--batch_size",
            str(BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
            "--output_dir",
            str(out_dir),
        ]
        print("RUN   ", " ".join(cmd))
        with open(log, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:  # type: ignore
                print(line, end="")
                lf.write(line)
            proc.wait()
        print("DONE  ", ds, arch, "exit_code=", proc.returncode)


if __name__ == "__main__":
    main()


