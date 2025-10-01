#!/usr/bin/env python3
"""
Batch 3: Train all architectures on TotalSegmentator dataset.
Estimated time: ~4-5 hours for 3 models (UNet, UNETR, SegResNet)
Run this after Batch 2 (MSD Liver) is complete.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime


DATASET = "totalsegmentator"
ARCHITECTURES = ["unet", "unetr", "segresnet"]

IO_CONFIG = {"in_channels": 1, "out_channels": 2}
MAX_EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 2
OUTPUT_BASE = Path("results/colab_runs")


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"BATCH 3: TotalSegmentator Dataset Training")
    print(f"Architectures: {', '.join(ARCHITECTURES)}")
    print(f"Configuration: MAX_EPOCHS={MAX_EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
    print(f"Estimated time: ~4-5 hours total (~1.5-2 hours per model)")
    print("="*80)
    
    completed_count = 0
    total_runs = len(ARCHITECTURES)
    
    for arch in ARCHITECTURES:
        out_dir = OUTPUT_BASE / f"{DATASET}_{arch}"
        best = out_dir / "best.pth"
        
        if best.exists():
            print(f"\n[SKIP] {DATASET} + {arch} - checkpoint exists at {best}")
            completed_count += 1
            continue
            
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        cmd = [
            sys.executable,
            "-u",
            "scripts/train_model.py",
            "--dataset",
            DATASET,
            "--architecture",
            arch,
            "--in_channels",
            str(IO_CONFIG["in_channels"]),
            "--out_channels",
            str(IO_CONFIG["out_channels"]),
            "--max_epochs",
            str(MAX_EPOCHS),
            "--batch_size",
            str(BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
            "--output_dir",
            str(out_dir),
        ]
        
        print(f"\n{'='*80}")
        print(f"[RUN {completed_count + 1}/{total_runs}] {DATASET} + {arch}")
        print(f"Output: {out_dir}")
        print(f"Log: {log}")
        print(f"Estimated time: ~56 hours")
        print(f"{'='*80}\n")
        
        with open(log, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            proc.wait()
            
        completed_count += 1
        print(f"\n[COMPLETED {completed_count}/{total_runs}] {DATASET} + {arch} (exit code: {proc.returncode})")
        
    print(f"\n{'='*80}")
    print(f"BATCH 3 COMPLETE: {completed_count}/{total_runs} runs finished")
    print(f"ALL BATCHES COMPLETE - Run scripts/evaluate_models.py for final results")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
