#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained 3D medical segmentation models.

This script evaluates all trained model checkpoints and generates:
- Per-model metrics (Dice, IoU, Hausdorff distance)
- Comparative analysis across architectures and datasets
- Summary tables and visualizations
- Inference speed benchmarking
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import torch
import numpy as np
import time
from typing import Dict, List, Any

# Add project root to path
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.utils import create_dataloaders
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.trainer import Trainer


def benchmark_inference(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> Dict[str, float]:
    """Benchmark inference speed with proper CUDA synchronization."""
    device = next(model.parameters()).device
    
    # Warmup
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # Convert to ms
    
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    
    return {
        "latency_ms_per_volume_mean": mean_ms,
        "latency_ms_per_volume_std": std_ms,
        "throughput_vol_s": 1000.0 / mean_ms,
    }


def get_environment_info() -> Dict[str, Any]:
    """Get GPU and environment information."""
    info = {
        "num_gpus": torch.cuda.device_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        })
    
    return info


def evaluate_model_checkpoint(
    checkpoint_path: Path,
    dataset: str,
    architecture: str,
    data_root: str,
    in_channels: int,
    out_channels: int,
    benchmark: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model checkpoint."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_model(
        architecture=architecture,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    result = {
        "dataset": dataset,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
    }
    
    # Standard evaluation
    if not benchmark:
        # Create validation dataloader
        _, val_loader = create_dataloaders(
            dataset_name=dataset,
            root_dir=data_root,
            batch_size=1,
            num_workers=0,
            patch_size=(128, 128, 128),
        )
        
        # Initialize trainer for evaluation
        loss_fn = get_loss("dice_ce")
        optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            output_dir=checkpoint_path.parent,
            max_epochs=1,
            num_classes=out_channels,
        )
        
        # Evaluate
        val_dice = trainer.validate(val_loader)
        result["val_dice"] = val_dice
    
    # Inference benchmarking
    if benchmark:
        # Create validation dataloader to get real evaluation data
        _, val_loader = create_dataloaders(
            dataset_name=dataset,
            root_dir=data_root,
            batch_size=1,
            num_workers=0,
            patch_size=(128, 128, 128),
        )
        
        # Get a real sample from the evaluation set
        sample_batch = next(iter(val_loader))
        sample_input = sample_batch["image"].to(device)
        
        # Run benchmark
        benchmark_results = benchmark_inference(model, sample_input)
        result.update(benchmark_results)
        
        # Add environment info
        result.update(get_environment_info())
    
    return result


def main():
    """Main evaluation function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate trained 3D medical segmentation models")
    # By default we run BOTH evaluation and inference benchmarking
    parser.add_argument("--no-eval", action="store_true", help="Skip validation evaluation")
    parser.add_argument("--no-benchmark-inference", action="store_true", help="Skip inference speed benchmarking")
    args = parser.parse_args()
    
    # Auto-detect dataset root based on environment
    import os
    is_colab = os.path.exists('/content')
    base_data_root = "/content/drive/MyDrive/datasets" if is_colab else str(Path.home() / "Downloads" / "datasets")
    
    # Configuration
    RESULTS_DIR = Path("results/colab_runs")
    # Ensure output directory exists so results JSON is always written
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_CONFIG = {
        "brats": {"in_channels": 4, "out_channels": 4, "data_root": base_data_root},
        "msd_liver": {"in_channels": 1, "out_channels": 3, "data_root": f"{base_data_root}/MSD/Task03_Liver"},
        # TotalSegmentator has 118 anatomical classes
        "totalsegmentator": {"in_channels": 1, "out_channels": 118, "data_root": f"{base_data_root}/TotalSegmentator"},
    }
    ARCHITECTURES = ["unet", "unetr", "segresnet"]
    
    print(f"Environment: {'Colab' if is_colab else 'Local'}")
    print(f"Base data root: {base_data_root}")
    run_eval = not args.no_eval
    run_benchmark = not args.no_benchmark_inference
    print(f"Modes: eval={'on' if run_eval else 'off'}, benchmark={'on' if run_benchmark else 'off'}")
    
    # Find all checkpoints
    checkpoints = list(RESULTS_DIR.glob("*/best.pth"))
    print(f"Found {len(checkpoints)} model checkpoints")
    
    # Evaluate each checkpoint
    eval_results = []
    bench_results = []
    for checkpoint_path in sorted(checkpoints):
        # Parse dataset and architecture from path
        parts = checkpoint_path.parent.name.split("_")
        if len(parts) >= 2:
            dataset = "_".join(parts[:-1])  # Handle multi-word datasets like "msd_liver"
            architecture = parts[-1]
            
            if dataset in DATASETS_CONFIG and architecture in ARCHITECTURES:
                print(f"Evaluating {dataset} + {architecture}...")
                
                config = DATASETS_CONFIG[dataset]
                try:
                    if run_eval:
                        eval_result = evaluate_model_checkpoint(
                            checkpoint_path=checkpoint_path,
                            dataset=dataset,
                            architecture=architecture,
                            data_root=config["data_root"],
                            in_channels=config["in_channels"],
                            out_channels=config["out_channels"],
                            benchmark=False,
                        )
                        eval_results.append(eval_result)
                        print(f"  Dice: {eval_result['val_dice']:.4f}")
                    if run_benchmark:
                        bench_result = evaluate_model_checkpoint(
                            checkpoint_path=checkpoint_path,
                            dataset=dataset,
                            architecture=architecture,
                            data_root=config["data_root"],
                            in_channels=config["in_channels"],
                            out_channels=config["out_channels"],
                            benchmark=True,
                        )
                        bench_results.append(bench_result)
                        print(f"  Latency: {bench_result['latency_ms_per_volume_mean']:.1f}±{bench_result['latency_ms_per_volume_std']:.1f} ms")
                        print(f"  Throughput: {bench_result['throughput_vol_s']:.2f} vol/s")
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"Skipping unknown combination: {dataset} + {architecture}")
    
    # Save detailed results and print summaries
    if run_eval:
        eval_df = pd.DataFrame(eval_results)
        if not eval_df.empty:
            eval_out = RESULTS_DIR / "evaluation_results.json"
            with open(eval_out, "w") as f:
                json.dump(eval_results, f, indent=2)
            try:
                legacy_eval = Path("results") / "evaluation_results.json"
                legacy_eval.parent.mkdir(parents=True, exist_ok=True)
                with open(legacy_eval, "w") as f:
                    json.dump(eval_results, f, indent=2)
                print(f"\nDetailed evaluation results saved to: {eval_out} and {legacy_eval}")
            except Exception:
                print(f"\nDetailed evaluation results saved to: {eval_out}")
            
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            pivot_df = eval_df.pivot(index="dataset", columns="architecture", values="val_dice")
            print("\nValidation Dice Scores:")
            print(pivot_df.round(4))
            print("\nModel Parameters (millions):")
            param_pivot = eval_df.pivot(index="dataset", columns="architecture", values="num_parameters")
            print((param_pivot / 1e6).round(2))
            print("\nModel Size (MB):")
            size_pivot = eval_df.pivot(index="dataset", columns="architecture", values="model_size_mb")
            print(size_pivot.round(1))
            print("\nBest Models by Dataset:")
            for dataset_name in eval_df["dataset"].unique():
                dataset_df = eval_df[eval_df["dataset"] == dataset_name]
                best_idx = dataset_df["val_dice"].idxmax()
                best_model = dataset_df.loc[best_idx]
                print(f"  {dataset_name}: {best_model['architecture']} (Dice: {best_model['val_dice']:.4f})")

    if run_benchmark:
        bench_df = pd.DataFrame(bench_results)
        if not bench_df.empty:
            bench_out = RESULTS_DIR / "inference_benchmark.json"
            with open(bench_out, "w") as f:
                json.dump(bench_results, f, indent=2)
            try:
                legacy_bench = Path("results") / "inference_benchmark.json"
                legacy_bench.parent.mkdir(parents=True, exist_ok=True)
                with open(legacy_bench, "w") as f:
                    json.dump(bench_results, f, indent=2)
                print(f"\nDetailed inference results saved to: {bench_out} and {legacy_bench}")
            except Exception:
                print(f"\nDetailed inference results saved to: {bench_out}")

            print("\n" + "="*80)
            print("INFERENCE BENCHMARK SUMMARY")
            print("="*80)
            print("\nInference Performance:")
            perf_df = bench_df[["dataset", "architecture", "latency_ms_per_volume_mean", "throughput_vol_s", "gpu_name", "num_gpus"]].copy()
            perf_df["latency_ms_per_volume_mean"] = perf_df["latency_ms_per_volume_mean"].round(1)
            perf_df["throughput_vol_s"] = perf_df["throughput_vol_s"].round(2)
            print(perf_df.to_string(index=False))
            print("\nLatency (ms/volume) by Dataset and Architecture:")
            latency_pivot = bench_df.pivot(index="dataset", columns="architecture", values="latency_ms_per_volume_mean")
            print(latency_pivot.round(1))
            print("\nThroughput (vol/s) by Dataset and Architecture:")
            throughput_pivot = bench_df.pivot(index="dataset", columns="architecture", values="throughput_vol_s")
            print(throughput_pivot.round(2))
            if "gpu_name" in bench_df.columns:
                env_info = bench_df[["gpu_name", "num_gpus", "torch_version", "cuda_version"]].iloc[0]
                print(f"\nEnvironment: {env_info['gpu_name']} (x{env_info['num_gpus']}), PyTorch {env_info['torch_version']}, CUDA {env_info['cuda_version']}")

    # Unified results file containing evaluation + inference + environment per model
    if (run_eval and 'eval_df' in locals() and not eval_df.empty) or (run_benchmark and 'bench_df' in locals() and not bench_df.empty):
        # Merge on dataset+architecture (and optionally checkpoint_path if present in both)
        def keyify(row):
            return (row.get('dataset'), row.get('architecture'), row.get('checkpoint_path'))
        combined_map = {}
        if run_eval and 'eval_df' in locals() and not eval_df.empty:
            for rec in eval_results:
                combined_map[keyify(rec)] = dict(rec)
        if run_benchmark and 'bench_df' in locals() and not bench_df.empty:
            for rec in bench_results:
                k = keyify(rec)
                if k in combined_map:
                    combined_map[k].update(rec)
                else:
                    combined_map[k] = dict(rec)
        combined_list = list(combined_map.values())
        unified_out = RESULTS_DIR / "evaluation_full.json"
        with open(unified_out, "w") as f:
            json.dump(combined_list, f, indent=2)
        try:
            legacy_unified = Path("results") / "evaluation_full.json"
            legacy_unified.parent.mkdir(parents=True, exist_ok=True)
            with open(legacy_unified, "w") as f:
                json.dump(combined_list, f, indent=2)
            print(f"\nUnified results saved to: {unified_out} and {legacy_unified}")
        except Exception:
            print(f"\nUnified results saved to: {unified_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
