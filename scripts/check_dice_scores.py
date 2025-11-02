#!/usr/bin/env python3
"""Check dice scores from checkpoint files."""

import sys
from pathlib import Path
import torch

def check_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location='cpu')
        epoch = ckpt.get('epoch', 'N/A')
        dice = ckpt.get('metrics', {}).get('val_dice', 'N/A')
        return {'path': str(path), 'epoch': epoch, 'dice': dice}
    except Exception as e:
        return {'path': str(path), 'error': str(e)}

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else list(Path('results/colab_runs').glob('**/best.pth'))
    
    if not paths:
        print("No checkpoints found")
        sys.exit(1)
    
    print(f"\nFound {len(paths)} checkpoint(s)\n")
    print(f"{'Checkpoint':<50} {'Epoch':<10} {'Dice':<10}")
    print("-" * 70)
    
    results = [check_checkpoint(Path(p)) for p in paths]
    for r in sorted(results, key=lambda x: x['path']):
        if 'error' in r:
            print(f"{Path(r['path']).name:<50} ERROR: {r['error']}")
        else:
            dice_str = f"{r['dice']:.4f}" if r['dice'] != 'N/A' else 'N/A'
            print(f"{Path(r['path']).parent.name:<50} {r['epoch']:<10} {dice_str:<10}")
    
    valid = [r for r in results if 'error' not in r and r['dice'] != 'N/A']
    if valid:
        best = max(valid, key=lambda x: x['dice'])
        print(f"\nBest: {Path(best['path']).parent.name} - Dice: {best['dice']:.4f}")
