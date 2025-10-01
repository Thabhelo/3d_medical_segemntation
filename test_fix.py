"""
Test the actual fix for one-hot encoding.
"""
import torch
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

B, H, W, D = 2, 96, 96, 96
num_classes = 4

# Simulated data
labels = torch.randint(0, num_classes, (B, 1, H, W, D))
logits = torch.randn(B, num_classes, H, W, D)

print("="*80)
print("PROPOSED FIX")
print("="*80)

# For predictions: argmax then one-hot
post_pred = AsDiscrete(argmax=True)
y_pred_indices = post_pred(logits)
print(f"After argmax: {y_pred_indices.shape} (should be [B, H, W, D])")

# Convert to one-hot manually
y_pred_onehot = one_hot(y_pred_indices.unsqueeze(1).long(), num_classes=num_classes)
print(f"After one_hot: {y_pred_onehot.shape} (should be [B, {num_classes}, H, W, D])")

# For labels: squeeze, ensure long dtype, then one-hot
labels_squeezed = labels.squeeze(1).long()  # [B, H, W, D]
print(f"Labels squeezed: {labels_squeezed.shape}")

y_true_onehot = one_hot(labels_squeezed.unsqueeze(1), num_classes=num_classes)
print(f"Labels one-hot: {y_true_onehot.shape} (should be [B, {num_classes}, H, W, D])")

# Test Dice
dice = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice(y_pred_onehot, y_true_onehot)
score = dice.aggregate()
print(f"\nDice score: {score}")
print("✓ Fix works!" if score.numel() > 0 else "✗ Still broken")

print("\n" + "="*80)
print("ALTERNATIVE: Let DiceMetric handle it with to_onehot_y parameter")
print("="*80)

dice2 = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Just argmax predictions, keep labels as indices
y_pred_indices2 = post_pred(logits)  # [B, H, W, D] with class indices
labels_indices = labels.squeeze(1).long()  # [B, H, W, D] with class indices

print(f"Pred indices shape: {y_pred_indices2.shape}")
print(f"Label indices shape: {labels_indices.shape}")

# Check if DiceMetric can handle class indices directly
try:
    dice2(y_pred_indices2.unsqueeze(1), labels_indices.unsqueeze(1))
    score2 = dice2.aggregate()
    print(f"Dice score (class indices): {score2}")
except Exception as e:
    print(f"✗ Class indices don't work: {e}")

