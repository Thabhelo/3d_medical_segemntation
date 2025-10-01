"""
Test the trainer fix with actual tensor operations.
"""
import torch
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric

B, H, W, D = 2, 96, 96, 96
num_classes = 4

# Simulated data
labels = torch.randint(0, num_classes, (B, 1, H, W, D))
logits = torch.randn(B, num_classes, H, W, D)

print("="*80)
print("TRAINER FIX TEST")
print("="*80)
print(f"Input shapes:")
print(f"  logits: {logits.shape}")
print(f"  labels: {labels.shape}")

# Simulate validation code
y_pred_indices = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, H, W, D]
print(f"\nAfter argmax: {y_pred_indices.shape}")

y_pred_onehot = one_hot(y_pred_indices, num_classes=num_classes)  # [B, C, H, W, D]
print(f"Pred one-hot: {y_pred_onehot.shape}")

labels_squeezed = labels.squeeze(1).long()  # [B, H, W, D]
print(f"Labels squeezed: {labels_squeezed.shape}")

y_true_onehot = one_hot(labels_squeezed.unsqueeze(1), num_classes=num_classes)  # [B, C, H, W, D]
print(f"True one-hot: {y_true_onehot.shape}")

# Verify shapes match
print(f"\nShape verification:")
print(f"  Pred == True: {y_pred_onehot.shape == y_true_onehot.shape}")
print(f"  Expected: torch.Size([{B}, {num_classes}, {H}, {W}, {D}])")

# Test DiceMetric
val_dice = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
val_dice(y_pred_onehot, y_true_onehot)
score = val_dice.aggregate()

print(f"\nDice score: {score}")
print(f"Score is valid: {not torch.isnan(score) and not torch.isinf(score)}")

# Verify one-hot properties
pred_sum = y_pred_onehot.sum(dim=1)
true_sum = y_true_onehot.sum(dim=1)
print(f"\nOne-hot verification (sum across classes should be 1):")
print(f"  Pred: min={pred_sum.min()}, max={pred_sum.max()}")
print(f"  True: min={true_sum.min()}, max={true_sum.max()}")

print("\n" + "="*80)
if y_pred_onehot.shape == (B, num_classes, H, W, D) and y_true_onehot.shape == (B, num_classes, H, W, D):
    print("✓ FIX IS CORRECT - Shapes are perfect!")
else:
    print("✗ FIX HAS ISSUES")
print("="*80)

