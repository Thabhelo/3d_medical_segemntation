"""
Debug script to verify tensor shapes and AsDiscrete behavior.
Run this in Colab to understand what's happening with the labels.
"""
import torch

# Simulate BraTS label from dataloader: [B, 1, H, W, D] with class indices 0-3
B, H, W, D = 2, 96, 96, 96
num_classes = 4

# Simulated labels (class indices)
labels = torch.randint(0, num_classes, (B, 1, H, W, D))
print(f"Original labels shape: {labels.shape}")
print(f"Labels unique values: {torch.unique(labels)}")
print(f"Labels dtype: {labels.dtype}")

# Simulated model predictions (logits)
logits = torch.randn(B, num_classes, H, W, D)
print(f"\nLogits shape: {logits.shape}")

# Test AsDiscrete for predictions
print("\n" + "="*80)
print("PREDICTIONS (logits -> one-hot)")
print("="*80)

try:
    from monai.transforms import AsDiscrete
    
    # This is what we currently do
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    y_pred = post_pred(logits)
    print(f"✓ post_pred output shape: {y_pred.shape}")
    print(f"  Expected: [{B}, {num_classes}, {H}, {W}, {D}]")
except Exception as e:
    print(f"✗ Error: {e}")

# Test AsDiscrete for labels
print("\n" + "="*80)
print("LABELS (class indices [B, 1, H, W, D] -> one-hot)")
print("="*80)

try:
    # Current approach - THIS IS THE PROBLEM
    post_label = AsDiscrete(to_onehot=num_classes)
    y = post_label(labels)
    print(f"✓ post_label output shape: {y.shape}")
    print(f"  Expected: [{B}, {num_classes}, {H}, {W}, {D}]")
    print(f"  Unique values in one-hot: {torch.unique(y)}")
    
    # Check if one-hot is correct
    if y.shape == (B, num_classes, H, W, D):
        # Sum across class dimension should be 1 (one-hot property)
        sum_check = y.sum(dim=1)
        print(f"  Sum across classes (should be all 1s): min={sum_check.min()}, max={sum_check.max()}")
    else:
        print(f"  ✗ SHAPE MISMATCH!")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test with squeezed labels
print("\n" + "="*80)
print("LABELS (squeezed first [B, H, W, D] -> one-hot)")
print("="*80)

try:
    labels_squeezed = labels.squeeze(1)  # Remove channel dim
    print(f"Squeezed labels shape: {labels_squeezed.shape}")
    
    post_label_fixed = AsDiscrete(to_onehot=num_classes)
    y_fixed = post_label_fixed(labels_squeezed)
    print(f"✓ post_label output shape: {y_fixed.shape}")
    print(f"  Expected: [{B}, {num_classes}, {H}, {W}, {D}]")
    print(f"  Unique values in one-hot: {torch.unique(y_fixed)}")
    
    # Check if one-hot is correct
    if y_fixed.shape == (B, num_classes, H, W, D):
        sum_check = y_fixed.sum(dim=1)
        print(f"  Sum across classes (should be all 1s): min={sum_check.min()}, max={sum_check.max()}")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test DiceMetric
print("\n" + "="*80)
print("DICE METRIC COMPUTATION")
print("="*80)

try:
    from monai.metrics import DiceMetric
    
    # Create mock one-hot predictions and labels
    y_pred_onehot = torch.zeros(B, num_classes, H, W, D)
    y_pred_onehot[:, 0, :, :, :] = 1  # All background
    
    y_true_onehot = torch.zeros(B, num_classes, H, W, D)
    y_true_onehot[:, 0, :, :, :] = 1  # All background
    
    dice = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice(y_pred_onehot, y_true_onehot)
    score = dice.aggregate()
    print(f"Dice score (all background, should be NaN or 0): {score}")
    
    # Now with some foreground
    dice.reset()
    y_pred_onehot[:, 1, :50, :, :] = 1
    y_pred_onehot[:, 0, :50, :, :] = 0
    y_true_onehot[:, 1, :50, :, :] = 1
    y_true_onehot[:, 0, :50, :, :] = 0
    
    dice(y_pred_onehot, y_true_onehot)
    score = dice.aggregate()
    print(f"Dice score (perfect match on class 1, should be 1.0): {score}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If AsDiscrete(to_onehot=N) doesn't work correctly with [B, 1, H, W, D],")
print("we need to squeeze labels to [B, H, W, D] before one-hot encoding.")
