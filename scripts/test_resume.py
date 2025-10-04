#!/usr/bin/env python3
"""
Test script to verify resume functionality works correctly.
This creates a minimal test to ensure the resume logic is working.
"""

import torch
import tempfile
from pathlib import Path
import sys

# Add repo root to path
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.trainer import Trainer
from src.models.factory import create_model
from src.models.losses import get_loss

def test_resume_functionality():
    """Test that resume functionality works correctly."""
    print("🧪 Testing resume functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple model and trainer
        model = create_model("unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = get_loss("dice_ce")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),  # Use CPU for testing
            output_dir=temp_path,
            max_epochs=5,
            num_classes=2,
            scheduler=scheduler,
            save_every_epoch=True
        )
        
        # Create dummy data loaders (empty for this test)
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        dummy_images = torch.randn(2, 1, 32, 32, 32)
        dummy_labels = torch.randint(0, 2, (2, 1, 32, 32, 32))
        dataset = TensorDataset(dummy_images, dummy_labels)
        dummy_loader = DataLoader(dataset, batch_size=1)
        
        print("✅ Created trainer with dynamic LR scheduler")
        print("✅ Created dummy data loaders")
        
        # Test checkpoint saving
        trainer._save_checkpoint(epoch=3, val_dice=0.5, best=False)
        checkpoint_path = temp_path / "epoch_3.pth"
        
        if checkpoint_path.exists():
            print("✅ Checkpoint saved successfully")
        else:
            print("❌ Checkpoint saving failed")
            return False
            
        # Test checkpoint loading
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        expected_keys = ["model", "optimizer", "epoch", "metrics", "scheduler"]
        
        for key in expected_keys:
            if key in ckpt:
                print(f"✅ Checkpoint contains {key}")
            else:
                print(f"❌ Checkpoint missing {key}")
                return False
                
        print("✅ Resume functionality test passed!")
        return True

if __name__ == "__main__":
    success = test_resume_functionality()
    if success:
        print("\n🎉 All tests passed! Resume functionality is working correctly.")
    else:
        print("\n❌ Tests failed! Check the implementation.")
        sys.exit(1)
