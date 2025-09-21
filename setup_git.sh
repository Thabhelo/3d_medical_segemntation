#!/bin/bash
# Git setup script for 3D Medical Segmentation project

echo "Setting up Git repository..."

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: 3D Medical Segmentation project setup

- Implemented dataset loaders for BraTS, MSD Liver, TotalSegmentator
- Added model architectures: UNet, UNETR, SegResNet
- Created training pipeline with MONAI integration
- Added configuration system with YAML files
- Set up evaluation metrics and visualization tools"

# Set up remote repository (will need to be updated with correct URL)
echo "To set up remote repository, run:"
echo "git remote add origin https://github.com/thabhelo/3d_medical_segmentation.git"
echo "git branch -M main"
echo "git push -u origin main"

echo "Git repository initialized successfully!"