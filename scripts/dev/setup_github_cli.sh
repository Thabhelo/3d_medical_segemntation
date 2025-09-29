#!/bin/bash
# GitHub CLI setup script

echo "Setting up GitHub CLI..."

# Check if GitHub CLI is already installed
if command -v gh &> /dev/null; then
    echo "GitHub CLI is already installed."
    gh --version
else
    echo "Installing GitHub CLI..."
    
    # Download and install GitHub CLI
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh -y
fi

echo "GitHub CLI setup complete!"
echo "To authenticate, run: gh auth login"
echo "To create a repository, run: gh repo create 3d_medical_segmentation --public"