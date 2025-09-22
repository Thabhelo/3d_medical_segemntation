#!/bin/bash

# 3D Medical Segmentation - Comprehensive Setup Script
# Author: Automated Setup System
# Description: One-command setup for local development or Google Colab

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Setup logging
SETUP_LOG="setup_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging function
log() {
    echo -e "$1" | tee -a "$SETUP_LOG"
}

# Progress bar function
progress_bar() {
    local current=$1
    local total=$2
    local step_name="$3"
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))

    printf "\r${CYAN}[%3d%%]${NC} " "$percentage"
    printf "${GREEN}%*s${NC}" "$completed" | tr ' ' '█'
    printf "${YELLOW}%*s${NC}" "$remaining" | tr ' ' '░'
    printf " %s" "$step_name"

    if [ "$current" -eq "$total" ]; then
        echo ""
    fi
}

# Header
print_header() {
    echo ""
    log "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
    log "${PURPLE}║           3D Medical Segmentation Setup Script              ║${NC}"
    log "${PURPLE}║              Automated Environment Setup                     ║${NC}"
    log "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Environment detection
detect_environment() {
    log "${BLUE}🔍 Detecting environment...${NC}"

    # Check if running in Google Colab
    if [ -n "$COLAB_GPU" ] || [ -d "/content" ]; then
        ENV_TYPE="colab"
        log "${GREEN}✓ Google Colab environment detected${NC}"
    # Check if running in WSL
    elif grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
        ENV_TYPE="wsl"
        log "${GREEN}✓ Windows WSL environment detected${NC}"
    # Check OS type
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ENV_TYPE="linux"
        log "${GREEN}✓ Linux environment detected${NC}"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ENV_TYPE="macos"
        log "${GREEN}✓ macOS environment detected${NC}"
    else
        ENV_TYPE="unknown"
        log "${YELLOW}⚠ Unknown environment, proceeding with Linux defaults${NC}"
        ENV_TYPE="linux"
    fi

    # Log system information
    log "${CYAN}System Information:${NC}"
    log "  OS: $(uname -s)"
    log "  Architecture: $(uname -m)"
    log "  Kernel: $(uname -r)"

    if command -v lscpu &> /dev/null; then
        CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
        log "  CPU Cores: $CPU_CORES"
    fi

    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -h | awk 'NR==2{printf "%.1f GB", $2}' 2>/dev/null || echo "unknown")
        log "  Memory: $MEMORY_GB"
    fi
}

# Check prerequisites
check_prerequisites() {
    local step=1
    local total_steps=5

    log "${BLUE}🔧 Checking prerequisites...${NC}"

    # Check Python
    progress_bar $step $total_steps "Checking Python installation"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        log "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        log "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    ((step++))

    # Check pip
    progress_bar $step $total_steps "Checking pip installation"
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version 2>&1 | cut -d' ' -f2)
        log "${GREEN}✓ pip $PIP_VERSION found${NC}"
    else
        log "${RED}✗ pip not found. Please install pip${NC}"
        exit 1
    fi
    ((step++))

    # Check git
    progress_bar $step $total_steps "Checking Git installation"
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version 2>&1 | cut -d' ' -f3)
        log "${GREEN}✓ Git $GIT_VERSION found${NC}"
    else
        log "${RED}✗ Git not found. Please install Git${NC}"
        exit 1
    fi
    ((step++))

    # Check CUDA (optional)
    progress_bar $step $total_steps "Checking CUDA availability"
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "unknown")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
        log "${GREEN}✓ CUDA $CUDA_VERSION detected${NC}"
        log "  GPU: $GPU_NAME"
        HAS_GPU=true
    else
        log "${YELLOW}⚠ CUDA not found - will use CPU training${NC}"
        HAS_GPU=false
    fi
    ((step++))

    # Check disk space
    progress_bar $step $total_steps "Checking disk space"
    if command -v df &> /dev/null; then
        DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}' 2>/dev/null || echo "unknown")
        log "${GREEN}✓ Available disk space: $DISK_SPACE${NC}"
    fi
    ((step++))

    echo ""
}

# Setup virtual environment
setup_virtual_environment() {
    local step=1
    local total_steps=4

    log "${BLUE}🏗️ Setting up virtual environment...${NC}"

    # Create virtual environment
    progress_bar $step $total_steps "Creating virtual environment"
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        log "${GREEN}✓ Virtual environment created${NC}"
    else
        log "${YELLOW}⚠ Virtual environment already exists${NC}"
    fi
    ((step++))

    # Activate virtual environment
    progress_bar $step $total_steps "Activating virtual environment"
    source .venv/bin/activate
    log "${GREEN}✓ Virtual environment activated${NC}"
    ((step++))

    # Upgrade pip
    progress_bar $step $total_steps "Upgrading pip"
    python -m pip install --upgrade pip --quiet
    log "${GREEN}✓ pip upgraded to latest version${NC}"
    ((step++))

    # Install wheel and setuptools
    progress_bar $step $total_steps "Installing build tools"
    pip install wheel setuptools --quiet
    log "${GREEN}✓ Build tools installed${NC}"
    ((step++))

    echo ""
}

# Install dependencies
install_dependencies() {
    local step=1
    local total_steps=6

    log "${BLUE}📦 Installing dependencies...${NC}"

    # Install core dependencies
    progress_bar $step $total_steps "Installing PyTorch"
    if [ "$HAS_GPU" = true ]; then
        pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 --quiet
        log "${GREEN}✓ PyTorch with CUDA support installed${NC}"
    else
        pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu --quiet
        log "${GREEN}✓ PyTorch (CPU-only) installed${NC}"
    fi
    ((step++))

    # Install MONAI
    progress_bar $step $total_steps "Installing MONAI"
    pip install "monai[all]==1.3.0" --quiet
    log "${GREEN}✓ MONAI medical imaging framework installed${NC}"
    ((step++))

    # Install scientific computing libraries
    progress_bar $step $total_steps "Installing scientific libraries"
    pip install "numpy==1.26.4" "scipy>=1.12,<1.14" --quiet
    log "${GREEN}✓ NumPy and SciPy installed${NC}"
    ((step++))

    # Install machine learning libraries
    progress_bar $step $total_steps "Installing ML libraries"
    pip install "scikit-learn>=1.3.0" "pandas>=2.0.0" --quiet
    log "${GREEN}✓ Scikit-learn and Pandas installed${NC}"
    ((step++))

    # Install visualization libraries
    progress_bar $step $total_steps "Installing visualization libraries"
    pip install "matplotlib>=3.7.0" "seaborn>=0.12.0" --quiet
    log "${GREEN}✓ Matplotlib and Seaborn installed${NC}"
    ((step++))

    # Install remaining requirements
    progress_bar $step $total_steps "Installing remaining requirements"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
        log "${GREEN}✓ All requirements from requirements.txt installed${NC}"
    else
        # Install individual packages if requirements.txt doesn't exist
        pip install nibabel SimpleITK PyYAML tqdm tensorboard jupyter --quiet
        log "${GREEN}✓ Medical imaging and utility libraries installed${NC}"
    fi
    ((step++))

    echo ""
}

# Verify installation
verify_installation() {
    local step=1
    local total_steps=5

    log "${BLUE}🔬 Verifying installation...${NC}"

    # Test Python imports
    progress_bar $step $total_steps "Testing core imports"
    python3 -c "
import sys
print(f'Python version: {sys.version.split()[0]}')
import torch
print(f'PyTorch version: {torch.__version__}')
import monai
print(f'MONAI version: {monai.__version__}')
import numpy as np
print(f'NumPy version: {np.__version__}')
" >> "$SETUP_LOG" 2>&1

    if [ $? -eq 0 ]; then
        log "${GREEN}✓ Core libraries import successfully${NC}"
    else
        log "${RED}✗ Failed to import core libraries${NC}"
        exit 1
    fi
    ((step++))

    # Test CUDA if available
    progress_bar $step $total_steps "Testing CUDA availability"
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
        log "${GREEN}✓ CUDA available with $GPU_COUNT GPU(s)${NC}"
        log "  Primary GPU: $GPU_NAME"
    else
        log "${YELLOW}⚠ CUDA not available - CPU training only${NC}"
    fi
    ((step++))

    # Test model creation
    progress_bar $step $total_steps "Testing model creation"
    python3 -c "
import sys
import os
sys.path.append('src')
from src.models.factory import create_model
model = create_model(architecture='unet', in_channels=1, out_channels=2)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
" >> "$SETUP_LOG" 2>&1

    if [ $? -eq 0 ]; then
        log "${GREEN}✓ Model creation test passed${NC}"
    else
        log "${YELLOW}⚠ Model creation test failed - check if project structure is correct${NC}"
    fi
    ((step++))

    # Check dataset directories
    progress_bar $step $total_steps "Checking dataset directories"
    DATASETS_DIR="$HOME/Downloads/datasets"
    if [ "$ENV_TYPE" = "colab" ]; then
        DATASETS_DIR="/content/drive/MyDrive/datasets"
    fi

    if [ -d "$DATASETS_DIR" ]; then
        DATASET_COUNT=$(find "$DATASETS_DIR" -maxdepth 1 -type d | wc -l)
        log "${GREEN}✓ Datasets directory found with $((DATASET_COUNT-1)) subdirectories${NC}"
        log "  Path: $DATASETS_DIR"
    else
        log "${YELLOW}⚠ Datasets directory not found${NC}"
        log "  Expected path: $DATASETS_DIR"
        log "  You'll need to download and organize datasets manually"
    fi
    ((step++))

    # Test training script
    progress_bar $step $total_steps "Testing training script"
    if [ -f "scripts/train_model.py" ]; then
        python3 scripts/train_model.py --help >> "$SETUP_LOG" 2>&1
        if [ $? -eq 0 ]; then
            log "${GREEN}✓ Training script is functional${NC}"
        else
            log "${YELLOW}⚠ Training script test failed${NC}"
        fi
    else
        log "${YELLOW}⚠ Training script not found${NC}"
    fi
    ((step++))

    echo ""
}

# Setup dataset directories
setup_datasets() {
    log "${BLUE}📁 Setting up dataset directories...${NC}"

    DATASETS_DIR="$HOME/Downloads/datasets"
    if [ "$ENV_TYPE" = "colab" ]; then
        DATASETS_DIR="/content/drive/MyDrive/datasets"
    fi

    # Create dataset directories
    mkdir -p "$DATASETS_DIR"/{BraTS,MSD,TotalSegmentator}

    log "${GREEN}✓ Dataset directories created:${NC}"
    log "  Main: $DATASETS_DIR"
    log "  - BraTS: $DATASETS_DIR/BraTS"
    log "  - MSD: $DATASETS_DIR/MSD"
    log "  - TotalSegmentator: $DATASETS_DIR/TotalSegmentator"

    # Create README with download instructions
    cat > "$DATASETS_DIR/README.md" << 'EOF'
# Dataset Setup Instructions

## Required Datasets

### 1. BraTS 2021
- Download from: https://www.synapse.org/Synapse:syn25829067
- Extract to: `BraTS/`
- Expected structure: `BraTS/training_data1_v2/BraTS-GLI-*/`

### 2. MSD Liver (Task 3)
- Download from: http://medicaldecathlon.com/
- Extract to: `MSD/`
- Expected structure: `MSD/Task03_Liver/`

### 3. TotalSegmentator
- Download from: https://github.com/wasserth/TotalSegmentator
- Extract to: `TotalSegmentator/`
- Expected structure: `TotalSegmentator/s*/`

## Verification
Run the setup script with --verify to check dataset integrity.
EOF

    log "${CYAN}📋 Dataset download instructions created in $DATASETS_DIR/README.md${NC}"
    echo ""
}

# Print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local         Setup for local development (default)"
    echo "  --colab         Setup for Google Colab"
    echo "  --verify        Only verify existing installation"
    echo "  --datasets      Setup dataset directories only"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full local setup"
    echo "  $0 --colab           # Colab setup"
    echo "  $0 --verify          # Verify installation"
    echo "  $0 --datasets        # Setup dataset dirs"
    echo ""
}

# Main setup function
main() {
    print_header

    # Parse arguments
    SETUP_TYPE="local"
    VERIFY_ONLY=false
    DATASETS_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --local)
                SETUP_TYPE="local"
                shift
                ;;
            --colab)
                SETUP_TYPE="colab"
                shift
                ;;
            --verify)
                VERIFY_ONLY=true
                shift
                ;;
            --datasets)
                DATASETS_ONLY=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done

    # Start timing
    START_TIME=$(date +%s)

    # Log setup start
    log "${CYAN}🚀 Starting setup process...${NC}"
    log "  Setup type: $SETUP_TYPE"
    log "  Log file: $SETUP_LOG"
    log "  Start time: $(date)"
    echo ""

    # Run setup steps
    detect_environment

    if [ "$DATASETS_ONLY" = true ]; then
        setup_datasets
    elif [ "$VERIFY_ONLY" = true ]; then
        verify_installation
    else
        check_prerequisites
        setup_virtual_environment
        install_dependencies
        verify_installation
        setup_datasets
    fi

    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    # Print success message
    echo ""
    log "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${GREEN}║                   SETUP COMPLETED!                        ║${NC}"
    log "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log "${CYAN}📊 Setup Summary:${NC}"
    log "  Duration: ${MINUTES}m ${SECONDS}s"
    log "  Log file: $SETUP_LOG"
    log "  Environment: $ENV_TYPE"
    log "  GPU available: $HAS_GPU"
    echo ""

    # Next steps
    log "${PURPLE}🎯 Next Steps:${NC}"
    if [ "$VERIFY_ONLY" = false ] && [ "$DATASETS_ONLY" = false ]; then
        log "1. Activate virtual environment: ${YELLOW}source .venv/bin/activate${NC}"
    fi
    if [ "$DATASETS_ONLY" = false ]; then
        log "2. Download datasets (see datasets/README.md)"
        log "3. Start training: ${YELLOW}python scripts/train_model.py --help${NC}"
        log "4. Or use Jupyter notebooks in notebooks/"
    fi
    echo ""

    log "${GREEN}✨ Ready for 3D medical image segmentation research!${NC}"
}

# Run main function
main "$@"