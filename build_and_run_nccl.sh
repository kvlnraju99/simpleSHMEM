#!/bin/bash

# =============================================================================
# Build Script for NCCL C++ Pipeline
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "NCCL C++ Pipeline - Build and Run Script"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# 1. Check Prerequisites
# =============================================================================

echo "Step 1: Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ nvcc not found. Please install CUDA Toolkit.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CUDA found:${NC} $(nvcc --version | grep release)"

# Check MPI
if ! command -v mpirun &> /dev/null; then
    echo -e "${RED}✗ mpirun not found. Please install MPI (OpenMPI or MPICH).${NC}"
    exit 1
fi
echo -e "${GREEN}✓ MPI found:${NC} $(mpirun --version | head -n1)"

# Check GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo -e "${RED}✗ Need 2 GPUs, but only found $GPU_COUNT${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $GPU_COUNT GPUs${NC}"

# Check NCCL
if [ ! -f "/usr/local/cuda/lib64/libnccl.so" ] && [ ! -f "/usr/local/cuda/lib64/libnccl.dylib" ]; then
    echo -e "${YELLOW}⚠ NCCL not found in standard location. Build may fail.${NC}"
else
    echo -e "${GREEN}✓ NCCL library found${NC}"
fi

echo ""

# =============================================================================
# 2. Detect CUDA Architecture
# =============================================================================

echo "Step 2: Detecting GPU architecture..."

# Get GPU architecture from first GPU
GPU_NAME=$(nvidia-smi -L | head -n1 | cut -d':' -f2 | cut -d'(' -f1 | xargs)
echo "  Detected GPU: $GPU_NAME"

# Map common GPUs to CUDA architectures
if [[ "$GPU_NAME" == *"A100"* ]]; then
    CUDA_ARCH="sm_80"
elif [[ "$GPU_NAME" == *"V100"* ]]; then
    CUDA_ARCH="sm_70"
elif [[ "$GPU_NAME" == *"RTX 3090"* ]] || [[ "$GPU_NAME" == *"RTX 3080"* ]]; then
    CUDA_ARCH="sm_86"
elif [[ "$GPU_NAME" == *"RTX 4090"* ]] || [[ "$GPU_NAME" == *"RTX 4080"* ]]; then
    CUDA_ARCH="sm_89"
elif [[ "$GPU_NAME" == *"H100"* ]]; then
    CUDA_ARCH="sm_90"
else
    CUDA_ARCH="sm_80"  # Default to A100
    echo -e "${YELLOW}  ⚠ Unknown GPU, defaulting to sm_80${NC}"
fi

echo -e "${GREEN}  Using CUDA architecture: $CUDA_ARCH${NC}"
echo ""

# =============================================================================
# 3. Set Environment Variables
# =============================================================================

echo "Step 3: Setting environment variables..."

# CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# MPI paths (try to auto-detect)
if [ -z "$MPI_HOME" ]; then
    MPI_BIN=$(which mpirun)
    if [ -n "$MPI_BIN" ]; then
        export MPI_HOME=$(dirname $(dirname $MPI_BIN))
        echo "  Auto-detected MPI_HOME: $MPI_HOME"
    else
        export MPI_HOME=/usr/local
    fi
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""

# =============================================================================
# 4. Build
# =============================================================================

echo "Step 4: Building nccl_pipeline..."

# Clean previous build
if [ -f "nccl_pipeline" ]; then
    rm nccl_pipeline
    echo "  Removed previous build"
fi

# Build using Makefile
if [ -f "Makefile.nccl" ]; then
    echo "  Using Makefile.nccl..."
    make -f Makefile.nccl CUDA_ARCH=$CUDA_ARCH
else
    # Manual build if Makefile doesn't exist
    echo "  Building manually..."
    nvcc -arch=$CUDA_ARCH -O3 -std=c++14 \
         -I$CUDA_HOME/include \
         -I$MPI_HOME/include \
         -L$CUDA_HOME/lib64 \
         -L$MPI_HOME/lib \
         nccl_pipeline.cpp \
         -o nccl_pipeline \
         -lnccl -lcudart -lcublas -lcurand -lmpi
fi

if [ ! -f "nccl_pipeline" ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# =============================================================================
# 5. Run
# =============================================================================

echo "Step 5: Running NCCL pipeline with 2 GPUs..."
echo "============================================================"
echo ""

# Run with 2 processes
mpirun -np 2 ./nccl_pipeline

echo ""
echo "============================================================"
echo -e "${GREEN}✓ Script completed successfully${NC}"
echo "============================================================"
