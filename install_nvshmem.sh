#!/bin/bash
# Quick NVSHMEM Installation Script for HPC

echo "=== NVSHMEM Installation ==="

# Step 1: Download NVSHMEM
echo "Step 1: Downloading NVSHMEM..."
mkdir -p ~/software
cd ~/software

# Download NVSHMEM 3.4.5 (latest stable, CUDA 12.x, x86_64)
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Step 2: Extract
echo "Step 2: Extracting..."
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Step 3: Set environment
echo "Step 3: Setting environment..."
export NVSHMEM_HOME=~/software/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

# Add to bashrc for persistence (only if not already added)
if ! grep -q "NVSHMEM_HOME" ~/.bashrc; then
  echo "export NVSHMEM_HOME=~/software/libnvshmem-linux-x86_64-3.4.5_cuda12-archive" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
  echo "Added NVSHMEM environment variables to ~/.bashrc"
fi

echo ""
echo "✓ NVSHMEM installed to: $NVSHMEM_HOME"
echo ""
echo "Verify installation:"
ls -la $NVSHMEM_HOME/include/nvshmem.h
ls -la $NVSHMEM_HOME/lib/libnvshmem_host.so

echo ""
echo "Next steps:"
echo "1. source ~/.bashrc  # or re-login to load environment variables"
echo "2. cd ~/simpleSHMEM"
echo "3. Update Makefile with NVSHMEM_HOME=$NVSHMEM_HOME"
echo "4. make clean && make"
echo "5. Run: srun --gres=gpu:2 mpirun -np 2 ./nvshmem_pipeline"
