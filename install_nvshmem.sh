#!/bin/bash
# Quick NVSHMEM Installation Script for HPC

echo "=== NVSHMEM Installation ==="

# Step 1: Download NVSHMEM
echo "Step 1: Downloading NVSHMEM..."
cd ~/software

# Download latest NVSHMEM (CUDA 12.x, x86_64)
wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.11.0/txz/nvshmem_2.11.0-6_cuda12_x86_64.txz

# Step 2: Extract
echo "Step 2: Extracting..."
tar -xf nvshmem_2.11.0-6_cuda12_x86_64.txz

# Step 3: Set environment
echo "Step 3: Setting environment..."
export NVSHMEM_HOME=~/software/nvshmem_2.11.0-6

# Add to bashrc for persistence
echo "export NVSHMEM_HOME=~/software/nvshmem_2.11.0-6" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo ""
echo "✓ NVSHMEM installed to: $NVSHMEM_HOME"
echo ""
echo "Verify installation:"
ls -la $NVSHMEM_HOME/include/nvshmem.h
ls -la $NVSHMEM_HOME/lib/libnvshmem.so

echo ""
echo "Next steps:"
echo "1. cd ~/simpleSHMEM"
echo "2. Update Makefile with NVSHMEM_HOME=$NVSHMEM_HOME"
echo "3. make clean && make"
echo "4. Run: srun --gres=gpu:2 mpirun -np 2 ./nvshmem_pipeline"
