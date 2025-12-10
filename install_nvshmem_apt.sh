#!/bin/bash
# NVSHMEM Installation via APT Package Manager
# This method is more reliable on HPC clusters

echo "=== NVSHMEM Installation (APT Method) ==="

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_VERSION=$VERSION_ID
    echo "Detected OS: $NAME $VERSION"
else
    echo "Cannot detect OS version. Assuming Ubuntu 22.04"
    OS_VERSION="22.04"
fi

# Convert version to repo format (e.g., 22.04 -> ubuntu2204)
UBUNTU_VERSION=$(echo $OS_VERSION | sed 's/\.//g')
REPO_VERSION="ubuntu${UBUNTU_VERSION}"

echo "Using repository version: $REPO_VERSION"

# Step 1: Add NVIDIA CUDA repository (if not already added)
echo "Step 1: Adding NVIDIA CUDA repository..."

if [ ! -f /etc/apt/sources.list.d/cuda-$REPO_VERSION-x86_64.list ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/$REPO_VERSION/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
else
    echo "CUDA repository already configured"
fi

# Step 2: Install NVSHMEM packages
echo "Step 2: Installing NVSHMEM via APT..."

# Install dev and runtime packages for CUDA 12
sudo apt-get install -y \
    libnvshmem3-cuda-12 \
    libnvshmem3-dev-cuda-12

# Step 3: Find installation location
echo "Step 3: Locating NVSHMEM installation..."

NVSHMEM_HOME=$(dpkg -L libnvshmem3-dev-cuda-12 | grep include/nvshmem.h | sed 's|/include/nvshmem.h||')

if [ -z "$NVSHMEM_HOME" ]; then
    echo "ERROR: Could not find NVSHMEM installation path"
    exit 1
fi

echo "✓ NVSHMEM installed to: $NVSHMEM_HOME"

# Step 4: Set environment variables
echo "Step 4: Setting environment variables..."

# Add to bashrc for persistence (only if not already added)
if ! grep -q "NVSHMEM_HOME" ~/.bashrc; then
    echo "export NVSHMEM_HOME=$NVSHMEM_HOME" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added NVSHMEM environment variables to ~/.bashrc"
fi

# Verify installation
echo ""
echo "Verify installation:"
ls -la $NVSHMEM_HOME/include/nvshmem.h
ls -la $NVSHMEM_HOME/lib/libnvshmem_host.so

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "1. source ~/.bashrc"
echo "2. Update Makefile with NVSHMEM_HOME=$NVSHMEM_HOME"
echo "3. cd ~/simpleSHMEM && make clean && make"
echo "4. Run: srun --gres=gpu:2 mpirun -np 2 ./nvshmem_pipeline"
