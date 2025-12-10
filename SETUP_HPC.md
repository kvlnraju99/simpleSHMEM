# HPC Setup Guide for NVSHMEM Pipeline

Complete setup instructions for running NVSHMEM pipeline on HPC cluster with 2 A100 GPUs.

## Step 1: Check for Existing NVSHMEM Installation

```bash
# Check if NVSHMEM module exists
module avail nvshmem

# If found, load it:
module load nvshmem/2.x.x  # Replace with actual version

# Check environment variable
echo $NVSHMEM_HOME

# If no module, search for installation
find /opt -name "nvshmem.h" 2>/dev/null | head -1
find /usr/local -name "nvshmem.h" 2>/dev/null | head -1
```

## Step 2: Install NVSHMEM (If Not Found)

### Option A: Download Pre-built Binary (Recommended)

```bash
# Create installation directory
mkdir -p ~/software
cd ~/software

# Download NVSHMEM from NVIDIA (requires NVIDIA Developer account)
# Visit: https://developer.nvidia.com/nvshmem
# Download: nvshmem_2.x.x_<cuda-version>_<arch>.txz

# Extract (replace with actual filename)
tar -xf nvshmem_2.11.0-6_cuda12_x86_64.txz

# Set environment variable
export NVSHMEM_HOME=~/software/nvshmem_2.11.0-6
echo "export NVSHMEM_HOME=~/software/nvshmem_2.11.0-6" >> ~/.bashrc
```

### Option B: Build from Source (If Binary Not Available)

```bash
cd ~/software

# Download source
wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.11.0/source/nvshmem_src_2.11.0-6.txz
tar -xf nvshmem_src_2.11.0-6.txz
cd nvshmem_src_2.11.0-6

# Load required modules
module load cuda/12.1
module load openmpi/4.1.x  # Or your MPI

# Build
make -j8 CUDA_HOME=$CUDA_HOME MPI_HOME=$MPI_HOME

# Install to home directory
make install PREFIX=~/software/nvshmem

# Set environment
export NVSHMEM_HOME=~/software/nvshmem
echo "export NVSHMEM_HOME=~/software/nvshmem" >> ~/.bashrc
```

## Step 3: Load Required Modules

```bash
# Load CUDA
module load cuda/12.1  # Or latest available

# Load MPI (required for NVSHMEM)
module load openmpi/4.1.x  # Or your cluster's MPI

# Verify
which nvcc
which mpirun
echo $NVSHMEM_HOME
```

## Step 4: Update Makefile

```bash
cd ~/simpleSHMEM

# Edit Makefile
nano Makefile  # or vim/vi

# Update these lines with actual paths:
# NVSHMEM_HOME = ~/software/nvshmem  # Your path from Step 2
# CUDA_HOME = /usr/local/cuda        # From 'which nvcc'
# CUDA_ARCH = sm_80                  # sm_80 for A100
```

Or use this one-liner to auto-update:

```bash
# Automatic Makefile update
cat > Makefile << 'EOF'
NVSHMEM_HOME ?= $(HOME)/software/nvshmem
CUDA_HOME ?= /usr/local/cuda

NVCC = $(CUDA_HOME)/bin/nvcc
NVSHMEM_INCLUDE = -I$(NVSHMEM_HOME)/include
NVSHMEM_LIB = -L$(NVSHMEM_HOME)/lib -lnvshmem

CUDA_ARCH ?= sm_80
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

all: nvshmem_pipeline

nvshmem_pipeline: nvshmem_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) $(NVSHMEM_INCLUDE) $< -o $@ $(NVSHMEM_LIB) -lcurand

clean:
	rm -f nvshmem_pipeline

.PHONY: all clean
EOF
```

## Step 5: Compile

```bash
cd ~/simpleSHMEM

# Clean previous builds
make clean

# Compile
make

# Verify binary created
ls -lh nvshmem_pipeline
```

## Step 6: Run on HPC

### Option A: Interactive Session

```bash
# Request interactive node with 2 GPUs
srun --gres=gpu:2 --time=00:30:00 --pty bash

# Inside the session:
module load cuda/12.1 openmpi
cd ~/simpleSHMEM

# Run
mpirun -np 2 ./nvshmem_pipeline
```

### Option B: SLURM Batch Job

Create `run_nvshmem.sh`:

```bash
cat > run_nvshmem.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=nvshmem_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --output=nvshmem_%j.out

# Load modules
module load cuda/12.1
module load openmpi

# Set NVSHMEM path if not in environment
export NVSHMEM_HOME=~/software/nvshmem

# Run
cd ~/simpleSHMEM
mpirun -np 2 ./nvshmem_pipeline
EOF

# Submit job
sbatch run_nvshmem.sh

# Check status
squeue -u $USER

# View output when done
cat nvshmem_*.out
```

## Step 7: Run NCCL Version for Comparison

```bash
# Python NCCL version (easier to test first)
python NCLL.py > nccl_output.txt

# Compare final predictions
grep "Final iteration predictions" nccl_output.txt
grep "Final iteration predictions" nvshmem_*.out
```

## Troubleshooting

### Error: "nvshmem.h not found"

```bash
# Check NVSHMEM_HOME is set
echo $NVSHMEM_HOME
ls $NVSHMEM_HOME/include/nvshmem.h

# Update Makefile with correct path
```

### Error: "libcuda.so not found"

```bash
# Load CUDA module
module load cuda/12.1

# Check library path
echo $LD_LIBRARY_PATH
```

### Error: "MPI not found"

```bash
# Load MPI module
module load openmpi

# Check mpirun
which mpirun
```

### Compilation warnings about sm_52

```bash
# Update CUDA_ARCH in Makefile to sm_80 (for A100)
```

## Quick Start (Copy-Paste All)

```bash
# Complete setup in one go (adjust paths as needed)
module load cuda/12.1 openmpi
export NVSHMEM_HOME=~/software/nvshmem  # Set your path
cd ~/simpleSHMEM
make clean && make
srun --gres=gpu:2 mpirun -np 2 ./nvshmem_pipeline
```
