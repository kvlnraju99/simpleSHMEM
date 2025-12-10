# NCCL CUDA Pipeline - Build Instructions

## Overview

`nccl_pipeline.cu` is a CUDA C++ implementation that matches the behavior of `NCLL.py`, using NCCL for GPU-to-GPU communication just like the Python version.

## Key Features

✓ **Same architecture**: 2-GPU pipeline (2048→1024→10)  
✓ **Same communication pattern**: Uses `ncclSend()`/`ncclRecv()` like Python's `dist.send()`/`dist.recv()`  
✓ **Full batch transfer**: Sends entire batch (128 samples) at once  
✓ **CUDA kernels**: Custom linear layer, ReLU, weight initialization (matching NVSHMEM version)

## Comparison with Other Implementations

| Implementation        | Language       | Communication         | Memory Model    |
| --------------------- | -------------- | --------------------- | --------------- |
| `NCLL.py`             | Python/PyTorch | NCCL (dist.send/recv) | PyTorch tensors |
| `nccl_pipeline.cu`    | CUDA C++       | NCCL (ncclSend/Recv)  | GPU malloc      |
| `nvshmem_pipeline.cu` | CUDA C++       | NVSHMEM (put/get)     | Symmetric heap  |

## Prerequisites

1. **CUDA Toolkit** (with NCCL included)

   - CUDA 11.0 or later recommended
   - NCCL is bundled with modern CUDA installations

2. **MPI Implementation**

   - OpenMPI or MPICH
   - Used for process management and initial synchronization

3. **2 NVIDIA GPUs**
   - Can be any CUDA-capable GPUs

## Build Instructions

### Option 1: Using the build script

```bash
./build_and_run_nccl.sh
```

### Option 2: Manual build with Makefile

```bash
# Clean previous builds
make -f Makefile.nccl clean

# Build
make -f Makefile.nccl

# Run
mpirun -np 2 ./nccl_pipeline
```

### Option 3: Direct nvcc compilation

```bash
nvcc -arch=sm_80 -O3 -std=c++14 \\
     -Xcompiler -fopenmp \\
     -I/usr/local/cuda/include \\
     -I$MPI_HOME/include \\
     -L/usr/local/cuda/lib64 \\
     -L$MPI_HOME/lib \\
     nccl_pipeline.cu -o nccl_pipeline \\
     -lnccl -lcudart -lcublas -lcurand -lmpi

# Run with MPI
mpirun -np 2 ./nccl_pipeline
```

## Configuration

Edit the following in `nccl_pipeline.cu`:

```c
#define BATCH_SIZE 128      // Batch size
#define D_IN 2048          // Input dimension
#define D_HIDDEN 1024      // Hidden dimension
#define D_OUT 10           // Output dimension
#define NUM_ITERATIONS 5    // Number of iterations
#define CUDA_ARCH sm_80    // GPU architecture (in Makefile)
```

## Expected Output

```
✓ Found 2 GPUs. Starting NCCL pipeline on 2 GPUs...
  GPU 0: NVIDIA A100-SXM4-40GB
  GPU 1: NVIDIA A100-SXM4-40GB
============================================================

[Rank 0] NCCL initialized on GPU 0
[Rank 1] NCCL initialized on GPU 1
[Rank 0] Initialized Stage (Linear 2048->1024)
[Rank 1] Initialized Stage (Linear 1024->10)
[Rank 0] Iter 0: Sent activations shape [128, 1024], norm=...
[Rank 1] Iter 0: Received activations shape [128, 1024], norm=...
[Rank 1] Iter 0: Final output shape [128, 10], norm=...
...
============================================================
[NCCL Pipeline] Completed 5 iterations
[NCCL Pipeline] Total time: 0.0123s
[NCCL Pipeline] Avg time per iteration: 0.0025s
============================================================
```

## How It Works

### Communication Pattern (matching Python version)

**GPU 0 (Sender):**

1. Generate random input data
2. Forward pass through first layer (2048→1024 + ReLU)
3. `ncclSend()` activations to GPU 1
4. Wait for completion with `cudaDeviceSynchronize()`

**GPU 1 (Receiver):**

1. `ncclRecv()` activations from GPU 0
2. Wait for completion with `cudaDeviceSynchronize()`
3. Forward pass through second layer (1024→10 + ReLU)
4. Compute final output

This is **identical** to the Python version's logic, just using C++ NCCL API instead of PyTorch's distributed primitives.

## Troubleshooting

### "nvcc: command not found"

Add CUDA to your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "cannot find -lnccl"

NCCL is included with CUDA 11+. If missing:

```bash
# Ubuntu/Debian
sudo apt-get install libnccl2 libnccl-dev

# Or download from: https://developer.nvidia.com/nccl
```

### "cannot find -lmpi"

Install MPI:

```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS
brew install open-mpi
```

### Architecture mismatch

Update `CUDA_ARCH` in Makefile.nccl:

- sm_70 for V100
- sm_80 for A100
- sm_86 for RTX 30xx
- sm_89 for RTX 40xx

## Comparison with NVSHMEM

**NCCL advantages:**

- More widely available (bundled with CUDA)
- Easier to install and configure
- Better compatibility with standard MPI workflows

**NVSHMEM advantages:**

- True one-sided communication (no receiver involvement)
- Symmetric memory model
- Potentially lower latency for small messages

Both implementations use identical CUDA kernels, so performance differences will purely reflect the communication library overhead.
