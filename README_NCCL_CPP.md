# NCCL Pipeline Parallelism - C++ Implementation

C++ implementation of the NCCL pipeline parallelism demo, equivalent to the Python version but using native NCCL C++ APIs.

## Requirements

- **Hardware**: 2 NVIDIA GPUs (ideal: A100s with NVLink)
- **CUDA**: 11.8+ or 12.1+
- **NCCL**: 2.10+ (bundled with CUDA Toolkit)
- **MPI**: OpenMPI 4.0+ or MPICH 3.3+
- **Compiler**: NVCC (from CUDA Toolkit)

## Installation

### 1. Install CUDA Toolkit

```bash
# CUDA includes NCCL, cuBLAS, and cuRAND
# Download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Install MPI

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
```

**macOS:**

```bash
brew install open-mpi
```

**From source (if needed):**

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar -xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
```

### 3. Verify Installations

```bash
# Check CUDA
nvcc --version

# Check NCCL (should be in CUDA installation)
ls /usr/local/cuda/lib64/libnccl*

# Check MPI
mpirun --version
```

## Building

### Option 1: Using Makefile

```bash
# Build
make -f Makefile.nccl

# Or build and run
make -f Makefile.nccl run
```

### Option 2: Manual Compilation

```bash
# Adjust paths as needed for your system
nvcc -arch=sm_80 -O3 -std=c++14 \
     -I/usr/local/cuda/include \
     -I/usr/local/include \
     -L/usr/local/cuda/lib64 \
     -L/usr/local/lib \
     nccl_pipeline.cpp \
     -o nccl_pipeline \
     -lnccl -lcudart -lcublas -lcurand -lmpi
```

**Architecture flags** (change based on your GPU):

- `sm_70` - V100
- `sm_80` - A100
- `sm_86` - RTX 3090
- `sm_89` - RTX 4090

## Running

```bash
# Run with 2 MPI processes (1 per GPU)
mpirun -np 2 ./nccl_pipeline

# Run with specific GPU visibility (optional)
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./nccl_pipeline

# Run with verbose MPI output (for debugging)
mpirun -np 2 --display-map ./nccl_pipeline
```

## Expected Output

```
✓ Found 2 GPUs. Starting NCCL pipeline on 2 GPUs...
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
============================================================

[Rank 0] NCCL process group initialized on GPU 0
[Rank 1] NCCL process group initialized on GPU 1
  Initialized Stage 0 (Linear 2048->1024)
  Initialized Stage 1 (Linear 1024->10)
[Rank 0] Iter 0: Sent activations shape [128, 1024], norm=...
[Rank 1] Iter 0: Received activations shape [128, 1024], norm=...
[Rank 1] Iter 0: Final output shape [128, 10], norm=...
...
============================================================
[NCCL Pipeline] Completed 5 iterations
[NCCL Pipeline] Total time: 0.XXXX s
[NCCL Pipeline] Avg time per iteration: 0.XXXX s
============================================================
```

## What It Does

Identical functionality to the Python version:

1. **Model Partitioning**:

   - GPU 0: Linear(2048 → 1024) + ReLU
   - GPU 1: Linear(1024 → 10) + ReLU

2. **Pipeline Execution**:

   - GPU 0: Generate random input → Compute layer 1 → Send to GPU 1
   - GPU 1: Receive from GPU 0 → Compute layer 2 → Output results
   - Repeat for 5 iterations

3. **Reproducibility**:
   - Uses seed 42 for weight initialization
   - Same random data generation per iteration

## Code Structure

```
nccl_pipeline.cpp
├── CUDA Kernels
│   └── relu_kernel()          # ReLU activation
├── Helper Functions
│   ├── CUDA_CHECK()            # CUDA error handling
│   ├── NCCL_CHECK()            # NCCL error handling
│   ├── MPI_CHECK()             # MPI error handling
│   └── compute_norm()          # L2 norm calculation
├── PipelineStage class
│   ├── constructor             # Initialize weights with cuRAND
│   └── forward()               # Linear + ReLU using cuBLAS
└── run_nccl_pipeline()        # Main pipeline logic
    └── main()                  # MPI initialization
```

## Key Differences from Python

| Aspect           | Python (PyTorch)            | C++ (Native NCCL)     |
| ---------------- | --------------------------- | --------------------- |
| Process spawning | `mp.spawn()`                | MPI                   |
| NCCL init        | `dist.init_process_group()` | `ncclCommInitRank()`  |
| Linear layer     | `nn.Linear()`               | cuBLAS SGEMM          |
| Weight init      | PyTorch defaults            | cuRAND Gaussian       |
| Send/Recv        | `dist.send/recv()`          | `ncclSend/ncclRecv()` |

## Performance Notes

- **NCCL Backend**: Same high-performance NCCL backend as PyTorch
- **NVLink**: Automatically detected and used if available
- **Zero-copy**: Direct GPU-to-GPU transfers
- **Overhead**: C++ version has lower overhead than Python (no Python interpreter, faster startup)

## Troubleshooting

### Error: `libnccl.so: cannot open shared object file`

```bash
# Add NCCL to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: `libmpi.so: cannot open shared object file`

```bash
# Add MPI to library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Error: `NCCL WARN Cuda failure 'invalid device ordinal'`

- Make sure you have 2 GPUs available
- Check: `nvidia-smi` shows 2 GPUs
- Use `CUDA_VISIBLE_DEVICES=0,1`

### Error: `MPI_Init_thread failed`

- Check MPI installation: `mpirun --version`
- Try reinstalling MPI

### Compilation errors: `cublas.h not found`

```bash
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
```

## Next Steps

Once this C++ NCCL implementation works, you can compare it with:

- The Python NCCL version (`NCLL.py`)
- The NVSHMEM version (`nvshmem_pipeline.cu`)

All three should produce similar results with different communication mechanisms.
