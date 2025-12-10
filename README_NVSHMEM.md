# NVSHMEM Pipeline Implementation

CUDA C++ implementation of the same pipeline parallelism demo using NVSHMEM instead of NCCL.

## Prerequisites

- NVSHMEM installed (download from NVIDIA Developer)
- CUDA 11.8+ or 12.1+
- 2 NVIDIA GPUs (A100 recommended)
- MPI (OpenMPI or MVAPICH2-GDR)

## Installation

### 1. Install NVSHMEM

```bash
# Download from: https://developer.nvidia.com/nvshmem
# Extract and set environment variable
export NVSHMEM_HOME=/path/to/nvshmem
```

### 2. Update Makefile

Edit `Makefile` and set:

```makefile
NVSHMEM_HOME = /path/to/your/nvshmem
CUDA_HOME = /usr/local/cuda  # or your CUDA path
CUDA_ARCH = sm_80  # sm_80 for A100, sm_86 for RTX 3090
```

### 3. Compile

```bash
make
```

## Running

### With MPI (Recommended)

```bash
mpirun -np 2 ./nvshmem_pipeline
```

### With SHMEM Launcher

```bash
nvshmrun -np 2 ./nvshmem_pipeline
```

## Micro-batching Configuration

The implementation uses **offset-based micro-batching** with a single large symmetric buffer:

```c
#define NUM_MICRO_BATCHES 4       // Split batch into 4 micro-batches
#define MICRO_BATCH_SIZE 32       // 128 / 4 = 32 samples per micro-batch
```

**How it works:**

- Allocates ONE large symmetric buffer for all micro-batches: `NUM_MICRO_BATCHES * MICRO_BATCH_SIZE * D_HIDDEN`
- Each micro-batch accesses its portion via offset: `intermediate + (mb * MICRO_BATCH_SIZE * D_HIDDEN)`
- GPU 0 sends micro-batches using non-blocking `nvshmem_put_nbi()` without immediate `quiet()`
- Enables **overlapped computation and communication** between GPUs

**To disable micro-batching:** Set `NUM_MICRO_BATCHES = 1` in the code.

## Expected Output

```
✓ Found 2 GPUs. Starting NVSHMEM pipeline on 2 GPUs...
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
============================================================

[Rank 0] NVSHMEM initialized on GPU 0
[Rank 1] NVSHMEM initialized on GPU 1
[Rank 0] Initialized Stage (Linear 2048->1024)
[Rank 1] Initialized Stage (Linear 1024->10)
[Rank 0] Iter 0: Computed activations, norm=25.XXXX
[Rank 0] Iter 0: Sent activations via NVSHMEM
[Rank 1] Iter 0: Received activations, norm=25.XXXX
[Rank 1] Iter 0: Final output, norm=8.XXXX
...
[Rank 1] Final iteration predictions (first 3 samples):
  Sample 0: [0.000000, 0.000000, 0.063081, ...]
  Sample 1: [0.000000, 0.266141, 0.158202, ...]
  Sample 2: [0.000000, 0.065695, 0.272367, ...]

============================================================
[NVSHMEM Pipeline] Completed 5 iterations
[NVSHMEM Pipeline] Total time: 0.XXXX s
[NVSHMEM Pipeline] Avg time per iteration: 0.XXXX s
============================================================
```

## Key Differences from NCCL Version

| Aspect             | NCCL (Python)       | NVSHMEM (CUDA)                    |
| ------------------ | ------------------- | --------------------------------- |
| **Language**       | Python + PyTorch    | CUDA C++                          |
| **Communication**  | `dist.send/recv`    | `nvshmem_put`                     |
| **Memory**         | PyTorch tensors     | Symmetric heap (`nvshmem_malloc`) |
| **Forward Pass**   | PyTorch layers      | Custom CUDA kernels               |
| **Initialization** | `torch.manual_seed` | `curand` with seed                |

## NVSHMEM-Specific Features

1. **Symmetric Memory**: Uses `nvshmem_malloc()` for inter-PE communication
2. **One-sided Communication**: `nvshmem_put_nbi()` (non-blocking put)
3. **Direct GPU-to-GPU**: No CPU involvement in data transfer
4. **Global Barriers**: `nvshmem_barrier_all()` for synchronization

## Verification

To compare NCCL vs NVSHMEM outputs:

1. Run NCCL version: `python NCLL.py`
2. Run NVSHMEM version: `mpirun -np 2 ./nvshmem_pipeline`
3. Compare the final predictions - they should be very similar (small numerical differences are OK)

## Troubleshooting

**Error: "NVSHMEM initialization failed"**

- Check `$NVSHMEM_HOME` is set correctly
- Verify MPI is installed and working

**Error: "cudaErrorInvalidDevice"**

- Make sure you have 2 GPUs available
- Check GPU visibility: `nvidia-smi`

**Compilation errors**

- Verify CUDA arch matches your GPU (sm_80 for A100)
- Check NVSHMEM paths in Makefile

## Performance Notes

- NVSHMEM should be slightly faster than NCCL for small messages
- On A100 with NVLink, both should be very fast (< 0.1s for 5 iterations)
- NVSHMEM uses one-sided communication (PUT) vs two-sided (SEND/RECV)
