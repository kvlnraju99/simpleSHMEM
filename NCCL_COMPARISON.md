# NCCL Implementation Comparison: Python vs C++

## Overview

This document compares the Python (PyTorch) and C++ (native NCCL) implementations of the same pipeline parallelism demo.

## Files

| Implementation | Files                                                                               |
| -------------- | ----------------------------------------------------------------------------------- |
| **Python**     | `NCLL.py`                                                                           |
| **C++**        | `nccl_pipeline.cpp`, `Makefile.nccl`, `build_and_run_nccl.sh`, `README_NCCL_CPP.md` |

## Quick Start

### Python Version

```bash
python NCLL.py
```

### C++ Version

```bash
# Option 1: Automated script
./build_and_run_nccl.sh

# Option 2: Manual build
make -f Makefile.nccl
mpirun -np 2 ./nccl_pipeline
```

## Side-by-Side Comparison

### 1. Process Spawning

| Python                          | C++                       |
| ------------------------------- | ------------------------- |
| `torch.multiprocessing.spawn()` | `mpirun -np 2`            |
| Automatic, built into PyTorch   | Requires MPI installation |

### 2. NCCL Initialization

**Python:**

```python
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

**C++:**

```cpp
ncclUniqueId nccl_id;
ncclGetUniqueId(&nccl_id);  // Rank 0
MPI_Bcast(&nccl_id, ...);   // Broadcast to all
ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);
```

### 3. Neural Network Layer

**Python:**

```python
class PipelineStage(nn.Module):
    def __init__(self, in_features, out_features, stage_id):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))
```

**C++:**

```cpp
class PipelineStage {
    float* d_weights;
    cublasHandle_t cublas_handle;

    void forward(float* d_input, float* d_output, int batch_size) {
        // Matrix multiplication using cuBLAS
        cublasSgemm(...);

        // ReLU activation using custom CUDA kernel
        relu_kernel<<<blocks, threads>>>(d_output, size);
    }
};
```

### 4. Weight Initialization

**Python:**

```python
# Automatic with nn.Linear
self.fc = nn.Linear(in_features, out_features)
```

**C++:**

```cpp
// Manual with cuRAND
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, seed);
curandGenerateNormal(gen, d_weights, size, 0.0f, 1.0f/sqrt(in_features));
```

### 5. Communication

**Python:**

```python
# Send
dist.send(tensor=intermediate, dst=1)

# Receive
dist.recv(tensor=recv_buffer, src=0)
```

**C++:**

```cpp
// Send
ncclSend(d_output, size, ncclFloat, 1, nccl_comm, stream);

// Receive
ncclRecv(d_recv_buffer, size, ncclFloat, 0, nccl_comm, stream);
```

### 6. Random Input Generation

**Python:**

```python
inputs = torch.randn(BATCH_SIZE, D_IN, device=rank)
```

**C++:**

```cpp
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandGenerateNormal(gen, d_input, BATCH_SIZE * D_IN, 0.0f, 1.0f);
```

## Functional Equivalence

Both implementations:

- ✅ Split a 2-layer network across 2 GPUs
- ✅ Use NCCL for GPU-to-GPU communication
- ✅ Execute 5 pipeline iterations
- ✅ Use seed 42 for reproducibility
- ✅ Print activations and norms at each step
- ✅ Report total and average iteration time

## Dependencies

| Component                | Python                | C++                       |
| ------------------------ | --------------------- | ------------------------- |
| **NCCL**                 | Bundled with PyTorch  | Bundled with CUDA Toolkit |
| **Process launcher**     | Built-in `mp.spawn()` | Requires MPI              |
| **Linear algebra**       | Built-in PyTorch      | Requires cuBLAS           |
| **Random generation**    | Built-in PyTorch      | Requires cuRAND           |
| **Activation functions** | Built-in PyTorch      | Custom CUDA kernels       |

## Code Complexity

| Metric                     | Python      | C++                                           |
| -------------------------- | ----------- | --------------------------------------------- |
| Lines of code              | 160         | 400+                                          |
| External dependencies      | 1 (PyTorch) | 3 (MPI, CUDA Toolkit with NCCL/cuBLAS/cuRAND) |
| Abstraction level          | High        | Low                                           |
| Explicit memory management | No          | Yes                                           |
| Error handling             | Minimal     | Extensive macros                              |

## Performance Characteristics

### Expected Performance

- **Similar throughput**: Both use the same NCCL backend for communication
- **C++ lower startup overhead**: No Python interpreter
- **C++ lower memory overhead**: No PyTorch framework overhead
- **Similar GPU compute**: Both use same CUDA operations

### When to Use Each

**Use Python version when:**

- Rapid prototyping
- Already using PyTorch ecosystem
- Want simpler code
- Don't need absolute maximum performance

**Use C++ version when:**

- Need maximum performance
- Building production systems
- Want fine-grained control
- Integrating with existing C++ codebase
- Minimizing dependencies

## Running Both for Comparison

```bash
# Run Python version
echo "=== Python Version ==="
python NCLL.py

# Run C++ version
echo ""
echo "=== C++ Version ==="
./build_and_run_nccl.sh
```

Both should produce similar output with comparable timings.

## Next Steps

After verifying both NCCL implementations work correctly, you can:

1. Compare performance between Python and C++
2. Compare with NVSHMEM implementation (`nvshmem_pipeline.cu`)
3. Analyze communication patterns and bottlenecks
4. Extend with more complex models or pipeline stages

## Key Takeaways

1. **Same NCCL backend**: Both implementations use identical underlying NCCL library
2. **Different abstractions**: Python hides complexity, C++ exposes it
3. **Trade-offs**: Simplicity vs. Control vs. Performance
4. **Learning**: C++ version teaches low-level GPU programming concepts
