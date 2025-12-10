# NCCL Pipeline Parallelism Demo

Simple 2-GPU pipeline parallelism implementation using PyTorch's NCCL backend.

## Requirements

- 2 NVIDIA GPUs (ideal: A100s with NVLink)
- CUDA 11.8+ or 12.1+
- PyTorch with CUDA support

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install PyTorch directly (recommended - match your CUDA version):
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Run NCCL Pipeline

```bash
python NCLL.py
```

### Expected Output

```
✓ Found 2 GPUs. Starting NCCL pipeline on 2 GPUs...
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
============================================================

[Rank 0] NCCL process group initialized on GPU 0
[Rank 1] NCCL process group initialized on GPU 1
 Initialized Stage 0 (Linear 2048->1024)
 Initialized Stage 1 (Linear 1024->10)
[Rank 0] Iter 0: Sent activations shape torch.Size([128, 1024]), norm=...
[Rank 1] Iter 0: Received activations shape torch.Size([128, 1024]), norm=...
...
============================================================
[NCCL Pipeline] Completed 5 iterations
[NCCL Pipeline] Total time: 0.XXXX s
[NCCL Pipeline] Avg time per iteration: 0.XXXX s
============================================================
```

## What It Does

1. **Splits a 2-layer neural network across 2 GPUs:**

   - GPU 0: Linear(2048 → 1024) + ReLU
   - GPU 1: Linear(1024 → 10) + ReLU

2. **Pipeline execution:**

   - GPU 0 generates random input data
   - GPU 0 computes first layer
   - GPU 0 sends activations to GPU 1 via NCCL
   - GPU 1 receives activations
   - GPU 1 computes second layer
   - Repeat for 5 iterations

3. **Reproducible results:**
   - Uses fixed random seed (42)
   - Same weights and data every run
   - Perfect for comparing with NVSHMEM implementation

## Code Structure

```
NCLL.py
├── PipelineStage          # Neural network layer (Linear + ReLU)
├── setup_process_group()  # Initialize NCCL
├── cleanup()              # Cleanup NCCL
└── run_nccl_pipeline()    # Main pipeline logic
```

## Notes

- NCCL is bundled with PyTorch CUDA installation (no separate install needed)
- Script automatically detects and uses NVLink if available between GPUs
- Communication uses NCCL's point-to-point send/recv operations
- Currently inference-only (no backward pass)

## Troubleshooting

**Error: "CUDA is not available"**

- Install PyTorch with CUDA support (see Installation section)

**Error: "This script requires 2 GPUs"**

- You need at least 2 NVIDIA GPUs to run this script

**Error: "Address already in use"**

- Change port in `setup_process_group()`: `os.environ['MASTER_PORT'] = '12356'`
