import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import time

# =============================================================================
# Configuration & Helper Classes
# =============================================================================

class PipelineStage(nn.Module):
    """
    Represents a single stage of the distributed model.
    """
    def __init__(self, in_features, out_features, stage_id):
        super().__init__()
        self.stage_id = stage_id
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))

def setup_process_group(rank, world_size):
    """
    Initializes the NCCL process group.
    
    NCCL requires a Rendezvous mechanism. Here we use a local file/TCP method
    implied by 'localhost'. In a multi-node setup, MASTER_ADDR would be the IP
    of the head node.
    """
    # FIX: Proper environment variable syntax
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group with NCCL backend.
    # NCCL is the only backend in PyTorch that supports P2P on GPU.
    # This establishes peer-to-peer communication channels between GPUs.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Critical: Set the device for this process to avoid cross-device access errors.
    # Each process must only operate on its assigned GPU.
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] NCCL process group initialized on GPU {rank}")

def cleanup():
    """Destroys the process group to release resources."""
    dist.destroy_process_group()

# =============================================================================
# NCCL Pipeline Logic
# =============================================================================

def run_nccl_pipeline(rank, world_size):
    # Set seed for reproducibility - same weights and data every run
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    setup_process_group(rank, world_size)
    
    # -------------------------------------------------------------------------
    # 1. Model Partitioning
    # -------------------------------------------------------------------------
    # Define problem dimensions
    BATCH_SIZE = 128
    D_IN = 2048
    D_HIDDEN = 1024
    D_OUT = 10
    
    # Partition the model:
    # Rank 0: Layer 1 (Input -> Hidden)
    # Rank 1: Layer 2 (Hidden -> Output)
    if rank == 0:
        model = PipelineStage(D_IN, D_HIDDEN, stage_id=0).cuda(rank)
        print(f" Initialized Stage 0 (Linear {D_IN}->{D_HIDDEN})")
    else:
        model = PipelineStage(D_HIDDEN, D_OUT, stage_id=1).cuda(rank)
        print(f" Initialized Stage 1 (Linear {D_HIDDEN}->{D_OUT})")

    # -------------------------------------------------------------------------
    # 2. Pipeline Execution Loop (Micro-batch Simulation)
    # -------------------------------------------------------------------------
    # We simulate 5 iterations to demonstrate the communication pattern.
    NUM_ITERATIONS = 5
    
    # Pre-allocate receive buffers on Rank 1.
    # NCCL requires the receiver to know the exact shape of the incoming tensor.
    if rank == 1:
        recv_buffer = torch.zeros(BATCH_SIZE, D_HIDDEN, device=rank)

    # Simple barrier to ensure both ranks are ready before timing
    dist.barrier()
    start_time = time.time()

    for i in range(NUM_ITERATIONS):
        if rank == 0:
            # --- Rank 0: Compute ---
            # Generate random input (simulating a dataloader)
            inputs = torch.randn(BATCH_SIZE, D_IN, device=rank)
            
            # Forward pass
            with torch.no_grad():
                intermediate = model(inputs)
            
            # --- Rank 0: Communicate (Send) ---
            # NCCL send is blocking on CPU but async on GPU.
            # This ensures the data is enqueued for transmission.
            dist.send(tensor=intermediate, dst=1)
            print(f"[Rank 0] Iter {i}: Sent activations shape {intermediate.shape}, norm={intermediate.norm().item():.4f}")

        elif rank == 1:
            # --- Rank 1: Communicate (Recv) ---
            # NCCL recv blocks CPU until data arrives from Rank 0.
            # This guarantees data is ready before we compute.
            dist.recv(tensor=recv_buffer, src=0)
            
            print(f"[Rank 1] Iter {i}: Received activations shape {recv_buffer.shape}, norm={recv_buffer.norm().item():.4f}")
            
            # --- Rank 1: Compute ---
            with torch.no_grad():
                outputs = model(recv_buffer)
            
            print(f"[Rank 1] Iter {i}: Final output shape {outputs.shape}, norm={outputs.norm().item():.4f}")
            
            # Show actual predictions on last iteration
            if i == NUM_ITERATIONS - 1:
                print(f"[Rank 1] Final iteration predictions (first 3 samples):")
                print(f"  Logits: {outputs[:3].cpu().numpy()}")

    dist.barrier()
    end_time = time.time()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[NCCL Pipeline] Completed {NUM_ITERATIONS} iterations")
        print(f"[NCCL Pipeline] Total time: {end_time - start_time:.4f}s")
        print(f"[NCCL Pipeline] Avg time per iteration: {(end_time - start_time)/NUM_ITERATIONS:.4f}s")
        print(f"{'='*60}")
    
    cleanup()

if __name__ == "__main__":
    # Verify we have 2 GPUs before attempting NCCL initialization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires 2 GPUs.")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(f"This script requires 2 GPUs, but only {num_gpus} GPU(s) available.")
    
    print(f"✓ Found {num_gpus} GPUs. Starting NCCL pipeline on 2 GPUs...")
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
    print("="*60 + "\n")
    
    WORLD_SIZE = 2
    # Use PyTorch's multiprocessing launcher to spawn one process per GPU
    # Each process will run run_nccl_pipeline with its assigned rank
    mp.spawn(run_nccl_pipeline, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)