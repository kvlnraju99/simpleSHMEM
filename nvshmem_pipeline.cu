/*
 * NVSHMEM Pipeline Parallelism Demo
 *
 * Equivalent to NCLL.py but using NVSHMEM for GPU-to-GPU communication.
 * Implements a 2-GPU pipeline with same architecture:
 * - GPU 0: Linear(2048 -> 1024) + ReLU
 * - GPU 1: Linear(1024 -> 10) + ReLU
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <stdlib.h>

// Configuration
#define BATCH_SIZE 128
#define D_IN 2048
#define D_HIDDEN 1024
#define D_OUT 10
#define NUM_ITERATIONS 5
#define SEED 42

// Error checking macros
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NVSHMEM_CHECK(call)                                                    \
  do {                                                                         \
    int err = call;                                                            \
    if (err != 0) {                                                            \
      fprintf(stderr, "NVSHMEM error at %s:%d: code %d\n", __FILE__, __LINE__, \
              err);                                                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =============================================================================
// CUDA Kernels
// =============================================================================

// Initialize weights with fixed seed (like PyTorch's Kaiming Uniform)
__global__ void init_weights_kernel(float *weights, int size,
                                    unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    // Simple uniform initialization [-0.1, 0.1]
    weights[idx] = (curand_uniform(&state) - 0.5f) * 0.2f;
  }
}

// Generate random input data
__global__ void generate_input_kernel(float *data, int size, unsigned long seed,
                                      int iteration) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed + iteration, idx, 0, &state);
    data[idx] = curand_normal(&state);
  }
}

// Linear layer: out = weight @ input + bias
__global__ void
linear_forward_kernel(const float *input,  // [batch, in_features]
                      const float *weight, // [out_features, in_features]
                      const float *bias,   // [out_features]
                      float *output,       // [batch, out_features]
                      int batch_size, int in_features, int out_features) {
  int batch_idx = blockIdx.y;
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx < batch_size && out_idx < out_features) {
    float sum = bias[out_idx];
    for (int i = 0; i < in_features; i++) {
      sum += weight[out_idx * in_features + i] *
             input[batch_idx * in_features + i];
    }
    output[batch_idx * out_features + out_idx] = sum;
  }
}

// ReLU activation: out = max(0, in)
__global__ void relu_kernel(float *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = fmaxf(0.0f, data[idx]);
  }
}

// Compute L2 norm for validation
__global__ void compute_norm_kernel(const float *data, float *partial_sums,
                                    int size) {
  __shared__ float shared_sum[256];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  float sum = 0.0f;
  if (idx < size) {
    sum = data[idx] * data[idx];
  }
  shared_sum[tid] = sum;
  __syncthreads();

  // Reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = shared_sum[0];
  }
}

// =============================================================================
// Model Structure
// =============================================================================

struct PipelineStage {
  float *weight;
  float *bias;
  int in_features;
  int out_features;
  int gpu_id;
};

void init_stage(PipelineStage *stage, int in_features, int out_features,
                int gpu_id) {
  stage->in_features = in_features;
  stage->out_features = out_features;
  stage->gpu_id = gpu_id;

  CUDA_CHECK(cudaSetDevice(gpu_id));

  // Allocate weights and biases
  int weight_size = out_features * in_features;
  CUDA_CHECK(cudaMalloc(&stage->weight, weight_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&stage->bias, out_features * sizeof(float)));

  // Initialize with seed
  int threads = 256;
  int blocks_w = (weight_size + threads - 1) / threads;
  int blocks_b = (out_features + threads - 1) / threads;

  init_weights_kernel<<<blocks_w, threads>>>(stage->weight, weight_size,
                                             SEED + gpu_id);
  init_weights_kernel<<<blocks_b, threads>>>(stage->bias, out_features,
                                             SEED + gpu_id + 1000);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("[Rank %d] Initialized Stage (Linear %d->%d)\n", gpu_id, in_features,
         out_features);
}

void forward_stage(PipelineStage *stage, const float *input, float *output,
                   int batch_size) {
  CUDA_CHECK(cudaSetDevice(stage->gpu_id));

  // Linear layer
  dim3 block(256);
  dim3 grid((stage->out_features + block.x - 1) / block.x, batch_size);
  linear_forward_kernel<<<grid, block>>>(input, stage->weight, stage->bias,
                                         output, batch_size, stage->in_features,
                                         stage->out_features);

  // ReLU activation
  int size = batch_size * stage->out_features;
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  relu_kernel<<<blocks, threads>>>(output, size);

  CUDA_CHECK(cudaDeviceSynchronize());
}

float compute_norm(const float *data, int size, int gpu_id) {
  CUDA_CHECK(cudaSetDevice(gpu_id));

  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  float *d_partial;
  CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

  compute_norm_kernel<<<blocks, threads>>>(data, d_partial, size);

  float *h_partial = (float *)malloc(blocks * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float total = 0.0f;
  for (int i = 0; i < blocks; i++) {
    total += h_partial[i];
  }

  free(h_partial);
  CUDA_CHECK(cudaFree(d_partial));

  return sqrtf(total);
}

// =============================================================================
// Main Pipeline
// =============================================================================

int main(int argc, char **argv) {
  // Initialize NVSHMEM
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = NULL;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  if (n_pes != 2) {
    if (my_pe == 0) {
      fprintf(stderr, "This program requires exactly 2 PEs (GPUs), got %d\n",
              n_pes);
    }
    nvshmem_finalize();
    return 1;
  }

  // Set GPU device
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    if (my_pe == 0) {
      fprintf(stderr, "This program requires 2 GPUs, found %d\n", num_devices);
    }
    nvshmem_finalize();
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(my_pe));

  if (my_pe == 0) {
    printf("✓ Found %d GPUs. Starting NVSHMEM pipeline on 2 GPUs...\n",
           num_devices);
    cudaDeviceProp prop;
    for (int i = 0; i < 2; i++) {
      CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
      printf("  GPU %d: %s\n", i, prop.name);
    }
    printf("============================================================\n\n");
  }

  printf("[Rank %d] NVSHMEM initialized on GPU %d\n", my_pe, my_pe);

  // Create pipeline stages
  PipelineStage stage;
  if (my_pe == 0) {
    init_stage(&stage, D_IN, D_HIDDEN, my_pe);
  } else {
    init_stage(&stage, D_HIDDEN, D_OUT, my_pe);
  }

  // Allocate buffers
  float *input = NULL;
  float *output = NULL;
  float *intermediate = NULL; // NVSHMEM symmetric memory for full batch

  if (my_pe == 0) {
    CUDA_CHECK(cudaMalloc(&input, BATCH_SIZE * D_IN * sizeof(float)));
    // Allocate symmetric buffer for full batch (no micro-batching)
    intermediate =
        (float *)nvshmem_malloc(BATCH_SIZE * D_HIDDEN * sizeof(float));
  } else {
    // GPU 1 allocates same-sized symmetric buffer
    intermediate =
        (float *)nvshmem_malloc(BATCH_SIZE * D_HIDDEN * sizeof(float));
    CUDA_CHECK(cudaMalloc(&output, BATCH_SIZE * D_OUT * sizeof(float)));
  }

  nvshmem_barrier_all();

  // Pipeline execution
  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    if (my_pe == 0) {
      // === GPU 0: Generate, Compute, Send (like NCCL) ===

      // Generate random input for entire batch
      int size = BATCH_SIZE * D_IN;
      int threads = 256;
      int blocks = (size + threads - 1) / threads;
      generate_input_kernel<<<blocks, threads>>>(input, size, SEED, iter);
      CUDA_CHECK(cudaDeviceSynchronize());

      // Forward pass through first stage (full batch)
      forward_stage(&stage, input, intermediate, BATCH_SIZE);

      // Compute norm for validation
      float norm = compute_norm(intermediate, BATCH_SIZE * D_HIDDEN, my_pe);
      printf("[Rank 0] Iter %d: Sent activations shape [%d, %d], norm=%.4f\n",
             iter, BATCH_SIZE, D_HIDDEN, norm);

      // NVSHMEM: Send to PE 1 (blocking via quiet)
      nvshmem_float_put_nbi(intermediate,          // dest on PE 1
                            intermediate,          // source on PE 0
                            BATCH_SIZE * D_HIDDEN, // full batch size
                            1);                    // target PE
      nvshmem_quiet(); // Wait for transfer to complete

    } else {
      // === GPU 1: Receive, Compute (like NCCL) ===

      // Verify received data
      float norm = compute_norm(intermediate, BATCH_SIZE * D_HIDDEN, my_pe);
      printf("[Rank 1] Iter %d: Received activations shape [%d, %d], "
             "norm=%.4f\n",
             iter, BATCH_SIZE, D_HIDDEN, norm);

      // Forward pass through second stage (full batch)
      forward_stage(&stage, intermediate, output, BATCH_SIZE);

      float out_norm = compute_norm(output, BATCH_SIZE * D_OUT, my_pe);
      printf("[Rank 1] Iter %d: Final output shape [%d, %d], norm=%.4f\n", iter,
             BATCH_SIZE, D_OUT, out_norm);

      // Show predictions on last iteration (first 3 samples)
      if (iter == NUM_ITERATIONS - 1) {
        float h_output[3 * D_OUT];
        CUDA_CHECK(cudaMemcpy(h_output, output, 3 * D_OUT * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[Rank 1] Final iteration predictions (first 3 samples):\n");
        for (int i = 0; i < 3; i++) {
          printf("  Logits: [");
          for (int j = 0; j < D_OUT; j++) {
            printf("%.6f", h_output[i * D_OUT + j]);
            if (j < D_OUT - 1)
              printf(" ");
          }
          printf("]\n");
        }
      }
    }

    // Synchronize both PEs before next iteration
    nvshmem_barrier_all();
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  if (my_pe == 0) {
    printf("\n============================================================\n");
    printf("[NVSHMEM Pipeline] Completed %d iterations\n", NUM_ITERATIONS);
    printf("[NVSHMEM Pipeline] Total time: %.4fs\n", milliseconds / 1000.0f);
    printf("[NVSHMEM Pipeline] Avg time per iteration: %.4fs\n",
           milliseconds / 1000.0f / NUM_ITERATIONS);
    printf("============================================================\n");
  }

  // Cleanup
  if (my_pe == 0) {
    CUDA_CHECK(cudaFree(input));
  } else {
    CUDA_CHECK(cudaFree(output));
  }
  nvshmem_free(intermediate);
  CUDA_CHECK(cudaFree(stage.weight));
  CUDA_CHECK(cudaFree(stage.bias));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  nvshmem_finalize();
  return 0;
}
