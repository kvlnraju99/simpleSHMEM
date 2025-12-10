# Makefile for NVSHMEM Pipeline

# Modify these paths based on your NVSHMEM installation
NVSHMEM_HOME ?= /path/to/nvshmem
CUDA_HOME ?= /usr/local/cuda

NVCC = $(CUDA_HOME)/bin/nvcc
NVSHMEM_INCLUDE = -I$(NVSHMEM_HOME)/include
NVSHMEM_LIB = -L$(NVSHMEM_HOME)/lib -lnvshmem

CUDA_ARCH ?= sm_80  # A100 compute capability
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

all: nvshmem_pipeline

nvshmem_pipeline: nvshmem_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) $(NVSHMEM_INCLUDE) $< -o $@ $(NVSHMEM_LIB) -lcurand

clean:
	rm -f nvshmem_pipeline

.PHONY: all clean
