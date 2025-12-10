# Makefile for NVSHMEM Pipeline

# Modify these paths based on your NVSHMEM installation
# Default assumes you ran install_nvshmem.sh
NVSHMEM_HOME ?= $(HOME)/software/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
CUDA_HOME ?= /usr/local/cuda

NVCC = $(CUDA_HOME)/bin/nvcc
NVSHMEM_INCLUDE = -I$(NVSHMEM_HOME)/include
# NVSHMEM 3.x uses libnvshmem_host instead of libnvshmem
NVSHMEM_LIB = -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device

CUDA_ARCH ?= sm_80  # A100 compute capability
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

all: nvshmem_pipeline

nvshmem_pipeline: nvshmem_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) $(NVSHMEM_INCLUDE) $< -o $@ $(NVSHMEM_LIB) -lcurand

clean:
	rm -f nvshmem_pipeline

.PHONY: all clean
