#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

const int allocating_size = 1 * 1024 * 1024 * 1024;

void *cpu_p;
void *gpu_p;

void cpu_alloc()
{
    cpu_p = malloc(allocating_size);
}

void gpu_alloc()
{
    cudaError_t sonuc = cudaMalloc(&gpu_p, allocating_size);
    printf("%d\n",sonuc);
    // assert(sonuc == cudaSuccess); //! 
}

int main()
{
    cpu_alloc();

    gpu_alloc();

    return 0;
}

/*
compile:    nvcc -arch compute_50 kernel.cu
run:        ./a.out 
*/
