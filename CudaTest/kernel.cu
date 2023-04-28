#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

const int alan_boyut = 1 * 1024 * 1024 * 1024;

void *cpu_p;
void *gpu_p;

void cpu_alloc()
{
    cpu_p = malloc(alan_boyut);
}

void gpu_alloc()
{
    cudaError_t sonuc = cudaMalloc(&gpu_p, alan_boyut);
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
