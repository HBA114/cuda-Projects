#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

const int number = 1024;
const int allocating_size = number * sizeof(int);

void *cpu_p;
void *gpu_p;

void cpu_alloc()
{
    cpu_p = malloc(allocating_size);
}

void cpu_set_numbers()
{
    int *cpu_int32 = (int *)cpu_p;
    for (int i = 0; i < number; i++)
        cpu_int32[i] = (i + 1) * 2;
}

void cpu_free()
{
    free(cpu_p);
}

void cpu_memory_to_gpu_memory()
{
    cudaError_t result = cudaMemcpy(gpu_p, cpu_p, allocating_size, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
}

void gpu_memory_to_cpu_memory()
{
    cudaError_t result = cudaMemcpy(cpu_p, gpu_p, allocating_size, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
}

void gpu_alloc()
{
    cudaError_t result = cudaMalloc(&gpu_p, allocating_size);
    assert(result == cudaSuccess);
}

void gpu_free()
{
    cudaError_t result = cudaFree(gpu_p);
    assert(result == cudaSuccess);
}

__global__ void gpu_add(int *gpu_numbers)
{
    int id = threadIdx.x;

    gpu_numbers[id] = gpu_numbers[id] + 100;
}

int main()
{
    cpu_alloc();
    cpu_set_numbers();

    int *cpu_int32 = (int *)cpu_p;
    for (int i = 0; i < number; i++)
        printf("%d \n", cpu_int32[i]);

    printf("\n ------------------------------ \n\n");
    gpu_alloc();
    cpu_memory_to_gpu_memory();

    // execute
    //! if nnumber is greater than 1024,
    //! gpu_add will not executed as expected
    gpu_add<<<1, number>>>((int *)gpu_p);

    cudaError_t result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    gpu_memory_to_cpu_memory();

    cpu_int32 = (int *)cpu_p;
    for (int i = 0; i < number; i++)
        printf("%d \n", cpu_int32[i]);

    gpu_free();
    cpu_free();

    printf("Completed.\n");

    return 0;
}
/*
compile: nvcc -arch compute_50 kernerl.cu
*/
