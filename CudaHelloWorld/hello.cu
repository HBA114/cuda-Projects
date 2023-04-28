#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>();
    printf("Hello World from CPU!\n");
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

/*
compile:    nvcc -arch compute_50 hello.cu
run:        ./a.out 
*/