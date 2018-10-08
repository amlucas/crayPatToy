#include <mpi.h>

#include "cptoy.h"

CPToy::CPToy(int size) :
    size(size), data(nullptr)
{
    cudaMalloc(&data, sizeof(float) * size);
}

CPToy::~CPToy()
{
    cudaFree(data);
}

__global__ void kernel_fill(float val, int n, float *data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        data[i] = val;
}

void cuda_test(float value) {

    const int nthreads = 128;
    const int nblocks = (n + nthreads - 1) / nthreads;

    kernel_fill <<<nblocks, nthreads>>> (value, size, data);
}

void mpi_test(float value) {
    float sum = 0;
    // TODO
}
