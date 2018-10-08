#include <mpi.h>

#include "cptoy.h"

CPToy::CPToy(int size) :
    size(size), data(nullptr)
{
    MPI_Init(nullptr, nullptr);
    cudaMalloc(&data, sizeof(float) * size);
}

CPToy::~CPToy()
{
    cudaFree(data);
    MPI_Finalize();
}

__global__ void kernel_fill(float val, int n, float *data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        data[i] = val;
}

void CPToy::cuda_test(float value) {

    const int nthreads = 128;
    const int nblocks = (size + nthreads - 1) / nthreads;

    kernel_fill <<<nblocks, nthreads>>> (value, size, data);
}

void CPToy::mpi_test(float value) {
    int rank, root = 0;
    float sum = 0;


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(&value, &sum, 1, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

    if (rank == root)
        printf("sum of values: %g\n", sum);
}
