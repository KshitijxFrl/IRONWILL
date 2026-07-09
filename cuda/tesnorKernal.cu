#include<cuda_runtime.h>
#include"../include/tesnorKernal.h"

//simple utility tensors

__global__ void addTensorK(float* __restrict__ mA, const float* __restrict__ mB, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){mA[id] += mB[id];}


}

void addTensor(float* A, Tensor& B){

    int thread_per_block = 256;
    int block_per_grid = (B.getTensorSize() + thread_per_block - 1) / thread_per_block;

    addTensorK<<<block_per_grid, thread_per_block>>>(A, B.getTenorGpuData(), B.getTensorSize());

}


//COPY TENSOR
__global__ void copyTensorK(float* A, float* B, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        B[id] = A[id];
    }
}

void copyTensorRaw(float* A, float* B, int N){
    int thread_per_block = 256;
    int block_per_grid = (N + thread_per_block - 1) / thread_per_block;

    copyTensorK<<<block_per_grid, thread_per_block>>>(A, B, N);
}