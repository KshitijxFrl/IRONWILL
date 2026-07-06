#include"../include/matmul2d.h"
#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<map>
#include<utility>



__global__ void matmul(float* mA, float* mB, float* mC, int N, int element_per_row_matA, int total_col_matC){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float temp = 0.0;
        
        //these two give the which is row of mat a and which col of mat b to use for matmul for matC with index id
        
        int row_num_A = (id / total_col_matC) * element_per_row_matA; // we need to multiply thhe row num to get the index for the first element from that row bu skipping total number of element per row
        int col_num_B = id % total_col_matC; // we dont need to multiply anything to get the start index of col as col num and start index for col are same

        for(int i = 0; i < element_per_row_matA; i++){
            temp += mA[row_num_A] * mB[col_num_B];
            row_num_A = row_num_A + 1;
            col_num_B = col_num_B + total_col_matC; 
        }

        mC[id] = temp;
    }

}

__global__ void matmulAT_B(float* mA, float* mB, float* mC, int N, int rowA, int colA, int rowB, int colB){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float temp = 0.0;
        
        //these two give the which is row of mat a and which col of mat b to use for matmul for matC with index id
        
        int rowC = id / colB; 
        int colCidx = id % colB; 

        for(int k = 0; k < rowA; k++)
        {
            temp += mA[k * colA + rowC] * mB[k * colB + colCidx];
        }

        mC[id] = temp;
    }

}

__global__ void matmulA_BT(float* mA, float* mB, float* mC,int N,int rowA, int colA,int rowB, int colB)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){

        int rowC = id / rowB;
        int colCidx = id % rowB;

        float temp = 0.0f;

        for(int k = 0; k < colA; k++){
            temp += mA[rowC * colA + k] *mB[colCidx * colB + k];
        }

        mC[id] = temp;
    }
}


void matmul2D(Tensor& A, Tensor& B, Tensor& C){
    // we are assuming matA = m x k and matB = k x n so we have matC = m x n
    size_t mat_sizeA = A.getTensorSize() * sizeof(float);
    size_t mat_sizeB = B.getTensorSize() * sizeof(float);

    int m = A.getTensorShape().front().first;
    int k = A.getTensorShape().back().first; // or the it can be B->shape.front().first doesnt matter
    int n = B.getTensorShape().back().first;


    int totalColums_matC  = n; 
 
    
    size_t mat_sizeC = C.getTensorSize() * sizeof(float);


    int threads_per_block = 256;
    //how many block are required (just a safe 1D formula not a hard rule)
    //blocks_per_grid = (dataset_size + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid = (C.getTensorSize() + threads_per_block - 1) / threads_per_block;

    matmul<<<blocks_per_grid, threads_per_block>>>(A.getTenorGpuData(), B.getTenorGpuData(), C.getTenorGpuData(), C.getTensorSize(), k, totalColums_matC);

    return;
}

void matmul2D_AT_B(Tensor& A, Tensor& B, Tensor& C){

    int colA = A.getTensorShape().back().first;

    int colB = B.getTensorShape().back().first;

    int rowA = A.getTensorSize() / colA;
    int rowB = B.getTensorSize() / colB;

    int threads_per_block = 256;
    int blocks_per_grid = (C.getTensorSize() + threads_per_block - 1) / threads_per_block;

    matmulAT_B<<<blocks_per_grid, threads_per_block>>>(A.getTenorGpuData(),B.getTenorGpuData(),C.getTenorGpuData(),C.getTensorSize(),rowA,colA,rowB,colB);
}

void matmul2D_A_BT(Tensor& A, Tensor& B, Tensor& C){

    int rowA = A.getTensorShape().front().first;
    int colA = A.getTensorShape().back().first;

    int rowB = B.getTensorShape().front().first;
    int colB = B.getTensorShape().back().first;

    int threads_per_block = 256;
    int blocks_per_grid =(C.getTensorSize() + threads_per_block - 1) /threads_per_block;

    matmulA_BT<<<blocks_per_grid, threads_per_block>>>(A.getTenorGpuData(),B.getTenorGpuData(),C.getTenorGpuData(),C.getTensorSize(),rowA,colA,rowB,colB);
}
