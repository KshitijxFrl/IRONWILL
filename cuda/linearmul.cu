#include"../include/linearmul.h"
#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<map>
#include<utility>


__global__ void linearmul(float* mA, float* mB, float* mC, float* bias,int N, int element_per_row_matA, int total_col_matC){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float temp = 0.0;
        
        //these two give the which is row of mat a and which col of mat b to use for matmul for matC with index id
        
        int row_num_A = (id / total_col_matC) * element_per_row_matA; // we need to multiply the row num to get the index for the first element from that row bu skipping total number of element per row
        int col_num_B = id % total_col_matC; // we dont need to multiply anything to get the start index of col as col num and start index for col are same

        for(int i = 0; i < element_per_row_matA; i++){
            temp += mA[row_num_A] * mB[col_num_B];
            row_num_A = row_num_A + 1;
            col_num_B = col_num_B + total_col_matC; 
        }

        mC[id] = temp + bias[id % total_col_matC];
    }

}



void linearmul2D(Tensor& A, Tensor& B, Tensor& out, Tensor& bias){
    // we are assuming matA = m x k and matB = k x n so we have matC = m x n
    size_t mat_sizeA = A.getTensorSize() * sizeof(float);
    size_t mat_sizeB = B.getTensorSize() * sizeof(float);
    size_t mat_sizeOut = out.getTensorSize() * sizeof(float);
    size_t bias_size = bias.getTensorSize() * sizeof(float);

    int m = A.getTensorShape().front().first * A.getTensorShape()[1].first;
    int k = A.getTensorShape().back().first; // or the it can be B->shape.front().first doesnt matter
    int n = B.getTensorShape().back().first;


    int totalColums_matOut  = n; 

    
    int threads_per_block = 256;
    //how many block are required (just a safe 1D formula not a hard rule)
    //blocks_per_grid = (dataset_size + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid = (out.getTensorSize() + threads_per_block - 1) / threads_per_block;

    linearmul<<<blocks_per_grid, threads_per_block>>>(A.getTenorGpuData(), B.getTenorGpuData(), out.getTenorGpuData(), bias.getTenorGpuData(), out.getTensorSize(), k, totalColums_matOut);

    return;
}



__global__ void biasGradK(float* mBias, float* mTarget, int outFeatures, int rows){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j < outFeatures){
        float sum = 0.0f;

        for(int r = 0; r < rows; r++){
            sum += mTarget[r * outFeatures + j];
        }

        mBias[j] = sum;
    }
}

void biasGradient(Tensor& B, Tensor& grad_out){
    int outFeatures = B.getTensorSize();

    int gradOutSize = grad_out.getTensorSize();
    int rows = gradOutSize / outFeatures;

    int threads_per_block = 256;
    int blocks_per_grid = (outFeatures + threads_per_block - 1) / threads_per_block;

    biasGradK<<<blocks_per_grid, threads_per_block>>>(B.getTenorGpuData(),grad_out.getTenorGpuData(),outFeatures,rows);
}
