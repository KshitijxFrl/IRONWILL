#include<cuda_runtime.h>
#include"../include/embedding_lookup.h"


__global__ void lookup(float* ten_inData, float* ten_emb, float* ten_newX, int dModel, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        int token_id = (int)ten_inData[id];
        int vector_start = id * dModel;
        int emb_start = token_id * dModel;


        for(int i = 0; i < dModel; i++){
            ten_newX[vector_start + i] = ten_emb[emb_start + i];       
        }
    }

}

void embeddingLookUp(Tensor& inData, Tensor& emb, Tensor& newX, int d_model){
    
int thread_per_block = 256;
int block_per_thread = (newX.getTensorSize() + thread_per_block - 1) / thread_per_block; 

lookup<<<block_per_thread, thread_per_block>>>(inData.getTenorGpuData(), emb.getTenorGpuData(), newX.getTenorGpuData(), d_model, inData.getTensorSize());

}


__global__ void embeddingBackwardK(
    float* tokenIds, float* gradOut, float* embGrad,int dModel,int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        int token_id = (int)tokenIds[id];

        int gradOutStart = id * dModel;
        int embGradStart = token_id * dModel;

        for(int i = 0; i < dModel; i++){
            //first time using it didt get comfertabel too much so only using it here for now 
            atomicAdd(&embGrad[embGradStart + i],gradOut[gradOutStart + i]);
        }
    }
}

void embeddingBackward(Tensor& tokenIds,Tensor& gradOut,Tensor& embGrad,int d_model){
   
    int thread_per_block = 256;
    int N = tokenIds.getTensorSize();

    int block_per_grid = (N + thread_per_block - 1) / thread_per_block;

    embGrad.fill(0.0f);

    embeddingBackwardK<<<block_per_grid, thread_per_block>>>(tokenIds.getTenorGpuData(),gradOut.getTenorGpuData(),embGrad.getTenorGpuData(),d_model,N);
}