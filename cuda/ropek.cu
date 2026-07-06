#include<cuda_runtime.h>
#include"../include/rope.h"

__global__ void rotaion(float* mInData, int sd, int h, int hd, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tokenSize = h * hd;

    if(id < N){
        if(id % tokenSize == 0){
            int global_token_idx = id / (h * hd);

            int position = global_token_idx % sd;

            for(int j = 0; j < h; j++){
                int headStart = id + (j * hd);
                int pairIDX = 0;

                //minimum 2 element are required
                for(int i = 1; i < hd; i += 2){

                    float theta =position * (1.0f / powf(10000.0f,(2.0f * pairIDX) / (float)hd));                    
        
                    
                    float even = mInData[headStart + i - 1];
                    float odd  = mInData[headStart + i];

                    //even
                    mInData[headStart + i - 1] = even * cosf(theta)- odd * sinf(theta);

                    //odd
                    mInData[headStart + i] = even * sinf(theta) + odd * cosf(theta);

                    pairIDX++;
                    
                }

            }                
        }
    }

}

//in data sape (batch, seq, head, head_dimention)
void simpleRoPE(Tensor& in_data, int seq_dimention,int head, int head_dimention){
    int thread_per_block = 256; 
    int block_per_grid = (in_data.getTensorSize() + thread_per_block - 1) / thread_per_block;
    

    rotaion<<<block_per_grid, thread_per_block>>>(in_data.getTenorGpuData(), seq_dimention, head, head_dimention, in_data.getTensorSize());

}

//Backward Rope

__global__ void rotationBackwardK(float* mGradData, int sd, int h, int hd, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tokenSize = h * hd;

    if(id < N){
        if(id % tokenSize == 0){
            int global_token_idx = id / (h * hd);
            int position = global_token_idx % sd;

            for(int j = 0; j < h; j++){
                int headStart = id + (j * hd);
                int pairIDX = 0;

                for(int i = 1; i < hd; i += 2){
                    float theta = position * (1.0f / powf(10000.0f, (2.0f * pairIDX) / (float)hd));

                    float evenGrad = mGradData[headStart + i - 1];
                    float oddGrad  = mGradData[headStart + i];

                    mGradData[headStart + i - 1] = evenGrad * cosf(theta) + oddGrad * sinf(theta);

                    mGradData[headStart + i] = -evenGrad * sinf(theta) + oddGrad * cosf(theta);

                    pairIDX++;
                }
            }
        }
    }
}

void simpleRoPEBackward(Tensor& grad_data, int seq_dimention, int head, int head_dimention){
    int thread_per_block = 256;
    int block_per_grid = (grad_data.getTensorSize() + thread_per_block - 1) / thread_per_block;

    rotationBackwardK<<<block_per_grid, thread_per_block>>>(grad_data.getTenorGpuData(),seq_dimention,head,head_dimention,grad_data.getTensorSize());
}