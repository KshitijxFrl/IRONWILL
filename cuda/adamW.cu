#include<cuda_runtime.h>
#include"../include/adamW.h"

//can be way faster - going with this
__global__ void adamWUpdateK(float* data,float* gradient,float* m,float* v,float learningRate,float beta1,float beta2,float epsilon,float weightDecay,float beta1Pow,float beta2Pow,int N){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){

        float weight = data[id];
        float grad = gradient[id];
        

        m[id] = beta1 * m[id] + (1.0f - beta1) * grad;
        v[id] = beta2 * v[id] + (1.0f - beta2) * grad * grad;

        float mHat = m[id] / (1.0f - beta1Pow);
        float vHat = v[id] / (1.0f - beta2Pow);


        float adamUpdate = mHat / (sqrtf(vHat) + epsilon);


        data[id] = weight - learningRate * adamUpdate - learningRate * weightDecay * weight;
    }
}


void adamWUpdate(Tensor& data,Tensor& gradient,Tensor& m,Tensor& v,float learningRate,float beta1,float beta2,float epsilon,float weightDecay,int stepCount){

    int thread_per_block = 256;
    int block_per_grid = (data.getTensorSize() + thread_per_block - 1) / thread_per_block;

    float beta1Pow = powf(beta1, (float)stepCount);
    float beta2Pow = powf(beta2, (float)stepCount);

    adamWUpdateK<<<block_per_grid, thread_per_block>>>(data.getTenorGpuData(),gradient.getTenorGpuData(),m.getTenorGpuData(),v.getTenorGpuData(),learningRate,beta1,beta2,epsilon,weightDecay,beta1Pow,beta2Pow,data.getTensorSize());
}