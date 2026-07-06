#include<cuda_runtime.h>
#include"../include/silu.h"

__global__ void onlySiluK(float* gate, float* out, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float x = gate[id];
        float sig = 1.0f / (1.0f + expf(-x));
        out[id] = x * sig;
    }
}


__global__ void siluK(float* prjGate, float* prjUP, float* prjGateOut, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float x = prjGate[id];
        prjGateOut[id] = (x * (1.0f /(1.0f +expf(-x)))) * prjUP[id];
        
    }

}


__global__ void swigluLocalBackwardK(float* gradHidden, float* gate, float* up, float* siluGate, float* gradUp, float* gradGate, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        float gh = gradHidden[id];
        float g  = gate[id];
        float u  = up[id];

        float sig = 1.0f / (1.0f + expf(-g));

        float silu_derivative = sig + g * sig * (1.0f - sig);

        gradUp[id] = gh * siluGate[id];

        gradGate[id] = gh * u * silu_derivative;
    }
}


void silu(Tensor& Gatein, Tensor& Up, Tensor& Gateout){

    int thread_per_block = 256;
    int block_per_grid = (Gatein.getTensorSize() + thread_per_block - 1)/thread_per_block;

    siluK<<<block_per_grid, thread_per_block>>>(Gatein.getTenorGpuData(), Up.getTenorGpuData(), Gateout.getTenorGpuData(), Gatein.getTensorSize());

}

void onlySilu(Tensor& Gatein, Tensor& Gateout){
    int thread_per_block = 256;
    int block_per_grid = (Gatein.getTensorSize() + thread_per_block - 1) / thread_per_block;

    onlySiluK<<<block_per_grid, thread_per_block>>>(Gatein.getTenorGpuData(),Gateout.getTenorGpuData(),Gatein.getTensorSize());
}

void swigluLocalBackward(Tensor& gradHidden,Tensor& gate,Tensor& up,Tensor& siluGate,Tensor& gradUp,Tensor& gradGate){
    int thread_per_block = 256;
    int block_per_grid = (gradHidden.getTensorSize() + thread_per_block - 1) / thread_per_block;

    swigluLocalBackwardK<<<block_per_grid, thread_per_block>>>(gradHidden.getTenorGpuData(),gate.getTenorGpuData(),up.getTenorGpuData(),siluGate.getTenorGpuData(),gradUp.getTenorGpuData(),gradGate.getTenorGpuData(),gradHidden.getTensorSize());
}