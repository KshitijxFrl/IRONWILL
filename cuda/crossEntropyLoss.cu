#include<cuda_runtime.h>
#include"../include/crossEntropyLoss.h"
#include<iostream>
#include<vector>

__global__ void crossEntropyLossK(float* logits,float* target,float* gradLogits,float* losses,int rows,int vocabSize,int validCount,int padID){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows){
        int targetToken = (int)target[row];
        int rowStart = row * vocabSize;

        if(targetToken == padID || validCount <= 0){
            losses[row] = 0.0f;

            for(int j = 0; j < vocabSize; j++){
                gradLogits[rowStart + j] = 0.0f;
            }

            return;
        }

        if(targetToken < 0) targetToken = 0;
        if(targetToken >= vocabSize) targetToken = vocabSize - 1;

        float maxLogit = logits[rowStart];

        for(int j = 1; j < vocabSize; j++){
            float val = logits[rowStart + j];

            if(val > maxLogit){
                maxLogit = val;
            }
        }

        float expSum = 0.0f;

        for(int j = 0; j < vocabSize; j++){expSum += expf(logits[rowStart + j] - maxLogit);}

        float logSumExp = logf(expSum);

        float targetLogit = logits[rowStart + targetToken];

        losses[row] = -(targetLogit - maxLogit - logSumExp);

        for(int j = 0; j < vocabSize; j++){
            float prob = expf(logits[rowStart + j] - maxLogit) / expSum;

            gradLogits[rowStart + j] = prob;
        }

        gradLogits[rowStart + targetToken] -= 1.0f;

        for(int j = 0; j < vocabSize; j++){gradLogits[rowStart + j] /= validCount;}

        losses[row] /= validCount;
    }
}


//This is another example for threminet where it is hyper focus on being a LLM builder
//may be changed in the future and improved upon 

//will shift to full gpu later
float crossEntropyLoss(Tensor& logits, Tensor& target, Tensor& gradLogits){
    int vocabSize = logits.getTensorShape().back().first;
    int rows = logits.getTensorSize() / vocabSize;
    int padID = 0;

    std::vector<float> cpuTargets = target.downloadata();
    int validCount = 0;

    for(int i = 0; i < cpuTargets.size(); i++){
        if((int)cpuTargets[i] != padID){
            validCount++;
        }
    }

    Tensor losses({rows});

    int thread_per_block = 256;
    int block_per_grid = (rows + thread_per_block - 1) / thread_per_block;

    crossEntropyLossK<<<block_per_grid, thread_per_block>>>(
        logits.getTenorGpuData(),
        target.getTenorGpuData(),
        gradLogits.getTenorGpuData(),
        losses.getTenorGpuData(),
        rows,
        vocabSize,
        validCount,
        padID
    );

    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess){
        std::cerr << "crossEntropyLoss kernel failed: "<< cudaGetErrorString(err)<< std::endl;
        losses.clear();
        return 0.0f;
    }

    std::vector<float> cpuLosses = losses.downloadata();

    float finalLoss = 0.0f;

    for(int i = 0; i < rows; i++){
        finalLoss += cpuLosses[i];
    }

    losses.clear();

    return finalLoss;
}
