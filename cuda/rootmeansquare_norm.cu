#include<cuda_runtime.h>
#include<float.h>
#include"../include/rootmeansquare_norm.h"

__global__ void rmsNormOnlyK(float* mVals, float* mRms, float* mNorm, int maxCols, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        int row = id / maxCols;
        mNorm[id] = mVals[id] / mRms[row];
    }
}

__global__ void rmsOfTokenK(float* mVals, float* rms_of_token, int maxCols, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float currMean = 0.0f;

    if(id < N){
        if(id % maxCols == 0){
            for(int i = 0; i < maxCols; i++){
                currMean += mVals[id + i] * mVals[id + i];
            }

            currMean /= maxCols;
            currMean = sqrtf(currMean + FLT_EPSILON);

            int row = id / maxCols;
            rms_of_token[row] = currMean;
        }
    }
}


__global__ void gammaGradK(float* mGammaGrad, float* mGradOut, float* mNorm, int featureLen, int rows){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j < featureLen){
        float sum = 0.0f;

        for(int r = 0; r < rows; r++){
            int idx = r * featureLen + j;
            sum += mGradOut[idx] * mNorm[idx];
        }

        mGammaGrad[j] = sum;
    }
}



__global__ void rmsnK(float* mVals, float* mNewVals, float* mGamma, int maxCols, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float currMean = 0.0;

    if(id < N){
        if(id % maxCols == 0){
            for(int i = 0; i < maxCols; i++){
                //square
                currMean += mVals[id + i] * mVals[id + i];
            }
            //mean;
            currMean /= maxCols;
            //square
            //we are adding e or FLT_EPSILON to prevent rms becoming zero

            currMean = sqrtf(currMean + FLT_EPSILON);

            //Normalising token vector with currMean
            for(int j = 0; j < maxCols; j++){
              mNewVals[id + j] = (mVals[id + j] / currMean) * mGamma[j];    
            }

            currMean = 0;
            
        }


    }
}


void rootMeanSquareNorm(Tensor& A, Tensor& newA, Tensor& gamma){
    size_t matsize_A = A.getTensorSize() * sizeof(float);
    size_t matsize_gamma = gamma.getTensorSize() * sizeof(float);

    int thread_per_block = 256;
    int block_per_grid   = (A.getTensorSize() +  thread_per_block - 1) / thread_per_block;

    rmsnK<<<block_per_grid, thread_per_block>>>(A.getTenorGpuData(), newA.getTenorGpuData(), gamma.getTenorGpuData(), A.getTensorShape().back().first, A.getTensorSize());
    
}

void gammaGrad(Tensor& gammaGradient, Tensor& gradOut, Tensor& norm){
    int featureLen = gammaGradient.getTensorSize();
    int rows = gradOut.getTensorSize() / featureLen;

    int thread_per_block = 256;
    int block_per_grid = (featureLen + thread_per_block - 1) / thread_per_block;

    gammaGradK<<<block_per_grid, thread_per_block>>>(gammaGradient.getTenorGpuData(),gradOut.getTenorGpuData(),norm.getTenorGpuData(),featureLen,rows);
}

void rmsOfToken(Tensor& A, Tensor& rms_of_token){
    int thread_per_block = 256;
    int block_per_grid = (A.getTensorSize() + thread_per_block - 1) / thread_per_block;

    int maxCols = A.getTensorShape().back().first;

    rmsOfTokenK<<<block_per_grid, thread_per_block>>>(A.getTenorGpuData(),rms_of_token.getTenorGpuData(),maxCols,A.getTensorSize());
}

void rmsNormOnly(Tensor& A, Tensor& rms_of_token, Tensor& norm){
    int thread_per_block = 256;
    int block_per_grid = (A.getTensorSize() + thread_per_block - 1) / thread_per_block;

    int maxCols = A.getTensorShape().back().first;

    rmsNormOnlyK<<<block_per_grid, thread_per_block>>>(A.getTenorGpuData(),rms_of_token.getTenorGpuData(),norm.getTenorGpuData(),maxCols,A.getTensorSize());
}


__global__ void rmsBackwardK(float* mGradOut,float* mGamma,float* mInput,float* mRms,float* mGradIn,int featureLen,int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        int row = id / featureLen;
        int col = id % featureLen;

        int rowStart = row * featureLen;

        float rms = mRms[row];

        float dot = 0.0f;

        for(int j = 0; j < featureLen; j++){
            float gradNorm_j = mGradOut[rowStart + j] * mGamma[j];
            float x_j = mInput[rowStart + j];

            dot += gradNorm_j * x_j;
        }

        float x = mInput[id];
        float gradNorm = mGradOut[id] * mGamma[col];

        float rms3 = rms * rms * rms;

        mGradIn[id] = (gradNorm / rms) - (x * dot / (featureLen * rms3));
    }
}

void rmsBackward(Tensor& gradOut, Tensor& gamma, Tensor& input, Tensor& rms_of_token, Tensor& gradIn){
    int thread_per_block = 256;
    int block_per_grid = (gradOut.getTensorSize() + thread_per_block - 1) / thread_per_block;

    int featureLen = gradOut.getTensorShape().back().first;

    rmsBackwardK<<<block_per_grid, thread_per_block>>>(gradOut.getTenorGpuData(),gamma.getTenorGpuData(),input.getTenorGpuData(),rms_of_token.getTenorGpuData(),gradIn.getTenorGpuData(),featureLen,gradOut.getTensorSize());
}