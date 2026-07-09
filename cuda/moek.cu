#include<cuda_runtime.h>
#include"../include/moek.h"

__global__ void routerSoftmaxTop1K(float* routerLogits,float* routerProb,float* expertChoice,int rows,int numExperts){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows){
        int rowStart = row * numExperts;

        float maxVal = routerLogits[rowStart];

        for(int e = 1; e < numExperts; e++){
            float val = routerLogits[rowStart + e];

            if(val > maxVal){
                maxVal = val;
            }
        }

        float expSum = 0.0f;

        for(int e = 0; e < numExperts; e++){
            float expVal = expf(routerLogits[rowStart + e] - maxVal);
            routerProb[rowStart + e] = expVal;
            expSum += expVal;
        }

        int bestExpert = 0; //dont accidentley start with any other dont change this or dont do wired pass
                            //U there lookin at this comment the above cooment is due to the moe desined for this iterration loopk at the top of moe cpp to get a breif idea what i am talking


        float bestProb = -1.0f;

        for(int e = 0; e < numExperts; e++){
            routerProb[rowStart + e] = routerProb[rowStart + e] / expSum;

            if(routerProb[rowStart + e] > bestProb){
                bestProb = routerProb[rowStart + e];
                bestExpert = e;
            }
        }

        expertChoice[row] = (float)bestExpert;
    }
}

void routerSoftmaxTop1(Tensor& routerLogits,Tensor& routerProb,Tensor& expertChoice,int batchLen,int seqLen,int numExperts){

    int rows = batchLen * seqLen;

    int thread_per_block = 256;
    int block_per_grid = (rows + thread_per_block - 1) / thread_per_block;

    routerSoftmaxTop1K<<<block_per_grid, thread_per_block>>>(routerLogits.getTenorGpuData(),routerProb.getTenorGpuData(),expertChoice.getTenorGpuData(),rows,numExperts);
}


////


__global__ void moeCombineForwardK(float** expertOutputs,float* routerProb,float* expertChoice,float* out,int rows,int dModel,int numExperts,int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        int row = id / dModel;

        int selectedExpert = (int)expertChoice[row];

        if(selectedExpert < 0) selectedExpert = 0;

        if(selectedExpert >= numExperts) selectedExpert = numExperts - 1;
        

        float gateProb = routerProb[row * numExperts + selectedExpert];

        out[id] = expertOutputs[selectedExpert][id] * gateProb;
    }
}

void moeCombineForward(std::vector<Tensor*>& expertOutputs,Tensor& routerProb,Tensor& expertChoice,Tensor& out,int batchLen,int seqLen,int dModel,int numExperts){
    
    float** hExpertPtrs = new float*[numExperts];

    for(int i = 0; i < numExperts; i++){
        hExpertPtrs[i] = expertOutputs[i]->getTenorGpuData();
    }

    float** dExpertPtrs = nullptr;

    cudaMalloc(&dExpertPtrs, numExperts * sizeof(float*));

    cudaMemcpy(dExpertPtrs,hExpertPtrs,numExperts * sizeof(float*),cudaMemcpyHostToDevice);

    int N = out.getTensorSize();
    int rows = batchLen * seqLen;

    int thread_per_block = 256;
    int block_per_grid = (N + thread_per_block - 1) / thread_per_block;

    moeCombineForwardK<<<block_per_grid, thread_per_block>>>(dExpertPtrs,routerProb.getTenorGpuData(),expertChoice.getTenorGpuData(),out.getTenorGpuData(),rows,dModel,numExperts,N);
    cudaDeviceSynchronize();

    cudaFree(dExpertPtrs);
    delete[] hExpertPtrs;
}


///

__global__ void moeBackwardRouterAndExpertsK(
    float* gradOut,float** expertOutputs,float* routerProb,float* expertChoice,float** gradExpertOutputs,float* gradRouterLogits,int rows,int dModel,int numExperts){

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows){
        int selectedExpert = (int)expertChoice[row];

        if(selectedExpert < 0) selectedExpert = 0;

        if(selectedExpert >= numExperts) selectedExpert = numExperts - 1;
        

        float selectedProb = routerProb[row * numExperts + selectedExpert];

        int rowStart = row * dModel;

        float gradSelectedProb = 0.0f;

        for(int d = 0; d < dModel; d++){
            int idx = rowStart + d;

            float go = gradOut[idx];
            float expertVal = expertOutputs[selectedExpert][idx];

            gradExpertOutputs[selectedExpert][idx] = go * selectedProb;

            gradSelectedProb += go * expertVal;
        }

        float dot = gradSelectedProb * selectedProb;

        for(int e = 0; e < numExperts; e++){
            float prob = routerProb[row * numExperts + e];

            float gradProb = 0.0f;

            if(e == selectedExpert){
                gradProb = gradSelectedProb;
            }

            gradRouterLogits[row * numExperts + e] = prob * (gradProb - dot);
        }
    }
}

void moeBackwardRouterAndExperts(Tensor& gradOut,std::vector<Tensor*>& expertOutputs,Tensor& routerProb,Tensor& expertChoice,std::vector<Tensor*>& gradExpertOutputs,Tensor& gradRouterLogits,int batchLen,int seqLen,int dModel,int numExperts){
    
    for(int i = 0; i < numExperts; i++){
        gradExpertOutputs[i]->fill(0.0f);
    }

    gradRouterLogits.fill(0.0f);

    float** hExpertPtrs = new float*[numExperts];
    float** hGradExpertPtrs = new float*[numExperts];

    for(int i = 0; i < numExperts; i++){
        hExpertPtrs[i] = expertOutputs[i]->getTenorGpuData();
        hGradExpertPtrs[i] = gradExpertOutputs[i]->getTenorGpuData();
    }

    float** dExpertPtrs = nullptr;
    float** dGradExpertPtrs = nullptr;

    cudaMalloc(&dExpertPtrs, numExperts * sizeof(float*));
    cudaMalloc(&dGradExpertPtrs, numExperts * sizeof(float*));

    cudaMemcpy(dExpertPtrs,hExpertPtrs,numExperts * sizeof(float*),cudaMemcpyHostToDevice);

    cudaMemcpy(dGradExpertPtrs,hGradExpertPtrs,numExperts * sizeof(float*),cudaMemcpyHostToDevice);

    int rows = batchLen * seqLen;

    int thread_per_block = 256;
    int block_per_grid = (rows + thread_per_block - 1) / thread_per_block;

    moeBackwardRouterAndExpertsK<<<block_per_grid, thread_per_block>>>(gradOut.getTenorGpuData(),dExpertPtrs,routerProb.getTenorGpuData(),expertChoice.getTenorGpuData(),dGradExpertPtrs,gradRouterLogits.getTenorGpuData(),rows,dModel,numExperts);
    cudaDeviceSynchronize();

    cudaFree(dExpertPtrs);
    cudaFree(dGradExpertPtrs);

    delete[] hExpertPtrs;
    delete[] hGradExpertPtrs;
}

////


__global__ void moeLoadBalanceStatsK(float* routerProb,float* expertChoice,float* expertFraction,float* meanRouterProb,float* lossParts,int rows,int numExperts){
    
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if(e < numExperts){
        float count = 0.0f;
        float probSum = 0.0f;

        for(int r = 0; r < rows; r++){
            int chosen = (int)expertChoice[r];

            if(chosen == e){
                count += 1.0f;
            }

            probSum += routerProb[r * numExperts + e];
        }

        float f = count / rows;
        float p = probSum / rows;

        expertFraction[e] = f;
        meanRouterProb[e] = p;

        lossParts[e] = f * p;
    }
}

__global__ void moeLoadBalanceGradK(float* routerProb,float* expertFraction,float* gradRouterLogits,int rows,int numExperts,float loadBalanceWeight){
    
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows){

        float dot = 0.0f;

        for(int e = 0; e < numExperts; e++){
            float gradProb = loadBalanceWeight * numExperts * expertFraction[e] / rows;
            float prob = routerProb[row * numExperts + e];

            dot += gradProb * prob;
        }

        for(int e = 0; e < numExperts; e++){
            float gradProb = loadBalanceWeight * numExperts * expertFraction[e] / rows;
            float prob = routerProb[row * numExperts + e];

            float gradLogit = prob * (gradProb - dot);

            gradRouterLogits[row * numExperts + e] += gradLogit;
        }
    }
}


float moeLoadBalanceLossAndGrad(Tensor& routerProb,Tensor& expertChoice,Tensor& gradRouterLogits,int batchLen,int seqLen,int numExperts,float loadBalanceWeight){
    
    int rows = batchLen * seqLen;

    Tensor expertFraction({numExperts});
    Tensor meanRouterProb({numExperts});
    Tensor lossParts({numExperts});

    int thread_per_block = 256;
    int block_per_grid_stats = (numExperts + thread_per_block - 1) / thread_per_block;

    moeLoadBalanceStatsK<<<block_per_grid_stats, thread_per_block>>>(routerProb.getTenorGpuData(),expertChoice.getTenorGpuData(),expertFraction.getTenorGpuData(),meanRouterProb.getTenorGpuData(),lossParts.getTenorGpuData(),rows,numExperts);

    int block_per_grid_rows = (rows + thread_per_block - 1) / thread_per_block;

    moeLoadBalanceGradK<<<block_per_grid_rows, thread_per_block>>>(routerProb.getTenorGpuData(),expertFraction.getTenorGpuData(),gradRouterLogits.getTenorGpuData(),rows,numExperts,loadBalanceWeight);

    cudaDeviceSynchronize();

    std::vector<float> cpuLossParts = lossParts.downloadata();

    float lb_Loss = 0.0f;

    for(int i = 0; i < numExperts; i++) lb_Loss += cpuLossParts[i];
    

    lb_Loss = lb_Loss * numExperts * loadBalanceWeight;

    expertFraction.clear();
    meanRouterProb.clear();
    lossParts.clear();

    return lb_Loss;
}

