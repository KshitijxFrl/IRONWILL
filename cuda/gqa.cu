#include<cuda_runtime.h>
#include<float.h>
#include <stdexcept>
#include"../include/gqa.h"

// group_size = heads / kv_heads;

// kv_head = q_head / group_size;

constexpr int MAX_SEQ = 512;

__device__ void dotqkHold_V(int seqLen, int headDim, int V_head_start, int jumpToNext_V_head, float qkHold[], float* V, float* outX, int currIdx){
    int currVOffset = 0;
    

    for(int i = 0; i < headDim; i++){
        outX[currIdx + i] = 0.0f;
        
        currVOffset = V_head_start;

        for(int j = 0; j < seqLen; j++){
            outX[currIdx + i] += qkHold[j] * V[currVOffset + i];
            currVOffset += jumpToNext_V_head;
        }
        
    }

}

__device__ void dotQK_with_SoftMax(int seqLen, int headDim, int queryToken, int k_head_start, int jumpToNext_K_head, float qkH[], float* Q, float* K){
    
    float valSum = 0.0f;
    float maxScore = -FLT_MAX;
    float scale = 1.0f / sqrtf((float)headDim);

    for(int i = 0; i < seqLen; i++){
        if(i > queryToken){
            qkH[i] = 0.0f;
            k_head_start += jumpToNext_K_head;
            continue;
        }

        qkH[i] = 0.0f;

        for(int j = 0; j < headDim; j++){
            qkH[i] +=  Q[j] * K[k_head_start + j];

        }

        qkH[i] *= scale;
        
        if(maxScore < qkH[i]) maxScore = qkH[i];
        
        k_head_start += jumpToNext_K_head;             
        
    }

    for(int k = 0; k < seqLen; k++){
        if(k > queryToken){
            qkH[k] = 0.0f;
            continue;
        }

        qkH[k] = expf(qkH[k] - maxScore);
        valSum += qkH[k];
    }    

    for(int m = 0; m < seqLen; m++){
        if(m > queryToken){
            qkH[m] = 0.0f;
            continue;
        }

        qkH[m] /= valSum;
    }


}


// working on the out tesnor not groping kv or q or anything what ever the our index deman it get served
__global__ void gqaK(float* prjQ, float* prjK, float* prjV, float* prjO, float* attentionScores, int b, int sl, int hd, int h, int kvh, int N){
    
    int headId = blockIdx.x * blockDim.x + threadIdx.x;

    int id = headId * hd;

    if(id < N){
        int group_size_per_KV = h / kvh;

        float* currQ = &prjQ[id];
        
        int curr_batch = id / (sl * h * hd);
        

        //curr_Qhead and k_head_start are the starting index of the required head
        int local = id % (sl * h * hd);

        int curr_q_token = local / (h * hd);
        int curr_Qhead = (local / hd) % h;


        int kv_head = (curr_Qhead  / group_size_per_KV);
        
        int batchOffset = curr_batch * (sl * kvh * hd);
        int kvOffset = kv_head * hd;

        int k_head_start = batchOffset + kvOffset;
        

        float QK_hold[MAX_SEQ];

        dotQK_with_SoftMax(sl, hd, curr_q_token, k_head_start, kvh * hd, QK_hold, currQ, prjK);

        // save softmax scores
        int scoreStart = headId * sl;

        for(int i = 0; i < sl; i++){
            attentionScores[scoreStart + i] = QK_hold[i];
        }

        dotqkHold_V(sl, hd, k_head_start, kvh * hd, QK_hold, prjV, prjO, id);

    }

}

void attention_GQA(Tensor& Q, Tensor& K, Tensor& V, Tensor& Out, Tensor& AttentionScores, int batchlen, int seqlen, int head, int kv_head, int head_dim){

    int thread_per_block = 256;
    int totalHeads = Out.getTensorSize() / head_dim;

    int block_per_grid = (totalHeads + thread_per_block - 1) / thread_per_block;

    if (seqlen > MAX_SEQ) {
        throw std::runtime_error("Sequence length exceeds MAX_SEQ");
    }

    gqaK<<<block_per_grid, thread_per_block>>>(Q.getTenorGpuData(), K.getTenorGpuData(), V.getTenorGpuData(), Out.getTenorGpuData(), AttentionScores.getTenorGpuData(), batchlen, seqlen, head_dim, head, kv_head, Out.getTensorSize());
}


//So this is the backward and i tried something new with this It took time its a single flow fucntion 


//expanded fucntion for readability change when its clear or keep it if wanted
__global__ void gqaBackwardK(
    float* gradOut,          // gradAttentionOut [B, T, H, HD]
    float* Q,                // prevQ [B, T, H, HD]
    float* K,                // prevK [B, T, KVH, HD]
    float* V,                // prevV [B, T, KVH, HD]
    float* attentionScores,  // [B, T, H, T]

    float* gradQ,            // [B, T, H, HD]
    float* gradK,            // [B, T, KVH, HD]
    float* gradV,            // [B, T, KVH, HD]

    int b,int sl,int h,int kvh,int hd,int N){
    int headId = blockIdx.x * blockDim.x + threadIdx.x;
    int qStart = headId * hd;

    if(qStart < N){
        int group_size_per_KV = h / kvh;
        float scale = 1.0f / sqrtf((float)hd);

        int curr_batch = qStart / (sl * h * hd);

        int local = qStart % (sl * h * hd);

        int curr_q_token = local / (h * hd);
        int curr_q_head = (local / hd) % h;

        int curr_kv_head = curr_q_head / group_size_per_KV;

        int qOffset = qStart;
        int gradOutOffset = qStart;

        int scoreStart = headId * sl;

        int kvBatchOffset = curr_batch * (sl * kvh * hd);
        int kvHeadOffset = curr_kv_head * hd;

        float dScore[MAX_SEQ];

        //this is just dV and dAttentionScores
        for(int s = 0; s < sl; s++){
            float score = attentionScores[scoreStart + s];

            int vStart = kvBatchOffset + (s * kvh * hd) + kvHeadOffset;

            float dot = 0.0f;

            for(int d = 0; d < hd; d++){
                float go = gradOut[gradOutOffset + d];
                float vVal = V[vStart + d];

                dot += go * vVal;

                

                //nice utility 
                atomicAdd(&gradV[vStart + d], score * go);
            }

            dScore[s] = dot;
        }

        
        float softmaxDot = 0.0f;

        for(int s = 0; s < sl; s++){
            softmaxDot += dScore[s] * attentionScores[scoreStart + s];
        }

        //calcu dQ and dK
        for(int d = 0; d < hd; d++){
            float qVal = Q[qOffset + d];
            float gradQSum = 0.0f;

            for(int s = 0; s < sl; s++){
                float score = attentionScores[scoreStart + s];

                float dLogit = score * (dScore[s] - softmaxDot);
             
                dLogit *= scale;

                int kStart = kvBatchOffset + (s * kvh * hd) + kvHeadOffset;

                float kVal = K[kStart + d];

                gradQSum += dLogit * kVal;
                atomicAdd(&gradK[kStart + d], dLogit * qVal);
            }

            gradQ[qOffset + d] = gradQSum;
        }
    }
}

void attention_GQA_Backward(Tensor& gradOut, Tensor& Q,Tensor& K,Tensor& V,Tensor& attentionScores,Tensor& gradQ,Tensor& gradK,Tensor& gradV,int batchlen,int seqlen,int head,int kv_head,int head_dim){
    int thread_per_block = 256;
    int totalHeads = gradOut.getTensorSize() / head_dim;

    int block_per_grid = (totalHeads + thread_per_block - 1) / thread_per_block;

    if(seqlen > MAX_SEQ) throw std::runtime_error("Sequence length exceeds MAX_SEQ");
    

    gradQ.fill(0.0f);
    gradK.fill(0.0f);
    gradV.fill(0.0f);

    gqaBackwardK<<<block_per_grid, thread_per_block>>>(gradOut.getTenorGpuData(),Q.getTenorGpuData(),K.getTenorGpuData(),V.getTenorGpuData(),attentionScores.getTenorGpuData(),gradQ.getTenorGpuData(),gradK.getTenorGpuData(),gradV.getTenorGpuData(),batchlen,seqlen,head,kv_head,head_dim,gradOut.getTensorSize());
}
