#ifndef GQA_H
#define GQA_H

#include "tensor.h"

void attention_GQA(
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Out,
    Tensor& AttentionScores,
    int batchlen,
    int seqlen,
    int head,
    int kv_head,
    int head_dim
);

void attention_GQA_Backward(
    Tensor& gradOut,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& attentionScores,
    Tensor& gradQ,
    Tensor& gradK,
    Tensor& gradV,
    int batchlen,
    int seqlen,
    int head,
    int kv_head,
    int head_dim
);

#endif