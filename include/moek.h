#ifndef MOEK_H
#define MOEK_H

#include"tensor.h"

void routerSoftmaxTop1(Tensor& routerLogits,Tensor& routerProb,Tensor& expertChoice,int batchLen,int seqLen,int numExperts);

void moeCombineForward(std::vector<Tensor*>& expertOutputs,Tensor& routerProb,Tensor& expertChoice,Tensor& out,int batchLen,int seqLen,int dModel,int numExperts);

void moeBackwardRouterAndExperts(Tensor& gradOut,std::vector<Tensor*>& expertOutputs,Tensor& routerProb,Tensor& expertChoice,std::vector<Tensor*>& gradExpertOutputs,Tensor& gradRouterLogits,int batchLen,int seqLen,int dModel,int numExperts);

float moeLoadBalanceLossAndGrad(Tensor& routerProb,Tensor& expertChoice,Tensor& gradRouterLogits,int batchLen,int seqLen,int numExperts,float loadBalanceWeight);

#endif