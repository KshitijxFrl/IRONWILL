#ifndef MOE_H
#define MOE_H

#include <vector>

#include "module.h"
#include "tensor.h"
#include "parameter.h"
#include "linear.h"
#include "SwiGLU.h"

class MoE : public Module{
private:
    int batchLen;
    int seqLen;
    int dModel;
    int hiddenDim;

    int numExperts;
    int topK;

    Linear* router;

    std::vector<SWIGLU*> experts;

    Tensor* prevInput;
    Tensor* prevRouterLogits;
    Tensor* prevRouterProb;
    Tensor* prevExpertChoice;

    std::vector<Tensor*> prevExpertOutputs;

    float loadBalanceWeight;
    float lastLoadBalanceLoss;

public:
    MoE(
        int bl,
        int sl,
        int dm,
        int hidden,
        int expertsCount = 3
    );

    ~MoE();

    Tensor* feedForward(Tensor& in_data) override;

    Tensor* backward(Tensor& gradOut) override;
    
    std::vector<Parameter*> getParameter() override;
    
    float getLastLoadBalanceLoss();

};

#endif