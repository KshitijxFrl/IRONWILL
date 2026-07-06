#include"include/moe.h"
#include"include/moek.h"

#include <stdexcept>
#include <iostream>



MoE::MoE(int bl,int sl,int dm,int hidden,int expertsCount){
    this->batchLen = bl;
    this->seqLen = sl;
    this->dModel = dm;
    this->hiddenDim = hidden;

    this->numExperts = expertsCount;
    this->topK = 1;

    this->loadBalanceWeight = 0.01f;
    this->lastLoadBalanceLoss = 0.0f;

    this->router = new Linear(this->batchLen, this->dModel, this->numExperts);

    for(int i = 0; i < expertsCount; i++){
        SWIGLU* expert = new SWIGLU(bl, sl, dm, hidden);
        this->experts.push_back(expert);
    }

    this->prevInput = nullptr;
    this->prevRouterLogits = nullptr;
    this->prevRouterProb = nullptr;
    this->prevExpertChoice = nullptr;

}

//a little bloated fix when actually use this 
MoE::~MoE(){
    if(this->router != nullptr){
        delete this->router;
        this->router = nullptr;
    }

    for(int i = 0; i < this->experts.size(); i++){
        if(this->experts[i] != nullptr){
            delete this->experts[i];
            this->experts[i] = nullptr;
        }
    }

    this->experts.clear();

    if(this->prevRouterLogits != nullptr){
        this->prevRouterLogits->clear();
        delete this->prevRouterLogits;
        this->prevRouterLogits = nullptr;
    }

    if(this->prevRouterProb != nullptr){
        this->prevRouterProb->clear();
        delete this->prevRouterProb;
        this->prevRouterProb = nullptr;
    }

    if(this->prevExpertChoice != nullptr){
        this->prevExpertChoice->clear();
        delete this->prevExpertChoice;
        this->prevExpertChoice = nullptr;
    }

    for(int i = 0; i < this->prevExpertOutputs.size(); i++){
        if(this->prevExpertOutputs[i] != nullptr){
            this->prevExpertOutputs[i]->clear();
            delete this->prevExpertOutputs[i];
            this->prevExpertOutputs[i] = nullptr;
        }
    }

    this->prevExpertOutputs.clear();

    this->prevInput = nullptr;
}

std::vector<Parameter*> MoE::getParameter(){
    std::vector<Parameter*> params;

    std::vector<Parameter*> routerParams = this->router->getParameter();

    for(int i = 0; i < routerParams.size(); i++){
        params.push_back(routerParams[i]);
    }

    for(int i = 0; i < this->experts.size(); i++){
        std::vector<Parameter*> expertParams = this->experts[i]->getParameter();

        for(int j = 0; j < expertParams.size(); j++){
            params.push_back(expertParams[j]);
        }
    }

    return params;
}

float MoE::getLastLoadBalanceLoss(){return this->lastLoadBalanceLoss;}


//////////////////////////////////////

Tensor* MoE::feedForward(Tensor& in_data){
    this->prevInput = &in_data;

    Tensor* routerLogits = this->router->feedForward(in_data);

    Tensor* routerProb = new Tensor({this->batchLen, this->seqLen, this->numExperts});

    Tensor* expertChoice = new Tensor({this->batchLen, this->seqLen});

    routerSoftmaxTop1(*routerLogits,*routerProb,*expertChoice,this->batchLen,this->seqLen,this->numExperts);

    this->prevRouterLogits = routerLogits;
    this->prevRouterProb = routerProb;
    this->prevExpertChoice = expertChoice;

    this->prevExpertOutputs.clear();

    for(int i = 0; i < this->experts.size(); i++){
        Tensor* expertOut = this->experts[i]->feedForward(in_data);
        this->prevExpertOutputs.push_back(expertOut);
    }

    Tensor* out = new Tensor({this->batchLen, this->seqLen, this->dModel});

    moeCombineForward(this->prevExpertOutputs,*this->prevRouterProb,*this->prevExpertChoice,*out,this->batchLen,this->seqLen,this->dModel,this->numExperts);

    return out;
}

Tensor* MoE::backward(Tensor& gradOut){
    std::vector<Tensor*> gradExpertOutputs;

    for(int i = 0; i < this->numExperts; i++){
        Tensor* gradExpert = new Tensor({this->batchLen, this->seqLen, this->dModel});
        gradExpertOutputs.push_back(gradExpert);
    }

    Tensor* gradRouterLogits = new Tensor({this->batchLen, this->seqLen, this->numExperts});

    moeBackwardRouterAndExperts(gradOut,this->prevExpertOutputs,*this->prevRouterProb,*this->prevExpertChoice,gradExpertOutputs,*gradRouterLogits,this->batchLen,this->seqLen,this->dModel,this->numExperts);

    this->lastLoadBalanceLoss = moeLoadBalanceLossAndGrad(*this->prevRouterProb,*this->prevExpertChoice,*gradRouterLogits,this->batchLen,this->seqLen,this->numExperts,this->loadBalanceWeight);


    Tensor* gradIn = this->router->backward(*gradRouterLogits);

    for(int i = 0; i < this->numExperts; i++){
        Tensor* expertGradIn = this->experts[i]->backward(*gradExpertOutputs[i]);

        gradIn->addTensorToTensor(*expertGradIn);

        expertGradIn->clear();
        delete expertGradIn;
    }

    for(int i = 0; i < gradExpertOutputs.size(); i++){
        gradExpertOutputs[i]->clear();
        delete gradExpertOutputs[i];
        gradExpertOutputs[i] = nullptr;
    }


    //clean up
    gradExpertOutputs.clear();

    gradRouterLogits->clear();
    delete gradRouterLogits;

    this->prevRouterLogits->clear();
    delete this->prevRouterLogits;
    this->prevRouterLogits = nullptr;

    this->prevRouterProb->clear();
    delete this->prevRouterProb;
    this->prevRouterProb = nullptr;

    this->prevExpertChoice->clear();
    delete this->prevExpertChoice;
    this->prevExpertChoice = nullptr;

    for(int i = 0; i < this->prevExpertOutputs.size(); i++){
        this->prevExpertOutputs[i]->clear();
        delete this->prevExpertOutputs[i];
        this->prevExpertOutputs[i] = nullptr;
    }

    this->prevExpertOutputs.clear();

    this->prevInput = nullptr;

    return gradIn;
}

