#include"include/rmsNorm.h"
#include"include/rootmeansquare_norm.h"

RMSNormal::RMSNormal(int featureLen){
    this->gamma.data.create({featureLen});
    this->gamma.data.fill(1.0f);
    
    this->gamma.gradient.create({featureLen});
    this->gamma.gradient.fill(0.0f);

}

std::vector<Parameter*> RMSNormal::getParameter(){
    return {&this->gamma};
}

Tensor* RMSNormal::feedForward(Tensor& in_data){
    this->prevInput = &in_data;

    Tensor* updated_InData = new Tensor();
    std::vector<int> uidShape;

    auto prevInputShape = this->prevInput->getTensorShape();

    

    for(int i = 0 ; i < prevInputShape.size(); i++){
        uidShape.push_back(prevInputShape[i].first);
    }

    updated_InData->create(uidShape);
    this->rms_of_token = new Tensor({prevInputShape[0].first, prevInputShape[1].first});

    rmsOfToken(in_data, *this->rms_of_token);
    rootMeanSquareNorm(in_data, *updated_InData, this->gamma.data);
    
    return updated_InData;
}


Tensor* RMSNormal::backward(Tensor& gradOut){
    Tensor* norm = new Tensor();
    Tensor* gradIn = new Tensor();

    std::vector<int> nShape;

    auto prevInputShape = this->prevInput->getTensorShape();

    for(int i = 0; i < prevInputShape.size(); i++){
        nShape.push_back(prevInputShape[i].first);
    }

    norm->create(nShape);
    gradIn->create(nShape);

    rmsNormOnly(*this->prevInput, *this->rms_of_token, *norm);

    gammaGrad(this->gamma.gradient, gradOut, *norm);

    rmsBackward(gradOut,this->gamma.data,*this->prevInput,*this->rms_of_token,*gradIn);

    //clearing of memory section (this is almost same int every "backward" careful and dont delete this->prevInputs)
    norm->clear();
    delete norm;

    this->rms_of_token->clear();
    delete this->rms_of_token;
    this->rms_of_token = nullptr;

    this->prevInput = nullptr;

    return gradIn;
}