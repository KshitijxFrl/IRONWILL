#include"include/residualADD.h"

ResidualAdd::ResidualAdd(Module& AL){
    this->actionLayer = &AL;
}

std::vector<Parameter*> ResidualAdd::getParameter(){
    return this->actionLayer->getParameter();
}

//apply to those layer which dont change the shape and size of the out they generate
Tensor* ResidualAdd::feedForward(Tensor& in_data){
    Tensor* temp = this->actionLayer->feedForward(in_data);

    Tensor* out = new Tensor();

    std::vector<int> outShape;
    auto inShape = in_data.getTensorShape();

    for(int i = 0; i < inShape.size(); i++){
        outShape.push_back(inShape[i].first);
    }

    out->create(outShape);

    in_data.copyTensor(*out);

    out->addTensorToTensor(*temp);

    temp->clear();
    delete temp;

    return out;
}

Tensor* ResidualAdd::backward(Tensor& gradOut){
    Tensor* actionGrad = this->actionLayer->backward(gradOut);

    actionGrad->addTensorToTensor(gradOut);

    return actionGrad;
}

void ResidualAdd::clearCache(){
    if(this->actionLayer != nullptr){
        this->actionLayer->clearCache();
    }

    this->prevInput = nullptr;
}
