#include "include/optimizer.h"
#include "include/adamW.h"


#include <cmath>
#include <stdexcept>


Optimizer::Optimizer(
    std::vector<Parameter*> parameters,float lr,float b1,float b2,float eps,float wd){
    this->params = parameters;

    this->learningRate = lr;
    this->beta1 = b1;
    this->beta2 = b2;
    this->epsilon = eps;
    this->weightDecay = wd;

    this->stepCount = 0;

    for(int i = 0; i < this->params.size(); i++){
        Tensor* mTensor = new Tensor();
        Tensor* vTensor = new Tensor();

        std::vector<int> shape;

        auto dataShape = this->params[i]->data.getTensorShape();

        for(int j = 0; j < dataShape.size(); j++){
            shape.push_back(dataShape[j].first);
        }

        mTensor->create(shape);
        vTensor->create(shape);

        mTensor->fill(0.0f);
        vTensor->fill(0.0f);

        this->m.push_back(mTensor);
        this->v.push_back(vTensor);
    }
}



Optimizer::~Optimizer(){
    for(int i = 0; i < this->m.size(); i++){
        if(this->m[i] != nullptr){
            this->m[i]->clear();
            delete this->m[i];
            this->m[i] = nullptr;
        }
    }

    for(int i = 0; i < this->v.size(); i++){
        if(this->v[i] != nullptr){
            this->v[i]->clear();
            delete this->v[i];
            this->v[i] = nullptr;
        }
    }

    this->m.clear();
    this->v.clear();

    this->params.clear();
}

void Optimizer::zeroGrad(){
    for(int i = 0; i < this->params.size(); i++){this->params[i]->gradient.fill(0.0f);}}

void Optimizer::adamW(){
    this->stepCount++;

    for(int i = 0; i < this->params.size(); i++){
        adamWUpdate(this->params[i]->data,this->params[i]->gradient,*this->m[i],*this->v[i],this->learningRate,this->beta1,this->beta2,this->epsilon,this->weightDecay,this->stepCount);
    }
}



void Optimizer::setLearningRate(float lr){this->learningRate = lr;}

float Optimizer::getLearningRate(){return this->learningRate;}