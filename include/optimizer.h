#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <string>
#include <vector>

#include "tensor.h"
#include "parameter.h"

class Optimizer{
private:
    std::vector<Parameter*> params;

    std::vector<Tensor*> m;
    std::vector<Tensor*> v;

    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;

    int stepCount;

public:
    Optimizer(std::vector<Parameter*> parameters,float lr,float b1 = 0.9f,float b2 = 0.999f,float eps = 1e-8f,float wd = 0.01f);

    ~Optimizer();

    void zeroGrad();

    void adamW();

    void saveState(std::string fileName);

    bool loadState(std::string fileName);

    void setLearningRate(float lr);

    float getLearningRate();
};

#endif
