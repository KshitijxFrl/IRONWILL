#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"
#include "parameter.h"

class Linear : public Module{
    public:
    Parameter weights;
    Parameter bias;

    Tensor* prevInput;

    int batchLenght;
    int inputFeatures;
    int outputFeatures;

    Linear(int batch_len, int in_size, int out_size);

    Tensor* feedForward(Tensor& in_data) override;
    virtual std::vector<Parameter*> getParameter() override;
    Tensor* backward(Tensor& gradOut) override;

};



#endif