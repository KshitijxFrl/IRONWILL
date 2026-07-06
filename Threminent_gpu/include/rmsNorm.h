#ifndef RMSNORM_H
#define RMSNORM_H

#include"module.h"

class RMSNormal: public Module{
    Parameter gamma; 

    Tensor* prevInput;
    Tensor* rms_of_token;
    
    public:
    RMSNormal(int featureLen);

    Tensor* feedForward(Tensor& in_data) override;
    virtual std::vector<Parameter*> getParameter() override;
    Tensor* backward(Tensor& gradOut) override;

};


#endif