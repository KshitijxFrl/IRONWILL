#ifndef RESIDUALADD_H
#define RESIDUALADD_H

#include"module.h"

class ResidualAdd : public Module{
    Tensor* prevInput;
    Module* actionLayer;


    public:
        ResidualAdd(Module& AL);
        Tensor* feedForward(Tensor& in_data) override;
        virtual std::vector<Parameter*> getParameter() override;
        Tensor* backward(Tensor& gradOut) override;
        void clearCache() override;

};

#endif
