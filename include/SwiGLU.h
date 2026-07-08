#ifndef SWIGLU_H
#define SWIGLU_H

#include"module.h"

class SWIGLU:public Module{
    private:    
        Parameter wUP;
        Parameter wDown;
        Parameter Wgate_in;

        Tensor* prevInput;
        Tensor* prev_prjGateIn;
        Tensor* prev_prjGateOut;
        Tensor* prev_prjUp;

        int batchLen;
        int currentDimention;
        int seqLen;
        int hiddenDimention;


    public:
        SWIGLU(int bl, int sl, int cd, int hid);

        Tensor* feedForward(Tensor& in_data) override;
        virtual std::vector<Parameter*> getParameter() override;
        Tensor* backward(Tensor& gradOut) override;
        void clearCache() override;


};



#endif
