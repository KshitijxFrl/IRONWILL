#ifndef MODULE_H
#define MODULE_H

#include"tensor.h"
#include"parameter.h"


class Module{    
    public:
        virtual ~Module() = default;
        virtual Tensor* feedForward(Tensor& in_data)   = 0;    
        virtual std::vector<Parameter*> getParameter() = 0;
        virtual Tensor* backward(Tensor& gradOut) = 0;
        virtual void clearCache(){}
};




#endif
