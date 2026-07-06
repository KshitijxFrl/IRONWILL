#ifndef PARAMETER_H
#define PARAMETER_H
#include "tensor.h"


class Parameter{
    public:
        Tensor data;
        Tensor gradient;
};


#endif