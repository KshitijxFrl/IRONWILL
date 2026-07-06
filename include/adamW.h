#ifndef ADAMW_H
#define ADAMW_H

#include"tensor.h"

void adamWUpdate(Tensor& data,Tensor& gradient,Tensor& m,Tensor& v,float learningRate,float beta1,float beta2,float epsilon,float weightDecay,int stepCount);

#endif