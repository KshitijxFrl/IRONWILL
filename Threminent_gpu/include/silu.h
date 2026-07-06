#ifndef SILU_H
#define SILU_H

#include"tensor.h"

void silu(Tensor& Gatein, Tensor& Up, Tensor& Gateout);
void onlySilu(Tensor& Gatein, Tensor& Gateout);
void swigluLocalBackward(Tensor& gradHidden,Tensor& gate,Tensor& up,Tensor& siluGate,Tensor& gradUp,Tensor& gradGate);

#endif