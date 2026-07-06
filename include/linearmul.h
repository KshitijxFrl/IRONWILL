#ifndef LINEARMUL_H
#define LINEARMUL_H

#include "tensor.h"

void linearmul2D(Tensor& A, Tensor& B, Tensor& out, Tensor& bias);
void biasGradient(Tensor& B, Tensor& grad_out);

#endif
