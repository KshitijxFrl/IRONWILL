#ifndef MATMUL2D_H
#define MATMUL2D_H

#include "tensor.h"

void matmul2D(Tensor& A, Tensor& B, Tensor& C);
void matmul2D_AT_B(Tensor& A, Tensor& B, Tensor& C);
void matmul2D_A_BT(Tensor& A, Tensor& B, Tensor& C);

#endif