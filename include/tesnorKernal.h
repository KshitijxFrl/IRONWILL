#ifndef TENSORKERNAL_H
#define TENSORKERNAL_H

#include"tensor.h"

void addTensor(float* A, Tensor& B);
void copyTensorRaw(float* A, float* B, int N);

#endif
