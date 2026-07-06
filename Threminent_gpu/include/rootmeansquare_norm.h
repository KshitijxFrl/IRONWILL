#ifndef ROOTMEANSQUARENORM_H
#define ROOTMEANSQUARENORM_H

#include"tensor.h"

void rootMeanSquareNorm(Tensor& A, Tensor& newA, Tensor& gamma);
void gammaGrad(Tensor& gammaGradient, Tensor& gradOut, Tensor& norm);
void rmsOfToken(Tensor& A, Tensor& rms_of_token);
void rmsNormOnly(Tensor& A, Tensor& rms_of_token, Tensor& norm);
void rmsBackward(Tensor& gradOut, Tensor& gamma, Tensor& input, Tensor& rms_of_token, Tensor& gradIn);

#endif