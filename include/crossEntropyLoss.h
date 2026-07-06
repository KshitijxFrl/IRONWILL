#ifndef CROSSENTROPYLOSS_H
#define CROSSENTROPYLOSS_H

#include"tensor.h"

float crossEntropyLoss(Tensor& logits, Tensor& target, Tensor& gradLogits);

#endif