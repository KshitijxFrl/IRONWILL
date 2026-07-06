#ifndef EMBEDDINGLOOKUP_H 
#define EMBEDDINGLOOKUP_H

#include"../include/tensor.h"

void embeddingLookUp(Tensor& inData, Tensor& emb, Tensor& newX, int d_model);
void embeddingBackward(Tensor& tokenIds,Tensor& gradOut,Tensor& embGrad,int d_model);


#endif