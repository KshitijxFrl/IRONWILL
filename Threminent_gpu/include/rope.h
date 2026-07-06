#ifndef ROPE_H
#define ROPE_H

#include"module.h"

// class RoPE : public Module{
//     int head_dim;

//     public:

//     RoPE(int hd);

//     Tensor* feedForward(Tensor& in_data) override;
//     virtual std::vector<Parameter*> getParameter() override;


// };


void simpleRoPE(Tensor& in_data, int seq_dimention,int head, int head_dimention);
void simpleRoPEBackward(Tensor& grad_data, int seq_dimention, int head, int head_dimention);

#endif
