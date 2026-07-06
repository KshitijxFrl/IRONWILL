#ifndef ATTENTION_H
#define ATTENTION_H

#include"module.h"

class AttentionGQA : public Module{
    private:
        
        Parameter wQ;
        Parameter wK;
        Parameter wV;

        Parameter wO;

        Tensor* prevData;
        Tensor* prevQ;
        Tensor* prevK;
        Tensor* prevV;
        Tensor* prevAttentionScores;
        Tensor* prevAttentionOut;



        int batchLen;
        int seqLen;
        int head;
        int head_dim;
        int kv_head;

        int currFeatures;
        int currKVFeatures;

    public:

        AttentionGQA(int bl, int sl, int kvh, int h, int hd);
        Tensor* feedForward(Tensor& in_data) override;
        virtual std::vector<Parameter*> getParameter() override;
        Tensor* backward(Tensor& gradOut) override;


};


#endif