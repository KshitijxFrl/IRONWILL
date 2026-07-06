#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "module.h"
#include "parameter.h"

class Embedding : public Module {
    private:
        int sequenceLength;
        int d_model;
        int vocabLength;
        int batchLength;

        Parameter emedding_lookup;

        Tensor* prevInput;


    public:
        Embedding(int batch_len ,int seq_len, int d_model, int vocab_len);
        Tensor* feedForward(Tensor& in_data) override;
        virtual std::vector<Parameter*> getParameter() override;
        Tensor* backward(Tensor& gradOut) override;

};


#endif