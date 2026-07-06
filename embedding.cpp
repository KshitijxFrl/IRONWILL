#include"include/embedding.h"
#include"include/module.h"
#include"include/embedding_lookup.h"
#include<iostream>

Embedding::Embedding(int batch_len, int seq_len, int d_model, int vocab_len){
    this->sequenceLength = seq_len;
    this->d_model        = d_model;
    this->vocabLength    = vocab_len;
    this->batchLength    = batch_len;


    this->emedding_lookup.data.create({vocab_len, d_model});
    this->emedding_lookup.data.random();

    this->emedding_lookup.gradient.create({vocab_len, d_model});
    this->emedding_lookup.gradient.fill(0.0f);    
    

}

std::vector<Parameter*>  Embedding::getParameter(){
    return {&this->emedding_lookup};
}


//in_data size = (batchsize, sequenceLength)
Tensor* Embedding::feedForward(Tensor& in_data){
    this->prevInput = &in_data;

    Tensor* X = new Tensor({this->batchLength, this->sequenceLength, this->d_model});

    embeddingLookUp(in_data, this->emedding_lookup.data, *X, this->d_model);

    return X;
}


Tensor* Embedding::backward(Tensor& gradOut){
    embeddingBackward(*this->prevInput,gradOut,this->emedding_lookup.gradient,this->d_model);

    this->prevInput = nullptr;

    return nullptr;
}