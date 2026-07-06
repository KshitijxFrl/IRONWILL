#include"include/attention.h"
#include"include/matmul2d.h"
#include"include/rope.h"
#include"include/gqa.h"


// The way this perticular attention should work is that inData tensor should have 512 features per token embedded vector
AttentionGQA::AttentionGQA(int bl, int sl, int kvh, int h, int hd){
    this->batchLen = bl;
    this->seqLen = sl;
    this->kv_head = kvh;
    this->head = h;
    this->head_dim = hd;

    this->currFeatures = this->head * this->head_dim;
    this->currKVFeatures = this->kv_head * this->head_dim;

    this->wQ.data.create({this->currFeatures, this->currFeatures});
    this->wQ.gradient.create({this->currFeatures, this->currFeatures});

    this->wK.data.create({this->currFeatures, this->currKVFeatures});
    this->wK.gradient.create({this->currFeatures, this->currKVFeatures});

    this->wV.data.create({this->currFeatures, this->currKVFeatures});
    this->wV.gradient.create({this->currFeatures, this->currKVFeatures});

    this->wO.data.create({this->currFeatures, this->currFeatures});
    this->wO.gradient.create({this->currFeatures, this->currFeatures});

    this->wQ.data.random();
    this->wK.data.random();
    this->wV.data.random();
    this->wO.data.random();

    this->wQ.gradient.fill(0.0f);
    this->wK.gradient.fill(0.0f);
    this->wV.gradient.fill(0.0f);
    this->wO.gradient.fill(0.0f);

    this->prevData = nullptr;
    this->prevQ = nullptr;
    this->prevK = nullptr;
    this->prevV = nullptr;
    this->prevAttentionScores = nullptr;
    this->prevAttentionOut = nullptr;
}

std::vector<Parameter*> AttentionGQA::getParameter(){
    return {&this->wQ, &this->wK, &this->wV, &this->wO};
}

//in data has shape of (batch, seq, d_model)
Tensor* AttentionGQA::feedForward(Tensor& in_data){

    this->prevData = &in_data;    
    this->prevAttentionScores = new Tensor({this->batchLen, this->seqLen, this->head, this->seqLen});

    Tensor* Q = new Tensor({this->batchLen, this->seqLen, this->currFeatures});
    Tensor* K = new Tensor({this->batchLen, this->seqLen, this->currKVFeatures});
    Tensor* V = new Tensor({this->batchLen, this->seqLen, this->currKVFeatures});

    // X * wQ
    matmul2D(in_data, this->wQ.data, *Q);
    // X * wK
    matmul2D(in_data, this->wK.data, *K);
    // X * wV
    matmul2D(in_data, this->wV.data, *V);


    //Applying rope
    simpleRoPE(*Q, this->seqLen, this->head, this->head_dim);
    simpleRoPE(*K, this->seqLen, this->kv_head, this->head_dim);

    //resahpe (so we dont need to like actuall do the reshape as the kernal find the offset inside the flat memory)

    Tensor* attentionOut = new Tensor({batchLen, seqLen, currFeatures});
    
    
    attention_GQA(*Q,*K,*V,*attentionOut,*this->prevAttentionScores,batchLen,seqLen,head,kv_head,head_dim);

    Tensor* out_data = new Tensor({batchLen, seqLen, currFeatures});

    matmul2D(*attentionOut, this->wO.data, *out_data);

    this->prevQ = Q;
    this->prevK = K;
    this->prevV = V;
    this->prevAttentionOut = attentionOut;

    return out_data;
}

Tensor* AttentionGQA::backward(Tensor& gradOut){

    Tensor* gradAttentionOut = new Tensor({this->batchLen, this->seqLen, this->currFeatures});

    matmul2D_AT_B(*this->prevAttentionOut, gradOut, this->wO.gradient);
    matmul2D_A_BT(gradOut, this->wO.data, *gradAttentionOut);


    Tensor* gradQ = new Tensor({this->batchLen, this->seqLen, this->currFeatures});
    Tensor* gradK = new Tensor({this->batchLen, this->seqLen, this->currKVFeatures});
    Tensor* gradV = new Tensor({this->batchLen, this->seqLen, this->currKVFeatures});


    attention_GQA_Backward(*gradAttentionOut,*this->prevQ,*this->prevK,*this->prevV,*this->prevAttentionScores,*gradQ,*gradK,*gradV,this->batchLen,this->seqLen,this->head,this->kv_head,this->head_dim);


    simpleRoPEBackward(*gradQ, this->seqLen, this->head, this->head_dim);
    simpleRoPEBackward(*gradK, this->seqLen, this->kv_head, this->head_dim);

    matmul2D_AT_B(*this->prevData, *gradQ, this->wQ.gradient);
    matmul2D_AT_B(*this->prevData, *gradK, this->wK.gradient);
    matmul2D_AT_B(*this->prevData, *gradV, this->wV.gradient);


    Tensor* gradInQ = new Tensor({this->batchLen, this->seqLen, this->currFeatures});
    Tensor* gradInK = new Tensor({this->batchLen, this->seqLen, this->currFeatures});
    Tensor* gradInV = new Tensor({this->batchLen, this->seqLen, this->currFeatures});

    matmul2D_A_BT(*gradQ, this->wQ.data, *gradInQ);
    matmul2D_A_BT(*gradK, this->wK.data, *gradInK);
    matmul2D_A_BT(*gradV, this->wV.data, *gradInV);


    gradInQ->addTensorToTensor(*gradInK);
    gradInQ->addTensorToTensor(*gradInV);

    //Cleaning up 
    gradAttentionOut->clear();
    gradQ->clear();
    gradK->clear();
    gradV->clear();
    gradInK->clear();
    gradInV->clear();



    delete gradAttentionOut;
    delete gradQ;
    delete gradK;
    delete gradV;
    delete gradInK;
    delete gradInV;

    this->prevQ->clear();
    delete this->prevQ;
    this->prevQ = nullptr;

    this->prevK->clear();
    delete this->prevK;
    this->prevK = nullptr;

    this->prevV->clear();
    delete this->prevV;
    this->prevV = nullptr;

    this->prevAttentionOut->clear();
    delete this->prevAttentionOut;
    this->prevAttentionOut = nullptr;

    this->prevAttentionScores->clear();
    delete this->prevAttentionScores;
    this->prevAttentionScores = nullptr;


    this->prevData = nullptr;

    return gradInQ;
}