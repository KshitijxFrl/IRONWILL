#include"include/SwiGLU.h"
#include"include/silu.h"
#include"include/matmul2d.h"

SWIGLU::SWIGLU(int bl, int sl, int cd, int hid){
    this->batchLen = bl;
    this->seqLen = sl;
    this->currentDimention = cd;
    this->hiddenDimention = hid;

    this->Wgate_in.data.create({this->currentDimention, this->hiddenDimention});
    this->Wgate_in.gradient.create({this->currentDimention, this->hiddenDimention});

    this->Wgate_in.data.random();
    this->Wgate_in.gradient.fill(0.0f);


    this->wUP.data.create({this->currentDimention, this->hiddenDimention});
    this->wUP.gradient.create({this->currentDimention, this->hiddenDimention});    

    this->wUP.data.random();
    this->wUP.gradient.fill(0.0f);

    this->wDown.data.create({this->hiddenDimention, this->currentDimention});
    this->wDown.gradient.create({this->hiddenDimention, this->currentDimention});

    this->wDown.data.random();
    this->wDown.gradient.fill(0.0f);

}

std::vector<Parameter*> SWIGLU::getParameter(){
    return {&this->Wgate_in, &this->wUP, &this->wDown};
}


Tensor* SWIGLU::feedForward(Tensor& in_data){
    
    auto inShape = in_data.getTensorShape();
    
    this->prevInput = &in_data;

    //this is basically X_out
    Tensor* prjDown = new Tensor({this->batchLen, this->seqLen, this->currentDimention});

    Tensor* prjGateIn = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});
    Tensor* prjGateOut = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});

    Tensor* prjUp = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});

    this->prev_prjGateIn = prjGateIn;
    this->prev_prjUp = prjUp;
    
    matmul2D(in_data, this->Wgate_in.data, *prjGateIn);
    matmul2D(in_data, this->wUP.data, *prjUp);

    silu(*prjGateIn, *prjUp, *prjGateOut);

    this->prev_prjGateOut = prjGateOut;

    matmul2D(*prjGateOut, this->wDown.data, *prjDown);


    return prjDown;
    

}


Tensor* SWIGLU::backward(Tensor& gradOut){
    Tensor* gradHidden = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});
    Tensor* siluGate = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});

    Tensor* gradUp = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});
    Tensor* gradGate = new Tensor({this->batchLen, this->seqLen, this->hiddenDimention});

    Tensor* gradInGate = new Tensor({this->batchLen, this->seqLen, this->currentDimention});
    Tensor* gradInUp = new Tensor({this->batchLen, this->seqLen, this->currentDimention});
    

    matmul2D_AT_B(*this->prev_prjGateOut, gradOut, this->wDown.gradient);
    matmul2D_A_BT(gradOut, this->wDown.data, *gradHidden);

    onlySilu(*this->prev_prjGateIn, *siluGate);

    swigluLocalBackward(*gradHidden,*this->prev_prjGateIn,*this->prev_prjUp,*siluGate,*gradUp,*gradGate);

    matmul2D_AT_B(*this->prevInput, *gradGate, this->Wgate_in.gradient);
    matmul2D_AT_B(*this->prevInput, *gradUp, this->wUP.gradient);    

    matmul2D_A_BT(*gradGate, this->Wgate_in.data, *gradInGate);
    matmul2D_A_BT(*gradUp, this->wUP.data, *gradInUp);

    // final gradIn = gradInGate + gradInUp. Insted of a new tesnor using gradInGate for gradIn
    gradInGate->addTensorToTensor(*gradInUp);

    gradHidden->clear();
    siluGate->clear();
    gradUp->clear();
    gradGate->clear();
    gradInUp->clear();


    delete gradHidden;
    delete siluGate;
    delete gradUp;
    delete gradGate;
    delete gradInUp;

    this->prev_prjGateIn->clear();
    delete this->prev_prjGateIn;
    this->prev_prjGateIn = nullptr;

    this->prev_prjGateOut->clear();
    delete this->prev_prjGateOut;
    this->prev_prjGateOut = nullptr;

    this->prev_prjUp->clear();
    delete this->prev_prjUp;
    this->prev_prjUp = nullptr;

    
    this->prevInput = nullptr;

    return gradInGate;
    
}