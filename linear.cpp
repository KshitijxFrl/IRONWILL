#include"include/linear.h"
#include"include/matmul2d.h"
#include"include/linearmul.h"

#include<iostream>

//  output_features are the the neuron per layer and every neuron take every features(column) of x/token multiply by corresponding weight add them togather and its bias the answer which result in a single new feature for the same token/column of x 
//  increse/decrease  output_features or neurons will result in increse/decrease of features. of x/token
//  Linear does not create or hold neurons but it create a sigular marix of all the neuron input weights 
//  number of features (column of x) decides how many weights (rows per column in weight matix) per neuron have as every feature requied a weight
//  every token/row of x use same input weights of a neuron
//  matmul make evey thing work work all at once

Linear :: Linear(int batch_len, int input_features, int output_features){
    
    this->batchLenght    = batch_len;
    this->inputFeatures  = input_features;
    this->outputFeatures = output_features;    

    this->weights.data.create({input_features, output_features});
    this->weights.gradient.create({input_features, output_features});

    this->weights.data.random();
    this->weights.gradient.fill(0.0f);
    
    this->bias.data.create({1, output_features});
    this->bias.gradient.create({1, output_features});

    this->bias.data.random();
    this->bias.gradient.fill(0.0f);

}

std::vector<Parameter*> Linear::getParameter(){
    return {&this->weights, &this->bias};
}

Tensor* Linear::feedForward(Tensor& in_data){
    this->prevInput = &in_data;

    //can use input features here
    Tensor* X_out = new Tensor({this->batchLenght, in_data.getTensorShape()[1].first, this->outputFeatures});

    linearmul2D(in_data, this->weights.data, *X_out, this->bias.data);

    return X_out;
}


Tensor* Linear::backward(Tensor& gradOut){
    matmul2D_AT_B(*this->prevInput, gradOut, this->weights.gradient);
    biasGradient(this->bias.gradient, gradOut);

    Tensor* gradIn = new Tensor();
    std::vector<int> giShape;
    auto prevInputShape = this->prevInput->getTensorShape();

    for(int i = 0 ; i < prevInputShape.size(); i++){
        giShape.push_back(prevInputShape[i].first);
    }

    gradIn->create(giShape);
    matmul2D_A_BT(gradOut, this->weights.data, *gradIn);

    this->prevInput = nullptr;

    return gradIn;

}
