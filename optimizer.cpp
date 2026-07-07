#include "include/optimizer.h"
#include "include/adamW.h"


#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>


Optimizer::Optimizer(
    std::vector<Parameter*> parameters,float lr,float b1,float b2,float eps,float wd){
    this->params = parameters;

    this->learningRate = lr;
    this->beta1 = b1;
    this->beta2 = b2;
    this->epsilon = eps;
    this->weightDecay = wd;

    this->stepCount = 0;

    for(int i = 0; i < this->params.size(); i++){
        Tensor* mTensor = new Tensor();
        Tensor* vTensor = new Tensor();

        std::vector<int> shape;

        auto dataShape = this->params[i]->data.getTensorShape();

        for(int j = 0; j < dataShape.size(); j++){
            shape.push_back(dataShape[j].first);
        }

        mTensor->create(shape);
        vTensor->create(shape);

        mTensor->fill(0.0f);
        vTensor->fill(0.0f);

        this->m.push_back(mTensor);
        this->v.push_back(vTensor);
    }
}



Optimizer::~Optimizer(){
    for(int i = 0; i < this->m.size(); i++){
        if(this->m[i] != nullptr){
            this->m[i]->clear();
            delete this->m[i];
            this->m[i] = nullptr;
        }
    }

    for(int i = 0; i < this->v.size(); i++){
        if(this->v[i] != nullptr){
            this->v[i]->clear();
            delete this->v[i];
            this->v[i] = nullptr;
        }
    }

    this->m.clear();
    this->v.clear();

    this->params.clear();
}

void Optimizer::zeroGrad(){
    for(int i = 0; i < this->params.size(); i++){this->params[i]->gradient.fill(0.0f);}}

void Optimizer::adamW(){
    this->stepCount++;

    for(int i = 0; i < this->params.size(); i++){
        adamWUpdate(this->params[i]->data,this->params[i]->gradient,*this->m[i],*this->v[i],this->learningRate,this->beta1,this->beta2,this->epsilon,this->weightDecay,this->stepCount);
    }
}

bool Optimizer::saveState(std::string fileName){
    cudaError_t syncErr = cudaDeviceSynchronize();

    if(syncErr != cudaSuccess){
        std::cerr << "Skipping optimizer state save because CUDA is unhealthy: "
                  << cudaGetErrorString(syncErr) << std::endl;
        return false;
    }

    std::string tempFileName = fileName + ".tmp";
    std::ofstream out(tempFileName, std::ios::binary);

    if(!out.is_open()){
        std::cerr << "Failed to open optimizer state file: " << tempFileName << std::endl;
        return false;
    }

    out.write(reinterpret_cast<char*>(&this->stepCount), sizeof(int));

    if(!out.good()){
        std::cerr << "Failed to write optimizer step count: " << tempFileName << std::endl;
        out.close();
        std::remove(tempFileName.c_str());
        return false;
    }

    for(int i = 0; i < this->m.size(); i++){
        std::vector<float> temp;

        if(!this->m[i]->downloadDataChecked(temp)){
            std::cerr << "Aborting optimizer state save because m tensor download failed: "
                      << tempFileName << std::endl;
            out.close();
            std::remove(tempFileName.c_str());
            return false;
        }

        out.write(reinterpret_cast<char*>(temp.data()), temp.size() * sizeof(float));

        if(!out.good()){
            std::cerr << "Failed to write optimizer m state: " << tempFileName << std::endl;
            out.close();
            std::remove(tempFileName.c_str());
            return false;
        }
    }

    for(int i = 0; i < this->v.size(); i++){
        std::vector<float> temp;

        if(!this->v[i]->downloadDataChecked(temp)){
            std::cerr << "Aborting optimizer state save because v tensor download failed: "
                      << tempFileName << std::endl;
            out.close();
            std::remove(tempFileName.c_str());
            return false;
        }

        out.write(reinterpret_cast<char*>(temp.data()), temp.size() * sizeof(float));

        if(!out.good()){
            std::cerr << "Failed to write optimizer v state: " << tempFileName << std::endl;
            out.close();
            std::remove(tempFileName.c_str());
            return false;
        }
    }

    out.close();

    if(!out.good()){
        std::cerr << "Failed to finish optimizer state file: " << tempFileName << std::endl;
        std::remove(tempFileName.c_str());
        return false;
    }

    if(std::rename(tempFileName.c_str(), fileName.c_str()) != 0){
        std::cerr << "Failed to replace optimizer state file: " << fileName << std::endl;
        std::remove(tempFileName.c_str());
        return false;
    }

    return true;
}

bool Optimizer::loadState(std::string fileName){
    std::ifstream in(fileName, std::ios::binary);

    if(!in.is_open()){
        std::cerr << "Failed to open optimizer state file: " << fileName << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(&this->stepCount), sizeof(int));

    if(!in.good()){
        std::cerr << "Failed to read optimizer step count: " << fileName << std::endl;
        in.close();
        return false;
    }

    for(int i = 0; i < this->m.size(); i++){
        std::vector<float> temp(this->m[i]->getTensorSize());
        in.read(reinterpret_cast<char*>(temp.data()), temp.size() * sizeof(float));

        if(!in.good()){
            std::cerr << "Failed to read optimizer m state: " << fileName << std::endl;
            in.close();
            return false;
        }

        this->m[i]->uploadData(temp);
    }

    for(int i = 0; i < this->v.size(); i++){
        std::vector<float> temp(this->v[i]->getTensorSize());
        in.read(reinterpret_cast<char*>(temp.data()), temp.size() * sizeof(float));

        if(!in.good()){
            std::cerr << "Failed to read optimizer v state: " << fileName << std::endl;
            in.close();
            return false;
        }

        this->v[i]->uploadData(temp);
    }

    in.close();
    return true;
}



void Optimizer::setLearningRate(float lr){this->learningRate = lr;}

float Optimizer::getLearningRate(){return this->learningRate;}
