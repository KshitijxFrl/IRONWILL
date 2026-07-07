#include<iostream>
#include<cstdlib>
#include<random>

#include"include/tensor.h"
#include"include/tesnorKernal.h"

// random with seed
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 1.0); 

//||HELPER||
float randV(){

    //current center is 0.5 and new center will be 0
    float val = dis(gen);

    //distance from center
    float centerDis = val - 0.5;
    //scaling down by chenking how much smaller the small val is when compare to bigger val
    float shrinkVal = 0.02 / 0.5;

    float randVal = centerDis * shrinkVal;
    
    return randVal;
}

// Adavance feature paused until it make sense
// void Tensor::to_gpu(){
//     //doing nothing for now
// }

// void Tensor::to_cpu(){
//     //doing nothing for now
// }

int Tensor::computeOffset(std::vector<int> id){
    int target = 0;
    
    //std::cout<<this->shape.size()<<" "<<id.size()<<std::endl;
    if(this->shape.size() != id.size()){
        std::cerr<<"Bad Tensor : (Input Miss Match)."<<std::endl;
        return -1;
    } 
    
    
    for(int i = 0; i < this->shape.size(); i++){
        //std::cout<<id[i]<<" "<<this->shape[i].first<<std::endl;
        if(id[i] >= 0 && id[i] < this->shape[i].first) target += id[i] * this->shape[i].second;
        else{
            std::cerr<<"Bad Tensor : (Input Miss Match)."<<std::endl;
            return -1;
        }        
        
    }

    return target;
}

void Tensor::uploadData(std::vector<float>& data){
    cudaError_t err = cudaMemcpy(this->gpu_data, data.data(), this->size * sizeof(float), cudaMemcpyHostToDevice);

    if(err != cudaSuccess){
        std::cerr << "Tensor upload failed: " << cudaGetErrorString(err) << std::endl;
    }
}

std::vector<float> Tensor::downloadata(){
    std::vector<float> temp(this->size);

    cudaError_t err = cudaMemcpy(temp.data(), this->gpu_data, this->size * sizeof(float), cudaMemcpyDeviceToHost);

    if(err != cudaSuccess){
        std::cerr << "Tensor download failed: " << cudaGetErrorString(err) << std::endl;
    }

    return temp;
}

bool Tensor::downloadDataChecked(std::vector<float>& out){
    out.resize(this->size);

    cudaError_t err = cudaMemcpy(out.data(), this->gpu_data, this->size * sizeof(float), cudaMemcpyDeviceToHost);

    if(err != cudaSuccess){
        std::cerr << "Tensor checked download failed: " << cudaGetErrorString(err) << std::endl;
        out.clear();
        return false;
    }

    return true;
}

Tensor::Tensor(){
    this->gpu_data = nullptr;
    this->size = 0;
}


Tensor::Tensor(std::vector<int> newShape){  
    this->shape.resize(newShape.size());

    this->shape.back().first  = newShape.back(); 
    this->shape.back().second = 1;
    
    for(int i =  newShape.size() - 2; i >= 0  ; i--){
        this->shape[i].first  = newShape[i]; 
        this->shape[i].second = this->shape[i + 1].first * this->shape[i + 1].second;    
    }

    //update the size and number of elements;

    this->size = (this->shape.front().first * this->shape.front().second);

    cudaMalloc(&this->gpu_data, this->size * sizeof(float));
}

Tensor::~Tensor(){
    if(this->gpu_data) cudaFree(this->gpu_data);
}

void Tensor::create(std::vector<int> newShape){
    //get the shape done 
    this->shape.resize(newShape.size());

    this->shape.back().first  = newShape.back(); 
    this->shape.back().second = 1;
    
    for(int i =  newShape.size() - 2; i >= 0  ; i--){
        this->shape[i].first  = newShape[i]; 
        this->shape[i].second = this->shape[i + 1].first * this->shape[i + 1].second;    
    }

    //update the size and number of elements;

    this->size = (this->shape.front().first * this->shape.front().second);

    cudaMalloc(&this->gpu_data, this->size * sizeof(float));
}

void Tensor::clear(){
    if(this->gpu_data != nullptr){
        cudaFree(this->gpu_data);
        this->gpu_data = nullptr;
    }

    this->shape.clear();
    this->size = 0;
}


void Tensor::fill(float val){
    std::vector<float> temp(this->size, val);
    cudaMemcpy(this->gpu_data, temp.data(), this->size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::random(){
    std::vector<float> temp(this->size);

    for(int i = 0; i < this->size; i++){
        temp[i] = randV(); 
    }
    cudaMemcpy(this->gpu_data, temp.data(), this->size * sizeof(float), cudaMemcpyHostToDevice);
}


int Tensor::getTensorSize(){return this->size;}

void Tensor::reshapeTensor(std::vector<int> newShape){
    
    //clear the old specs
    this->shape.clear();
    
    //get the shape done 
    this->shape.resize(newShape.size());

    this->shape.back().first  = newShape.back(); 
    this->shape.back().second = 1;
    
    for(int i =  newShape.size() - 2; i >= 0  ; i--){
        this->shape[i].first  = newShape[i]; 
        this->shape[i].second = this->shape[i + 1].first * this->shape[i + 1].second;    
    }

    //update the size and number of elements;

    this->size = (this->shape.front().first * this->shape.front().second);


}


bool Tensor::isSameShape(Tensor& t){
    auto curr = this->shape;
    auto temp = t.shape;

    if(curr.size() != temp.size()) return false;

    for(int i = 0; i < curr.size(); i++){
        if(curr[i].first != temp[i].first) return false;
    }

    return true;

}

std::vector<std::pair<int,int>>& Tensor::getTensorShape(){return this->shape;}


void Tensor::setData(std::vector<int> id, float val){

    int idx = this->computeOffset(id);

    cudaMemcpy(this->gpu_data + idx, &val, sizeof(float), cudaMemcpyHostToDevice);    
}

float Tensor::getData(std::vector<int> id){
    int idx = this->computeOffset(id);
    float val;
    cudaMemcpy(&val, this->gpu_data + idx, sizeof(float), cudaMemcpyDeviceToHost);    

    return val;
}

float* Tensor::getTenorGpuData(){
    return this->gpu_data;
}

void Tensor::addTensorToTensor(Tensor& B){
    if(this->size != B.getTensorSize()){
        std::cerr<<"Problem adding Tensor to Tesnor: (Different Size)";
        return;
    }
    
    auto bShape = B.getTensorShape();
    
    if(this->shape.size() == bShape.size()){
        int x = this->shape.size();

        for(int i = 0; i < x; i++){
            if(this->shape[i].first != bShape[i].first || this->shape[i].second != bShape[i].second){
                std::cerr<<"Problem adding Tensor to Tesnor: (Different Shape)";
                return;
            }
        }

    }else{
        std::cerr<<"Problem adding Tensor to Tesnor: (Different Shape)";
        return;
    }    

    addTensor(this->gpu_data, B);
}

void Tensor::copyTensor(Tensor& B){
    if(this->size != B.getTensorSize()){
        std::cerr << "Problem copying Tensor to Tensor: (Different Size)" << std::endl;
        return;
    }

    auto bShape = B.getTensorShape();

    if(this->shape.size() == bShape.size()){
        int x = this->shape.size();

        for(int i = 0; i < x; i++){
            if(this->shape[i].first != bShape[i].first || this->shape[i].second != bShape[i].second){
                std::cerr << "Problem copying Tensor to Tensor: (Different Shape)" << std::endl;
                return;
            }
        }
    }else{
        std::cerr << "Problem copying Tensor to Tensor: (Different Shape)" << std::endl;
        return;
    }

    copyTensorRaw(this->gpu_data, B.getTenorGpuData(), this->size);
}
