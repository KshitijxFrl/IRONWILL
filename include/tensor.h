#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

class Tensor{
    private:
        float *gpu_data = nullptr;

        int size = 0;
        std::vector<std::pair<int,int>> shape;

        int computeOffset(std::vector<int> id);

    public:
        Tensor();
        Tensor(std::vector<int> newShape);
        ~Tensor();

        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;        

        //float* getTensorData();    
        
        //Advance systems for sending data to cpu or gpu 
        //void   to_gpu();
        //void   to_cpu();
        
        void   create(std::vector<int> newShape);
        void   fill(float val);
        void   random(); 
        void   clear(); 

        int getTensorSize();
        
        std::vector<std::pair<int,int>>& getTensorShape();
        void reshapeTensor(std::vector<int> newShape); //very critical fucntion it only change how tesnor is viewed by making changes in dimention or adding/ removing dimentions. Number of element still remain same. 
        bool isSameShape(Tensor& t);
        
        void uploadData(std::vector<float>& data);
        std::vector<float> downloadata();
        bool downloadDataChecked(std::vector<float>& out);

        void  setData(std::vector<int> id, float val);
        float getData(std::vector<int> id);

        float* getTenorGpuData();

        void addTensorToTensor(Tensor& B);
        void copyTensor(Tensor& B);

};


#endif
