#include"include/filehandler.h"
#include<iostream>
#include<fstream>
#include<cstdio>
#include<cuda_runtime.h>


bool FileHandler::setParameter(const std::vector<Module*>& model, std::string FILENAME){
    cudaError_t syncErr = cudaDeviceSynchronize();

    if(syncErr != cudaSuccess){
        std::cerr << "Skipping checkpoint save because CUDA is unhealthy: "
                  << cudaGetErrorString(syncErr) << std::endl;
        return false;
    }

    std::string tempFileName = FILENAME + ".tmp";
    std::ofstream outfile(tempFileName, std::ios::binary);

    if(!outfile.is_open()){
        std::cerr<<"Failed to open "<<tempFileName<<std::endl;
        return false;
    } 
    

    for(auto layer: model){
        auto parameter = layer->getParameter();
        for(auto i : parameter){
            std::vector<float> temp;

            if(!i->data.downloadDataChecked(temp)){
                std::cerr << "Aborting checkpoint save because tensor download failed: "
                          << tempFileName << std::endl;
                outfile.close();
                std::remove(tempFileName.c_str());
                return false;
            }

            //i->data.getMatData().data() function returns a direct memory address (a pointer) to the first element inside the vector.
            outfile.write(reinterpret_cast<const char*>(temp.data()), i->data.getTensorSize() * sizeof(float));

            if(!outfile.good()){
                std::cerr << "Failed while writing checkpoint temp file: " << tempFileName << std::endl;
                outfile.close();
                std::remove(tempFileName.c_str());
                return false;
            }
        }
    }

    outfile.close();

    if(!outfile.good()){
        std::cerr << "Failed to finish checkpoint temp file: " << tempFileName << std::endl;
        std::remove(tempFileName.c_str());
        return false;
    }

    if(std::rename(tempFileName.c_str(), FILENAME.c_str()) != 0){
        std::cerr << "Failed to replace checkpoint file: " << FILENAME << std::endl;
        std::remove(tempFileName.c_str());
        return false;
    }

    return true;
}

bool FileHandler::fetchParameter(const std::vector<Module*>& model, std::string FILENAME){
    std::ifstream infile(FILENAME, std::ios::binary);

    if(!infile.is_open()){
        std::cerr << "Failed to open " << FILENAME << std::endl;
        return false;
    }

    std::vector<float> temp;

    for(auto layer: model){
        auto parameter = layer->getParameter();
        for(auto i : parameter){
            temp.resize(i->data.getTensorSize());
            infile.read(reinterpret_cast<char*>(temp.data()), i->data.getTensorSize() * sizeof(float));

            if(!infile.good()){
                std::cerr << "Failed to read checkpoint tensor from: " << FILENAME << std::endl;
                infile.close();
                return false;
            }

            i->data.uploadData(temp);
        }
    }    
    
    infile.close();
    return true;
}
