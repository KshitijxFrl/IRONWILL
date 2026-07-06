#include"include/filehandler.h"
#include<iostream>
#include<fstream>


void FileHandler::setParameter(const std::vector<Module*>& model, std::string FILENAME){
    std::ofstream outfile(FILENAME, std::ios::binary);

    if(!outfile.is_open()){
        std::cerr<<"Failed to open "<<FILENAME<<std::endl;
        return;
    } 
    

    for(auto layer: model){
        auto parameter = layer->getParameter();
        for(auto i : parameter){
            auto temp = i->data.downloadata();
            //i->data.getMatData().data() function returns a direct memory address (a pointer) to the first element inside the vector.
            outfile.write(reinterpret_cast<const char*>(temp.data()), i->data.getTensorSize() * sizeof(float));
        }
    }
    outfile.close();
}

void FileHandler::fetchParameter(const std::vector<Module*>& model, std::string FILENAME){
    std::ifstream infile(FILENAME, std::ios::binary);

    if(!infile.is_open()){
        std::cerr << "Failed to open " << FILENAME << std::endl;
        return;
    }

    std::vector<float> temp;

    for(auto layer: model){
        auto parameter = layer->getParameter();
        for(auto i : parameter){
            temp.resize(i->data.getTensorSize());
            infile.read(reinterpret_cast<char*>(temp.data()), i->data.getTensorSize() * sizeof(float));
            i->data.uploadData(temp);
        }
    }    
    
    infile.close();
}