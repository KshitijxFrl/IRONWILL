#ifndef FILEHANDLER_H
#define FILEHANDLER_H

#include <vector>
#include <string>
#include "module.h"

class FileHandler{
    public:
        bool setParameter(const std::vector<Module*>& model, std::string FILENAME);
        void fetchParameter(const std::vector<Module*>& model, std::string FILENAME);
};




#endif
