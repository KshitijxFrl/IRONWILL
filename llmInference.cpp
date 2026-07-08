#include "include/llmInference.h"

#include "include/filehandler.h"
#include "include/llmAssembly.h"

#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

bool inferenceFileExists(std::string fileName){
    std::ifstream file(fileName);
    bool exists = file.is_open();
    file.close();
    return exists;
}

int argmaxLastPosition(Tensor& logits,int position,int vocabSize){
    std::vector<float> cpuLogits = logits.downloadata();

    int rowStart = position * vocabSize;
    int bestToken = 0;
    float bestVal = cpuLogits[rowStart];

    for(int i = 1; i < vocabSize; i++){
        float val = cpuLogits[rowStart + i];

        if(val > bestVal){
            bestVal = val;
            bestToken = i;
        }
    }

    return bestToken;
}

void clearInferenceCaches(std::vector<Module*>& layers){
    for(int i = 0; i < layers.size(); i++){
        if(layers[i] != nullptr){
            layers[i]->clearCache();
        }
    }
}

void fillPromptTensor(std::vector<int>& tokens,Tensor& xInput,int seqLen){
    std::vector<float> inputData(seqLen, 0.0f);

    int start = 0;

    if(tokens.size() > seqLen){
        start = tokens.size() - seqLen;
    }

    int writeCount = tokens.size() - start;

    for(int i = 0; i < writeCount; i++){
        inputData[i] = (float)tokens[start + i];
    }

    xInput.uploadData(inputData);
}

std::string runPromptInference(
    std::string prompt,
    TokenUtility& tokenizer,
    Embedding& embedding,
    std::vector<Module*>& layers,
    Linear& outputHead,
    int seqLen,
    int vocabSize,
    int maxNewTokens,
    std::string checkpointFile
){
    std::vector<Module*> fullModel;

    fullModel.push_back(&embedding);

    for(int i = 0; i < layers.size(); i++){
        fullModel.push_back(layers[i]);
    }

    fullModel.push_back(&outputHead);

    if(!inferenceFileExists(checkpointFile)){
        std::cerr << "Inference checkpoint missing: " << checkpointFile << std::endl;
        return "";
    }

    FileHandler fileHandler;
    if(!fileHandler.fetchParameter(fullModel, checkpointFile)){
        std::cerr << "Failed to load inference checkpoint: " << checkpointFile << std::endl;
        return "";
    }

    std::vector<int> generated = tokenizer.encode(prompt);

    Tensor xInput({1, seqLen});

    for(int step = 0; step < maxNewTokens; step++){
        fillPromptTensor(generated,xInput,seqLen);

        std::vector<Tensor*> activationsToClean;

        Tensor* logits = modelForward(embedding,layers,outputHead,xInput,activationsToClean);

        cudaError_t forwardErr = cudaDeviceSynchronize();

        if(forwardErr != cudaSuccess){
            std::cerr << "Inference forward failed: " << cudaGetErrorString(forwardErr) << std::endl;

            if(logits != nullptr){
                logits->clear();
                delete logits;
                logits = nullptr;
            }

            cleanActivations(activationsToClean);
            clearInferenceCaches(layers);
            break;
        }

        int position = generated.size() - 1;

        if(position >= seqLen){
            position = seqLen - 1;
        }

        int nextToken = argmaxLastPosition(*logits,position,vocabSize);

        logits->clear();
        delete logits;
        logits = nullptr;

        cleanActivations(activationsToClean);
        clearInferenceCaches(layers);

        generated.push_back(nextToken);

        if(nextToken == tokenizer.getEosID()){
            break;
        }
    }

    xInput.clear();

    return tokenizer.decode(generated);
}
