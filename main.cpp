#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "include/llmAssembly.h"
#include "include/llmInference.h"
#include "include/tokenUtility.h"

#include "include/module.h"
#include "include/tensor.h"
#include "include/embedding.h"
#include "include/linear.h"
#include "include/rmsNorm.h"
#include "include/attention.h"
#include "include/residualADD.h"
#include "include/moe.h"

bool localFileExists(std::string fileName){
    std::ifstream file(fileName);
    bool exists = file.is_open();
    file.close();
    return exists;
}

bool localDirectoryExists(std::string dirName){
    struct stat info;

    if(stat(dirName.c_str(), &info) != 0){
        return false;
    }

    return S_ISDIR(info.st_mode);
}

void createLocalDirectory(std::string dirName){
    if(localDirectoryExists(dirName)){
        return;
    }

    if(mkdir(dirName.c_str(), 0755) != 0){
        std::cerr << "Failed to create directory: " << dirName << std::endl;
    }
}

std::string joinPath(std::string folder,std::string fileName){
    if(folder == "" || folder == "."){
        return fileName;
    }

    if(folder.back() == '/'){
        return folder + fileName;
    }

    return folder + "/" + fileName;
}

int main(int argc, char** argv){
    int cudaDeviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);

    if(cudaStatus != cudaSuccess || cudaDeviceCount == 0){
        std::cerr << "No CUDA capable device available(no nvidia gpu): "<< cudaGetErrorString(cudaStatus)<< std::endl;
        return 1;
    }

    srand(time(NULL));

    // Edit these folders when moving datasets, generated token bins, or vocab files.
    std::string datasetLocation = "dataset";
    std::string binLocation     = "binFiles";
    std::string vocabLocation   = "vocab";

    std::string mergeFile    = joinPath(datasetLocation, "manual_merges.txt");
    std::string rawTrainFile = joinPath(datasetLocation, "train.jsonl");
    std::string rawValFile   = joinPath(datasetLocation, "validation.jsonl");
    std::string rawTestFile  = joinPath(datasetLocation, "test.jsonl");
    std::string vocabFile    = joinPath(vocabLocation, "vocab.json");
    std::string trainBinFile = joinPath(binLocation, "train_tokens.bin");
    std::string valBinFile   = joinPath(binLocation, "val_tokens.bin");
    std::string testBinFile  = joinPath(binLocation, "test_tokens.bin");
    
    bool promptMode = (argc > 1 && std::string(argv[1]) == "prompt");

    createLocalDirectory(binLocation);
    createLocalDirectory(vocabLocation);

    // -------------------------
    // IRONWILL_V1 config
    // -------------------------
    int batchLen = 1;
    int seqLen = 512;

    int dModel = 512;

    int head = 8;
    int kvHead = 4;
    int headDim = 64;        // head * headDim = dModel

    int hiddenDim = 4096;
    int blockCount = 4;
    int expertCount = 4;

    int totalSteps = 15000;//10000
    float learningRate = 0.0003f;

    // -------------------------
    // Tokenizer preparation
    // -------------------------
    TokenUtility tokenizer;

    tokenizer.loadManualMerges(mergeFile);

    if(promptMode){
        if(!localFileExists(vocabFile)){
            std::cerr << "Prompt mode needs existing vocab file: " << vocabFile << std::endl;
            return 1;
        }

        tokenizer.loadVocab(vocabFile);
    }else{
        if(localFileExists(vocabFile)){
            std::cout << "Using existing vocab file: " << vocabFile << std::endl;
        }else{
            std::cout << "Creating vocab file: " << vocabFile << std::endl;
            tokenizer.createVocabFromJsonl(
                rawTrainFile,
                vocabFile
            );
        }

        tokenizer.loadVocab(vocabFile);

        if(localFileExists(trainBinFile)){
            std::cout << "Using existing train token bin: " << trainBinFile << std::endl;
        }else{
            std::cout << "Creating train token bin: " << trainBinFile << std::endl;
            tokenizer.encodeJsonlToPaddedBin(
                rawTrainFile,
                trainBinFile,
                seqLen
            );
        }

        if(rawValFile != "" && localFileExists(rawValFile)){
            if(localFileExists(valBinFile)){
                std::cout << "Using existing validation token bin: " << valBinFile << std::endl;
            }else{
                std::cout << "Creating validation token bin: " << valBinFile << std::endl;
                tokenizer.encodeJsonlToPaddedBin(
                    rawValFile,
                    valBinFile,
                    seqLen
                );
            }
        }else{
            valBinFile = "";
        }

        if(rawTestFile != "" && localFileExists(rawTestFile)){
            if(localFileExists(testBinFile)){
                std::cout << "Using existing test token bin: " << testBinFile << std::endl;
            }else{
                std::cout << "Creating test token bin: " << testBinFile << std::endl;
                tokenizer.encodeJsonlToPaddedBin(
                    rawTestFile,
                    testBinFile,
                    seqLen
                );
            }
        }else{
            testBinFile = "";
        }
    }

    int vocabSize = tokenizer.getVocabSize();

    std::cout << "Final vocab size: " << vocabSize << std::endl;

    if(head * headDim != dModel){
        std::cerr << "Bad config: head * headDim must equal dModel" << std::endl;
        return 1;
    }

    if(head % kvHead != 0){
        std::cerr << "Bad config: head must be divisible by kvHead" << std::endl;
        return 1;
    }

    if(seqLen > 512){
        std::cerr << "Bad config: AttentionGQA supports max seqLen 512" << std::endl;
        return 1;
    }

    std::cout << "Building IRONWILL_V1 model..." << std::endl;

    Embedding embedding(batchLen, seqLen, dModel, vocabSize);;

    std::vector<Module*> layers;
    std::vector<Module*> ownedActionLayers;



    // -------------------------
    // IRONWILL_V1 blocks:
    // RMSNorm -> Residual Attention -> RMSNorm -> Residual MoE
    // -------------------------
    for(int block = 0; block < blockCount; block++){
        RMSNormal* normAttn   = new RMSNormal(dModel);

        AttentionGQA* attn    = new AttentionGQA(batchLen,seqLen,kvHead,head,headDim);

        ResidualAdd* resAttn  = new ResidualAdd(*attn);

        RMSNormal* normMoe    = new RMSNormal(dModel);

        MoE* moe              = new MoE(batchLen,seqLen,dModel,hiddenDim,expertCount);

        ResidualAdd* resMoe   = new ResidualAdd(*moe);

        layers.push_back(normAttn);
        layers.push_back(resAttn);
        layers.push_back(normMoe);
        layers.push_back(resMoe);

        ownedActionLayers.push_back(attn);
        ownedActionLayers.push_back(moe);
    }

    RMSNormal* finalNorm = new RMSNormal(dModel);
    layers.push_back(finalNorm);

    // Output head: [B, T, dModel] -> [B, T, vocabSize]
    Linear outputHead(batchLen, dModel, vocabSize);

    long long embeddingParams = (long long)vocabSize * dModel;
    long long outputHeadParams = (long long)dModel * vocabSize + vocabSize;
    long long attnParamsPerBlock =
        (long long)dModel * dModel +
        (long long)dModel * (kvHead * headDim) +
        (long long)dModel * (kvHead * headDim) +
        (long long)dModel * dModel;
    long long moeParamsPerExpert =
        (long long)dModel * hiddenDim +
        (long long)dModel * hiddenDim +
        (long long)hiddenDim * dModel;
    long long moeParamsPerBlock =
        (long long)dModel * expertCount + expertCount +
        (long long)expertCount * moeParamsPerExpert;
    long long normParams =
        (long long)(blockCount * 2 + 1) * dModel;
    long long estimatedParams =
        embeddingParams +
        outputHeadParams +
        (long long)blockCount * (attnParamsPerBlock + moeParamsPerBlock) +
        normParams;

    std::cout << "IRONWILL_V1 config: "
              << "blocks=" << blockCount
              << ", experts=" << expertCount
              << ", dModel=" << dModel
              << ", hiddenDim=" << hiddenDim
              << ", seqLen=" << seqLen
              << ", heads=" << head
              << ", kvHeads=" << kvHead
              << std::endl;
    std::cout << "Estimated parameters: " << estimatedParams << std::endl;

    if(promptMode){
        std::string prompt = "";

        for(int i = 2; i < argc; i++){
            if(prompt.size() > 0){
                prompt += " ";
            }

            prompt += argv[i];
        }

        if(prompt == ""){
            std::cerr << "Prompt mode needs text after: prompt" << std::endl;
        }else{
            std::string output = runPromptInference(prompt,tokenizer,embedding,layers,outputHead,seqLen,vocabSize,32);

            std::cout << output << std::endl;
        }
    }else{
        std::cout << "Starting training..." << std::endl;

        trainLoop(trainBinFile,valBinFile,testBinFile,embedding,layers,outputHead,batchLen,seqLen,vocabSize,totalSteps,learningRate,500,10,100,1);

        std::cout << "Training finished." << std::endl;
    }

    for(int i = 0; i < layers.size(); i++){
        if(layers[i] != nullptr){
            delete layers[i];
            layers[i] = nullptr;
        }
    }

    layers.clear();

    for(int i = 0; i < ownedActionLayers.size(); i++){
        if(ownedActionLayers[i] != nullptr){
            delete ownedActionLayers[i];
            ownedActionLayers[i] = nullptr;
        }
    }

    ownedActionLayers.clear();

    return 0;
}
