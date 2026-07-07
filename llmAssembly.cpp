#include "include/llmAssembly.h"
#include "include/filehandler.h"
#include "include/crossEntropyLoss.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>

// xInput  = input going inside the model
// xTarget = values which we need to compare with the model output to compute loss 


std::vector<int> loadTokenBin(std::string fileName){
    std::ifstream file(fileName, std::ios::binary);

    if(!file.is_open()){
        std::cerr << "Failed to open token bin: " << fileName << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    long long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    long long tokenCount = fileSize / sizeof(int);

    std::vector<int> tokens(tokenCount);

    file.read(
        reinterpret_cast<char*>(tokens.data()),
        tokenCount * sizeof(int)
    );

    file.close();

    std::cout << "Loaded tokens: " << tokenCount << std::endl;

    return tokens;
}

void createBatch(std::vector<int>& tokens,Tensor& xInput,Tensor& xTarget,int batchLen,int seqLen,int batchIndex){
    
    std::vector<float> inputData(batchLen * seqLen);
    std::vector<float> targetData(batchLen * seqLen);

    int recordLen = seqLen + 1;
    int sampleCount = tokens.size() / recordLen;
    int startSample = batchIndex * batchLen;

    if(startSample + batchLen > sampleCount){
        std::cerr << "Not enough tokens to create batch !!! " << std::endl;
        return;
    }

    for(int b = 0; b < batchLen; b++){
        int sampleStart = (startSample + b) * recordLen;

        for(int t = 0; t < seqLen; t++){
            int idx = b * seqLen + t;

            inputData[idx] = (float)tokens[sampleStart + t];
            targetData[idx] = (float)tokens[sampleStart + t + 1];
        }
    }

    xInput.uploadData(inputData);
    xTarget.uploadData(targetData);
}



std::vector<Parameter*> collectAllParameters(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead){
    
    
    std::vector<Parameter*> params;
    std::vector<Parameter*> embeddingParams = embedding.getParameter();

    for(int i = 0; i < embeddingParams.size(); i++){
        params.push_back(embeddingParams[i]);
    }

    for(int i = 0; i < layers.size(); i++){
        std::vector<Parameter*> layerParams = layers[i]->getParameter();

        for(int j = 0; j < layerParams.size(); j++){
            params.push_back(layerParams[j]);
        }
    }

    std::vector<Parameter*> outputParams = outputHead.getParameter();

    for(int i = 0; i < outputParams.size(); i++){
        params.push_back(outputParams[i]);
    }

    return params;
}



Tensor* modelForward(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead,Tensor& xInput,std::vector<Tensor*>& activationsToClean){
    
    Tensor* x = embedding.feedForward(xInput);

    activationsToClean.push_back(x);

    for(int i = 0; i < layers.size(); i++){
        Tensor* next = layers[i]->feedForward(*x);

        activationsToClean.push_back(next);

        x = next;
    }

    Tensor* logits = outputHead.feedForward(*x);

    return logits;
}


Tensor* modelBackward(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead,Tensor& gradLogits){
    
    Tensor* grad = outputHead.backward(gradLogits);

    for(int i = (int)layers.size() - 1; i >= 0; i--){
        Tensor* nextGrad = layers[i]->backward(*grad);

        grad->clear();
        delete grad;

        grad = nextGrad;
    }

    Tensor* embeddingGrad = embedding.backward(*grad);

    grad->clear();
    delete grad;

    return embeddingGrad;
}

void cleanActivations(std::vector<Tensor*>& activationsToClean){
    for(int i = 0; i < activationsToClean.size(); i++){
        if(activationsToClean[i] != nullptr){
            activationsToClean[i]->clear();
            delete activationsToClean[i];
            activationsToClean[i] = nullptr;
        }
    }

    activationsToClean.clear();
}

bool pathExists(std::string path){
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

long long pathFileSize(std::string path){
    struct stat info;

    if(stat(path.c_str(), &info) != 0){
        return 0;
    }

    return (long long)info.st_size;
}

void createDirectory(std::string path){
    if(pathExists(path)){
        return;
    }

    if(mkdir(path.c_str(), 0755) != 0){
        std::cerr << "Failed to create directory: " << path << std::endl;
    }
}

std::pair<int,std::string> findLatestNumberedCheckpoint(std::string checkpointDir){
    int latestStep = -1;
    std::string latestPath = "";

    if(!pathExists(checkpointDir)){
        return {latestStep, latestPath};
    }

    DIR* dir = opendir(checkpointDir.c_str());

    if(dir == nullptr){
        return {latestStep, latestPath};
    }

    dirent* entry;

    while((entry = readdir(dir)) != nullptr){
        std::string fileName = entry->d_name;
        std::string fullPath = checkpointDir + "/" + fileName;

        struct stat info;

        if(stat(fullPath.c_str(), &info) != 0) continue;
        if(!S_ISREG(info.st_mode)) continue;

        std::string prefix = "checkpoint_step_";
        std::string suffix = ".bin";

        if(fileName.size() <= prefix.size() + suffix.size()) continue;
        if(fileName.substr(0, prefix.size()) != prefix) continue;
        if(fileName.substr(fileName.size() - suffix.size()) != suffix) continue;

        std::string stepText = fileName.substr(
            prefix.size(),
            fileName.size() - prefix.size() - suffix.size()
        );

        int step = -1;

        try{
            step = std::stoi(stepText);
        }
        catch(...){
            continue;
        }

        if(step > latestStep){
            latestStep = step;
            latestPath = fullPath;
        }
    }

    closedir(dir);

    return {latestStep, latestPath};
}

std::string optimizerCheckpointPath(std::string modelCheckpointPath){
    std::string suffix = ".bin";

    if(modelCheckpointPath.size() >= suffix.size()){
        std::string ending = modelCheckpointPath.substr(modelCheckpointPath.size() - suffix.size());

        if(ending == suffix){
            return modelCheckpointPath.substr(0, modelCheckpointPath.size() - suffix.size()) + ".optim";
        }
    }

    return modelCheckpointPath + ".optim";
}

void cleanupOldNumberedCheckpoints(std::string checkpointDir,int keepCount){
    if(keepCount <= 0) return;
    if(!pathExists(checkpointDir)) return;

    DIR* dir = opendir(checkpointDir.c_str());

    if(dir == nullptr){
        return;
    }

    std::vector<std::pair<int,std::string>> checkpoints;
    dirent* entry;

    while((entry = readdir(dir)) != nullptr){
        std::string fileName = entry->d_name;
        std::string fullPath = checkpointDir + "/" + fileName;

        struct stat info;

        if(stat(fullPath.c_str(), &info) != 0) continue;
        if(!S_ISREG(info.st_mode)) continue;

        std::string prefix = "checkpoint_step_";
        std::string suffix = ".bin";

        if(fileName.size() <= prefix.size() + suffix.size()) continue;
        if(fileName.substr(0, prefix.size()) != prefix) continue;
        if(fileName.substr(fileName.size() - suffix.size()) != suffix) continue;

        std::string stepText = fileName.substr(
            prefix.size(),
            fileName.size() - prefix.size() - suffix.size()
        );

        int step = -1;

        try{
            step = std::stoi(stepText);
        }
        catch(...){
            continue;
        }

        checkpoints.push_back({step,fullPath});
    }

    closedir(dir);

    while((int)checkpoints.size() > keepCount){
        int oldestIndex = 0;

        for(int i = 1; i < checkpoints.size(); i++){
            if(checkpoints[i].first < checkpoints[oldestIndex].first){
                oldestIndex = i;
            }
        }

        std::string modelFile = checkpoints[oldestIndex].second;
        std::string optimFile = optimizerCheckpointPath(modelFile);

        std::cout << "Deleting old checkpoint: " << modelFile << std::endl;
        std::remove(modelFile.c_str());
        std::remove(optimFile.c_str());

        checkpoints.erase(checkpoints.begin() + oldestIndex);
    }
}

void saveBestMetric(std::string bestMetricFile,float bestLoss){
    std::string tempFileName = bestMetricFile + ".tmp";
    std::ofstream bestFile(tempFileName);

    if(!bestFile.is_open()){
        std::cerr << "Failed to open best metric temp file: " << tempFileName << std::endl;
        return;
    }

    bestFile << bestLoss << std::endl;
    bestFile.close();

    if(!bestFile.good()){
        std::cerr << "Failed to write best metric temp file: " << tempFileName << std::endl;
        std::remove(tempFileName.c_str());
        return;
    }

    if(std::rename(tempFileName.c_str(), bestMetricFile.c_str()) != 0){
        std::cerr << "Failed to replace best metric file: " << bestMetricFile << std::endl;
        std::remove(tempFileName.c_str());
    }
}

bool saveCheckpointWithOptimizer(
    FileHandler& fileHandler,
    Optimizer& optimizer,
    std::vector<Module*>& fullModel,
    std::string checkpointFile
){
    bool modelSaved = fileHandler.setParameter(fullModel, checkpointFile);

    if(!modelSaved){
        std::cerr << "Checkpoint model save failed: " << checkpointFile << std::endl;
        return false;
    }

    bool optimizerSaved = optimizer.saveState(optimizerCheckpointPath(checkpointFile));

    if(!optimizerSaved){
        std::cerr << "Checkpoint optimizer save failed: "
                  << optimizerCheckpointPath(checkpointFile) << std::endl;
        return false;
    }

    return true;
}

void appendProgress(std::string progressFile,int step,float trainLoss,float valLoss){
    bool needsHeader = true;

    if(pathExists(progressFile) && pathFileSize(progressFile) > 0){
        needsHeader = false;
    }

    std::ofstream progress(progressFile, std::ios::app);

    if(!progress.is_open()){
        std::cerr << "Failed to open progress file: " << progressFile << std::endl;
        return;
    }

    if(needsHeader){
        progress << "step,train_loss,val_loss\n";
    }

    progress << step << "," << trainLoss << "," << valLoss << "\n";
    progress.close();
}

void createProgressFile(std::string progressFile){
    if(pathExists(progressFile) && pathFileSize(progressFile) > 0){
        return;
    }

    std::ofstream progress(progressFile, std::ios::app);

    if(!progress.is_open()){
        std::cerr << "Failed to open progress file: " << progressFile << std::endl;
        return;
    }

    progress << "step,train_loss,val_loss\n";
    progress.close();
}

float evaluateLoss(
    std::vector<int>& evalTokens,
    Embedding& embedding,
    std::vector<Module*>& layers,
    Linear& outputHead,
    int batchLen,
    int seqLen,
    int vocabSize,
    int evalBatches
){
    int recordLen = seqLen + 1;
    int evalSampleCount = evalTokens.size() / recordLen;
    int availableBatches = evalSampleCount / batchLen;

    if(availableBatches <= 0){
        std::cerr << "Not enough tokens for validation !!!" << std::endl;
        return -1.0f;
    }

    if(evalBatches <= 0){
        return -1.0f;
    }

    if(evalBatches > availableBatches){
        evalBatches = availableBatches;
    }

    Tensor xInput({batchLen, seqLen});
    Tensor xTarget({batchLen, seqLen});
    Tensor gradLogits({batchLen, seqLen, vocabSize});

    float totalLoss = 0.0f;

    for(int i = 0; i < evalBatches; i++){
        createBatch(evalTokens,xInput,xTarget,batchLen,seqLen,i);

        std::vector<Tensor*> activationsToClean;

        Tensor* logits = modelForward(embedding,layers,outputHead,xInput,activationsToClean);

        float loss = crossEntropyLoss(*logits,xTarget,gradLogits);

        totalLoss += loss;

        logits->clear();
        delete logits;
        logits = nullptr;

        cleanActivations(activationsToClean);
    }

    xInput.clear();
    xTarget.clear();
    gradLogits.clear();

    return totalLoss / evalBatches;
}

void trainLoop(
    std::string trainBinFile,
    std::string valBinFile,
    std::string testBinFile,
    Embedding& embedding,
    std::vector<Module*>& layers,
    Linear& outputHead,
    int batchLen,
    int seqLen,
    int vocabSize,
    int totalSteps,
    float learningRate,
    int saveEvery,
    int logEvery,
    int valEvery,
    int evalBatches
){
    std::string checkpointDir = "model_checkpoints";
    std::string progressDir = "progress";

    createDirectory(checkpointDir);
    createDirectory(progressDir);

    std::string latestCheckpointFile = checkpointDir + "/checkpoint_latest.bin";
    std::string bestCheckpointFile = checkpointDir + "/checkpoint_best.bin";
    std::string bestMetricFile = checkpointDir + "/checkpoint_best_metric.txt";
    std::string progressFile = progressDir + "/training_progress.csv";

    createProgressFile(progressFile);
    
    
    std::vector<int> trainTokens = loadTokenBin(trainBinFile);

    int recordLen = seqLen + 1;
    int trainSampleCount = trainTokens.size() / recordLen;
    int availableTrainBatches = trainSampleCount / batchLen;

    if(availableTrainBatches <= 0){
        std::cerr << "Not enough tokens for training !!!" << std::endl;
        return;
    }

    std::vector<int> valTokens;
    bool hasValidation = false;

    if(valBinFile == ""){
        std::cerr << "Warning: validation token file not provided." << std::endl;
    }else if(!pathExists(valBinFile)){
        std::cerr << "Warning: validation token file missing: " << valBinFile << std::endl;
    }else{
        valTokens = loadTokenBin(valBinFile);
        int valRecordLen = seqLen + 1;
        int valSampleCount = valTokens.size() / valRecordLen;

        if(valSampleCount / batchLen > 0){
            hasValidation = true;
        }else{
            std::cerr << "Warning: validation token file too small." << std::endl;
        }
    }

    if(testBinFile == ""){
        std::cerr << "Warning: test token file not provided." << std::endl;
    }else if(!pathExists(testBinFile)){
        std::cerr << "Warning: test token file missing: " << testBinFile << std::endl;
    }else{
        std::vector<int> testTokens = loadTokenBin(testBinFile);
        int testRecordLen = seqLen + 1;
        int testSampleCount = testTokens.size() / testRecordLen;

        if(testSampleCount / batchLen <= 0){
            std::cerr << "Warning: test token file too small." << std::endl;
        }
    }

    std::vector<Module*> fullModel;

    fullModel.push_back(&embedding);

    for(int i = 0; i < layers.size(); i++){
        fullModel.push_back(layers[i]);
    }

    fullModel.push_back(&outputHead);

    FileHandler fileHandler;
    int startStep = 0;
    std::string optimizerStateToLoad = "";

    std::pair<int,std::string> latestNumbered = findLatestNumberedCheckpoint(checkpointDir);

    if(latestNumbered.first >= 0){
        std::cout << "Loading checkpoint: " << latestNumbered.second << std::endl;
        fileHandler.fetchParameter(fullModel, latestNumbered.second);
        optimizerStateToLoad = optimizerCheckpointPath(latestNumbered.second);
        startStep = latestNumbered.first + 1;
        std::cout << "Resuming from step: " << startStep << std::endl;
    }else if(pathExists(latestCheckpointFile)){
        std::cout << "Loading checkpoint: " << latestCheckpointFile << std::endl;
        fileHandler.fetchParameter(fullModel, latestCheckpointFile);
        optimizerStateToLoad = optimizerCheckpointPath(latestCheckpointFile);
        startStep = 0;
        std::cout << "Loaded latest checkpoint without numbered step. Starting from step 0." << std::endl;
    }else{
        std::cout << "No checkpoint found. Starting from random initialized weights." << std::endl;
    }

    std::vector<Parameter*> params = collectAllParameters(
        embedding,
        layers,
        outputHead
    );

    Optimizer optimizer(params, learningRate);

    if(optimizerStateToLoad != ""){
        if(pathExists(optimizerStateToLoad)){
            std::cout << "Loading optimizer state: " << optimizerStateToLoad << std::endl;
            optimizer.loadState(optimizerStateToLoad);
        }else{
            std::cerr << "Warning: optimizer state missing: " << optimizerStateToLoad << std::endl;
        }
    }

    Tensor xInput({batchLen, seqLen});
    Tensor xTarget({batchLen, seqLen});
    Tensor gradLogits({batchLen, seqLen, vocabSize});
    float lastLoss = -1.0f;
    int lastStep = startStep - 1;
    float bestLoss = -1.0f;

    if(pathExists(bestMetricFile)){
        std::ifstream bestFile(bestMetricFile);

        if(bestFile.is_open()){
            bestFile >> bestLoss;
            bestFile.close();
            std::cout << "Loaded best checkpoint metric: " << bestLoss << std::endl;
        }
    }

    int maxTrainSteps = totalSteps;

    if(maxTrainSteps > availableTrainBatches){
        maxTrainSteps = availableTrainBatches;
    }

    if(startStep >= maxTrainSteps){
        std::cout << "No full batches left to train from step: " << startStep << std::endl;
    }

    for(int step = startStep; step < maxTrainSteps; step++){
        optimizer.zeroGrad();

        createBatch(trainTokens,xInput,xTarget,batchLen,seqLen,step);

        std::vector<Tensor*> activationsToClean;

        Tensor* logits = modelForward(embedding,layers,outputHead,xInput,activationsToClean);

        float loss = crossEntropyLoss(*logits,xTarget,gradLogits);
        lastLoss = loss;
        lastStep = step;

        Tensor* embeddingGrad = modelBackward(embedding,layers,outputHead,gradLogits);

        if(embeddingGrad != nullptr){
            embeddingGrad->clear();
            delete embeddingGrad;
            embeddingGrad = nullptr;
        }

        optimizer.adamW();

        logits->clear();
        delete logits;
        logits = nullptr;

        cleanActivations(activationsToClean);

        if(logEvery > 0 && step % logEvery == 0){
            float valLoss = -1.0f;

            if(hasValidation && valEvery > 0 && step % valEvery == 0){
                valLoss = evaluateLoss(
                    valTokens,
                    embedding,
                    layers,
                    outputHead,
                    batchLen,
                    seqLen,
                    vocabSize,
                    evalBatches
                );
            }

            appendProgress(progressFile,step,loss,valLoss);

            float compareLoss = loss;

            if(valLoss >= 0.0f){
                compareLoss = valLoss;
            }

            if(bestLoss < 0.0f || compareLoss < bestLoss){
                bestLoss = compareLoss;

                std::cout << "Saving best checkpoint: " << bestCheckpointFile
                          << " Best Loss: " << bestLoss
                          << std::endl;

                if(saveCheckpointWithOptimizer(fileHandler,optimizer,fullModel,bestCheckpointFile)){
                    saveBestMetric(bestMetricFile,bestLoss);
                }else{
                    std::cerr << "Best checkpoint save failed. Best metric was not updated." << std::endl;
                }
            }

            std::cout << "Step: " << step
                      << " Train Loss: " << loss
                      << " Val Loss: " << valLoss
                      << std::endl;
        }


        if(saveEvery > 0 && step > 0 && step % saveEvery == 0){
            std::string saveName = checkpointDir + "/checkpoint_step_" + std::to_string(step) + ".bin";

            std::cout << "Saving checkpoint: " << saveName << std::endl;

            if(saveCheckpointWithOptimizer(fileHandler,optimizer,fullModel,saveName)){
                cleanupOldNumberedCheckpoints(checkpointDir,2);
            }else{
                std::cerr << "Numbered checkpoint save failed. Old checkpoints were kept." << std::endl;
            }
        }
    }

    if(lastStep >= startStep){
        float finalValLoss = -1.0f;

        if(hasValidation){
            finalValLoss = evaluateLoss(
                valTokens,
                embedding,
                layers,
                outputHead,
                batchLen,
                seqLen,
                vocabSize,
                evalBatches
            );
        }

        appendProgress(progressFile,lastStep,lastLoss,finalValLoss);

        float compareLoss = lastLoss;

        if(finalValLoss >= 0.0f){
            compareLoss = finalValLoss;
        }

        if(bestLoss < 0.0f || compareLoss < bestLoss){
            bestLoss = compareLoss;

            std::cout << "Saving best checkpoint: " << bestCheckpointFile
                      << " Best Loss: " << bestLoss
                      << std::endl;

            if(saveCheckpointWithOptimizer(fileHandler,optimizer,fullModel,bestCheckpointFile)){
                saveBestMetric(bestMetricFile,bestLoss);
            }else{
                std::cerr << "Best checkpoint save failed. Best metric was not updated." << std::endl;
            }
        }
    }

    std::cout << "Saving final checkpoint: " << latestCheckpointFile << std::endl;
    saveCheckpointWithOptimizer(fileHandler,optimizer,fullModel,latestCheckpointFile);

    xInput.clear();
    xTarget.clear();
    gradLogits.clear();
}
