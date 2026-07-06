#ifndef LLMASSEMBLY_H
#define LLMASSEMBLY_H

#include <string>
#include <vector>
#include <utility>

#include "tensor.h"
#include "module.h"
#include "parameter.h"
#include "embedding.h"
#include "linear.h"
#include "optimizer.h"

std::vector<int> loadTokenBin(std::string fileName);

void createBatch(std::vector<int>& tokens,Tensor& xInput,Tensor& xTarget,int batchLen,int seqLen,int batchIndex);

std::vector<Parameter*> collectAllParameters(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead);

Tensor* modelForward(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead,Tensor& xInput,std::vector<Tensor*>& activationsToClean);

Tensor* modelBackward(Embedding& embedding,std::vector<Module*>& layers,Linear& outputHead,Tensor& gradLogits);

void cleanActivations(std::vector<Tensor*>& activationsToClean);

std::pair<int,std::string> findLatestNumberedCheckpoint(std::string checkpointDir);

float evaluateLoss(
    std::vector<int>& evalTokens,
    Embedding& embedding,
    std::vector<Module*>& layers,
    Linear& outputHead,
    int batchLen,
    int seqLen,
    int vocabSize,
    int evalBatches
);

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
    int saveEvery = 50,
    int logEvery = 10,
    int valEvery = 50,
    int evalBatches = 3
);

#endif
