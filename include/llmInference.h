#ifndef LLMINFERENCE_H
#define LLMINFERENCE_H

#include <string>
#include <vector>

#include "embedding.h"
#include "linear.h"
#include "module.h"
#include "tokenUtility.h"

std::string runPromptInference(
    std::string prompt,
    TokenUtility& tokenizer,
    Embedding& embedding,
    std::vector<Module*>& layers,
    Linear& outputHead,
    int seqLen,
    int vocabSize,
    int maxNewTokens,
    std::string checkpointFile = "model_checkpoints/checkpoint_latest.bin"
);

#endif
