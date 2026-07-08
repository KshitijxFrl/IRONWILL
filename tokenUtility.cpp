#include "include/tokenUtility.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

TokenUtility::TokenUtility(){
    this->padID = 0;
    this->bosID = 1;
    this->eosID = 2;
    this->unkID = 3;

    this->vocabMAP["<PAD>"]   = 0;
    this->vocabMAP["<BOS>"]   = 1;
    this->vocabMAP["<EOS>"]   = 2;
    this->vocabMAP["<UNK>"]   = 3;
    this->vocabMAP["[ARITH]"] = 4;
    this->vocabMAP["[ALG]"]   = 5;
    this->vocabMAP["[CALC]"]  = 6;
    this->vocabMAP["[TRIG]"]  = 7;

    this->idToTokenMAP[0] = "<PAD>";
    this->idToTokenMAP[1] = "<BOS>";
    this->idToTokenMAP[2] = "<EOS>";
    this->idToTokenMAP[3] = "<UNK>";
    this->idToTokenMAP[4] = "[ARITH]";
    this->idToTokenMAP[5] = "[ALG]";
    this->idToTokenMAP[6] = "[CALC]";
    this->idToTokenMAP[7] = "[TRIG]";
}

bool TokenUtility::isSymbol(char c){
    std::string symbolList = "{}[]()/<>^*+-.:;=";
    return symbolList.find(c) != symbolList.npos;
}

bool TokenUtility::isNum(char c){
    std::string numList = "1234567890";
    return numList.find(c) != numList.npos;
}

void TokenUtility::addTokenToList(
    std::string& prevToken,
    std::vector<std::string>& currTokenList
){
    if(prevToken == "") return;

    currTokenList.push_back(prevToken);
    prevToken.clear();
}

std::vector<std::string> TokenUtility::tokenizeString(std::string& targetStr){
    std::vector<std::string> currTokenList;

    if(targetStr.empty()) return currTokenList;

    std::string currToken = "";
    bool currTokenNum = false;

    int size = targetStr.size();
    int currIDX = 0;

    if(targetStr[currIDX] == '['){
        while(currIDX < size){
            if(targetStr[currIDX] == ']'){
                currToken.push_back(targetStr[currIDX]);
                currIDX++;
                break;
            }

            currToken.push_back(targetStr[currIDX]);
            currIDX++;
        }

        addTokenToList(currToken, currTokenList);
    }

    while(currIDX < size){
        if(targetStr[currIDX] == ' ' || targetStr[currIDX] == '\n'){
            addTokenToList(currToken, currTokenList);

            currIDX++;
            currTokenNum = false;
            continue;
        }

        if(isSymbol(targetStr[currIDX])){
            currTokenNum = false;

            addTokenToList(currToken, currTokenList);

            currToken.push_back(targetStr[currIDX]);

            addTokenToList(currToken, currTokenList);

            currIDX++;
            continue;
        }

        if(isNum(targetStr[currIDX])){
            if(currTokenNum == false){
                currTokenNum = true;

                addTokenToList(currToken, currTokenList);

                currToken.push_back(targetStr[currIDX]);
                currIDX++;
                continue;
            }else{
                currToken.push_back(targetStr[currIDX]);
                currIDX++;
                continue;
            }
        }

        if(!isNum(targetStr[currIDX]) && currTokenNum == true){
            currTokenNum = false;
            addTokenToList(currToken, currTokenList);
        }

        currToken.push_back(targetStr[currIDX]);
        currIDX++;
    }

    addTokenToList(currToken, currTokenList);

    return currTokenList;
}

void TokenUtility::loadManualMerges(std::string mergeFileName){
    std::ifstream file(mergeFileName);

    if(!file.is_open()){
        std::cerr << "Failed to open merge file: " << mergeFileName << std::endl;
        return;
    }

    this->mergeRules.clear();

    std::string line;

    while(std::getline(file, line)){
        if(line.empty()) continue;

        std::stringstream ss(line);

        std::string first;
        std::string second;

        ss >> first >> second;

        if(first == "" || second == "") continue;

        this->mergeRules.push_back({first, second});
    }

    file.close();
}

std::vector<std::string> TokenUtility::applyManualMerges(
    std::vector<std::string>& tokens
){
    std::vector<std::string> curr = tokens;

    for(int r = 0; r < this->mergeRules.size(); r++){
        std::string first = this->mergeRules[r].first;
        std::string second = this->mergeRules[r].second;

        std::vector<std::string> merged;

        int i = 0;

        while(i < curr.size()){
            if(i + 1 < curr.size() && curr[i] == first && curr[i + 1] == second){
                merged.push_back(first + second);
                i += 2;
            }else{
                merged.push_back(curr[i]);
                i++;
            }
        }

        curr = merged;
    }

    return curr;
}

void TokenUtility::addTokenToVocab(
    const std::string& token,
    int& nextID
){
    if(this->vocabMAP.find(token) == this->vocabMAP.end()){
        this->vocabMAP[token] = nextID;
        this->idToTokenMAP[nextID] = token;
        nextID++;
    }
}

void TokenUtility::createVocabFromJsonl(
    std::string jsonlFileName,
    std::string vocabFileName,
    int maxSamples
){
    std::ifstream file(jsonlFileName);

    if(!file.is_open()){
        std::cerr << "Failed to open jsonl file: " << jsonlFileName << std::endl;
        return;
    }

    int nextID = 8;
    int sampleCount = 0;

    std::string line;

    while(std::getline(file, line)){
        if(line.empty()) continue;

        if(maxSamples != -1 && sampleCount >= maxSamples) break;

        try{
            json j = json::parse(line);
            std::string content = j["text"];

            std::vector<std::string> tokens = tokenizeString(content);
            std::vector<std::string> mergedTokens = applyManualMerges(tokens);

            for(int i = 0; i < mergedTokens.size(); i++){
                addTokenToVocab(mergedTokens[i], nextID);
            }

            sampleCount++;

            if(sampleCount % 10000 == 0){
                std::cout << "Vocab processed samples: " << sampleCount << std::endl;
            }
        }
        catch(json::parse_error& e){
            std::cerr << "Syntax error on line: " << line << " -> " << e.what() << std::endl;
        }
    }

    file.close();

    saveVocab(vocabFileName);

    std::cout << "Vocab saved to: " << vocabFileName << std::endl;
    std::cout << "Vocab size: " << this->vocabMAP.size() << std::endl;
}

void TokenUtility::saveVocab(std::string vocabFileName){
    json j;

    for(auto& item : this->vocabMAP){
        j[item.first] = item.second;
    }

    std::ofstream file(vocabFileName);

    if(!file.is_open()){
        std::cerr << "Failed to save vocab file: " << vocabFileName << std::endl;
        return;
    }

    file << j.dump(4);

    file.close();
}

void TokenUtility::loadVocab(std::string vocabFileName){
    std::ifstream file(vocabFileName);

    if(!file.is_open()){
        std::cerr << "Failed to open vocab file: " << vocabFileName << std::endl;
        return;
    }

    json j;
    file >> j;

    file.close();

    this->vocabMAP.clear();
    this->idToTokenMAP.clear();

    for(auto it = j.begin(); it != j.end(); ++it){
        std::string token = it.key();
        int id = it.value();

        this->vocabMAP[token] = id;
        this->idToTokenMAP[id] = token;
    }

    this->padID = this->vocabMAP["<PAD>"];
    this->bosID = this->vocabMAP["<BOS>"];
    this->eosID = this->vocabMAP["<EOS>"];
    this->unkID = this->vocabMAP["<UNK>"];
}

std::vector<int> TokenUtility::encode(std::string text){
    std::vector<int> ids;

    ids.push_back(this->bosID);

    std::vector<std::string> tokens = tokenizeString(text);
    std::vector<std::string> mergedTokens = applyManualMerges(tokens);

    for(int i = 0; i < mergedTokens.size(); i++){
        std::string token = mergedTokens[i];

        if(this->vocabMAP.find(token) != this->vocabMAP.end()){
            ids.push_back(this->vocabMAP[token]);
        }else{
            ids.push_back(this->unkID);
        }
    }

    ids.push_back(this->eosID);

    return ids;
}

std::vector<int> TokenUtility::encodePrompt(std::string text){
    std::vector<int> ids;

    ids.push_back(this->bosID);

    std::vector<std::string> tokens = tokenizeString(text);
    std::vector<std::string> mergedTokens = applyManualMerges(tokens);

    for(int i = 0; i < mergedTokens.size(); i++){
        std::string token = mergedTokens[i];

        if(this->vocabMAP.find(token) != this->vocabMAP.end()){
            ids.push_back(this->vocabMAP[token]);
        }else{
            ids.push_back(this->unkID);
        }
    }

    return ids;
}

std::string TokenUtility::decode(std::vector<int>& tokenIDs){
    std::string out = "";

    for(int i = 0; i < tokenIDs.size(); i++){
        int id = tokenIDs[i];

        if(id == this->padID) continue;
        if(id == this->bosID) continue;
        if(id == this->eosID) continue;

        if(this->idToTokenMAP.find(id) == this->idToTokenMAP.end()){
            continue;
        }

        std::string token = this->idToTokenMAP[id];

        if(token == "<UNK>") continue;

        if(out.size() > 0){
            out += " ";
        }

        out += token;
    }

    return out;
}

void TokenUtility::encodeJsonlToBin(
    std::string jsonlFileName,
    std::string outputBinFileName,
    int maxSamples
){
    std::ifstream file(jsonlFileName);

    if(!file.is_open()){
        std::cerr << "Failed to open jsonl file: " << jsonlFileName << std::endl;
        return;
    }

    std::ofstream outfile(outputBinFileName, std::ios::binary);

    if(!outfile.is_open()){
        std::cerr << "Failed to open output bin file: " << outputBinFileName << std::endl;
        file.close();
        return;
    }

    int sampleCount = 0;
    long long tokenCount = 0;

    std::string line;

    while(std::getline(file, line)){
        if(line.empty()) continue;

        if(maxSamples != -1 && sampleCount >= maxSamples) break;

        try{
            json j = json::parse(line);
            std::string content = j["text"];

            std::vector<int> ids = encode(content);

            outfile.write(
                reinterpret_cast<const char*>(ids.data()),
                ids.size() * sizeof(int)
            );

            tokenCount += ids.size();
            sampleCount++;

            if(sampleCount % 10000 == 0){
                std::cout << "Encoded samples: " << sampleCount
                          << " tokens: " << tokenCount << std::endl;
            }
        }
        catch(json::parse_error& e){
            std::cerr << "Syntax error on line: " << line << " -> " << e.what() << std::endl;
        }
    }

    file.close();
    outfile.close();

    std::cout << "Encoded bin saved to: " << outputBinFileName << std::endl;
    std::cout << "Samples encoded: " << sampleCount << std::endl;
    std::cout << "Total tokens: " << tokenCount << std::endl;
}

void TokenUtility::encodeJsonlToPaddedBin(
    std::string jsonlFileName,
    std::string outputBinFileName,
    int seqLen,
    int maxSamples
){
    std::ifstream file(jsonlFileName);

    if(!file.is_open()){
        std::cerr << "Failed to open jsonl file: " << jsonlFileName << std::endl;
        return;
    }

    std::ofstream outfile(outputBinFileName, std::ios::binary);

    if(!outfile.is_open()){
        std::cerr << "Failed to open output bin file: " << outputBinFileName << std::endl;
        file.close();
        return;
    }

    int sampleCount = 0;
    long long tokenCount = 0;
    int recordLen = seqLen + 1;

    std::string line;

    while(std::getline(file, line)){
        if(line.empty()) continue;

        if(maxSamples != -1 && sampleCount >= maxSamples) break;

        try{
            json j = json::parse(line);
            std::string content = j["text"];

            std::vector<int> ids = encode(content);
            std::vector<int> padded(recordLen, this->padID);

            int copyLen = ids.size();

            if(copyLen > recordLen){
                copyLen = recordLen;
            }

            for(int i = 0; i < copyLen; i++){
                padded[i] = ids[i];
            }

            if(ids.size() > recordLen){
                padded[recordLen - 1] = this->eosID;
            }

            outfile.write(
                reinterpret_cast<const char*>(padded.data()),
                recordLen * sizeof(int)
            );

            tokenCount += recordLen;
            sampleCount++;

            if(sampleCount % 10000 == 0){
                std::cout << "Encoded padded samples: " << sampleCount
                          << " tokens: " << tokenCount << std::endl;
            }
        }
        catch(json::parse_error& e){
            std::cerr << "Syntax error on line: " << line << " -> " << e.what() << std::endl;
        }
    }

    file.close();
    outfile.close();

    std::cout << "Encoded padded bin saved to: " << outputBinFileName << std::endl;
    std::cout << "Samples encoded: " << sampleCount << std::endl;
    std::cout << "Total tokens: " << tokenCount << std::endl;
}



//Extra Utility
int TokenUtility::getVocabSize(){return this->vocabMAP.size();}

int TokenUtility::getPadID(){return this->padID;}

int TokenUtility::getBosID(){return this->bosID;}

int TokenUtility::getEosID(){return this->eosID;}

int TokenUtility::getUnkID(){return this->unkID;}
