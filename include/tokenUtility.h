#ifndef TOKENUTILITY_H
#define TOKENUTILITY_H

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class TokenUtility{
private:
    std::unordered_map<std::string, int> vocabMAP;
    std::unordered_map<int, std::string> idToTokenMAP;

    std::vector<std::pair<std::string, std::string>> mergeRules;

    int padID;
    int bosID;
    int eosID;
    int unkID;

private:
    bool isSymbol(char c);
    bool isNum(char c);

    void addTokenToList(
        std::string& prevToken,
        std::vector<std::string>& currTokenList
    );

    std::vector<std::string> tokenizeString(std::string& targetStr);

    std::vector<std::string> applyManualMerges(
        std::vector<std::string>& tokens
    );

    void addTokenToVocab(
        const std::string& token,
        int& nextID
    );

public:
    TokenUtility();

    void loadManualMerges(std::string mergeFileName);

    void createVocabFromJsonl(
        std::string jsonlFileName,
        std::string vocabFileName,
        int maxSamples = -1
    );

    void saveVocab(std::string vocabFileName);

    void loadVocab(std::string vocabFileName);

    std::vector<int> encode(std::string text);
    std::vector<int> encodePrompt(std::string text);

    std::string decode(std::vector<int>& tokenIDs);

    void encodeJsonlToBin(
        std::string jsonlFileName,
        std::string outputBinFileName,
        int maxSamples = -1
    );

    void encodeJsonlToPaddedBin(
        std::string jsonlFileName,
        std::string outputBinFileName,
        int seqLen,
        int maxSamples = -1
    );

    int getVocabSize();

    //these are extra utility so dont worry about it can be removed
    int getPadID();
    int getBosID();
    int getEosID();
    int getUnkID();
};

#endif
