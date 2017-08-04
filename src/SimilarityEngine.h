#pragma once

#include "Dictionary.h"
#include <string>
#include <vector>

struct SimilarityEngine {
	virtual bool load(const std::string &fileName, bool verbose) = 0;
	virtual float similarity(wordID fixedWord, wordID dynWord) = 0;
	virtual int getPopularity(wordID id) = 0;
	virtual wordID getID(const std::string &s) = 0;
	virtual const std::string &getWord(wordID id) = 0;
	virtual bool wordExists(const std::string &word) = 0;
	virtual float stat(wordID s) = 0;
	virtual std::vector<wordID> getCommonWords(int vocabularySize) = 0;
	virtual ~SimilarityEngine() {}
};