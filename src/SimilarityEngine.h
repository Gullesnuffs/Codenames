#pragma once

#include <string>
#include <vector>
#include "Dictionary.h"

struct SimilarityEngine {
	virtual bool load(const std::string &fileName, bool verbose) = 0;
	virtual float similarity(wordID fixedWord, wordID dynWord) = 0;
	virtual bool wordExists(const std::string &word) = 0;
	virtual float stat(wordID s) = 0;
	virtual ~SimilarityEngine() {}
};